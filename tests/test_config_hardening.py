"""Settings validation, CORS/host middleware, error handling, logging redaction (Phase 5).
NOT VERIFIED when written: not executed by the author."""

import logging

import pytest
from fastapi.testclient import TestClient

from core.config import Settings
from core.logging_config import JsonFormatter, RedactingFilter, redact
from main import create_app

# 48 chars / 16 distinct: passes the stronger SECRET_KEY rules (length >= 32, >= 8 distinct
# characters, no placeholder). The old value "k" * 40 is now rejected by design.
GOOD_KEY = "0123456789abcdef" * 3


def _prod(**overrides):
    values = dict(
        ENVIRONMENT="production",
        DEBUG=False,
        SECRET_KEY=GOOD_KEY,
        DATABASE_URL="postgresql+psycopg://u:p@db:5432/shop",
        ALLOWED_HOSTS="shop.example.com,localhost",
        CORS_ALLOWED_ORIGINS="https://shop.example.com",
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


# ------------------------------------------------------------------ settings
def test_valid_production_settings_load():
    s = _prod()
    assert s.is_production and s.DEBUG is False
    assert s.allowed_hosts_list == ["shop.example.com", "localhost"]
    assert s.cors_origins_list == ["https://shop.example.com"]


@pytest.mark.parametrize(
    "overrides,fragment",
    [
        ({"SECRET_KEY": ""}, "SECRET_KEY"),
        ({"SECRET_KEY": "change-this-secret-key"}, "SECRET_KEY"),
        ({"SECRET_KEY": "short"}, "SECRET_KEY"),
        ({"DEBUG": True}, "DEBUG"),
        ({"DATABASE_URL": "sqlite:///./x.db"}, "DATABASE_URL"),
        ({"ALLOWED_HOSTS": "*"}, "ALLOWED_HOSTS"),
        ({"ALLOWED_HOSTS": ""}, "ALLOWED_HOSTS"),
        ({"CORS_ALLOWED_ORIGINS": "*"}, "CORS_ALLOWED_ORIGINS"),
    ],
)
def test_production_fails_fast_on_unsafe_settings(overrides, fragment):
    with pytest.raises(ValueError) as exc:
        _prod(**overrides)
    assert fragment in str(exc.value)


def test_staging_is_validated_like_production():
    with pytest.raises(ValueError):
        _prod(ENVIRONMENT="staging", SECRET_KEY="")


def test_error_message_never_contains_the_secret_value():
    with pytest.raises(ValueError) as exc:
        _prod(SECRET_KEY="change-this-secret-key", DATABASE_URL="sqlite:///./x.db")
    assert "change-this-secret-key" not in str(exc.value)


def test_development_defaults_are_usable_and_safe():
    s = Settings(_env_file=None, ENVIRONMENT="development", SECRET_KEY="")
    assert s.DEBUG is False  # safe default; local .env opts in
    assert s.SECRET_KEY and s.SECRET_KEY != "change-this-secret-key"


def test_bare_postgres_url_is_rewritten_to_psycopg3_driver():
    s = Settings(_env_file=None, DATABASE_URL="postgresql://u:p@h/db")
    assert s.DATABASE_URL.startswith("postgresql+psycopg://")


def test_redis_readiness_requirement_defaults():
    assert _prod().redis_required_for_readiness is True
    assert Settings(_env_file=None).redis_required_for_readiness is False
    assert _prod(CACHE_ENABLED=False).redis_required_for_readiness is False
    assert _prod(READINESS_REQUIRE_REDIS=False).redis_required_for_readiness is False


# ------------------------------------------------------------ CORS + hosts
def test_cors_only_allows_configured_origins():
    app = create_app(_prod())
    c = TestClient(app, base_url="https://shop.example.com")
    ok = c.get("/health", headers={"Origin": "https://shop.example.com"})
    assert ok.headers.get("access-control-allow-origin") == "https://shop.example.com"
    assert ok.headers.get("access-control-allow-credentials") == "true"
    bad = c.get("/health", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in bad.headers


def test_no_cors_middleware_when_no_origins_configured():
    app = create_app(_prod(CORS_ALLOWED_ORIGINS=""))
    c = TestClient(app, base_url="https://shop.example.com")
    resp = c.get("/health", headers={"Origin": "https://shop.example.com"})
    assert "access-control-allow-origin" not in resp.headers


def test_untrusted_host_is_rejected():
    app = create_app(_prod())
    assert TestClient(app, base_url="http://shop.example.com").get("/health").status_code == 200
    assert TestClient(app, base_url="http://evil.example.com").get("/health").status_code == 400


# ------------------------------------------------------------ error handling
def test_unhandled_error_response_leaks_nothing(caplog):
    app = create_app(_prod())

    @app.get("/_boom")
    def boom():
        raise RuntimeError("SELECT * FROM users WHERE password='hunter2' at /srv/app/secret.py")

    c = TestClient(app, base_url="http://localhost", raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR):
        resp = c.get("/_boom")
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Internal server error"}
    for leaked in ("SELECT", "hunter2", "secret.py", "Traceback"):
        assert leaked not in resp.text


def test_http_exception_format_is_unchanged():
    resp = TestClient(create_app(_prod()), base_url="http://localhost").get("/nonexistent-route")
    assert resp.status_code == 404
    assert set(resp.json()) == {"detail"}


def test_docs_can_be_disabled():
    app = create_app(_prod(ENABLE_API_DOCS=False))
    c = TestClient(app, base_url="http://localhost")
    assert c.get("/docs").status_code == 404
    assert c.get("/openapi.json").status_code == 404


def test_request_id_header_is_set():
    resp = TestClient(create_app(_prod()), base_url="http://localhost").get("/health")
    assert len(resp.headers["x-request-id"]) == 32


# ------------------------------------------------------------------- logging
def test_redact_masks_llm_provider_api_keys():
    # Key-shaped strings are built at run time (test data for the masking regex, not credentials).
    import secrets

    for key in ("gsk_" + secrets.token_hex(16), "sk-proj-" + secrets.token_hex(12)):
        masked = redact(f"request failed for {key} at provider")
        assert key not in masked and "***" in masked
        assert key[8:] not in masked


def test_redact_masks_credentials():
    assert "hunter2" not in redact("redis://:hunter2@redis:6379/0")
    assert "S3cr3t" not in redact("postgresql+psycopg://app:S3cr3t@db/shop")
    assert "eyJabc.def.ghi" not in redact("Authorization: Bearer eyJabc.def.ghi")
    assert "abc123" not in redact("login failed password=abc123")
    assert redact("GET /products -> 200") == "GET /products -> 200"


def test_redacting_filter_cleans_log_records_and_json_output():
    record = logging.LogRecord("t", logging.ERROR, __file__, 1, "cannot reach %s", ("redis://:pw123@redis:6379/0",), None)
    assert RedactingFilter().filter(record)
    out = JsonFormatter().format(record)
    assert "pw123" not in out and '"level": "ERROR"' in out


# ----------------------------------------------------------- startup contract
def test_main_does_not_create_schema_on_startup():
    import inspect

    import main

    assert "create_all" not in inspect.getsource(main)


@pytest.mark.parametrize("bad_key", ["short-but-secret-abc", "x" * 31, ""])
def test_no_unsafe_secret_value_appears_in_error_or_its_repr(bad_key):
    with pytest.raises(ValueError) as exc:
        _prod(SECRET_KEY=bad_key, DATABASE_URL="postgresql+psycopg://dbuser:dbpass@db/shop")
    text = f"{exc.value} {exc.value!r}"
    assert "SECRET_KEY is invalid" in text
    for value in (bad_key, "dbpass"):
        if value:
            assert value not in text
