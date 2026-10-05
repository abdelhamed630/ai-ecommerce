"""SAFE config batch: secret placeholders, SECRET_KEY strength, JWT ALGORITHM
allow-list, numeric bounds, Redis/Celery URL schemes, CORS / ALLOWED_HOSTS shape
and CELERY_TASK_ALWAYS_EAGER in production.

These tests never read `.env` / `.env.docker`. Templates are taken only from the
tracked `.env.example` and `.env.docker.example`. Every Settings object is built
with `_env_file=None`.
"""

from pathlib import Path

import pytest
from dotenv import dotenv_values

from core.config import (
    ALLOWED_JWT_ALGORITHMS,
    ConfigurationError,
    Settings,
    is_placeholder_secret,
)

ROOT = Path(__file__).resolve().parent.parent
GOOD_KEY = "0123456789abcdef" * 3  # 48 chars, 16 distinct


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


def _dev(**overrides):
    return Settings(_env_file=None, **overrides)


def _rejected(factory, **overrides):
    """Return the text of the error raised (any ValueError subclass), or fail."""
    with pytest.raises(ValueError) as exc:
        factory(**overrides)
    return f"{exc.value} {exc.value!r}"


# ------------------------------------------------- SECRET_KEY: placeholders
@pytest.mark.parametrize(
    "key",
    [
        "CHANGE_ME_TO_A_RANDOM_SECRET_KEY",  # 32 chars: only the placeholder rule can reject it
        "change_me_to_a_random_secret_key",
        "Change-Me-To-A-Random-Secret-Key",
        "CHANGEME" + "a1b2c3d4" * 4,
        "0123456789abcdef-CHANGE_ME-0123456789abcdef",
    ],
)
def test_production_rejects_change_me_placeholders_of_any_case(key):
    assert len(key) >= 32
    text = _rejected(_prod, SECRET_KEY=key)
    assert "SECRET_KEY is invalid" in text and "placeholder" in text
    assert key not in text


@pytest.mark.parametrize(
    "key",
    [
        "secret",
        "SECRET",
        "Secret-Key",
        "SECRET_KEY",
        "ChangeMe",
        "CHANGE-ME",
        " change-this-secret-key ",
        "Change-This-Secret-Key",
        "DEV-SECRET-KEY-NOT-FOR-PRODUCTION",
        "Dev_Secret_Key_Not_For_Production",
        "",
        "   ",
    ],
)
def test_secret_blocklist_is_case_and_separator_insensitive(key):
    assert is_placeholder_secret(key) is True
    _rejected(_prod, SECRET_KEY=key)


def test_dev_fallback_marker_is_rejected_even_inside_a_long_key():
    text = _rejected(_prod, SECRET_KEY="My-Not-For-Production-Key-0123456789abcdef")
    assert "placeholder" in text


def test_real_looking_keys_are_not_placeholders():
    for key in (GOOD_KEY, "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"):
        assert is_placeholder_secret(key) is False
        assert _prod(SECRET_KEY=key).SECRET_KEY == key


def test_staging_is_also_checked_for_placeholders():
    _rejected(_prod, ENVIRONMENT="staging", SECRET_KEY="CHANGE_ME_TO_A_RANDOM_SECRET_KEY")


def test_placeholder_rule_is_production_only_by_design():
    # Development keeps its documented behaviour (no hard failure for local checkouts).
    assert _dev(ENVIRONMENT="development", SECRET_KEY="CHANGE_ME").SECRET_KEY == "CHANGE_ME"


# ------------------------------------------------- SECRET_KEY: strength
def test_secret_key_needs_at_least_32_characters():
    assert "shorter than 32" in _rejected(_prod, SECRET_KEY="a1b2c3d4" * 3 + "e5f6a7b")  # 31 chars
    assert _prod(SECRET_KEY="a1b2c3d4" * 4).SECRET_KEY  # exactly 32 is accepted


def test_low_entropy_key_is_rejected():
    assert "distinct" in _rejected(_prod, SECRET_KEY="k" * 40)
    assert "distinct" in _rejected(_prod, SECRET_KEY="ab" * 30)


def test_key_with_surrounding_whitespace_is_rejected():
    assert "whitespace" in _rejected(_prod, SECRET_KEY=" " + GOOD_KEY)
    assert "whitespace" in _rejected(_prod, SECRET_KEY=GOOD_KEY + "\n")


def test_secret_error_text_never_contains_the_key():
    for key in ("CHANGE_ME_TO_A_RANDOM_SECRET_KEY", "k" * 40, "weak-but-unique-value-xyz"):
        assert key not in _rejected(_prod, SECRET_KEY=key)


# ------------------------------------------------- placeholder passwords in URLs
@pytest.mark.parametrize(
    "name,url",
    [
        ("DATABASE_URL", "postgresql+psycopg://postgres:CHANGE_ME_TO_A_STRONG_DATABASE_PASSWORD@postgres:5432/shop"),
        ("DATABASE_URL", "postgresql+psycopg://postgres:change_me@postgres:5432/shop"),
        ("REDIS_URL", "redis://:CHANGE_ME_TO_A_STRONG_REDIS_PASSWORD@redis:6379/0"),
        ("CELERY_BROKER_URL", "redis://:Change-Me-Redis@redis:6379/1"),
        ("CELERY_RESULT_BACKEND", "redis://default:CHANGEME@redis:6379/2"),
    ],
)
def test_production_rejects_placeholder_passwords_in_urls_without_leaking_them(name, url):
    text = _rejected(_prod, **{name: url})
    assert name in text and "placeholder" in text
    password = url.split("@")[0].rsplit(":", 1)[1]
    assert password not in text


def test_urls_without_a_password_are_not_rejected_by_the_placeholder_rule():
    # Forcing a Redis password is a separate (breaking) decision, not part of this batch.
    s = _prod(REDIS_URL="redis://redis:6379/0")
    assert s.REDIS_URL == "redis://redis:6379/0"


# ------------------------------------------------- ALGORITHM allow-list
def test_algorithm_default_is_hs256():
    assert _dev().ALGORITHM == "HS256"


@pytest.mark.parametrize("alg", ALLOWED_JWT_ALGORITHMS)
def test_allowed_algorithms_load(alg):
    assert _dev(ALGORITHM=alg).ALGORITHM == alg
    assert _prod(ALGORITHM=alg).ALGORITHM == alg


@pytest.mark.parametrize("alg", ["none", "None", "", "hs256", "HS1", "HS512 ", "RS256", "ES256", "PS256"])
def test_other_algorithms_are_rejected_in_every_environment(alg):
    assert "ALGORITHM" in _rejected(_dev, ALGORITHM=alg)
    assert "ALGORITHM" in _rejected(_prod, ALGORITHM=alg)


def test_allow_list_is_exactly_the_three_hmac_algorithms():
    assert ALLOWED_JWT_ALGORITHMS == ("HS256", "HS384", "HS512")


# ------------------------------------------------- numeric bounds
@pytest.mark.parametrize("value", [1, 60, 10080])
def test_access_token_minutes_accepts_sane_values(value):
    assert _dev(ACCESS_TOKEN_EXPIRE_MINUTES=value).ACCESS_TOKEN_EXPIRE_MINUTES == value


@pytest.mark.parametrize("value", [0, -1, 10081, 525600])
def test_access_token_minutes_rejects_out_of_range(value):
    assert "ACCESS_TOKEN_EXPIRE_MINUTES" in _rejected(_dev, ACCESS_TOKEN_EXPIRE_MINUTES=value)


# (setting, accepted boundary values, rejected values)
NUMERIC_BOUNDS = [
    ("DB_POOL_SIZE", [1, 10, 100], [0, -1, 101]),
    ("DB_MAX_OVERFLOW", [0, 20, 200], [-1, 201]),
    ("DB_POOL_TIMEOUT_SECONDS", [1, 30, 300], [0, -1, 301]),
    ("DB_POOL_RECYCLE_SECONDS", [30, 1800, 86400], [0, 29, -1, 86401]),
    ("CELERY_RESULT_EXPIRES_SECONDS", [1, 86400, 2592000], [0, -1, 2592001]),
    ("CELERY_BEAT_CHURN_TRAIN_INTERVAL_SECONDS", [0, 604800, 31536000], [-1, 31536001]),
    ("CELERY_BEAT_SEGMENTATION_INTERVAL_SECONDS", [0, 86400, 31536000], [-1, 31536001]),
    ("CACHE_SOCKET_TIMEOUT_SECONDS", [0.01, 0.5, 30], [0, -1, 30.1, "nan", "inf"]),
    ("CACHE_FAILURE_BACKOFF_SECONDS", [0, 5.0, 3600], [-0.1, 3600.1, "nan", "inf"]),
    ("RECOMMENDATION_CACHE_TTL_SECONDS", [0, 300, 604800], [-1, 604801]),
    ("CHURN_CACHE_TTL_SECONDS", [0, 600, 604800], [-1, 604801]),
    ("SEGMENTATION_CACHE_TTL_SECONDS", [0, 300, 604800], [-1, 604801]),
    ("CF_MAX_NEIGHBORS", [1, 50, 10000], [0, -1, 10001]),
    ("HYBRID_CONTENT_WEIGHT", [0, 0.5, 3], [-0.1, "nan", "inf"]),
    ("HYBRID_COLLABORATIVE_WEIGHT", [0, 0.5, 3], [-0.1, "nan", "inf"]),
    ("SEGMENTATION_N_CLUSTERS", [1, 4, 10], [0, -1, 101]),
    ("SEGMENTATION_MAX_CLUSTERS", [4, 10, 100], [0, -1, 101]),
    ("SEGMENTATION_RANDOM_STATE", [0, 42, 2**32 - 1], [-1, 2**32]),
    ("CHURN_INACTIVITY_DAYS", [1, 60, 3650], [0, -1, 3651]),
    ("CHURN_SNAPSHOT_COUNT", [1, 3, 24], [0, -1, 25]),
    ("CHURN_RECENT_ACTIVITY_DAYS", [1, 30, 3650], [0, -1, 3651]),
    ("CHURN_MIN_SAMPLES", [1, 50, 1_000_000], [0, -1, 1_000_001]),
    ("CHURN_MIN_CLASS_SAMPLES", [1, 5, 1_000_000], [0, -1, 1_000_001]),
    ("CHURN_VALIDATION_FRACTION", [0.01, 0.25, 0.99], [0, 1, -0.1, 1.5, "nan"]),
    ("CHURN_LOGREG_C", [0.001, 1.0, 1e6], [0, -1, 1e6 + 1, "nan", "inf"]),
    ("CHURN_RANDOM_STATE", [0, 42, 2**32 - 1], [-1, 2**32]),
    ("CHURN_RISK_MEDIUM_THRESHOLD", [0, 0.4, 0.69], [-0.1, 1.1]),
    ("CHURN_RISK_HIGH_THRESHOLD", [0.5, 0.7, 1], [-0.1, 1.1]),
]


@pytest.mark.parametrize(
    "name,value",
    [(n, v) for n, good, _ in NUMERIC_BOUNDS for v in good],
    ids=lambda x: str(x),
)
def test_numeric_settings_accept_boundary_values(name, value):
    assert getattr(_dev(**{name: value}), name) == pytest.approx(float(value))


@pytest.mark.parametrize(
    "name,value",
    [(n, v) for n, _, bad in NUMERIC_BOUNDS for v in bad],
    ids=lambda x: str(x),
)
def test_numeric_settings_reject_out_of_range_values(name, value):
    assert name in _rejected(_dev, **{name: value})


def test_bounds_also_apply_when_the_value_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("DB_POOL_SIZE", "0")
    assert "DB_POOL_SIZE" in _rejected(_dev)
    monkeypatch.setenv("DB_POOL_SIZE", "10")
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "999999")
    assert "ACCESS_TOKEN_EXPIRE_MINUTES" in _rejected(_dev)


def test_all_defaults_satisfy_their_own_bounds():
    s = _dev()
    assert s.DB_POOL_SIZE == 10 and s.DB_MAX_OVERFLOW == 20
    assert s.ACCESS_TOKEN_EXPIRE_MINUTES == 60
    _prod()  # production defaults are valid too


# ------------------------------------------------- cross-field consistency
def test_default_cluster_count_must_not_exceed_the_maximum():
    text = _rejected(_dev, SEGMENTATION_N_CLUSTERS=11, SEGMENTATION_MAX_CLUSTERS=10)
    assert "SEGMENTATION_N_CLUSTERS" in text
    assert _dev(SEGMENTATION_N_CLUSTERS=10, SEGMENTATION_MAX_CLUSTERS=10)


def test_hybrid_weights_cannot_both_be_zero():
    assert "HYBRID" in _rejected(_dev, HYBRID_CONTENT_WEIGHT=0, HYBRID_COLLABORATIVE_WEIGHT=0)
    assert _dev(HYBRID_CONTENT_WEIGHT=0, HYBRID_COLLABORATIVE_WEIGHT=1)


@pytest.mark.parametrize("medium,high", [(0.7, 0.7), (0.8, 0.7)])
def test_churn_thresholds_must_be_ordered(medium, high):
    text = _rejected(_dev, CHURN_RISK_MEDIUM_THRESHOLD=medium, CHURN_RISK_HIGH_THRESHOLD=high)
    assert "CHURN_RISK_MEDIUM_THRESHOLD" in text
    assert _dev(CHURN_RISK_MEDIUM_THRESHOLD=0, CHURN_RISK_HIGH_THRESHOLD=1)


# ------------------------------------------------- CELERY_TASK_ALWAYS_EAGER
@pytest.mark.parametrize("env", ["production", "staging"])
def test_eager_celery_is_rejected_in_production_like_environments(env):
    assert "CELERY_TASK_ALWAYS_EAGER" in _rejected(_prod, ENVIRONMENT=env, CELERY_TASK_ALWAYS_EAGER=True)


def test_eager_celery_remains_allowed_outside_production():
    assert _dev(CELERY_TASK_ALWAYS_EAGER=True).CELERY_TASK_ALWAYS_EAGER is True
    assert _dev(ENVIRONMENT="test", CELERY_TASK_ALWAYS_EAGER=True).CELERY_TASK_ALWAYS_EAGER is True


def test_production_with_eager_false_loads():
    assert _prod(CELERY_TASK_ALWAYS_EAGER=False).CELERY_TASK_ALWAYS_EAGER is False


# ------------------------------------------------- Redis / Celery URL schemes
URL_SETTINGS = ["REDIS_URL", "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"]


@pytest.mark.parametrize("name", URL_SETTINGS)
@pytest.mark.parametrize(
    "url",
    [
        "redis://localhost:6379/0",
        "rediss://redis.internal:6380/1",
        "redis://:s3cr3t@redis:6379/2",
        "REDIS://localhost:6379/0",
    ],
)
def test_redis_schemes_are_accepted(name, url):
    assert getattr(_dev(**{name: url}), name) == url
    assert getattr(_prod(**{name: url.replace("s3cr3t", GOOD_KEY[:10])}), name)


@pytest.mark.parametrize("name", URL_SETTINGS)
@pytest.mark.parametrize(
    "url",
    ["http://localhost:6379/0", "amqp://guest@localhost//", "memory://", "localhost:6379", "redis", "", "   ", "//redis:6379/0"],
)
def test_other_schemes_are_rejected_in_every_environment(name, url):
    assert name in _rejected(_dev, **{name: url})
    assert name in _rejected(_prod, **{name: url})


def test_scheme_error_does_not_leak_the_url_password():
    text = _rejected(_dev, REDIS_URL="http://:hunter2pw@redis:6379/0")
    assert "REDIS_URL" in text and "hunter2pw" not in text


# ------------------------------------------------- CORS_ALLOWED_ORIGINS shape
@pytest.mark.parametrize(
    "value",
    [
        "",
        "https://shop.example.com",
        "https://shop.example.com,https://admin.example.com",
        " https://a.example.com , http://localhost:3000 ,",
        "http://127.0.0.1:5173",
        "https://a.example.com:8443",
    ],
)
def test_valid_cors_origins_load(value):
    assert _prod(CORS_ALLOWED_ORIGINS=value).cors_origins_list == [v.strip() for v in value.split(",") if v.strip()]


def test_wildcard_cors_is_still_dev_only():
    assert _dev(CORS_ALLOWED_ORIGINS="*").cors_origins_list == ["*"]
    assert "CORS_ALLOWED_ORIGINS" in _rejected(_prod, CORS_ALLOWED_ORIGINS="*")


@pytest.mark.parametrize(
    "value",
    [
        "shop.example.com",  # no scheme
        "https://shop.example.com/",  # trailing slash never matches an Origin header
        "https://shop.example.com/app",
        "ftp://shop.example.com",
        "https://",
        "https://a.example.com:99999",
        "https://a.example.com:abc",
        "https://user:pw@a.example.com",
        "https://a.example.com?x=1",
        "https://a.example.com?",
        "https://a.example.com#",
        "http://a b.example.com",
        "null",
        "https://ok.example.com,shop.example.com",  # one bad entry among good ones
    ],
)
def test_malformed_cors_origins_are_rejected_in_every_environment(value):
    assert "CORS_ALLOWED_ORIGINS" in _rejected(_dev, CORS_ALLOWED_ORIGINS=value)
    assert "CORS_ALLOWED_ORIGINS" in _rejected(_prod, CORS_ALLOWED_ORIGINS=value)


def test_cors_error_does_not_echo_entries():
    text = _rejected(_dev, CORS_ALLOWED_ORIGINS="https://user:pw123@a.example.com")
    assert "pw123" not in text and "entry #1" in text


# ------------------------------------------------- ALLOWED_HOSTS shape
@pytest.mark.parametrize(
    "value",
    [
        "shop.example.com",
        "shop.example.com,localhost",
        "localhost,127.0.0.1,api,shop.example.com",
        "*.example.com,example.com",
        "EXAMPLE.com",
        "my_service,api-2",
        " a.example.com , b.example.com ,",
    ],
)
def test_valid_allowed_hosts_load(value):
    assert _prod(ALLOWED_HOSTS=value).allowed_hosts_list == [v.strip() for v in value.split(",") if v.strip()]


def test_wildcard_host_is_still_dev_only_and_empty_is_rejected_in_production():
    assert _dev(ALLOWED_HOSTS="*").allowed_hosts_list == ["*"]
    assert "ALLOWED_HOSTS" in _rejected(_prod, ALLOWED_HOSTS="*")
    assert "ALLOWED_HOSTS" in _rejected(_prod, ALLOWED_HOSTS="")


@pytest.mark.parametrize(
    "value",
    [
        "https://shop.example.com",
        "shop.example.com:8000",  # Starlette strips the port from Host, so this could never match
        "shop.example.com/",
        "shop.example.com/path",
        "shop example.com",
        "-bad.example.com",
        "bad-.example.com",
        "a..example.com",
        ".example.com",
        "*.",
        "*example.com",
        "exa$mple.com",
        "a" * 64 + ".example.com",
        "ok.example.com,https://bad.example.com",
    ],
)
def test_malformed_allowed_hosts_are_rejected_in_every_environment(value):
    assert "ALLOWED_HOSTS" in _rejected(_dev, ALLOWED_HOSTS=value)
    assert "ALLOWED_HOSTS" in _rejected(_prod, ALLOWED_HOSTS=value)


# ------------------------------------------------- tracked example templates only
def _example(name):
    return {k: v for k, v in dotenv_values(ROOT / name).items() if v is not None}


def test_env_example_loads_as_a_valid_development_configuration():
    values = _example(".env.example")
    s = Settings(_env_file=None, **values)
    assert s.ENVIRONMENT == "development" and not s.is_production
    assert s.ALGORITHM == "HS256"
    assert s.cors_origins_list == ["http://localhost:3000", "http://localhost:5173"]


def _compose_style_settings(values):
    """Mimic how docker-compose.yml assembles the app's environment from .env.docker."""
    return dict(
        ENVIRONMENT="production",
        SECRET_KEY=values["SECRET_KEY"],
        DATABASE_URL=(
            f"postgresql+psycopg://{values['POSTGRES_USER']}:{values['POSTGRES_PASSWORD']}"
            f"@postgres:5432/{values['POSTGRES_DB']}"
        ),
        REDIS_URL=f"redis://:{values['REDIS_PASSWORD']}@redis:6379/0",
        CELERY_BROKER_URL=f"redis://:{values['REDIS_PASSWORD']}@redis:6379/1",
        CELERY_RESULT_BACKEND=f"redis://:{values['REDIS_PASSWORD']}@redis:6379/2",
        ALLOWED_HOSTS=values["ALLOWED_HOSTS"],
        CORS_ALLOWED_ORIGINS=values["CORS_ALLOWED_ORIGINS"],
        LOG_LEVEL=values["LOG_LEVEL"],
        LOG_FORMAT=values["LOG_FORMAT"],
        ENABLE_API_DOCS=values["ENABLE_API_DOCS"],
        CELERY_BEAT_ENABLED=values["CELERY_BEAT_ENABLED"],
    )


def test_every_change_me_value_in_the_docker_template_is_detected_as_a_placeholder():
    values = _example(".env.docker.example")
    placeholders = {k: v for k, v in values.items() if "CHANGE_ME" in v}
    assert {"SECRET_KEY", "POSTGRES_PASSWORD", "REDIS_PASSWORD"} <= set(placeholders)
    for key, value in placeholders.items():
        assert is_placeholder_secret(value), key


def test_unedited_docker_template_is_refused_in_production_without_leaking_its_values():
    values = _example(".env.docker.example")
    # _env_file=None: never let a developer's real .env leak into this test.
    text = _rejected(Settings, _env_file=None, **_compose_style_settings(values))
    for name in ("SECRET_KEY", "DATABASE_URL", "REDIS_URL", "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"):
        assert name in text, name
    for key in ("SECRET_KEY", "POSTGRES_PASSWORD", "REDIS_PASSWORD"):
        assert values[key] not in text


def test_docker_template_is_accepted_once_every_change_me_is_replaced():
    values = _example(".env.docker.example")
    filled = {
        k: ("Zx9" + GOOD_KEY if "CHANGE_ME" in v else v)
        for k, v in values.items()
    }
    s = Settings(_env_file=None, **_compose_style_settings(filled))
    assert s.is_production and s.ENABLE_API_DOCS is False
    assert s.allowed_hosts_list == ["localhost", "127.0.0.1", "api", "shop.example.com"]
    assert isinstance(s, Settings) and not isinstance(s, ConfigurationError)


# ------------------------------------------------- extra edge cases (added with the implementation)
def test_error_reasons_are_specific_not_a_fixed_blurb():
    # Each failure names only ITS reason, so the assertions above are not vacuous.
    placeholder_only = _rejected(_prod, SECRET_KEY="CHANGE_ME_TO_A_RANDOM_SECRET_KEY")
    assert "placeholder" in placeholder_only
    assert "shorter" not in placeholder_only and "distinct" not in placeholder_only
    entropy_only = _rejected(_prod, SECRET_KEY="k" * 40)
    assert "distinct" in entropy_only and "placeholder" not in entropy_only
    short_only = _rejected(_prod, SECRET_KEY="a1b2c3d4" * 3 + "e5f6a7b")
    assert "shorter than 32" in short_only and "placeholder" not in short_only


def test_secret_rules_are_enforced_in_staging_too_for_weak_keys():
    _rejected(_prod, ENVIRONMENT="staging", SECRET_KEY="k" * 40)


def test_every_unsafe_secret_is_rejected_whatever_its_case():
    from core.config import UNSAFE_SECRET_KEYS

    for key in UNSAFE_SECRET_KEYS:
        for variant in (key, key.upper(), key.title(), key.replace("-", "_"), f"  {key}  "):
            assert is_placeholder_secret(variant) is True, repr(variant)


def test_an_embedded_newline_cannot_slip_through_the_shape_checks():
    # Surrounding whitespace is stripped per entry (so "x\n" is just "x"); a newline INSIDE an
    # entry must still be rejected. Guards against `$`-style matching that tolerates a final "\n".
    assert "ALLOWED_HOSTS" in _rejected(_dev, ALLOWED_HOSTS="a.example.com\nb.example.com")
    assert "CORS_ALLOWED_ORIGINS" in _rejected(_dev, CORS_ALLOWED_ORIGINS="https://a.example.com\nhttps://b.example.com")
    assert _dev(ALLOWED_HOSTS="shop.example.com\n").allowed_hosts_list == ["shop.example.com"]


@pytest.mark.parametrize("value", ["https://a.example.com:0", "HTTPS://a.example.com", "https://a.example.com:65536"])
def test_more_malformed_cors_origins(value):
    assert "CORS_ALLOWED_ORIGINS" in _rejected(_dev, CORS_ALLOWED_ORIGINS=value)


def test_cors_port_boundaries_and_ipv6_literal_are_accepted():
    for origin in ("https://a.example.com:1", "https://a.example.com:65535", "http://[::1]:3000"):
        assert _dev(CORS_ALLOWED_ORIGINS=origin).cors_origins_list == [origin]


def test_redis_url_with_surrounding_whitespace_is_not_newly_rejected():
    # Whitespace tolerance is deliberate: this batch must not break a setup that works today.
    assert _dev(REDIS_URL=" redis://localhost:6379/0").REDIS_URL.strip() == "redis://localhost:6379/0"


def test_consistency_errors_never_contain_the_secret_key():
    text = _rejected(_prod, SECRET_KEY=GOOD_KEY, SEGMENTATION_N_CLUSTERS=11, SEGMENTATION_MAX_CLUSTERS=10)
    assert GOOD_KEY not in text
    text = _rejected(_prod, SECRET_KEY=GOOD_KEY, DB_POOL_SIZE=0)
    assert GOOD_KEY not in text


def test_all_problems_are_reported_together_in_production():
    text = _rejected(_prod, SECRET_KEY="k" * 40, CELERY_TASK_ALWAYS_EAGER=True, REDIS_URL="redis://:change_me@redis:6379/0")
    for name in ("SECRET_KEY", "CELERY_TASK_ALWAYS_EAGER", "REDIS_URL"):
        assert name in text
