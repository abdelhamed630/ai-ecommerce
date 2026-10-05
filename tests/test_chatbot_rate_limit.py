"""POST /chatbot/chat rate limiting: limits, Redis behaviour, degraded mode, no LLM call after 429."""
import pytest
from pydantic import SecretStr, ValidationError

from core import rate_limit
from core.config import Settings, settings
from tests.helpers_chatbot import FakeGroqClient, client, new_user_headers
from services import llm_service


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    fake = FakeGroqClient()
    monkeypatch.setattr(settings, "GROQ_API_KEY", SecretStr("placeholder-not-a-real-key"))
    monkeypatch.setattr(settings, "GROQ_MODEL", "test-model")
    llm_service.reset_client()
    monkeypatch.setattr(llm_service, "_get_client", lambda api_key: fake)
    rate_limit.reset_rate_limits()
    yield fake
    llm_service.reset_client()
    rate_limit.reset_rate_limits()


def _chat(headers, message="hi"):
    return client.post("/chatbot/chat", json={"message": message}, headers=headers)


def test_default_limit_is_20_per_minute(monkeypatch):
    monkeypatch.delenv("CHATBOT_RATE_LIMIT_PER_MINUTE", raising=False)  # tests/conftest.py sets it to 0
    assert Settings(_env_file=None).CHATBOT_RATE_LIMIT_PER_MINUTE == 20


def test_the_environment_variable_sets_the_limit(monkeypatch):
    monkeypatch.setenv("CHATBOT_RATE_LIMIT_PER_MINUTE", "7")
    assert Settings(_env_file=None).CHATBOT_RATE_LIMIT_PER_MINUTE == 7


def test_under_at_and_over_the_limit(monkeypatch, fake_llm):
    monkeypatch.setattr(settings, "CHATBOT_RATE_LIMIT_PER_MINUTE", 3)
    h = new_user_headers()
    assert [_chat(h).status_code for _ in range(2)] == [200, 200]  # under
    assert _chat(h).status_code == 200  # exactly at the limit: still allowed
    over = _chat(h)
    assert over.status_code == 429
    retry = int(over.headers["Retry-After"])
    assert 1 <= retry <= rate_limit.WINDOW_SECONDS + 1
    assert len(fake_llm.calls) == 3  # the 429 never reached the LLM


def test_limits_are_per_user(monkeypatch):
    monkeypatch.setattr(settings, "CHATBOT_RATE_LIMIT_PER_MINUTE", 1)
    a, b = new_user_headers(), new_user_headers()
    assert _chat(a).status_code == 200
    assert _chat(a).status_code == 429
    assert _chat(b).status_code == 200


def test_zero_disables_the_limit(monkeypatch):
    monkeypatch.setattr(settings, "CHATBOT_RATE_LIMIT_PER_MINUTE", 0)
    h = new_user_headers()
    assert all(_chat(h).status_code == 200 for _ in range(25))


@pytest.mark.parametrize("value", ["-1", "abc", "1.5", "10001"])
def test_invalid_configuration_is_rejected(value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, CHATBOT_RATE_LIMIT_PER_MINUTE=value)


def test_unauthenticated_and_invalid_token_requests_are_401_not_counted(monkeypatch):
    monkeypatch.setattr(settings, "CHATBOT_RATE_LIMIT_PER_MINUTE", 1)
    assert client.post("/chatbot/chat", json={"message": "hi"}).status_code == 401
    bad = {"Authorization": "Bearer not-a-token"}
    assert client.post("/chatbot/chat", json={"message": "hi"}, headers=bad).status_code == 401
    assert rate_limit._hits == {}


def test_stream_endpoint_shares_the_same_counter(monkeypatch, fake_llm):
    monkeypatch.setattr(settings, "CHATBOT_RATE_LIMIT_PER_MINUTE", 1)
    h = new_user_headers()
    assert _chat(h).status_code == 200
    blocked = client.post("/chatbot/chat/stream", json={"message": "hi"}, headers=h)
    assert blocked.status_code == 429 and "Retry-After" in blocked.headers
    assert len(fake_llm.calls) == 1


# --- Redis (fakeredis speaks real redis-py semantics; real Redis is NOT VERIFIED here) -------------


def test_redis_counter_is_atomic_expiring_and_namespaced(redis_cache, monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "staging")
    assert rate_limit.check_rate_limit("chatbot", 7, limit=2) is None
    assert rate_limit.check_rate_limit("chatbot", 7, limit=2) is None
    assert rate_limit.check_rate_limit("chatbot", 7, limit=2) is not None
    keys = redis_cache.keys("*ratelimit*")
    assert len(keys) == 1
    key = keys[0]
    assert key.startswith(f"{settings.CACHE_KEY_PREFIX}:staging:ratelimit:chatbot:7:")
    assert int(redis_cache.get(key)) == 3
    assert 0 < redis_cache.ttl(key) <= rate_limit.WINDOW_SECONDS + 1
    assert rate_limit._hits == {}  # Redis answered: the in-process fallback was not used


def test_keys_differ_between_users_and_environments(redis_cache, monkeypatch):
    rate_limit.check_rate_limit("chatbot", 1, limit=5)
    rate_limit.check_rate_limit("chatbot", 2, limit=5)
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    rate_limit.check_rate_limit("chatbot", 1, limit=5)
    assert len(redis_cache.keys("*ratelimit*")) == 3


def test_environment_name_cannot_inject_key_separators(redis_cache, monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "prod:../x *")
    rate_limit.check_rate_limit("chatbot", 1, limit=5)
    (key,) = redis_cache.keys("*ratelimit*")
    assert key.split(":")[1] == "prod_.._x__"  # separators / wildcards / spaces replaced


def test_limit_is_enforced_through_redis_end_to_end(redis_cache, monkeypatch, fake_llm):
    monkeypatch.setattr(settings, "CHATBOT_RATE_LIMIT_PER_MINUTE", 2)
    h = new_user_headers()
    assert [_chat(h).status_code for _ in range(3)] == [200, 200, 429]
    assert len(fake_llm.calls) == 2


def test_redis_failure_degrades_to_the_local_limiter_and_never_crashes(redis_cache, monkeypatch, fake_llm):
    """Redis dies mid-flight: still limited (NOT fail-open), per process, chatbot keeps answering."""
    monkeypatch.setattr(settings, "CHATBOT_RATE_LIMIT_PER_MINUTE", 2)

    def broken_pipeline(*a, **k):
        raise ConnectionError("redis is down")

    monkeypatch.setattr(redis_cache, "pipeline", broken_pipeline)
    h = new_user_headers()
    assert [_chat(h).status_code for _ in range(3)] == [200, 200, 429]
    assert len(fake_llm.calls) == 2


def test_degraded_mode_logs_a_warning_at_most_once_a_minute(redis_cache, monkeypatch, caplog):
    def broken_pipeline(*a, **k):
        raise ConnectionError("redis is down")

    monkeypatch.setattr(redis_cache, "pipeline", broken_pipeline)
    with caplog.at_level("WARNING", logger="core.rate_limit"):
        for _ in range(3):
            rate_limit.check_rate_limit("chatbot", 1, limit=100)
    degraded = [r for r in caplog.records if "in-memory limiter" in r.getMessage()]
    assert len(degraded) <= 1


def test_cache_disabled_uses_the_local_limiter_without_a_warning(monkeypatch, caplog):
    monkeypatch.setattr(settings, "CACHE_ENABLED", False)
    with caplog.at_level("WARNING", logger="core.rate_limit"):
        assert rate_limit.check_rate_limit("chatbot", 1, limit=1) is None
        assert rate_limit.check_rate_limit("chatbot", 1, limit=1) is not None
    assert not [r for r in caplog.records if "in-memory limiter" in r.getMessage()]
