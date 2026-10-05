"""POST /chatbot/chat/stream (Server-Sent Events). The provider is ALWAYS faked.

Real Groq streaming behaviour and a real client disconnect over a socket are NOT
covered here (no network access in the test environment): they need a manual check.
"""
import pytest
from pydantic import SecretStr

from core import rate_limit
from core.config import settings
from services import chatbot_service, llm_service
from tests.helpers_chatbot import FakeGroqClient, FakeStream, client, new_user_headers, parse_sse, token
from database.database import SessionLocal
from models.product import Product

SECRET_KEY_VALUE = "placeholder-key-must-never-appear-0123"


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    fake = FakeGroqClient()
    monkeypatch.setattr(settings, "GROQ_API_KEY", SecretStr(SECRET_KEY_VALUE))
    monkeypatch.setattr(settings, "GROQ_MODEL", "test-model")
    llm_service.reset_client()
    monkeypatch.setattr(llm_service, "_get_client", lambda api_key: fake)
    rate_limit.reset_rate_limits()
    yield fake
    llm_service.reset_client()
    rate_limit.reset_rate_limits()


@pytest.fixture(scope="module")
def headers():
    return new_user_headers()


@pytest.fixture
def seed_product():
    ids = []

    def _seed(**fields):
        fields.setdefault("price", 10.0)
        fields.setdefault("stock", 2)
        db = SessionLocal()
        try:
            product = Product(**fields)
            db.add(product)
            db.commit()
            db.refresh(product)
            ids.append(product.id)
            return product.id
        finally:
            db.close()

    yield _seed
    if ids:
        db = SessionLocal()
        try:
            db.query(Product).filter(Product.id.in_(ids)).delete(synchronize_session=False)
            db.commit()
        finally:
            db.close()


def _stream(headers, message="hello", **extra):
    return client.post("/chatbot/chat/stream", json={"message": message, **extra}, headers=headers)


def test_requires_authentication():
    assert client.post("/chatbot/chat/stream", json={"message": "hi"}).status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert client.post("/chatbot/chat/stream", json={"message": "hi"}, headers=bad).status_code == 401


def test_streams_products_then_tokens_then_done(headers, seed_product, fake_llm):
    t = token()
    pid = seed_product(name=f"Gadget {t}", price=42.0, stock=2)
    r = _stream(headers, f"do you have {t}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    events = parse_sse(r.text)
    assert [name for name, _ in events] == ["products", "token", "token", "token", "done"]
    assert [p["id"] for p in events[0][1]["products"]] == [pid]
    assert "".join(d["text"] for n, d in events if n == "token") == "Hello there"
    assert "seller_id" not in events[0][1]["products"][0]


def test_the_provider_is_asked_for_a_stream_with_the_same_safety_prompt(headers, fake_llm):
    _stream(headers, "hello")
    call = fake_llm.calls[0]
    assert call["stream"] is True
    assert call["messages"][0] == {"role": "system", "content": llm_service.SYSTEM_PROMPT}
    assert "<products>" in call["messages"][-1]["content"]
    assert call["max_completion_tokens"] == llm_service.MAX_OUTPUT_TOKENS


def test_history_is_validated_bounded_and_forwarded(headers, fake_llm, monkeypatch):
    assert _stream(headers, history=[{"role": "system", "content": "x"}]).status_code == 422
    assert _stream(headers, history=[{"role": "user", "content": "x"}] * 51).status_code == 422
    monkeypatch.setattr(settings, "CHATBOT_MAX_HISTORY_TURNS", 1)
    turns = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "newer"}]
    assert _stream(headers, "now", history=turns).status_code == 200
    assert [m["content"] for m in fake_llm.calls[-1]["messages"][1:-1]] == ["newer"]


def test_invalid_body_is_rejected_before_any_llm_call(headers, fake_llm):
    assert client.post("/chatbot/chat/stream", json={"message": ""}, headers=headers).status_code == 422
    assert client.post("/chatbot/chat/stream", json={"message": "x" * 2001}, headers=headers).status_code == 422
    assert fake_llm.calls == []


def test_rate_limit_applies_before_the_llm(monkeypatch, fake_llm):
    monkeypatch.setattr(settings, "CHATBOT_RATE_LIMIT_PER_MINUTE", 2)
    h = new_user_headers()
    assert _stream(h).status_code == 200
    assert _stream(h).status_code == 200
    blocked = _stream(h)
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1
    assert "text/event-stream" not in blocked.headers.get("content-type", "")
    assert len(fake_llm.calls) == 2


def test_rate_limit_is_per_user_on_the_stream(monkeypatch):
    monkeypatch.setattr(settings, "CHATBOT_RATE_LIMIT_PER_MINUTE", 1)
    a, b = new_user_headers(), new_user_headers()
    assert _stream(a).status_code == 200
    assert _stream(a).status_code == 429
    assert _stream(b).status_code == 200


def test_recommendation_requests_use_the_engine_on_the_stream(headers, monkeypatch, fake_llm):
    from services import recommendation_cache_service

    calls = []
    monkeypatch.setattr(
        recommendation_cache_service,
        "get_user_recommendations_cached",
        lambda db, user_id, limit: calls.append(user_id)
        or [{"product": {"id": 1, "name": "Rec", "price": 1.0, "stock": 1}}],
    )
    events = parse_sse(_stream(headers, "رشحلي حاجة").text)
    assert len(calls) == 1
    assert events[0][1]["products"][0]["name"] == "Rec"
    assert "personalized recommendations" in fake_llm.calls[0]["messages"][-1]["content"]


def test_provider_error_before_any_text_becomes_a_safe_error_event(headers, fake_llm):
    fake_llm.error = RuntimeError("401 invalid key " + SECRET_KEY_VALUE)
    r = _stream(headers, "hello")
    assert r.status_code == 200
    events = parse_sse(r.text)
    assert [n for n, _ in events] == ["products", "error"]
    assert events[1][1]["message"] == chatbot_service.UNAVAILABLE_ANSWER
    assert SECRET_KEY_VALUE not in r.text and "401" not in r.text and "Traceback" not in r.text


def test_error_message_is_arabic_for_arabic_questions(headers, fake_llm):
    fake_llm.error = RuntimeError("down")
    events = parse_sse(_stream(headers, "عندكم ايه؟").text)
    assert events[-1] == ("error", {"message": chatbot_service.UNAVAILABLE_ANSWER_AR})


def test_provider_error_in_the_middle_of_the_stream(headers, fake_llm):
    fake_llm.stream_fail_after = 1  # one token, then the provider breaks
    r = _stream(headers, "hello")
    events = parse_sse(r.text)
    assert [n for n, _ in events] == ["products", "token", "error"]
    assert "secret-detail-123" not in r.text and "provider exploded" not in r.text
    assert fake_llm.last_stream.closed is True


def test_missing_api_key_is_a_safe_error_event(headers, monkeypatch, fake_llm):
    monkeypatch.setattr(settings, "GROQ_API_KEY", SecretStr(""))
    events = parse_sse(_stream(headers, "hello").text)
    assert [n for n, _ in events] == ["products", "error"]
    assert fake_llm.calls == []


def test_empty_provider_stream_is_an_error(headers, fake_llm):
    fake_llm.stream_pieces = []
    events = parse_sse(_stream(headers, "hello").text)
    assert [n for n, _ in events] == ["products", "error"]


def test_the_provider_stream_is_closed_after_a_normal_finish(headers, fake_llm):
    _stream(headers, "hello")
    assert fake_llm.last_stream.closed is True


def test_no_secret_ever_appears_in_the_stream(headers, fake_llm):
    r = _stream(headers, "hello")
    assert SECRET_KEY_VALUE not in r.text


def test_the_non_streaming_endpoint_is_unchanged(headers, fake_llm):
    r = client.post("/chatbot/chat", json={"message": "hello"}, headers=headers)
    assert r.status_code == 200
    assert set(r.json()) == {"answer", "products"}
    assert "stream" not in fake_llm.calls[-1]


# --- unit level: early consumer exit (what a client disconnect causes) ---------------------------


def test_closing_the_generator_early_closes_the_provider_stream(fake_llm):
    gen = llm_service.stream_answer("hello", [])
    assert next(gen) == "Hel"
    gen.close()  # the consumer went away
    assert fake_llm.last_stream.closed is True


def test_closing_the_event_stream_early_closes_the_provider_stream(fake_llm):
    context = chatbot_service.ChatContext(message="hello")
    events = chatbot_service.stream_chat_events(context)
    assert next(events).startswith("event: products")
    assert next(events).startswith("event: token")
    events.close()
    assert fake_llm.last_stream.closed is True


def test_sse_frames_are_single_line_json(fake_llm):
    fake_llm.stream_pieces = ["line1\nline2", "end"]
    frames = list(chatbot_service.stream_chat_events(chatbot_service.ChatContext(message="hello")))
    token_frame = frames[1]
    assert token_frame.count("\n") == 3  # event line, one data line, blank separator
    assert "\\n" in token_frame  # the newline inside the text is JSON-escaped, not raw
    assert token_frame.startswith("event: token\ndata: ")


def test_stream_chunks_without_choices_or_content_are_skipped(fake_llm):
    from types import SimpleNamespace

    class Odd(FakeStream):
        def __iter__(self):
            yield SimpleNamespace(choices=[])
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))])

    fake_llm.chat.completions.create = lambda **kwargs: Odd([])
    assert list(llm_service.stream_answer("hello", [])) == ["ok"]
