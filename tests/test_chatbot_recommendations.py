"""Chatbot <-> recommendation integration: intent routing, no duplicated algorithm, safe fallback."""
import pytest
from pydantic import SecretStr

from core import rate_limit
from core.config import settings
from database.database import SessionLocal
from models.product import Product
from services import llm_service, product_search_service, recommendation_cache_service
from tests.helpers_chatbot import FakeGroqClient, client, new_user_headers, token


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


@pytest.fixture
def seed_product():
    ids = []

    def _seed(**fields):
        fields.setdefault("price", 10.0)
        fields.setdefault("stock", 3)
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


def _chat(headers, message):
    return client.post("/chatbot/chat", json={"message": message}, headers=headers)


@pytest.mark.parametrize(
    "message, expected",
    [
        ("رشحلي لابتوب", ["لابتوب"]),
        ("رشحلي لابتوب جيمينج", ["لابتوب", "جيمينج"]),
        ("recommend a laptop", ["laptop"]),
        ("what do you recommend laptop", ["laptop"]),
    ],
)
def test_the_request_verb_is_never_a_search_term(message, expected):
    terms = product_search_service.extract_search_terms(message)
    assert terms == expected
    assert "رشحلي" not in terms and "recommend" not in terms


@pytest.mark.parametrize("message", ["رشحلي حاجة", "what do you recommend?", "recommend something"])
def test_generic_requests_use_the_recommendation_service(message, seed_product, monkeypatch, fake_llm):
    pid = seed_product(name=f"Rec {token()}")
    calls = []

    def fake_engine(db, user_id, limit):
        calls.append((user_id, limit))
        return [{"product": {"id": pid, "name": "Rec", "price": 10.0, "stock": 3}}]

    monkeypatch.setattr(recommendation_cache_service, "get_user_recommendations_cached", fake_engine)
    body = _chat(new_user_headers(), message).json()
    assert [p["id"] for p in body["products"]] == [pid]
    assert len(calls) == 1 and calls[0][1] == 5
    assert "personalized recommendations" in fake_llm.calls[0]["messages"][-1]["content"]


def test_the_engine_is_asked_for_the_authenticated_user_only(monkeypatch):
    seen = []
    monkeypatch.setattr(
        recommendation_cache_service,
        "get_user_recommendations_cached",
        lambda db, user_id, limit: seen.append(user_id) or [],
    )
    _chat(new_user_headers(), "رشحلي حاجة")
    _chat(new_user_headers(), "رشحلي حاجة")
    assert len(seen) == 2 and seen[0] != seen[1]


def test_a_product_request_is_searched_not_recommended(seed_product, monkeypatch, fake_llm):
    t = token()
    pid = seed_product(name=f"لابتوب {t}")
    monkeypatch.setattr(
        recommendation_cache_service,
        "get_user_recommendations_cached",
        lambda *a, **k: pytest.fail("the recommendation engine must not run for a product request"),
    )
    body = _chat(new_user_headers(), f"رشحلي لابتوب {t}").json()
    assert [p["id"] for p in body["products"]] == [pid]
    assert "personalized recommendations" not in fake_llm.calls[0]["messages"][-1]["content"]


def test_engine_failure_falls_back_to_search_without_leaking_details(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("engine down: internal-secret-trace")

    monkeypatch.setattr(recommendation_cache_service, "get_user_recommendations_cached", boom)
    r = _chat(new_user_headers(), "رشحلي حاجة")
    assert r.status_code == 200
    assert "internal-secret-trace" not in r.text and "Traceback" not in r.text
    assert set(r.json()) == {"answer", "products"}


def test_engine_failure_on_a_specific_request_still_searches(seed_product, monkeypatch):
    t = token()
    pid = seed_product(name=f"Phone {t}")
    monkeypatch.setattr(
        recommendation_cache_service,
        "get_user_recommendations_cached",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")),
    )
    body = _chat(new_user_headers(), f"recommend {t}").json()
    assert [p["id"] for p in body["products"]] == [pid]
