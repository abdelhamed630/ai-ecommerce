"""Phase 4: Redis cache tests (core.cache, cache_invalidation, and the two
cache-aside services), using the `redis_cache` fixture (fakeredis) from
tests/conftest.py — real redis-py client semantics, no external service.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from core.cache import cache, churn_key, recommendations_key, segmentation_key
from core.config import settings
from database.database import Base, get_db
from main import app
from models.interaction import InteractionType, ProductInteraction
from models.product import Product
from models.user import User
from services import cache_invalidation, recommendation_cache_service

_DB = "./test_redis_cache.db"
_engine = create_engine(f"sqlite:///{_DB}", connect_args={"check_same_thread": False})
_Session = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
client = TestClient(app)


def _override():
    s = _Session()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture(scope="module", autouse=True)
def _isolated_db():
    Base.metadata.create_all(bind=_engine)
    app.dependency_overrides[get_db] = _override
    yield
    app.dependency_overrides.pop(get_db, None)
    _engine.dispose()
    import os

    if os.path.exists(_DB):
        os.remove(_DB)


@pytest.fixture
def db():
    s = _Session()
    for model in (ProductInteraction, Product, User):
        s.query(model).delete()
    s.commit()
    yield s
    s.close()


def _user():
    email = f"cache_{uuid.uuid4().hex[:10]}@example.com"
    pw = "StrongPass123!"
    r = client.post(
        "/auth/register",
        json={"email": email, "full_name": "C", "password": pw, "confirm_password": pw},
    )
    assert r.status_code == 200, r.text
    token = client.post("/auth/login", data={"username": email, "password": pw}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    with _Session() as s:
        uid = s.query(User).filter(User.email == email).first().id
    return headers, uid


# ---------------- basic get/set/delete (28, 29, 30) ----------------

def test_cache_miss_returns_none(redis_cache):
    assert cache.get("nope") is None


def test_cache_hit_after_set(redis_cache):
    cache.set("k1", {"a": 1}, ttl=60)
    assert cache.get("k1") == {"a": 1}


def test_cache_hash_field_get_set(redis_cache):
    cache.set("h1", [1, 2, 3], ttl=60, field="limit:5")
    assert cache.get("h1", field="limit:5") == [1, 2, 3]
    assert cache.get("h1", field="limit:10") is None


def test_ttl_is_applied_and_configurable(redis_cache):
    cache.set("k2", "v", ttl=42)
    ttl = redis_cache.ttl("k2")
    assert 0 < ttl <= 42


def test_zero_or_negative_ttl_never_caches(redis_cache):
    assert cache.set("k3", "v", ttl=0) is False
    assert cache.get("k3") is None


def test_delete_removes_key(redis_cache):
    cache.set("k4", "v", ttl=60)
    cache.delete("k4")
    assert cache.get("k4") is None


def test_delete_prefix_removes_only_matching_keys(redis_cache):
    cache.set(segmentation_key(2), {"n_clusters": 2}, ttl=60)
    cache.set(segmentation_key(4), {"n_clusters": 4}, ttl=60)
    cache.set(recommendations_key(1), {"x": 1}, ttl=60, field="limit:10")
    from core.cache import segmentation_prefix

    cache.delete_prefix(segmentation_prefix())
    assert cache.get(segmentation_key(2)) is None
    assert cache.get(segmentation_key(4)) is None
    assert cache.get(recommendations_key(1), field="limit:10") == {"x": 1}


# ---------------- key isolation (27, 33, 34) ----------------

def test_recommendation_and_churn_keys_are_isolated_per_user(redis_cache):
    assert recommendations_key(1) != recommendations_key(2)
    assert churn_key(1) != churn_key(2)
    cache.set(recommendations_key(1), {"user": 1}, ttl=60, field="limit:10")
    cache.set(recommendations_key(2), {"user": 2}, ttl=60, field="limit:10")
    assert cache.get(recommendations_key(1), field="limit:10") == {"user": 1}
    assert cache.get(recommendations_key(2), field="limit:10") == {"user": 2}


def test_recommendation_cache_does_not_leak_between_users(redis_cache, db, monkeypatch):
    monkeypatch.setattr(settings, "RECOMMENDATION_CACHE_TTL_SECONDS", 60)
    headers_a, uid_a = _user()
    headers_b, uid_b = _user()
    # prime distinct cache entries directly (cheap, deterministic)
    recommendation_cache_service.get_user_recommendations_cached(db, uid_a, 10)
    recommendation_cache_service.get_user_recommendations_cached(db, uid_b, 10)
    a_cached = cache.get(recommendations_key(uid_a), field="limit:10")
    b_cached = cache.get(recommendations_key(uid_b), field="limit:10")
    assert a_cached is not None and b_cached is not None
    # invalidating A must never touch B
    cache_invalidation.invalidate_user_caches(uid_a)
    assert cache.get(recommendations_key(uid_a), field="limit:10") is None
    assert cache.get(recommendations_key(uid_b), field="limit:10") == b_cached


def test_churn_cache_does_not_leak_between_users(redis_cache, monkeypatch):
    monkeypatch.setattr(settings, "CHURN_CACHE_TTL_SECONDS", 60)
    cache.set(churn_key(1), {"churn_probability": 0.1}, ttl=60)
    cache.set(churn_key(2), {"churn_probability": 0.9}, ttl=60)
    assert cache.get(churn_key(1))["churn_probability"] == 0.1
    assert cache.get(churn_key(2))["churn_probability"] == 0.9
    cache_invalidation.invalidate_user_caches(1)
    assert cache.get(churn_key(1)) is None
    assert cache.get(churn_key(2))["churn_probability"] == 0.9


def test_churn_endpoint_cache_key_comes_only_from_jwt(redis_cache, monkeypatch):
    """Even if a client tried to address another user's cache, the key is
    built server-side from the authenticated id only (core.cache.churn_key
    takes a plain int, never request input) — verified at the unit level
    since /churn/me has no user_id parameter to attempt this through."""
    monkeypatch.setattr(settings, "CHURN_CACHE_TTL_SECONDS", 60)
    cache.set(churn_key(999), {"churn_probability": 0.5}, ttl=60)
    assert cache.get(churn_key(999)) is not None
    assert cache.get(churn_key(998)) is None


# ---------------- Redis unavailable (32) ----------------

def test_cache_disabled_setting_bypasses_redis_entirely(redis_cache, monkeypatch):
    monkeypatch.setattr(settings, "CACHE_ENABLED", False)
    assert cache.set("k5", "v", ttl=60) is False
    assert cache.get("k5") is None


def test_broken_client_degrades_to_miss_without_raising(monkeypatch):
    class Boom:
        def get(self, *a, **k):
            raise ConnectionError("down")

        def hget(self, *a, **k):
            raise ConnectionError("down")

        def set(self, *a, **k):
            raise ConnectionError("down")

        def delete(self, *a, **k):
            raise ConnectionError("down")

    monkeypatch.setattr(settings, "CACHE_ENABLED", True)
    cache.set_client(Boom())
    assert cache.get("anything") is None  # never raises
    assert cache.set("anything", "v", ttl=60) is False
    assert cache.delete("anything") is False
    cache.reset()


def test_route_using_cache_still_works_when_redis_is_down(db, monkeypatch):
    """The endpoint must return normal data (falling through to the DB/ML
    path), not a 500, when the cache backend is broken."""
    headers, uid = _user()

    class Boom:
        def get(self, *a, **k):
            raise ConnectionError("down")

        def hget(self, *a, **k):
            raise ConnectionError("down")

        def set(self, *a, **k):
            raise ConnectionError("down")

    monkeypatch.setattr(settings, "CACHE_ENABLED", True)
    cache.set_client(Boom())
    try:
        r = client.get("/recommendations/me", headers=headers)
        assert r.status_code == 200, r.text
        assert "recommendations" in r.json()
    finally:
        cache.reset()


def test_failure_backoff_skips_redis_briefly_after_an_error(monkeypatch):
    calls = {"n": 0}

    class FlakyThenBoom:
        def get(self, *a, **k):
            calls["n"] += 1
            raise ConnectionError("down")

    monkeypatch.setattr(settings, "CACHE_ENABLED", True)
    monkeypatch.setattr(settings, "CACHE_FAILURE_BACKOFF_SECONDS", 10.0)
    cache.set_client(FlakyThenBoom())
    cache.get("x")  # fails once, triggers back-off
    assert calls["n"] == 1
    cache.get("y")  # within back-off window: must not touch the client again
    assert calls["n"] == 1
    cache.reset()


# ---------------- cache invalidation on behaviour change (31, 35) ----------------

def test_interaction_invalidates_that_users_recommendation_and_churn_cache(redis_cache, db, monkeypatch):
    monkeypatch.setattr(settings, "RECOMMENDATION_CACHE_TTL_SECONDS", 60)
    monkeypatch.setattr(settings, "CHURN_CACHE_TTL_SECONDS", 60)
    headers, uid = _user()
    product = Product(name="Widget", price=5.0, stock=10)
    db.add(product)
    db.commit()

    cache.set(recommendations_key(uid), {"x": 1}, ttl=60, field="limit:10")
    cache.set(churn_key(uid), {"churn_probability": 0.2}, ttl=60)

    resp = client.post(
        "/interactions/",
        json={"product_id": product.id, "interaction_type": "VIEW"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text

    assert cache.get(recommendations_key(uid), field="limit:10") is None
    assert cache.get(churn_key(uid)) is None


def test_segmentation_cache_ttl_only_not_invalidated_by_orders(redis_cache, monkeypatch):
    """Documented in docs/customer_segmentation.md: segmentation summaries are
    TTL-only; ordinary user activity does not evict them."""
    monkeypatch.setattr(settings, "SEGMENTATION_CACHE_TTL_SECONDS", 60)
    cache.set(segmentation_key(4), {"n_clusters": 4}, ttl=60)
    cache_invalidation.invalidate_user_caches(123)
    assert cache.get(segmentation_key(4)) == {"n_clusters": 4}


# ---------------- query-count / no-op cost checks ----------------

def test_cached_recommendation_hit_skips_the_database(redis_cache, db, monkeypatch):
    monkeypatch.setattr(settings, "RECOMMENDATION_CACHE_TTL_SECONDS", 60)
    headers, uid = _user()
    recommendation_cache_service.get_user_recommendations_cached(db, uid, 10)  # warms cache

    calls = []
    listener = lambda *a, **k: calls.append(1)
    event.listen(_engine, "before_cursor_execute", listener)
    try:
        second = client.get("/recommendations/me", headers=headers)
    finally:
        event.remove(_engine, "before_cursor_execute", listener)
    assert second.status_code == 200
    # only the auth lookup (current_user) should query the db; recommendations
    # themselves must come from cache, not be recomputed.
    assert len(calls) <= 1


# ---------------- end-to-end integration (35-38) ----------------

def test_segmentation_refresh_endpoint_queues_real_task(db, monkeypatch):
    """POST /segmentation/refresh (admin) -> Celery task -> the same persisted
    segmentation pipeline as POST /segmentation/run (no duplicated algorithm)."""
    from celery_app import celery_app

    monkeypatch.setattr(celery_app.conf, "task_always_eager", True)
    celery_app.loader.import_default_modules()

    headers, uid = _user()
    with _Session() as s:
        s.query(User).filter(User.id == uid).first().role = __import__(
            "models.user", fromlist=["UserRole"]
        ).UserRole.ADMIN
        s.commit()

    resp = client.post("/segmentation/refresh", headers=headers)
    assert resp.status_code == 202, resp.text
    task_id = resp.json()["task_id"]

    # The task itself ran synchronously (task_always_eager) and reused
    # run_customer_segmentation without duplicating any clustering logic
    # (see test_segmentation_task_runs_eagerly_and_reuses_existing_pipeline).
    # GET /tasks/{id} then queries the REAL configured result backend
    # (settings.CELERY_RESULT_BACKEND); this sandbox has no Redis server
    # running, so that lookup correctly reports 503 "unavailable" rather than
    # fabricating a status — the same failure-handling contract
    # test_task_dispatch_get_status_reflects_eager_result exercises directly
    # against a real (in-process) backend.
    status_resp = client.get(f"/tasks/{task_id}", headers=headers)
    assert status_resp.status_code in (200, 503), status_resp.text
    if status_resp.status_code == 200:
        assert status_resp.json()["state"] in ("SUCCESS", "PENDING")
