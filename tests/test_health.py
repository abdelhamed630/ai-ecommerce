"""Health endpoints (Phase 5). NOT VERIFIED when written: not executed by the author."""

import fakeredis
import pytest
from fastapi.testclient import TestClient

from core.cache import cache
from core.config import settings
from main import app

client = TestClient(app)


def test_health_is_liveness_and_needs_no_dependencies(monkeypatch):
    # Must succeed even if the database dependency is unusable.
    from database.database import get_db

    def _broken_db():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    app.dependency_overrides[get_db] = _broken_db
    try:
        for path in ("/health", "/health/live"):
            resp = client.get(path)
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_ready_ok_with_database_and_cache_disabled():
    resp = client.get("/health/ready")  # conftest disables the cache
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "checks": {"database": "ok", "redis": "skipped"}}


def test_ready_reports_redis_ok(monkeypatch):
    monkeypatch.setattr(settings, "CACHE_ENABLED", True)
    cache.set_client(fakeredis.FakeRedis(decode_responses=True))
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["checks"] == {"database": "ok", "redis": "ok"}


class _DeadRedis:
    def ping(self):
        raise ConnectionError("redis://:supersecret@redis:6379/0 refused")


def test_ready_redis_down_is_503_when_required(monkeypatch):
    monkeypatch.setattr(settings, "CACHE_ENABLED", True)
    monkeypatch.setattr(settings, "READINESS_REQUIRE_REDIS", True)
    cache.set_client(_DeadRedis())
    resp = client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["checks"] == {"database": "ok", "redis": "unavailable"}
    assert "supersecret" not in resp.text  # no connection details leak


def test_ready_redis_down_is_200_when_optional(monkeypatch):
    monkeypatch.setattr(settings, "CACHE_ENABLED", True)
    monkeypatch.setattr(settings, "READINESS_REQUIRE_REDIS", False)
    cache.set_client(_DeadRedis())
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["checks"]["redis"] == "unavailable"


def test_ready_database_down_is_503_without_leaking_details():
    from database.database import get_db

    class _BrokenSession:
        def execute(self, *a, **k):
            raise RuntimeError("could not connect postgresql://app:dbpassword@postgres/shop")

    def _override():
        yield _BrokenSession()

    app.dependency_overrides[get_db] = _override
    try:
        resp = client.get("/health/ready")
    finally:
        app.dependency_overrides.pop(get_db, None)
    assert resp.status_code == 503
    assert resp.json()["checks"]["database"] == "unavailable"
    assert "dbpassword" not in resp.text and "postgresql://" not in resp.text
