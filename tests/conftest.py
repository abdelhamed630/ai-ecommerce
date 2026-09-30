"""Shared test isolation (Phase 4 + Phase 5).

Test-database isolation (Phase 5)
---------------------------------
This module runs BEFORE any application module is imported, and forces the
application's default database to a dedicated TEST database:

- TEST_DATABASE_URL, if set (e.g. a PostgreSQL database), otherwise
- a throw-away SQLite file in the system temp directory (deleted at the end).

The developer's DATABASE_URL / .env database (ecommerce.db, a PostgreSQL dev or
production database) is therefore never touched by the suite. For a non-SQLite
TEST_DATABASE_URL the database name must contain "test": the schema is DROPPED
and recreated, so anything else is refused. Redis and Celery URLs are likewise
pointed at dedicated Redis DB numbers (override with TEST_REDIS_URL).

Other test modules keep their own per-module SQLite databases (unchanged).

Phase 4 isolation
-----------------
- Cache is DISABLED by default so pre-existing tests (which mutate data
  directly and expect fresh results) never touch Redis. Cache tests opt in
  with the `redis_cache` fixture (fakeredis) below.
- The churn model artifact path is redirected to a per-test temp directory so
  no test can read or write the real ml/artifacts/ model.
"""

import os
import shutil
import tempfile
from pathlib import Path

_TMP_DIR = Path(tempfile.mkdtemp(prefix="ai_ecommerce_tests_"))
_TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{(_TMP_DIR / 'default_test.db').as_posix()}"

if not _TEST_DATABASE_URL.startswith("sqlite"):
    _db_name = _TEST_DATABASE_URL.rsplit("/", 1)[-1].split("?", 1)[0]
    if "test" not in _db_name.lower():
        raise RuntimeError(
            "Refusing to run tests: TEST_DATABASE_URL must point to a database whose name contains 'test' "
            "(its schema is dropped and recreated)."
        )

_TEST_REDIS = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379").rstrip("/")
os.environ["ENVIRONMENT"] = "test"
os.environ["DEBUG"] = "false"
os.environ["DATABASE_URL"] = _TEST_DATABASE_URL
os.environ["SECRET_KEY"] = "test-only-secret-key-not-for-production-0123456789"
os.environ["REDIS_URL"] = f"{_TEST_REDIS}/15"
os.environ["CELERY_BROKER_URL"] = f"{_TEST_REDIS}/14"
os.environ["CELERY_RESULT_BACKEND"] = f"{_TEST_REDIS}/13"
os.environ["MEDIA_ROOT"] = str(_TMP_DIR / "media")
os.environ["ALLOWED_HOSTS"] = "*"
os.environ["CORS_ALLOWED_ORIGINS"] = ""

# Load settings NOW (they pick the values above up), then remove the two Celery
# variables from the process environment. Celery itself reads CELERY_BROKER_URL /
# CELERY_RESULT_BACKEND from os.environ at runtime and lets them OVERRIDE the
# broker/backend passed to any `Celery(...)` constructor, which would silently
# redirect test-local apps (e.g. broker="memory://") to Redis. settings keeps the
# isolated values, so the project's own celery_app still uses the dedicated
# test Redis DB numbers.
import core.config as _core_config  # noqa: E402

assert _core_config.settings.CELERY_BROKER_URL.endswith("/14")
os.environ.pop("CELERY_BROKER_URL", None)
os.environ.pop("CELERY_RESULT_BACKEND", None)

import fakeredis
import pytest

from core.cache import cache
from core.config import settings
from database.database import Base, engine
from services import churn_service
import models  # noqa: F401,E402  (registers every model on Base.metadata)
from models import cart, customer_segment, interaction, order, payment, product, user  # noqa: F401,E402


@pytest.fixture(scope="session", autouse=True)
def _default_test_database():
    """Schema for the application's default (test) database.

    main.py no longer runs create_all(); production uses Alembic. Tests that use
    the default SessionLocal get their schema here, on the isolated test DB only.
    """
    # (compare without the driver part: settings may rewrite postgresql:// -> postgresql+psycopg://)
    assert settings.DATABASE_URL.split("://", 1)[1] == _TEST_DATABASE_URL.split("://", 1)[1], (
        "tests must run against the test database"
    )
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    engine.dispose()  # release SQLite file handles (Windows) before deleting
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def _phase4_isolation(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "CACHE_ENABLED", False)
    monkeypatch.setattr(settings, "CHURN_MODEL_PATH", str(tmp_path / "churn" / "churn_model.joblib"))
    cache.reset()
    churn_service.clear_artifact_cache()
    yield
    cache.reset()
    churn_service.clear_artifact_cache()


@pytest.fixture
def redis_cache(monkeypatch):
    """Cache enabled against an in-memory fake Redis (real redis-py semantics)."""
    client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(settings, "CACHE_ENABLED", True)
    cache.set_client(client)
    yield client
    client.flushall()
    cache.reset()
