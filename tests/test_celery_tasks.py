"""Phase 4: Celery application + task tests.

`CELERY_TASK_ALWAYS_EAGER=True` runs tasks synchronously in-process (no
broker/worker needed) while still going through Celery's real apply_async /
AsyncResult machinery, so these are true Celery tests, not just calling the
underlying functions.
"""

import uuid

import pytest

# Importing the app (unused directly below) guarantees Base.metadata.create_all
# has run against the shared database before any test in this file uses
# database.database.SessionLocal directly — needed so this file also works
# when run standalone, not just as part of the full suite.
from main import app  # noqa: F401
from database.database import SessionLocal
from models.interaction import InteractionType, ProductInteraction
from models.order import Order, OrderStatus
from models.product import Product
from models.user import User
from services import task_dispatch


@pytest.fixture
def eager(monkeypatch):
    """Fresh Celery app config: eager execution, for a clean import."""
    from celery_app import celery_app

    monkeypatch.setattr(celery_app.conf, "task_always_eager", True)
    celery_app.loader.import_default_modules()
    return celery_app


# ---------------- app / registration (22, 23, 24) ----------------

def test_celery_app_initializes_without_touching_broker():
    """Importing celery_app must not raise or open a connection (no Redis
    needs to be running for this)."""
    from celery_app import celery_app

    assert celery_app.main == "ai_ecommerce"
    assert celery_app.conf.task_serializer == "json"


def test_fastapi_import_does_not_build_celery_app():
    """main.py must never import celery_app at module load time."""
    import ast

    with open("main.py") as f:
        tree = ast.parse(f.read())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    module_names = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert "celery_app" not in imported
    assert "celery_app" not in module_names


def test_churn_training_task_registered(eager):
    assert "churn.train_model" in eager.tasks


def test_segmentation_refresh_task_registered(eager):
    assert "segmentation.refresh" in eager.tasks


def test_recommendation_cache_task_registered(eager):
    assert "recommendations.refresh_user_cache" in eager.tasks


# ---------------- eager execution (25) ----------------

def test_segmentation_task_runs_eagerly_and_reuses_existing_pipeline(eager, monkeypatch):
    calls = {}

    def fake_run(db, reference_date=None):
        calls["ran"] = True
        return {"customers_processed": 0, "segments": {}, "model_version": "v"}

    monkeypatch.setattr("tasks.segmentation.run_customer_segmentation", fake_run)
    result = eager.tasks["segmentation.refresh"].apply_async().get(timeout=5)
    assert calls.get("ran") is True
    assert result["status"] == "completed"


def test_churn_training_task_runs_eagerly_with_insufficient_data(eager):
    result = eager.tasks["churn.train_model"].apply_async().get(timeout=5)
    assert result["status"] == "insufficient_data"


def test_recommendation_cache_task_runs_eagerly(eager):
    session = SessionLocal()
    try:
        user = User(email=f"t_{uuid.uuid4().hex[:8]}@ex.com", hashed_password="x")
        session.add(user)
        session.commit()
        result = eager.tasks["recommendations.refresh_user_cache"].apply_async(
            args=[user.id], kwargs={"limit": 5}
        ).get(timeout=5)
        assert result["status"] == "refreshed" and result["user_id"] == user.id
        assert 0 <= result["items"] <= 5
    finally:
        session.query(User).filter(User.id == user.id).delete()
        session.commit()
        session.close()


# ---------------- task_dispatch submission (37, plus failure handling 26) ----------------

def test_task_dispatch_submit_returns_a_task_id(eager):
    task_id = task_dispatch.submit("segmentation.refresh")
    assert isinstance(task_id, str) and task_id


def test_task_dispatch_get_status_reflects_eager_result(monkeypatch):
    """A fresh app with a real (in-process, no external service) result
    backend, so a task's result is actually retrievable by id afterwards —
    the same round-trip GET /tasks/{id} relies on against a real worker's
    Redis backend."""
    from celery import Celery

    app = Celery(
        "test_status",
        broker="memory://",
        backend="cache+memory://",
        include=["tasks.churn"],
    )
    app.conf.task_always_eager = True
    app.conf.task_store_eager_result = True
    app.loader.import_default_modules()

    monkeypatch.setattr(task_dispatch, "get_celery_app", lambda: app)

    task_id = task_dispatch.submit("churn.train_model")
    status = task_dispatch.get_status(task_id)
    assert status["task_id"] == task_id
    assert status["state"] == "SUCCESS"
    assert status["result"]["status"] == "insufficient_data"


def test_task_dispatch_raises_documented_error_when_broker_unreachable(monkeypatch):
    """Failure handling (26): submission failures raise a typed error rather
    than pretending the task was queued."""
    from celery_app import celery_app

    monkeypatch.setattr(celery_app.conf, "task_always_eager", False)
    monkeypatch.setattr(
        "core.config.settings.CELERY_BROKER_URL", "redis://127.0.0.1:1/0"
    )
    # Force a fresh broker connection attempt against a closed port.
    celery_app.loader.import_default_modules()
    with pytest.raises(task_dispatch.TaskQueueUnavailableError):
        task_dispatch.submit("segmentation.refresh")


# ---------------- churn training task produces a loadable model (37) ----------------

def test_churn_training_task_produces_a_loadable_model(eager, tmp_path, monkeypatch):
    """Isolated database: the task calls `database.database.SessionLocal()`
    internally, so that attribute is monkeypatched to point at a throwaway
    engine/session rather than touching the shared app database other test
    files rely on."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from core.config import settings
    from database import database
    from database.database import Base
    from services import churn_service

    from tests.helpers_churn import WINDOW, seed_customers, naive_utc_now

    db_path = tmp_path / "celery_churn_test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    IsolatedSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(database, "SessionLocal", IsolatedSession)

    monkeypatch.setattr(settings, "CHURN_MODEL_PATH", str(tmp_path / "m.joblib"))
    # Match tests/helpers_churn.py's own churn cadence (WINDOW) and the sample
    # sizes used by test_training_persists_loadable_model, so the temporal
    # split has both classes in training and validation.
    monkeypatch.setattr(settings, "CHURN_INACTIVITY_DAYS", WINDOW)
    monkeypatch.setattr(settings, "CHURN_SNAPSHOT_COUNT", 3)
    monkeypatch.setattr(settings, "CHURN_MIN_SAMPLES", 50)
    monkeypatch.setattr(settings, "CHURN_MIN_CLASS_SAMPLES", 5)
    churn_service.clear_artifact_cache()

    session = IsolatedSession()
    try:
        now = naive_utc_now()
        seed_customers(session, now, n=80)

        result = eager.tasks["churn.train_model"].apply_async().get(timeout=30)
        assert result["status"] == "trained"

        artifact = churn_service.get_artifact()
        assert artifact.model_version == result["model_version"]
    finally:
        session.close()
        engine.dispose()
        churn_service.clear_artifact_cache()
