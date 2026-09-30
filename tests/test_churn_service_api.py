"""Churn service + API tests (isolated SQLite database)."""

import os
import uuid
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from database.database import Base, get_db
from main import app
from ml.churn import model as cm
from ml.churn import predict as cp
from models.customer_segment import CustomerSegment
from models.interaction import InteractionType, ProductInteraction
from models.order import Order, OrderStatus
from models.product import Product
from models.user import User
from services import churn_service as svc
from tests.helpers_churn import WINDOW, naive_utc_now, seed_customers

_DB = "./test_churn_service.db"
_engine = create_engine(f"sqlite:///{_DB}", connect_args={"check_same_thread": False})
_Session = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
client = TestClient(app)

CONFIG = cm.TrainingConfig(inactivity_days=WINDOW, snapshot_count=3, min_samples=50, min_class_samples=5)


def _override():
    s = _Session()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture(scope="module", autouse=True)
def _isolated():
    Base.metadata.create_all(bind=_engine)
    app.dependency_overrides[get_db] = _override
    yield
    app.dependency_overrides.pop(get_db, None)
    _engine.dispose()
    if os.path.exists(_DB):
        os.remove(_DB)


@pytest.fixture
def db():
    s = _Session()
    for model in (CustomerSegment, ProductInteraction, Order, Product, User):
        s.query(model).delete()
    s.commit()
    yield s
    s.close()


@pytest.fixture
def now():
    return naive_utc_now()


def _register():
    email = f"api_{uuid.uuid4().hex[:10]}@example.com"
    pw = "StrongPass123!"
    r = client.post("/auth/register", json={"email": email, "full_name": "C", "password": pw, "confirm_password": pw})
    assert r.status_code == 200, r.text
    tok = client.post("/auth/login", data={"username": email, "password": pw}).json()["access_token"]
    s = _Session()
    uid = s.query(User).filter(User.email == email).first().id
    s.close()
    return {"Authorization": f"Bearer {tok}"}, uid


def _orders_for(db, uid, now, days_ago_list, status=OrderStatus.COMPLETED):
    for d in days_ago_list:
        db.add(Order(user_id=uid, status=status, total_price=40.0, created_at=now - timedelta(days=d)))
    db.commit()


# ---------------- data access: cutoff / leakage / labels ----------------

def test_aggregates_use_only_completed_orders_before_cutoff(db, now):
    (u,) = [User(email=f"a{uuid.uuid4().hex[:6]}@x.com", hashed_password="x")]
    db.add(u); db.commit()
    _orders_for(db, u.id, now, [50, 40])
    _orders_for(db, u.id, now, [35], OrderStatus.CANCELLED)
    _orders_for(db, u.id, now, [30], OrderStatus.PENDING)
    _orders_for(db, u.id, now, [5, 1])  # completed but AFTER the cutoff below
    cutoff = now - timedelta(days=20)
    (agg,) = svc.load_aggregates(db, cutoff, recent_days=30)
    assert agg.order_count == 2 and agg.total_spent == pytest.approx(80.0)
    assert agg.last_order_at <= cutoff
    assert svc.load_aggregates(db, now - timedelta(days=60), 30) == []  # no orders yet -> not a customer


def test_future_events_never_change_features_but_do_change_labels(db, now):
    u = User(email=f"a{uuid.uuid4().hex[:6]}@x.com", hashed_password="x")
    db.add(u); db.commit()
    p = Product(name="P", price=1.0, stock=1); db.add(p); db.commit()
    _orders_for(db, u.id, now, [100, 90])
    cutoff = now - timedelta(days=60)
    before = svc.load_aggregates(db, cutoff, 30)
    labels_before = svc.load_purchasers_in_window(db, cutoff, WINDOW)

    _orders_for(db, u.id, now, [45, 10])  # future relative to cutoff
    db.add(ProductInteraction(user_id=u.id, product_id=p.id, interaction_type=InteractionType.VIEW,
                              created_at=now - timedelta(days=20)))
    db.commit()

    assert svc.load_aggregates(db, cutoff, 30) == before  # features unchanged
    assert labels_before == set()
    assert svc.load_purchasers_in_window(db, cutoff, WINDOW) == {u.id}  # 45 days ago is inside (cutoff, cutoff+30]
    # 10 days ago is beyond the window end (cutoff + 30d = 30 days ago) and does not count:
    _ = svc.load_purchasers_in_window(db, cutoff, 5)
    assert _ == set()


def test_training_dataset_labels_match_definition(db, now):
    seed_customers(db, now, 40)
    ds = svc.build_training_dataset(db, now, CONFIG)
    assert len(ds.cutoffs) == 3 and ds.cutoffs[-1] == now - timedelta(days=WINDOW)
    # Loyal customers (even index) always buy in the window after the newest cutoff.
    users = db.query(User).order_by(User.id).all()
    loyal = {u.id for i, u in enumerate(users) if i % 2 == 0}
    newest = [i for i, s in enumerate(ds.snapshot_index) if s == 2]
    for i in newest:
        if ds.customer_ids[i] in loyal:
            assert ds.y[i] == 0


def test_feature_loading_query_count_is_constant(db, now):
    def count(fn):
        n = []
        cb = lambda *a, **k: n.append(1)
        event.listen(_engine, "before_cursor_execute", cb)
        try:
            fn()
        finally:
            event.remove(_engine, "before_cursor_execute", cb)
        return len(n)

    seed_customers(db, now, 5)
    small = count(lambda: svc.load_aggregates(db, now, 30))
    seed_customers(db, now, 40)
    assert count(lambda: svc.load_aggregates(db, now, 30)) == small == 2


# ---------------- training ----------------

def test_training_with_too_little_data_fails_safely(db, now):
    seed_customers(db, now, 4)
    with pytest.raises(cm.InsufficientChurnDataError):
        svc.train_and_persist(db, as_of=now, config=CONFIG)
    assert not os.path.exists(svc.settings.CHURN_MODEL_PATH)  # nothing persisted


def test_training_persists_loadable_model(db, now):
    seed_customers(db, now, 80)
    summary = svc.train_and_persist(db, as_of=now, config=CONFIG)
    assert summary["status"] == "trained" and summary["split_strategy"] == "temporal"
    assert 0.5 < summary["metrics"]["roc_auc"] <= 1.0
    art = cp.load_artifact(svc.settings.CHURN_MODEL_PATH)
    assert art.model_version == summary["model_version"]
    assert art.config["inactivity_days"] == WINDOW
    assert svc.get_model_info()["model_version"] == summary["model_version"]


# ---------------- API ----------------

def test_churn_me_requires_auth():
    assert client.get("/churn/me").status_code == 401
    assert client.get("/churn/me?user_id=1").status_code == 401


def test_churn_me_without_model_returns_documented_503(db):
    headers, _ = _register()
    r = client.get("/churn/me", headers=headers)
    assert r.status_code == 503
    assert "not available" in r.json()["detail"]


def test_churn_me_returns_probability(db, now):
    seed_customers(db, now, 80)
    svc.train_and_persist(db, as_of=now, config=CONFIG)
    headers, uid = _register()
    _orders_for(db, uid, now, [80, 60, 40, 20, 2])
    r = client.get("/churn/me", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["eligible"] is True and 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert body["model_version"] and body["risk_thresholds"] == {"medium": 0.4, "high": 0.7}


def test_churn_me_recent_buyer_is_lower_risk_than_lapsed_buyer(db, now):
    seed_customers(db, now, 80)
    svc.train_and_persist(db, as_of=now, config=CONFIG)
    (h1, u1), (h2, u2) = _register(), _register()
    _orders_for(db, u1, now, [90, 70, 50, 30, 10, 1])
    _orders_for(db, u2, now, [150, 140, 130, 120])
    p_recent = client.get("/churn/me", headers=h1).json()["churn_probability"]
    p_lapsed = client.get("/churn/me", headers=h2).json()["churn_probability"]
    assert p_recent < p_lapsed


def test_churn_me_cannot_impersonate(db, now):
    seed_customers(db, now, 80)
    svc.train_and_persist(db, as_of=now, config=CONFIG)
    (ha, ua), (hb, ub) = _register(), _register()
    _orders_for(db, ua, now, [90, 70, 50, 30, 10, 1])
    _orders_for(db, ub, now, [150, 140, 130, 120])
    own_b = client.get("/churn/me", headers=hb).json()["churn_probability"]
    for q in (f"?user_id={ua}", f"?userId={ua}", f"?customer_id={ua}"):
        assert client.get(f"/churn/me{q}", headers=hb).json()["churn_probability"] == pytest.approx(own_b, abs=1e-6)


def test_user_without_completed_orders_is_not_eligible(db, now):
    seed_customers(db, now, 80)
    svc.train_and_persist(db, as_of=now, config=CONFIG)
    headers, uid = _register()
    _orders_for(db, uid, now, [3], OrderStatus.CANCELLED)
    body = client.get("/churn/me", headers=headers).json()
    assert body["eligible"] is False and body["churn_probability"] is None
    assert body["risk_level"] is None and body["reason"] == "no_completed_orders"


def test_unusable_artifact_returns_503(db, now):
    seed_customers(db, now, 80)
    svc.train_and_persist(db, as_of=now, config=CONFIG)
    with open(svc.settings.CHURN_MODEL_PATH, "wb") as f:
        f.write(b"garbage")
    headers, _ = _register()
    assert client.get("/churn/me", headers=headers).status_code == 503


def test_model_reloaded_only_when_file_changes(db, now, monkeypatch):
    seed_customers(db, now, 80)
    svc.train_and_persist(db, as_of=now, config=CONFIG)
    calls = []
    real = cp.load_artifact
    monkeypatch.setattr(cp, "load_artifact", lambda p: calls.append(p) or real(p))
    svc.get_artifact(); svc.get_artifact(); svc.get_artifact()
    assert len(calls) == 1
    svc.train_and_persist(db, as_of=now, config=CONFIG)  # rewrites file
    svc.get_artifact()
    assert len(calls) == 2


def test_admin_churn_endpoints_are_protected(db):
    assert client.get("/churn/model").status_code == 401
    assert client.post("/churn/train").status_code == 401
    headers, _ = _register()
    assert client.get("/churn/model", headers=headers).status_code == 403
    assert client.post("/churn/train", headers=headers).status_code == 403
    assert client.get("/tasks/abc", headers=headers).status_code == 403
