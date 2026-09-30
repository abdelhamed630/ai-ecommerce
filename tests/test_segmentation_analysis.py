"""Service + API tests for Phase 3 segmentation analysis.

Uses its OWN isolated SQLite database (same pattern as
tests/test_recommendations.py) so customer counts and statistics are exact.
"""

import os
import uuid
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from core.config import settings
from database.database import Base, get_db
from main import app
from models.customer_segment import CustomerSegment
from models.interaction import InteractionType, ProductInteraction
from models.order import Order, OrderStatus
from models.product import Product
from models.user import User, UserRole
from services import segmentation_analysis_service as svc

_DB_PATH = "./test_segmentation_analysis.db"
_engine = create_engine(f"sqlite:///{_DB_PATH}", connect_args={"check_same_thread": False})
_Session = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

client = TestClient(app)
NOW = datetime(2026, 9, 28, 12, 0, 0)


def _override_get_db():
    db = _Session()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module", autouse=True)
def _isolated_db():
    Base.metadata.create_all(bind=_engine)
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)
    _engine.dispose()
    if os.path.exists(_DB_PATH):
        os.remove(_DB_PATH)


@pytest.fixture
def db():
    """Fresh, empty tables for every test."""
    session = _Session()
    for model in (CustomerSegment, ProductInteraction, Order, Product, User):
        session.query(model).delete()
    session.commit()
    yield session
    session.close()


# ---------------- data helpers ----------------

def _user(db, role=UserRole.USER):
    user = User(email=f"u_{uuid.uuid4().hex[:10]}@example.com", hashed_password="x", role=role)
    db.add(user)
    db.commit()
    return user


def _product(db):
    product = Product(name="Widget", price=1.0, stock=1)
    db.add(product)
    db.commit()
    return product


def _order(db, user, total, days_ago, status=OrderStatus.COMPLETED):
    db.add(
        Order(user_id=user.id, status=status, total_price=total, created_at=NOW - timedelta(days=days_ago))
    )
    db.commit()


def _interact(db, user, product, kind, times=1):
    for _ in range(times):
        db.add(ProductInteraction(user_id=user.id, product_id=product.id, interaction_type=kind))
    db.commit()


def _analyze(db, **kw):
    return svc.analyze_customer_segments(db, reference_date=NOW, **kw).result


def _login(email_user):
    """Real JWT for a DB-created user: register through the API for the
    account, then return headers for it."""
    email = f"api_{uuid.uuid4().hex[:10]}@example.com"
    pw = "StrongPass123!"
    r = client.post(
        "/auth/register",
        json={"email": email, "full_name": "T", "password": pw, "confirm_password": pw},
    )
    assert r.status_code == 200, r.text
    r = client.post("/auth/login", data={"username": email, "password": pw})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}, email


def _headers(role=UserRole.USER):
    headers, email = _login(None)
    if role != UserRole.USER:
        s = _Session()
        s.query(User).filter(User.email == email).first().role = role
        s.commit()
        s.close()
    return headers, email


# ---------------- service: eligibility ----------------

def test_empty_database(db):
    result = _analyze(db)
    assert result.customer_count == 0 and result.profiles == [] and result.assignments == {}


def test_only_completed_orders_count(db):
    buyer, other = _user(db), _user(db)
    _order(db, buyer, 100.0, 5)
    _order(db, buyer, 50.0, 2)
    for status in (OrderStatus.CANCELLED, OrderStatus.PENDING, OrderStatus.CONFIRMED):
        _order(db, buyer, 9999.0, 1, status)
        _order(db, other, 9999.0, 1, status)  # never completed -> not a customer

    (activity,) = svc.load_customer_activities(db)
    assert activity.customer_id == buyer.id
    assert activity.order_count == 2
    assert activity.total_spent == pytest.approx(150.0)

    result = _analyze(db)
    assert result.customer_count == 1
    (profile,) = result.profiles
    assert profile.average_frequency == 2
    assert profile.average_monetary == pytest.approx(150.0)
    assert profile.average_recency == 2  # latest COMPLETED order, not the pending one
    assert profile.average_order_value == pytest.approx(75.0)


def test_cancelled_only_customer_is_not_segmented(db):
    user = _user(db)
    _order(db, user, 500.0, 3, OrderStatus.CANCELLED)
    assert _analyze(db).customer_count == 0


def test_views_and_cart_adds_are_counted_for_customers(db):
    buyer, p = _user(db), _product(db)
    _order(db, buyer, 10.0, 1)
    _interact(db, buyer, p, InteractionType.VIEW, 3)
    _interact(db, buyer, p, InteractionType.CART_ADD, 2)
    _interact(db, buyer, p, InteractionType.PURCHASE, 5)  # not a behavioural feature
    (a,) = svc.load_customer_activities(db)
    assert (a.views_count, a.cart_add_count) == (3, 2)


def test_browse_only_users_excluded_by_default_included_when_configured(db, monkeypatch):
    buyer, browser, p = _user(db), _user(db), _product(db)
    _order(db, buyer, 10.0, 1)
    _interact(db, browser, p, InteractionType.VIEW, 4)

    assert [a.customer_id for a in svc.load_customer_activities(db)] == [buyer.id]

    monkeypatch.setattr(settings, "SEGMENTATION_INCLUDE_NON_PURCHASERS", True)
    ids = {a.customer_id: a for a in svc.load_customer_activities(db, True)}
    assert set(ids) == {buyer.id, browser.id}
    assert ids[browser.id].order_count == 0 and ids[browser.id].views_count == 4
    assert _analyze(db).customer_count == 2  # zero-monetary customer handled


def test_small_datasets_do_not_crash(db):
    a, b = _user(db), _user(db)
    _order(db, a, 500.0, 1)
    assert _analyze(db, n_clusters=4).n_clusters == 1
    _order(db, b, 5.0, 200)
    r = _analyze(db, n_clusters=4)
    assert r.n_clusters == 2 and r.requested_n_clusters == 4


def test_multiple_customers_statistics(db):
    users = [_user(db) for _ in range(6)]
    for u in users[:3]:
        _order(db, u, 1000.0, 2)
        _order(db, u, 1000.0, 10)
    for u in users[3:]:
        _order(db, u, 10.0, 300)
    r = _analyze(db, n_clusters=2)
    assert [p.customer_count for p in r.profiles] == [3, 3]
    assert r.profiles[0].average_monetary == pytest.approx(2000.0)
    assert r.profiles[0].average_frequency == 2
    assert r.profiles[1].average_recency == 300


def test_query_count_does_not_grow_with_customers(db):
    def count_queries(fn):
        n = []
        listener = lambda *a, **k: n.append(1)
        event.listen(_engine, "before_cursor_execute", listener)
        try:
            fn()
        finally:
            event.remove(_engine, "before_cursor_execute", listener)
        return len(n)

    for _ in range(3):
        u = _user(db)
        _order(db, u, 10.0, 1)
    small = count_queries(lambda: svc.load_customer_activities(db))
    for _ in range(30):
        u = _user(db)
        _order(db, u, 10.0, 1)
    assert count_queries(lambda: svc.load_customer_activities(db)) == small


# ---------------- API: authorization ----------------

def test_segments_requires_authentication():
    assert client.get("/segmentation/segments").status_code == 401


@pytest.mark.parametrize("role", [UserRole.USER, UserRole.SELLER])
def test_segments_forbidden_for_non_admin(role):
    headers, _ = _headers(role)
    assert client.get("/segmentation/segments", headers=headers).status_code == 403


def test_segments_empty_dataset_ok_for_admin(db):
    headers, _ = _headers(UserRole.ADMIN)
    r = client.get("/segmentation/segments", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["customer_count"] == 0 and body["segments"] == [] and body["n_clusters"] == 0
    assert body["assignments"] is None


def test_segments_admin_response_shape_and_no_personal_data(db):
    users = [_user(db) for _ in range(5)]
    for i, u in enumerate(users):
        _order(db, u, 100.0 * (i + 1), 5 * (i + 1))
    headers, _ = _headers(UserRole.ADMIN)
    r = client.get("/segmentation/segments?n_clusters=2&include_assignments=true", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["algorithm"] == "kmeans"
    assert body["requested_n_clusters"] == 2 and body["n_clusters"] == 2
    assert body["features_used"] == settings.SEGMENTATION_CLUSTER_FEATURES
    assert sum(s["customer_count"] for s in body["segments"]) == 5
    assert set(body["segments"][0]) == {
        "segment_id", "customer_count", "average_recency", "average_frequency",
        "average_monetary", "average_order_value", "average_views", "average_cart_adds",
    }
    assert body["assignments_total"] == 5
    assert all(set(a) == {"customer_id", "segment_id"} for a in body["assignments"])
    assert "@" not in r.text  # no emails anywhere


def test_segments_assignment_pagination(db):
    for i in range(5):
        _order(db, _user(db), 10.0 * (i + 1), i + 1)
    headers, _ = _headers(UserRole.ADMIN)
    page = client.get(
        "/segmentation/segments?n_clusters=2&include_assignments=true&limit=2&offset=1", headers=headers
    ).json()
    assert page["assignments_total"] == 5 and len(page["assignments"]) == 2


def test_segments_default_k_from_config_and_cap(db):
    for i in range(6):
        _order(db, _user(db), 10.0 * (i + 1) ** 2, 10 * (i + 1))
    headers, _ = _headers(UserRole.ADMIN)
    body = client.get("/segmentation/segments", headers=headers).json()
    assert body["requested_n_clusters"] == settings.SEGMENTATION_N_CLUSTERS
    too_big = settings.SEGMENTATION_MAX_CLUSTERS + 1
    assert client.get(f"/segmentation/segments?n_clusters={too_big}", headers=headers).status_code == 422
    assert client.get("/segmentation/segments?n_clusters=0", headers=headers).status_code == 422


def test_segments_endpoint_is_read_only(db):
    for i in range(4):
        _order(db, _user(db), 10.0 * (i + 1), i + 1)
    headers, _ = _headers(UserRole.ADMIN)
    client.get("/segmentation/segments", headers=headers)
    assert db.query(CustomerSegment).count() == 0


# ---------------- API: current user's segment ----------------

def test_my_segment_requires_authentication():
    assert client.get("/segmentation/me").status_code == 401


def test_my_segment_not_segmented_yet(db):
    headers, _ = _headers()
    r = client.get("/segmentation/me", headers=headers)
    assert r.status_code == 200
    assert r.json()["segmented"] is False and r.json()["segment_label"] is None


def test_my_segment_returns_only_own_row_and_ignores_user_id(db):
    headers, email = _headers()
    other = _user(db)
    me = db.query(User).filter(User.email == email).first()
    when = datetime(2026, 9, 1)
    for user, cluster, label in ((me, 1, "Low Engagement"), (other, 0, "Champions / Loyal")):
        db.add(
            CustomerSegment(
                user_id=user.id, cluster_id=cluster, segment_label=label, recency=5,
                frequency=2, monetary=42.0, model_version="rfm_kmeans_v1", calculated_at=when,
            )
        )
    db.commit()

    r = client.get(f"/segmentation/me?user_id={other.id}&customer_id={other.id}", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["segmented"] is True
    assert body["segment_label"] == "Low Engagement" and body["segment_id"] == 1
    assert "user_id" not in body
