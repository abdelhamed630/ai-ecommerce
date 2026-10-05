import uuid

import pytest
from fastapi.testclient import TestClient

from database.database import SessionLocal
from main import app
from models.user import User, UserRole
from ml.segmentation.model import SEGMENTATION_MODEL_VERSION
import api.segmentation as segmentation_api
from services.segmentation_service import InsufficientSegmentationDataError

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers — mirrors tests/test_orders.py's conventions (real app, real db,
# unique emails, register/login through the actual API).
# ---------------------------------------------------------------------------

def _unique_email():
    return f"user_{uuid.uuid4().hex[:10]}@example.com"


def _register_and_login():
    email = _unique_email()
    password = "StrongPass123!"

    resp = client.post(
        "/auth/register",
        json={
            "email": email,
            "full_name": "Segmentation Test User",
            "password": password,
            "confirm_password": password,
        },
    )
    assert resp.status_code == 200, resp.text

    resp = client.post("/auth/login", data={"username": email, "password": password})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}, email


def _promote_role(email: str, role: UserRole) -> None:
    """No API self-promotes to SELLER or ADMIN (by design — see
    scripts/set_user_role.py), so tests promote a real registered user
    directly at the DB level, the same way tests/test_payments.py uses
    SessionLocal against the real app db."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        user.role = role
        db.commit()
    finally:
        db.close()


@pytest.fixture
def admin_headers():
    headers, email = _register_and_login()
    _promote_role(email, UserRole.ADMIN)
    return headers


@pytest.fixture
def seller_headers():
    headers, email = _register_and_login()
    _promote_role(email, UserRole.SELLER)
    return headers


@pytest.fixture
def normal_headers():
    headers, _ = _register_and_login()
    return headers


def _create_product(headers, name="Segmentation Test Product", price=100.0, stock=1000):
    resp = client.post(
        "/products/",
        json={
            "name": name,
            "description": "test product",
            "price": price,
            "stock": stock,
            "image_url": None,
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _create_completed_order(headers, product_id, admin_headers, quantity=1):
    """Adds `quantity` of `product_id` to the CUSTOMER's cart, creates an
    order from it, then advances it PENDING -> CONFIRMED -> COMPLETED via the
    real API (the same two-step pattern as tests/test_orders.py).

    Roles: the customer (`headers`, a regular user) does the cart + order
    steps. PATCH /orders/{id}/complete is ADMIN/SELLER-only, so the two
    advance steps are performed with `admin_headers`. The order is still
    created by — and stays owned by — the customer (order.user_id), which is
    what segmentation counts; that is asserted below."""
    resp = client.post(
        "/cart/items", json={"product_id": product_id, "quantity": quantity}, headers=headers
    )
    assert resp.status_code == 200, resp.text

    resp = client.post("/orders/", headers=headers)
    assert resp.status_code == 200, resp.text
    order = resp.json()

    resp = client.patch(f"/orders/{order['id']}/complete", headers=admin_headers)  # -> CONFIRMED
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "CONFIRMED"

    resp = client.patch(f"/orders/{order['id']}/complete", headers=admin_headers)  # -> COMPLETED
    assert resp.status_code == 200, resp.text
    completed = resp.json()
    assert completed["status"] == "COMPLETED"
    # Ownership / customer association must survive the admin's actions.
    assert completed["user_id"] == order["user_id"]
    return completed


_FAKE_RESULT = {
    "customers_processed": 8,
    "segments": {
        "Champions / Loyal": 2,
        "Potential Loyalist": 2,
        "Low Engagement": 2,
        "At Risk / Churned": 2,
    },
    "model_version": SEGMENTATION_MODEL_VERSION,
}


# ---------------------------------------------------------------------------
# 1-3. AuthN / AuthZ
# ---------------------------------------------------------------------------

def test_unauthenticated_request_rejected():
    resp = client.post("/segmentation/run")
    assert resp.status_code == 401


def test_authenticated_normal_user_rejected(normal_headers):
    resp = client.post("/segmentation/run", headers=normal_headers)
    assert resp.status_code == 403


def test_authenticated_seller_user_rejected(seller_headers):
    resp = client.post("/segmentation/run", headers=seller_headers)
    assert resp.status_code == 403


def test_authorized_admin_user_allowed(monkeypatch, admin_headers):
    monkeypatch.setattr(
        "api.segmentation.run_customer_segmentation", lambda *a, **k: dict(_FAKE_RESULT)
    )
    resp = client.post("/segmentation/run", headers=admin_headers)
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# 4, 8, 9. Response structure / delegation — mocked so these assertions
# don't depend on however many completed-order customers already exist in
# the shared app database from other test files (see also the real,
# end-to-end integration test further below).
# ---------------------------------------------------------------------------

def test_successful_response_structure(monkeypatch, admin_headers):
    monkeypatch.setattr(
        "api.segmentation.run_customer_segmentation", lambda *a, **k: dict(_FAKE_RESULT)
    )
    resp = client.post("/segmentation/run", headers=admin_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"customers_processed", "segments", "model_version"}


def test_model_version_appears_in_response(monkeypatch, admin_headers):
    monkeypatch.setattr(
        "api.segmentation.run_customer_segmentation", lambda *a, **k: dict(_FAKE_RESULT)
    )
    resp = client.post("/segmentation/run", headers=admin_headers)
    assert resp.json()["model_version"] == SEGMENTATION_MODEL_VERSION


def test_segment_counts_returned_correctly(monkeypatch, admin_headers):
    monkeypatch.setattr(
        "api.segmentation.run_customer_segmentation", lambda *a, **k: dict(_FAKE_RESULT)
    )
    resp = client.post("/segmentation/run", headers=admin_headers)
    body = resp.json()
    assert body["customers_processed"] == 8
    assert body["segments"] == _FAKE_RESULT["segments"]


def test_endpoint_calls_the_existing_orchestration_service(monkeypatch, admin_headers):
    calls = []

    def _spy(db, reference_date=None):
        calls.append(reference_date)
        return dict(_FAKE_RESULT)

    monkeypatch.setattr("api.segmentation.run_customer_segmentation", _spy)

    resp = client.post("/segmentation/run", headers=admin_headers)
    assert resp.status_code == 200
    assert len(calls) == 1  # exactly one delegation, no duplicated logic in the API layer


# ---------------------------------------------------------------------------
# 5-6. Error mapping (mocked, so the API's own error-handling is tested in
# isolation from the ML pipeline's data requirements)
# ---------------------------------------------------------------------------

def test_empty_completed_order_history_returns_safe_empty_result(monkeypatch, admin_headers):
    empty_result = {"customers_processed": 0, "segments": {}, "model_version": SEGMENTATION_MODEL_VERSION}
    monkeypatch.setattr(
        "api.segmentation.run_customer_segmentation", lambda *a, **k: dict(empty_result)
    )
    resp = client.post("/segmentation/run", headers=admin_headers)
    assert resp.status_code == 200
    assert resp.json() == empty_result


def test_insufficient_customers_returns_correct_http_error(monkeypatch, admin_headers):
    def _raise(*args, **kwargs):
        raise InsufficientSegmentationDataError(
            "Cannot train the configured segmentation model: found 2 customer(s) "
            "with completed orders, but the configured model requires at least 4."
        )

    monkeypatch.setattr("api.segmentation.run_customer_segmentation", _raise)

    resp = client.post("/segmentation/run", headers=admin_headers)
    assert resp.status_code == 400
    assert "4" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 7. reference_date validation
# ---------------------------------------------------------------------------

def test_invalid_reference_date_returns_validation_error(admin_headers):
    resp = client.post(
        "/segmentation/run", json={"reference_date": "not-a-date"}, headers=admin_headers
    )
    assert resp.status_code == 422


def test_valid_reference_date_is_passed_through(monkeypatch, admin_headers):
    calls = []

    def _spy(db, reference_date=None):
        calls.append(reference_date)
        return dict(_FAKE_RESULT)

    monkeypatch.setattr("api.segmentation.run_customer_segmentation", _spy)

    resp = client.post(
        "/segmentation/run", json={"reference_date": "2026-09-27"}, headers=admin_headers
    )
    assert resp.status_code == 200
    assert calls[0] is not None
    assert str(calls[0].date()) == "2026-09-27"


# ---------------------------------------------------------------------------
# 11. No arbitrary model parameters accepted
# ---------------------------------------------------------------------------

def test_endpoint_rejects_arbitrary_n_clusters(admin_headers):
    resp = client.post(
        "/segmentation/run",
        json={"n_clusters": 2, "reference_date": "2026-09-27"},
        headers=admin_headers,
    )
    assert resp.status_code == 422  # extra="forbid" rejects the unknown field


def test_endpoint_rejects_arbitrary_random_state(admin_headers):
    resp = client.post(
        "/segmentation/run", json={"random_state": 7}, headers=admin_headers
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Real, end-to-end integration: no mocking, hits the actual orchestration
# service and persists real CustomerSegment rows through the live API.
# Exact totals aren't asserted (the shared app db accumulates completed
# orders from other test files), only that the wiring genuinely works.
# ---------------------------------------------------------------------------

def test_real_end_to_end_run_via_api(admin_headers):
    product = _create_product(admin_headers, name=f"Widget-{uuid.uuid4().hex[:6]}")

    # 4 distinct customers with a real completed order each — enough to
    # satisfy the configured K=4 even in isolation from other test files.
    for _ in range(4):
        headers, _ = _register_and_login()
        _create_completed_order(headers, product["id"], admin_headers, quantity=1)

    resp = client.post("/segmentation/run", headers=admin_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["customers_processed"] >= 4
    assert body["model_version"] == SEGMENTATION_MODEL_VERSION
    assert isinstance(body["segments"], dict)
    assert sum(body["segments"].values()) == body["customers_processed"]


# ---------------------------------------------------------------------------
# 12. Existing APIs still work
# ---------------------------------------------------------------------------

def test_existing_products_endpoint_still_works():
    resp = client.get("/products/")
    assert resp.status_code == 200


def test_existing_auth_me_endpoint_still_works(normal_headers):
    resp = client.get("/auth/me", headers=normal_headers)
    assert resp.status_code == 200
    assert "email" in resp.json()


def test_root_endpoint_still_works():
    resp = client.get("/")
    assert resp.status_code == 200
