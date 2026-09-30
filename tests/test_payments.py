import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from database.database import SessionLocal
from main import app
from models.payment import Payment
from models.user import User, UserRole
from services import payment_service
from schemas.payment import PaymentCreate

client = TestClient(app)


def _unique_email():
    return f"user_{uuid.uuid4().hex[:10]}@example.com"


def _promote_to_seller(email: str) -> None:
    """No API self-promotes to SELLER (by design), so test fixtures that
    need to create a product promote a real registered user directly at
    the DB level — same convention as
    tests/test_segmentation_api.py's _promote_to_admin. Payment behavior
    itself is unaffected: nothing in this project restricts payment
    actions by role."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        user.role = UserRole.SELLER
        db.commit()
    finally:
        db.close()


def _register_and_login():
    email = _unique_email()
    password = "StrongPass123!"

    resp = client.post(
        "/auth/register",
        json={
            "email": email,
            "full_name": "Payment Test User",
            "password": password,
            "confirm_password": password,
        },
    )
    assert resp.status_code == 200, resp.text

    # Promoted to SELLER so this same account can also create the test
    # products it then orders/pays for below (product creation requires
    # SELLER or ADMIN; this file is not testing product authorization, so
    # this is a fixture-only change).
    _promote_to_seller(email)

    resp = client.post("/auth/login", data={"username": email, "password": password})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _create_product(headers, name="Laptop", price=1000.0, stock=10):
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


def _create_order(headers, price=1000.0, stock=10, quantity=1):
    product = _create_product(headers, price=price, stock=stock)
    resp = client.post(
        "/cart/items", json={"product_id": product["id"], "quantity": quantity}, headers=headers
    )
    assert resp.status_code == 200, resp.text
    order_resp = client.post("/orders/", headers=headers)
    assert order_resp.status_code == 200, order_resp.text
    return order_resp.json()


@pytest.fixture
def auth_headers():
    return _register_and_login()


# 1. Create payment successfully
def test_create_payment_successfully(auth_headers):
    order = _create_order(auth_headers, price=100.0, quantity=2)
    resp = client.post(
        "/payments/",
        json={"order_id": order["id"], "payment_method": "CARD"},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text


# 2. Payment starts as PENDING
def test_payment_starts_pending(auth_headers):
    order = _create_order(auth_headers, price=50.0, quantity=1)
    resp = client.post(
        "/payments/",
        json={"order_id": order["id"], "payment_method": "CARD"},
        headers=auth_headers,
    )
    assert resp.json()["status"] == "PENDING"


# 3. Correct amount is copied from Order.total_price
def test_amount_matches_order_total(auth_headers):
    order = _create_order(auth_headers, price=75.0, quantity=3)
    resp = client.post(
        "/payments/",
        json={"order_id": order["id"], "payment_method": "CARD"},
        headers=auth_headers,
    )
    assert resp.json()["amount"] == pytest.approx(order["total_price"])
    assert resp.json()["amount"] == pytest.approx(225.0)


# 4. Client cannot choose an arbitrary amount
def test_client_cannot_choose_amount(auth_headers):
    order = _create_order(auth_headers, price=100.0, quantity=1)
    resp = client.post(
        "/payments/",
        json={"order_id": order["id"], "payment_method": "CARD", "amount": 1.0},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    # the bogus "amount" field is simply ignored by the schema
    assert resp.json()["amount"] == pytest.approx(100.0)


# 5. Create CARD payment
def test_create_card_payment(auth_headers):
    order = _create_order(auth_headers)
    resp = client.post(
        "/payments/",
        json={"order_id": order["id"], "payment_method": "CARD"},
        headers=auth_headers,
    )
    assert resp.json()["payment_method"] == "CARD"


# 6. Create CASH_ON_DELIVERY payment
def test_create_cod_payment(auth_headers):
    order = _create_order(auth_headers)
    resp = client.post(
        "/payments/",
        json={"order_id": order["id"], "payment_method": "CASH_ON_DELIVERY"},
        headers=auth_headers,
    )
    assert resp.json()["payment_method"] == "CASH_ON_DELIVERY"
    assert resp.json()["status"] == "PENDING"


# 7. Cannot create payment for non-existing order
def test_cannot_pay_nonexistent_order(auth_headers):
    resp = client.post(
        "/payments/",
        json={"order_id": 999999, "payment_method": "CARD"},
        headers=auth_headers,
    )
    assert resp.status_code == 404


# 8. Cannot create payment for another user's order
def test_cannot_pay_another_users_order():
    headers_a = _register_and_login()
    headers_b = _register_and_login()
    order = _create_order(headers_a)

    resp = client.post(
        "/payments/",
        json={"order_id": order["id"], "payment_method": "CARD"},
        headers=headers_b,
    )
    assert resp.status_code == 404


# 9. Cannot create payment for CANCELLED order
def test_cannot_pay_cancelled_order(auth_headers):
    order = _create_order(auth_headers)
    client.patch(f"/orders/{order['id']}/cancel", headers=auth_headers)

    resp = client.post(
        "/payments/",
        json={"order_id": order["id"], "payment_method": "CARD"},
        headers=auth_headers,
    )
    assert resp.status_code == 400


# 10. Cannot create second successful payment for same order
def test_cannot_create_second_paid_payment(auth_headers):
    order = _create_order(auth_headers)
    resp1 = client.post(
        "/payments/",
        json={"order_id": order["id"], "payment_method": "CARD"},
        headers=auth_headers,
    )
    payment_id = resp1.json()["id"]
    client.patch(f"/payments/{payment_id}/pay", headers=auth_headers)  # PENDING -> PAID

    resp2 = client.post(
        "/payments/",
        json={"order_id": order["id"], "payment_method": "CARD"},
        headers=auth_headers,
    )
    assert resp2.status_code == 400


# 11. Get payment successfully
def test_get_payment_successfully(auth_headers):
    order = _create_order(auth_headers)
    created = client.post(
        "/payments/",
        json={"order_id": order["id"], "payment_method": "CARD"},
        headers=auth_headers,
    ).json()

    resp = client.get(f"/payments/{created['id']}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["id"] == created["id"]


# 12. Cannot get another user's payment
def test_cannot_get_another_users_payment():
    headers_a = _register_and_login()
    headers_b = _register_and_login()
    order = _create_order(headers_a)
    created = client.post(
        "/payments/",
        json={"order_id": order["id"], "payment_method": "CARD"},
        headers=headers_a,
    ).json()

    resp = client.get(f"/payments/{created['id']}", headers=headers_b)
    assert resp.status_code == 404


# 13. Get payment by order
def test_get_payment_by_order(auth_headers):
    order = _create_order(auth_headers)
    created = client.post(
        "/payments/",
        json={"order_id": order["id"], "payment_method": "CARD"},
        headers=auth_headers,
    ).json()

    resp = client.get(f"/payments/order/{order['id']}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["id"] == created["id"]


# 14. Cannot get another user's order payment
def test_cannot_get_another_users_order_payment():
    headers_a = _register_and_login()
    headers_b = _register_and_login()
    order = _create_order(headers_a)
    client.post(
        "/payments/",
        json={"order_id": order["id"], "payment_method": "CARD"},
        headers=headers_a,
    )

    resp = client.get(f"/payments/order/{order['id']}", headers=headers_b)
    assert resp.status_code == 404


# 15. PENDING -> PAID works
def test_pending_to_paid_works(auth_headers):
    order = _create_order(auth_headers)
    payment = client.post(
        "/payments/", json={"order_id": order["id"], "payment_method": "CARD"}, headers=auth_headers
    ).json()

    resp = client.patch(f"/payments/{payment['id']}/pay", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "PAID"


# 16. PAID -> PAID fails
def test_paid_to_paid_fails(auth_headers):
    order = _create_order(auth_headers)
    payment = client.post(
        "/payments/", json={"order_id": order["id"], "payment_method": "CARD"}, headers=auth_headers
    ).json()
    client.patch(f"/payments/{payment['id']}/pay", headers=auth_headers)

    resp = client.patch(f"/payments/{payment['id']}/pay", headers=auth_headers)
    assert resp.status_code == 400


# 17. PENDING -> FAILED works
def test_pending_to_failed_works(auth_headers):
    order = _create_order(auth_headers)
    payment = client.post(
        "/payments/", json={"order_id": order["id"], "payment_method": "CARD"}, headers=auth_headers
    ).json()

    resp = client.patch(f"/payments/{payment['id']}/fail", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "FAILED"


# 18. PAID -> FAILED fails
def test_paid_to_failed_fails(auth_headers):
    order = _create_order(auth_headers)
    payment = client.post(
        "/payments/", json={"order_id": order["id"], "payment_method": "CARD"}, headers=auth_headers
    ).json()
    client.patch(f"/payments/{payment['id']}/pay", headers=auth_headers)

    resp = client.patch(f"/payments/{payment['id']}/fail", headers=auth_headers)
    assert resp.status_code == 400


# 19. FAILED -> PAID fails
def test_failed_to_paid_fails(auth_headers):
    order = _create_order(auth_headers)
    payment = client.post(
        "/payments/", json={"order_id": order["id"], "payment_method": "CARD"}, headers=auth_headers
    ).json()
    client.patch(f"/payments/{payment['id']}/fail", headers=auth_headers)

    resp = client.patch(f"/payments/{payment['id']}/pay", headers=auth_headers)
    assert resp.status_code == 400


# 20. PAID -> REFUNDED works
def test_paid_to_refunded_works(auth_headers):
    order = _create_order(auth_headers)
    payment = client.post(
        "/payments/", json={"order_id": order["id"], "payment_method": "CARD"}, headers=auth_headers
    ).json()
    client.patch(f"/payments/{payment['id']}/pay", headers=auth_headers)

    resp = client.patch(f"/payments/{payment['id']}/refund", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "REFUNDED"


# 21. REFUNDED -> REFUNDED fails
def test_refunded_to_refunded_fails(auth_headers):
    order = _create_order(auth_headers)
    payment = client.post(
        "/payments/", json={"order_id": order["id"], "payment_method": "CARD"}, headers=auth_headers
    ).json()
    client.patch(f"/payments/{payment['id']}/pay", headers=auth_headers)
    client.patch(f"/payments/{payment['id']}/refund", headers=auth_headers)

    resp = client.patch(f"/payments/{payment['id']}/refund", headers=auth_headers)
    assert resp.status_code == 400


# 22. REFUNDED -> PAID fails
def test_refunded_to_paid_fails(auth_headers):
    order = _create_order(auth_headers)
    payment = client.post(
        "/payments/", json={"order_id": order["id"], "payment_method": "CARD"}, headers=auth_headers
    ).json()
    client.patch(f"/payments/{payment['id']}/pay", headers=auth_headers)
    client.patch(f"/payments/{payment['id']}/refund", headers=auth_headers)

    resp = client.patch(f"/payments/{payment['id']}/pay", headers=auth_headers)
    assert resp.status_code == 400


# 23. REFUNDED -> FAILED fails
def test_refunded_to_failed_fails(auth_headers):
    order = _create_order(auth_headers)
    payment = client.post(
        "/payments/", json={"order_id": order["id"], "payment_method": "CARD"}, headers=auth_headers
    ).json()
    client.patch(f"/payments/{payment['id']}/pay", headers=auth_headers)
    client.patch(f"/payments/{payment['id']}/refund", headers=auth_headers)

    resp = client.patch(f"/payments/{payment['id']}/fail", headers=auth_headers)
    assert resp.status_code == 400


# 24. Cannot refund PENDING payment
def test_cannot_refund_pending_payment(auth_headers):
    order = _create_order(auth_headers)
    payment = client.post(
        "/payments/", json={"order_id": order["id"], "payment_method": "CARD"}, headers=auth_headers
    ).json()

    resp = client.patch(f"/payments/{payment['id']}/refund", headers=auth_headers)
    assert resp.status_code == 400


# 25. Cannot refund FAILED payment
def test_cannot_refund_failed_payment(auth_headers):
    order = _create_order(auth_headers)
    payment = client.post(
        "/payments/", json={"order_id": order["id"], "payment_method": "CARD"}, headers=auth_headers
    ).json()
    client.patch(f"/payments/{payment['id']}/fail", headers=auth_headers)

    resp = client.patch(f"/payments/{payment['id']}/refund", headers=auth_headers)
    assert resp.status_code == 400


# 26. Cannot pay another user's payment
def test_cannot_pay_another_users_payment():
    headers_a = _register_and_login()
    headers_b = _register_and_login()
    order = _create_order(headers_a)
    payment = client.post(
        "/payments/", json={"order_id": order["id"], "payment_method": "CARD"}, headers=headers_a
    ).json()

    resp = client.patch(f"/payments/{payment['id']}/pay", headers=headers_b)
    assert resp.status_code == 404


# 27. Cannot refund another user's payment
def test_cannot_refund_another_users_payment():
    headers_a = _register_and_login()
    headers_b = _register_and_login()
    order = _create_order(headers_a)
    payment = client.post(
        "/payments/", json={"order_id": order["id"], "payment_method": "CARD"}, headers=headers_a
    ).json()
    client.patch(f"/payments/{payment['id']}/pay", headers=headers_a)

    resp = client.patch(f"/payments/{payment['id']}/refund", headers=headers_b)
    assert resp.status_code == 404


# 28. JWT is required
def test_jwt_required_on_all_payment_endpoints():
    assert client.post("/payments/", json={"order_id": 1, "payment_method": "CARD"}).status_code == 401
    assert client.get("/payments/1").status_code == 401
    assert client.get("/payments/order/1").status_code == 401
    assert client.patch("/payments/1/pay").status_code == 401
    assert client.patch("/payments/1/fail").status_code == 401
    assert client.patch("/payments/1/refund").status_code == 401


# 29. Database failure rolls back correctly
def test_transaction_rollback_on_failure(auth_headers):
    order = _create_order(auth_headers, price=60.0, quantity=1)

    db = SessionLocal()
    try:
        with patch.object(db, "commit", side_effect=RuntimeError("forced failure")):
            with pytest.raises(RuntimeError):
                payment_service.create_payment(
                    db, order["user_id"], PaymentCreate(order_id=order["id"], payment_method="CARD")
                )
    finally:
        db.rollback()
        db.close()

    # Verify nothing persisted for this order.
    fresh_db = SessionLocal()
    try:
        payments = fresh_db.query(Payment).filter(Payment.order_id == order["id"]).all()
        assert payments == []
    finally:
        fresh_db.close()

    # A real payment can still be created afterward — DB is not corrupted.
    resp = client.post(
        "/payments/", json={"order_id": order["id"], "payment_method": "CARD"}, headers=auth_headers
    )
    assert resp.status_code == 200


# 30-33. Existing feature regressions are covered by running the full suite
# (test_auth.py, test_products.py, test_cart.py, test_orders.py) alongside
# this file — see the final pytest run in the delivery report.
