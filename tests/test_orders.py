import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from database.database import SessionLocal
from main import app
from models.cart import Cart, CartItem
from models.product import Product
from models.user import User, UserRole
from services import order_service

client = TestClient(app)


def _unique_email():
    return f"user_{uuid.uuid4().hex[:10]}@example.com"


def _promote_to_seller(email: str) -> None:
    """No API self-promotes to SELLER (by design), so test fixtures that
    need to create a product promote a real registered user directly at
    the DB level — same convention as
    tests/test_segmentation_api.py's _promote_to_admin. Order/cart
    behavior itself is unaffected: nothing in this project restricts
    cart/order actions by role, so the same account can be a SELLER for
    product-creation purposes and still act as a normal buyer below."""
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
            "full_name": "Order Test User",
            "password": password,
            "confirm_password": password,
        },
    )
    assert resp.status_code == 200, resp.text

    # Promoted to SELLER so this same account can also create the test
    # products it then adds to its own cart/orders below (product
    # creation requires SELLER or ADMIN; this file is not testing product
    # authorization, so this is a fixture-only change).
    _promote_to_seller(email)

    resp = client.post("/auth/login", data={"username": email, "password": password})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}, email


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


def _add_to_cart(headers, product_id, quantity):
    resp = client.post(
        "/cart/items", json={"product_id": product_id, "quantity": quantity}, headers=headers
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.fixture
def auth_headers():
    headers, _ = _register_and_login()
    return headers


# 1. Create order from cart
def test_create_order_from_cart(auth_headers):
    product = _create_product(auth_headers, stock=10)
    _add_to_cart(auth_headers, product["id"], 2)

    resp = client.post("/orders/", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "PENDING"
    assert len(data["items"]) == 1
    assert data["items"][0]["quantity"] == 2


# 2. Empty cart
def test_create_order_with_empty_cart(auth_headers):
    resp = client.post("/orders/", headers=auth_headers)
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Cannot create an order from an empty cart"


# 3. Multiple products
def test_create_order_multiple_products(auth_headers):
    p1 = _create_product(auth_headers, name="Item A", price=100.0, stock=10)
    p2 = _create_product(auth_headers, name="Item B", price=50.0, stock=10)
    _add_to_cart(auth_headers, p1["id"], 2)
    _add_to_cart(auth_headers, p2["id"], 3)

    resp = client.post("/orders/", headers=auth_headers)
    data = resp.json()
    assert len(data["items"]) == 2


# 4. Correct subtotal
def test_order_item_correct_subtotal(auth_headers):
    product = _create_product(auth_headers, name="Priced", price=25.0, stock=10)
    _add_to_cart(auth_headers, product["id"], 4)

    resp = client.post("/orders/", headers=auth_headers)
    item = resp.json()["items"][0]
    assert item["subtotal"] == pytest.approx(100.0)


# 5. Correct total_price
def test_order_correct_total_price(auth_headers):
    p1 = _create_product(auth_headers, name="Item C", price=100.0, stock=10)
    p2 = _create_product(auth_headers, name="Item D", price=50.0, stock=10)
    _add_to_cart(auth_headers, p1["id"], 2)
    _add_to_cart(auth_headers, p2["id"], 3)

    resp = client.post("/orders/", headers=auth_headers)
    assert resp.json()["total_price"] == pytest.approx(350.0)


# 6. Correct quantity
def test_order_item_correct_quantity(auth_headers):
    product = _create_product(auth_headers, name="Qty Test", stock=10)
    _add_to_cart(auth_headers, product["id"], 7)

    resp = client.post("/orders/", headers=auth_headers)
    assert resp.json()["items"][0]["quantity"] == 7


# 7. Product price snapshot
def test_order_item_price_snapshot(auth_headers):
    product = _create_product(auth_headers, name="Snapshot Price", price=100.0, stock=10)
    _add_to_cart(auth_headers, product["id"], 1)
    order = client.post("/orders/", headers=auth_headers).json()

    # price changes after the order was placed
    client.patch(
        f"/products/{product['id']}", json={"price": 200.0}, headers=auth_headers
    )

    fetched = client.get(f"/orders/{order['id']}", headers=auth_headers).json()
    assert fetched["items"][0]["product_price"] == 100.0
    assert fetched["total_price"] == 100.0


# 8. Product name snapshot
def test_order_item_name_snapshot(auth_headers):
    product = _create_product(auth_headers, name="Original Name", stock=10)
    _add_to_cart(auth_headers, product["id"], 1)
    order = client.post("/orders/", headers=auth_headers).json()

    client.patch(
        f"/products/{product['id']}", json={"name": "Renamed"}, headers=auth_headers
    )

    fetched = client.get(f"/orders/{order['id']}", headers=auth_headers).json()
    assert fetched["items"][0]["product_name"] == "Original Name"


# 9. Stock reduced after order
def test_stock_reduced_after_order(auth_headers):
    product = _create_product(auth_headers, name="Stock Test", stock=10)
    _add_to_cart(auth_headers, product["id"], 3)
    client.post("/orders/", headers=auth_headers)

    resp = client.get(f"/products/{product['id']}")
    assert resp.json()["stock"] == 7


# 10. Insufficient stock
def test_insufficient_stock_rejects_order(auth_headers):
    product = _create_product(auth_headers, name="Low Stock", stock=2)
    _add_to_cart(auth_headers, product["id"], 2)  # allowed at cart level given current stock

    # reduce stock further via another order from a different user to simulate depletion
    other_headers, _ = _register_and_login()
    _add_to_cart(other_headers, product["id"], 1)
    client.post("/orders/", headers=other_headers)  # stock now 1

    resp = client.post("/orders/", headers=auth_headers)  # original cart still wants 2
    assert resp.status_code == 400
    assert "stock" in resp.json()["detail"].lower()


# 11. Multiple products with one insufficient stock -> whole order fails
def test_multiple_products_one_insufficient_stock_fails_entirely(auth_headers):
    p_ok = _create_product(auth_headers, name="OK Stock", price=10.0, stock=10)
    p_low = _create_product(auth_headers, name="Insufficient", price=10.0, stock=5)
    _add_to_cart(auth_headers, p_ok["id"], 2)
    _add_to_cart(auth_headers, p_low["id"], 5)  # valid when added

    # p_low's stock gets depleted by someone else before this order is placed
    other_headers, _ = _register_and_login()
    _add_to_cart(other_headers, p_low["id"], 4)
    client.post("/orders/", headers=other_headers)  # p_low stock now 1, cart above still wants 5

    resp = client.post("/orders/", headers=auth_headers)
    assert resp.status_code == 400

    # p_ok's stock must NOT have been touched — the whole order failed atomically
    ok_after = client.get(f"/products/{p_ok['id']}").json()
    assert ok_after["stock"] == 10

    # Cart still has both items (order was not created)
    cart = client.get("/cart/", headers=auth_headers).json()
    assert len(cart["items"]) == 2


# 12. Cart cleared after successful order
def test_cart_cleared_after_successful_order(auth_headers):
    product = _create_product(auth_headers, name="Clear Cart Test", stock=10)
    _add_to_cart(auth_headers, product["id"], 2)
    client.post("/orders/", headers=auth_headers)

    cart = client.get("/cart/", headers=auth_headers).json()
    assert cart["items"] == []


# 13. Cart NOT cleared if order fails
def test_cart_not_cleared_if_order_fails(auth_headers):
    product = _create_product(auth_headers, name="Fail Path", price=10.0, stock=5)
    _add_to_cart(auth_headers, product["id"], 3)  # valid at add-to-cart time

    # Stock gets depleted by someone else after the item was added to the cart.
    other_headers, _ = _register_and_login()
    _add_to_cart(other_headers, product["id"], 4)
    client.post("/orders/", headers=other_headers)  # stock now 1

    resp = client.post("/orders/", headers=auth_headers)  # original cart still wants 3
    assert resp.status_code == 400

    cart = client.get("/cart/", headers=auth_headers).json()
    assert len(cart["items"]) == 1
    assert cart["items"][0]["quantity"] == 3


# 14. Get my orders
def test_get_my_orders(auth_headers):
    product = _create_product(auth_headers, stock=10)
    _add_to_cart(auth_headers, product["id"], 1)
    client.post("/orders/", headers=auth_headers)

    resp = client.get("/orders/", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


# 15. Get my order
def test_get_my_single_order(auth_headers):
    product = _create_product(auth_headers, stock=10)
    _add_to_cart(auth_headers, product["id"], 1)
    order = client.post("/orders/", headers=auth_headers).json()

    resp = client.get(f"/orders/{order['id']}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["id"] == order["id"]


# 16. Cannot access another user's order
def test_cannot_access_another_users_order():
    headers_a, _ = _register_and_login()
    headers_b, _ = _register_and_login()

    product = _create_product(headers_a, name="Owned By A", stock=10)
    _add_to_cart(headers_a, product["id"], 1)
    order = client.post("/orders/", headers=headers_a).json()

    resp = client.get(f"/orders/{order['id']}", headers=headers_b)
    assert resp.status_code == 404

    resp = client.get("/orders/", headers=headers_b)
    ids = [o["id"] for o in resp.json()]
    assert order["id"] not in ids


# 17. Order not found
def test_order_not_found(auth_headers):
    resp = client.get("/orders/999999", headers=auth_headers)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Order not found"


# 18. Cancel pending order
def test_cancel_pending_order(auth_headers):
    product = _create_product(auth_headers, stock=10)
    _add_to_cart(auth_headers, product["id"], 2)
    order = client.post("/orders/", headers=auth_headers).json()
    assert order["status"] == "PENDING"

    resp = client.patch(f"/orders/{order['id']}/cancel", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "CANCELLED"


# 19. Cancel confirmed order
def test_cancel_confirmed_order(auth_headers):
    product = _create_product(auth_headers, stock=10)
    _add_to_cart(auth_headers, product["id"], 1)
    order = client.post("/orders/", headers=auth_headers).json()
    confirmed = client.patch(f"/orders/{order['id']}/complete", headers=auth_headers).json()
    assert confirmed["status"] == "CONFIRMED"

    resp = client.patch(f"/orders/{order['id']}/cancel", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "CANCELLED"


# 20. Cannot cancel completed order
def test_cannot_cancel_completed_order(auth_headers):
    product = _create_product(auth_headers, stock=10)
    _add_to_cart(auth_headers, product["id"], 1)
    order = client.post("/orders/", headers=auth_headers).json()
    client.patch(f"/orders/{order['id']}/complete", headers=auth_headers)  # -> CONFIRMED
    completed = client.patch(f"/orders/{order['id']}/complete", headers=auth_headers).json()
    assert completed["status"] == "COMPLETED"

    resp = client.patch(f"/orders/{order['id']}/cancel", headers=auth_headers)
    assert resp.status_code == 400


# 21. Cannot cancel already cancelled order
def test_cannot_cancel_already_cancelled_order(auth_headers):
    product = _create_product(auth_headers, stock=10)
    _add_to_cart(auth_headers, product["id"], 1)
    order = client.post("/orders/", headers=auth_headers).json()
    client.patch(f"/orders/{order['id']}/cancel", headers=auth_headers)

    resp = client.patch(f"/orders/{order['id']}/cancel", headers=auth_headers)
    assert resp.status_code == 400


# 22. Stock restored after cancellation
def test_stock_restored_after_cancellation(auth_headers):
    product = _create_product(auth_headers, name="Restore Stock", stock=10)
    _add_to_cart(auth_headers, product["id"], 4)
    order = client.post("/orders/", headers=auth_headers).json()

    after_order = client.get(f"/products/{product['id']}").json()
    assert after_order["stock"] == 6

    client.patch(f"/orders/{order['id']}/cancel", headers=auth_headers)

    after_cancel = client.get(f"/products/{product['id']}").json()
    assert after_cancel["stock"] == 10


# 23. Confirm order
def test_confirm_order(auth_headers):
    product = _create_product(auth_headers, stock=10)
    _add_to_cart(auth_headers, product["id"], 1)
    order = client.post("/orders/", headers=auth_headers).json()

    resp = client.patch(f"/orders/{order['id']}/complete", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "CONFIRMED"


# 24. Complete order
def test_complete_order(auth_headers):
    product = _create_product(auth_headers, stock=10)
    _add_to_cart(auth_headers, product["id"], 1)
    order = client.post("/orders/", headers=auth_headers).json()

    client.patch(f"/orders/{order['id']}/complete", headers=auth_headers)  # PENDING -> CONFIRMED
    resp = client.patch(f"/orders/{order['id']}/complete", headers=auth_headers)  # -> COMPLETED
    assert resp.status_code == 200
    assert resp.json()["status"] == "COMPLETED"


# 25. Invalid status transition
def test_invalid_status_transition_after_completed(auth_headers):
    product = _create_product(auth_headers, stock=10)
    _add_to_cart(auth_headers, product["id"], 1)
    order = client.post("/orders/", headers=auth_headers).json()
    client.patch(f"/orders/{order['id']}/complete", headers=auth_headers)  # CONFIRMED
    client.patch(f"/orders/{order['id']}/complete", headers=auth_headers)  # COMPLETED

    resp = client.patch(f"/orders/{order['id']}/complete", headers=auth_headers)
    assert resp.status_code == 400


# 26. Unauthorized create order
def test_unauthorized_create_order():
    resp = client.post("/orders/")
    assert resp.status_code == 401


# 27. Unauthorized get orders
def test_unauthorized_get_orders():
    assert client.get("/orders/").status_code == 401
    assert client.get("/orders/1").status_code == 401


# 28. Unauthorized cancel
def test_unauthorized_cancel():
    resp = client.patch("/orders/1/cancel")
    assert resp.status_code == 401


# 29. Unauthorized complete
def test_unauthorized_complete():
    resp = client.patch("/orders/1/complete")
    assert resp.status_code == 401


# 30. Transaction rollback on failure
def test_transaction_rollback_on_failure(auth_headers):
    headers, email = auth_headers, None
    product_headers, product_email = _register_and_login()
    product = _create_product(product_headers, name="Rollback Test", price=20.0, stock=10)
    _add_to_cart(headers, product["id"], 3)

    db = SessionLocal()
    try:
        # Fetch the actual user tied to `headers` via the cart we just built
        cart = db.query(Cart).join(CartItem).filter(CartItem.product_id == product["id"]).first()
        user_id = cart.user_id

        with patch.object(db, "commit", side_effect=RuntimeError("forced failure")):
            with pytest.raises(RuntimeError):
                order_service.create_order_from_cart(db, user_id)
    finally:
        db.rollback()
        db.close()

    # Verify nothing persisted: cart item still present, stock unchanged
    fresh_db = SessionLocal()
    try:
        db_product = fresh_db.query(Product).filter(Product.id == product["id"]).first()
        assert db_product.stock == 10  # unchanged, rollback worked

        cart_items = (
            fresh_db.query(CartItem).filter(CartItem.product_id == product["id"]).all()
        )
        assert len(cart_items) == 1
        assert cart_items[0].quantity == 3
    finally:
        fresh_db.close()

    cart = client.get("/cart/", headers=headers).json()
    assert len(cart["items"]) == 1
    assert cart["items"][0]["quantity"] == 3
