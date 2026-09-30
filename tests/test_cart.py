import uuid

import pytest
from fastapi.testclient import TestClient

from database.database import SessionLocal
from main import app
from models.user import User, UserRole

client = TestClient(app)


def _unique_email():
    return f"user_{uuid.uuid4().hex[:10]}@example.com"


def _promote_to_seller(email: str) -> None:
    """No API self-promotes to SELLER (by design), so test fixtures that
    need to create a product promote a real registered user directly at
    the DB level — same convention as
    tests/test_segmentation_api.py's _promote_to_admin."""
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
            "full_name": "Cart Test User",
            "password": password,
            "confirm_password": password,
        },
    )
    assert resp.status_code == 200, resp.text

    resp = client.post("/auth/login", data={"username": email, "password": password})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _create_product(name="Laptop", price=1500.0, stock=5):
    # Product creation now requires a SELLER (or ADMIN) account. This
    # helper is only test setup (cart behavior, not product authorization
    # is under test here), so use a throwaway user promoted to SELLER.
    email = _unique_email()
    password = "StrongPass123!"
    resp = client.post(
        "/auth/register",
        json={
            "email": email,
            "full_name": "Cart Test Seller",
            "password": password,
            "confirm_password": password,
        },
    )
    assert resp.status_code == 200, resp.text
    _promote_to_seller(email)

    resp = client.post("/auth/login", data={"username": email, "password": password})
    assert resp.status_code == 200, resp.text
    headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}

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


@pytest.fixture
def auth_headers():
    return _register_and_login()


# 1. Get empty cart
def test_get_empty_cart(auth_headers):
    resp = client.get("/cart/", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total_items"] == 0
    assert data["total_price"] == 0


# 2. Add product to cart
def test_add_product_to_cart(auth_headers):
    product = _create_product(stock=5)
    resp = client.post(
        "/cart/items",
        json={"product_id": product["id"], "quantity": 2},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["quantity"] == 2
    assert data["items"][0]["product"]["id"] == product["id"]


# 3. Add same product twice
def test_add_same_product_twice_increments_quantity(auth_headers):
    product = _create_product(name="Mouse", stock=10)
    client.post(
        "/cart/items",
        json={"product_id": product["id"], "quantity": 2},
        headers=auth_headers,
    )
    resp = client.post(
        "/cart/items",
        json={"product_id": product["id"], "quantity": 3},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["quantity"] == 5


# 4. Update cart item
def test_update_cart_item(auth_headers):
    product = _create_product(name="Keyboard", stock=10)
    add_resp = client.post(
        "/cart/items",
        json={"product_id": product["id"], "quantity": 5},
        headers=auth_headers,
    )
    item_id = add_resp.json()["items"][0]["id"]

    resp = client.patch(
        f"/cart/items/{item_id}", json={"quantity": 2}, headers=auth_headers
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["items"][0]["quantity"] == 2


# 5. Remove cart item
def test_remove_cart_item(auth_headers):
    product = _create_product(name="Monitor", stock=10)
    add_resp = client.post(
        "/cart/items",
        json={"product_id": product["id"], "quantity": 1},
        headers=auth_headers,
    )
    item_id = add_resp.json()["items"][0]["id"]

    resp = client.delete(f"/cart/items/{item_id}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["items"] == []


# 6. Clear cart
def test_clear_cart(auth_headers):
    p1 = _create_product(name="Pen", stock=10)
    p2 = _create_product(name="Notebook", stock=10)
    client.post("/cart/items", json={"product_id": p1["id"], "quantity": 1}, headers=auth_headers)
    client.post("/cart/items", json={"product_id": p2["id"], "quantity": 1}, headers=auth_headers)

    resp = client.delete("/cart/", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["items"] == []


# 7. Product not found
def test_add_nonexistent_product_returns_404(auth_headers):
    resp = client.post(
        "/cart/items", json={"product_id": 999999, "quantity": 1}, headers=auth_headers
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Product not found"


# 8. Cart item not found
def test_update_nonexistent_cart_item_returns_404(auth_headers):
    resp = client.patch(
        "/cart/items/999999", json={"quantity": 2}, headers=auth_headers
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Cart item not found"


# 9. Quantity = 0
def test_add_quantity_zero_is_rejected(auth_headers):
    product = _create_product(name="Zero Qty", stock=5)
    resp = client.post(
        "/cart/items", json={"product_id": product["id"], "quantity": 0}, headers=auth_headers
    )
    assert resp.status_code == 422


# 10. Negative quantity
def test_add_negative_quantity_is_rejected(auth_headers):
    product = _create_product(name="Negative Qty", stock=5)
    resp = client.post(
        "/cart/items", json={"product_id": product["id"], "quantity": -3}, headers=auth_headers
    )
    assert resp.status_code == 422


# 11. Quantity أكبر من stock
def test_add_quantity_more_than_stock_is_rejected(auth_headers):
    product = _create_product(name="Limited Stock", stock=3)
    resp = client.post(
        "/cart/items", json={"product_id": product["id"], "quantity": 10}, headers=auth_headers
    )
    assert resp.status_code == 400
    assert "stock" in resp.json()["detail"].lower()


# 12. Adding to another user's cart ممنوع
def test_user_cannot_access_another_users_cart_item():
    headers_a = _register_and_login()
    headers_b = _register_and_login()

    product = _create_product(name="Shared Product", stock=10)
    add_resp = client.post(
        "/cart/items", json={"product_id": product["id"], "quantity": 1}, headers=headers_a
    )
    item_id = add_resp.json()["items"][0]["id"]

    # user B tries to update/delete user A's cart item
    resp = client.patch(
        f"/cart/items/{item_id}", json={"quantity": 2}, headers=headers_b
    )
    assert resp.status_code == 404

    resp = client.delete(f"/cart/items/{item_id}", headers=headers_b)
    assert resp.status_code == 404

    # user A's item is still intact
    cart_a = client.get("/cart/", headers=headers_a).json()
    assert len(cart_a["items"]) == 1
    assert cart_a["items"][0]["quantity"] == 1


# 13. User بدون Cart يتم إنشاء Cart له تلقائيًا
def test_cart_auto_created_for_new_user(auth_headers):
    resp = client.get("/cart/", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert "user_id" in data


# 14. Correct subtotal
def test_correct_subtotal(auth_headers):
    product = _create_product(name="Priced Item", price=25.5, stock=10)
    resp = client.post(
        "/cart/items", json={"product_id": product["id"], "quantity": 3}, headers=auth_headers
    )
    item = resp.json()["items"][0]
    assert item["subtotal"] == pytest.approx(76.5)


# 15. Correct total_items
def test_correct_total_items(auth_headers):
    p1 = _create_product(name="Item A", price=10.0, stock=10)
    p2 = _create_product(name="Item B", price=5.0, stock=10)
    client.post("/cart/items", json={"product_id": p1["id"], "quantity": 2}, headers=auth_headers)
    resp = client.post(
        "/cart/items", json={"product_id": p2["id"], "quantity": 4}, headers=auth_headers
    )
    assert resp.json()["total_items"] == 6


# 16. Correct total_price
def test_correct_total_price(auth_headers):
    p1 = _create_product(name="Item C", price=10.0, stock=10)
    p2 = _create_product(name="Item D", price=5.0, stock=10)
    client.post("/cart/items", json={"product_id": p1["id"], "quantity": 2}, headers=auth_headers)
    resp = client.post(
        "/cart/items", json={"product_id": p2["id"], "quantity": 4}, headers=auth_headers
    )
    assert resp.json()["total_price"] == pytest.approx(40.0)


# 17. Unauthorized requests
def test_cart_endpoints_require_authentication():
    assert client.get("/cart/").status_code == 401
    assert client.post("/cart/items", json={"product_id": 1, "quantity": 1}).status_code == 401
    assert client.patch("/cart/items/1", json={"quantity": 1}).status_code == 401
    assert client.delete("/cart/items/1").status_code == 401
    assert client.delete("/cart/").status_code == 401


# 18. Duplicate product in same cart (DB-level unique constraint never violated
# because add_item_to_cart always updates the existing row instead of inserting a new one)
def test_duplicate_product_in_cart_does_not_create_second_row(auth_headers):
    product = _create_product(name="Unique Constraint Test", stock=20)
    client.post("/cart/items", json={"product_id": product["id"], "quantity": 1}, headers=auth_headers)
    client.post("/cart/items", json={"product_id": product["id"], "quantity": 1}, headers=auth_headers)
    resp = client.post(
        "/cart/items", json={"product_id": product["id"], "quantity": 1}, headers=auth_headers
    )
    data = resp.json()
    matching_items = [i for i in data["items"] if i["product_id"] == product["id"]]
    assert len(matching_items) == 1
    assert matching_items[0]["quantity"] == 3
