import uuid

from fastapi.testclient import TestClient

from database.database import SessionLocal
from main import app
from models.user import User, UserRole

client = TestClient(app)


def _unique_email():
    return f"user_{uuid.uuid4().hex[:10]}@example.com"


def _promote_role(email: str, role: UserRole) -> None:
    """No API sets SELLER/ADMIN (by design — see scripts/set_user_role.py),
    so tests promote a real registered user directly at the DB level, the
    same way tests/test_segmentation_api.py's _promote_to_admin does."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        user.role = role
        db.commit()
    finally:
        db.close()


def _register_and_login(role: UserRole = UserRole.USER):
    email = _unique_email()
    password = "StrongPass123!"
    resp = client.post(
        "/auth/register",
        json={
            "email": email,
            "full_name": "Test User",
            "password": password,
            "confirm_password": password,
        },
    )
    assert resp.status_code == 200, resp.text

    if role != UserRole.USER:
        _promote_role(email, role)

    resp = client.post("/auth/login", data={"username": email, "password": password})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}, email


def _get_token():
    """A plain USER token — kept for the pre-existing auth/validation
    tests below that aren't about product authorization specifically."""
    headers, _ = _register_and_login(UserRole.USER)
    return headers


def _get_seller_headers():
    headers, _ = _register_and_login(UserRole.SELLER)
    return headers


def _get_admin_headers():
    headers, _ = _register_and_login(UserRole.ADMIN)
    return headers


def _create_product(headers, name="Laptop", price=1200.0, stock=10):
    return client.post(
        "/products/",
        json={
            "name": name,
            "description": "A test product",
            "price": price,
            "stock": stock,
            "image_url": None,
        },
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Existing behavior: unauthenticated / basic CRUD happy path (as SELLER,
# since product creation now requires SELLER or ADMIN — see the role
# authorization matrix section below for the USER-forbidden cases).
# ---------------------------------------------------------------------------

def test_create_product_without_auth():
    resp = client.post(
        "/products/",
        json={
            "name": "No Auth Product",
            "description": "test",
            "price": 10.0,
            "stock": 1,
        },
    )
    assert resp.status_code == 401


def test_create_product_with_auth():
    headers = _get_seller_headers()
    resp = _create_product(headers)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["name"] == "Laptop"
    assert data["price"] == 1200.0


def test_get_products():
    headers = _get_seller_headers()
    _create_product(headers, name="Mouse")
    resp = client.get("/products/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_product_by_id():
    headers = _get_seller_headers()
    create_resp = _create_product(headers, name="Keyboard")
    product_id = create_resp.json()["id"]

    resp = client.get(f"/products/{product_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Keyboard"


def test_get_missing_product():
    resp = client.get("/products/999999")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Product not found"


def test_update_product_full():
    headers = _get_seller_headers()
    create_resp = _create_product(headers, name="Monitor", price=300.0, stock=5)
    product_id = create_resp.json()["id"]

    resp = client.patch(
        f"/products/{product_id}",
        json={"name": "Monitor Pro", "price": 350.0, "stock": 8},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Monitor Pro"
    assert data["price"] == 350.0
    assert data["stock"] == 8


def test_update_product_partial():
    headers = _get_seller_headers()
    create_resp = _create_product(headers, name="Webcam", price=80.0, stock=3)
    product_id = create_resp.json()["id"]

    resp = client.patch(
        f"/products/{product_id}", json={"stock": 20}, headers=headers
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["stock"] == 20
    assert data["name"] == "Webcam"  # unchanged
    assert data["price"] == 80.0  # unchanged


def test_delete_product():
    headers = _get_seller_headers()
    create_resp = _create_product(headers, name="Speaker")
    product_id = create_resp.json()["id"]

    resp = client.delete(f"/products/{product_id}", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["message"] == "Product deleted successfully"

    get_resp = client.get(f"/products/{product_id}")
    assert get_resp.status_code == 404


def test_delete_missing_product():
    headers = _get_seller_headers()
    resp = client.delete("/products/999999", headers=headers)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Product not found"


def test_invalid_price():
    headers = _get_seller_headers()
    resp = client.post(
        "/products/",
        json={
            "name": "Test Product",
            "description": "Test",
            "price": -100,
            "stock": -5,
            "image_url": None,
        },
        headers=headers,
    )
    assert resp.status_code == 422


def test_invalid_stock():
    headers = _get_seller_headers()
    resp = client.post(
        "/products/",
        json={
            "name": "Valid Name",
            "description": "Test",
            "price": 10.0,
            "stock": -1,
            "image_url": None,
        },
        headers=headers,
    )
    assert resp.status_code == 422


def test_invalid_short_name():
    headers = _get_seller_headers()
    resp = client.post(
        "/products/",
        json={
            "name": "A",
            "description": "Test",
            "price": 10.0,
            "stock": 1,
            "image_url": None,
        },
        headers=headers,
    )
    assert resp.status_code == 422


# --- category/brand ---

def test_create_product_without_category_and_brand():
    headers = _get_seller_headers()
    resp = client.post(
        "/products/",
        json={"name": "No Metadata Product", "price": 10.0, "stock": 1},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["category"] is None
    assert data["brand"] is None


def test_create_product_with_category_and_brand():
    headers = _get_seller_headers()
    resp = client.post(
        "/products/",
        json={
            "name": "Gaming Laptop",
            "description": "a laptop",
            "price": 1500.0,
            "stock": 5,
            "category": "Electronics",
            "brand": "ExampleBrand",
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["category"] == "Electronics"
    assert data["brand"] == "ExampleBrand"


def test_product_response_contains_category_and_brand_keys():
    headers = _get_seller_headers()
    resp = client.post(
        "/products/",
        json={"name": "Key Check Product", "price": 5.0, "stock": 1},
        headers=headers,
    )
    data = resp.json()
    assert "category" in data
    assert "brand" in data


def test_update_product_category():
    headers = _get_seller_headers()
    created = client.post(
        "/products/",
        json={"name": "Category Update Product", "price": 20.0, "stock": 2},
        headers=headers,
    ).json()

    resp = client.patch(
        f"/products/{created['id']}", json={"category": "Kitchen"}, headers=headers
    )
    assert resp.status_code == 200
    assert resp.json()["category"] == "Kitchen"
    assert resp.json()["brand"] is None  # untouched


def test_update_product_brand():
    headers = _get_seller_headers()
    created = client.post(
        "/products/",
        json={"name": "Brand Update Product", "price": 20.0, "stock": 2},
        headers=headers,
    ).json()

    resp = client.patch(
        f"/products/{created['id']}", json={"brand": "AnotherBrand"}, headers=headers
    )
    assert resp.status_code == 200
    assert resp.json()["brand"] == "AnotherBrand"
    assert resp.json()["category"] is None  # untouched


# ---------------------------------------------------------------------------
# Role authorization matrix
# ---------------------------------------------------------------------------

# --- USER: forbidden from every mutation, browsing still allowed ---

def test_user_can_browse_products():
    resp = client.get("/products/")
    assert resp.status_code == 200


def test_user_cannot_create_product():
    headers = _get_token()
    resp = _create_product(headers, name="User Attempt")
    assert resp.status_code == 403


def test_user_cannot_update_product():
    seller_headers = _get_seller_headers()
    product = _create_product(seller_headers, name="Owned By Seller").json()

    user_headers = _get_token()
    resp = client.patch(
        f"/products/{product['id']}", json={"price": 1.0}, headers=user_headers
    )
    assert resp.status_code == 403


def test_user_cannot_delete_product():
    seller_headers = _get_seller_headers()
    product = _create_product(seller_headers, name="Owned By Seller 2").json()

    user_headers = _get_token()
    resp = client.delete(f"/products/{product['id']}", headers=user_headers)
    assert resp.status_code == 403


# --- SELLER: full control of own products, forbidden from others' ---

def test_seller_created_product_has_seller_id_set_to_self():
    headers, email = _register_and_login(UserRole.SELLER)
    resp = _create_product(headers, name="Seller Owned Product")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    db = SessionLocal()
    try:
        seller = db.query(User).filter(User.email == email).first()
        assert data["seller_id"] == seller.id
    finally:
        db.close()


def test_seller_can_update_own_product():
    headers = _get_seller_headers()
    product = _create_product(headers, name="Seller's Own", price=10.0).json()

    resp = client.patch(
        f"/products/{product['id']}", json={"price": 15.0}, headers=headers
    )
    assert resp.status_code == 200
    assert resp.json()["price"] == 15.0


def test_seller_can_delete_own_product():
    headers = _get_seller_headers()
    product = _create_product(headers, name="Seller's Own 2").json()

    resp = client.delete(f"/products/{product['id']}", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["message"] == "Product deleted successfully"


def test_seller_cannot_update_another_sellers_product():
    seller_a_headers = _get_seller_headers()
    seller_b_headers = _get_seller_headers()

    product = _create_product(seller_a_headers, name="Seller A's Product").json()

    resp = client.patch(
        f"/products/{product['id']}", json={"price": 999.0}, headers=seller_b_headers
    )
    assert resp.status_code == 403


def test_seller_cannot_delete_another_sellers_product():
    seller_a_headers = _get_seller_headers()
    seller_b_headers = _get_seller_headers()

    product = _create_product(seller_a_headers, name="Seller A's Product 2").json()

    resp = client.delete(f"/products/{product['id']}", headers=seller_b_headers)
    assert resp.status_code == 403

    # Confirm it wasn't actually deleted
    get_resp = client.get(f"/products/{product['id']}")
    assert get_resp.status_code == 200


# --- ADMIN: can manage any product; admin-created products have no owner ---

def test_admin_can_create_product():
    headers = _get_admin_headers()
    resp = _create_product(headers, name="Admin Created Product")
    assert resp.status_code == 200, resp.text


def test_admin_created_product_has_null_seller_id():
    headers = _get_admin_headers()
    resp = _create_product(headers, name="Admin Owns Nothing")
    assert resp.status_code == 200, resp.text
    assert resp.json()["seller_id"] is None


def test_admin_can_update_any_product():
    seller_headers = _get_seller_headers()
    admin_headers = _get_admin_headers()
    product = _create_product(seller_headers, name="Managed By Admin").json()

    resp = client.patch(
        f"/products/{product['id']}", json={"price": 42.0}, headers=admin_headers
    )
    assert resp.status_code == 200
    assert resp.json()["price"] == 42.0


def test_admin_can_delete_any_product():
    seller_headers = _get_seller_headers()
    admin_headers = _get_admin_headers()
    product = _create_product(seller_headers, name="Deleted By Admin").json()

    resp = client.delete(f"/products/{product['id']}", headers=admin_headers)
    assert resp.status_code == 200
    assert resp.json()["message"] == "Product deleted successfully"


# --- Additional coverage: browsing per role, NULL-owner products, spoofing ---

def test_authenticated_user_and_seller_can_browse_products():
    for headers in (_get_token(), _get_seller_headers(), _get_admin_headers()):
        resp = client.get("/products/", headers=headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


def test_seller_cannot_modify_admin_created_product_with_no_owner():
    admin_headers = _get_admin_headers()
    seller_headers = _get_seller_headers()
    product = _create_product(admin_headers, name="Ownerless Product").json()
    assert product["seller_id"] is None

    assert (
        client.patch(
            f"/products/{product['id']}", json={"price": 1.0}, headers=seller_headers
        ).status_code
        == 403
    )
    assert (
        client.delete(f"/products/{product['id']}", headers=seller_headers).status_code
        == 403
    )


def test_client_supplied_seller_id_is_ignored_on_create():
    headers_a, email_a = _register_and_login(UserRole.SELLER)
    headers_b, _ = _register_and_login(UserRole.SELLER)

    other = _create_product(headers_b, name="Other Seller Product").json()
    resp = client.post(
        "/products/",
        json={
            "name": "Spoofed Owner",
            "price": 10.0,
            "stock": 1,
            "seller_id": other["seller_id"],
        },
        headers=headers_a,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["seller_id"] != other["seller_id"]

    db = SessionLocal()
    try:
        seller_a = db.query(User).filter(User.email == email_a).first()
        assert resp.json()["seller_id"] == seller_a.id
    finally:
        db.close()


def test_client_supplied_seller_id_is_ignored_on_update():
    headers_a = _get_seller_headers()
    headers_b, _ = _register_and_login(UserRole.SELLER)
    product = _create_product(headers_a, name="Keep My Owner").json()
    original_owner = product["seller_id"]

    other = _create_product(headers_b, name="Someone Else's").json()
    resp = client.patch(
        f"/products/{product['id']}",
        json={"price": 5.0, "seller_id": other["seller_id"]},
        headers=headers_a,
    )
    assert resp.status_code == 200
    assert resp.json()["seller_id"] == original_owner


def test_unauthenticated_update_and_delete_return_401():
    assert client.patch("/products/1", json={"price": 1.0}).status_code == 401
    assert client.delete("/products/1").status_code == 401
