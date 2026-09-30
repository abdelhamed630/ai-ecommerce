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
    tests/test_segmentation_api.py's _promote_to_admin. Interaction
    behavior itself is unaffected: nothing in this project restricts
    interaction actions by role."""
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
            "full_name": "Interaction Test User",
            "password": password,
            "confirm_password": password,
        },
    )
    assert resp.status_code == 200, resp.text

    # Promoted to SELLER so this same account can also create the test
    # products it then records interactions against below (product
    # creation requires SELLER or ADMIN; this file is not testing product
    # authorization, so this is a fixture-only change).
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


@pytest.fixture
def auth_headers():
    return _register_and_login()


# 1. Authenticated user can record VIEW
def test_record_view_interaction(auth_headers):
    product = _create_product(auth_headers)
    resp = client.post(
        "/interactions/",
        json={"product_id": product["id"], "interaction_type": "VIEW"},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["interaction_type"] == "VIEW"


# 2. Authenticated user can record CART_ADD
def test_record_cart_add_interaction(auth_headers):
    product = _create_product(auth_headers)
    resp = client.post(
        "/interactions/",
        json={"product_id": product["id"], "interaction_type": "CART_ADD"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["interaction_type"] == "CART_ADD"


# 3. Authenticated user can record PURCHASE
def test_record_purchase_interaction(auth_headers):
    product = _create_product(auth_headers)
    resp = client.post(
        "/interactions/",
        json={"product_id": product["id"], "interaction_type": "PURCHASE"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["interaction_type"] == "PURCHASE"


# 4. Unauthenticated request is rejected
def test_unauthenticated_request_rejected():
    resp = client.post("/interactions/", json={"product_id": 1, "interaction_type": "VIEW"})
    assert resp.status_code == 401
    assert client.get("/interactions/me").status_code == 401
    assert client.post("/interactions/view/1").status_code == 401


# 5. Non-existing product returns 404
def test_nonexistent_product_returns_404(auth_headers):
    resp = client.post(
        "/interactions/",
        json={"product_id": 999999, "interaction_type": "VIEW"},
        headers=auth_headers,
    )
    assert resp.status_code == 404


# 6. User ID comes from JWT
def test_user_id_comes_from_jwt(auth_headers):
    product = _create_product(auth_headers)
    resp = client.post(
        "/interactions/",
        json={"product_id": product["id"], "interaction_type": "VIEW"},
        headers=auth_headers,
    )
    me = client.get("/auth/me", headers=auth_headers).json()
    assert resp.json()["user_id"] == me["id"]


# 7. Client cannot impersonate another user
def test_client_cannot_impersonate_another_user():
    headers_a = _register_and_login()
    headers_b = _register_and_login()
    product = _create_product(headers_a)

    resp = client.post(
        "/interactions/",
        json={"product_id": product["id"], "interaction_type": "VIEW", "user_id": 999999},
        headers=headers_b,
    )
    assert resp.status_code == 200
    me_b = client.get("/auth/me", headers=headers_b).json()
    assert resp.json()["user_id"] == me_b["id"]  # bogus user_id in body is ignored


# 8. Client cannot submit arbitrary user_id (schema simply has no such field)
def test_arbitrary_user_id_field_is_ignored(auth_headers):
    product = _create_product(auth_headers)
    resp = client.post(
        "/interactions/",
        json={"product_id": product["id"], "interaction_type": "VIEW", "user_id": 1},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    me = client.get("/auth/me", headers=auth_headers).json()
    assert resp.json()["user_id"] == me["id"]


# 9. User can retrieve own interactions
def test_user_can_retrieve_own_interactions(auth_headers):
    product = _create_product(auth_headers)
    client.post(
        "/interactions/",
        json={"product_id": product["id"], "interaction_type": "VIEW"},
        headers=auth_headers,
    )
    resp = client.get("/interactions/me", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


# 10. User cannot retrieve another user's private interaction data
def test_cannot_retrieve_another_users_interactions():
    headers_a = _register_and_login()
    headers_b = _register_and_login()
    product = _create_product(headers_a)
    client.post(
        "/interactions/",
        json={"product_id": product["id"], "interaction_type": "VIEW"},
        headers=headers_a,
    )

    resp_b = client.get("/interactions/me", headers=headers_b)
    assert resp_b.status_code == 200
    assert resp_b.json() == []  # user B has no interactions of their own


# 11. Product interaction endpoint works
def test_product_interaction_summary_endpoint(auth_headers):
    product = _create_product(auth_headers)
    client.post(
        "/interactions/",
        json={"product_id": product["id"], "interaction_type": "VIEW"},
        headers=auth_headers,
    )
    resp = client.get(f"/interactions/product/{product['id']}", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["product_id"] == product["id"]
    assert data["interaction_counts"]["VIEW"] >= 1
    # aggregate only — no per-user breakdown is exposed
    assert "user_id" not in data
    assert "interactions" not in data


# 12. Product view endpoint works
def test_product_view_endpoint(auth_headers):
    product = _create_product(auth_headers)
    resp = client.post(f"/interactions/view/{product['id']}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["interaction_type"] == "VIEW"
    assert resp.json()["product_id"] == product["id"]


# 13. Invalid interaction type is rejected
def test_invalid_interaction_type_rejected(auth_headers):
    product = _create_product(auth_headers)
    resp = client.post(
        "/interactions/",
        json={"product_id": product["id"], "interaction_type": "NOT_A_REAL_TYPE"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


# 14. created_at is generated by server
def test_created_at_generated_by_server(auth_headers):
    product = _create_product(auth_headers)
    resp = client.post(
        "/interactions/",
        json={
            "product_id": product["id"],
            "interaction_type": "VIEW",
            "created_at": "1999-01-01T00:00:00",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert not resp.json()["created_at"].startswith("1999")


# 15. Interaction is correctly linked to Product
def test_interaction_linked_to_product(auth_headers):
    product = _create_product(auth_headers, name="Linked Product")
    resp = client.post(
        "/interactions/",
        json={"product_id": product["id"], "interaction_type": "VIEW"},
        headers=auth_headers,
    )
    assert resp.json()["product_id"] == product["id"]


# 16. Interaction is correctly linked to User
def test_interaction_linked_to_user(auth_headers):
    product = _create_product(auth_headers)
    resp = client.post(
        "/interactions/",
        json={"product_id": product["id"], "interaction_type": "VIEW"},
        headers=auth_headers,
    )
    me = client.get("/auth/me", headers=auth_headers).json()
    assert resp.json()["user_id"] == me["id"]


# 17. Multiple interactions for same product can be recorded
def test_multiple_interactions_same_product(auth_headers):
    product = _create_product(auth_headers)
    for _ in range(3):
        resp = client.post(
            "/interactions/",
            json={"product_id": product["id"], "interaction_type": "VIEW"},
            headers=auth_headers,
        )
        assert resp.status_code == 200

    mine = client.get("/interactions/me", headers=auth_headers).json()
    matching = [i for i in mine if i["product_id"] == product["id"]]
    assert len(matching) == 3


# 21. CART_ADD is automatically recorded after successful cart add
def test_cart_add_automatically_recorded(auth_headers):
    product = _create_product(auth_headers, name="Auto Cart Track")
    resp = client.post(
        "/cart/items", json={"product_id": product["id"], "quantity": 1}, headers=auth_headers
    )
    assert resp.status_code == 200

    mine = client.get("/interactions/me", headers=auth_headers).json()
    cart_add_events = [
        i for i in mine if i["product_id"] == product["id"] and i["interaction_type"] == "CART_ADD"
    ]
    assert len(cart_add_events) == 1


# 22. PURCHASE is automatically recorded after successful order creation
def test_purchase_automatically_recorded(auth_headers):
    product = _create_product(auth_headers, name="Auto Purchase Track", stock=10)
    client.post(
        "/cart/items", json={"product_id": product["id"], "quantity": 2}, headers=auth_headers
    )
    order_resp = client.post("/orders/", headers=auth_headers)
    assert order_resp.status_code == 200

    mine = client.get("/interactions/me", headers=auth_headers).json()
    purchase_events = [
        i for i in mine if i["product_id"] == product["id"] and i["interaction_type"] == "PURCHASE"
    ]
    assert len(purchase_events) == 1
