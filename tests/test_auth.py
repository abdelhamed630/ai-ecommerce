import uuid

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _unique_email():
    return f"user_{uuid.uuid4().hex[:10]}@example.com"


def _register(email=None, password="StrongPass123!"):
    email = email or _unique_email()
    resp = client.post(
        "/auth/register",
        json={
            "email": email,
            "full_name": "Test User",
            "password": password,
            "confirm_password": password,
        },
    )
    return email, password, resp


def test_register_user():
    email, password, resp = _register()
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["email"] == email
    assert "password" not in data
    assert "hashed_password" not in data


def test_register_duplicate_email():
    email, password, resp = _register()
    assert resp.status_code == 200

    resp2 = client.post(
        "/auth/register",
        json={
            "email": email,
            "full_name": "Another User",
            "password": password,
            "confirm_password": password,
        },
    )
    assert resp2.status_code == 400
    assert resp2.json()["detail"] == "Email already registered"


def test_register_password_mismatch():
    email = _unique_email()
    resp = client.post(
        "/auth/register",
        json={
            "email": email,
            "full_name": "Test User",
            "password": "StrongPass123!",
            "confirm_password": "DifferentPass456!",
        },
    )
    assert resp.status_code == 422


def test_login_success():
    email, password, resp = _register()
    assert resp.status_code == 200

    login_resp = client.post(
        "/auth/login", data={"username": email, "password": password}
    )
    assert login_resp.status_code == 200, login_resp.text
    data = login_resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password():
    email, password, resp = _register()
    assert resp.status_code == 200

    login_resp = client.post(
        "/auth/login", data={"username": email, "password": "WrongPassword!"}
    )
    assert login_resp.status_code == 401


def test_me_with_token():
    email, password, resp = _register()
    login_resp = client.post(
        "/auth/login", data={"username": email, "password": password}
    )
    token = login_resp.json()["access_token"]

    me_resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == email


def test_me_without_token():
    resp = client.get("/auth/me")
    assert resp.status_code == 401
