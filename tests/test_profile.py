import os
import uuid

import pytest
from fastapi.testclient import TestClient

from core.config import settings
from database.database import SessionLocal
from main import app
from models.user import User, UserRole
from services import avatar_service

client = TestClient(app)

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
WEBP = b"RIFF" + b"\x24\x00\x00\x00" + b"WEBP" + b"\x00" * 64


@pytest.fixture(autouse=True)
def _isolated_media_root(tmp_path, monkeypatch):
    # avatar_service reads settings.MEDIA_ROOT at call time.
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(tmp_path))


def _unique_email():
    return f"user_{uuid.uuid4().hex[:10]}@example.com"


def _unique_username():
    return f"u_{uuid.uuid4().hex[:12]}"


def _register(role=UserRole.USER, password="StrongPass123!"):
    email = _unique_email()
    resp = client.post(
        "/auth/register",
        json={"email": email, "full_name": "Profile Tester",
              "password": password, "confirm_password": password},
    )
    assert resp.status_code == 200, resp.text
    if role != UserRole.USER:
        db = SessionLocal()
        try:
            db.query(User).filter(User.email == email).first().role = role
            db.commit()
        finally:
            db.close()
    return email, password


def _login(email, password):
    return client.post("/auth/login", data={"username": email, "password": password})


def _headers(role=UserRole.USER):
    email, password = _register(role)
    token = _login(email, password).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _upload(headers, data, content_type="image/png", filename="a.png"):
    return client.post(
        "/users/me/avatar", files={"file": (filename, data, content_type)}, headers=headers
    )


# ---------------------------------------------------------------- GET /users/me

@pytest.mark.parametrize("role", [UserRole.USER, UserRole.SELLER, UserRole.ADMIN])
def test_get_my_profile_all_roles(role):
    resp = client.get("/users/me", headers=_headers(role))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["role"] == role.value
    for key in ("id", "email", "username", "first_name", "last_name", "phone",
                "bio", "avatar_url", "created_at", "updated_at"):
        assert key in data
    assert "hashed_password" not in data and "password" not in data


def test_get_my_profile_unauthenticated():
    assert client.get("/users/me").status_code == 401


# -------------------------------------------------------------- PATCH /users/me

def test_update_each_profile_field_and_persist():
    headers = _headers()
    username = _unique_username()
    body = {"first_name": "Ada", "last_name": "Lovelace", "username": username,
            "phone": "+20 100 123 4567", "bio": "Hello there"}
    resp = client.patch("/users/me", json=body, headers=headers)
    assert resp.status_code == 200, resp.text
    for k, v in body.items():
        assert resp.json()[k] == v

    again = client.get("/users/me", headers=headers).json()
    for k, v in body.items():
        assert again[k] == v


def test_partial_update_leaves_other_fields():
    headers = _headers()
    client.patch("/users/me", json={"first_name": "Ada", "bio": "keep"}, headers=headers)
    resp = client.patch("/users/me", json={"last_name": "L"}, headers=headers)
    assert resp.json()["first_name"] == "Ada"
    assert resp.json()["bio"] == "keep"
    assert resp.json()["last_name"] == "L"


def test_username_is_normalized_to_lowercase():
    headers = _headers()
    name = _unique_username()
    resp = client.patch("/users/me", json={"username": name.upper()}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["username"] == name.lower()


def test_duplicate_username_rejected():
    headers_a, headers_b = _headers(), _headers()
    name = _unique_username()
    assert client.patch("/users/me", json={"username": name}, headers=headers_a).status_code == 200
    resp = client.patch("/users/me", json={"username": name.upper()}, headers=headers_b)
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Username already taken"


def test_resubmitting_own_username_is_ok():
    headers = _headers()
    name = _unique_username()
    client.patch("/users/me", json={"username": name}, headers=headers)
    assert client.patch("/users/me", json={"username": name}, headers=headers).status_code == 200


@pytest.mark.parametrize("body", [
    {"username": "ab"},
    {"username": "has space"},
    {"username": "x" * 31},
    {"username": None},
    {"first_name": "x" * 51},
    {"bio": "x" * 501},
    {"phone": "abc"},
    {"phone": "12"},
])
def test_invalid_profile_input_returns_422(body):
    assert client.patch("/users/me", json=body, headers=_headers()).status_code == 422


def test_role_cannot_be_changed_via_profile():
    headers = _headers()
    for role in ("ADMIN", "SELLER"):
        assert client.patch("/users/me", json={"role": role}, headers=headers).status_code == 422
    assert client.get("/users/me", headers=headers).json()["role"] == "USER"


def test_is_admin_and_other_protected_fields_rejected():
    headers = _headers()
    for body in ({"is_admin": True}, {"hashed_password": "x"}, {"id": 1},
                 {"email": "new@example.com"}, {"created_at": "2020-01-01T00:00:00"},
                 {"updated_at": "2020-01-01T00:00:00"}):
        assert client.patch("/users/me", json=body, headers=headers).status_code == 422, body
    me = client.get("/users/me", headers=headers).json()
    assert me["role"] == "USER"


def test_patch_profile_unauthenticated():
    assert client.patch("/users/me", json={"bio": "x"}).status_code == 401


def test_updated_at_is_present_after_update():
    headers = _headers()
    resp = client.patch("/users/me", json={"bio": "changed"}, headers=headers)
    assert resp.json()["updated_at"] is not None


# ------------------------------------------------------------------- password

def test_change_password_success_and_new_password_authenticates():
    email, old = _register()
    headers = {"Authorization": f"Bearer {_login(email, old).json()['access_token']}"}
    new = "BrandNewPass456!"
    resp = client.patch(
        "/users/me/password",
        json={"current_password": old, "new_password": new, "confirm_new_password": new},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert "password" not in resp.text.lower().replace("password updated successfully", "")
    assert _login(email, old).status_code == 401
    assert _login(email, new).status_code == 200


def test_existing_token_stays_valid_after_password_change():
    # Documented behavior: JWTs are stateless and are not revoked.
    email, old = _register()
    headers = {"Authorization": f"Bearer {_login(email, old).json()['access_token']}"}
    new = "BrandNewPass456!"
    client.patch("/users/me/password",
                 json={"current_password": old, "new_password": new, "confirm_new_password": new},
                 headers=headers)
    assert client.get("/users/me", headers=headers).status_code == 200


def test_wrong_current_password_rejected():
    email, old = _register()
    headers = {"Authorization": f"Bearer {_login(email, old).json()['access_token']}"}
    new = "BrandNewPass456!"
    resp = client.patch(
        "/users/me/password",
        json={"current_password": "WrongPass999!", "new_password": new, "confirm_new_password": new},
        headers=headers,
    )
    assert resp.status_code == 400
    assert _login(email, old).status_code == 200  # unchanged


def test_password_confirmation_mismatch_is_422():
    email, old = _register()
    headers = {"Authorization": f"Bearer {_login(email, old).json()['access_token']}"}
    resp = client.patch(
        "/users/me/password",
        json={"current_password": old, "new_password": "BrandNewPass456!",
              "confirm_new_password": "Different789!"},
        headers=headers,
    )
    assert resp.status_code == 422


@pytest.mark.parametrize("weak", ["short1", "onlyletters", "12345678"])
def test_weak_new_password_is_422(weak):
    email, old = _register()
    headers = {"Authorization": f"Bearer {_login(email, old).json()['access_token']}"}
    resp = client.patch(
        "/users/me/password",
        json={"current_password": old, "new_password": weak, "confirm_new_password": weak},
        headers=headers,
    )
    assert resp.status_code == 422


def test_new_password_same_as_current_rejected():
    email, old = _register()
    headers = {"Authorization": f"Bearer {_login(email, old).json()['access_token']}"}
    resp = client.patch(
        "/users/me/password",
        json={"current_password": old, "new_password": old, "confirm_new_password": old},
        headers=headers,
    )
    assert resp.status_code == 400


def test_change_password_unauthenticated():
    resp = client.patch("/users/me/password",
                        json={"current_password": "a", "new_password": "b1234567", "confirm_new_password": "b1234567"})
    assert resp.status_code == 401


# --------------------------------------------------------------------- avatar

@pytest.mark.parametrize("data,ctype,name,ext", [
    (JPEG, "image/jpeg", "me.jpg", ".jpg"),
    (PNG, "image/png", "me.png", ".png"),
    (WEBP, "image/webp", "me.webp", ".webp"),
])
def test_valid_avatar_upload(data, ctype, name, ext):
    headers = _headers()
    resp = _upload(headers, data, ctype, name)
    assert resp.status_code == 200, resp.text
    url = resp.json()["avatar_url"]
    assert url.startswith("/media/avatars/") and url.endswith(ext)
    stored = os.path.join(settings.MEDIA_ROOT, "avatars", os.path.basename(url))
    assert os.path.isfile(stored)
    assert client.get("/users/me", headers=headers).json()["avatar_url"] == url


def test_avatar_stored_name_ignores_client_filename():
    resp = _upload(_headers(), PNG, "image/png", "../../evil.png")
    assert resp.status_code == 200
    assert "evil" not in resp.json()["avatar_url"] and ".." not in resp.json()["avatar_url"]


def test_non_image_content_rejected():
    assert _upload(_headers(), b"#!/bin/sh\nrm -rf /\n", "text/plain", "x.sh").status_code == 415


def test_disguised_file_rejected_by_content_check():
    # Claims to be a PNG (name + content type) but the bytes are a script.
    assert _upload(_headers(), b"<?php system($_GET['c']); ?>", "image/png", "x.png").status_code == 415


def test_svg_and_gif_rejected():
    headers = _headers()
    assert _upload(headers, b"<svg xmlns='http://www.w3.org/2000/svg'/>", "image/svg+xml", "a.svg").status_code == 415
    assert _upload(headers, b"GIF89a" + b"\x00" * 32, "image/gif", "a.gif").status_code == 415


def test_oversized_avatar_rejected():
    big = JPEG + b"\x00" * avatar_service.MAX_AVATAR_BYTES
    assert _upload(_headers(), big, "image/jpeg", "big.jpg").status_code == 413


def test_empty_avatar_rejected():
    assert _upload(_headers(), b"", "image/png", "e.png").status_code == 400


def test_avatar_upload_unauthenticated():
    resp = client.post("/users/me/avatar", files={"file": ("a.png", PNG, "image/png")})
    assert resp.status_code == 401


def test_replacing_avatar_removes_old_file():
    headers = _headers()
    first = _upload(headers, PNG, "image/png").json()["avatar_url"]
    second = _upload(headers, JPEG, "image/jpeg", "b.jpg").json()["avatar_url"]
    assert first != second
    directory = os.path.join(settings.MEDIA_ROOT, "avatars")
    assert not os.path.exists(os.path.join(directory, os.path.basename(first)))
    assert os.path.exists(os.path.join(directory, os.path.basename(second)))


def test_delete_avatar():
    headers = _headers()
    url = _upload(headers, PNG, "image/png").json()["avatar_url"]
    resp = client.delete("/users/me/avatar", headers=headers)
    assert resp.status_code == 200
    assert client.get("/users/me", headers=headers).json()["avatar_url"] is None
    assert not os.path.exists(os.path.join(settings.MEDIA_ROOT, "avatars", os.path.basename(url)))


def test_delete_nonexistent_avatar_does_not_crash():
    headers = _headers()
    assert client.delete("/users/me/avatar", headers=headers).status_code == 200
    assert client.delete("/users/me/avatar", headers=headers).status_code == 200


def test_delete_avatar_when_file_already_missing():
    headers = _headers()
    url = _upload(headers, PNG, "image/png").json()["avatar_url"]
    os.remove(os.path.join(settings.MEDIA_ROOT, "avatars", os.path.basename(url)))
    assert client.delete("/users/me/avatar", headers=headers).status_code == 200


def test_delete_avatar_unauthenticated():
    assert client.delete("/users/me/avatar").status_code == 401


def test_remove_avatar_file_ignores_paths_outside_avatar_dir(tmp_path):
    outside = tmp_path / "keep.txt"
    outside.write_text("x")
    avatar_service.remove_avatar_file("/media/avatars/../keep.txt")
    avatar_service.remove_avatar_file("/etc/passwd")
    assert outside.exists()


# ------------------------------------------------------------------- isolation

def test_user_a_cannot_see_or_change_user_b():
    email_a, pw_a = _register()
    email_b, pw_b = _register()
    headers_a = {"Authorization": f"Bearer {_login(email_a, pw_a).json()['access_token']}"}
    headers_b = {"Authorization": f"Bearer {_login(email_b, pw_b).json()['access_token']}"}

    client.patch("/users/me", json={"bio": "B's private bio"}, headers=headers_b)
    b_id = client.get("/users/me", headers=headers_b).json()["id"]

    # /users/me only ever returns the caller's own profile
    me_a = client.get("/users/me", headers=headers_a).json()
    assert me_a["email"] == email_a and me_a["bio"] != "B's private bio"

    # No route addresses another user by id
    assert client.get(f"/users/{b_id}", headers=headers_a).status_code in (404, 405)
    assert client.patch(f"/users/{b_id}", json={"bio": "hacked"}, headers=headers_a).status_code in (404, 405)
    assert client.patch(f"/users/{b_id}/password", json={}, headers=headers_a).status_code in (404, 405)
    assert client.delete(f"/users/{b_id}/avatar", headers=headers_a).status_code in (404, 405)

    # A's own writes never touch B
    client.patch("/users/me", json={"bio": "A's bio"}, headers=headers_a)
    client.delete("/users/me/avatar", headers=headers_a)
    assert client.get("/users/me", headers=headers_b).json()["bio"] == "B's private bio"


def test_avatar_upload_only_affects_caller():
    headers_a, headers_b = _headers(), _headers()
    _upload(headers_a, PNG, "image/png")
    assert client.get("/users/me", headers=headers_b).json()["avatar_url"] is None


def test_registration_still_creates_plain_user_with_new_fields_empty():
    email, pw = _register()
    token = _login(email, pw).json()["access_token"]
    me = client.get("/users/me", headers={"Authorization": f"Bearer {token}"}).json()
    assert me["role"] == "USER" and me["username"] is None and me["avatar_url"] is None
