# User Profile

Every authenticated user (USER, SELLER, ADMIN) has a profile on their own
`users` row. All profile endpoints act on the **authenticated user only**
(`/users/me`). There is no `/users/{id}` route: reading or editing another
user's profile is not possible (admin user management is a future phase).

Authentication is unchanged: JWT via `get_current_user`, role read from the
database. No profile route accepts `role`, `is_admin`, `hashed_password`,
`id`, `email`, `created_at` or `updated_at`.

## Profile data

| Field | Notes |
|---|---|
| `id`, `email` | read-only. `email` is unique and is the JWT `sub` claim |
| `username` | optional, unique, stored lowercase, 3-30 chars (`A-Z a-z 0-9 . _ -`), cannot be cleared once set |
| `first_name`, `last_name` | optional, max 50 |
| `full_name` | legacy field set at registration; returned read-only, not synced with first/last name |
| `phone` | optional, 7-20 chars: digits, spaces, `()`, `-`, optional leading `+` |
| `bio` | optional, max 500 |
| `avatar_url` | optional relative URL, e.g. `/media/avatars/user7_<uuid>.png` |
| `role` | read-only (`USER` / `SELLER` / `ADMIN`) |
| `created_at`, `updated_at` | read-only; `updated_at` changes on every profile update |

Existing users start with all optional fields empty; no names or usernames
are invented for them.

## Endpoints

### `GET /users/me`
Returns the caller's profile. `401` without a valid token.

### `PATCH /users/me`
JSON body with any of `first_name`, `last_name`, `username`, `phone`, `bio`
(only sent fields change; `""`/`null` clears first_name, last_name, phone, bio).

- Unknown/protected fields (`role`, `is_admin`, `email`, `hashed_password`, ...) -> `422`
  (the schema uses `extra="forbid"`, so it is rejected rather than silently ignored).
- Invalid values (length, username/phone format) -> `422`.
- Username already used by another user -> `400 "Username already taken"`
  (case-insensitive; also handled if two requests race on the DB unique index).

**Why email is not editable:** the JWT identifies the user by email (`sub`).
Changing it would invalidate every token the user holds and requires a
re-verification flow. That belongs to a separate phase.

### `PATCH /users/me/password`
Body: `current_password`, `new_password`, `confirm_new_password`.

- Wrong current password -> `400` (401 is reserved for missing/invalid tokens).
- Confirmation mismatch -> `422`.
- New password must be 8+ characters, at most 72 bytes (bcrypt limit), with at least one
  letter and one digit -> otherwise `422`. Registration had no strength rule, so this
  rule applies to password changes only; registration is unchanged.
- New password equal to the current one -> `400`.
- Hashed with the existing `hash_password` (bcrypt). Response is
  `{"message": "Password updated successfully"}`; passwords/hashes are never returned or logged.

**Existing tokens:** JWTs are stateless and there is no revocation list, so tokens
issued before a password change stay valid until they expire
(`ACCESS_TOKEN_EXPIRE_MINUTES`, default 60). The old password stops working for new
logins immediately. Token revocation was deliberately not added; if needed it is a
separate design (e.g. a `token_version` claim).

### `POST /users/me/avatar`
`multipart/form-data` with a `file` part.

- Allowed: JPEG, PNG, WEBP. Max 2 MB (`413` above that).
- `415` if the declared content type is not allowed **or** the file's leading bytes
  are not a real JPEG/PNG/WEBP header (a renamed script or SVG/GIF is rejected). The
  filename/extension is never trusted or used.
- `400` for an empty file.
- Replacing an avatar deletes the previous file after the new one is saved.
- Returns the updated profile.

### `DELETE /users/me/avatar`
Clears `avatar_url` and deletes the stored file. Idempotent: `200` even if there is
no avatar or the file is already gone.

## Avatar storage

- Files are written to `<MEDIA_ROOT>/avatars/` (`MEDIA_ROOT` setting, default `./media`,
  git-ignored) under server-generated names `user<id>_<uuid>.<jpg|png|webp>`.
- Only the relative URL is stored in `users.avatar_url`; **no image bytes are stored in the database**.
- `main.py` serves only the avatars directory at `/media/avatars/`.
- No cloud storage (S3) yet; no Pillow dependency was added.

## Security restrictions

- Every route resolves the user from the JWT; nothing takes a user id from the client.
- `role`/`is_admin` cannot be set through any profile route (422).
- Delete only removes files inside the avatars directory, even if `avatar_url` were tampered with.

## Existing databases

`create_all()` does not alter existing tables. Before running the app or the tests
against an existing `ecommerce.db`:

```
python scripts/migrate_add_user_profile_fields.py
```

It adds the new columns and a unique index on `username`, is idempotent, and never
touches users, passwords, ids or roles.

## Known limitations

- The 2 MB limit is enforced after the framework has received the upload (no
  streaming rejection or reverse-proxy limit yet).
- Only the image header is verified, not a full decode.
