# Authentication & Authorization

This document describes how the API authenticates a request (JWT) and
then authorizes it (role check), and how product ownership fits into
that. It reflects the current, final state of the role/authorization
foundation added in this refactor — see "Recommended next phase" at the
end for what's deliberately *not* covered here.

## Roles

There are exactly three roles, defined once, canonically, as
`UserRole` in `models/user.py`:

- **USER** — a normal customer. Can browse products, use the cart,
  place orders, make payments, record interactions, get
  recommendations, and trigger their own customer segmentation data
  (i.e. they show up as a segmented customer; they cannot *run* the
  segmentation job itself — that's ADMIN-only, see below).
- **SELLER** — a merchant/vendor. Everything a USER can do, plus:
  create products, and update/delete the products *they* created.
- **ADMIN** — a system administrator. Full access: can manage any
  product regardless of owner, and can run the segmentation job.

Every user has exactly one role, stored in `User.role`
(`models/user.py`). There is no notion of multiple roles per user, and
no role hierarchy is encoded beyond what each authorization dependency
explicitly checks (e.g. ADMIN is allowed wherever SELLER is, but only
because the product endpoints explicitly say
`Depends(get_current_admin_or_seller_user)`, not because ADMIN
"inherits" SELLER).

## Why public registration always creates USER

`POST /auth/register` accepts `schemas.user.UserCreate`, which has no
`role` (and no `is_admin`) field at all. It is not that the field is
present but ignored — it's simply not part of the schema, so a request
body like:

```json
{ "email": "...", "password": "...", "role": "ADMIN" }
```

has its `role` key silently dropped by pydantic before it ever reaches
`services.user_service.create_user`, which never sets `role` explicitly
and so gets the model's default: `UserRole.USER`. There is no code path
in the public API, under any input, that creates a SELLER or ADMIN
account. This is deliberate: client-supplied authorization data is
never trusted, full stop.

## Admin / seller provisioning

Because there's no public self-promotion endpoint, assigning SELLER or
ADMIN is an explicit, out-of-band, database-level operation:

```
python scripts/set_user_role.py --email someone@example.com --role SELLER
python scripts/set_user_role.py --email someone@example.com --role ADMIN
```

The script requires an exact target email and an exact target role, is
idempotent (running it again with the same arguments is a harmless
no-op), and never creates or deletes a user — only updates the `role`
of one that already exists. There is intentionally no HTTP endpoint for
this.

This project's own test suite uses the same convention for test setup:
tests that need a SELLER or ADMIN account register a normal user
through the real `/auth/register` API and then promote it directly via
a `SessionLocal` write (see `tests/test_segmentation_api.py`'s
`_promote_role`, `tests/test_products.py`'s `_promote_role`, and the
`_promote_to_seller` helpers in `tests/test_cart.py`,
`tests/test_orders.py`, `tests/test_payments.py`,
`tests/test_interactions.py`, and `tests/test_recommendations.py`).

## Authentication: JWT → current user

Authentication is unchanged by this refactor. `core.security`:

- `hash_password` / `verify_password` — bcrypt via passlib, untouched.
- `create_access_token` / `decode_access_token` — JWT encode/decode
  (HS256, via `python-jose`), untouched.
- `get_current_user(token, db)` — decodes the JWT, looks up the user by
  the `sub` (email) claim, rejects with `401` on a missing/invalid/
  expired token or unknown user, and rejects with `403` if the user is
  inactive. This is the *only* place that touches the JWT. Every
  authorization dependency below is built on top of it, not alongside
  it.

**The JWT carries no role.** `create_access_token` is only ever called
with `{"sub": user.email}`. This is deliberate: if it embedded the
user's role at login time, promoting or demoting a user later (e.g. via
`set_user_role.py`) would have no effect until they logged in again,
since the stale role in their existing token would keep being trusted.
Instead, `get_current_user` re-reads `role` from the database on every
request, so a role change takes effect immediately, on the very next
request — no re-login needed, no risk of a revoked admin continuing to
act as one until their token expires.

## Authorization: current user → role check

Authorization is a second, separate step layered on top of
`get_current_user`. `core.security` provides:

- **`require_roles(*allowed_roles)`** — a dependency *factory*, and the
  one generic mechanism everything else below is built from. Call it
  with the roles you want to allow — e.g.
  `Depends(require_roles(UserRole.ADMIN, UserRole.SELLER))` — and it
  returns a FastAPI dependency that runs `get_current_user` first (so
  you still get a normal `401` for a bad/missing token), then checks
  `current_user.role` against the allowed set, raising `403` if it
  doesn't match.
- **`get_current_admin_user`** — equivalent to
  `require_roles(UserRole.ADMIN)`, kept as its own named dependency
  since "admin only" is by far the most common case (used by
  `POST /segmentation/run`).
- **`get_current_seller_user`** — equivalent to
  `require_roles(UserRole.SELLER)`. Not currently used by any endpoint
  (nothing in this phase is SELLER-*only*), but available for a future
  seller-specific endpoint.
- **`get_current_admin_or_seller_user`** — equivalent to
  `require_roles(UserRole.ADMIN, UserRole.SELLER)`. Used by
  `POST /products/`, `PATCH /products/{id}`, and
  `DELETE /products/{id}` — both roles may call these; a *further*,
  per-resource ownership check (below) then narrows what a SELLER
  specifically may touch.

All four share one underlying check (`_require_role`), so there is
exactly one implementation of "compare `current_user.role` against an
allowed set and raise 403" in the codebase — not four copies of it, and
not three copies of `get_current_user` either. `get_current_user`
itself is unchanged and is still the *only* function that touches the
JWT.

## Product ownership

`models.product.Product` has a nullable `seller_id` (FK to `users.id`):

- **A SELLER-created product**: `seller_id = current_user.id`. Always
  set server-side, from the authenticated user making the request —
  `schemas.product.ProductCreate` has no `seller_id` field, so it can
  never be supplied by the client.
- **An ADMIN-created product**: `seller_id = NULL`. ADMIN is a system
  administrator role, not automatically a seller/vendor, so an
  admin-created product has no owner. (It's still fully manageable —
  by any ADMIN — same as any other product; "no owner" only affects
  what a *SELLER* is allowed to do with it, which is nothing, since
  they don't own it.)
- **A legacy product** (created before this column existed): also
  `seller_id = NULL`. No historical owner is invented or guessed — see
  `scripts/migrate_add_product_seller_id.py`.

Enforcement (`api/products.py`):

| Endpoint                  | Role check                           | Extra check                              |
| -------------------------- | ------------------------------------- | ----------------------------------------- |
| `GET /products/`           | none — open to everyone, incl. anonymous | —                                      |
| `GET /products/{id}`       | none — open to everyone, incl. anonymous | —                                      |
| `POST /products/`          | `get_current_admin_or_seller_user`    | none — `seller_id` derived from role/user |
| `PATCH /products/{id}`     | `get_current_admin_or_seller_user`    | SELLER: `product.seller_id == current_user.id`, else `403` |
| `DELETE /products/{id}`    | `get_current_admin_or_seller_user`    | same ownership check as PATCH             |

The ownership check can't be expressed as a role-only dependency
because it depends on which *specific* product is being touched — it
loads the product first (`404` if it doesn't exist), then, only if the
caller is a SELLER (an ADMIN skips this entirely), compares
`product.seller_id` to `current_user.id`.

## Permission matrix

| Action              | USER | SELLER | ADMIN |
| ------------------- | ---- | ------ | ----- |
| Browse products      | YES  | YES    | YES   |
| Create product        | NO   | YES    | YES   |
| Update own product    | NO   | YES    | YES   |
| Update any product    | NO   | NO     | YES   |
| Delete own product    | NO   | YES    | YES   |
| Delete any product    | NO   | NO     | YES   |
| Run segmentation      | NO   | NO     | YES   |

Cart, orders, payments, interactions, and recommendations are not
role-restricted at all in this phase — any authenticated user (USER,
SELLER, or ADMIN alike) can use them, exactly as before this refactor.
Only endpoints where a role distinction was explicitly required
(product mutation, segmentation) were changed; everything else keeps
its pre-existing "just needs a valid logged-in user" behavior
(`get_current_user`), with per-user data isolation (e.g. "can't see
another user's cart/order") preserved unchanged.

## Migration: `is_admin` → `role`

Earlier versions of this project used a single `User.is_admin` boolean
instead of a role system. If your database predates this refactor and
still has that column:

```
python scripts/migrate_is_admin_to_role.py
```

This adds the `role` column if missing, and if `is_admin` is still
present, backfills: `is_admin = 1 → role = 'ADMIN'`,
`is_admin = 0 → role = 'USER'`. No user is ever auto-promoted to
SELLER by this migration — SELLER is always assigned explicitly via
`set_user_role.py`. The migration is idempotent (re-running it only ever
promotes users still at `USER` whose legacy flag is set, so it can never
undo a role assigned afterwards), never deletes a user,
never touches passwords or IDs, and never drops the legacy `is_admin`
column — it's simply left in place, unused (the application no longer
reads it anywhere). See the script's own docstring for full details and
verification steps.

Similarly, `scripts/migrate_add_product_seller_id.py` adds the
nullable `seller_id` column to an existing `products` table, with
every existing row getting `seller_id = NULL` (no owner invented).

Both migrations are only needed for a database that predates this
refactor — a fresh database (via `Base.metadata.create_all()`, as
`main.py` already does on startup) gets both columns directly.
