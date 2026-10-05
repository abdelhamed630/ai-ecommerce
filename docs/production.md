# Production guide (Phase 5)

> **Verification status.** Phase 5 was written in a sandbox with no network, Docker,
> PostgreSQL or Redis, and without the Python dependencies installed. Nothing in this
> document has been *executed* unless it says VERIFIED. See the verification matrix in the
> Phase 5 report and the "Verify it yourself" section of the README.

## 1. Configuration reference

All settings are environment variables (or `.env` for local development); see
`core/config.py` for the authoritative list. Phase 5 additions and changes:

| Variable | Default | Notes |
|---|---|---|
| `ENVIRONMENT` | `development` | `production` / `staging` enable fail-fast validation. |
| `DEBUG` | `false` | Was `true`. Local dev opts in through `.env`. Rejected in production. |
| `SECRET_KEY` | *(empty)* | Outside production an empty value falls back to a dev-only key and logs a warning. In production it must be >= 32 chars (no leading/trailing whitespace), have >= 8 distinct characters, and not be a placeholder (any `CHANGE_ME`/`change-this`/`not-for-production` marker, matched case-insensitively), otherwise startup aborts. Generate: `openssl rand -hex 32`. |
| `ALGORITHM` | `HS256` | JWT signing algorithm; only `HS256`, `HS384`, `HS512` are accepted (exact, case-sensitive), in every environment. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Must be between 1 and 10080 (7 days). |
| `DATABASE_URL` | `sqlite:///./ecommerce.db` | Production: `postgresql+psycopg://USER:PASSWORD@HOST:5432/DB`. Bare `postgresql://` is rewritten to the psycopg 3 driver. SQLite is rejected in production. |
| `DB_POOL_SIZE` / `DB_MAX_OVERFLOW` / `DB_POOL_TIMEOUT_SECONDS` / `DB_POOL_RECYCLE_SECONDS` | 10 / 20 / 30 / 1800 | Bounds: 1-100 / 0-200 / 1-300 / 30-86400. Non-SQLite only. `pool_pre_ping` is always on. Total connections per API process <= size + overflow; multiply by `WEB_CONCURRENCY`, and keep the sum below PostgreSQL `max_connections` (default 100). |
| `ALLOWED_HOSTS` | `*` | Comma-separated. Production: explicit names, no `*`. Must include names used by health checks (`localhost`, `127.0.0.1`). Each entry must be a bare hostname, `*.domain` or `*`: no scheme, port or path (Starlette matches the Host header without its port). Malformed entries are rejected in every environment. |
| `CORS_ALLOWED_ORIGINS` | *(empty)* | Comma-separated origins. Empty = no CORS middleware (no cross-origin browser access). `*` only outside production, and then without credentials. Each entry must be `http(s)://host[:port]` with no path, trailing slash, credentials, query or fragment; malformed entries are rejected in every environment. |
| `ENABLE_API_DOCS` | `true` | Serves `/docs`, `/redoc`, `/openapi.json`. Set `false` if you do not want the API surface published. |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `text` | `LOG_FORMAT=json` for log aggregation. |
| `READINESS_REQUIRE_REDIS` | *(auto)* | Auto: Redis is required for `/health/ready` in production (cache enabled), optional elsewhere. |
| `REDIS_URL`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` | localhost defaults | Must start with `redis://` or `rediss://` (every environment). In production a `CHANGE_ME`-style password inside the URL is rejected, as is `CELERY_TASK_ALWAYS_EAGER=true`. In Docker they point at the `redis` service with a password. |
| `CELERY_BEAT_ENABLED` and two interval settings | `false` / weekly / daily | See section 5. |
| `CHURN_MODEL_PATH`, `MEDIA_ROOT` | relative paths | Docker sets both to the `/data` volumes. |

Numeric tuning settings (pool, cache TTLs, Celery intervals, hybrid/segmentation/churn parameters) are range-checked at startup, and related settings must be consistent (`SEGMENTATION_N_CLUSTERS` <= `SEGMENTATION_MAX_CLUSTERS`, hybrid weights not both 0, `CHURN_RISK_MEDIUM_THRESHOLD` < `CHURN_RISK_HIGH_THRESHOLD`); see `core/config.py` for each range.

Configuration errors name the offending setting but never print secret values.

## 2. Database: PostgreSQL + Alembic

* Schema management is **Alembic only**. `main.py` no longer calls `create_all()`; the
  application assumes the schema is already migrated. Tests build their own schemas
  (`tests/conftest.py`).
* `0001_baseline_schema` was **written by hand** from the models (autogenerate could not be
  run). `0002_phase5_order_indexes` adds two indexes. `tests/test_alembic_migrations.py`
  compares `upgrade head` with `Base.metadata` on SQLite (and on PostgreSQL when
  `ALEMBIC_TEST_DATABASE_URL` is set) so any mistake in the baseline should show up there.
* New database: `alembic upgrade head`.
* **Existing SQLite/dev database** created by the old `create_all()` + `scripts/migrate_*.py`:
  `alembic stamp 0001` then `alembic upgrade head`. Only stamp a database that really has the
  Phase 4 schema; the old `scripts/migrate_*.py` files are historical and are not needed for
  new databases.
* Data migration from SQLite to PostgreSQL is **not provided**.

### PostgreSQL compatibility audit

| Topic | Finding |
|---|---|
| Enums (`role`, order/payment status, interaction type, payment method) | All use `native_enum=False` -> plain `VARCHAR`, portable, no PostgreSQL `ENUM` types, no CHECK constraints. Unchanged. |
| UUID / JSON columns | None in the schema. |
| Timestamps | `DateTime(timezone=True)` with `server_default=now()`; PostgreSQL stores `timestamptz`. Application code that compares naive/aware datetimes (churn/segmentation use `to_naive_utc`) was **not** exercised on PostgreSQL: NOT VERIFIED. |
| Unique constraints / NULLs | `username` unique with NULLs, `transaction_reference` unique nullable: PostgreSQL allows multiple NULLs, same as SQLite. |
| Foreign keys | Declared; no `ON DELETE` rules (default `NO ACTION`). PostgreSQL enforces FKs strictly, SQLite (without the pragma) does not: deleting a referenced row that used to "work" on SQLite can now fail. NOT VERIFIED against the test-suite on PostgreSQL. |
| Booleans, text | Portable. |
| Sequences | PostgreSQL uses SERIAL/identity from `Integer` primary keys. |

### Money fields (audit; **not changed**)

Every monetary value is a `Float` (`double precision` on PostgreSQL):

| Table.column | Meaning |
|---|---|
| `products.price` | current list price |
| `orders.total_price` | snapshot of the order total |
| `order_items.product_price`, `order_items.subtotal` | price/subtotal snapshots |
| `payments.amount` | snapshot of `orders.total_price` |
| `customer_segments.monetary` | RFM monetary value (analytics, not accounting) |

Binary floating point cannot represent most decimal amounts exactly, so sums and equality
checks can drift by fractions of a cent. Converting to `Numeric(12, 2)` was **deliberately not
done**: SQLAlchemy would return `Decimal`, which cannot be mixed with the `float` arithmetic in
the services/ML code and changes JSON/pydantic behaviour and test equality. Recommended future
work: `Numeric(12,2)` (or integer minor units) + a `Decimal`-aware service layer + a data
migration, as its own change with its own tests. Tracked as technical debt.

### Index audit

Existing indexes already cover: every `user_id` FK (`orders`, `payments`, `product_interactions`,
`customer_segments`; `carts` via its unique constraint), `products.seller_id`, `payments.order_id`,
`product_interactions` (`user_id`, `product_id`, `interaction_type`, `created_at` + three
composites), `cart_items.cart_id` (leading column of a unique constraint).

Added (migration `0002`, justified by query patterns):

* `ix_orders_status_created_at (status, created_at)`: churn/segmentation filter
  `status = COMPLETED [AND created_at <= cutoff]` then `GROUP BY user_id`.
* `ix_order_items_order_id (order_id)`: `Order.items` loading and the RFM
  `Order JOIN OrderItem`. PostgreSQL does not index foreign keys automatically.

Deliberately **not** added: `order_items.product_id`, `cart_items.product_id`,
`products.created_at` (recommendation fallback ordering on a small catalog): no query pattern
found that needs them yet. Whether PostgreSQL's planner uses the new indexes at your data size
is NOT VERIFIED (small tables are usually scanned sequentially).

## 3. Application server

`uvicorn main:app --workers N --proxy-headers --no-access-log` (see the Dockerfile `CMD`).
Gunicorn was **not** introduced: the `uvicorn.workers` module is deprecated in current uvicorn
releases (the replacement is the separate `uvicorn-worker` package) and the project's exact
versions are unpinned, so uvicorn's own multi-process mode is the lower-risk choice.
Requests are logged by the application's middleware (method, path, status, duration,
`X-Request-ID`; never query strings, headers or bodies).

Development: `uvicorn main:app --reload`.

## 4. Health endpoints

* `GET /health` (and `/health/live`): liveness, touches no dependency: `{"status":"ok"}`.
* `GET /health/ready`: `SELECT 1` on the database, Redis `PING` when the cache is enabled.
  `200 {"status":"ready","checks":{"database":"ok","redis":"ok"}}` or
  `503 {"status":"not_ready", ...}`. Values are only `ok` / `unavailable` / `skipped`; no
  exception text, hostnames or credentials. Redis is fail-open for requests, so it only
  fails readiness when required (default: production).

## 5. Redis and Celery

* The cache abstraction is unchanged (fail-open, key isolation by authenticated user id).
* Redis in compose has `appendonly yes` + password because it also holds Celery messages/results.
* Worker: `celery -A celery_app.celery_app worker --loglevel=info`. Each task opens and closes its
  **own** `SessionLocal()` (unchanged from Phase 4); request-scoped sessions are never shared.
* Task retries/acks-late/time limits were **not** added: task behaviour was left untouched.
  Failed tasks are stored as `FAILURE` and visible via `GET /tasks/{id}` (admin).
* **Beat (scheduled jobs) is off by default.** Enable with `CELERY_BEAT_ENABLED=true` and run exactly one
  beat process (`docker compose --profile beat up -d`). Weekly churn training and daily
  segmentation refresh reuse the existing tasks. Segmentation refresh and churn training read whole
  order/interaction tables in-process; only schedule them once you have measured runtime on real
  data. Otherwise trigger them manually (`POST /churn/train`, `POST /segmentation/refresh` as admin).

## 6. Model artifacts

* Churn model: single joblib file at `CHURN_MODEL_PATH` (Docker: `/data/artifacts/churn/churn_model.joblib`
  on the `model_artifacts` volume, shared by the API and the worker). Not baked into the image
  (`ml/artifacts/` is in `.gitignore` and `.dockerignore`).
* The worker writes it atomically after training; API processes reload it when the file's mtime/size
  changes; the worker also clears the cached churn predictions in Redis.
* If the file is missing, churn prediction endpoints return the existing "model unavailable" error
  until training is run (`POST /churn/train`).
* Backup/replace: copy the file (and its metadata) while the worker is idle; replacing the file is
  picked up by the API without restart. joblib files are pickles: only load files you produced
  yourself.
* `ml/segmentation/predict.py` `save_artifacts/load_artifacts` (default `ml/artifacts/segmentation`)
  are not called by the running application (the persisted segmentation state is in the
  `customer_segments` table), so no volume is provisioned for them.

## 7. Docker

See the README for commands. Design notes: one image (`python:3.12-slim`, non-root uid 10001) used for
`api`, `celery_worker`, `celery_beat`; only Nginx publishes a port; `postgres`/`redis` have healthchecks and
`api`/`celery_worker` wait for them with `depends_on: condition: service_healthy` (no sleeps);
migrations are an explicit one-off command, never run by the API on startup. Nginx runs its master
process as root (official image default) with unprivileged workers. **TLS is not configured**:
terminate HTTPS at your load balancer/ingress or add certificates to `nginx/nginx.conf`.

Python 3.12 is used because current numpy/pandas/scikit-learn releases require >= 3.11.

## 8. Security review (summary)

| Area | Result |
|---|---|
| SECRET_KEY | No unsafe default; production fails fast. |
| DEBUG | Defaults false; rejected in production (Starlette debug tracebacks are therefore off). |
| CORS / hosts | Configurable; production rejects `*`. |
| Errors | Unhandled exceptions return `{"detail":"Internal server error"}`; details only in redacted server logs. FastAPI's default 422 validation responses still echo the offending input to the *caller* (unchanged format). |
| Logging | Redaction filter for URL credentials, bearer tokens and `password=`-style pairs; no bodies/queries logged. Redaction is defense in depth, not a substitute for not logging secrets. |
| Auth | Unchanged: JWT (HS256) + bcrypt; roles read from the DB; admin-only endpoints unchanged. No rate limiting or token revocation exists (limitation). |
| Tokens | `create_access_token` uses timezone-aware `datetime.now(timezone.utc)` for `exp` (previously the deprecated `utcnow()`); the token structure is unchanged. |
| Cache/user isolation | Unchanged (keys built from the authenticated user id only). |
| Secrets in repo | `.env` is git-ignored and excluded from the Docker image; only `*.example` files are tracked. **The uploaded zip contained a `.env`: rotate its `SECRET_KEY` if it was ever shared.** |
| Docker | Non-root app user; no published DB/Redis ports; secrets come from `--env-file`. Redis password is visible in the container's process list (`redis-server --requirepass`); acceptable inside a private network, use Docker secrets/a managed Redis if that is not enough. |

## 9. Known limitations / technical debt

* Float money columns (section 2).
* No SQLite -> PostgreSQL data migration tool.
* The main test-suite's per-module tests still use their own SQLite databases; only tests using the
  default `SessionLocal` (and the health/migration tests) can run on PostgreSQL (`TEST_DATABASE_URL`).
* No rate limiting, account lockout, token refresh/revocation.
* No Celery retries/time limits; no dead-letter handling.
* Avatars are stored on a local volume (single-node); use object storage for multiple API hosts.
* Dependencies are unpinned (see README, "Dependency pinning").
* Payments are simulated (no real provider integration).
