# AI E-Commerce API

FastAPI backend for an e-commerce store with AI features: content-based, collaborative and hybrid
product recommendations, RFM/K-Means customer segmentation and churn prediction. Redis caching and
Celery background jobs; PostgreSQL + Alembic + Docker for production.

> **Verification status of Phase 5 (productionization):** written without access to Docker,
> PostgreSQL, Redis or installed dependencies, so the new files were only syntax-checked.
> The 530-test Phase 4 baseline passed on the developer's machine; the suite has **not** been
> re-run since Phase 5. Run the checks in [Verify it yourself](#verify-it-yourself) before trusting any of it.
> Details: [docs/production.md](docs/production.md).

## Architecture

```
Client
  |
Nginx (reverse proxy; TLS terminated upstream)
  |
FastAPI (uvicorn workers)  --- auth (JWT, roles from DB), REST routers, services
  |-- PostgreSQL   (SQLAlchemy; schema owned by Alembic)
  |-- Redis        (cache, fail-open)  +  Celery broker/result backend
  |-- ML artifacts (churn model on a volume)
Celery worker  --- churn.train_model, segmentation.refresh, recommendations.refresh_user_cache
Celery beat    --- optional schedule (off by default)
```

| Concern | Where |
|---|---|
| Authentication / roles | `core/security.py`, `api/auth.py` (JWT authentication only; role read from DB) |
| Recommendations | `ml/recommendation/*` (content-based, collaborative, hybrid), `services/recommendation_service.py`, `api/recommendations.py` |
| Segmentation | `ml/segmentation/*`, `services/segmentation_*.py`, `api/segmentation.py` |
| Churn | `ml/churn/*`, `services/churn_service.py`, `api/churn.py` |
| Caching | `core/cache.py` (Redis, per-user keys, TTLs in settings) |
| Background jobs | `celery_app.py`, `tasks/*`, `services/task_dispatch.py`, `api/tasks.py` |
| Migrations | `alembic/` |
| Health | `api/health.py` |

Feature docs: [recommendations](docs/recommendations.md), [segmentation](docs/customer_segmentation.md),
[churn](docs/churn_prediction.md), [authentication](docs/authentication.md), [profile](docs/profile.md).

## Local development (SQLite)

Python 3.11+ is required (current numpy/pandas/scikit-learn); the Docker image uses 3.12.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
cp .env.example .env                                   # Windows: Copy-Item .env.example .env
alembic upgrade head                                   # creates ecommerce.db (the app no longer creates tables itself)
uvicorn main:app --reload
```

Already have a development `ecommerce.db` from earlier phases? Do **not** re-run the baseline:

```bash
alembic stamp 0001
alembic upgrade head        # adds the two Phase 5 indexes
```

API docs: http://localhost:8000/docs (unless `ENABLE_API_DOCS=false`).

## Tests

```bash
python -m pytest -q
```

The default database used by tests is isolated: a temp SQLite file, or `TEST_DATABASE_URL` (its name
must contain `test`; its schema is dropped and recreated). Your `DATABASE_URL`/`.env` database is never
used. Tests use fakeredis and Celery eager mode; no Redis/PostgreSQL is required.

Optional PostgreSQL run (needs a running PostgreSQL with an empty `*_test` database):

```bash
export TEST_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/ai_ecommerce_test
export ALEMBIC_TEST_DATABASE_URL=$TEST_DATABASE_URL
python -m pytest -q
```

## Configuration

Copy `.env.example` (development) or `.env.docker.example` (Docker). Full reference and the production
rules (fail-fast on weak `SECRET_KEY`, `DEBUG`, wildcard hosts/CORS, SQLite) are in
[docs/production.md](docs/production.md#1-configuration-reference). Never commit `.env*` files other than
the `*.example` ones.

## PostgreSQL and migrations

```bash
export DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME
alembic upgrade head          # the ONLY way production schemas are created/changed
alembic current
alembic downgrade -1          # destructive for 0001: back up first
alembic revision --autogenerate -m "describe change"   # then review the generated file
```

## Docker (production-like stack)

```bash
cp .env.docker.example .env.docker          # replace every CHANGE_ME
docker compose --env-file .env.docker build
docker compose --env-file .env.docker run --rm api alembic upgrade head   # explicit migration
docker compose --env-file .env.docker up -d
curl http://localhost:8080/health/ready
```

Services: `api`, `postgres`, `redis`, `celery_worker`, `nginx` (published on `NGINX_PORT`, default 8080), and
`celery_beat` behind the `beat` profile. Postgres and Redis are not published. Volumes: `postgres_data`,
`redis_data`, `model_artifacts` (churn model), `media_data` (avatars), `beat_data`.

Create the first admin without an HTTP endpoint:
`docker compose --env-file .env.docker run --rm api python scripts/set_user_role.py --help`
(register the user through `/auth/register` first).

## Celery

```bash
celery -A celery_app.celery_app worker --loglevel=info      # worker
celery -A celery_app.celery_app beat --loglevel=info        # optional; needs CELERY_BEAT_ENABLED=true
```

Admin endpoints queue jobs: `POST /churn/train`, `POST /segmentation/refresh`; status: `GET /tasks/{id}`.

## Health endpoints

`GET /health` (liveness), `GET /health/live`, `GET /health/ready` (database + Redis when required; 503 if not ready).

## Redis

Used as the cache (per-user keys, TTL settings, fail-open) and the Celery broker/result backend. In
Docker it has a password and append-only persistence.

## CI

`.github/workflows/ci.yml` runs the full suite against PostgreSQL + Redis service containers (Python 3.12) and a
separate migration job (`alembic upgrade head`, downgrade/upgrade, `alembic check`). **It has never been run.**

## Verify it yourself

```bash
python -m pytest -q                                   # expect the old 530 + new tests to pass
alembic upgrade head && alembic check                 # on a scratch DATABASE_URL
docker compose --env-file .env.docker build
docker compose --env-file .env.docker run --rm api alembic upgrade head
docker compose --env-file .env.docker up -d && docker compose --env-file .env.docker ps
curl -i http://localhost:8080/health/ready
docker compose --env-file .env.docker exec celery_worker celery -A celery_app.celery_app inspect ping
```

Then, with an admin token: `POST /churn/train`, poll `GET /tasks/{task_id}`; `POST /segmentation/refresh`.

## Dependency pinning

`requirements.txt` is unpinned (as in earlier phases) so it matches the environment the 530 tests passed in.
For reproducible images, run `pip freeze > requirements.lock` in that environment and install with
`pip install -r requirements.txt -c requirements.lock`.

## Known limitations

See [docs/production.md](docs/production.md#9-known-limitations--technical-debt): Float money columns,
no SQLite->PostgreSQL data migration, no rate limiting, no Celery retries, single-node avatar storage,
simulated payments, no TLS configuration in Nginx.
