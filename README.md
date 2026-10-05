# AI E-Commerce Backend

FastAPI backend for an e-commerce store with AI features: a Groq-powered, product-aware shopping chatbot
(with streaming), content-based / collaborative / hybrid recommendations, RFM customer segmentation and churn
prediction. PostgreSQL + Alembic for data, Redis + Celery for caching and background jobs, Docker + Nginx for
deployment.

## Features

- **FastAPI** REST API with OpenAPI docs
- **PostgreSQL** (SQLAlchemy 2) with **Alembic** migrations
- **Redis** (cache, rate-limit counters, Celery broker/result backend) and **Celery** workers
- **Docker Compose** stack and **Nginx** reverse proxy
- **JWT authentication** with roles `USER` / `SELLER` / `ADMIN` (role read from the database)
- **Product APIs** with DB-side filtering (`q`, `category`, `min_price`, `max_price`, `skip`, `limit`)
- **AI chatbot** (Groq, OpenAI-compatible API) that only talks about real products from the database
- **Product-aware recommendations** inside chatbot answers, plus content-based, collaborative and hybrid recommenders
- **Conversation history** (client-sent turns, server trims to `CHATBOT_MAX_HISTORY_TURNS`)
- **Streaming responses** (Server-Sent Events) at `POST /chatbot/chat/stream`
- **Rate limiting**: per-user chatbot limit in Redis (`CHATBOT_RATE_LIMIT_PER_MINUTE`) plus a per-IP Nginx limit
- **Arabic text normalization** shared by product search and the chatbot
- Customer segmentation, churn prediction, cart / orders / simulated payments, admin user management
- **Automated tests** (pytest)

## Verification status

Reported by the maintainer from their local Docker environment (not re-run in the documentation pass):
1215 tests collected, 1212 passed, 3 skipped, 0 failed; Alembic at `0003` (head); containers `api`, `postgres`,
`redis`, `celery_worker`, `nginx` healthy; `/health/ready` reports `database: ok`, `redis: ok`; registration, login,
`/auth/me`, products, chatbot and streaming verified. The GitHub Actions workflow (`.github/workflows/ci.yml`)
has **not** been run yet.

## Architecture

```
Browser / frontend
        |
        v
   Nginx :8080  (reverse proxy; per-IP limit + no buffering on /chatbot/; TLS terminated upstream)
        |
        v
   FastAPI (uvicorn workers)  --- JWT auth, REST routers, services
        |-- PostgreSQL   (schema owned by Alembic)
        |-- Redis        (cache, chatbot rate limit, Celery broker/results)
        |-- Groq API     (chatbot LLM; optional, answers degrade gracefully without a key)
        |-- ML artifacts (churn model on a volume)

   Celery worker --- churn.train_model, segmentation.refresh, recommendations.refresh_user_cache
   Celery beat   --- optional schedule (off by default)
```

## Project structure

```
api/          FastAPI routers (auth, products, chatbot, cart, orders, payments, admin, health, ...)
core/         config, security (JWT/roles), cache, rate limiting, logging, text normalization
database/     SQLAlchemy engine/session
models/       SQLAlchemy models
schemas/      Pydantic request/response schemas
services/     business logic (products, chatbot, LLM client, recommendations, ...)
ml/           recommendation, segmentation and churn code
tasks/        Celery tasks
alembic/      migrations (0001 -> 0002 -> 0003)
nginx/        nginx.conf
scripts/      admin/bootstrap and historical migration helpers
tests/        pytest suite
docs/         feature docs and the frontend API contract
```

| Concern | Where |
|---|---|
| Authentication / roles | `core/security.py`, `api/auth.py` |
| Chatbot | `api/chatbot.py`, `services/chatbot_service.py`, `services/llm_service.py`, `services/product_search_service.py` |
| Rate limiting | `core/rate_limit.py`, `nginx/nginx.conf` |
| Recommendations | `ml/recommendation/*`, `services/recommendation_service.py` |
| Segmentation / churn | `ml/segmentation/*`, `ml/churn/*`, matching services and routers |
| Background jobs | `celery_app.py`, `tasks/*`, `api/tasks.py` |

## Requirements

- Docker and Docker Compose for the full stack, or PostgreSQL and Redis installed locally
- Python (the Docker image uses 3.12); dependencies in `requirements.txt` (runtime) and `requirements-dev.txt` (tests)
- A Groq API key for the chatbot LLM (optional; without it the chatbot answers "temporarily unavailable" and still returns products)

## Environment setup

Never commit real secrets. Only `.env.example` and `.env.docker.example` are tracked; `.env` and `.env.docker`
are git-ignored.

Local development:

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
cp .env.example .env                                   # Windows: Copy-Item .env.example .env
# edit .env: DATABASE_URL (PostgreSQL), SECRET_KEY (openssl rand -hex 32), GROQ_API_KEY (optional)
alembic upgrade head
uvicorn main:app --reload
```

Docker: see below. Full variable reference: [docs/production.md](docs/production.md#1-configuration-reference).

## Docker setup

```bash
cp .env.docker.example .env.docker          # replace every CHANGE_ME; use URL-safe values
docker compose --env-file .env.docker config
docker compose --env-file .env.docker build
docker compose --env-file .env.docker run --rm api alembic upgrade head
docker compose --env-file .env.docker up -d
docker compose --env-file .env.docker ps
curl http://localhost:8080/health/ready
```

Services: `api`, `postgres`, `redis`, `celery_worker`, `nginx` (published on `NGINX_PORT`, default 8080), and
`celery_beat` behind the `beat` profile. Postgres and Redis are not published. Never run `docker compose down -v`
unless you intend to delete the database and Redis volumes.

Create the first admin without an HTTP endpoint (register the user through `/auth/register` first):
`docker compose --env-file .env.docker run --rm api python scripts/set_user_role.py --help`

## Database migrations

```bash
alembic current
alembic heads
alembic upgrade head          # the ONLY way production schemas are created/changed
alembic revision --autogenerate -m "describe change"   # then review the generated file
```

## Running tests

```bash
python -m pytest -q
```

Tests are isolated from your real database: a temporary database file, or `TEST_DATABASE_URL` (its name must
contain `test`; its schema is dropped and recreated). Tests use fakeredis and Celery eager mode. To run against
PostgreSQL, set `TEST_DATABASE_URL` and `ALEMBIC_TEST_DATABASE_URL` to an empty `*_test` database.

## API documentation

- Interactive docs: `http://localhost:8080/docs` (Docker) or `http://localhost:8000/docs` (local), unless `ENABLE_API_DOCS=false`
- Schema snapshot: `openapi.json` in the repository root
- Frontend contract: [docs/FRONTEND_API.md](docs/FRONTEND_API.md)
- Feature docs: [authentication](docs/authentication.md), [chatbot](docs/chatbot.md), [profile](docs/profile.md),
  [recommendations](docs/recommendations.md), [segmentation](docs/customer_segmentation.md),
  [churn](docs/churn_prediction.md), [production](docs/production.md)

## Frontend integration

Base URL through Nginx: `http://localhost:8080`. Send `Authorization: Bearer <access_token>`.
`POST /auth/login` takes form-encoded `username` (the email) and `password`. The chatbot stream is
`POST` + Server-Sent Events, so use `fetch`, not `EventSource`.
**CORS:** `CORS_ALLOWED_ORIGINS` in `.env.docker` must contain the exact origin of the frontend
(e.g. `https://shop.example.com`); while it is empty, browsers on another origin are blocked. `ALLOWED_HOSTS`
must contain the hostname used to reach the API. Details: [docs/FRONTEND_API.md](docs/FRONTEND_API.md).

## Celery

```bash
celery -A celery_app.celery_app worker --loglevel=info      # worker
celery -A celery_app.celery_app beat --loglevel=info        # optional; needs CELERY_BEAT_ENABLED=true
```

Admin endpoints queue jobs: `POST /churn/train`, `POST /segmentation/refresh`; status: `GET /tasks/{id}`.

## Security notes

- Secrets live only in git-ignored env files; the Docker image excludes them (`.dockerignore`).
- Production mode (`ENVIRONMENT=production`) refuses to start with a weak or placeholder `SECRET_KEY`, `DEBUG=true`,
  SQLite, wildcard hosts/CORS or placeholder passwords.
- Public registration always creates a `USER`; admin/seller roles are assigned only via `scripts/set_user_role.py`
  or the protected admin API.
- Chatbot endpoints require authentication and are rate-limited per user (Redis) and per IP (Nginx).
- Unhandled errors return a generic 500; details are only in redacted server logs.
- If a key or `SECRET_KEY` was ever shared outside your machine, rotate it.
- TLS is not configured in Nginx: terminate HTTPS at your load balancer/ingress.

## Known limitations

See [docs/production.md](docs/production.md#9-known-limitations--technical-debt): Float money columns,
no SQLite->PostgreSQL data migration, no token refresh/revocation or account lockout (rate limiting exists only for the
chatbot), no Celery retries, single-node avatar storage, simulated payments, no TLS configuration in Nginx,
unpinned dependencies.
