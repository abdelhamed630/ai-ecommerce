# syntax=docker/dockerfile:1
# One image for api, celery_worker and celery_beat (different commands in
# docker-compose.yml). Verified locally by the maintainer (image builds, containers healthy).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Unprivileged user; writable data dirs are mounted as volumes in compose.
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app

# Dependency layer first: only rebuilt when requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code (see .dockerignore: no .env, tests, media, artifacts, DB files).
COPY . .

# Model artifacts and uploads live OUTSIDE the code, on volumes.
RUN mkdir -p /data/artifacts /data/media /data/beat \
 && chown -R app:app /data /app
ENV CHURN_MODEL_PATH=/data/artifacts/churn/churn_model.joblib \
    MEDIA_ROOT=/data/media \
    ENVIRONMENT=production \
    DEBUG=false

USER app

EXPOSE 8000

# Liveness only (/health touches no dependency). Host header 127.0.0.1 must be
# accepted by ALLOWED_HOSTS.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# uvicorn's built-in multi-process mode (no gunicorn: `uvicorn.workers` is
# deprecated in current uvicorn releases and would add a dependency).
# WEB_CONCURRENCY = number of worker processes.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-2} --proxy-headers --no-access-log"]
