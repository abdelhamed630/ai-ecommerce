"""Health endpoints (no authentication, no secrets, no internals in responses).

GET /health        liveness  - the process is up. Touches NO dependency.
GET /health/live   alias of /health.
GET /health/ready  readiness - can this instance serve traffic? Checks the
                   database, and Redis when required (see
                   settings.redis_required_for_readiness). 200 when ready,
                   503 otherwise; failures are reported only as
                   "ok" / "unavailable" / "skipped", never as exception text.

Redis is fail-open for requests (cache misses fall back to the database), so
outside production a Redis outage only shows as "unavailable" in the body
while the status stays 200.
"""

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.cache import cache
from core.config import settings
from database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Liveness probe")
def health():
    return {"status": "ok"}


@router.get("/health/live", summary="Liveness probe (alias)")
def health_live():
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness probe")
def health_ready(db: Session = Depends(get_db)):
    checks = {}

    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        logger.error("readiness: database check failed (%s)", type(exc).__name__)
        checks["database"] = "unavailable"

    if not settings.CACHE_ENABLED:
        checks["redis"] = "skipped"
    else:
        checks["redis"] = "ok" if cache.ping() else "unavailable"

    ready = checks["database"] == "ok" and (
        checks["redis"] == "ok" or not settings.redis_required_for_readiness
    )
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "checks": checks},
    )
