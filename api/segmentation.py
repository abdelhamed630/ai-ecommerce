"""
On-demand Customer Segmentation trigger.

ARCHITECTURE RULE: this endpoint is intentionally a thin authentication +
authorization + delegation layer only.

    API (this file)
        -> authentication / authorization
        -> services.segmentation_service.run_customer_segmentation
        -> response

No pandas, RFM, K-Means, interpretation, or database-batching logic lives
here — all of that stays in services/segmentation_service.py and the
ml/segmentation/ modules it calls, exactly as before this endpoint
existed. This module only decides who may call it, validates the request
shape, translates the service's own domain exception into an HTTP error,
and returns the service's result.

PERFORMANCE NOTE: POST /run still runs synchronously, inside the HTTP
request, matching its documented "runs synchronously" contract below. An
asynchronous alternative now exists: POST /refresh queues the same
underlying pipeline as a Celery task (see tasks/segmentation.py) instead of
blocking the request/response cycle. GET /segments additionally caches its
aggregate summary in Redis. See docs/customer_segmentation.md section 14.
"""

from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from core.config import settings
from core.security import get_current_admin_user, get_current_user
from database.database import get_db
from models.user import User
from schemas.segmentation import (
    MySegmentOut,
    SegmentationAnalysisOut,
    SegmentationRunRequest,
    SegmentationRunResult,
)
from core.cache import cache, segmentation_key
from schemas.churn import TaskSubmittedOut
from services import segmentation_analysis_service, task_dispatch
from services.segmentation_service import (
    InsufficientSegmentationDataError,
    run_customer_segmentation,
)

router = APIRouter(prefix="/segmentation", tags=["Segmentation"])


@router.post(
    "/run",
    response_model=SegmentationRunResult,
    summary="Run customer segmentation now",
    description=(
        "Recomputes RFM (Recency, Frequency, Monetary) for every customer "
        "with at least one COMPLETED order, re-clusters them with the "
        "application's configured K-Means model (K=4), and persists each "
        "customer's CURRENT segment. This replaces any previous segment "
        "for that customer — it does not keep a history of past runs.\n\n"
        "**Authentication**: a valid JWT is required.\n\n"
        "**Authorization**: the authenticated user must have the ADMIN "
        "role (`User.role`). SELLER and USER accounts receive "
        "`403 Forbidden`.\n\n"
        "**reference_date** (optional): an ISO date (`YYYY-MM-DD`) to use "
        "as the RFM reference point. If omitted, the existing RFM "
        "calculation's own default is used. Malformed values are rejected "
        "with `422` before reaching the segmentation pipeline.\n\n"
        "**Important**: the model's configuration (K, random_state, "
        "feature columns, algorithm) is fixed by the application and "
        "cannot be supplied by the client. If there are fewer customers "
        "with completed orders than the configured K, the model is NOT "
        "trained with a reduced K — the request fails with `400` instead.\n\n"
        "**Runs synchronously** — the request blocks until the full "
        "pipeline finishes. This may become a background job as the "
        "customer base grows; that is not implemented yet."
    ),
)
def run_segmentation(
    request: SegmentationRunRequest = SegmentationRunRequest(),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
) -> SegmentationRunResult:
    reference_date = (
        pd.Timestamp(request.reference_date) if request.reference_date else None
    )

    try:
        return run_customer_segmentation(db, reference_date=reference_date)
    except InsufficientSegmentationDataError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except SQLAlchemyError:
        # Never leak raw tracebacks/driver internals to API clients.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="A database error occurred while running segmentation.",
        )


@router.get(
    "/segments",
    response_model=SegmentationAnalysisOut,
    summary="Customer segments and statistics (admin)",
    description=(
        "Read-only, on-demand analysis: clusters customers with at least one "
        "COMPLETED order (recency, frequency, monetary; log1p + standardization + "
        "KMeans) and returns per-segment statistics. Nothing is persisted.\n\n"
        "**Authorization**: ADMIN only (`403` otherwise, `401` without a token).\n\n"
        "**n_clusters**: requested K, `1` to `SEGMENTATION_MAX_CLUSTERS` "
        "(default from `SEGMENTATION_N_CLUSTERS`). For small datasets the "
        "effective `n_clusters` in the response can be lower than requested.\n\n"
        "**include_assignments**: also return `{customer_id, segment_id}` pairs "
        "(paginated with `limit`/`offset`). Off by default; no emails or other "
        "personal data are ever returned."
    ),
)
def segmentation_segments(
    n_clusters: Optional[int] = Query(default=None, ge=1),
    include_assignments: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
) -> SegmentationAnalysisOut:
    if n_clusters is not None and n_clusters > settings.SEGMENTATION_MAX_CLUSTERS:
        raise HTTPException(
            status_code=422,
            detail=f"n_clusters must be between 1 and {settings.SEGMENTATION_MAX_CLUSTERS}",
        )

    effective_k = n_clusters if n_clusters is not None else settings.SEGMENTATION_N_CLUSTERS
    # Only the aggregate summary is cached (no per-customer assignments).
    cache_key = segmentation_key(effective_k)
    if not include_assignments:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    try:
        analysis = segmentation_analysis_service.analyze_customer_segments(
            db, n_clusters=n_clusters
        )
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="A database error occurred while analyzing segments.",
        )

    result = analysis.result
    payload = {
        "algorithm": "kmeans",
        "features_used": list(result.features_used),
        "requested_n_clusters": result.requested_n_clusters,
        "n_clusters": result.n_clusters,
        "customer_count": result.customer_count,
        "reference_date": analysis.reference_date,
        "segments": [vars(p) for p in result.profiles],
    }
    if not include_assignments:
        cache.set(
            cache_key,
            SegmentationAnalysisOut.model_validate(payload).model_dump(mode="json"),
            settings.SEGMENTATION_CACHE_TTL_SECONDS,
        )
    if include_assignments:
        pairs = sorted(result.assignments.items())
        payload["assignments_total"] = len(pairs)
        payload["assignments"] = [
            {"customer_id": cid, "segment_id": sid} for cid, sid in pairs[offset : offset + limit]
        ]
    return payload


@router.get(
    "/me",
    response_model=MySegmentOut,
    summary="My customer segment",
    description=(
        "Returns the authenticated user's OWN current segment as last "
        "persisted by `POST /segmentation/run`. The user always comes from "
        "the JWT; there is no user_id parameter (any supplied one is "
        "ignored). If the user has never been segmented, `segmented` is "
        "`false` and all other fields are null."
    ),
)
def my_segment(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MySegmentOut:
    segment = segmentation_analysis_service.get_my_segment(db, current_user.id)
    if segment is None:
        return MySegmentOut(segmented=False)
    return MySegmentOut(
        segmented=True,
        segment_id=segment.cluster_id,
        segment_label=segment.segment_label,
        recency=segment.recency,
        frequency=segment.frequency,
        monetary=segment.monetary,
        model_version=segment.model_version,
        calculated_at=segment.calculated_at,
    )


@router.post(
    "/refresh",
    response_model=TaskSubmittedOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a segmentation refresh (admin)",
    description=(
        "Queues the `segmentation.refresh` Celery task, which runs the same persisted "
        "pipeline as `POST /segmentation/run` (K fixed at the model's default; the "
        "configurable on-demand analysis is a separate, read-only path). Poll "
        "`GET /tasks/{task_id}`. `503` if the queue is unreachable (nothing queued)."
    ),
)
def refresh_segmentation(current_user: User = Depends(get_current_admin_user)):
    from tasks.segmentation import REFRESH_TASK_NAME

    try:
        return {"task_id": task_dispatch.submit(REFRESH_TASK_NAME), "status": "queued"}
    except task_dispatch.TaskQueueUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
