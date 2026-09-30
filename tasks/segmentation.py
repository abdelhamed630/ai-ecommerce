"""Segmentation background tasks. Reuses the existing persisted pipeline."""

from celery_app import celery_app
from database import database
from services import cache_invalidation
from services.segmentation_service import (
    InsufficientSegmentationDataError,
    run_customer_segmentation,
)

REFRESH_TASK_NAME = "segmentation.refresh"


@celery_app.task(name=REFRESH_TASK_NAME)
def refresh_customer_segmentation() -> dict:
    """Runs services.segmentation_service.run_customer_segmentation (the same
    persisted RFM/K-Means pipeline as POST /segmentation/run; no algorithm is
    duplicated here) and evicts cached segmentation summaries.

    Fewer customers than the model's fixed K is an expected state and is
    returned as {"status": "insufficient_data"}; other errors fail the task.
    """
    db = database.SessionLocal()
    try:
        summary = run_customer_segmentation(db)
        cache_invalidation.invalidate_segmentation_caches()
        return {"status": "completed", **summary}
    except InsufficientSegmentationDataError as exc:
        return {"status": "insufficient_data", "reason": str(exc)}
    finally:
        db.close()
