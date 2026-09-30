"""Churn background tasks."""

from celery_app import celery_app
from database import database
from services import churn_service

TRAIN_TASK_NAME = "churn.train_model"


@celery_app.task(name=TRAIN_TASK_NAME)
def train_churn_model() -> dict:
    """Build the historical dataset, train, evaluate, persist the artifact.

    Not enough data is an expected outcome, returned as
    {"status": "insufficient_data", ...} (nothing persisted, no fake metrics).
    Unexpected errors propagate so Celery records the task as FAILURE.
    """
    db = database.SessionLocal()
    try:
        return churn_service.train_and_persist(db)
    except churn_service.InsufficientChurnDataError as exc:
        return {"status": "insufficient_data", "reason": str(exc)}
    finally:
        db.close()
