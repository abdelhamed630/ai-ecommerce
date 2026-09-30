"""Churn prediction API.

- GET  /churn/me      authenticated user's OWN churn probability (JWT identity only)
- GET  /churn/model   ADMIN: metadata + evaluation metrics of the current model
- POST /churn/train   ADMIN: queue a training job (202); never trains in-request
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.security import get_current_admin_user, get_current_user
from database.database import get_db
from models.user import User
from schemas.churn import ChurnPredictionOut, TaskSubmittedOut
from services import churn_service, task_dispatch

router = APIRouter(prefix="/churn", tags=["Churn"])


@router.get(
    "/me",
    response_model=ChurnPredictionOut,
    summary="My churn probability",
    description=(
        "Returns the authenticated user's own `churn_probability` (0-1) and a "
        "configurable `risk_level` band. The user always comes from the JWT; no "
        "`user_id` parameter exists (any supplied one is ignored). Users with no "
        "completed order get `eligible=false` and a null probability. `503` if no "
        "model has been trained yet (or the artifact is unusable): predictions are "
        "never fabricated."
    ),
)
def my_churn(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        return churn_service.get_churn_prediction(db, current_user.id)
    except churn_service.ChurnModelUnavailableError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Churn model is not available yet: no model has been trained.",
        )
    except churn_service.ChurnArtifactError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Churn model is unusable and must be retrained.",
        )


@router.get("/model", summary="Current churn model metadata (admin)")
def churn_model_info(current_user: User = Depends(get_current_admin_user)):
    try:
        return churn_service.get_model_info()
    except churn_service.ChurnModelUnavailableError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No trained churn model.")
    except churn_service.ChurnArtifactError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Churn metadata is unreadable."
        )


@router.post(
    "/train",
    response_model=TaskSubmittedOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue churn model training (admin)",
    description=(
        "Queues the `churn.train_model` Celery task and returns its id "
        "(poll `GET /tasks/{task_id}`). `503` if the queue is unreachable: in that "
        "case nothing was queued."
    ),
)
def train_churn(current_user: User = Depends(get_current_admin_user)):
    from tasks.churn import TRAIN_TASK_NAME

    try:
        return {"task_id": task_dispatch.submit(TRAIN_TASK_NAME), "status": "queued"}
    except task_dispatch.TaskQueueUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
