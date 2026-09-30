"""Background task status (ADMIN only)."""

from fastapi import APIRouter, Depends, HTTPException, status

from core.security import get_current_admin_user
from models.user import User
from schemas.churn import TaskStatusOut
from services import task_dispatch

router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.get("/{task_id}", response_model=TaskStatusOut, summary="Background task status (admin)")
def task_status(task_id: str, current_user: User = Depends(get_current_admin_user)):
    try:
        return task_dispatch.get_status(task_id)
    except task_dispatch.TaskQueueUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
