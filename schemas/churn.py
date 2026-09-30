from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel


class ChurnPredictionOut(BaseModel):
    """GET /churn/me. `churn_probability` is P(no completed order in the next
    CHURN_INACTIVITY_DAYS days), not a certainty. `risk_level` is a business
    band derived from configurable thresholds."""

    eligible: bool
    churn_probability: Optional[float] = None
    risk_level: Optional[str] = None
    reason: Optional[str] = None  # "no_completed_orders" when not eligible
    model_version: str
    trained_at: str
    features_as_of: str
    risk_thresholds: Dict[str, float]


class TaskSubmittedOut(BaseModel):
    task_id: str
    status: str = "queued"


class TaskStatusOut(BaseModel):
    task_id: str
    state: str
    result: Optional[Any] = None
    error: Optional[str] = None
