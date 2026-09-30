from datetime import date, datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class SegmentationRunRequest(BaseModel):
    """Request body for POST /segmentation/run.

    Only `reference_date` may be supplied. The segmentation model's own
    configuration — K, random_state, feature columns, algorithm — is fixed
    by the application (see ml/segmentation/model.py) and is never
    accepted from a client. `extra="forbid"` makes that explicit: sending
    any other field (e.g. an attempted `n_clusters`) is rejected as a
    validation error rather than silently ignored.

    `reference_date` is a real `date`, so pydantic rejects a malformed
    value (e.g. "not-a-date") with a validation error before it ever
    reaches the segmentation service or pandas.
    """

    reference_date: Optional[date] = None

    model_config = ConfigDict(extra="forbid")


class SegmentationRunResult(BaseModel):
    """Mirrors the summary dict returned by
    services.segmentation_service.run_customer_segmentation."""

    customers_processed: int
    segments: Dict[str, int]
    model_version: str


class SegmentProfileOut(BaseModel):
    """Aggregate statistics for one segment. No customer identities."""

    segment_id: int
    customer_count: int
    average_recency: float  # days since last completed order
    average_frequency: float  # completed orders
    average_monetary: float  # total spent
    average_order_value: float
    average_views: float
    average_cart_adds: float


class SegmentAssignmentOut(BaseModel):
    customer_id: int
    segment_id: int


class SegmentationAnalysisOut(BaseModel):
    """Response of GET /segmentation/segments (admin only)."""

    algorithm: str
    features_used: List[str]
    requested_n_clusters: int
    n_clusters: int  # effective K (smaller than requested for small datasets)
    customer_count: int
    reference_date: datetime
    segments: List[SegmentProfileOut]
    # Only present when include_assignments=true; paginated by limit/offset.
    assignments: Optional[List[SegmentAssignmentOut]] = None
    assignments_total: Optional[int] = None


class MySegmentOut(BaseModel):
    """Response of GET /segmentation/me: the caller's OWN persisted segment."""

    segmented: bool
    segment_id: Optional[int] = None
    segment_label: Optional[str] = None
    recency: Optional[int] = None
    frequency: Optional[int] = None
    monetary: Optional[float] = None
    model_version: Optional[str] = None
    calculated_at: Optional[datetime] = None
