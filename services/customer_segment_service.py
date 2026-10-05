"""
Persistence layer for calculated customer segments.

This module ONLY reads/writes `CustomerSegment` rows. It deliberately does
NOT compute RFM, does NOT run K-Means, and does NOT decide segment labels —
callers (a future orchestration layer) are expected to already have a
fully-calculated result (cluster id, label, RFM values, model version,
when it was calculated) and hand it to `upsert_current_segment`.

    RFM Adapter -> Existing Segmentation Pipeline -> Segment Result
        -> CustomerSegmentService (this module) -> CustomerSegment DB table
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ml.segmentation.model import SEGMENTATION_MODEL_VERSION
from models.customer_segment import CustomerSegment


def get_current_segment(db: Session, user_id: int) -> Optional[CustomerSegment]:
    """Returns the user's current persisted segment, or None if never calculated."""
    return db.query(CustomerSegment).filter(CustomerSegment.user_id == user_id).first()


def upsert_current_segment(
    db: Session,
    user_id: int,
    cluster_id: int,
    segment_label: str,
    recency: int,
    frequency: int,
    monetary: float,
    model_version: str = SEGMENTATION_MODEL_VERSION,
    calculated_at: Optional[datetime] = None,
    commit: bool = True,
) -> CustomerSegment:
    """Creates or replaces the user's current segment (at most one row per user).

    If a segment already exists for `user_id`, it is updated in place
    (same row/id) rather than inserting a second row — `user_id` is
    unique on this table (see models/customer_segment.py), so this method
    is the only supported way to write a segmentation result.

    `calculated_at` defaults to now (when this call runs) if not supplied
    by the caller — it is intentionally distinct from `created_at`/
    `updated_at`, which track this row's own lifecycle, not when the
    segmentation pipeline actually ran.

    `commit` defaults to True, preserving this function's original
    single-call behavior (commit + refresh immediately) for any caller
    persisting one user's segment on its own. Batch callers processing many
    users in one run (see services/segmentation_service.py) should pass
    `commit=False` so all of a run's upserts share a single transaction —
    the row is still flushed (so constraint violations surface immediately
    and the assigned/updated id is available), just not committed. The
    caller is then responsible for calling `db.commit()` once, and for
    `db.rollback()` if anything in the batch fails.
    """
    resolved_calculated_at = calculated_at or datetime.now(timezone.utc)

    segment = get_current_segment(db, user_id)
    if segment is None:
        segment = CustomerSegment(user_id=user_id)
        db.add(segment)

    segment.cluster_id = cluster_id
    segment.segment_label = segment_label
    segment.recency = recency
    segment.frequency = frequency
    segment.monetary = monetary
    segment.model_version = model_version
    segment.calculated_at = resolved_calculated_at

    if commit:
        db.commit()
        db.refresh(segment)
    else:
        db.flush()
    return segment
