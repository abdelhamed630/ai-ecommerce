from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from database.database import Base

# The segmentation model/algorithm version that produced a given row.
# Kept here (next to the model it stamps) rather than duplicated across
# callers — see ml/segmentation/model.py's SEGMENTATION_MODEL_VERSION,
# which is the single source of truth and is re-exported here only for
# convenience of anything importing the ORM model directly.
from ml.segmentation.model import SEGMENTATION_MODEL_VERSION  # noqa: F401


class CustomerSegment(Base):
    """The latest calculated RFM/K-Means segmentation result for a user.

    One-current-record-per-user by design (see `uq_customer_segment_user_id`
    below): this table stores the CURRENT segment, not a history of every
    past calculation. Recalculating a user's segment replaces this row
    in-place (see services/customer_segment_service.py's upsert) rather than
    inserting a new one.

    `cluster_id` (the raw K-Means cluster index) and `segment_label` (the
    human-readable interpretation, e.g. "Champions / Loyal") are stored as
    two separate columns on purpose: K-Means cluster ids are arbitrary and
    can shift after every retraining run, so cluster_id must never be
    treated as having a fixed business meaning. Only `segment_label` —
    produced by ml.segmentation.interpretation at calculation time — is
    safe to display or reason about long-term. See
    docs/customer_segmentation.md for the full rationale.
    """

    __tablename__ = "customer_segments"
    __table_args__ = (
        # At most one CURRENT segment per user (see class docstring).
        UniqueConstraint("user_id", name="uq_customer_segment_user_id"),
        CheckConstraint("recency >= 0", name="ck_customer_segment_recency_non_negative"),
        CheckConstraint("monetary >= 0", name="ck_customer_segment_monetary_non_negative"),
        CheckConstraint("frequency > 0", name="ck_customer_segment_frequency_positive"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Raw K-Means output — never assume a fixed meaning for this value
    # (see class docstring / docs/customer_segmentation.md).
    cluster_id = Column(Integer, nullable=False)

    # Human-readable interpretation of cluster_id, produced by
    # ml.segmentation.interpretation at calculation time. Stored
    # independently of cluster_id so the label survives a retraining run
    # that reshuffles which integer means what.
    segment_label = Column(String, nullable=False)

    # RFM profile that produced this segment, stored (not just the cluster
    # id) so the segment can be explained, debugged, and checked for drift
    # later without re-running the pipeline. See adapters.py / features.py
    # for how these are computed from Orders/OrderItems.
    recency = Column(Integer, nullable=False)
    frequency = Column(Integer, nullable=False)
    monetary = Column(Float, nullable=False)

    # Which segmentation model/algorithm version produced this row (e.g.
    # "rfm_kmeans_v1"). See ml/segmentation/model.py for the single
    # configured value — never hardcode this string elsewhere.
    model_version = Column(String, nullable=False, default=SEGMENTATION_MODEL_VERSION)

    # When the segmentation pipeline actually ran — distinct from the
    # customer's last order date and from created_at/updated_at (which
    # track this ROW's lifecycle, not the calculation itself). E.g. a
    # customer's last order was 2026-09-20 but the segment covering it
    # might not be calculated until 2026-09-27.
    calculated_at = Column(DateTime(timezone=True), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # One-directional relationship only — nothing added to the User model,
    # consistent with Order/Payment/Cart/ProductInteraction elsewhere in
    # this project.
    user = relationship("User")
