"""On-demand customer segmentation analysis (Phase 3): data layer + orchestration.

    Database (2 aggregate queries, no per-user queries)
        -> ml.segmentation.features.build_customer_features   (R/F/M, AOV, views, cart adds)
        -> ml.segmentation.clustering.segment_customers       (log1p, StandardScaler, KMeans)
        -> SegmentationAnalysis (assignments + segment statistics)

This service is READ-ONLY: it never writes to the database. It is independent
of the recommendation engine and of the persisted `CustomerSegment` rows
written by POST /segmentation/run (services/segmentation_service.py); the
current user's persisted segment is read with `get_my_segment`.

Purchase eligibility matches the existing adapter
(ml/segmentation/adapters.COMPLETED_PURCHASE_STATUS): only orders with
status COMPLETED count. PENDING, CONFIRMED and CANCELLED orders never
contribute to recency, frequency or monetary. Monetary is `Order.total_price`
(the price snapshot taken when the order was created, equal to the sum of its
item subtotals).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.config import settings
from ml.segmentation import clustering, features
from ml.segmentation.adapters import COMPLETED_PURCHASE_STATUS
from models.customer_segment import CustomerSegment
from models.interaction import InteractionType, ProductInteraction
from models.order import Order
from services import customer_segment_service


@dataclass(frozen=True)
class SegmentationAnalysis:
    result: clustering.SegmentationResult
    reference_date: datetime
    include_non_purchasers: bool


def load_customer_activities(
    db: Session, include_non_purchasers: bool = False
) -> List[features.CustomerActivity]:
    """One row per customer, aggregated in SQL.

    Query 1: completed orders grouped by user (count, sum, max date).
    Query 2: VIEW / CART_ADD counts grouped by (user, type), restricted to
    customers with a completed order via a subquery (no giant IN list)
    unless `include_non_purchasers` is set.
    """
    order_rows = (
        db.query(
            Order.user_id,
            func.count(Order.id),
            func.coalesce(func.sum(Order.total_price), 0.0),
            func.max(Order.created_at),
        )
        .filter(Order.status == COMPLETED_PURCHASE_STATUS)
        .group_by(Order.user_id)
        .all()
    )
    orders: Dict[int, tuple] = {row[0]: row[1:] for row in order_rows}

    interaction_query = db.query(
        ProductInteraction.user_id,
        ProductInteraction.interaction_type,
        func.count(ProductInteraction.id),
    ).filter(
        ProductInteraction.interaction_type.in_(
            [InteractionType.VIEW, InteractionType.CART_ADD]
        )
    )
    if not include_non_purchasers:
        purchasers = select(Order.user_id).where(Order.status == COMPLETED_PURCHASE_STATUS)
        interaction_query = interaction_query.filter(ProductInteraction.user_id.in_(purchasers))
    interaction_rows = interaction_query.group_by(
        ProductInteraction.user_id, ProductInteraction.interaction_type
    ).all()

    views: Dict[int, int] = {}
    cart_adds: Dict[int, int] = {}
    for user_id, interaction_type, count in interaction_rows:
        target = views if interaction_type == InteractionType.VIEW else cart_adds
        target[user_id] = count

    customer_ids = set(orders)
    if include_non_purchasers:
        customer_ids |= set(views) | set(cart_adds)

    activities = []
    for customer_id in sorted(customer_ids):
        order_count, total_spent, last_order_at = orders.get(customer_id, (0, 0.0, None))
        activities.append(
            features.CustomerActivity(
                customer_id=customer_id,
                order_count=order_count,
                total_spent=total_spent,
                last_order_at=last_order_at,
                views_count=views.get(customer_id, 0),
                cart_add_count=cart_adds.get(customer_id, 0),
            )
        )
    return activities


def analyze_customer_segments(
    db: Session,
    n_clusters: Optional[int] = None,
    reference_date: Optional[datetime] = None,
) -> SegmentationAnalysis:
    """Cluster all eligible customers and return assignments + statistics.

    `n_clusters` defaults to settings.SEGMENTATION_N_CLUSTERS. Range
    validation (<= SEGMENTATION_MAX_CLUSTERS) is the API's job; the ML layer
    only requires >= 1 and shrinks K for small datasets. `reference_date`
    (recency baseline) defaults to the current UTC time.
    """
    requested = n_clusters if n_clusters is not None else settings.SEGMENTATION_N_CLUSTERS
    reference = reference_date or datetime.now(timezone.utc)
    include_non_purchasers = settings.SEGMENTATION_INCLUDE_NON_PURCHASERS

    activities = load_customer_activities(db, include_non_purchasers)
    customer_features = features.build_customer_features(activities, reference)
    result = clustering.segment_customers(
        customer_features,
        n_clusters=requested,
        random_state=settings.SEGMENTATION_RANDOM_STATE,
        cluster_features=settings.SEGMENTATION_CLUSTER_FEATURES,
    )
    return SegmentationAnalysis(
        result=result, reference_date=reference, include_non_purchasers=include_non_purchasers
    )


def get_my_segment(db: Session, user_id: int) -> Optional[CustomerSegment]:
    """The caller's own persisted segment (from POST /segmentation/run), or
    None. `user_id` must be the authenticated user's id."""
    return customer_segment_service.get_current_segment(db, user_id)
