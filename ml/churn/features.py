"""Churn features from per-customer aggregates. Pure Python/NumPy.

The database layer (services/churn_service.py) aggregates events into a
`ChurnAggregate` using ONLY rows with timestamp <= the cutoff, for both
training and inference. `build_feature_row` then derives the model features,
so training and prediction share one definition.

Leakage guard: if an aggregate contains a timestamp after the cutoff, feature
building raises instead of silently using the future.

Features (all measured at the cutoff):
  recency_days              days since the last completed order
  frequency                 completed orders so far
  monetary                  total spent on completed orders so far
  average_order_value       monetary / frequency
  views_count               VIEW events so far
  cart_add_count            CART_ADD events so far
  days_since_last_activity  days since the latest order OR interaction
  avg_days_between_orders   (last - first order) / (orders - 1); missing (NaN) with < 2 orders
  customer_lifetime_days    days since the first order OR interaction
  recent_activity_count     orders + interactions in the last `recent_days` days

Not used: PURCHASE interaction counts (the completed order is the authoritative
purchase record and would just duplicate `frequency`).
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

import numpy as np

FEATURE_NAMES = (
    "recency_days",
    "frequency",
    "monetary",
    "average_order_value",
    "views_count",
    "cart_add_count",
    "days_since_last_activity",
    "avg_days_between_orders",
    "customer_lifetime_days",
    "recent_activity_count",
)
# Right-skewed counts/currency (log1p before scaling); the rest are used as is.
SKEWED_FEATURES = (
    "frequency",
    "monetary",
    "average_order_value",
    "views_count",
    "cart_add_count",
    "recent_activity_count",
)

_SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class ChurnAggregate:
    """Aggregates of events with timestamp <= cutoff for one customer."""

    customer_id: int
    order_count: int = 0
    total_spent: float = 0.0
    first_order_at: Optional[datetime] = None
    last_order_at: Optional[datetime] = None
    recent_order_count: int = 0
    views_count: int = 0
    cart_add_count: int = 0
    first_interaction_at: Optional[datetime] = None
    last_interaction_at: Optional[datetime] = None
    recent_interaction_count: int = 0


def to_naive_utc(moment: Optional[datetime]) -> Optional[datetime]:
    if moment is None:
        return None
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


def _count(value) -> int:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return 0
    return int(n) if math.isfinite(n) and n > 0 else 0


def _amount(value) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return 0.0
    return n if math.isfinite(n) and n > 0 else 0.0


def _days(later: datetime, earlier: datetime) -> float:
    return max(0.0, (later - earlier).total_seconds() / _SECONDS_PER_DAY)


def is_eligible(aggregate: ChurnAggregate) -> bool:
    """Churn is only defined for customers with >= 1 completed order."""
    return _count(aggregate.order_count) >= 1


def build_feature_row(aggregate: ChurnAggregate, cutoff: datetime) -> Optional[dict]:
    """Feature dict {name: float} or None if the customer is not eligible.
    Missing values are NaN (imputed inside the fitted pipeline)."""
    cutoff = to_naive_utc(cutoff)
    if not is_eligible(aggregate):
        return None

    first_order = to_naive_utc(aggregate.first_order_at)
    last_order = to_naive_utc(aggregate.last_order_at)
    first_inter = to_naive_utc(aggregate.first_interaction_at)
    last_inter = to_naive_utc(aggregate.last_interaction_at)
    for moment in (first_order, last_order, first_inter, last_inter):
        if moment is not None and moment > cutoff:
            raise ValueError(
                f"aggregate for customer {aggregate.customer_id} contains data after the "
                f"cutoff ({moment} > {cutoff}); refusing to build features (target leakage)"
            )

    orders = _count(aggregate.order_count)
    monetary = _amount(aggregate.total_spent)
    last_activity_candidates = [m for m in (last_order, last_inter) if m is not None]
    first_activity_candidates = [m for m in (first_order, first_inter) if m is not None]

    avg_gap = float("nan")
    if orders >= 2 and first_order is not None and last_order is not None:
        avg_gap = _days(last_order, first_order) / (orders - 1)

    return {
        "recency_days": _days(cutoff, last_order) if last_order is not None else float("nan"),
        "frequency": float(orders),
        "monetary": monetary,
        "average_order_value": monetary / orders,
        "views_count": float(_count(aggregate.views_count)),
        "cart_add_count": float(_count(aggregate.cart_add_count)),
        "days_since_last_activity": (
            _days(cutoff, max(last_activity_candidates)) if last_activity_candidates else float("nan")
        ),
        "avg_days_between_orders": avg_gap,
        "customer_lifetime_days": (
            _days(cutoff, min(first_activity_candidates)) if first_activity_candidates else float("nan")
        ),
        "recent_activity_count": float(
            _count(aggregate.recent_order_count) + _count(aggregate.recent_interaction_count)
        ),
    }


def rows_to_matrix(rows: Iterable[dict], feature_names=FEATURE_NAMES) -> np.ndarray:
    """Rows -> 2-D float array in `feature_names` order (missing key -> NaN)."""
    data: List[List[float]] = [
        [float(row.get(name, float("nan"))) for name in feature_names] for row in rows
    ]
    return np.array(data, dtype=float).reshape(len(data), len(feature_names))
