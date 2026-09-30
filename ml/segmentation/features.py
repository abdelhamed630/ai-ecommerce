"""
RFM (Recency, Frequency, Monetary) feature generation for the Customer
Segmentation pipeline.

Expects already-cleaned transaction data (see preprocessing.py) — this
module does not itself remove invalid rows.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from ml.segmentation.preprocessing import validate_required_columns

RFM_INPUT_COLUMNS = ("CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice")


def compute_rfm(
    df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Computes Recency, Frequency, and Monetary per customer.

    - Recency: days between `reference_date` and the customer's most recent
      invoice. `reference_date` is configurable rather than hardcoded — if
      omitted, it defaults to one day after the latest invoice date present
      in `df`, matching the validated offline pipeline's convention.
    - Frequency: number of *unique* invoices for the customer.
    - Monetary: total spend, sum(Quantity * UnitPrice).

    Returns a DataFrame with columns: CustomerID, Recency, Frequency, Monetary.
    """
    validate_required_columns(df, RFM_INPUT_COLUMNS)

    working = df.copy()
    working["InvoiceDate"] = pd.to_datetime(working["InvoiceDate"])
    working["TotalPrice"] = working["Quantity"] * working["UnitPrice"]

    if reference_date is None:
        reference_date = working["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = working.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda dates: (reference_date - dates.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    return rfm


def apply_rfm_transformations(rfm: pd.DataFrame) -> pd.DataFrame:
    """Adds log1p-transformed Frequency/Monetary columns.

    Frequency and Monetary are heavily right-skewed in retail transaction
    data (a small number of customers buy far more often / spend far more
    than the median), which would otherwise dominate a distance-based
    algorithm like KMeans. log1p compresses that skew.

    Recency is intentionally left untouched here — it does not have the
    same extreme-outlier skew problem, and the validated pipeline only
    transforms Frequency/Monetary.

    Returns a new DataFrame with `Frequency_log` and `Monetary_log` columns
    added; `rfm` itself is not mutated.
    """
    validate_required_columns(rfm, ("Frequency", "Monetary"))

    transformed = rfm.copy()
    transformed["Frequency_log"] = np.log1p(transformed["Frequency"])
    transformed["Monetary_log"] = np.log1p(transformed["Monetary"].clip(lower=0))
    return transformed


# ---------------------------------------------------------------------------
# Plain-Python customer features (Phase 3). No pandas/SQLAlchemy needed: the
# service layer aggregates rows in SQL into `CustomerActivity` records and this
# module derives the customer-level features from them.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CustomerActivity:
    """Per-customer aggregates as loaded from the database.

    order_count / total_spent / last_order_at describe COMPLETED orders only
    (the caller is responsible for that filter). views_count / cart_add_count
    count VIEW / CART_ADD interaction events.
    """

    customer_id: int
    order_count: int = 0
    total_spent: float = 0.0
    last_order_at: Optional[datetime] = None
    views_count: int = 0
    cart_add_count: int = 0


@dataclass(frozen=True)
class CustomerFeatures:
    customer_id: int
    recency: float  # days since last completed order (see build_customer_features)
    frequency: int  # completed orders
    monetary: float  # total spent on completed orders
    average_order_value: float  # monetary / frequency, 0.0 without orders
    views_count: int
    cart_add_count: int
    has_purchases: bool


def _clean_count(value) -> int:
    """Missing / NaN / negative counts become 0."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(number) or number < 0:
        return 0
    return int(number)


def _clean_amount(value) -> float:
    """Missing / NaN / infinite / negative money becomes 0.0."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number) or number < 0:
        return 0.0
    return number


def _to_naive_utc(moment: datetime) -> datetime:
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


def build_customer_features(
    activities: Iterable[CustomerActivity], reference_date: datetime
) -> List[CustomerFeatures]:
    """Derive per-customer features. Output is sorted by customer_id.

    - recency: whole days from the customer's last completed order to
      `reference_date` (never negative). Customers WITHOUT a completed order
      have no recency; it is imputed as the largest recency among customers
      that do have one (i.e. "at least as inactive as anyone who bought"), or
      0 when nobody has purchased. Imputation is an explicit, documented
      choice, not a real measurement.
    - frequency: completed-order count. monetary: total spent.
    - average_order_value: monetary / frequency (0.0 when frequency is 0).
    - Missing/NaN/negative counts and amounts are treated as 0.

    Raises ValueError on duplicate customer ids (the input must be one
    aggregate row per customer).
    """
    reference = _to_naive_utc(reference_date)
    ordered = sorted(activities, key=lambda a: a.customer_id)

    seen = set()
    for activity in ordered:
        if activity.customer_id in seen:
            raise ValueError(f"duplicate customer_id in activities: {activity.customer_id}")
        seen.add(activity.customer_id)

    recencies = {}
    for activity in ordered:
        frequency = _clean_count(activity.order_count)
        if activity.last_order_at is not None and frequency > 0:
            recencies[activity.customer_id] = max(
                0, (reference - _to_naive_utc(activity.last_order_at)).days
            )
    imputed_recency = max(recencies.values()) if recencies else 0

    features: List[CustomerFeatures] = []
    for activity in ordered:
        frequency = _clean_count(activity.order_count)
        monetary = _clean_amount(activity.total_spent) if frequency > 0 else 0.0
        has_purchases = frequency > 0
        features.append(
            CustomerFeatures(
                customer_id=activity.customer_id,
                recency=float(recencies.get(activity.customer_id, imputed_recency)),
                frequency=frequency,
                monetary=monetary,
                average_order_value=(monetary / frequency) if frequency > 0 else 0.0,
                views_count=_clean_count(activity.views_count),
                cart_add_count=_clean_count(activity.cart_add_count),
                has_purchases=has_purchases,
            )
        )
    return features
