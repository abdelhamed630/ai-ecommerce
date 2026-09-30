"""Historical snapshots -> labelled churn dataset. Pure Python.

Churn definition
----------------
For a snapshot with cutoff T and window W = CHURN_INACTIVITY_DAYS:

  population: customers with >= 1 completed order at or before T
  features  : computed from events with timestamp <= T ONLY
  label     : churned = 1 if the customer has NO completed order in (T, T + W],
              else 0

Snapshots are spaced W days apart, newest first: cutoffs are
`as_of - W`, `as_of - 2W`, ... so that the label window of every snapshot
ends at or before `as_of` (no label ever looks into the future).
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Sequence, Set

import numpy as np

from ml.churn.features import (
    FEATURE_NAMES,
    ChurnAggregate,
    build_feature_row,
    rows_to_matrix,
    to_naive_utc,
)


@dataclass(frozen=True)
class Snapshot:
    cutoff: datetime
    aggregates: Sequence[ChurnAggregate]  # built from events <= cutoff only
    # Customers with a completed order in (cutoff, cutoff + window]. Used ONLY
    # to derive labels; never a feature source.
    purchasers_in_window: Set[int]


@dataclass
class ChurnDataset:
    X: np.ndarray
    y: np.ndarray
    customer_ids: List[int] = field(default_factory=list)
    snapshot_index: List[int] = field(default_factory=list)  # 0 = oldest cutoff
    cutoffs: List[datetime] = field(default_factory=list)
    feature_names: tuple = FEATURE_NAMES

    def __len__(self) -> int:
        return len(self.y)


def snapshot_cutoffs(as_of: datetime, window_days: int, count: int) -> List[datetime]:
    """Cutoffs oldest-first. The newest is `as_of - window_days`."""
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    if count < 1:
        raise ValueError("count must be >= 1")
    as_of = to_naive_utc(as_of)
    return [as_of - timedelta(days=window_days * k) for k in range(count, 0, -1)]


def label_churn(customer_id: int, purchasers_in_window: Set[int]) -> int:
    return 0 if customer_id in purchasers_in_window else 1


def build_dataset(snapshots: Sequence[Snapshot]) -> ChurnDataset:
    """Build X / y from snapshots (ordered oldest -> newest)."""
    rows: List[dict] = []
    labels: List[int] = []
    ids: List[int] = []
    snap_idx: List[int] = []
    for index, snapshot in enumerate(snapshots):
        cutoff = to_naive_utc(snapshot.cutoff)
        for aggregate in sorted(snapshot.aggregates, key=lambda a: a.customer_id):
            row = build_feature_row(aggregate, cutoff)
            if row is None:
                continue
            rows.append(row)
            labels.append(label_churn(aggregate.customer_id, snapshot.purchasers_in_window))
            ids.append(aggregate.customer_id)
            snap_idx.append(index)
    return ChurnDataset(
        X=rows_to_matrix(rows),
        y=np.array(labels, dtype=int),
        customer_ids=ids,
        snapshot_index=snap_idx,
        cutoffs=[to_naive_utc(s.cutoff) for s in snapshots],
    )
