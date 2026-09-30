"""Pure customer clustering (Phase 3): features -> StandardScaler -> KMeans.

No FastAPI, SQLAlchemy or pandas. Input is a list of
`features.CustomerFeatures`; output is a `SegmentationResult` with one segment
id per customer plus explainable per-segment statistics. The scaler + KMeans
pipeline is the existing `model.build_segmentation_pipeline`.

Preprocessing
-------------
- frequency, monetary, views and cart_adds are right-skewed counts/currency,
  so they are `log1p`-transformed (as in the validated RFM pipeline); recency
  is used as-is.
- All selected features are then standardized (zero mean / unit variance) so
  no single feature dominates purely through its scale. Zero-variance
  features are left unscaled by StandardScaler (no division by zero).

Small / degenerate data (never raises for these)
------------------------------------------------
The effective number of clusters is `min(requested, number of DISTINCT
feature vectors)`: KMeans cannot make more non-empty clusters than there are
distinct points. So 0 customers -> empty result, 1 customer or all-identical
vectors -> one segment, 2 customers -> at most 2 segments. The result reports
both the requested and the effective K.

Segment ids
-----------
KMeans ids are arbitrary, so ids are re-assigned deterministically: segments
are ordered by average monetary (descending), then average recency
(ascending), then size (descending), then the original KMeans label.
Segment 0 is therefore the highest-spending segment of THIS run. Ids carry no
business meaning across runs and no subjective label is attached; interpret a
segment through its statistics.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ml.segmentation.features import CustomerFeatures
from ml.segmentation.model import build_segmentation_pipeline

FEATURE_ACCESSORS = {
    "recency": lambda f: f.recency,
    "frequency": lambda f: f.frequency,
    "monetary": lambda f: f.monetary,
    "views": lambda f: f.views_count,
    "cart_adds": lambda f: f.cart_add_count,
}
ALLOWED_CLUSTER_FEATURES: Tuple[str, ...] = tuple(FEATURE_ACCESSORS)
DEFAULT_CLUSTER_FEATURES: Tuple[str, ...] = ("recency", "frequency", "monetary")
# Skewed features that get log1p before scaling.
LOG_TRANSFORMED = frozenset({"frequency", "monetary", "views", "cart_adds"})

PROFILE_DECIMALS = 4


@dataclass(frozen=True)
class SegmentProfile:
    segment_id: int
    customer_count: int
    average_recency: float
    average_frequency: float
    average_monetary: float
    average_order_value: float
    average_views: float
    average_cart_adds: float


@dataclass(frozen=True)
class SegmentationResult:
    assignments: Dict[int, int]  # customer_id -> segment_id
    profiles: List[SegmentProfile]  # ordered by segment_id
    requested_n_clusters: int
    n_clusters: int  # effective K actually used
    customer_count: int
    features_used: Tuple[str, ...]


def _validate_feature_names(names: Sequence[str]) -> Tuple[str, ...]:
    names = tuple(names)
    if not names:
        raise ValueError("at least one clustering feature is required")
    unknown = [n for n in names if n not in FEATURE_ACCESSORS]
    if unknown:
        raise ValueError(
            f"unknown clustering feature(s) {unknown}; allowed: {list(ALLOWED_CLUSTER_FEATURES)}"
        )
    # Preserve order, drop duplicates.
    return tuple(dict.fromkeys(names))


def build_feature_matrix(
    features: Sequence[CustomerFeatures], feature_names: Sequence[str]
) -> np.ndarray:
    """Raw (untransformed, unscaled) matrix, one row per customer."""
    names = _validate_feature_names(feature_names)
    return np.array(
        [[float(FEATURE_ACCESSORS[n](f)) for n in names] for f in features], dtype=float
    ).reshape(len(features), len(names))


def transform_feature_matrix(raw: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """Apply log1p to the skewed columns (negatives/NaN clipped to 0 first)."""
    names = _validate_feature_names(feature_names)
    out = np.nan_to_num(raw.astype(float), nan=0.0, posinf=0.0, neginf=0.0).copy()
    for col, name in enumerate(names):
        if name in LOG_TRANSFORMED:
            out[:, col] = np.log1p(np.clip(out[:, col], 0.0, None))
    return out


def _profile(segment_id: int, members: Sequence[CustomerFeatures]) -> SegmentProfile:
    def mean(values) -> float:
        return round(float(np.mean(values)), PROFILE_DECIMALS)

    return SegmentProfile(
        segment_id=segment_id,
        customer_count=len(members),
        average_recency=mean([m.recency for m in members]),
        average_frequency=mean([m.frequency for m in members]),
        average_monetary=mean([m.monetary for m in members]),
        average_order_value=mean([m.average_order_value for m in members]),
        average_views=mean([m.views_count for m in members]),
        average_cart_adds=mean([m.cart_add_count for m in members]),
    )


def segment_customers(
    features: Sequence[CustomerFeatures],
    n_clusters: int = 4,
    random_state: int = 42,
    cluster_features: Sequence[str] = DEFAULT_CLUSTER_FEATURES,
) -> SegmentationResult:
    """Cluster customers and return assignments + per-segment statistics."""
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    names = _validate_feature_names(cluster_features)

    features = sorted(features, key=lambda f: f.customer_id)
    ids = [f.customer_id for f in features]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate customer_id in features")

    if not features:
        return SegmentationResult({}, [], n_clusters, 0, 0, names)

    transformed = transform_feature_matrix(build_feature_matrix(features, names), names)
    distinct_vectors = len(np.unique(transformed, axis=0))
    effective_k = min(n_clusters, distinct_vectors)

    if effective_k <= 1:
        raw_labels = np.zeros(len(features), dtype=int)
    else:
        pipeline = build_segmentation_pipeline(
            n_clusters=effective_k, random_state=random_state
        )
        raw_labels = pipeline.fit_predict(transformed)

    members_by_label: Dict[int, List[CustomerFeatures]] = {}
    for feature, label in zip(features, raw_labels):
        members_by_label.setdefault(int(label), []).append(feature)

    def sort_key(label: int):
        members = members_by_label[label]
        return (
            -float(np.mean([m.monetary for m in members])),
            float(np.mean([m.recency for m in members])),
            -len(members),
            label,
        )

    ordered_labels = sorted(members_by_label, key=sort_key)
    new_id = {label: i for i, label in enumerate(ordered_labels)}

    assignments = {
        feature.customer_id: new_id[int(label)] for feature, label in zip(features, raw_labels)
    }
    profiles = [_profile(new_id[label], members_by_label[label]) for label in ordered_labels]

    return SegmentationResult(
        assignments=assignments,
        profiles=profiles,
        requested_n_clusters=n_clusters,
        n_clusters=len(profiles),
        customer_count=len(features),
        features_used=names,
    )
