"""
K-Means clustering pipeline for Customer Segmentation.

Wraps StandardScaler + KMeans in a single sklearn Pipeline so scaling and
clustering are always applied together and consistently, both when training
and when predicting on new data later (see predict.py).
"""

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# K=4 is the currently VALIDATED business configuration (see
# docs/customer_segmentation.md for why K=4 was chosen over the
# higher-silhouette K=2). It is a default, not a hardcoded constraint —
# callers can pass a different `n_clusters` to evaluate alternatives later.
DEFAULT_N_CLUSTERS = 4
DEFAULT_RANDOM_STATE = 42

# Identifies which segmentation algorithm/config produced a persisted
# CustomerSegment row (see models/customer_segment.py). Bump this (e.g. to
# "rfm_kmeans_v2") whenever the algorithm, feature set, or K changes in a
# way that makes old and new segments not directly comparable. This is the
# single source of truth for the value — never hardcode the string
# elsewhere.
SEGMENTATION_MODEL_VERSION = "rfm_kmeans_v1"

# The feature set the model is trained on: raw Recency plus the log1p-
# transformed Frequency/Monetary from features.py.
FEATURE_COLUMNS: Tuple[str, str, str] = ("Recency", "Frequency_log", "Monetary_log")


def build_segmentation_pipeline(
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Builds an (unfit) scaler + KMeans pipeline.

    `n_clusters` and `random_state` are both configurable and deterministic
    — the same inputs always produce the same trained model.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10),
            ),
        ]
    )


def train_segmentation_model(
    rfm_transformed: pd.DataFrame,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[Pipeline, np.ndarray]:
    """Trains the segmentation pipeline on already-transformed RFM data.

    `rfm_transformed` is expected to already have the columns produced by
    `features.apply_rfm_transformations` (i.e. Recency, Frequency_log,
    Monetary_log all present).

    Returns the fitted pipeline and the cluster label assigned to each row
    (in the same order as `rfm_transformed`).
    """
    missing = [c for c in feature_columns if c not in rfm_transformed.columns]
    if missing:
        raise ValueError(f"Missing required feature column(s): {missing}")

    pipeline = build_segmentation_pipeline(n_clusters=n_clusters, random_state=random_state)
    X = rfm_transformed[list(feature_columns)]
    labels = pipeline.fit_predict(X)
    return pipeline, labels


def predict_clusters(
    pipeline: Pipeline,
    rfm_transformed: pd.DataFrame,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> np.ndarray:
    """Predicts cluster ids for already-transformed RFM data using a fitted pipeline."""
    missing = [c for c in feature_columns if c not in rfm_transformed.columns]
    if missing:
        raise ValueError(f"Missing required feature column(s): {missing}")

    X = rfm_transformed[list(feature_columns)]
    return pipeline.predict(X)
