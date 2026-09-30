"""
Interpretation layer for Customer Segmentation.

K-Means cluster IDs (0, 1, 2, 3, ...) are arbitrary and can change every
time the model is retrained — cluster "1" today might be "Champions" but
could be "At Risk" after the next training run on different data. This
module NEVER assumes a fixed mapping like "cluster 1 = Champions". Instead,
it always derives a human-readable label from each cluster's *actual*
Recency/Frequency/Monetary profile, recomputed fresh every time.

Labeling rules (documented explicitly, not hidden in a lookup table):

  - High spend (Monetary >= median) AND recent (Recency <= median)
        -> "Champions / Loyal"
  - Very inactive (Recency > 1.5x median) AND infrequent (Frequency <= median)
        -> "At Risk / Churned"
  - Frequent AND high spend (both >= median), but not recent enough to be
    a Champion
        -> "Potential Loyalist"
  - Everything else (recent-ish but low frequency/spend)
        -> "Low Engagement"

These thresholds are intentionally simple and transparent (median-relative,
not machine-learned) so the reasoning behind any label is always
inspectable. They mirror the validated offline pipeline's heuristic.
"""

from typing import Dict

import pandas as pd

RFM_COLUMNS = ("Recency", "Frequency", "Monetary")


def build_cluster_profiles(
    rfm_with_clusters: pd.DataFrame, cluster_col: str = "Cluster"
) -> pd.DataFrame:
    """Aggregates mean Recency/Frequency/Monetary and size per cluster.

    `rfm_with_clusters` must contain the raw (untransformed) Recency,
    Frequency, and Monetary columns plus a cluster-id column — profiles are
    computed on the original business units (days, counts, currency), not
    the log-transformed/scaled features used internally by the model.
    """
    missing = [c for c in (*RFM_COLUMNS, cluster_col) if c not in rfm_with_clusters.columns]
    if missing:
        raise ValueError(f"Missing required column(s) for profiling: {missing}")

    profile = rfm_with_clusters.groupby(cluster_col)[list(RFM_COLUMNS)].mean()
    profile["Count"] = rfm_with_clusters[cluster_col].value_counts()
    return profile


def interpret_cluster(profile_row: pd.Series, medians: pd.Series) -> str:
    """Derives a single human-readable label from one cluster's RFM profile.

    `medians` is the per-metric median taken ACROSS cluster profiles (not
    across individual customers) — see `label_cluster_profiles` — so the
    comparison is always relative to the other clusters actually found in
    this training run, never a fixed number.
    """
    if profile_row["Recency"] <= medians["Recency"] and profile_row["Monetary"] >= medians["Monetary"]:
        return "Champions / Loyal"
    if profile_row["Recency"] > medians["Recency"] * 1.5 and profile_row["Frequency"] <= medians["Frequency"]:
        return "At Risk / Churned"
    if profile_row["Frequency"] >= medians["Frequency"] and profile_row["Monetary"] >= medians["Monetary"]:
        return "Potential Loyalist"
    return "Low Engagement"


def label_cluster_profiles(profile: pd.DataFrame) -> pd.DataFrame:
    """Adds a `Label` column to a cluster profile DataFrame (from build_cluster_profiles)."""
    medians = profile[list(RFM_COLUMNS)].median()
    labeled = profile.copy()
    labeled["Label"] = labeled.apply(lambda row: interpret_cluster(row, medians), axis=1)
    return labeled


def get_segment_label_map(
    rfm_with_clusters: pd.DataFrame, cluster_col: str = "Cluster"
) -> Dict[int, str]:
    """Builds a fresh {cluster_id: label} map from the current training run's data.

    This is the ONLY place a cluster id is ever associated with a label —
    always computed from actual cluster behavior, never hardcoded. Callers
    (e.g. predict.py) should always go through this map rather than
    assuming what a given cluster id means.
    """
    profile = build_cluster_profiles(rfm_with_clusters, cluster_col=cluster_col)
    labeled = label_cluster_profiles(profile)
    return labeled["Label"].to_dict()
