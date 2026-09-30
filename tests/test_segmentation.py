import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.segmentation import features, interpretation, model, predict, preprocessing


# ---------------------------------------------------------------------------
# 1. Dataset cleaning
# ---------------------------------------------------------------------------

def _raw_transactions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "InvoiceNo": ["1001", "1002", "C1003", "1004", "1005", "1006"],
            "CustomerID": [1, None, 3, 4, 5, 6],
            "InvoiceDate": pd.to_datetime(
                ["2024-01-01"] * 6
            ),
            "Quantity": [2, 3, 1, -1, 4, 5],
            "UnitPrice": [10.0, 5.0, 20.0, 15.0, -3.0, 8.0],
        }
    )


def test_cleaning_removes_missing_customer_id():
    df = _raw_transactions_df()
    cleaned = preprocessing.clean_transactions(df)
    assert cleaned["CustomerID"].isnull().sum() == 0
    assert 2 not in cleaned["CustomerID"].tolist()  # row with None CustomerID is gone


def test_cleaning_removes_cancelled_invoices():
    df = _raw_transactions_df()
    cleaned = preprocessing.clean_transactions(df)
    assert not cleaned["InvoiceNo"].astype(str).str.startswith("C").any()


def test_cleaning_removes_invalid_quantity():
    df = _raw_transactions_df()
    cleaned = preprocessing.clean_transactions(df)
    assert (cleaned["Quantity"] > 0).all()
    assert 4 not in cleaned["CustomerID"].tolist()  # the negative-quantity row's customer


def test_cleaning_removes_invalid_price():
    df = _raw_transactions_df()
    cleaned = preprocessing.clean_transactions(df)
    assert (cleaned["UnitPrice"] > 0).all()
    assert 5 not in cleaned["CustomerID"].tolist()  # the negative-price row's customer


def test_cleaning_missing_column_raises():
    df = _raw_transactions_df().drop(columns=["UnitPrice"])
    with pytest.raises(ValueError):
        preprocessing.clean_transactions(df)


# ---------------------------------------------------------------------------
# 2. RFM calculation
# ---------------------------------------------------------------------------

def _clean_transactions_for_rfm() -> pd.DataFrame:
    # Customer 1: two invoices, most recent 2024-01-10
    # Customer 2: one invoice, most recent 2024-01-05
    return pd.DataFrame(
        {
            "InvoiceNo": ["A1", "A2", "B1"],
            "CustomerID": [1, 1, 2],
            "InvoiceDate": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-05"]),
            "Quantity": [2, 3, 4],
            "UnitPrice": [10.0, 5.0, 8.0],
        }
    )


def test_recency_calculated_correctly():
    df = _clean_transactions_for_rfm()
    reference_date = pd.Timestamp("2024-01-15")
    rfm = features.compute_rfm(df, reference_date=reference_date)

    customer_1 = rfm[rfm["CustomerID"] == 1].iloc[0]
    customer_2 = rfm[rfm["CustomerID"] == 2].iloc[0]

    assert customer_1["Recency"] == 5   # 2024-01-15 - 2024-01-10
    assert customer_2["Recency"] == 10  # 2024-01-15 - 2024-01-05


def test_frequency_counts_unique_invoices():
    df = _clean_transactions_for_rfm()
    rfm = features.compute_rfm(df, reference_date=pd.Timestamp("2024-01-15"))

    customer_1 = rfm[rfm["CustomerID"] == 1].iloc[0]
    customer_2 = rfm[rfm["CustomerID"] == 2].iloc[0]

    assert customer_1["Frequency"] == 2  # invoices A1, A2
    assert customer_2["Frequency"] == 1  # invoice B1


def test_monetary_calculated_correctly():
    df = _clean_transactions_for_rfm()
    rfm = features.compute_rfm(df, reference_date=pd.Timestamp("2024-01-15"))

    customer_1 = rfm[rfm["CustomerID"] == 1].iloc[0]
    customer_2 = rfm[rfm["CustomerID"] == 2].iloc[0]

    assert customer_1["Monetary"] == pytest.approx(2 * 10.0 + 3 * 5.0)  # 35.0
    assert customer_2["Monetary"] == pytest.approx(4 * 8.0)  # 32.0


def test_reference_date_defaults_to_day_after_latest_invoice():
    df = _clean_transactions_for_rfm()
    rfm = features.compute_rfm(df)  # no reference_date passed
    customer_1 = rfm[rfm["CustomerID"] == 1].iloc[0]
    # latest invoice overall is 2024-01-10 -> default reference date 2024-01-11
    assert customer_1["Recency"] == 1


# ---------------------------------------------------------------------------
# 3. Transformation
# ---------------------------------------------------------------------------

def test_log1p_applied_to_frequency_and_monetary_only():
    rfm = pd.DataFrame(
        {"CustomerID": [1, 2], "Recency": [5, 100], "Frequency": [3, 1], "Monetary": [150.0, 20.0]}
    )
    transformed = features.apply_rfm_transformations(rfm)

    assert transformed["Frequency_log"].tolist() == pytest.approx(np.log1p([3, 1]).tolist())
    assert transformed["Monetary_log"].tolist() == pytest.approx(np.log1p([150.0, 20.0]).tolist())
    # Recency must remain untouched by the log transform
    assert transformed["Recency"].tolist() == [5, 100]


def test_transformation_does_not_mutate_input():
    rfm = pd.DataFrame(
        {"CustomerID": [1], "Recency": [5], "Frequency": [3], "Monetary": [150.0]}
    )
    features.apply_rfm_transformations(rfm)
    assert "Frequency_log" not in rfm.columns  # original untouched


# ---------------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------------

def _synthetic_rfm_transformed(n_per_group: int = 15) -> pd.DataFrame:
    """Four well-separated synthetic groups so KMeans has real structure to find."""
    rng = np.random.default_rng(0)
    groups = [
        {"Recency": 5, "Frequency": 20, "Monetary": 5000},   # champions
        {"Recency": 60, "Frequency": 5, "Monetary": 1000},   # potential loyalists
        {"Recency": 15, "Frequency": 2, "Monetary": 300},    # low engagement
        {"Recency": 200, "Frequency": 1, "Monetary": 100},   # at risk
    ]
    rows = []
    for customer_id, group in enumerate(groups):
        for _ in range(n_per_group):
            rows.append(
                {
                    "CustomerID": len(rows) + 1,
                    "Recency": max(1, group["Recency"] + rng.integers(-2, 3)),
                    "Frequency": max(1, group["Frequency"] + rng.integers(-1, 2)),
                    "Monetary": max(1.0, group["Monetary"] + rng.normal(0, 50)),
                }
            )
    rfm = pd.DataFrame(rows)
    return features.apply_rfm_transformations(rfm)


def test_kmeans_trains_successfully():
    rfm_transformed = _synthetic_rfm_transformed()
    pipeline, labels = model.train_segmentation_model(rfm_transformed)
    assert len(labels) == len(rfm_transformed)
    assert set(labels).issubset({0, 1, 2, 3})


def test_model_uses_n_clusters_4_by_default():
    rfm_transformed = _synthetic_rfm_transformed()
    pipeline, _ = model.train_segmentation_model(rfm_transformed)
    assert pipeline.named_steps["kmeans"].n_clusters == 4


def test_predictions_deterministic_with_fixed_random_state():
    rfm_transformed = _synthetic_rfm_transformed()
    _, labels_1 = model.train_segmentation_model(rfm_transformed, random_state=42)
    _, labels_2 = model.train_segmentation_model(rfm_transformed, random_state=42)
    assert list(labels_1) == list(labels_2)


def test_model_missing_feature_column_raises():
    rfm_transformed = _synthetic_rfm_transformed().drop(columns=["Monetary_log"])
    with pytest.raises(ValueError):
        model.train_segmentation_model(rfm_transformed)


# ---------------------------------------------------------------------------
# 5. Interpretation
# ---------------------------------------------------------------------------

def test_labels_derived_from_profile_not_from_cluster_id():
    # Deliberately non-sequential, "shuffled" cluster ids to prove the labeling
    # logic never assumes e.g. "cluster 1 = Champions".
    rfm_with_clusters = pd.DataFrame(
        {
            "CustomerID": range(8),
            "Recency": [3, 4, 200, 190, 60, 65, 20, 18],
            "Frequency": [20, 18, 1, 1, 5, 4, 2, 2],
            "Monetary": [5000, 4800, 100, 90, 1200, 1100, 300, 280],
            "Cluster": [7, 7, 2, 2, 9, 9, 0, 0],  # arbitrary non-0-indexed ids
        }
    )
    label_map = interpretation.get_segment_label_map(rfm_with_clusters)

    assert label_map[7] == "Champions / Loyal"
    assert label_map[2] == "At Risk / Churned"
    # every cluster id that actually occurs must have a label
    assert set(label_map.keys()) == {7, 2, 9, 0}


def test_interpretation_same_profiles_different_ids_give_same_labels():
    base = pd.DataFrame(
        {
            "CustomerID": range(8),
            "Recency": [3, 4, 200, 190, 60, 65, 20, 18],
            "Frequency": [20, 18, 1, 1, 5, 4, 2, 2],
            "Monetary": [5000, 4800, 100, 90, 1200, 1100, 300, 280],
        }
    )
    version_a = base.copy()
    version_a["Cluster"] = [1, 1, 2, 2, 3, 3, 0, 0]
    version_b = base.copy()
    version_b["Cluster"] = [9, 9, 5, 5, 1, 1, 2, 2]  # same groups, different ids

    labels_a = set(interpretation.get_segment_label_map(version_a).values())
    labels_b = set(interpretation.get_segment_label_map(version_b).values())
    assert labels_a == labels_b


def test_build_cluster_profiles_missing_column_raises():
    df = pd.DataFrame({"CustomerID": [1], "Recency": [5], "Cluster": [0]})
    with pytest.raises(ValueError):
        interpretation.build_cluster_profiles(df)


# ---------------------------------------------------------------------------
# 6. Prediction
# ---------------------------------------------------------------------------

def test_trained_model_can_predict_on_valid_rfm_input():
    rfm_transformed = _synthetic_rfm_transformed()
    pipeline, labels = model.train_segmentation_model(rfm_transformed)

    rfm_with_clusters = rfm_transformed.copy()
    rfm_with_clusters["Cluster"] = labels
    label_map = interpretation.get_segment_label_map(rfm_with_clusters)

    new_customer = pd.DataFrame(
        {"CustomerID": [999], "Recency": [4], "Frequency": [19], "Monetary": [4900.0]}
    )
    new_transformed = features.apply_rfm_transformations(new_customer)

    result = predict.predict_segment(new_transformed, pipeline, label_map)
    assert len(result) == 1
    assert result.iloc[0]["CustomerID"] == 999
    assert result.iloc[0]["SegmentLabel"] in label_map.values()


def test_save_and_load_artifacts_round_trip():
    rfm_transformed = _synthetic_rfm_transformed()
    pipeline, labels = model.train_segmentation_model(rfm_transformed)

    rfm_with_clusters = rfm_transformed.copy()
    rfm_with_clusters["Cluster"] = labels
    label_map = interpretation.get_segment_label_map(rfm_with_clusters)

    tmp_dir = tempfile.mkdtemp()
    try:
        predict.save_artifacts(pipeline, label_map, artifacts_dir=tmp_dir)
        loaded_pipeline, loaded_label_map, loaded_feature_columns = predict.load_artifacts(
            artifacts_dir=tmp_dir
        )

        assert loaded_feature_columns == list(model.FEATURE_COLUMNS)
        assert loaded_label_map == label_map

        new_customer = pd.DataFrame(
            {"CustomerID": [1000], "Recency": [3], "Frequency": [21], "Monetary": [5100.0]}
        )
        new_transformed = features.apply_rfm_transformations(new_customer)
        original_result = predict.predict_segment(new_transformed, pipeline, label_map)
        loaded_result = predict.predict_segment(
            new_transformed, loaded_pipeline, loaded_label_map
        )
        assert original_result["Cluster"].tolist() == loaded_result["Cluster"].tolist()
        assert original_result["SegmentLabel"].tolist() == loaded_result["SegmentLabel"].tolist()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_load_artifacts_missing_raises():
    empty_dir = tempfile.mkdtemp()
    try:
        with pytest.raises(FileNotFoundError):
            predict.load_artifacts(artifacts_dir=empty_dir)
    finally:
        shutil.rmtree(empty_dir, ignore_errors=True)
