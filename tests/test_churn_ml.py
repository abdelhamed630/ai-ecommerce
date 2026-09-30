"""Pure-Python churn ML tests (no DB, no FastAPI)."""

import json
import math
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pytest

from ml.churn import dataset as ds
from ml.churn import features as cf
from ml.churn import model as cm
from ml.churn import predict as cp

CUT = datetime(2026, 6, 1, 12, 0, 0)


def _agg(cid=1, orders=3, spent=90.0, first=90, last=10, views=4, carts=1, inter_last=5, recent_o=1, recent_i=2):
    return cf.ChurnAggregate(
        customer_id=cid, order_count=orders, total_spent=spent,
        first_order_at=CUT - timedelta(days=first), last_order_at=CUT - timedelta(days=last),
        recent_order_count=recent_o, views_count=views, cart_add_count=carts,
        first_interaction_at=CUT - timedelta(days=first - 5),
        last_interaction_at=CUT - timedelta(days=inter_last), recent_interaction_count=recent_i,
    )


def _synthetic_snapshots(n=60, seed=0):
    """Two snapshots; low recency -> retained, high recency -> churned (+ noise)."""
    rng = np.random.RandomState(seed)
    snaps = []
    for s, cutoff in enumerate([CUT - timedelta(days=30), CUT]):
        aggs, buyers = [], set()
        for cid in range(1, n + 1):
            last = int(rng.randint(1, 80))
            orders = int(rng.randint(1, 8))
            aggs.append(
                cf.ChurnAggregate(
                    customer_id=cid, order_count=orders, total_spent=orders * 25.0,
                    first_order_at=cutoff - timedelta(days=last + 60), last_order_at=cutoff - timedelta(days=last),
                    recent_order_count=int(last < 30), views_count=int(rng.randint(0, 9)),
                    cart_add_count=int(rng.randint(0, 3)),
                    first_interaction_at=cutoff - timedelta(days=last + 60),
                    last_interaction_at=cutoff - timedelta(days=last), recent_interaction_count=int(last < 30),
                )
            )
            p_buy = 0.85 if last < 30 else 0.15
            if rng.rand() < p_buy:
                buyers.add(cid)
        snaps.append(ds.Snapshot(cutoff, aggs, buyers))
    return snaps


CONFIG = cm.TrainingConfig(inactivity_days=30, min_samples=40, min_class_samples=5)


# ---- 1. feature extraction
def test_feature_extraction_values():
    row = cf.build_feature_row(_agg(), CUT)
    assert row["recency_days"] == 10.0
    assert row["frequency"] == 3.0 and row["monetary"] == 90.0
    assert row["average_order_value"] == 30.0
    assert row["views_count"] == 4.0 and row["cart_add_count"] == 1.0
    assert row["days_since_last_activity"] == 5.0
    assert row["avg_days_between_orders"] == pytest.approx((90 - 10) / 2)
    assert row["customer_lifetime_days"] == 90.0
    assert row["recent_activity_count"] == 3.0
    assert set(row) == set(cf.FEATURE_NAMES)


def test_customers_without_orders_are_not_eligible():
    assert cf.build_feature_row(cf.ChurnAggregate(customer_id=1, views_count=9), CUT) is None


# ---- 2. historical cutoff
def test_snapshot_cutoffs_end_before_as_of():
    as_of = datetime(2026, 9, 28)
    cutoffs = ds.snapshot_cutoffs(as_of, 60, 3)
    assert cutoffs == sorted(cutoffs)
    assert cutoffs[-1] == as_of - timedelta(days=60)
    assert all(c + timedelta(days=60) <= as_of for c in cutoffs)  # label window never in the future
    assert len(set(cutoffs)) == 3
    with pytest.raises(ValueError):
        ds.snapshot_cutoffs(as_of, 0, 3)


# ---- 3. label generation
def test_label_generation():
    assert ds.label_churn(1, {2, 3}) == 1
    assert ds.label_churn(2, {2, 3}) == 0
    snap = ds.Snapshot(CUT, [_agg(1), _agg(2), cf.ChurnAggregate(customer_id=3)], {2})
    d = ds.build_dataset([snap])
    assert d.customer_ids == [1, 2]  # ineligible customer 3 dropped
    assert d.y.tolist() == [1, 0]


# ---- 4. no target leakage
def test_future_data_in_aggregate_is_rejected():
    future = _agg(last=-3)  # last order 3 days AFTER the cutoff
    with pytest.raises(ValueError, match="leakage"):
        cf.build_feature_row(future, CUT)
    with pytest.raises(ValueError, match="leakage"):
        cf.build_feature_row(_agg(inter_last=-1), CUT)


def test_labels_do_not_influence_features():
    a = ds.build_dataset([ds.Snapshot(CUT, [_agg(1), _agg(2)], set())])
    b = ds.build_dataset([ds.Snapshot(CUT, [_agg(1), _agg(2)], {1, 2})])
    assert np.array_equal(a.X, b.X, equal_nan=True)
    assert a.y.tolist() == [1, 1] and b.y.tolist() == [0, 0]


# ---- 5/6. missing and zero values
def test_missing_values_become_nan_or_zero():
    single = cf.build_feature_row(_agg(orders=1, spent=10, first=10, last=10), CUT)
    assert math.isnan(single["avg_days_between_orders"])
    bare = cf.build_feature_row(
        cf.ChurnAggregate(1, order_count=1, total_spent=None, first_order_at=CUT, last_order_at=CUT,
                          views_count=None, cart_add_count=float("nan")), CUT)
    assert bare["monetary"] == 0.0 and bare["views_count"] == 0.0 and bare["cart_add_count"] == 0.0
    assert bare["days_since_last_activity"] == 0.0


def test_zero_values_are_safe_and_predictable():
    snaps = _synthetic_snapshots()
    result = cm.train_churn_model(ds.build_dataset(snaps), CONFIG)
    zero_row = {name: 0.0 for name in cf.FEATURE_NAMES}
    nan_row = {name: float("nan") for name in cf.FEATURE_NAMES}
    for row in (zero_row, nan_row):
        X = cf.rows_to_matrix([row])
        p = result.pipeline.predict_proba(X)[0, 1]
        assert 0.0 <= p <= 1.0 and math.isfinite(p)


# ---- 7. probability range + training/eval + 14 metrics
def test_training_produces_valid_probabilities_and_metrics():
    dataset = ds.build_dataset(_synthetic_snapshots())
    result = cm.train_churn_model(dataset, CONFIG)
    assert result.split_strategy == "temporal"
    assert result.n_train + result.n_validation == result.n_total == len(dataset)
    proba = result.pipeline.predict_proba(dataset.X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    m = result.metrics
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= m[key] <= 1.0
    assert m["roc_auc"] > 0.7  # planted signal must be learnable
    assert m["decision_threshold"] == 0.5 and "majority_class_accuracy" in m


def test_temporal_split_validation_is_newest_snapshot_only():
    dataset = ds.build_dataset(_synthetic_snapshots())
    train_idx, val_idx, strategy = cm.split_dataset(dataset, CONFIG)
    snaps = np.array(dataset.snapshot_index)
    assert set(snaps[val_idx]) == {1} and set(snaps[train_idx]) == {0}
    assert not set(train_idx) & set(val_idx)


def test_single_snapshot_uses_stratified_split():
    dataset = ds.build_dataset(_synthetic_snapshots()[1:])
    result = cm.train_churn_model(dataset, CONFIG)
    assert result.split_strategy == "stratified_random"


def test_metrics_single_class_validation_reports_undefined_auc():
    dataset = ds.build_dataset(_synthetic_snapshots())
    result = cm.train_churn_model(dataset, CONFIG)
    X = dataset.X[:10]
    m = cm.evaluate(result.pipeline, X, np.ones(10, dtype=int))
    assert m["roc_auc"] is None and "roc_auc_note" in m


def test_training_is_reproducible():
    d = ds.build_dataset(_synthetic_snapshots())
    a = cm.train_churn_model(d, CONFIG)
    b = cm.train_churn_model(d, CONFIG)
    assert np.allclose(a.pipeline.predict_proba(d.X), b.pipeline.predict_proba(d.X))
    assert a.metrics == b.metrics


# ---- 12. small / degenerate datasets fail safely
def test_small_dataset_raises_insufficient_data():
    tiny = ds.build_dataset([ds.Snapshot(CUT, [_agg(i) for i in range(1, 6)], {1})])
    with pytest.raises(cm.InsufficientChurnDataError, match="at least"):
        cm.train_churn_model(tiny, CONFIG)
    with pytest.raises(cm.InsufficientChurnDataError):
        cm.train_churn_model(ds.build_dataset([]), CONFIG)


def test_single_class_dataset_raises():
    aggs = [_agg(i) for i in range(1, 61)]
    one_class = ds.build_dataset([ds.Snapshot(CUT, aggs, set())])  # everyone churned
    with pytest.raises(cm.InsufficientChurnDataError, match="retained"):
        cm.train_churn_model(one_class, CONFIG)


# ---- 8/9/10/13/15 persistence, loading, prediction
def _train_and_save(tmp_path):
    dataset = ds.build_dataset(_synthetic_snapshots())
    result = cm.train_churn_model(dataset, CONFIG)
    path = str(tmp_path / "art" / "churn_model.joblib")
    version = cp.save_artifact(path, result.pipeline, datetime(2026, 6, 1, 8, 0, 0),
                               CONFIG.as_dict(), result.metrics, {"n_total": result.n_total})
    return path, version, result, dataset


def test_artifact_roundtrip_and_contents(tmp_path):
    path, version, result, dataset = _train_and_save(tmp_path)
    assert version == "churn_logreg_v1-20260601080000"
    payload = joblib.load(path)
    assert set(payload) == {"artifact_schema", "model_version", "model_family", "trained_at",
                            "feature_names", "config", "metrics", "info", "pipeline"}
    assert payload["config"] == CONFIG.as_dict()
    art = cp.load_artifact(path)
    assert art.model_version == version and art.feature_names == cf.FEATURE_NAMES
    meta = json.load(open(cp.metadata_path(path)))
    assert meta["model_version"] == version and meta["metrics"]["roc_auc"] == result.metrics["roc_auc"]
    assert "pipeline" not in meta
    assert not [f for f in os.listdir(os.path.dirname(path)) if f.endswith(".tmp")]


def test_loaded_model_predicts_same_as_trained_model(tmp_path):
    path, _, result, dataset = _train_and_save(tmp_path)
    art = cp.load_artifact(path)
    row = dict(zip(cf.FEATURE_NAMES, dataset.X[3]))
    expected = result.pipeline.predict_proba(dataset.X[3:4])[0, 1]
    p = cp.predict_probability(art, row)
    assert p == pytest.approx(expected) and 0.0 <= p <= 1.0


def test_prediction_orders_risk_sensibly(tmp_path):
    path, *_ = _train_and_save(tmp_path)
    art = cp.load_artifact(path)
    base = cf.build_feature_row(_agg(last=2, recent_o=2, recent_i=4), CUT)
    stale = cf.build_feature_row(_agg(last=78, recent_o=0, recent_i=0, inter_last=78), CUT)
    assert cp.predict_probability(art, stale) > cp.predict_probability(art, base)


def test_missing_model_raises_clear_error(tmp_path):
    with pytest.raises(cp.ChurnModelUnavailableError):
        cp.load_artifact(str(tmp_path / "nope.joblib"))
    with pytest.raises(cp.ChurnModelUnavailableError):
        cp.read_metadata(str(tmp_path / "nope.joblib"))


def test_corrupt_and_incompatible_artifacts_rejected(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a model")
    with pytest.raises(cp.ChurnArtifactError):
        cp.load_artifact(str(bad))
    path, *_ = _train_and_save(tmp_path)
    payload = joblib.load(path)
    payload["feature_names"] = payload["feature_names"][:-1] + ["something_else"]
    joblib.dump(payload, path)
    with pytest.raises(cp.ChurnArtifactError, match="feature"):
        cp.load_artifact(path)


def test_feature_consistency_training_vs_inference():
    assert cf.FEATURE_NAMES == tuple(cm.FEATURE_NAMES)
    row = cf.build_feature_row(_agg(), CUT)
    d = ds.build_dataset([ds.Snapshot(CUT, [_agg()], set())])
    assert np.array_equal(cf.rows_to_matrix([row]), d.X, equal_nan=True)
    shuffled = {k: row[k] for k in reversed(list(row))}
    assert np.array_equal(cf.rows_to_matrix([shuffled]), d.X, equal_nan=True)  # order by name, not dict order


# ---- 11. risk thresholds
def test_risk_thresholds_boundaries():
    assert cp.risk_level(0.0, 0.4, 0.7) == "LOW"
    assert cp.risk_level(0.3999, 0.4, 0.7) == "LOW"
    assert cp.risk_level(0.4, 0.4, 0.7) == "MEDIUM"
    assert cp.risk_level(0.6999, 0.4, 0.7) == "MEDIUM"
    assert cp.risk_level(0.7, 0.4, 0.7) == "HIGH"
    assert cp.risk_level(1.0, 0.4, 0.7) == "HIGH"
    assert cp.risk_level(0.5, 0.2, 0.9) == "MEDIUM"  # configurable
    for bad in ((0.7, 0.4), (0.5, 0.5), (-0.1, 0.5), (0.2, 1.5)):
        with pytest.raises(ValueError):
            cp.risk_level(0.5, *bad)
