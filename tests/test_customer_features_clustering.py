"""Pure-Python tests for Phase 3 customer features + clustering.

No database, no FastAPI: plain dataclasses in, plain results out.
"""

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from ml.segmentation.clustering import (
    ALLOWED_CLUSTER_FEATURES,
    build_feature_matrix,
    segment_customers,
    transform_feature_matrix,
)
from ml.segmentation.features import CustomerActivity, build_customer_features

REF = datetime(2026, 9, 28, 12, 0, 0)


def _act(cid, orders=0, spent=0.0, days_ago=None, views=0, carts=0):
    last = REF - timedelta(days=days_ago) if days_ago is not None else None
    return CustomerActivity(cid, orders, spent, last, views, carts)


def _feats(activities):
    return build_customer_features(activities, REF)


def _spread_customers(n_per_group=3):
    """Three clearly separated behaviours: big/recent, mid, small/old."""
    acts, cid = [], 1
    for orders, spent, days in [(20, 5000.0, 2), (5, 600.0, 60), (1, 20.0, 300)]:
        for i in range(n_per_group):
            acts.append(_act(cid, orders + i % 2, spent + i, days + i))
            cid += 1
    return _feats(acts)


# ---------------- feature calculation (1-9) ----------------

def test_recency_is_whole_days_since_last_order():
    (f,) = _feats([_act(1, 1, 10.0, days_ago=7)])
    assert f.recency == 7.0


def test_recency_never_negative_for_future_order():
    (f,) = _feats([CustomerActivity(1, 1, 10.0, REF + timedelta(days=3))])
    assert f.recency == 0.0


def test_recency_handles_timezone_aware_dates():
    aware = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
    (f,) = build_customer_features([CustomerActivity(1, 1, 5.0, aware)], REF)
    assert f.recency == 7.0


def test_frequency_and_monetary_and_aov():
    (f,) = _feats([_act(1, orders=4, spent=200.0, days_ago=1)])
    assert f.frequency == 4
    assert f.monetary == 200.0
    assert f.average_order_value == 50.0
    assert f.has_purchases is True


def test_multiple_customers_sorted_by_id():
    feats = _feats([_act(3, 1, 1.0, 1), _act(1, 2, 2.0, 2), _act(2, 3, 3.0, 3)])
    assert [f.customer_id for f in feats] == [1, 2, 3]
    assert [f.frequency for f in feats] == [2, 3, 1]


def test_missing_and_invalid_values_become_zero():
    bad = CustomerActivity(1, None, float("nan"), None, -5, None)
    (f,) = _feats([bad])
    assert (f.frequency, f.monetary, f.average_order_value) == (0, 0.0, 0.0)
    assert (f.views_count, f.cart_add_count) == (0, 0)
    assert f.has_purchases is False
    assert math.isfinite(f.recency)


def test_zero_monetary_with_orders_is_safe():
    (f,) = _feats([_act(1, orders=2, spent=0.0, days_ago=5)])
    assert f.monetary == 0.0 and f.average_order_value == 0.0


def test_customers_without_purchases_get_imputed_recency():
    feats = _feats([_act(1, 1, 10.0, days_ago=30), _act(2, 1, 10.0, days_ago=90), _act(3, views=4)])
    no_purchase = feats[2]
    assert no_purchase.has_purchases is False
    assert no_purchase.recency == 90.0  # largest observed recency
    assert no_purchase.views_count == 4


def test_nobody_purchased_recency_is_zero():
    (f,) = _feats([_act(1, views=2)])
    assert f.recency == 0.0


def test_duplicate_customer_ids_rejected():
    with pytest.raises(ValueError):
        _feats([_act(1, 1, 1.0, 1), _act(1, 1, 1.0, 1)])


# ---------------- scaling / transforms (14) ----------------

def test_log_transform_applies_only_to_skewed_columns():
    raw = np.array([[10.0, 3.0, 100.0]])
    out = transform_feature_matrix(raw, ("recency", "frequency", "monetary"))
    assert out[0, 0] == 10.0
    assert out[0, 1] == pytest.approx(math.log1p(3))
    assert out[0, 2] == pytest.approx(math.log1p(100))


def test_transform_is_safe_for_zero_negative_and_nan():
    raw = np.array([[0.0, 0.0, -5.0], [1.0, np.nan, 0.0]])
    out = transform_feature_matrix(raw, ("recency", "frequency", "monetary"))
    assert np.isfinite(out).all()
    assert out[0, 2] == 0.0


def test_scaling_prevents_large_feature_from_dominating():
    """Monetary spans 6 orders of magnitude; standardized clustering must still
    separate customers by recency when recency is the only real signal."""
    acts = []
    for i in range(6):
        acts.append(_act(i + 1, 1, 100.0, days_ago=2 + i))  # recent
    for i in range(6):
        acts.append(_act(i + 7, 1, 100.0, days_ago=300 + i))  # old
    r = segment_customers(_feats(acts), n_clusters=2)
    recent = {r.assignments[i] for i in range(1, 7)}
    old = {r.assignments[i] for i in range(7, 13)}
    assert len(recent) == 1 and len(old) == 1 and recent != old


def test_unknown_feature_and_bad_k_rejected():
    with pytest.raises(ValueError):
        build_feature_matrix(_spread_customers(), ("recency", "nope"))
    with pytest.raises(ValueError):
        segment_customers(_spread_customers(), n_clusters=0)
    with pytest.raises(ValueError):
        segment_customers(_spread_customers(), cluster_features=())


# ---------------- clustering (10-13, 15-18) ----------------

def test_empty_dataset():
    r = segment_customers([], n_clusters=4)
    assert r.assignments == {} and r.profiles == [] and r.n_clusters == 0 and r.customer_count == 0


def test_one_customer():
    r = segment_customers(_feats([_act(1, 2, 50.0, 3)]), n_clusters=4)
    assert r.n_clusters == 1 and r.requested_n_clusters == 4
    assert r.assignments == {1: 0}
    assert r.profiles[0].customer_count == 1


def test_two_customers_two_clusters_when_different():
    r = segment_customers(_feats([_act(1, 10, 900.0, 1), _act(2, 1, 5.0, 200)]), n_clusters=4)
    assert r.n_clusters == 2
    assert set(r.assignments.values()) == {0, 1}


def test_dataset_smaller_than_requested_k():
    r = segment_customers(_feats([_act(i, i, 10.0 * i, i) for i in range(1, 4)]), n_clusters=8)
    assert r.n_clusters == 3
    assert sum(p.customer_count for p in r.profiles) == 3


def test_identical_feature_vectors_single_segment_no_crash():
    acts = [_act(i, 2, 100.0, 10) for i in range(1, 7)]
    r = segment_customers(_feats(acts), n_clusters=4)
    assert r.n_clusters == 1
    assert set(r.assignments.values()) == {0}
    assert r.profiles[0].customer_count == 6


def test_kmeans_output_shape_and_segment_counts():
    feats = _spread_customers()
    r = segment_customers(feats, n_clusters=3)
    assert r.n_clusters == 3
    assert len(r.assignments) == len(feats) == r.customer_count
    assert [p.segment_id for p in r.profiles] == [0, 1, 2]
    assert sum(p.customer_count for p in r.profiles) == len(feats)
    counts = {}
    for sid in r.assignments.values():
        counts[sid] = counts.get(sid, 0) + 1
    assert counts == {p.segment_id: p.customer_count for p in r.profiles}
    # the three planted groups are recovered
    assert all(p.customer_count == 3 for p in r.profiles)


def test_segment_statistics_match_manual_means():
    feats = _spread_customers()
    r = segment_customers(feats, n_clusters=3)
    by_id = {f.customer_id: f for f in feats}
    for p in r.profiles:
        members = [by_id[c] for c, s in r.assignments.items() if s == p.segment_id]
        assert p.average_recency == pytest.approx(np.mean([m.recency for m in members]), abs=1e-3)
        assert p.average_frequency == pytest.approx(np.mean([m.frequency for m in members]), abs=1e-3)
        assert p.average_monetary == pytest.approx(np.mean([m.monetary for m in members]), abs=1e-3)
        assert p.average_order_value == pytest.approx(
            np.mean([m.average_order_value for m in members]), abs=1e-3
        )
        assert p.average_views == pytest.approx(np.mean([m.views_count for m in members]), abs=1e-3)
        assert p.average_cart_adds == pytest.approx(
            np.mean([m.cart_add_count for m in members]), abs=1e-3
        )


def test_segment_ids_ordered_by_average_monetary_desc():
    r = segment_customers(_spread_customers(), n_clusters=3)
    monetary = [p.average_monetary for p in r.profiles]
    assert monetary == sorted(monetary, reverse=True)


def test_deterministic_and_input_order_independent():
    feats = _spread_customers()
    a = segment_customers(feats, n_clusters=3, random_state=7)
    b = segment_customers(list(reversed(feats)), n_clusters=3, random_state=7)
    assert a.assignments == b.assignments
    assert a.profiles == b.profiles


def test_optional_behavioural_features_can_be_clustered_on():
    acts = [_act(i, 1, 50.0, 10, views=v) for i, v in enumerate([1, 2, 1, 200, 220, 210], start=1)]
    r = segment_customers(_feats(acts), n_clusters=2, cluster_features=("views",))
    assert r.features_used == ("views",)
    low = {r.assignments[i] for i in (1, 2, 3)}
    high = {r.assignments[i] for i in (4, 5, 6)}
    assert len(low) == 1 and len(high) == 1 and low != high


def test_allowed_features_documented():
    assert set(ALLOWED_CLUSTER_FEATURES) == {"recency", "frequency", "monetary", "views", "cart_adds"}
