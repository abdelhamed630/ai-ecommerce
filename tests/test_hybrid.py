"""Unit tests for ml/recommendation/hybrid.py (pure combination logic)."""

import pytest

from ml.recommendation import hybrid as hy
from ml.recommendation.hybrid import HybridWeights, combine, normalize_score

CONTENT = {1: 3.0, 2: 1.5, 3: 0.3}
COLLAB = {2: 3.0, 4: 6.0, 5: 0.6}


# ---- normalization ----------------------------------------------------------

def test_normalize_score_bounds_and_monotonic():
    values = [0.0, 0.1, 1.0, 3.0, 10.0, 1e6]
    normalized = [normalize_score(v) for v in values]
    assert normalized == sorted(normalized)
    assert normalized[0] == 0.0
    assert all(0.0 <= n < 1.0 for n in normalized)
    assert normalize_score(3.0, saturation=3.0) == pytest.approx(0.5)
    assert normalize_score(-5.0) == 0.0


def test_normalize_rejects_bad_saturation():
    with pytest.raises(ValueError):
        normalize_score(1.0, saturation=0)


def test_weights_are_normalized_and_validated():
    w = HybridWeights(content=1, collaborative=3)
    assert (w.content, w.collaborative) == (pytest.approx(0.25), pytest.approx(0.75))
    d = HybridWeights()
    assert (d.content, d.collaborative) == (0.5, 0.5)
    with pytest.raises(ValueError):
        HybridWeights(content=-1, collaborative=1)
    with pytest.raises(ValueError):
        HybridWeights(content=0, collaborative=0)


# ---- 17/18. single-signal modes ---------------------------------------------

def test_content_only():
    result = combine(CONTENT, {})
    assert [c.product_id for c in result] == [1, 2, 3]
    assert all(c.source == hy.SOURCE_CONTENT and c.collaborative_score == 0.0 for c in result)


def test_collaborative_only():
    result = combine({}, COLLAB)
    assert [c.product_id for c in result] == [4, 2, 5]
    assert all(c.source == hy.SOURCE_COLLABORATIVE and c.content_score == 0.0 for c in result)


def test_weight_zero_disables_a_signal():
    only_content = combine(CONTENT, COLLAB, HybridWeights(1, 0))
    assert {c.product_id for c in only_content} == set(CONTENT)
    assert all(c.source == hy.SOURCE_CONTENT for c in only_content)

    only_collab = combine(CONTENT, COLLAB, HybridWeights(0, 1))
    assert {c.product_id for c in only_collab} == set(COLLAB)
    assert all(c.source == hy.SOURCE_COLLABORATIVE for c in only_collab)


def test_empty_inputs():
    assert combine({}, {}) == []
    assert combine({1: 0.0}, {2: 0.0}) == []


# ---- 19/20/21. combining, merging, sorting ----------------------------------

def test_combines_both_candidate_sets():
    result = combine(CONTENT, COLLAB)
    assert {c.product_id for c in result} == set(CONTENT) | set(COLLAB)


def test_duplicates_are_merged_with_weighted_sum():
    result = combine(CONTENT, COLLAB)
    ids = [c.product_id for c in result]
    assert len(ids) == len(set(ids))

    both = next(c for c in result if c.product_id == 2)
    assert both.source == hy.SOURCE_HYBRID
    expected = 0.5 * normalize_score(1.5) + 0.5 * normalize_score(3.0)
    assert both.hybrid_score == pytest.approx(expected)
    assert both.content_score == pytest.approx(normalize_score(1.5))
    assert both.collaborative_score == pytest.approx(normalize_score(3.0))


def test_merged_candidate_beats_single_signal_with_same_strength():
    result = combine({1: 3.0, 2: 3.0}, {2: 3.0})
    assert [c.product_id for c in result] == [2, 1]


def test_sorted_descending_numeric_and_bounded():
    result = combine(CONTENT, COLLAB)
    scores = [c.hybrid_score for c in result]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(s, float) and 0.0 < s < 1.0 for s in scores)


def test_ties_break_by_product_id():
    result = combine({9: 1.0, 3: 1.0, 5: 1.0}, {})
    assert [c.product_id for c in result] == [3, 5, 9]


def test_limit_respected():
    assert len(combine(CONTENT, COLLAB, limit=2)) == 2
    assert combine(CONTENT, COLLAB, limit=2) == combine(CONTENT, COLLAB)[:2]
    assert combine(CONTENT, COLLAB, limit=0) == []
    assert len(combine(CONTENT, COLLAB, limit=100)) == 5


# ---- 22. exclusions ----------------------------------------------------------

def test_excluded_products_never_returned():
    result = combine(CONTENT, COLLAB, exclude_product_ids={2, 4})
    assert {c.product_id for c in result} == {1, 3, 5}


# ---- scale independence -------------------------------------------------------

def test_one_signal_cannot_dominate_by_scale():
    # A huge raw collaborative score is capped by the collaborative weight.
    result = combine({1: 3.0}, {2: 1e9}, HybridWeights(0.5, 0.5))
    by_id = {c.product_id: c for c in result}
    assert by_id[2].hybrid_score < 0.5
    assert by_id[1].hybrid_score == pytest.approx(0.25)


def test_weights_shift_the_ranking():
    content_heavy = combine({1: 3.0}, {2: 3.0}, HybridWeights(0.9, 0.1))
    collab_heavy = combine({1: 3.0}, {2: 3.0}, HybridWeights(0.1, 0.9))
    assert content_heavy[0].product_id == 1
    assert collab_heavy[0].product_id == 2
