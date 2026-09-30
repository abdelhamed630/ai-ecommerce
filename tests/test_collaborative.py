"""Unit tests for ml/recommendation/collaborative.py (pure ML, no DB/API).

Weights below mirror the project's VIEW=1 < CART_ADD=2 < PURCHASE=3 rule; the
module itself is weight-agnostic and just consumes (user, product, weight).
"""

import numpy as np
import pytest

from ml.recommendation import collaborative as cf

VIEW, CART, BUY = 1.0, 2.0, 3.0

# user 1 (target) bought product 10.
# user 2 bought 10 and 11.      user 3 cart-added 10 and viewed 12.
# user 4 bought 13 only (no overlap with anyone).
ENTRIES = [
    (1, 10, BUY),
    (2, 10, BUY),
    (2, 11, BUY),
    (3, 10, CART),
    (3, 12, VIEW),
    (4, 13, BUY),
]


# ---- 1/2. matrix generation and weights ------------------------------------

def test_matrix_shape_ordering_and_values():
    im = cf.build_interaction_matrix(ENTRIES)
    assert im.user_ids == (1, 2, 3, 4)
    assert im.product_ids == (10, 11, 12, 13)
    dense = im.matrix.toarray()
    assert dense.shape == (4, 4)
    expected = np.array(
        [
            [3, 0, 0, 0],
            [3, 3, 0, 0],
            [2, 0, 1, 0],
            [0, 0, 0, 3],
        ],
        dtype=float,
    )
    assert np.array_equal(dense, expected)


def test_matrix_is_sparse():
    from scipy import sparse

    assert sparse.issparse(cf.build_interaction_matrix(ENTRIES).matrix)


def test_interaction_weights_are_applied_and_duplicates_summed():
    im = cf.build_interaction_matrix([(1, 10, VIEW), (1, 10, VIEW), (1, 10, BUY), (1, 11, CART)])
    row = dict(zip(im.product_ids, im.matrix.toarray()[0]))
    assert row == {10: 5.0, 11: 2.0}


def test_invalid_weights_ignored():
    im = cf.build_interaction_matrix(
        [(1, 10, 0.0), (1, 11, -2.0), (1, 12, float("nan")), (1, 13, float("inf")), (1, 14, 1.0)]
    )
    assert im.product_ids == (14,)


def test_empty_input_gives_empty_matrix():
    im = cf.build_interaction_matrix([])
    assert im.is_empty
    assert cf.find_similar_users(im, 1) == []
    assert cf.score_candidates(im, 1) == {}


# ---- 3. purchase influences more than view ---------------------------------

def _score_of_candidate(candidate_weight):
    entries = [(1, 10, BUY), (2, 10, BUY), (2, 11, candidate_weight)]
    return dict(cf.recommend(entries, 1, 10))[11]


def test_purchase_influences_recommendations_more_than_view():
    assert _score_of_candidate(BUY) > _score_of_candidate(CART) > _score_of_candidate(VIEW)


def test_purchased_candidate_outranks_viewed_candidate():
    entries = [(1, 10, BUY), (2, 10, BUY), (2, 11, BUY), (3, 10, BUY), (3, 12, VIEW)]
    ranked = [pid for pid, _ in cf.recommend(entries, 1, 10)]
    assert ranked.index(11) < ranked.index(12)


# ---- 4. similar users -------------------------------------------------------

def test_similar_users_identified_and_ordered():
    im = cf.build_interaction_matrix(ENTRIES)
    neighbors = cf.find_similar_users(im, 1)
    ids = [uid for uid, _ in neighbors]
    assert ids == [3, 2]  # cos(1,3)=0.894 > cos(1,2)=0.707; user 4 has no overlap; self excluded
    sims = [s for _, s in neighbors]
    assert sims == sorted(sims, reverse=True)
    assert all(0.0 < s <= 1.0 for s in sims)


def test_identical_users_have_similarity_one():
    im = cf.build_interaction_matrix([(1, 10, BUY), (1, 11, VIEW), (2, 10, BUY), (2, 11, VIEW)])
    assert cf.find_similar_users(im, 1) == [(2, pytest.approx(1.0))]


def test_max_neighbors_limits_neighbors():
    im = cf.build_interaction_matrix(ENTRIES)
    assert len(cf.find_similar_users(im, 1, max_neighbors=1)) == 1
    assert cf.find_similar_users(im, 1, max_neighbors=0) == []


# ---- 5/6/13/14/15/16. candidates -------------------------------------------

def test_products_of_similar_users_are_recommended():
    result = dict(cf.recommend(ENTRIES, 1, 10))
    assert set(result) == {11, 12}  # 13 belongs to a non-overlapping user


def test_already_interacted_products_are_excluded():
    ids = [pid for pid, _ in cf.recommend(ENTRIES, 1, 10)]
    assert 10 not in ids


def test_explicit_exclusions_respected():
    ids = [pid for pid, _ in cf.recommend(ENTRIES, 1, 10, exclude_product_ids={11})]
    assert ids == [12]


def test_no_duplicates_limit_numeric_and_sorted():
    entries = ENTRIES + [(5, 10, BUY), (5, 11, VIEW), (5, 14, CART), (6, 10, VIEW), (6, 14, VIEW)]
    full = cf.recommend(entries, 1, 100)
    ids = [pid for pid, _ in full]
    scores = [s for _, s in full]

    assert len(ids) == len(set(ids)) and len(ids) >= 3
    assert all(isinstance(s, float) and s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)

    assert len(cf.recommend(entries, 1, 2)) == 2
    assert cf.recommend(entries, 1, 2) == full[:2]
    assert cf.recommend(entries, 1, 0) == []
    assert cf.recommend(entries, 1, -3) == []


def test_ties_are_deterministic_by_product_id():
    entries = [(1, 10, BUY), (2, 10, BUY), (2, 12, BUY), (2, 11, BUY)]
    assert [pid for pid, _ in cf.recommend(entries, 1, 10)] == [11, 12]


# ---- 7/8. the target user's own data is what is used -----------------------

def test_current_users_interactions_are_used():
    from_user1 = {pid for pid, _ in cf.recommend(ENTRIES, 1, 10)}
    from_user2 = {pid for pid, _ in cf.recommend(ENTRIES, 2, 10)}
    assert from_user1 == {11, 12}
    assert from_user2 == {12}  # user 2 already has 10 and 11; only 3's product remains
    assert from_user1 != from_user2


def test_unknown_user_never_borrows_another_users_history():
    # A target id absent from the data must not be treated as any real user.
    assert cf.recommend(ENTRIES, 999, 10) == []
    im = cf.build_interaction_matrix(ENTRIES)
    assert cf.find_similar_users(im, 999) == []


def test_target_history_is_not_leaked_into_own_candidates():
    # Another user's interactions with the target's own products never make
    # them candidates for the target.
    entries = [(1, 10, BUY), (2, 10, BUY), (2, 11, BUY)]
    assert 10 not in dict(cf.recommend(entries, 1, 10))


# ---- 9-12. small/degenerate data never crashes -----------------------------

def test_user_with_no_interactions():
    assert cf.recommend([(2, 10, BUY), (3, 10, VIEW)], 1, 10) == []


def test_user_with_single_interaction():
    entries = [(1, 10, VIEW), (2, 10, BUY), (2, 11, BUY)]
    assert dict(cf.recommend(entries, 1, 10)).keys() == {11}


def test_single_user_only():
    assert cf.recommend([(1, 10, BUY), (1, 11, BUY)], 1, 10) == []


def test_no_similar_users():
    assert cf.recommend([(1, 10, BUY), (2, 11, BUY), (3, 12, BUY)], 1, 10) == []


def test_empty_interaction_data():
    assert cf.recommend([], 1, 10) == []
    assert cf.recommend(iter(()), 1, 10) == []


def test_all_products_already_seen():
    entries = [(1, 10, BUY), (1, 11, BUY), (2, 10, BUY), (2, 11, BUY)]
    assert cf.recommend(entries, 1, 10) == []
