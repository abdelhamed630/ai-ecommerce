"""Unit tests for ml/recommendation/content_based.py (pure ML, no DB/API)."""

import pytest

from ml.recommendation import content_based as cb

DOCS = {
    1: cb.build_document("Gaming Laptop", "asus rog gaming laptop electronics", "Electronics", "Asus"),
    2: cb.build_document("Gaming Mouse", "asus rog gaming mouse electronics", "Electronics", "Asus"),
    3: cb.build_document("Gaming Keyboard", "asus rog gaming keyboard electronics", "Electronics", "Asus"),
    4: cb.build_document("Kitchen Blender", "powerful kitchen blender appliance", "Kitchen", "Acme"),
}


@pytest.fixture(autouse=True)
def _fresh_cache():
    cb.clear_cache()
    yield
    cb.clear_cache()


def _index(docs=DOCS):
    ids = list(docs)
    return cb.get_or_build_index(ids, [docs[i] for i in ids])


def test_build_document_handles_missing_fields():
    assert cb.build_document(None, None, None, None) == ""
    assert cb.build_document("Name", None, "", None) == "name"
    assert cb.build_document("  Mixed   CASE ", "Desc", "Cat", "Brand") == "mixed case desc cat brand"


def test_similar_items_returned_and_target_excluded():
    result = _index().similar_to(1, 10)
    ids = [pid for pid, _ in result]
    assert 1 not in ids
    assert set(ids) == {2, 3, 4}


def test_sorted_descending_numeric_and_bounded():
    scores = [s for _, s in _index().similar_to(1, 10)]
    assert all(isinstance(s, float) and 0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_similar_content_ranks_above_unrelated():
    result = dict(_index().similar_to(1, 10))
    assert result[2] > result[4]
    assert result[3] > result[4]
    assert result[4] == 0.0  # no shared vocabulary


def test_limit_works():
    assert len(_index().similar_to(1, 1)) == 1
    assert len(_index().similar_to(1, 2)) == 2
    assert len(_index().similar_to(1, 100)) == 3  # larger than catalog
    assert _index().similar_to(1, 0) == []


def test_no_duplicate_recommendations():
    ids = [pid for pid, _ in _index().similar_to(1, 100)]
    assert len(ids) == len(set(ids))


def test_unknown_product_id_returns_empty():
    assert _index().similar_to(999, 5) == []


def test_identical_documents_score_one_and_ties_keep_id_order():
    docs = {10: "red cotton shirt", 11: "red cotton shirt", 12: "red cotton shirt", 13: "steel hammer"}
    result = _index(docs).similar_to(10, 10)
    assert [pid for pid, _ in result[:2]] == [11, 12]
    assert result[0][1] == pytest.approx(1.0)


def test_empty_and_missing_text_products_do_not_crash():
    docs = {1: "", 2: "blue denim jeans", 3: "blue denim jacket", 4: ""}
    idx = _index(docs)
    assert idx is not None
    assert dict(idx.similar_to(2, 10))[3] > 0
    assert all(s == 0.0 for _, s in idx.similar_to(1, 10))  # empty target


def test_all_empty_catalog_returns_none():
    assert _index({1: "", 2: "", 3: "   "}) is None


def test_stop_word_only_catalog_still_works():
    docs = {1: "the one", 2: "the one", 3: "the other"}
    idx = _index(docs)
    assert idx is not None
    assert idx.similar_to(1, 5)[0][0] == 2


def test_two_product_catalog_works():
    idx = _index({1: "wooden chair", 2: "wooden table"})
    assert [pid for pid, _ in idx.similar_to(1, 5)] == [2]


def test_index_is_reused_until_catalog_text_changes():
    first = _index()
    assert _index() is first  # same snapshot -> cached, not refitted

    changed = dict(DOCS)
    changed[4] = cb.build_document("Kitchen Blender", "now with gaming laptop vibes", "Kitchen", "Acme")
    second = _index(changed)
    assert second is not first
    assert dict(second.similar_to(1, 10))[4] > 0


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        cb.get_or_build_index([1, 2], ["only one"])


def test_accents_and_non_latin_text_supported():
    docs = {1: "قهوة عربية فاخرة", 2: "قهوة عربية محمصة", 3: "Café crème", 4: "hammer"}
    idx = _index(docs)
    assert idx.similar_to(1, 3)[0][0] == 2


# --- ContentIndex.score_profile (personalized content scoring, Phase 2) ---

def test_score_profile_matches_weighted_sum_of_pairwise_similarities():
    idx = _index()
    interest = {1: 3.0, 4: 1.0}
    scores = idx.score_profile(interest)
    for candidate in (2, 3):
        expected = sum(w * dict(idx.similar_to(pid, 10))[candidate] for pid, w in interest.items())
        assert scores[candidate] == pytest.approx(expected)


def test_score_profile_excludes_interacted_and_scales_with_weight():
    idx = _index()
    low = idx.score_profile({1: 1.0})
    high = idx.score_profile({1: 3.0})
    assert 1 not in low and 1 not in high
    assert high[2] == pytest.approx(3 * low[2])
    assert 4 not in low  # zero similarity -> not a candidate


def test_score_profile_handles_empty_and_unknown():
    idx = _index()
    assert idx.score_profile({}) == {}
    assert idx.score_profile({999: 3.0}) == {}
    assert idx.score_profile({1: 0.0}) == {}
