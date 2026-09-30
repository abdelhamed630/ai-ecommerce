"""Collaborative + Hybrid recommendations through the service layer and API.

Every test gets its OWN empty SQLite database (tmp_path), so the small,
fully-controlled catalogs below never mix with data from other test files.
Users/products/interactions are seeded through the ORM; the API tests use the
real register/login/interactions endpoints.
"""

import itertools
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from core.config import settings
from database.database import Base, get_db
from main import app
from ml.recommendation import content_based
from ml.recommendation.hybrid import HybridWeights
from models.interaction import InteractionType, ProductInteraction
from models.product import Product
from models.user import User
from services import recommendation_service as svc

VIEW, CART, BUY = InteractionType.VIEW, InteractionType.CART_ADD, InteractionType.PURCHASE

client = TestClient(app)
_counter = itertools.count(1)


class Env:
    """Isolated database + tiny seeding helpers."""

    def __init__(self, engine, session_factory):
        self.engine = engine
        self.Session = session_factory
        self.db = session_factory()

    def user(self):
        u = User(email=f"seed_{next(_counter)}_{uuid.uuid4().hex[:6]}@example.com", hashed_password="x")
        self.db.add(u)
        self.db.commit()
        return u.id

    def product(self, name, description=None, category=None, brand=None):
        p = Product(name=name, description=description, price=10.0, stock=5, category=category, brand=brand)
        self.db.add(p)
        self.db.commit()
        return p.id

    def interact(self, user_id, product_id, kind):
        self.db.add(ProductInteraction(user_id=user_id, product_id=product_id, interaction_type=kind))
        self.db.commit()

    def recs(self, user_id, limit=10, **kwargs):
        # fresh session so the service always sees committed state
        session = self.Session()
        try:
            return svc.get_user_recommendations(session, user_id, limit, **kwargs)
        finally:
            session.close()

    def cf_recs(self, user_id, limit=10):
        session = self.Session()
        try:
            return svc.get_collaborative_recommendations(session, user_id, limit)
        finally:
            session.close()


@pytest.fixture
def env(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'hybrid.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def _override():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    previous = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = _override
    content_based.clear_cache()
    e = Env(engine, factory)
    yield e
    e.db.close()
    if previous is None:
        app.dependency_overrides.pop(get_db, None)
    else:
        app.dependency_overrides[get_db] = previous
    content_based.clear_cache()
    engine.dispose()


def ids_of(items):
    return [i["product"].id for i in items]


def seed_gaming(env):
    """L (target's interest), M and K content-similar to L, Z unrelated text."""
    return {
        "L": env.product("Gaming Laptop", "asus rog gaming laptop electronics", "Electronics", "Asus"),
        "M": env.product("Gaming Mouse", "asus rog gaming mouse electronics", "Electronics", "Asus"),
        "K": env.product("Gaming Keyboard", "asus rog gaming keyboard electronics", "Electronics", "Asus"),
        "Z": env.product("Zebra Crossing Sign", "reflective roadside zebra signage", "Roadwork", "Acme"),
        "B": env.product("Kitchen Blender", "powerful kitchen blender appliance", "Kitchen", "Acme"),
    }


# ============================ Collaborative (service) ==========================

def test_cf_recommends_products_of_similar_users(env):
    p = seed_gaming(env)
    me, neighbor, stranger = env.user(), env.user(), env.user()
    env.interact(me, p["L"], BUY)
    env.interact(neighbor, p["L"], BUY)
    env.interact(neighbor, p["Z"], BUY)
    env.interact(stranger, p["B"], BUY)  # no overlap with me

    result = env.cf_recs(me)
    assert ids_of(result) == [p["Z"]]
    assert result[0]["recommendation_source"] == "collaborative"
    assert result[0]["content_score"] == 0.0 and result[0]["collaborative_score"] > 0


def test_cf_excludes_already_interacted_products(env):
    p = seed_gaming(env)
    me, neighbor = env.user(), env.user()
    env.interact(me, p["L"], BUY)
    env.interact(me, p["Z"], VIEW)
    env.interact(neighbor, p["L"], BUY)
    env.interact(neighbor, p["Z"], BUY)
    env.interact(neighbor, p["B"], BUY)

    assert ids_of(env.cf_recs(me)) == [p["B"]]


def test_cf_purchase_outranks_view_from_similar_users(env):
    p = seed_gaming(env)
    me, n1, n2 = env.user(), env.user(), env.user()
    env.interact(me, p["L"], BUY)
    env.interact(n1, p["L"], BUY)
    env.interact(n1, p["Z"], BUY)
    env.interact(n2, p["L"], BUY)
    env.interact(n2, p["B"], VIEW)

    result = env.cf_recs(me)
    assert ids_of(result) == [p["Z"], p["B"]]
    assert result[0]["score"] > result[1]["score"]


def test_changing_interaction_behavior_changes_collaborative_ranking(env):
    p = seed_gaming(env)
    me, n1, n2 = env.user(), env.user(), env.user()
    env.interact(me, p["L"], BUY)
    env.interact(n1, p["L"], BUY)
    env.interact(n1, p["Z"], VIEW)
    env.interact(n2, p["L"], BUY)
    env.interact(n2, p["B"], VIEW)

    before = ids_of(env.cf_recs(me))
    assert before == [p["Z"], p["B"]]  # identical behavior -> tie broken by product id

    env.interact(n2, p["B"], BUY)  # n2 now also purchased B
    after = ids_of(env.cf_recs(me))
    assert after == [p["B"], p["Z"]]


def test_cf_cold_start_and_degenerate_data_do_not_crash(env):
    p = seed_gaming(env)
    lonely, one_view, other = env.user(), env.user(), env.user()

    assert env.cf_recs(lonely) == []  # no interactions at all
    env.interact(one_view, p["L"], VIEW)  # single interaction, nobody else
    assert env.cf_recs(one_view) == []  # no similar users
    env.interact(other, p["B"], BUY)  # exists but no overlap
    assert env.cf_recs(one_view) == []


def test_cf_only_uses_the_requested_users_history(env):
    p = seed_gaming(env)
    me, neighbor, third = env.user(), env.user(), env.user()
    env.interact(me, p["L"], BUY)
    env.interact(neighbor, p["L"], BUY)
    env.interact(neighbor, p["Z"], BUY)

    assert ids_of(env.cf_recs(me)) == [p["Z"]]
    # `third` has no history; neighbor's/my history must not be attributed to them.
    assert env.cf_recs(third) == []
    # neighbor's own recommendations are derived from THEIR history, not mine
    assert ids_of(env.cf_recs(neighbor)) == []  # they already have everything I have


# ============================ Hybrid (service) =================================

def _hybrid_scenario(env):
    p = seed_gaming(env)
    me, neighbor = env.user(), env.user()
    env.interact(me, p["L"], BUY)
    env.interact(neighbor, p["L"], BUY)
    env.interact(neighbor, p["K"], BUY)
    env.interact(neighbor, p["Z"], BUY)
    return p, me


def test_hybrid_combines_both_signals(env):
    p, me = _hybrid_scenario(env)
    by_id = {i["product"].id: i for i in env.recs(me, 10)}

    assert by_id[p["K"]]["recommendation_source"] == "hybrid"  # content AND collaborative
    assert by_id[p["M"]]["recommendation_source"] == "content"  # content only
    assert by_id[p["Z"]]["recommendation_source"] == "collaborative"  # CF only
    assert by_id[p["K"]]["content_score"] > 0 and by_id[p["K"]]["collaborative_score"] > 0
    assert by_id[p["M"]]["collaborative_score"] == 0.0
    assert by_id[p["Z"]]["content_score"] == 0.0


def test_hybrid_merges_duplicates_and_ranks_merged_candidate_first(env):
    p, me = _hybrid_scenario(env)
    result = env.recs(me, 10)
    ids = ids_of(result)
    assert len(ids) == len(set(ids))
    assert ids[0] == p["K"]  # same content as M and same CF as Z, plus both
    k = result[0]
    assert k["score"] == pytest.approx(0.5 * k["content_score"] + 0.5 * k["collaborative_score"])


def test_hybrid_sorted_numeric_limited_and_excludes_interacted(env):
    p, me = _hybrid_scenario(env)
    env.interact(me, p["M"], VIEW)  # M is now interacted -> must disappear

    result = env.recs(me, 10)
    scores = [i["score"] for i in result]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(s, float) and s > 0 for s in scores)
    assert p["L"] not in ids_of(result) and p["M"] not in ids_of(result)
    assert len(env.recs(me, 1)) == 1
    top = env.recs(me, 1)[0]
    assert (top["product"].id, top["score"]) == (result[0]["product"].id, result[0]["score"])


def test_content_only_and_collaborative_only_weights(env):
    p, me = _hybrid_scenario(env)

    content_only = env.recs(me, 10, weights=HybridWeights(1, 0))
    assert set(ids_of(content_only)) == {p["K"], p["M"]}
    assert all(i["recommendation_source"] == "content" for i in content_only)

    collab_only = env.recs(me, 10, weights=HybridWeights(0, 1))
    assert set(ids_of(collab_only)) == {p["K"], p["Z"]}
    assert all(i["recommendation_source"] == "collaborative" for i in collab_only)


def test_weights_come_from_settings(env, monkeypatch):
    p, me = _hybrid_scenario(env)
    monkeypatch.setattr(settings, "HYBRID_CONTENT_WEIGHT", 0.0)
    monkeypatch.setattr(settings, "HYBRID_COLLABORATIVE_WEIGHT", 1.0)
    result = env.recs(me, 10)
    assert set(ids_of(result)) == {p["K"], p["Z"]}


def test_default_weights_are_balanced():
    assert settings.HYBRID_CONTENT_WEIGHT == 0.5
    assert settings.HYBRID_COLLABORATIVE_WEIGHT == 0.5


def test_purchase_beats_view_in_hybrid_score(env):
    p = seed_gaming(env)
    buyer, viewer = env.user(), env.user()
    env.interact(buyer, p["L"], BUY)
    env.interact(viewer, p["L"], VIEW)

    bought = {i["product"].id: i["score"] for i in env.recs(buyer)}
    viewed = {i["product"].id: i["score"] for i in env.recs(viewer)}
    assert bought[p["M"]] > viewed[p["M"]]


def test_changing_product_content_changes_content_score(env):
    p = seed_gaming(env)
    me = env.user()
    env.interact(me, p["L"], BUY)

    before = {i["product"].id: i["content_score"] for i in env.recs(me, 10)}
    assert p["B"] not in before  # blender shares no vocabulary with the laptop

    blender = env.db.get(Product, p["B"])
    blender.description = "asus rog gaming appliance electronics"
    env.db.commit()

    after = {i["product"].id: i["content_score"] for i in env.recs(me, 10)}
    assert after[p["B"]] > 0


def test_hybrid_reflects_both_signals_when_behavior_changes(env):
    p, me = _hybrid_scenario(env)
    base = {i["product"].id: i["score"] for i in env.recs(me, 10)}
    other = env.user()  # a second neighbor also bought Z -> Z gets stronger CF support
    env.interact(other, p["L"], BUY)
    env.interact(other, p["Z"], BUY)
    boosted = {i["product"].id: i["score"] for i in env.recs(me, 10)}
    assert boosted[p["Z"]] > base[p["Z"]]  # collaborative signal moved the hybrid score
    assert boosted[p["M"]] == pytest.approx(base[p["M"]])  # content-only item unaffected


# ============================ Cold start & fallbacks ===========================

def test_cold_start_uses_popular_products(env):
    p = seed_gaming(env)
    active, newcomer = env.user(), env.user()
    env.interact(active, p["Z"], BUY)

    result = env.recs(newcomer)
    assert ids_of(result)[0] == p["Z"]
    assert all(i["recommendation_source"] == "popular" for i in result)
    assert all(i["content_score"] is None and i["collaborative_score"] is None for i in result)


def test_cold_start_without_any_interaction_data_returns_newest_products(env):
    p = seed_gaming(env)
    result = env.recs(env.user(), limit=3)
    assert len(result) == 3
    assert set(ids_of(result)) <= set(p.values())


def test_no_personalized_signal_falls_back_to_popular_excluding_interacted(env):
    a = env.product("Zebra Crossing Sign", "reflective roadside signage")
    b = env.product("Quantum Widget", "experimental physics device")
    c = env.product("Mango Smoothie", "fresh tropical fruit drink")
    me, other = env.user(), env.user()
    env.interact(me, a, BUY)  # nothing textually similar, nobody overlapping
    env.interact(other, b, BUY)
    env.interact(other, b, BUY)
    env.interact(other, c, VIEW)

    result = env.recs(me)
    assert ids_of(result) == [b, c]
    assert a not in ids_of(result)
    assert all(i["recommendation_source"] == "popular" for i in result)


def test_empty_catalog_and_single_product_catalog(env):
    me = env.user()
    assert env.recs(me) == []
    assert env.cf_recs(me) == []

    only = env.product("Only Product", "the single item")
    env.interact(me, only, BUY)
    assert env.recs(me) == []
    assert env.cf_recs(me) == []


def test_missing_optional_fields_do_not_crash(env):
    a = env.product("Plain Widget")  # no description/category/brand
    b = env.product("Plain Gadget")
    c = env.product("Plain Gizmo", description="")
    me, neighbor = env.user(), env.user()
    env.interact(me, a, VIEW)
    env.interact(neighbor, a, VIEW)
    env.interact(neighbor, b, BUY)
    env.interact(neighbor, c, CART)

    result = env.recs(me, 10)
    assert {b, c} <= set(ids_of(result))
    assert all(isinstance(i["score"], float) for i in result)


def test_very_small_datasets_never_crash(env):
    a = env.product("Alpha", "first")
    b = env.product("Beta", "second")
    u1, u2 = env.user(), env.user()
    env.interact(u1, a, VIEW)
    env.interact(u2, a, VIEW)
    env.interact(u2, b, VIEW)
    assert ids_of(env.recs(u1)) == [b]
    assert env.recs(u2) == [] or a not in ids_of(env.recs(u2))


# ============================ Efficiency =======================================

def _count_queries(env, fn):
    statements = []

    def _capture(conn, cursor, statement, *_):
        statements.append(statement)

    event.listen(env.engine, "before_cursor_execute", _capture)
    try:
        fn()
    finally:
        event.remove(env.engine, "before_cursor_execute", _capture)
    return len(statements)


def test_query_count_does_not_grow_with_users_or_products(env):
    p = seed_gaming(env)
    me = env.user()
    env.interact(me, p["L"], BUY)
    small = _count_queries(env, lambda: env.recs(me, 10))

    for i in range(25):
        other = env.user()
        pid = env.product(f"Extra Item {i}", f"filler text {i}")
        env.interact(other, p["L"], BUY)
        env.interact(other, pid, VIEW)
    large = _count_queries(env, lambda: env.recs(me, 10))

    assert large == small  # no per-user / per-product queries


# ============================ API / security ===================================

def _register_and_login():
    email = f"api_{uuid.uuid4().hex[:10]}@example.com"
    password = "StrongPass123!"
    resp = client.post(
        "/auth/register",
        json={"email": email, "full_name": "Hybrid Test", "password": password, "confirm_password": password},
    )
    assert resp.status_code == 200, resp.text
    token = client.post("/auth/login", data={"username": email, "password": password}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    return headers, client.get("/auth/me", headers=headers).json()["id"]


def _api_interact(headers, product_id, kind):
    resp = client.post("/interactions/", json={"product_id": product_id, "interaction_type": kind}, headers=headers)
    assert resp.status_code == 200, resp.text


def test_me_endpoint_returns_hybrid_items_and_stays_backward_compatible(env):
    p = seed_gaming(env)
    headers, _ = _register_and_login()
    n = env.user()
    env.interact(n, p["L"], BUY)
    env.interact(n, p["Z"], BUY)
    _api_interact(headers, p["L"], "PURCHASE")

    resp = client.get("/recommendations/me?limit=10", headers=headers)
    assert resp.status_code == 200, resp.text
    items = resp.json()["recommendations"]
    assert items
    for item in items:
        assert {"product", "score"} <= set(item)  # original contract intact
        assert isinstance(item["score"], (int, float))
        assert item["product"]["id"] != p["L"]
    by_id = {i["product"]["id"]: i for i in items}
    assert by_id[p["Z"]]["recommendation_source"] == "collaborative"
    assert by_id[p["M"]]["recommendation_source"] == "content"
    scores = [i["score"] for i in items]
    assert scores == sorted(scores, reverse=True)


def test_me_endpoint_cannot_impersonate_another_user(env):
    p = seed_gaming(env)
    headers_a, id_a = _register_and_login()
    headers_b, _ = _register_and_login()
    n = env.user()
    env.interact(n, p["L"], BUY)
    env.interact(n, p["Z"], BUY)
    _api_interact(headers_a, p["L"], "PURCHASE")  # only A has history

    a_recs = client.get("/recommendations/me", headers=headers_a).json()["recommendations"]
    b_plain = client.get("/recommendations/me", headers=headers_b).json()["recommendations"]
    b_spoof = client.get(f"/recommendations/me?user_id={id_a}", headers=headers_b).json()["recommendations"]

    assert {i["recommendation_source"] for i in a_recs} <= {"hybrid", "content", "collaborative"}
    # B has no history: cold start, never A's personalized results
    assert b_spoof == b_plain
    assert {i["recommendation_source"] for i in b_spoof} == {"popular"}


def test_me_endpoint_requires_authentication():
    assert client.get("/recommendations/me").status_code == 401


def test_product_similarity_endpoint_is_unchanged(env):
    p = seed_gaming(env)
    headers, _ = _register_and_login()
    data = client.get(f"/recommendations/product/{p['L']}?limit=10", headers=headers).json()
    assert set(data) == {"product_id", "recommendations"}
    for rec in data["recommendations"]:
        assert set(rec) == {"product", "similarity_score"}
    assert p["L"] not in [r["product"]["id"] for r in data["recommendations"]]
