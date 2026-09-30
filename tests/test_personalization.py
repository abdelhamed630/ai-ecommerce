"""Phase 3 personalization tests for GET /recommendations/me.

Isolated SQLite database (same pattern as tests/test_recommendations.py) with
data inserted directly through the ORM so scenarios are exact; identities come
from real JWTs issued through the API.
"""

import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.database import Base, get_db
from main import app
from ml.recommendation import reasons
from models.interaction import InteractionType as IT
from models.interaction import ProductInteraction
from models.product import Product
from models.user import User
from services import recommendation_service

_DB_PATH = "./test_personalization.db"
_engine = create_engine(f"sqlite:///{_DB_PATH}", connect_args={"check_same_thread": False})
_Session = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
client = TestClient(app)


def _override_get_db():
    db = _Session()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module", autouse=True)
def _isolated_db():
    Base.metadata.create_all(bind=_engine)
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)
    _engine.dispose()
    if os.path.exists(_DB_PATH):
        os.remove(_DB_PATH)


@pytest.fixture
def db():
    session = _Session()
    for model in (ProductInteraction, Product, User):
        session.query(model).delete()
    session.commit()
    yield session
    session.close()


class Shopper:
    def __init__(self, db):
        email = f"p_{uuid.uuid4().hex[:10]}@example.com"
        pw = "StrongPass123!"
        r = client.post(
            "/auth/register",
            json={"email": email, "full_name": "P", "password": pw, "confirm_password": pw},
        )
        assert r.status_code == 200, r.text
        token = client.post("/auth/login", data={"username": email, "password": pw}).json()["access_token"]
        self.headers = {"Authorization": f"Bearer {token}"}
        self.db = db
        self.id = db.query(User).filter(User.email == email).first().id

    def act(self, product, kind, times=1):
        for _ in range(times):
            self.db.add(ProductInteraction(user_id=self.id, product_id=product.id, interaction_type=kind))
        self.db.commit()

    def recs(self, query=""):
        r = client.get(f"/recommendations/me{query}", headers=self.headers)
        assert r.status_code == 200, r.text
        return r.json()["recommendations"]


def _product(db, name, description=None, **kw):
    p = Product(name=name, description=description, price=10.0, stock=5, **kw)
    db.add(p)
    db.commit()
    return p


def _scenario(db):
    """A views P1. B viewed P1, P3, P4.
    Expected for A: P2 content-only, P3 collaborative-only, P4 hybrid."""
    p1 = _product(db, "Red running shoes", "red running shoes for runners")
    p2 = _product(db, "Red running sneakers", "red running sneakers for runners")
    p3 = _product(db, "Blue kitchen blender", "blender appliance for smoothies")
    p4 = _product(db, "Red running socks", "red running socks for runners")
    a, b = Shopper(db), Shopper(db)
    a.act(p1, IT.VIEW)
    for p in (p1, p3, p4):
        b.act(p, IT.VIEW)
    return a, b, (p1, p2, p3, p4)


# ---------------- reasons (pure) ----------------

def test_reason_text_per_source_and_kinds():
    assert reasons.build_reason("content", {"VIEW"}) == "Similar to products you viewed"
    assert reasons.build_reason("content", {"PURCHASE"}) == "Similar to products you purchased"
    assert reasons.build_reason("content", {"CART_ADD"}) == "Similar to products you added to your cart"
    assert reasons.build_reason("content", {"PURCHASE", "VIEW"}) == "Similar to products you viewed and purchased"
    assert (
        reasons.build_reason("content", {"VIEW", "CART_ADD", "PURCHASE"})
        == "Similar to products you viewed, added to your cart and purchased"
    )
    assert reasons.build_reason("collaborative", {"VIEW"}) == "Popular among users with similar activity"
    assert reasons.build_reason("popular") == "Popular among shoppers"
    assert reasons.build_reason("unknown") is None


def test_collaborative_and_popular_reasons_never_claim_content_similarity():
    for kinds in ({"VIEW"}, {"PURCHASE"}, set()):
        assert "Similar to" not in reasons.build_reason("collaborative", kinds)
        assert "similar activity" not in reasons.build_reason("popular", kinds)


def test_hybrid_reason_mentions_both_signals_and_is_deterministic():
    text = reasons.build_reason("hybrid", {"VIEW", "PURCHASE"})
    assert text == "Similar to products you viewed and purchased and popular among users with similar activity"
    assert text == reasons.build_reason("hybrid", ["PURCHASE", "VIEW"])


# ---------------- reasons match real sources ----------------

def test_reasons_match_actual_sources(db):
    a, _, (p1, p2, p3, p4) = _scenario(db)
    by_id = {i["product"]["id"]: i for i in a.recs()}
    assert by_id[p2.id]["recommendation_source"] == "content"
    assert by_id[p2.id]["reason"] == "Similar to products you viewed"
    assert by_id[p3.id]["recommendation_source"] == "collaborative"
    assert by_id[p3.id]["reason"] == "Popular among users with similar activity"
    assert by_id[p4.id]["recommendation_source"] == "hybrid"
    assert by_id[p4.id]["reason"] == (
        "Similar to products you viewed and popular among users with similar activity"
    )


def test_reason_reflects_purchase_history(db):
    a, _, (p1, p2, *_rest) = _scenario(db)
    a.db.query(ProductInteraction).filter(ProductInteraction.user_id == a.id).delete()
    a.act(p1, IT.PURCHASE)
    by_id = {i["product"]["id"]: i for i in a.recs()}
    assert by_id[p2.id]["reason"] == "Similar to products you purchased"


def test_view_only_user_never_told_purchased(db):
    a, _, _ = _scenario(db)
    assert all("purchased" not in (i["reason"] or "") for i in a.recs())


def test_cold_start_reason_is_popular(db):
    p1 = _product(db, "Alpha")
    p2 = _product(db, "Beta")
    other = Shopper(db)
    other.act(p2, IT.PURCHASE)
    me = Shopper(db)
    items = me.recs()
    assert {i["recommendation_source"] for i in items} == {"popular"}
    assert {i["reason"] for i in items} == {"Popular among shoppers"}
    assert items[0]["product"]["id"] == p2.id


# ---------------- identity ----------------

def test_uses_jwt_identity_not_client_input(db):
    a, b, _ = _scenario(db)
    plain = a.recs()
    for spoof in (f"?user_id={b.id}", f"?userId={b.id}", f"?user_id={b.id}&limit=10", "?email=x@y.z"):
        assert a.recs(spoof) == plain
    # B has a different history, so B's own view differs from A's.
    assert b.recs() != plain


def test_requires_authentication():
    assert client.get("/recommendations/me").status_code == 401
    assert client.get("/recommendations/me?user_id=1").status_code == 401


# ---------------- rules ----------------

def test_limit_respected_and_validated(db):
    products = [_product(db, f"Gadget number {i}", "gadget device") for i in range(8)]
    me = Shopper(db)
    me.act(products[0], IT.VIEW)
    assert len(me.recs("?limit=3")) == 3
    assert len(me.recs("?limit=1")) == 1
    assert len(me.recs("?limit=50")) == 7  # everything except the viewed product
    for bad in ("0", "51", "abc"):
        assert client.get(f"/recommendations/me?limit={bad}", headers=me.headers).status_code == 422


def test_already_interacted_products_excluded(db):
    a, _, (p1, *_rest) = _scenario(db)
    a.act(_rest[0], IT.CART_ADD)  # P2
    ids = [i["product"]["id"] for i in a.recs()]
    assert p1.id not in ids and _rest[0].id not in ids


def test_interacted_excluded_in_popular_fallback_too(db):
    p1, p2, p3 = (_product(db, n) for n in ("Alpha", "Beta", "Gamma"))
    other = Shopper(db)
    for p in (p1, p2, p3):
        other.act(p, IT.PURCHASE)
    me = Shopper(db)
    me.act(p1, IT.VIEW)  # nothing similar/overlapping beyond popularity
    ids = [i["product"]["id"] for i in me.recs()]
    assert p1.id not in ids


def test_no_duplicate_recommendations(db):
    a, b, _ = _scenario(db)
    for user in (a, b):
        ids = [i["product"]["id"] for i in user.recs("?limit=50")]
        assert len(ids) == len(set(ids))


def test_deterministic_ordering_including_ties(db):
    # Equal popularity everywhere: order must be by ascending product id
    # regardless of the order interactions were inserted.
    products = [_product(db, f"Item {i}") for i in range(5)]
    other = Shopper(db)
    for p in reversed(products):
        other.act(p, IT.VIEW)
    me = Shopper(db)
    first = [i["product"]["id"] for i in me.recs()]
    assert first == sorted(first) == [p.id for p in products]
    assert [i["product"]["id"] for i in me.recs()] == first


def test_repeated_calls_identical_for_personalized_results(db):
    a, _, _ = _scenario(db)
    assert a.recs() == a.recs()


def test_only_views_and_with_purchases_both_work(db):
    a, b, (p1, p2, p3, p4) = _scenario(db)
    assert a.recs()  # views only
    b.act(p2, IT.PURCHASE)
    assert isinstance(b.recs(), list)


def test_missing_product_metadata_does_not_crash(db):
    bare = _product(db, "Bare", None, category=None, brand=None, image_url=None)
    p1 = _product(db, "Red running shoes", "red running shoes")
    p2 = _product(db, "Red running socks", None)
    me = Shopper(db)
    me.act(p1, IT.VIEW)
    items = {i["product"]["id"]: i for i in me.recs("?limit=50")}
    assert p2.id in items
    assert items[p2.id]["product"]["description"] is None
    assert items[p2.id]["product"]["category"] is None and items[p2.id]["product"]["brand"] is None

    only_bare = Shopper(db)
    only_bare.act(bare, IT.VIEW)
    assert isinstance(only_bare.recs(), list)


def test_empty_catalog_and_user_without_history(db):
    me = Shopper(db)
    assert me.recs() == []


def test_interactions_with_deleted_products_do_not_use_up_slots(db):
    products = [_product(db, f"Thing {i}") for i in range(3)]
    other = Shopper(db)
    ghost = Product(name="Ghost", price=1.0, stock=1)
    db.add(ghost)
    db.commit()
    other.act(ghost, IT.PURCHASE, times=5)  # most popular...
    ghost_id = ghost.id
    db.delete(ghost)  # ...then deleted (SQLite does not enforce the FK)
    db.commit()
    me = Shopper(db)
    ids = [i["product"]["id"] for i in me.recs("?limit=3")]
    assert ghost_id not in ids


# ---------------- backward-compatible contract ----------------

def test_response_contract_is_backward_compatible(db):
    a, _, _ = _scenario(db)
    body = client.get("/recommendations/me", headers=a.headers).json()
    assert set(body) == {"recommendations"}
    for item in body["recommendations"]:
        # legacy fields
        assert {"product", "score"} <= set(item)
        assert isinstance(item["score"], float)
        # Phase 2 optional fields preserved
        assert {"recommendation_source", "content_score", "collaborative_score"} <= set(item)
        # Phase 3 additive field
        assert "reason" in item
        assert set(item) == {
            "product", "score", "recommendation_source", "content_score", "collaborative_score", "reason",
        }
        assert {"id", "name", "price", "created_at"} <= set(item["product"])
        assert isinstance(item["reason"], str)


def test_service_layer_items_include_reason(db):
    a, _, _ = _scenario(db)
    items = recommendation_service.get_user_recommendations(db, a.id, 10)
    assert items and all(i["reason"] for i in items)
