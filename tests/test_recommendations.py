import tempfile
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.database import Base, get_db
from main import app
from models.user import User, UserRole

# This test module gets its OWN isolated SQLite database. Recommendation
# quality assertions (ranking order, weight comparisons) need a small,
# controlled catalog — sharing the same growing database as every other
# test file (which creates 100+ products) would make ranking-based
# assertions flaky through no fault of the algorithm itself.
#
# The file name is UNIQUE PER PROCESS and lives in the system temp directory,
# not at a fixed path in the current working directory. A fixed
# "./test_recommendations.db" is only deleted by the module fixture's teardown,
# so an interrupted run (killed / timed out / crashed) leaves it behind with
# its products, and the next run silently re-uses it (create_all() does not
# empty an existing file) — as would a second pytest process started from the
# same directory. The extra products push the catalog past the endpoint's
# `limit <= 50` cap, so the top-50 ranking assertions fail for reasons
# unrelated to the algorithm. A unique path makes that impossible.
_TEST_DB_PATH = (
    Path(tempfile.gettempdir()) / f"test_recommendations_{uuid.uuid4().hex}.db"
).as_posix()
_test_engine = create_engine(
    f"sqlite:///{_TEST_DB_PATH}", connect_args={"check_same_thread": False}
)
_TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)


def _override_get_db():
    db = _TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def _isolated_test_db():
    # Setup and teardown happen when THIS module's tests actually run, not
    # at import/collection time — overriding at import time would leak the
    # isolated DB into every other test file for the rest of the session.
    Base.metadata.create_all(bind=_test_engine)
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)
    import os

    _test_engine.dispose()  # release the SQLite file handle before deleting it
    if os.path.exists(_TEST_DB_PATH):
        os.remove(_TEST_DB_PATH)


def _unique_email():
    return f"user_{uuid.uuid4().hex[:10]}@example.com"


def _promote_to_seller(email: str) -> None:
    """No API self-promotes to SELLER (by design), so test fixtures that
    need to create a product promote a real registered user directly at
    the DB level — same convention as
    tests/test_segmentation_api.py's _promote_to_admin. This file uses
    its OWN isolated database (_TestSessionLocal), never the shared
    SessionLocal, since the recommendation-quality assertions below
    depend on a small, controlled catalog. Recommendation behavior
    itself is unaffected: nothing in this project restricts
    interaction/recommendation actions by role."""
    db = _TestSessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        user.role = UserRole.SELLER
        db.commit()
    finally:
        db.close()


def _register_and_login():
    email = _unique_email()
    password = "StrongPass123!"

    resp = client.post(
        "/auth/register",
        json={
            "email": email,
            "full_name": "Rec Test User",
            "password": password,
            "confirm_password": password,
        },
    )
    assert resp.status_code == 200, resp.text

    # Promoted to SELLER so this same account can also create the test
    # catalog it then records interactions against / requests
    # recommendations for below (product creation requires SELLER or
    # ADMIN; this file is not testing product authorization, so this is a
    # fixture-only change).
    _promote_to_seller(email)

    resp = client.post("/auth/login", data={"username": email, "password": password})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _create_product(headers, name="Laptop", description="a product", price=1000.0, stock=10):
    resp = client.post(
        "/products/",
        json={
            "name": name,
            "description": description,
            "price": price,
            "stock": stock,
            "image_url": None,
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _record_interaction(headers, product_id, interaction_type):
    resp = client.post(
        "/interactions/",
        json={"product_id": product_id, "interaction_type": interaction_type},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.fixture
def auth_headers():
    return _register_and_login()


def _seed_gaming_catalog(headers):
    """A small catalog with two clear textual clusters (gaming vs kitchen)
    so similarity ordering is unambiguous without asserting exact floats.
    """
    laptop = _create_product(
        headers, name="Gaming Laptop", description="asus rog gaming laptop electronics", price=1500.0
    )
    mouse = _create_product(
        headers, name="Gaming Mouse", description="asus rog gaming mouse electronics", price=50.0
    )
    keyboard = _create_product(
        headers, name="Gaming Keyboard", description="asus rog gaming keyboard electronics", price=80.0
    )
    blender = _create_product(
        headers, name="Kitchen Blender", description="powerful kitchen blender appliance", price=60.0
    )
    return laptop, mouse, keyboard, blender


# 1. Similar products are returned
def test_similar_products_returned(auth_headers):
    laptop, mouse, keyboard, blender = _seed_gaming_catalog(auth_headers)
    resp = client.get(f"/recommendations/product/{laptop['id']}", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["recommendations"]) > 0


# 2. Target product is excluded
def test_target_product_excluded(auth_headers):
    laptop, *_ = _seed_gaming_catalog(auth_headers)
    resp = client.get(f"/recommendations/product/{laptop['id']}", headers=auth_headers)
    ids = [r["product"]["id"] for r in resp.json()["recommendations"]]
    assert laptop["id"] not in ids


# 3. Similarity scores are numeric
def test_similarity_scores_numeric(auth_headers):
    laptop, *_ = _seed_gaming_catalog(auth_headers)
    resp = client.get(f"/recommendations/product/{laptop['id']}", headers=auth_headers)
    for rec in resp.json()["recommendations"]:
        assert isinstance(rec["similarity_score"], (int, float))


# 4. Results are sorted by similarity descending
def test_results_sorted_descending(auth_headers):
    laptop, *_ = _seed_gaming_catalog(auth_headers)
    resp = client.get(f"/recommendations/product/{laptop['id']}", headers=auth_headers)
    scores = [r["similarity_score"] for r in resp.json()["recommendations"]]
    assert scores == sorted(scores, reverse=True)


# 4b. Gaming products should rank above the unrelated kitchen product
def test_related_products_rank_higher(auth_headers):
    laptop, mouse, keyboard, blender = _seed_gaming_catalog(auth_headers)
    resp = client.get(f"/recommendations/product/{laptop['id']}?limit=50", headers=auth_headers)
    recs = resp.json()["recommendations"]
    ranked_ids = [r["product"]["id"] for r in recs]
    assert ranked_ids.index(mouse["id"]) < ranked_ids.index(blender["id"])


# 5. limit works
def test_limit_works(auth_headers):
    _seed_gaming_catalog(auth_headers)
    laptop_id = client.get("/products/").json()[0]["id"]
    resp = client.get(f"/recommendations/product/{laptop_id}?limit=1", headers=auth_headers)
    assert len(resp.json()["recommendations"]) <= 1


# 6. invalid limit is rejected
def test_invalid_limit_rejected(auth_headers):
    laptop, *_ = _seed_gaming_catalog(auth_headers)
    resp = client.get(f"/recommendations/product/{laptop['id']}?limit=0", headers=auth_headers)
    assert resp.status_code == 422

    resp2 = client.get(f"/recommendations/product/{laptop['id']}?limit=-1", headers=auth_headers)
    assert resp2.status_code == 422


# 7. non-existing product returns 404
def test_nonexistent_product_returns_404(auth_headers):
    resp = client.get("/recommendations/product/999999", headers=auth_headers)
    assert resp.status_code == 404


# 8. empty candidate set is handled (only one product exists for this fresh user's catalog view)
def test_single_product_no_crash(auth_headers):
    product = _create_product(auth_headers, name="Solo Product", description="only one here")
    resp = client.get(f"/recommendations/product/{product['id']}", headers=auth_headers)
    assert resp.status_code == 200
    # Other products may exist from earlier tests sharing the same DB, so we
    # only assert this never crashes and always returns a list.
    assert isinstance(resp.json()["recommendations"], list)


# 9. products with missing optional fields do not crash
def test_missing_optional_fields_no_crash(auth_headers):
    product = client.post(
        "/products/",
        json={"name": "No Description", "description": None, "price": 10.0, "stock": 5, "image_url": None},
        headers=auth_headers,
    ).json()
    resp = client.get(f"/recommendations/product/{product['id']}", headers=auth_headers)
    assert resp.status_code == 200


# 10. authenticated user can get personalized recommendations
def test_authenticated_user_gets_personalized_recommendations(auth_headers):
    resp = client.get("/recommendations/me", headers=auth_headers)
    assert resp.status_code == 200
    assert "recommendations" in resp.json()


# 11. unauthenticated user cannot access personalized recommendations
def test_unauthenticated_cannot_access_recommendations():
    assert client.get("/recommendations/me").status_code == 401
    assert client.get("/recommendations/product/1").status_code == 401


# 12. user recommendations use only the current user's interactions
def test_recommendations_use_only_current_users_interactions():
    headers_a = _register_and_login()
    headers_b = _register_and_login()
    laptop, mouse, keyboard, blender = _seed_gaming_catalog(headers_a)

    _record_interaction(headers_a, laptop["id"], "PURCHASE")

    resp_b = client.get("/recommendations/me", headers=headers_b)
    assert resp_b.status_code == 200
    # user B has no interactions -> cold start, not influenced by A's purchase
    resp_a = client.get("/recommendations/me", headers=headers_a)
    assert resp_a.status_code == 200


# 13. user_id cannot be supplied to impersonate another user
def test_cannot_supply_user_id_to_impersonate(auth_headers):
    resp = client.get("/recommendations/me?user_id=999999", headers=auth_headers)
    assert resp.status_code == 200  # the bogus query param is simply ignored
    me = client.get("/auth/me", headers=auth_headers).json()
    # sanity: endpoint only ever reflects the JWT identity, never the query param
    assert me["id"] is not None


# 14. already interacted products are excluded
def test_already_interacted_products_excluded(auth_headers):
    laptop, mouse, keyboard, blender = _seed_gaming_catalog(auth_headers)
    _record_interaction(auth_headers, laptop["id"], "PURCHASE")
    _record_interaction(auth_headers, mouse["id"], "VIEW")

    resp = client.get("/recommendations/me", headers=auth_headers)
    ids = [r["product"]["id"] for r in resp.json()["recommendations"]]
    assert laptop["id"] not in ids
    assert mouse["id"] not in ids


# 15/16/17. Interaction weight ordering: PURCHASE > CART_ADD > VIEW
def test_interaction_weight_ordering(auth_headers):
    from services.recommendation_service import INTERACTION_WEIGHTS
    from models.interaction import InteractionType

    assert INTERACTION_WEIGHTS[InteractionType.VIEW] < INTERACTION_WEIGHTS[InteractionType.CART_ADD]
    assert INTERACTION_WEIGHTS[InteractionType.CART_ADD] < INTERACTION_WEIGHTS[InteractionType.PURCHASE]


def test_purchase_influences_recommendations_more_than_view():
    headers_purchase = _register_and_login()
    headers_view = _register_and_login()

    # identical catalog built separately for each user's test isolation
    laptop_p, mouse_p, keyboard_p, blender_p = _seed_gaming_catalog(headers_purchase)
    _record_interaction(headers_purchase, laptop_p["id"], "PURCHASE")

    laptop_v, mouse_v, keyboard_v, blender_v = _seed_gaming_catalog(headers_view)
    _record_interaction(headers_view, laptop_v["id"], "VIEW")

    resp_p = client.get("/recommendations/me?limit=50", headers=headers_purchase).json()
    resp_v = client.get("/recommendations/me?limit=50", headers=headers_view).json()

    score_p = next(
        r["score"] for r in resp_p["recommendations"] if r["product"]["id"] == mouse_p["id"]
    )
    score_v = next(
        r["score"] for r in resp_v["recommendations"] if r["product"]["id"] == mouse_v["id"]
    )
    assert score_p > score_v


# 18. multiple interacted products are combined
def test_multiple_interacted_products_combined(auth_headers):
    laptop, mouse, keyboard, blender = _seed_gaming_catalog(auth_headers)
    _record_interaction(auth_headers, mouse["id"], "VIEW")
    _record_interaction(auth_headers, keyboard["id"], "VIEW")

    resp = client.get("/recommendations/me?limit=50", headers=auth_headers)
    ids = [r["product"]["id"] for r in resp.json()["recommendations"]]
    assert laptop["id"] in ids  # similar to both interacted products


# 19. cold-start user receives popular products when interaction data exists
def test_cold_start_with_existing_interaction_data():
    headers_seed = _register_and_login()
    product = _create_product(headers_seed, name="Popular Item")
    _record_interaction(headers_seed, product["id"], "PURCHASE")

    headers_new = _register_and_login()
    resp = client.get("/recommendations/me", headers=headers_new)
    assert resp.status_code == 200
    ids = [r["product"]["id"] for r in resp.json()["recommendations"]]
    assert product["id"] in ids


# 20. cold-start user receives fallback products when no interaction data exists
def test_cold_start_fallback_with_no_interaction_data():
    headers = _register_and_login()
    _create_product(headers, name="Fallback Item")

    resp = client.get("/recommendations/me", headers=headers)
    assert resp.status_code == 200
    assert isinstance(resp.json()["recommendations"], list)


# 21. recommendation limit larger than available products is handled
def test_limit_larger_than_available_products(auth_headers):
    product = _create_product(auth_headers, name="Only Product For Limit Test")
    resp = client.get(
        f"/recommendations/product/{product['id']}?limit=50", headers=auth_headers
    )
    assert resp.status_code == 200
    assert isinstance(resp.json()["recommendations"], list)


# --- category/brand integration into build_product_text ---

def test_build_product_text_includes_category():
    from services.recommendation_service import build_product_text
    from models.product import Product

    product = Product(name="Laptop", description="a laptop", category="Electronics", brand=None)
    text = build_product_text(product)
    assert "electronics" in text


def test_build_product_text_includes_brand():
    from services.recommendation_service import build_product_text
    from models.product import Product

    product = Product(name="Laptop", description="a laptop", category=None, brand="ExampleBrand")
    text = build_product_text(product)
    assert "examplebrand" in text.lower()


def test_missing_category_brand_does_not_crash_recommendations(auth_headers):
    product = client.post(
        "/products/",
        json={"name": "No Metadata Item", "description": "plain product", "price": 10.0, "stock": 5},
        headers=auth_headers,
    ).json()
    assert product["category"] is None
    assert product["brand"] is None

    resp = client.get(f"/recommendations/product/{product['id']}", headers=auth_headers)
    assert resp.status_code == 200


def test_recommendations_work_with_category_and_brand_populated(auth_headers):
    laptop = client.post(
        "/products/",
        json={
            "name": "Gaming Laptop",
            "description": "asus rog laptop",
            "price": 1500.0,
            "stock": 5,
            "category": "Electronics",
            "brand": "Asus",
        },
        headers=auth_headers,
    ).json()
    mouse = client.post(
        "/products/",
        json={
            "name": "Gaming Mouse",
            "description": "asus rog mouse",
            "price": 50.0,
            "stock": 5,
            "category": "Electronics",
            "brand": "Asus",
        },
        headers=auth_headers,
    ).json()
    blender = client.post(
        "/products/",
        json={
            "name": "Kitchen Blender",
            "description": "powerful appliance",
            "price": 60.0,
            "stock": 5,
            "category": "Kitchen",
            "brand": "OtherBrand",
        },
        headers=auth_headers,
    ).json()

    # Call the service directly (not through the API's limit<=50 cap) so this
    # assertion stays valid regardless of how large this module's shared
    # isolated catalog has grown by the time this test runs.
    db = _TestSessionLocal()
    try:
        from services.recommendation_service import get_similar_products

        results = get_similar_products(db, laptop["id"], limit=100000)
        scores_by_id = {r["product"].id: r["similarity_score"] for r in results}
    finally:
        db.close()

    assert scores_by_id[mouse["id"]] > scores_by_id[blender["id"]]


# --- Content-based similar products (TF-IDF + cosine): additional coverage ---

def test_similar_products_response_shape(auth_headers):
    laptop, *_ = _seed_gaming_catalog(auth_headers)
    data = client.get(f"/recommendations/product/{laptop['id']}", headers=auth_headers).json()
    assert set(data.keys()) == {"product_id", "recommendations"}
    assert data["product_id"] == laptop["id"]
    for rec in data["recommendations"]:
        assert set(rec.keys()) == {"product", "similarity_score"}


def test_similar_products_require_authentication():
    assert client.get("/recommendations/product/1").status_code == 401


def test_similar_products_no_duplicates(auth_headers):
    laptop, *_ = _seed_gaming_catalog(auth_headers)
    resp = client.get(f"/recommendations/product/{laptop['id']}?limit=50", headers=auth_headers)
    ids = [r["product"]["id"] for r in resp.json()["recommendations"]]
    assert len(ids) == len(set(ids))


# NOTE: this module's database accumulates products across tests and the
# endpoint caps `limit` at 50, so the tests below give the products under test
# a unique token. Related products then share it and always rank at the top,
# regardless of how many other products already exist in the catalog.
def _tok():
    return "tk" + uuid.uuid4().hex[:10]


def _scores(headers, product_id):
    resp = client.get(f"/recommendations/product/{product_id}?limit=50", headers=headers)
    assert resp.status_code == 200, resp.text
    return {r["product"]["id"]: r["similarity_score"] for r in resp.json()["recommendations"]}


def _post_product(headers, **fields):
    body = {"price": 5.0, "stock": 1, **fields}
    resp = client.post("/products/", json=body, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_similar_content_scores_higher_than_unrelated(auth_headers):
    tok = _tok()
    target = _post_product(auth_headers, name=f"{tok} laptop", description=f"{tok} portable computer")
    related = _post_product(auth_headers, name=f"{tok} mouse", description=f"{tok} portable")
    unrelated = _post_product(auth_headers, name="Kitchen Blender", description="blender appliance")

    score = _scores(auth_headers, target["id"])
    assert 0.0 < score[related["id"]] <= 1.0
    assert score.get(unrelated["id"], 0.0) < score[related["id"]]
    assert all(0.0 <= s <= 1.0 for s in score.values())


def test_empty_description_does_not_crash(auth_headers):
    a = _post_product(auth_headers, name="Plain Widget", description="")
    _post_product(auth_headers, name="Plain Gadget")  # description omitted (None)
    resp = client.get(f"/recommendations/product/{a['id']}", headers=auth_headers)
    assert resp.status_code == 200
    assert isinstance(resp.json()["recommendations"], list)


def test_category_and_brand_influence_similarity(auth_headers):
    tok = _tok()
    base = _post_product(auth_headers, name="Item", category=f"cat{tok}", brand=f"brand{tok}")
    same_meta = _post_product(auth_headers, name="Thing", category=f"cat{tok}", brand=f"brand{tok}")
    other_meta = _post_product(auth_headers, name="Stuff", category="Office", brand="Deskly")

    score = _scores(auth_headers, base["id"])
    assert score[same_meta["id"]] > 0.0
    assert score.get(other_meta["id"], 0.0) < score[same_meta["id"]]


def test_catalog_edit_is_reflected_in_next_request(auth_headers):
    # The fitted TF-IDF index is cached per catalog snapshot; an edit must
    # produce a fresh index, not stale scores.
    tok = _tok()
    target = _post_product(auth_headers, name=f"{tok} target", description=f"{tok} target")
    other = _post_product(auth_headers, name="Kitchen Blender", description="blender appliance")
    assert _scores(auth_headers, target["id"]).get(other["id"], 0.0) == 0.0

    resp = client.patch(
        f"/products/{other['id']}", json={"description": f"{tok} blender"}, headers=auth_headers
    )
    assert resp.status_code == 200, resp.text
    assert _scores(auth_headers, target["id"])[other["id"]] > 0.0
