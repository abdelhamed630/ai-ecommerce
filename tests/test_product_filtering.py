"""GET /products/ filtering: q, category, min_price, max_price, skip, limit (database-side)."""
import pytest
from fastapi.testclient import TestClient

from api.products import MAX_LIST_LIMIT
from database.database import SessionLocal
from main import app
from models.product import Product
from tests.helpers_chatbot import token

client = TestClient(app)


@pytest.fixture
def seed():
    ids = []

    def _seed(**fields):
        fields.setdefault("price", 10.0)
        fields.setdefault("stock", 5)
        db = SessionLocal()
        try:
            product = Product(**fields)
            db.add(product)
            db.commit()
            db.refresh(product)
            ids.append(product.id)
            return product.id
        finally:
            db.close()

    yield _seed
    if ids:
        db = SessionLocal()
        try:
            db.query(Product).filter(Product.id.in_(ids)).delete(synchronize_session=False)
            db.commit()
        finally:
            db.close()


def _ids(response):
    assert response.status_code == 200, response.text
    return [p["id"] for p in response.json()]


def test_unauthenticated_listing_still_works_without_parameters():
    r = client.get("/products/")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_default_limit_is_unchanged_and_skip_limit_still_work(seed):
    t = token()
    ids = [seed(name=f"Item {t} {i}") for i in range(5)]
    assert _ids(client.get(f"/products/?q={t}")) == ids
    assert _ids(client.get(f"/products/?q={t}&skip=1&limit=2")) == ids[1:3]
    assert _ids(client.get(f"/products/?q={t}&limit=0")) == []


def test_pages_are_stable_and_ordered_by_id(seed):
    t = token()
    ids = [seed(name=f"Page {t} {i}") for i in range(4)]
    first = _ids(client.get(f"/products/?q={t}&limit=2"))
    second = _ids(client.get(f"/products/?q={t}&skip=2&limit=2"))
    assert first + second == ids


def test_limit_above_the_maximum_is_rejected():
    assert client.get(f"/products/?limit={MAX_LIST_LIMIT}").status_code == 200
    assert client.get(f"/products/?limit={MAX_LIST_LIMIT + 1}").status_code == 422


@pytest.mark.parametrize("params", ["skip=-1", "limit=-1", "skip=abc", "limit=abc"])
def test_invalid_pagination_is_rejected(params):
    assert client.get(f"/products/?{params}").status_code == 422


def test_q_is_case_insensitive_and_matches_all_words(seed):
    t = token()
    a = seed(name=f"Gaming Laptop {t}")
    seed(name=f"Gaming Mouse {t}")
    assert _ids(client.get(f"/products/?q=LAPTOP%20{t}")) == [a]
    assert _ids(client.get(f"/products/?q=gaming%20{t.upper()}")) != []
    assert len(_ids(client.get(f"/products/?q=gaming%20{t}"))) == 2


def test_q_searches_description_brand_and_category(seed):
    t = token()
    by_desc = seed(name=f"A {t}", description="fast charging cable")
    by_brand = seed(name=f"B {t}", brand="Anker")
    by_cat = seed(name=f"C {t}", category="Accessories")
    assert _ids(client.get(f"/products/?q=charging%20{t}")) == [by_desc]
    assert _ids(client.get(f"/products/?q=anker%20{t}")) == [by_brand]
    assert _ids(client.get(f"/products/?q=accessories%20{t}")) == [by_cat]


def test_q_keeps_every_word_including_common_ones(seed):
    t = token()
    pid = seed(name=f"The Best Do {t}")
    assert _ids(client.get(f"/products/?q=do%20{t}")) == [pid]  # not dropped as a stopword


def test_blank_q_is_ignored():
    assert _ids(client.get("/products/?q=%20%20")) == _ids(client.get("/products/"))


def test_q_wildcards_are_literal(seed):
    t = token()
    seed(name=f"Plain {t}")
    pid = seed(name=f"100% cotton {t}")
    # "%" is not a word character, so it is dropped by the tokenizer; "_" is kept and must be
    # matched literally (not as the single-character LIKE wildcard).
    assert _ids(client.get(f"/products/?q=100%25%20{t}")) == [pid]
    assert _ids(client.get(f"/products/?q=_%20{t}")) == []


def test_sql_injection_in_every_filter_is_harmless(seed):
    t = token()
    pid = seed(name=f"Safe {t}", category="Electronics")
    payload = "'; DROP TABLE products; --"
    for param in ("q", "category"):
        r = client.get("/products/", params={param: payload})
        assert r.status_code == 200 and r.json() == []
    assert client.get(f"/products/{pid}").status_code == 200  # the table is intact


def test_category_filter_is_case_insensitive_and_exact(seed):
    t = token()
    cat = f"Electronics{t}"
    a = seed(name=f"One {t}", category=cat)
    seed(name=f"Two {t}", category=f"{cat}Extra")
    seed(name=f"Three {t}")
    assert _ids(client.get(f"/products/?category={cat.lower()}")) == [a]
    assert _ids(client.get(f"/products/?category={cat.upper()}")) == [a]


def test_price_bounds_are_inclusive(seed):
    t = token()
    low = seed(name=f"Low {t}", price=499.99)
    edge_low = seed(name=f"EdgeLow {t}", price=500.0)
    edge_high = seed(name=f"EdgeHigh {t}", price=5000.0)
    high = seed(name=f"High {t}", price=5000.01)
    assert _ids(client.get(f"/products/?q={t}&min_price=500")) == [edge_low, edge_high, high]
    assert _ids(client.get(f"/products/?q={t}&max_price=5000")) == [low, edge_low, edge_high]
    assert _ids(client.get(f"/products/?q={t}&min_price=500&max_price=5000")) == [edge_low, edge_high]
    assert _ids(client.get(f"/products/?q={t}&min_price=500&max_price=500")) == [edge_low]


def test_all_filters_combine(seed):
    t = token()
    cat = f"Electronics{t}"
    hit = seed(name=f"Laptop {t}", category=cat, price=1200.0)
    seed(name=f"Laptop cheap {t}", category=cat, price=100.0)
    seed(name=f"Laptop other {t}", category=f"Other{t}", price=1200.0)
    seed(name=f"Phone {t}", category=cat, price=1200.0)
    params = {"q": f"laptop {t}", "category": cat, "min_price": 500, "max_price": 5000}
    assert _ids(client.get("/products/", params=params)) == [hit]


@pytest.mark.parametrize(
    "params",
    ["min_price=abc", "max_price=abc", "min_price=-1", "max_price=-0.5", "min_price=nan", "max_price=inf",
     "min_price=1e400", "min_price=1e13"],
)
def test_invalid_price_values_are_rejected(params):
    assert client.get(f"/products/?{params}").status_code == 422


def test_min_price_greater_than_max_price_is_rejected():
    r = client.get("/products/?min_price=100&max_price=50")
    assert r.status_code == 422
    assert "min_price" in r.json()["detail"]


def test_too_long_q_or_category_is_rejected():
    assert client.get("/products/", params={"q": "x" * 101}).status_code == 422
    assert client.get("/products/", params={"category": "x" * 101}).status_code == 422


@pytest.mark.parametrize(
    "stored, typed",
    [("سماعة", "سماعه"), ("سماعه", "سماعة"), ("أحذية", "احذيه"), ("مستشفى", "مستشفي")],
)
def test_arabic_spelling_variants_match_in_q(seed, stored, typed):
    t = token()
    pid = seed(name=f"{stored} {t}")
    assert _ids(client.get("/products/", params={"q": f"{typed} {t}"})) == [pid]


def test_arabic_category_matches_spelling_variants(seed):
    t = token()
    pid = seed(name=f"Item {t}", category="أحذية")
    assert _ids(client.get("/products/", params={"category": "احذيه", "q": t})) == [pid]


def test_exact_names_english_and_mixed_queries_still_work(seed):
    t = token()
    exact = seed(name=f"سماعة Sony WH-1000XM5 {t}")
    english = seed(name=f"iPhone 15 {t}")
    assert _ids(client.get("/products/", params={"q": f"سماعة Sony {t}"})) == [exact]
    assert _ids(client.get("/products/", params={"q": f"سماعه sony {t}"})) == [exact]  # mixed + variant
    assert _ids(client.get("/products/", params={"q": f"IPHONE {t}"})) == [english]
    assert _ids(client.get("/products/", params={"q": f"سماعة Sony WH-1000XM5 {t}"})) == [exact]


def test_filtering_does_not_load_the_whole_table(seed, monkeypatch):
    """The page size is applied in SQL: the statement carries LIMIT/OFFSET and the filters."""
    from sqlalchemy import event

    from database.database import engine

    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    t = token()
    seed(name=f"Sql {t}", price=700.0)
    event.listen(engine, "before_cursor_execute", capture)
    try:
        client.get(f"/products/?q={t}&min_price=500&limit=3")
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    sql = " ".join(s for s in statements if "FROM products" in s).upper()
    assert "LIMIT" in sql and "PRICE >=" in sql and "LIKE" in sql
