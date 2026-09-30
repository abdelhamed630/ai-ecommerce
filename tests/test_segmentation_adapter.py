import os

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.database import Base
from ml.segmentation import features
from ml.segmentation.adapters import compute_rfm_from_orders, get_completed_order_transactions
from models.cart import Cart, CartItem  # noqa: F401  (registers all mappers via Base)
from models.interaction import ProductInteraction  # noqa: F401
from models.order import Order, OrderItem, OrderStatus
from models.payment import Payment  # noqa: F401
from models.product import Product
from models.user import User

_TEST_DB_PATH = "./test_segmentation_adapter.db"
_engine = create_engine(f"sqlite:///{_TEST_DB_PATH}", connect_args={"check_same_thread": False})
_TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


@pytest.fixture(autouse=True)
def _isolated_db():
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)
    _engine.dispose()  # release pooled connections before deleting the file
    if os.path.exists(_TEST_DB_PATH):
        os.remove(_TEST_DB_PATH)


@pytest.fixture
def db():
    session = _TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


def _make_user(db, email: str) -> User:
    user = User(email=email, hashed_password="hashed", full_name="Test User")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _make_product(db, name: str, price: float, stock: int = 100) -> Product:
    product = Product(name=name, description="test", price=price, stock=stock)
    db.add(product)
    db.commit()
    db.refresh(product)
    return product


def _make_order(db, user: User, status: OrderStatus, items: list) -> Order:
    """`items` is a list of (product, quantity, historical_price) tuples."""
    total = sum(qty * price for _, qty, price in items)
    order = Order(user_id=user.id, status=status, total_price=total)
    db.add(order)
    db.commit()
    db.refresh(order)

    for product, qty, price in items:
        db.add(
            OrderItem(
                order_id=order.id,
                product_id=product.id,
                product_name=product.name,
                product_price=price,
                quantity=qty,
                subtotal=qty * price,
            )
        )
    db.commit()
    db.refresh(order)
    return order


# ---------------------------------------------------------------------------
# 1-3. Purchase eligibility by status
# ---------------------------------------------------------------------------

def test_completed_order_is_included(db):
    user = _make_user(db, "completed@example.com")
    product = _make_product(db, "Widget", price=10.0)
    _make_order(db, user, OrderStatus.COMPLETED, [(product, 2, 10.0)])

    transactions = get_completed_order_transactions(db)
    assert len(transactions) == 1
    assert transactions.iloc[0]["CustomerID"] == user.id


def test_cancelled_order_is_excluded(db):
    user = _make_user(db, "cancelled@example.com")
    product = _make_product(db, "Widget", price=10.0)
    _make_order(db, user, OrderStatus.CANCELLED, [(product, 2, 10.0)])

    transactions = get_completed_order_transactions(db)
    assert transactions.empty


def test_pending_order_is_excluded(db):
    user = _make_user(db, "pending@example.com")
    product = _make_product(db, "Widget", price=10.0)
    _make_order(db, user, OrderStatus.PENDING, [(product, 2, 10.0)])

    transactions = get_completed_order_transactions(db)
    assert transactions.empty


def test_confirmed_order_is_excluded(db):
    # CONFIRMED is deliberately not treated as a final purchase (see
    # adapters.py docstring) — it can still be cancelled.
    user = _make_user(db, "confirmed@example.com")
    product = _make_product(db, "Widget", price=10.0)
    _make_order(db, user, OrderStatus.CONFIRMED, [(product, 2, 10.0)])

    transactions = get_completed_order_transactions(db)
    assert transactions.empty


# ---------------------------------------------------------------------------
# 4-6. Multi-item orders, quantity, historical price
# ---------------------------------------------------------------------------

def test_multiple_order_items_produce_multiple_rows(db):
    user = _make_user(db, "multi@example.com")
    product_a = _make_product(db, "Item A", price=100.0)
    product_b = _make_product(db, "Item B", price=300.0)
    _make_order(
        db, user, OrderStatus.COMPLETED, [(product_a, 2, 100.0), (product_b, 1, 300.0)]
    )

    transactions = get_completed_order_transactions(db)
    assert len(transactions) == 2
    assert transactions["InvoiceNo"].nunique() == 1  # same order


def test_quantity_mapped_correctly(db):
    user = _make_user(db, "qty@example.com")
    product = _make_product(db, "Widget", price=10.0)
    _make_order(db, user, OrderStatus.COMPLETED, [(product, 7, 10.0)])

    transactions = get_completed_order_transactions(db)
    assert transactions.iloc[0]["Quantity"] == 7


def test_historical_price_used_not_current_product_price(db):
    user = _make_user(db, "historical@example.com")
    product = _make_product(db, "Widget", price=10.0)
    # Order was placed when the price was 10.0 (snapshot)...
    _make_order(db, user, OrderStatus.COMPLETED, [(product, 1, 10.0)])
    # ...but the product's live price has since changed.
    product.price = 999.0
    db.commit()

    transactions = get_completed_order_transactions(db)
    assert transactions.iloc[0]["UnitPrice"] == 10.0  # snapshot, not 999.0


# ---------------------------------------------------------------------------
# 7-9. Field mapping
# ---------------------------------------------------------------------------

def test_order_id_maps_to_invoice_no(db):
    user = _make_user(db, "invoice@example.com")
    product = _make_product(db, "Widget", price=10.0)
    order = _make_order(db, user, OrderStatus.COMPLETED, [(product, 1, 10.0)])

    transactions = get_completed_order_transactions(db)
    assert transactions.iloc[0]["InvoiceNo"] == order.id


def test_user_id_maps_to_customer_id(db):
    user = _make_user(db, "custid@example.com")
    product = _make_product(db, "Widget", price=10.0)
    _make_order(db, user, OrderStatus.COMPLETED, [(product, 1, 10.0)])

    transactions = get_completed_order_transactions(db)
    assert transactions.iloc[0]["CustomerID"] == user.id


def test_created_at_maps_to_invoice_date(db):
    user = _make_user(db, "date@example.com")
    product = _make_product(db, "Widget", price=10.0)
    order = _make_order(db, user, OrderStatus.COMPLETED, [(product, 1, 10.0)])

    transactions = get_completed_order_transactions(db)
    invoice_date = pd.to_datetime(transactions.iloc[0]["InvoiceDate"])
    assert invoice_date is not None
    # created_at is server-generated; just confirm it round-trips to a real timestamp
    assert not pd.isna(invoice_date)


# ---------------------------------------------------------------------------
# 10-12. Multiple orders / RFM integration
# ---------------------------------------------------------------------------

def test_multiple_orders_same_user_produce_multiple_invoice_nos(db):
    user = _make_user(db, "repeat@example.com")
    product = _make_product(db, "Widget", price=10.0)
    order_1 = _make_order(db, user, OrderStatus.COMPLETED, [(product, 1, 10.0)])
    order_2 = _make_order(db, user, OrderStatus.COMPLETED, [(product, 1, 10.0)])

    transactions = get_completed_order_transactions(db)
    assert set(transactions["InvoiceNo"]) == {order_1.id, order_2.id}


def test_frequency_correct_after_compute_rfm(db):
    user = _make_user(db, "freq@example.com")
    product = _make_product(db, "Widget", price=10.0)
    _make_order(db, user, OrderStatus.COMPLETED, [(product, 1, 10.0)])
    _make_order(db, user, OrderStatus.COMPLETED, [(product, 1, 10.0)])
    _make_order(db, user, OrderStatus.COMPLETED, [(product, 1, 10.0)])

    rfm = compute_rfm_from_orders(db)
    row = rfm[rfm["CustomerID"] == user.id].iloc[0]
    assert row["Frequency"] == 3


def test_monetary_correct_after_compute_rfm(db):
    user = _make_user(db, "monetary@example.com")
    product_a = _make_product(db, "Item A", price=100.0)
    product_b = _make_product(db, "Item B", price=300.0)
    _make_order(
        db, user, OrderStatus.COMPLETED, [(product_a, 2, 100.0), (product_b, 1, 300.0)]
    )

    rfm = compute_rfm_from_orders(db)
    row = rfm[rfm["CustomerID"] == user.id].iloc[0]
    assert row["Monetary"] == pytest.approx(2 * 100.0 + 1 * 300.0)  # 500.0


# ---------------------------------------------------------------------------
# 13-14. Empty data + reference date
# ---------------------------------------------------------------------------

def test_empty_purchase_history_handled_safely(db):
    # No orders at all in this isolated db.
    transactions = get_completed_order_transactions(db)
    assert transactions.empty
    assert list(transactions.columns) == ["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]

    rfm = compute_rfm_from_orders(db)
    assert rfm.empty
    assert list(rfm.columns) == ["CustomerID", "Recency", "Frequency", "Monetary"]


def test_reference_date_passed_through_correctly(db):
    user = _make_user(db, "refdate@example.com")
    product = _make_product(db, "Widget", price=10.0)
    _make_order(db, user, OrderStatus.COMPLETED, [(product, 1, 10.0)])

    transactions = get_completed_order_transactions(db)
    invoice_date = pd.to_datetime(transactions.iloc[0]["InvoiceDate"])

    reference_date = invoice_date + pd.Timedelta(days=30)
    rfm = compute_rfm_from_orders(db, reference_date=reference_date)
    row = rfm[rfm["CustomerID"] == user.id].iloc[0]
    assert row["Recency"] == 30

    # Sanity check: a different reference_date gives a different Recency,
    # proving it's actually being used rather than silently ignored.
    rfm_2 = compute_rfm_from_orders(db, reference_date=invoice_date + pd.Timedelta(days=5))
    row_2 = rfm_2[rfm_2["CustomerID"] == user.id].iloc[0]
    assert row_2["Recency"] == 5


def test_adapter_does_not_reimplement_rfm_math(db, monkeypatch):
    """Confirms compute_rfm_from_orders actually delegates to features.compute_rfm
    rather than recalculating RFM itself."""
    calls = []
    original_compute_rfm = features.compute_rfm

    def _spy(df, reference_date=None):
        calls.append(df)
        return original_compute_rfm(df, reference_date=reference_date)

    monkeypatch.setattr("ml.segmentation.adapters.compute_rfm", _spy)

    user = _make_user(db, "spy@example.com")
    product = _make_product(db, "Widget", price=10.0)
    _make_order(db, user, OrderStatus.COMPLETED, [(product, 1, 10.0)])

    compute_rfm_from_orders(db)
    assert len(calls) == 1
