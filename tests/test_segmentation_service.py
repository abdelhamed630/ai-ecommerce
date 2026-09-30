import os
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.database import Base
from ml.segmentation import interpretation as interpretation_module
from ml.segmentation import model as model_module
from ml.segmentation.model import SEGMENTATION_MODEL_VERSION
from models.cart import Cart, CartItem  # noqa: F401  (registers all mappers via Base)
from models.customer_segment import CustomerSegment
from models.interaction import InteractionType, ProductInteraction
from models.order import Order, OrderItem, OrderStatus
from models.payment import Payment  # noqa: F401
from models.product import Product
from models.user import User
from services.customer_segment_service import get_current_segment
from services.segmentation_service import (
    InsufficientSegmentationDataError,
    run_customer_segmentation,
)
import services.segmentation_service as segmentation_service

_TEST_DB_PATH = "./test_segmentation_service.db"
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_user(db, email: str) -> User:
    user = User(email=email, hashed_password="hashed", full_name="Test User")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _make_product(db, name: str, price: float, stock: int = 1000) -> Product:
    product = Product(name=name, description="test", price=price, stock=stock)
    db.add(product)
    db.commit()
    db.refresh(product)
    return product


def _make_order(
    db, user: User, status: OrderStatus, items: list, created_at: datetime = None
) -> Order:
    """`items` is a list of (product, quantity, historical_price) tuples.

    If `created_at` is given, the order's created_at (InvoiceDate) is
    overridden after insert so tests can control Recency deterministically
    — the default server_default=func.now() is fine for tests that don't
    care about specific recency values.
    """
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

    if created_at is not None:
        order.created_at = created_at
        db.commit()

    db.refresh(order)
    return order


def _make_interaction(db, user: User, product: Product, interaction_type: InteractionType):
    interaction = ProductInteraction(
        user_id=user.id, product_id=product.id, interaction_type=interaction_type
    )
    db.add(interaction)
    db.commit()
    return interaction


_NOW = datetime.utcnow()


def _days_ago(n: int) -> datetime:
    return _NOW - timedelta(days=n)


def _make_group_a_champion(db, product, email):
    """Frequent orders, high spending, very recent activity."""
    user = _make_user(db, email)
    for i in range(4):
        _make_order(
            db, user, OrderStatus.COMPLETED, [(product, 1, 200.0)], created_at=_days_ago(1 + i)
        )
    return user


def _make_group_b_moderate(db, product, email):
    """Moderate orders, moderate spending, moderate recency."""
    user = _make_user(db, email)
    for i in range(2):
        _make_order(
            db, user, OrderStatus.COMPLETED, [(product, 1, 50.0)], created_at=_days_ago(20 + i)
        )
    return user


def _make_group_c_low_recent(db, product, email):
    """One order, low spend, recent activity."""
    user = _make_user(db, email)
    _make_order(db, user, OrderStatus.COMPLETED, [(product, 1, 10.0)], created_at=_days_ago(2))
    return user


def _make_group_d_low_old(db, product, email):
    """One order, low spend, old (churned) activity."""
    user = _make_user(db, email)
    _make_order(db, user, OrderStatus.COMPLETED, [(product, 1, 10.0)], created_at=_days_ago(120))
    return user


def _seed_four_groups(db):
    product = _make_product(db, "Widget", price=10.0)
    users = {
        "a1": _make_group_a_champion(db, product, "a1@example.com"),
        "a2": _make_group_a_champion(db, product, "a2@example.com"),
        "b1": _make_group_b_moderate(db, product, "b1@example.com"),
        "b2": _make_group_b_moderate(db, product, "b2@example.com"),
        "c1": _make_group_c_low_recent(db, product, "c1@example.com"),
        "c2": _make_group_c_low_recent(db, product, "c2@example.com"),
        "d1": _make_group_d_low_old(db, product, "d1@example.com"),
        "d2": _make_group_d_low_old(db, product, "d2@example.com"),
    }
    return users, product


# ---------------------------------------------------------------------------
# 1-2. Empty database
# ---------------------------------------------------------------------------

def test_empty_database_returns_zero_summary_without_error(db):
    result = run_customer_segmentation(db)
    assert result["customers_processed"] == 0
    assert result["segments"] == {}
    assert result["model_version"] == SEGMENTATION_MODEL_VERSION


def test_empty_database_creates_no_segment_rows_and_does_not_train(db, monkeypatch):
    calls = []
    original_train = model_module.train_segmentation_model

    def _spy(*args, **kwargs):
        calls.append(1)
        return original_train(*args, **kwargs)

    monkeypatch.setattr("services.segmentation_service.train_segmentation_model", _spy)

    run_customer_segmentation(db)

    assert calls == []  # training never attempted
    assert db.query(CustomerSegment).count() == 0


# ---------------------------------------------------------------------------
# 3-5. Insufficient data (K=4 configured)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_customers", [1, 2, 3])
def test_insufficient_customers_raises_clear_error(db, n_customers):
    product = _make_product(db, "Widget", price=10.0)
    for i in range(n_customers):
        user = _make_user(db, f"insufficient{i}@example.com")
        _make_order(db, user, OrderStatus.COMPLETED, [(product, 1, 10.0)])

    with pytest.raises(InsufficientSegmentationDataError):
        run_customer_segmentation(db)

    # No partial/fake segments were created.
    assert db.query(CustomerSegment).count() == 0


def test_insufficient_data_does_not_reduce_k(db, monkeypatch):
    product = _make_product(db, "Widget", price=10.0)
    for i in range(2):
        user = _make_user(db, f"nok{i}@example.com")
        _make_order(db, user, OrderStatus.COMPLETED, [(product, 1, 10.0)])

    calls = []
    monkeypatch.setattr(
        "services.segmentation_service.train_segmentation_model",
        lambda *a, **k: calls.append(k.get("n_clusters")),
    )

    with pytest.raises(InsufficientSegmentationDataError):
        run_customer_segmentation(db)

    assert calls == []  # train_segmentation_model was never even called


def test_four_or_more_customers_allows_training(db):
    _seed_four_groups(db)
    result = run_customer_segmentation(db)
    assert result["customers_processed"] == 8
    assert sum(result["segments"].values()) == 8


# ---------------------------------------------------------------------------
# 6-9. Core orchestration correctness
# ---------------------------------------------------------------------------

def test_segment_record_exists_for_every_eligible_user(db):
    users, _ = _seed_four_groups(db)
    run_customer_segmentation(db)

    for user in users.values():
        assert get_current_segment(db, user.id) is not None


def test_every_record_has_valid_cluster_id_and_non_empty_label(db):
    users, _ = _seed_four_groups(db)
    run_customer_segmentation(db)

    for user in users.values():
        segment = get_current_segment(db, user.id)
        assert isinstance(segment.cluster_id, int)
        assert 0 <= segment.cluster_id < 4
        assert isinstance(segment.segment_label, str)
        assert segment.segment_label.strip() != ""


def test_model_version_and_calculated_at_populated(db):
    users, _ = _seed_four_groups(db)
    before = datetime.utcnow()
    run_customer_segmentation(db)
    after = datetime.utcnow()

    for user in users.values():
        segment = get_current_segment(db, user.id)
        assert segment.model_version == SEGMENTATION_MODEL_VERSION
        assert segment.calculated_at is not None
        assert before - timedelta(seconds=2) <= segment.calculated_at <= after + timedelta(seconds=2)


def test_rfm_values_match_database_transactions(db):
    users, product = _seed_four_groups(db)
    run_customer_segmentation(db)

    # Group A champion: 4 completed orders of 200.0 each.
    segment_a1 = get_current_segment(db, users["a1"].id)
    assert segment_a1.frequency == 4
    assert segment_a1.monetary == pytest.approx(800.0)
    assert segment_a1.recency <= 2  # most recent order was 1 day ago

    # Group D: 1 completed order of 10.0, 120 days ago.
    segment_d1 = get_current_segment(db, users["d1"].id)
    assert segment_d1.frequency == 1
    assert segment_d1.monetary == pytest.approx(10.0)
    assert segment_d1.recency >= 100


# ---------------------------------------------------------------------------
# 10-11. Eligibility: who gets processed
# ---------------------------------------------------------------------------

def test_user_without_completed_purchase_not_processed(db):
    users, product = _seed_four_groups(db)
    no_purchase_user = _make_user(db, "nopurchase@example.com")

    run_customer_segmentation(db)

    assert get_current_segment(db, no_purchase_user.id) is None


def test_cancelled_orders_do_not_affect_results(db):
    users, product = _seed_four_groups(db)
    # a1 also has a large CANCELLED order that must not count.
    _make_order(
        db, users["a1"], OrderStatus.CANCELLED, [(product, 1, 100000.0)], created_at=_days_ago(1)
    )

    run_customer_segmentation(db)

    segment_a1 = get_current_segment(db, users["a1"].id)
    assert segment_a1.frequency == 4  # unchanged: still just the 4 COMPLETED orders
    assert segment_a1.monetary == pytest.approx(800.0)  # cancelled order's amount excluded


# ---------------------------------------------------------------------------
# 12-14. Interactions, price snapshots, multi-item orders
# ---------------------------------------------------------------------------

def test_product_interactions_do_not_affect_segmentation(db):
    users, product = _seed_four_groups(db)
    # Lots of browsing/cart activity for a1 that must not influence RFM.
    for _ in range(10):
        _make_interaction(db, users["a1"], product, InteractionType.VIEW)
    _make_interaction(db, users["a1"], product, InteractionType.CART_ADD)

    run_customer_segmentation(db)

    segment_a1 = get_current_segment(db, users["a1"].id)
    assert segment_a1.frequency == 4
    assert segment_a1.monetary == pytest.approx(800.0)


def test_product_price_change_after_order_does_not_affect_historical_monetary(db):
    users, product = _seed_four_groups(db)
    # Live price changes drastically after all orders were already placed.
    product.price = 999999.0
    db.commit()

    run_customer_segmentation(db)

    segment_a1 = get_current_segment(db, users["a1"].id)
    assert segment_a1.monetary == pytest.approx(800.0)  # snapshot price (200.0), not 999999.0


def test_multiple_orders_for_one_customer_increase_frequency(db):
    users, product = _seed_four_groups(db)
    run_customer_segmentation(db)
    segment_a1 = get_current_segment(db, users["a1"].id)
    assert segment_a1.frequency == 4  # 4 separate completed orders


def test_multiple_items_in_one_order_contribute_correctly_to_monetary(db):
    users, product = _seed_four_groups(db)
    product_b = _make_product(db, "Gadget", price=75.0)
    _make_order(
        db,
        users["b1"],
        OrderStatus.COMPLETED,
        [(product, 3, 50.0), (product_b, 2, 75.0)],
        created_at=_days_ago(5),
    )

    run_customer_segmentation(db)

    segment_b1 = get_current_segment(db, users["b1"].id)
    # b1 started with 2 orders of 50.0 (see _make_group_b_moderate) plus this
    # multi-item order: 3*50.0 + 2*75.0 = 300.0.
    expected_monetary = 50.0 + 50.0 + (3 * 50.0 + 2 * 75.0)
    assert segment_b1.monetary == pytest.approx(expected_monetary)
    assert segment_b1.frequency == 3


# ---------------------------------------------------------------------------
# 15. Interpretation is delegated, not duplicated/hardcoded
# ---------------------------------------------------------------------------

def test_orchestration_passes_cluster_profiles_into_interpretation_layer(db, monkeypatch):
    """Confirms run_customer_segmentation delegates labeling to
    ml.segmentation.interpretation.get_segment_label_map with the actual
    computed cluster profiles, rather than hardcoding cluster_id->label."""
    calls = []
    original = interpretation_module.get_segment_label_map

    def _spy(rfm_with_clusters, cluster_col="Cluster"):
        calls.append(rfm_with_clusters.copy())
        return original(rfm_with_clusters, cluster_col=cluster_col)

    monkeypatch.setattr("services.segmentation_service.interpretation.get_segment_label_map", _spy)

    users, _ = _seed_four_groups(db)
    run_customer_segmentation(db)

    assert len(calls) == 1
    passed_df = calls[0]
    assert "Cluster" in passed_df.columns
    for col in ("Recency", "Frequency", "Monetary"):
        assert col in passed_df.columns
    assert len(passed_df) == 8


def test_champion_and_churned_profiles_get_different_labels(db):
    """Sanity check that clearly different behavioral profiles end up with
    different (not asserted-by-id) segment labels — verifying labeling is
    behavior-driven, without asserting any specific cluster_id."""
    users, _ = _seed_four_groups(db)
    run_customer_segmentation(db)

    champion_label = get_current_segment(db, users["a1"].id).segment_label
    churned_label = get_current_segment(db, users["d1"].id).segment_label
    assert champion_label != churned_label


# ---------------------------------------------------------------------------
# 16. Transaction safety
# ---------------------------------------------------------------------------

def test_failure_partway_through_rolls_back_entire_batch(db, monkeypatch):
    users, _ = _seed_four_groups(db)

    original_upsert = segmentation_service.upsert_current_segment
    call_count = {"n": 0}

    def _flaky_upsert(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 5:
            raise RuntimeError("simulated failure mid-batch")
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr("services.segmentation_service.upsert_current_segment", _flaky_upsert)

    with pytest.raises(RuntimeError):
        run_customer_segmentation(db)

    # Nothing from the failed batch was left half-committed.
    assert db.query(CustomerSegment).count() == 0


def test_upsert_does_not_commit_individually_in_batch_mode(db, monkeypatch):
    """Verifies the batch relies on ONE commit at the end, not a commit per
    customer (see services/customer_segment_service.py's `commit` flag)."""
    users, _ = _seed_four_groups(db)

    commit_calls = {"n": 0}
    original_commit = db.commit

    def _counting_commit():
        commit_calls["n"] += 1
        return original_commit()

    monkeypatch.setattr(db, "commit", _counting_commit)

    run_customer_segmentation(db)

    assert commit_calls["n"] == 1
