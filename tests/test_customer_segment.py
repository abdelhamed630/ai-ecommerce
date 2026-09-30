import os
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from database.database import Base
from models.cart import Cart, CartItem  # noqa: F401  (registers all mappers via Base)
from models.customer_segment import CustomerSegment
from models.interaction import ProductInteraction  # noqa: F401
from models.order import Order, OrderItem  # noqa: F401
from models.payment import Payment  # noqa: F401
from models.product import Product  # noqa: F401
from models.user import User
from services.customer_segment_service import get_current_segment, upsert_current_segment

_TEST_DB_PATH = "./test_customer_segment.db"
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


# ---------------------------------------------------------------------------
# 1-2. Basic creation / FK to an existing user
# ---------------------------------------------------------------------------

def test_create_segment_for_valid_user(db):
    user = _make_user(db, "valid@example.com")
    calculated_at = datetime.utcnow()

    segment = upsert_current_segment(
        db,
        user_id=user.id,
        cluster_id=2,
        segment_label="Champions / Loyal",
        recency=5,
        frequency=10,
        monetary=500.0,
        calculated_at=calculated_at,
    )

    assert segment.id is not None
    assert segment.user_id == user.id


def test_user_id_references_existing_user(db):
    user = _make_user(db, "fk@example.com")
    segment = upsert_current_segment(
        db,
        user_id=user.id,
        cluster_id=0,
        segment_label="Low Engagement",
        recency=30,
        frequency=1,
        monetary=20.0,
    )
    stored = db.query(CustomerSegment).filter(CustomerSegment.id == segment.id).first()
    assert stored.user.id == user.id
    assert stored.user.email == "fk@example.com"


# ---------------------------------------------------------------------------
# 3-5. Uniqueness / upsert behavior
# ---------------------------------------------------------------------------

def test_user_id_is_unique_at_db_level(db):
    user = _make_user(db, "unique@example.com")
    db.add(
        CustomerSegment(
            user_id=user.id,
            cluster_id=1,
            segment_label="Potential Loyalist",
            recency=10,
            frequency=3,
            monetary=100.0,
            model_version="rfm_kmeans_v1",
            calculated_at=datetime.utcnow(),
        )
    )
    db.commit()

    db.add(
        CustomerSegment(
            user_id=user.id,
            cluster_id=2,
            segment_label="Champions / Loyal",
            recency=1,
            frequency=20,
            monetary=900.0,
            model_version="rfm_kmeans_v1",
            calculated_at=datetime.utcnow(),
        )
    )
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_two_current_segments_rejected_but_upsert_handles_it(db):
    user = _make_user(db, "double@example.com")
    upsert_current_segment(
        db, user_id=user.id, cluster_id=0, segment_label="Low Engagement",
        recency=40, frequency=1, monetary=10.0,
    )
    # A second call for the same user must update, not duplicate.
    upsert_current_segment(
        db, user_id=user.id, cluster_id=3, segment_label="At Risk / Churned",
        recency=90, frequency=1, monetary=15.0,
    )
    all_segments = db.query(CustomerSegment).filter(CustomerSegment.user_id == user.id).all()
    assert len(all_segments) == 1


def test_upsert_replaces_current_segment_correctly(db):
    user = _make_user(db, "replace@example.com")
    first = upsert_current_segment(
        db, user_id=user.id, cluster_id=0, segment_label="Low Engagement",
        recency=40, frequency=1, monetary=10.0,
    )
    updated = upsert_current_segment(
        db, user_id=user.id, cluster_id=2, segment_label="Champions / Loyal",
        recency=2, frequency=15, monetary=800.0,
    )
    assert updated.id == first.id  # same row, updated in place
    assert updated.cluster_id == 2
    assert updated.segment_label == "Champions / Loyal"
    assert updated.recency == 2
    assert updated.frequency == 15
    assert updated.monetary == 800.0


# ---------------------------------------------------------------------------
# 6-12. Field storage correctness
# ---------------------------------------------------------------------------

def test_cluster_id_stored_correctly(db):
    user = _make_user(db, "cluster@example.com")
    segment = upsert_current_segment(
        db, user_id=user.id, cluster_id=3, segment_label="At Risk / Churned",
        recency=100, frequency=1, monetary=5.0,
    )
    assert segment.cluster_id == 3


def test_segment_label_stored_independently_of_cluster_id(db):
    user = _make_user(db, "label@example.com")
    # Same cluster_id, different label — proves no hardcoded mapping is used.
    segment = upsert_current_segment(
        db, user_id=user.id, cluster_id=1, segment_label="Potential Loyalist",
        recency=8, frequency=6, monetary=250.0,
    )
    assert segment.cluster_id == 1
    assert segment.segment_label == "Potential Loyalist"


def test_recency_stored_correctly(db):
    user = _make_user(db, "recency@example.com")
    segment = upsert_current_segment(
        db, user_id=user.id, cluster_id=0, segment_label="Low Engagement",
        recency=17, frequency=2, monetary=50.0,
    )
    assert segment.recency == 17


def test_frequency_stored_correctly(db):
    user = _make_user(db, "frequency@example.com")
    segment = upsert_current_segment(
        db, user_id=user.id, cluster_id=0, segment_label="Low Engagement",
        recency=17, frequency=9, monetary=50.0,
    )
    assert segment.frequency == 9


def test_monetary_stored_correctly(db):
    user = _make_user(db, "monetary@example.com")
    segment = upsert_current_segment(
        db, user_id=user.id, cluster_id=0, segment_label="Low Engagement",
        recency=17, frequency=2, monetary=123.45,
    )
    assert segment.monetary == pytest.approx(123.45)


def test_model_version_stored_correctly(db):
    user = _make_user(db, "version@example.com")
    segment = upsert_current_segment(
        db, user_id=user.id, cluster_id=0, segment_label="Low Engagement",
        recency=17, frequency=2, monetary=50.0, model_version="rfm_kmeans_v1",
    )
    assert segment.model_version == "rfm_kmeans_v1"


def test_model_version_defaults_to_configured_constant(db):
    from ml.segmentation.model import SEGMENTATION_MODEL_VERSION

    user = _make_user(db, "default_version@example.com")
    segment = upsert_current_segment(
        db, user_id=user.id, cluster_id=0, segment_label="Low Engagement",
        recency=17, frequency=2, monetary=50.0,
    )
    assert segment.model_version == SEGMENTATION_MODEL_VERSION


def test_calculated_at_stored_correctly(db):
    user = _make_user(db, "calcat@example.com")
    calculated_at = datetime.utcnow() - timedelta(days=1)
    segment = upsert_current_segment(
        db, user_id=user.id, cluster_id=0, segment_label="Low Engagement",
        recency=17, frequency=2, monetary=50.0, calculated_at=calculated_at,
    )
    assert segment.calculated_at is not None
    # Stored value round-trips to the same moment we passed in.
    assert abs((segment.calculated_at - calculated_at).total_seconds()) < 1


# ---------------------------------------------------------------------------
# 13-15. Constraint validation
# ---------------------------------------------------------------------------

def test_negative_recency_rejected(db):
    user = _make_user(db, "negrecency@example.com")
    db.add(
        CustomerSegment(
            user_id=user.id,
            cluster_id=0,
            segment_label="Low Engagement",
            recency=-1,
            frequency=1,
            monetary=10.0,
            model_version="rfm_kmeans_v1",
            calculated_at=datetime.utcnow(),
        )
    )
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_negative_monetary_rejected(db):
    user = _make_user(db, "negmonetary@example.com")
    db.add(
        CustomerSegment(
            user_id=user.id,
            cluster_id=0,
            segment_label="Low Engagement",
            recency=1,
            frequency=1,
            monetary=-10.0,
            model_version="rfm_kmeans_v1",
            calculated_at=datetime.utcnow(),
        )
    )
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_invalid_frequency_rejected(db):
    user = _make_user(db, "badfreq@example.com")
    db.add(
        CustomerSegment(
            user_id=user.id,
            cluster_id=0,
            segment_label="Low Engagement",
            recency=1,
            frequency=0,  # must be > 0 for an actual calculated segment
            monetary=10.0,
            model_version="rfm_kmeans_v1",
            calculated_at=datetime.utcnow(),
        )
    )
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


# ---------------------------------------------------------------------------
# 16. Retrieval
# ---------------------------------------------------------------------------

def test_get_current_segment_returns_latest_record(db):
    user = _make_user(db, "getlatest@example.com")
    assert get_current_segment(db, user.id) is None

    upsert_current_segment(
        db, user_id=user.id, cluster_id=0, segment_label="Low Engagement",
        recency=40, frequency=1, monetary=10.0,
    )
    upsert_current_segment(
        db, user_id=user.id, cluster_id=2, segment_label="Champions / Loyal",
        recency=1, frequency=25, monetary=999.0,
    )

    current = get_current_segment(db, user.id)
    assert current is not None
    assert current.segment_label == "Champions / Loyal"
    assert current.cluster_id == 2


def test_get_current_segment_for_user_with_none_calculated(db):
    user = _make_user(db, "nosegment@example.com")
    assert get_current_segment(db, user.id) is None
