"""datetime.utcnow() -> timezone-aware UTC, in the application code (SAFE batch).

Scope is exactly the three places that used it:
  * core/security.py              (JWT `exp` claim)
  * services/customer_segment_service.py  (default `calculated_at`)
  * services/segmentation_service.py       (run-level `calculated_at`)

The JWT structure and the stored values are unchanged: `exp` is still an integer
epoch, and `calculated_at` still lands in a DateTime(timezone=True) column.
"""

import ast
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from jose import JWTError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.config import ALLOWED_JWT_ALGORITHMS, settings
from core.security import create_access_token, decode_access_token
from database.database import Base
from models.cart import Cart, CartItem  # noqa: F401  (registers all mappers via Base)
from models.customer_segment import CustomerSegment  # noqa: F401
from models.interaction import ProductInteraction  # noqa: F401
from models.order import Order, OrderItem  # noqa: F401
from models.payment import Payment  # noqa: F401
from models.product import Product  # noqa: F401
from models.user import User
from services.customer_segment_service import get_current_segment, upsert_current_segment

ROOT = Path(__file__).resolve().parent.parent
APP_SOURCES = ["core", "services", "api", "models", "ml", "database", "tasks", "scripts", "alembic"]
APP_FILES = ["main.py", "celery_app.py"]


@pytest.fixture
def db():
    # Private in-memory database: touches neither the shared test DB nor any file.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _make_user(db, email="utc@example.com") -> User:
    user = User(email=email, hashed_password="hashed", full_name="UTC Test")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


# ------------------------------------------------------------ JWT (core/security.py)
def test_create_access_token_emits_no_utcnow_deprecation_warning():
    # On Python 3.12 the old implementation raised DeprecationWarning here.
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*utcnow.*", category=DeprecationWarning)
        assert create_access_token({"sub": "1"})


def test_access_token_exp_is_the_correct_utc_epoch():
    token = create_access_token({"sub": "1"})
    payload = decode_access_token(token)
    expected = _now_epoch() + settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    assert isinstance(payload["exp"], int)
    assert abs(payload["exp"] - expected) <= 5


def test_custom_expires_delta_is_honoured():
    payload = decode_access_token(create_access_token({"sub": "1"}, timedelta(minutes=5)))
    assert abs(payload["exp"] - (_now_epoch() + 300)) <= 5


def test_expired_token_is_rejected():
    token = create_access_token({"sub": "1"}, timedelta(seconds=-30))
    with pytest.raises(JWTError):
        decode_access_token(token)


def test_token_structure_is_unchanged():
    payload = decode_access_token(create_access_token({"sub": "42", "role": "customer"}))
    assert set(payload) == {"sub", "role", "exp"}
    assert payload["sub"] == "42" and payload["role"] == "customer"


@pytest.mark.parametrize("alg", ALLOWED_JWT_ALGORITHMS)
def test_tokens_round_trip_with_every_allowed_algorithm(monkeypatch, alg):
    monkeypatch.setattr(settings, "ALGORITHM", alg)
    payload = decode_access_token(create_access_token({"sub": "7"}))
    assert payload["sub"] == "7" and isinstance(payload["exp"], int)


# ------------------------------------------- customer_segment_service default timestamp
def test_upsert_default_calculated_at_is_timezone_aware_utc_before_commit(db):
    user = _make_user(db)
    segment = upsert_current_segment(
        db, user_id=user.id, cluster_id=0, segment_label="Champions / Loyal",
        recency=1, frequency=2, monetary=3.0, commit=False,
    )
    stamp = segment.calculated_at
    assert stamp.tzinfo is not None and stamp.utcoffset() == timedelta(0)
    assert abs((datetime.now(timezone.utc) - stamp).total_seconds()) < 5
    db.rollback()


def test_upsert_default_calculated_at_persists_the_current_utc_time(db):
    user = _make_user(db)
    upsert_current_segment(
        db, user_id=user.id, cluster_id=0, segment_label="Champions / Loyal",
        recency=1, frequency=2, monetary=3.0,
    )
    stored = get_current_segment(db, user.id).calculated_at
    # SQLite drops tzinfo on read; PostgreSQL (timestamptz) returns it. Compare wall-clock UTC.
    stored_utc = stored.astimezone(timezone.utc) if stored.tzinfo else stored.replace(tzinfo=timezone.utc)
    assert abs((datetime.now(timezone.utc) - stored_utc).total_seconds()) < 5


def test_upsert_keeps_an_explicit_calculated_at(db):
    user = _make_user(db)
    explicit = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    segment = upsert_current_segment(
        db, user_id=user.id, cluster_id=0, segment_label="Champions / Loyal",
        recency=1, frequency=2, monetary=3.0, calculated_at=explicit, commit=False,
    )
    assert segment.calculated_at == explicit
    db.rollback()


# ----------------------------------------------------------------- regression guard
def _utcnow_calls(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in {"utcnow", "utcfromtimestamp"}
    ]


def test_application_code_never_calls_utcnow():
    files = [ROOT / f for f in APP_FILES]
    for folder in APP_SOURCES:
        files += sorted((ROOT / folder).rglob("*.py"))
    assert files, "no application sources found"
    offenders = {str(p.relative_to(ROOT)): lines for p in files if (lines := _utcnow_calls(p))}
    assert not offenders, f"datetime.utcnow()/utcfromtimestamp() is deprecated; use timezone-aware UTC: {offenders}"
