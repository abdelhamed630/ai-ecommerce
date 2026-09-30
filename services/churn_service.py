"""Churn: data access + orchestration (training and per-user prediction).

    DB (SQL aggregation, events <= cutoff) -> ml.churn.features/dataset
        -> ml.churn.model (train, evaluate) -> ml.churn.predict (persist / load)

The SAME `load_aggregates` feeds training (cutoff = historical snapshot) and
inference (cutoff = now), so training and serving features cannot drift.
Training is an explicit operation (`train_and_persist`, run by the Celery task
or a script) and is never called from a request handler. Prediction never
trains: without a persisted model it raises ChurnModelUnavailableError.

Timestamps: `Order.created_at` is the purchase time (the project stores no
separate completion time). Order status is read as of NOW, so an order created
before a cutoff but cancelled later is excluded from that historical snapshot
(a small, documented look-ahead; see docs/churn_prediction.md).
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from core.cache import cache, churn_key, churn_prefix
from core.config import settings
from ml.churn import dataset as churn_dataset
from ml.churn import features as churn_features
from ml.churn import model as churn_model
from ml.churn import predict as churn_predict
from ml.segmentation.adapters import COMPLETED_PURCHASE_STATUS
from models.interaction import InteractionType, ProductInteraction
from models.order import Order

ChurnModelUnavailableError = churn_predict.ChurnModelUnavailableError
ChurnArtifactError = churn_predict.ChurnArtifactError
InsufficientChurnDataError = churn_model.InsufficientChurnDataError


def _naive_utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def training_config_from_settings() -> churn_model.TrainingConfig:
    return churn_model.TrainingConfig(
        inactivity_days=settings.CHURN_INACTIVITY_DAYS,
        snapshot_count=settings.CHURN_SNAPSHOT_COUNT,
        recent_activity_days=settings.CHURN_RECENT_ACTIVITY_DAYS,
        min_samples=settings.CHURN_MIN_SAMPLES,
        min_class_samples=settings.CHURN_MIN_CLASS_SAMPLES,
        validation_fraction=settings.CHURN_VALIDATION_FRACTION,
        class_weight=settings.CHURN_CLASS_WEIGHT,
        C=settings.CHURN_LOGREG_C,
        random_state=settings.CHURN_RANDOM_STATE,
    )


# ---------------------------------------------------------------- data access

def load_aggregates(
    db: Session,
    cutoff: datetime,
    recent_days: int,
    user_id: Optional[int] = None,
) -> List[churn_features.ChurnAggregate]:
    """Per-customer aggregates of events with timestamp <= cutoff.

    Two grouped queries (orders, interactions) regardless of customer count.
    Only customers with >= 1 completed order at or before the cutoff are
    returned. `user_id` restricts everything to one customer (inference).
    """
    cutoff = churn_features.to_naive_utc(cutoff)
    recent_start = cutoff - timedelta(days=recent_days)

    order_query = db.query(
        Order.user_id,
        func.count(Order.id),
        func.coalesce(func.sum(Order.total_price), 0.0),
        func.min(Order.created_at),
        func.max(Order.created_at),
        func.coalesce(func.sum(case((Order.created_at > recent_start, 1), else_=0)), 0),
    ).filter(Order.status == COMPLETED_PURCHASE_STATUS, Order.created_at <= cutoff)
    if user_id is not None:
        order_query = order_query.filter(Order.user_id == user_id)
    orders = {row[0]: row[1:] for row in order_query.group_by(Order.user_id).all()}
    if not orders:
        return []

    customers = select(Order.user_id).where(
        Order.status == COMPLETED_PURCHASE_STATUS, Order.created_at <= cutoff
    )
    inter_query = db.query(
        ProductInteraction.user_id,
        func.coalesce(
            func.sum(case((ProductInteraction.interaction_type == InteractionType.VIEW, 1), else_=0)), 0
        ),
        func.coalesce(
            func.sum(
                case((ProductInteraction.interaction_type == InteractionType.CART_ADD, 1), else_=0)
            ),
            0,
        ),
        func.min(ProductInteraction.created_at),
        func.max(ProductInteraction.created_at),
        func.coalesce(func.sum(case((ProductInteraction.created_at > recent_start, 1), else_=0)), 0),
    ).filter(ProductInteraction.created_at <= cutoff)
    if user_id is not None:
        inter_query = inter_query.filter(ProductInteraction.user_id == user_id)
    else:
        inter_query = inter_query.filter(ProductInteraction.user_id.in_(customers))
    interactions = {row[0]: row[1:] for row in inter_query.group_by(ProductInteraction.user_id).all()}

    aggregates = []
    for uid in sorted(orders):
        n, spent, first_o, last_o, recent_o = orders[uid]
        views, carts, first_i, last_i, recent_i = interactions.get(uid, (0, 0, None, None, 0))
        aggregates.append(
            churn_features.ChurnAggregate(
                customer_id=uid,
                order_count=n,
                total_spent=spent,
                first_order_at=first_o,
                last_order_at=last_o,
                recent_order_count=recent_o,
                views_count=views,
                cart_add_count=carts,
                first_interaction_at=first_i,
                last_interaction_at=last_i,
                recent_interaction_count=recent_i,
            )
        )
    return aggregates


def load_purchasers_in_window(
    db: Session, cutoff: datetime, window_days: int, user_id: Optional[int] = None
) -> Set[int]:
    """Customers with a completed order in (cutoff, cutoff + window]. Label source only."""
    cutoff = churn_features.to_naive_utc(cutoff)
    query = db.query(Order.user_id).filter(
        Order.status == COMPLETED_PURCHASE_STATUS,
        Order.created_at > cutoff,
        Order.created_at <= cutoff + timedelta(days=window_days),
    )
    if user_id is not None:
        query = query.filter(Order.user_id == user_id)
    return {row[0] for row in query.distinct().all()}


def build_training_dataset(
    db: Session, as_of: datetime, config: churn_model.TrainingConfig
) -> churn_dataset.ChurnDataset:
    cutoffs = churn_dataset.snapshot_cutoffs(as_of, config.inactivity_days, config.snapshot_count)
    snapshots = [
        churn_dataset.Snapshot(
            cutoff=cutoff,
            aggregates=load_aggregates(db, cutoff, config.recent_activity_days),
            purchasers_in_window=load_purchasers_in_window(db, cutoff, config.inactivity_days),
        )
        for cutoff in cutoffs
    ]
    return churn_dataset.build_dataset(snapshots)


# ------------------------------------------------------------------- training

def train_and_persist(
    db: Session,
    as_of: Optional[datetime] = None,
    config: Optional[churn_model.TrainingConfig] = None,
    model_path: Optional[str] = None,
) -> Dict:
    """Explicit training operation. Raises InsufficientChurnDataError when the
    data cannot support a reliable model (nothing is persisted then)."""
    config = config or training_config_from_settings()
    path = model_path or settings.CHURN_MODEL_PATH
    as_of = churn_features.to_naive_utc(as_of) if as_of else _naive_utc_now()

    dataset = build_training_dataset(db, as_of, config)
    result = churn_model.train_churn_model(dataset, config)

    trained_at = _naive_utc_now()
    info = {
        "as_of": as_of.isoformat(),
        "snapshot_cutoffs": [c.isoformat() for c in dataset.cutoffs],
        "split_strategy": result.split_strategy,
        "n_total": result.n_total,
        "n_train": result.n_train,
        "n_validation": result.n_validation,
        "churn_rate": result.churn_rate,
        "class_weight": config.class_weight,
    }
    version = churn_predict.save_artifact(
        path, result.pipeline, trained_at, config.as_dict(), result.metrics, info
    )
    clear_artifact_cache()
    cache.delete_prefix(churn_prefix())  # predictions from the old model are stale
    return {
        "status": "trained",
        "model_version": version,
        "trained_at": trained_at.isoformat(),
        "metrics": result.metrics,
        **{k: info[k] for k in ("split_strategy", "n_total", "n_train", "n_validation", "churn_rate")},
    }


# ----------------------------------------------------------------- prediction

_artifact_cache: Dict[str, object] = {}


def clear_artifact_cache() -> None:
    _artifact_cache.clear()


def get_artifact(path: Optional[str] = None) -> churn_predict.ChurnArtifact:
    """Load the persisted model, reloading only when the file changed."""
    path = path or settings.CHURN_MODEL_PATH
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        clear_artifact_cache()
        raise ChurnModelUnavailableError(f"no trained churn model at '{path}'")
    signature = (path, stat.st_mtime_ns, stat.st_size)
    if _artifact_cache.get("signature") != signature:
        artifact = churn_predict.load_artifact(path)
        _artifact_cache.update(signature=signature, artifact=artifact)
    return _artifact_cache["artifact"]  # type: ignore[return-value]


def get_model_info(path: Optional[str] = None) -> Dict:
    return churn_predict.read_metadata(path or settings.CHURN_MODEL_PATH)


def predict_for_user(db: Session, user_id: int, now: Optional[datetime] = None) -> Dict:
    """Churn prediction for ONE customer (`user_id` = the authenticated user)."""
    artifact = get_artifact()
    now = churn_features.to_naive_utc(now) if now else _naive_utc_now()
    medium, high = settings.CHURN_RISK_MEDIUM_THRESHOLD, settings.CHURN_RISK_HIGH_THRESHOLD
    churn_predict.validate_thresholds(medium, high)

    base = {
        "model_version": artifact.model_version,
        "trained_at": artifact.trained_at,
        "features_as_of": now.isoformat(),
        "risk_thresholds": {"medium": medium, "high": high},
    }
    recent_days = int(artifact.config.get("recent_activity_days", settings.CHURN_RECENT_ACTIVITY_DAYS))
    aggregates = load_aggregates(db, now, recent_days, user_id=user_id)
    row = churn_features.build_feature_row(aggregates[0], now) if aggregates else None
    if row is None:
        return {
            **base,
            "eligible": False,
            "churn_probability": None,
            "risk_level": None,
            "reason": "no_completed_orders",
        }
    probability = churn_predict.predict_probability(artifact, row)
    return {
        **base,
        "eligible": True,
        "churn_probability": probability,
        "risk_level": churn_predict.risk_level(probability, medium, high),
        "reason": None,
    }


def get_churn_prediction(db: Session, user_id: int) -> Dict:
    """Cached wrapper. The cache key comes only from the authenticated user id;
    a cache failure falls through to the model. Errors (no model) are never cached."""
    key = churn_key(user_id)
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = predict_for_user(db, user_id)
    cache.set(key, result, settings.CHURN_CACHE_TTL_SECONDS)
    return result
