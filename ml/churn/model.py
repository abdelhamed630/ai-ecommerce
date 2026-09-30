"""Churn model: sklearn Pipeline (preprocessing + LogisticRegression). Pure ML.

    log1p (skewed cols) -> median imputation (+missing indicator for the gap
    feature) -> StandardScaler -> LogisticRegression

The whole thing is ONE fitted Pipeline; it is persisted and reused unchanged
for inference (preprocessing is never refit at prediction time).

Evaluation split
----------------
- >= 2 snapshots with data: TEMPORAL split. Validation = the newest snapshot,
  training = all older snapshots. Training label windows all end at or before
  the validation cutoff, so nothing from the validation period reaches training.
- otherwise: stratified random split of the single snapshot.

Small or degenerate data raises `InsufficientChurnDataError`; metrics are never
made up. If the validation set has one class, ROC-AUC is reported as None.
"""

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ml.churn.dataset import ChurnDataset
from ml.churn.features import FEATURE_NAMES, SKEWED_FEATURES

MODEL_FAMILY = "churn_logreg_v1"
DECISION_THRESHOLD = 0.5  # only for the classification metrics below


class InsufficientChurnDataError(Exception):
    """Not enough (or degenerate) data for reliable training/evaluation."""


@dataclass(frozen=True)
class TrainingConfig:
    inactivity_days: int = 60
    snapshot_count: int = 3
    recent_activity_days: int = 30
    min_samples: int = 50
    min_class_samples: int = 5
    validation_fraction: float = 0.25
    class_weight: Optional[str] = "balanced"
    C: float = 1.0
    random_state: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingResult:
    pipeline: Pipeline
    metrics: dict
    split_strategy: str
    n_train: int
    n_validation: int
    n_total: int
    churn_rate: float


def build_pipeline(config: TrainingConfig) -> Pipeline:
    skewed = [i for i, n in enumerate(FEATURE_NAMES) if n in SKEWED_FEATURES]
    plain = [i for i, n in enumerate(FEATURE_NAMES) if n not in SKEWED_FEATURES]
    preprocess = ColumnTransformer(
        [
            (
                "skewed",
                Pipeline(
                    [
                        ("log1p", FunctionTransformer(np.log1p)),
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                skewed,
            ),
            (
                "plain",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="median", add_indicator=True, keep_empty_features=True
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                plain,
            ),
        ]
    )
    classifier = LogisticRegression(
        C=config.C,
        class_weight=config.class_weight,
        max_iter=1000,
        random_state=config.random_state,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def _check_size(y: np.ndarray, config: TrainingConfig) -> None:
    if len(y) < config.min_samples:
        raise InsufficientChurnDataError(
            f"only {len(y)} labelled sample(s); at least {config.min_samples} are required"
        )
    for label, name in ((0, "retained"), (1, "churned")):
        count = int((y == label).sum())
        if count < config.min_class_samples:
            raise InsufficientChurnDataError(
                f"only {count} {name} sample(s); at least {config.min_class_samples} per class "
                "are required to train and evaluate a churn model"
            )


def split_dataset(dataset: ChurnDataset, config: TrainingConfig):
    """Return (train_idx, val_idx, strategy)."""
    snap = np.array(dataset.snapshot_index)
    distinct = sorted(set(dataset.snapshot_index))
    if len(distinct) >= 2:
        newest = distinct[-1]
        return np.where(snap != newest)[0], np.where(snap == newest)[0], "temporal"
    indices = np.arange(len(dataset))
    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=config.validation_fraction,
            random_state=config.random_state,
            stratify=dataset.y,
        )
    except ValueError as exc:
        raise InsufficientChurnDataError(f"cannot create a stratified split: {exc}") from exc
    return np.sort(train_idx), np.sort(val_idx), "stratified_random"


def evaluate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    """Classification metrics on held-out data (decision threshold 0.5)."""
    proba = pipeline.predict_proba(X)[:, list(pipeline.classes_).index(1)]
    predicted = (proba >= DECISION_THRESHOLD).astype(int)
    single_class = len(set(y.tolist())) < 2
    metrics = {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": None if single_class else float(roc_auc_score(y, proba)),
        "decision_threshold": DECISION_THRESHOLD,
        "validation_churn_rate": float(np.mean(y)) if len(y) else None,
        # accuracy of always predicting the majority class: context for `accuracy`
        "majority_class_accuracy": float(max(np.mean(y), 1 - np.mean(y))) if len(y) else None,
    }
    if single_class:
        metrics["roc_auc_note"] = "undefined: the validation set contains a single class"
    return metrics


def train_churn_model(dataset: ChurnDataset, config: TrainingConfig) -> TrainingResult:
    _check_size(dataset.y, config)
    train_idx, val_idx, strategy = split_dataset(dataset, config)
    if len(val_idx) == 0 or len(train_idx) == 0:
        raise InsufficientChurnDataError("train or validation split is empty")
    y_train = dataset.y[train_idx]
    if len(set(y_train.tolist())) < 2:
        raise InsufficientChurnDataError("the training split contains a single class")

    pipeline = build_pipeline(config)
    pipeline.fit(dataset.X[train_idx], y_train)
    metrics = evaluate(pipeline, dataset.X[val_idx], dataset.y[val_idx])
    return TrainingResult(
        pipeline=pipeline,
        metrics=metrics,
        split_strategy=strategy,
        n_train=int(len(train_idx)),
        n_validation=int(len(val_idx)),
        n_total=int(len(dataset)),
        churn_rate=float(np.mean(dataset.y)),
    )
