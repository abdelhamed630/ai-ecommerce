"""Model artifact persistence + inference + risk bands. Pure (joblib, no DB).

Artifact = one joblib file (atomic write) with ONLY: the fitted sklearn
Pipeline, feature names, model version, training timestamp, training config,
metrics and data counts, plus a JSON sidecar (`metadata.json`, same info minus
the model) so metadata can be read without unpickling. No database objects.

Security note: joblib files are pickles. Only load artifacts from a path you
control (settings.CHURN_MODEL_PATH); never from user input.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import joblib
import numpy as np

from ml.churn.features import FEATURE_NAMES, rows_to_matrix
from ml.churn.model import MODEL_FAMILY

ARTIFACT_SCHEMA_VERSION = 1
RISK_LOW, RISK_MEDIUM, RISK_HIGH = "LOW", "MEDIUM", "HIGH"


class ChurnModelUnavailableError(Exception):
    """No trained churn model exists at the configured path."""


class ChurnArtifactError(Exception):
    """The artifact exists but is corrupt or incompatible with this code."""


@dataclass(frozen=True)
class ChurnArtifact:
    pipeline: Any
    feature_names: tuple
    model_version: str
    trained_at: str  # ISO 8601 UTC
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    info: Dict[str, Any]


def metadata_path(path: str) -> str:
    return os.path.join(os.path.dirname(path) or ".", "metadata.json")


def _atomic_write(path: str, writer) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_artifact(
    path: str,
    pipeline,
    trained_at: datetime,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    info: Dict[str, Any],
) -> str:
    """Persist the model; returns the model version."""
    version = f"{MODEL_FAMILY}-{trained_at:%Y%m%d%H%M%S}"
    metadata = {
        "artifact_schema": ARTIFACT_SCHEMA_VERSION,
        "model_version": version,
        "model_family": MODEL_FAMILY,
        "trained_at": trained_at.isoformat(),
        "feature_names": list(FEATURE_NAMES),
        "config": config,
        "metrics": metrics,
        "info": info,
    }
    payload = {**metadata, "pipeline": pipeline}
    _atomic_write(path, lambda tmp: joblib.dump(payload, tmp))
    _atomic_write(
        metadata_path(path),
        lambda tmp: open(tmp, "w").write(json.dumps(metadata, indent=2, default=str)),
    )
    return version


def load_artifact(path: str) -> ChurnArtifact:
    if not os.path.exists(path):
        raise ChurnModelUnavailableError(f"no trained churn model at '{path}'")
    try:
        payload = joblib.load(path)
        if payload["artifact_schema"] != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported artifact schema {payload['artifact_schema']}")
        artifact = ChurnArtifact(
            pipeline=payload["pipeline"],
            feature_names=tuple(payload["feature_names"]),
            model_version=payload["model_version"],
            trained_at=payload["trained_at"],
            config=payload["config"],
            metrics=payload["metrics"],
            info=payload["info"],
        )
    except ChurnModelUnavailableError:
        raise
    except Exception as exc:
        raise ChurnArtifactError(f"cannot read churn artifact: {exc}") from exc
    if artifact.feature_names != tuple(FEATURE_NAMES):
        raise ChurnArtifactError(
            "artifact feature names do not match the current feature definition; retrain the model"
        )
    return artifact


def read_metadata(path: str) -> Dict[str, Any]:
    """Metadata without unpickling the model."""
    meta_file = metadata_path(path)
    if not os.path.exists(meta_file):
        raise ChurnModelUnavailableError(f"no trained churn model at '{path}'")
    try:
        with open(meta_file) as handle:
            return json.load(handle)
    except Exception as exc:
        raise ChurnArtifactError(f"cannot read churn metadata: {exc}") from exc


def predict_probability(artifact: ChurnArtifact, feature_row: Dict[str, float]) -> float:
    """P(churn) in [0, 1] using the persisted, already-fitted pipeline."""
    X = rows_to_matrix([feature_row], artifact.feature_names)
    classes = list(artifact.pipeline.classes_)
    proba = float(artifact.pipeline.predict_proba(X)[0, classes.index(1)])
    return float(np.clip(proba, 0.0, 1.0))


def validate_thresholds(medium: float, high: float) -> None:
    if not (0.0 <= medium < high <= 1.0):
        raise ValueError("risk thresholds must satisfy 0 <= medium < high <= 1")


def risk_level(probability: float, medium: float, high: float) -> str:
    """LOW: p < medium; MEDIUM: medium <= p < high; HIGH: p >= high."""
    validate_thresholds(medium, high)
    if probability >= high:
        return RISK_HIGH
    if probability >= medium:
        return RISK_MEDIUM
    return RISK_LOW
