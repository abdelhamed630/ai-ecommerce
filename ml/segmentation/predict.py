"""
Model persistence and prediction for Customer Segmentation.

Deliberately independent of FastAPI: no import of `main`, `api/`,
`services/`, or any application code lives here. This module only knows how
to save/load a trained pipeline + its interpretation metadata, and how to
apply it to new RFM data. Wiring this into the running API (endpoints,
database-backed RFM computation from real Orders) is explicitly out of
scope for this task.
"""

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.segmentation.model import FEATURE_COLUMNS, predict_clusters

# Local artifacts directory, relative to the project root. Not committed by
# default (see .gitignore) — these are generated files, regenerable by
# re-running training, not source code.
DEFAULT_ARTIFACTS_DIR = "ml/artifacts/segmentation"

_MODEL_FILENAME = "pipeline.joblib"
_METADATA_FILENAME = "metadata.json"


def save_artifacts(
    pipeline: Pipeline,
    segment_label_map: Dict[int, str],
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
) -> None:
    """Persists the trained pipeline plus the metadata needed to interpret its output.

    Saves two files under `artifacts_dir`:
      - pipeline.joblib: the fitted sklearn Pipeline (scaler + KMeans).
      - metadata.json: the {cluster_id: label} map and the feature column
        order the pipeline expects, so predictions can be labeled correctly
        without re-deriving anything.
    """
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, out_dir / _MODEL_FILENAME)

    metadata = {
        "segment_label_map": {str(k): v for k, v in segment_label_map.items()},
        "feature_columns": list(feature_columns),
    }
    with open(out_dir / _METADATA_FILENAME, "w") as f:
        json.dump(metadata, f, indent=2)


def load_artifacts(
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
) -> Tuple[Pipeline, Dict[int, str], List[str]]:
    """Loads a previously trained pipeline and its interpretation metadata."""
    in_dir = Path(artifacts_dir)
    model_path = in_dir / _MODEL_FILENAME
    metadata_path = in_dir / _METADATA_FILENAME

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"No trained segmentation artifacts found in '{artifacts_dir}'. "
            "Train and save a model first."
        )

    pipeline: Pipeline = joblib.load(model_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    segment_label_map = {int(k): v for k, v in metadata["segment_label_map"].items()}
    feature_columns = metadata["feature_columns"]
    return pipeline, segment_label_map, feature_columns


def predict_segment(
    rfm_transformed: pd.DataFrame,
    pipeline: Pipeline,
    segment_label_map: Dict[int, str],
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Predicts cluster + human-readable segment label for new RFM data.

    `rfm_transformed` must already have `feature_columns` present (i.e. it
    went through `features.apply_rfm_transformations`) and should also carry
    `CustomerID` if the caller wants predictions traceable back to a
    customer.

    Returns a DataFrame with (CustomerID if present) + Cluster + SegmentLabel.
    Unknown cluster ids (should not normally happen) map to "Unknown".
    """
    clusters = predict_clusters(pipeline, rfm_transformed, feature_columns=feature_columns)

    result = pd.DataFrame({"Cluster": clusters})
    if "CustomerID" in rfm_transformed.columns:
        result.insert(0, "CustomerID", rfm_transformed["CustomerID"].values)

    result["SegmentLabel"] = result["Cluster"].map(
        lambda cluster_id: segment_label_map.get(int(cluster_id), "Unknown")
    )
    return result
