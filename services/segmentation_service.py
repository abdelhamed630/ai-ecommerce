"""
Application-layer orchestration for Customer Segmentation.

This is the "bridge" module described in docs/customer_segmentation.md's
architecture diagram — it wires together pieces that each already exist
and are independently tested, without duplicating any of their logic:

    Completed Orders
        -> ml.segmentation.adapters.compute_rfm_from_orders  (DB -> RFM)
        -> ml.segmentation.features.apply_rfm_transformations
        -> ml.segmentation.model.train_segmentation_model     (K-Means)
        -> ml.segmentation.interpretation.get_segment_label_map
        -> services.customer_segment_service.upsert_current_segment

No database querying happens outside of calling the existing adapter, no
K-Means logic is reimplemented, and no interpretation/label logic is
reimplemented — this module's only real job is sequencing those calls
correctly, deciding what "insufficient data" and "no data" mean at the
application level, and persisting the result in one transaction.

Deliberately NOT implemented here (a future phase, per project scope):
an API endpoint, a Celery task, Redis, or a scheduled/automatic trigger.
This module is called explicitly, e.g. from a script or a future endpoint.
"""

from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

from ml.segmentation import interpretation
from ml.segmentation.adapters import compute_rfm_from_orders
from ml.segmentation.features import apply_rfm_transformations
from ml.segmentation.model import (
    DEFAULT_N_CLUSTERS,
    DEFAULT_RANDOM_STATE,
    FEATURE_COLUMNS,
    SEGMENTATION_MODEL_VERSION,
    train_segmentation_model,
)
from services.customer_segment_service import upsert_current_segment
from sqlalchemy.orm import Session


class InsufficientSegmentationDataError(Exception):
    """Raised when there are fewer customers than the configured K.

    K-Means requires at least as many samples as clusters. Rather than
    silently reducing K to fit whatever data happens to be available (which
    would make segments computed at different times not comparable, and
    would hide a genuinely under-populated production database), this is
    surfaced as an explicit, catchable domain error. K itself is never
    auto-adjusted anywhere in this module.
    """


def run_customer_segmentation(
    db: Session, reference_date: Optional[pd.Timestamp] = None
) -> Dict:
    """Runs one full segmentation pass and persists the result for every eligible customer.

    Steps (each delegating to the existing, already-tested implementation):
      1. Pull completed-order RFM data via the existing adapter/`compute_rfm`
         (no RFM math is reimplemented here).
      2. If there are no completed transactions at all, return an empty
         summary without touching the database's CustomerSegment table and
         without attempting to train anything — a fresh installation with
         no completed orders yet is a normal, valid state, not an error.
      3. If there are fewer distinct customers than the configured
         `DEFAULT_N_CLUSTERS` (K), raise `InsufficientSegmentationDataError`
         rather than training with a reduced K.
      4. Apply the existing log1p transformation, train the existing
         K-Means pipeline (K and random_state come from
         `ml.segmentation.model`'s own constants — never redefined here),
         and derive segment labels via the existing interpretation layer.
      5. Persist one CustomerSegment row per customer, all in a single
         transaction: every upsert is flushed but not committed
         individually, then the whole batch is committed once. If anything
         fails partway through, the transaction is rolled back and the
         exception is re-raised — no half-updated batch is left committed.

    `reference_date`, if given, is passed straight through to the existing
    RFM calculation (via `compute_rfm_from_orders`); if omitted, that
    function's own default behavior applies unchanged. No second
    reference-date calculation is introduced here.

    Returns a summary dict:
        {
            "customers_processed": int,
            "segments": {label: count, ...},
            "model_version": str,
        }
    """
    rfm = compute_rfm_from_orders(db, reference_date=reference_date)

    if rfm.empty:
        return {
            "customers_processed": 0,
            "segments": {},
            "model_version": SEGMENTATION_MODEL_VERSION,
        }

    customer_count = len(rfm)
    if customer_count < DEFAULT_N_CLUSTERS:
        raise InsufficientSegmentationDataError(
            f"Cannot train the configured segmentation model: found "
            f"{customer_count} customer(s) with completed orders, but the "
            f"configured model requires at least {DEFAULT_N_CLUSTERS} "
            f"(K={DEFAULT_N_CLUSTERS}). Reducing K automatically is not "
            f"supported — either wait for more customers with completed "
            f"orders, or explicitly reconfigure the segmentation model."
        )

    # CustomerID is preserved through every step below (compute_rfm keeps
    # it, apply_rfm_transformations only adds Frequency_log/Monetary_log)
    # but is never passed into the model: FEATURE_COLUMNS
    # ("Recency", "Frequency_log", "Monetary_log") is the only thing
    # train_segmentation_model selects as X.
    rfm_transformed = apply_rfm_transformations(rfm)

    _pipeline, cluster_labels = train_segmentation_model(
        rfm_transformed,
        feature_columns=FEATURE_COLUMNS,
        n_clusters=DEFAULT_N_CLUSTERS,
        random_state=DEFAULT_RANDOM_STATE,
    )
    rfm_transformed["Cluster"] = cluster_labels

    # The ONLY place cluster ids are associated with human-readable
    # labels — always freshly derived from this run's actual cluster
    # profiles, never a hardcoded {cluster_id: label} mapping.
    segment_label_map = interpretation.get_segment_label_map(
        rfm_transformed, cluster_col="Cluster"
    )

    calculated_at = datetime.now(timezone.utc)
    segment_counts: Dict[str, int] = {}

    try:
        for row in rfm_transformed.itertuples(index=False):
            cluster_id = int(row.Cluster)
            segment_label = segment_label_map[cluster_id]

            upsert_current_segment(
                db,
                user_id=int(row.CustomerID),
                cluster_id=cluster_id,
                segment_label=segment_label,
                recency=int(row.Recency),
                frequency=int(row.Frequency),
                monetary=float(row.Monetary),
                model_version=SEGMENTATION_MODEL_VERSION,
                calculated_at=calculated_at,
                commit=False,
            )
            segment_counts[segment_label] = segment_counts.get(segment_label, 0) + 1

        db.commit()
    except Exception:
        db.rollback()
        raise

    return {
        "customers_processed": customer_count,
        "segments": segment_counts,
        "model_version": SEGMENTATION_MODEL_VERSION,
    }
