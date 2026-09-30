"""
Dataset loading and cleaning for the Customer Segmentation pipeline.

This module is intentionally independent of the FastAPI application — it
operates on a raw transaction DataFrame (e.g. the Online Retail dataset)
for OFFLINE model development and validation only. It is never imported by
`main.py` or any API route, and the dataset file itself is never bundled
into or read by the running application.

No model logic lives here — only loading, validation, and cleaning.
"""

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Columns the rest of the pipeline (features.py) actually depends on. Kept
# deliberately minimal rather than requiring every column the raw dataset
# happens to have (e.g. Description, Country aren't used for RFM).
REQUIRED_COLUMNS = (
    "InvoiceNo",
    "CustomerID",
    "InvoiceDate",
    "Quantity",
    "UnitPrice",
)


def load_transactions(path: str) -> pd.DataFrame:
    """Loads raw transaction data from a local file.

    Supports .xlsx/.xls (via openpyxl) and .csv. `path` is a caller-supplied
    string — never hardcoded — so this stays usable for offline training
    without coupling the pipeline to any particular machine's filesystem.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    suffix = file_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    if suffix == ".csv":
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported dataset file type: {suffix}")


def validate_required_columns(
    df: pd.DataFrame, required: Optional[Iterable[str]] = None
) -> None:
    """Raises ValueError if any required column is missing from `df`."""
    required_columns = set(required) if required is not None else set(REQUIRED_COLUMNS)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw transaction data into valid, purchase-only rows.

    Removes, in order:
      1. Rows with a missing CustomerID (can't be attributed to a customer).
      2. Cancelled invoices (InvoiceNo starting with "C").
      3. Rows with Quantity <= 0.
      4. Rows with UnitPrice <= 0.

    Returns a new DataFrame — the input is never mutated in place.
    """
    validate_required_columns(df)

    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["CustomerID"])
    cleaned = cleaned[~cleaned["InvoiceNo"].astype(str).str.startswith("C")]
    cleaned = cleaned[cleaned["Quantity"] > 0]
    cleaned = cleaned[cleaned["UnitPrice"] > 0]

    cleaned["CustomerID"] = cleaned["CustomerID"].astype(int)
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"])

    return cleaned.reset_index(drop=True)
