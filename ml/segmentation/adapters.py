"""
Adapter between the E-Commerce application's Orders/OrderItems and the
existing (already validated) RFM segmentation pipeline.

Conceptual flow:

    Orders (status == COMPLETED)
        -> OrderItems
        -> transaction-shaped DataFrame (CustomerID/InvoiceNo/InvoiceDate/
           Quantity/UnitPrice)
        -> ml.segmentation.features.compute_rfm()   (existing, unmodified)

This module does NOT reimplement any RFM math — it only reshapes real
application data into the exact column shape `compute_rfm()` already
expects, then delegates to it.

Field mapping (documented explicitly, since the external Online Retail
dataset used different names/units than this application):

    CustomerID  <- Order.user_id      (this app has no separate Customer
                                        model; User.id IS the customer id)
    InvoiceNo   <- Order.id
    InvoiceDate <- Order.created_at
    Quantity    <- OrderItem.quantity
    UnitPrice   <- OrderItem.product_price  (the PRICE SNAPSHOT captured at
                                        order-creation time, never the
                                        product's current live price — this
                                        is what makes historical RFM/Monetary
                                        figures stay correct even after a
                                        product's price changes later)

Purchase eligibility:

    Only orders with status == OrderStatus.COMPLETED are included.

    This project's Order status enum is PENDING / CONFIRMED / CANCELLED /
    COMPLETED (see models/order.py). CONFIRMED was deliberately NOT
    included here: the existing order_service.py transition rules allow
    CONFIRMED -> CANCELLED, meaning a CONFIRMED order is not yet a final,
    irreversible purchase — counting it in RFM could later be contradicted
    by a cancellation with no mechanism here to retroactively correct the
    segmentation. COMPLETED is the only status in this project with no
    outgoing transition at all (see order_service._ADVANCE_TRANSITIONS /
    _CANCELLABLE_STATES), i.e. it's truly final. CANCELLED and PENDING are
    excluded for the obvious reason that they are not completed purchases.
"""

from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from ml.segmentation.features import compute_rfm
from models.order import Order, OrderItem, OrderStatus

TRANSACTION_COLUMNS = ("CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice")

# The only Order status considered a completed, RFM-eligible purchase.
# Documented here as a single named constant rather than a scattered
# literal, and intentionally NOT including CONFIRMED (see module docstring).
COMPLETED_PURCHASE_STATUS = OrderStatus.COMPLETED


def get_completed_order_transactions(db: Session) -> pd.DataFrame:
    """Queries completed orders' items and returns them in transaction-row shape.

    One row per OrderItem — an order with multiple items produces multiple
    rows sharing the same CustomerID/InvoiceNo, exactly like a multi-line
    invoice in the original Online Retail dataset.

    Only the columns actually needed are selected (no full ORM objects, no
    Product join — OrderItem already carries the historical price/name
    snapshot), so this is a single query with no N+1 behavior.

    Returns an empty DataFrame (with the correct columns, zero rows) if
    there are no completed orders yet — never fabricated data.
    """
    rows = (
        db.query(
            Order.user_id.label("CustomerID"),
            Order.id.label("InvoiceNo"),
            Order.created_at.label("InvoiceDate"),
            OrderItem.quantity.label("Quantity"),
            OrderItem.product_price.label("UnitPrice"),
        )
        .join(OrderItem, OrderItem.order_id == Order.id)
        .filter(Order.status == COMPLETED_PURCHASE_STATUS)
        .all()
    )

    if not rows:
        return pd.DataFrame(columns=list(TRANSACTION_COLUMNS))

    return pd.DataFrame(rows, columns=list(TRANSACTION_COLUMNS))


def compute_rfm_from_orders(
    db: Session, reference_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """End-to-end: reads completed Orders/OrderItems and returns an RFM DataFrame.

    Thin wrapper around `get_completed_order_transactions` +
    `ml.segmentation.features.compute_rfm` — the RFM math itself is never
    reimplemented here.

    `reference_date` is passed straight through to `compute_rfm` (which
    already defaults it to one day after the latest transaction date if not
    given) — never hardcoded to "today" in this adapter.

    Returns an empty DataFrame with the expected RFM columns
    (CustomerID/Recency/Frequency/Monetary) if there are no completed
    orders — callers can check `.empty` rather than handling an exception
    for what is a normal, valid state (a fresh installation with no
    completed purchases yet).
    """
    transactions = get_completed_order_transactions(db)

    if transactions.empty:
        return pd.DataFrame(columns=["CustomerID", "Recency", "Frequency", "Monetary"])

    return compute_rfm(transactions, reference_date=reference_date)
