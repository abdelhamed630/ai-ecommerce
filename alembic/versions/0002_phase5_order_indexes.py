"""Phase 5: indexes justified by the churn/segmentation/order query patterns.

- ix_orders_status_created_at: `Order.status == COMPLETED [AND created_at <= cutoff]`
  followed by GROUP BY user_id (services/churn_service.py,
  services/segmentation_analysis_service.py, ml/segmentation/adapters.py).
- ix_order_items_order_id: FK used to load an order's items and by the RFM
  Order JOIN OrderItem. PostgreSQL does not create indexes for foreign keys.

Whether the PostgreSQL planner actually uses them at the current data size is NOT
VERIFIED (tiny tables are usually sequential-scanned regardless).

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-29
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_orders_status_created_at", "orders", ["status", "created_at"], unique=False)
    op.create_index("ix_order_items_order_id", "order_items", ["order_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_order_items_order_id", table_name="order_items")
    op.drop_index("ix_orders_status_created_at", table_name="orders")
