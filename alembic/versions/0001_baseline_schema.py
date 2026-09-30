"""Baseline schema: the database as of the end of Phase 4.

Constructed BY HAND from the SQLAlchemy models (models/*.py) — autogenerate could
not be run when this was written. tests/test_alembic_migrations.py compares the
result of `upgrade head` against Base.metadata, so any drift is caught by the
test suite.

Existing development databases that were created with create_all() and the
scripts/migrate_*.py helpers already contain exactly this schema: mark them with
`alembic stamp 0001` (do NOT run the baseline against them), then
`alembic upgrade head`.

Revision ID: 0001
Revises:
Create Date: 2026-09-29
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _ts(name: str) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), server_default=sa.func.now())


def upgrade() -> None:
    # ------------------------------------------------------------------ users
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("full_name", sa.String(), nullable=True),
        sa.Column("first_name", sa.String(length=50), nullable=True),
        sa.Column("last_name", sa.String(length=50), nullable=True),
        sa.Column("username", sa.String(length=30), nullable=True),
        sa.Column("phone", sa.String(length=20), nullable=True),
        sa.Column("bio", sa.String(length=500), nullable=True),
        sa.Column("avatar_url", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column(
            "role",
            sa.Enum("USER", "SELLER", "ADMIN", name="userrole", native_enum=False, length=20),
            server_default="USER",
            nullable=False,
        ),
        _ts("created_at"),
        _ts("updated_at"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_id", "users", ["id"], unique=False)
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_username", "users", ["username"], unique=True)

    # --------------------------------------------------------------- products
    op.create_table(
        "products",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("stock", sa.Integer(), nullable=True),
        sa.Column("image_url", sa.String(), nullable=True),
        sa.Column("category", sa.String(), nullable=True),
        sa.Column("brand", sa.String(), nullable=True),
        sa.Column("seller_id", sa.Integer(), nullable=True),
        _ts("created_at"),
        sa.ForeignKeyConstraint(["seller_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_products_id", "products", ["id"], unique=False)
    op.create_index("ix_products_name", "products", ["name"], unique=False)
    op.create_index("ix_products_category", "products", ["category"], unique=False)
    op.create_index("ix_products_brand", "products", ["brand"], unique=False)
    op.create_index("ix_products_seller_id", "products", ["seller_id"], unique=False)

    # ------------------------------------------------------------------ carts
    op.create_table(
        "carts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        _ts("created_at"),
        _ts("updated_at"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", name="uq_cart_user_id"),
    )
    op.create_index("ix_carts_id", "carts", ["id"], unique=False)

    op.create_table(
        "cart_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("cart_id", sa.Integer(), nullable=False),
        sa.Column("product_id", sa.Integer(), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        _ts("created_at"),
        _ts("updated_at"),
        sa.CheckConstraint("quantity > 0", name="ck_cart_item_quantity_positive"),
        sa.ForeignKeyConstraint(["cart_id"], ["carts.id"]),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("cart_id", "product_id", name="uq_cart_item_cart_product"),
    )
    op.create_index("ix_cart_items_id", "cart_items", ["id"], unique=False)

    # ----------------------------------------------------------------- orders
    op.create_table(
        "orders",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("PENDING", "CONFIRMED", "CANCELLED", "COMPLETED", name="orderstatus", native_enum=False, length=20),
            nullable=False,
        ),
        sa.Column("total_price", sa.Float(), nullable=False),
        _ts("created_at"),
        _ts("updated_at"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_orders_id", "orders", ["id"], unique=False)
    op.create_index("ix_orders_user_id", "orders", ["user_id"], unique=False)

    op.create_table(
        "order_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("order_id", sa.Integer(), nullable=False),
        sa.Column("product_id", sa.Integer(), nullable=False),
        sa.Column("product_name", sa.String(), nullable=False),
        sa.Column("product_price", sa.Float(), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("subtotal", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["order_id"], ["orders.id"]),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_order_items_id", "order_items", ["id"], unique=False)

    # --------------------------------------------------------------- payments
    op.create_table(
        "payments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("order_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("PENDING", "PAID", "FAILED", "REFUNDED", name="paymentstatus", native_enum=False, length=20),
            nullable=False,
        ),
        sa.Column(
            "payment_method",
            sa.Enum("CASH_ON_DELIVERY", "CARD", name="paymentmethod", native_enum=False, length=30),
            nullable=False,
        ),
        sa.Column("transaction_reference", sa.String(), nullable=True),
        _ts("created_at"),
        _ts("updated_at"),
        sa.CheckConstraint("amount > 0", name="ck_payment_amount_positive"),
        sa.ForeignKeyConstraint(["order_id"], ["orders.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("transaction_reference", name="uq_payment_transaction_reference"),
    )
    op.create_index("ix_payments_id", "payments", ["id"], unique=False)
    op.create_index("ix_payments_order_id", "payments", ["order_id"], unique=False)
    op.create_index("ix_payments_user_id", "payments", ["user_id"], unique=False)

    # ----------------------------------------------------- product_interactions
    op.create_table(
        "product_interactions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("product_id", sa.Integer(), nullable=False),
        sa.Column(
            "interaction_type",
            sa.Enum("VIEW", "CART_ADD", "PURCHASE", name="interactiontype", native_enum=False, length=20),
            nullable=False,
        ),
        _ts("created_at"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_product_interactions_id", "product_interactions", ["id"], unique=False)
    op.create_index("ix_product_interactions_user_id", "product_interactions", ["user_id"], unique=False)
    op.create_index("ix_product_interactions_product_id", "product_interactions", ["product_id"], unique=False)
    op.create_index("ix_product_interactions_interaction_type", "product_interactions", ["interaction_type"], unique=False)
    op.create_index("ix_product_interactions_created_at", "product_interactions", ["created_at"], unique=False)
    op.create_index("ix_interaction_user_product", "product_interactions", ["user_id", "product_id"], unique=False)
    op.create_index("ix_interaction_product_type", "product_interactions", ["product_id", "interaction_type"], unique=False)
    op.create_index("ix_interaction_type_created_at", "product_interactions", ["interaction_type", "created_at"], unique=False)

    # ------------------------------------------------------- customer_segments
    op.create_table(
        "customer_segments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("cluster_id", sa.Integer(), nullable=False),
        sa.Column("segment_label", sa.String(), nullable=False),
        sa.Column("recency", sa.Integer(), nullable=False),
        sa.Column("frequency", sa.Integer(), nullable=False),
        sa.Column("monetary", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("calculated_at", sa.DateTime(timezone=True), nullable=False),
        _ts("created_at"),
        _ts("updated_at"),
        sa.CheckConstraint("recency >= 0", name="ck_customer_segment_recency_non_negative"),
        sa.CheckConstraint("monetary >= 0", name="ck_customer_segment_monetary_non_negative"),
        sa.CheckConstraint("frequency > 0", name="ck_customer_segment_frequency_positive"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", name="uq_customer_segment_user_id"),
    )
    op.create_index("ix_customer_segments_id", "customer_segments", ["id"], unique=False)
    op.create_index("ix_customer_segments_user_id", "customer_segments", ["user_id"], unique=False)


def downgrade() -> None:
    # Reverse dependency order. Destroys ALL data: never run against production
    # without a backup.
    for table in (
        "customer_segments",
        "product_interactions",
        "payments",
        "order_items",
        "orders",
        "cart_items",
        "carts",
        "products",
        "users",
    ):
        op.drop_table(table)  # drops the table's indexes with it
