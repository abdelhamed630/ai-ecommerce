"""Index on products.price for the GET /products/ min_price / max_price range filters.

Additive only: no table is dropped or altered, no data is touched.

Deliberately NOT added: an index for the free-text `q` filter. It is a
`%term%` ILIKE over several columns after Arabic folding (REPLACE chain); a B-tree
cannot serve either, and a trigram (pg_trgm) index would need an extension plus
an expression index, which has to be designed against real data volume first.
`category` already has ix_products_category; the filter compares lower(category),
so that index is not used for it (fine at catalogue sizes; a functional index on
lower(category) is the follow-up if it ever matters).

Whether the PostgreSQL planner uses ix_products_price at the current data size is
NOT VERIFIED (tiny tables are usually sequentially scanned).

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-04
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_products_price", "products", ["price"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_products_price", table_name="products")
