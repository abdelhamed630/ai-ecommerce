"""
Manual, one-off migration: adds `seller_id` to an EXISTING `products`
table.

WHY THIS SCRIPT EXISTS
-----------------------
`Base.metadata.create_all()` (used everywhere else in this project) only
CREATES tables that don't exist yet — it never ALTERs an existing table to
add new columns. If you already have a local `ecommerce.db` with product
rows in it, restarting the app after this change will NOT add the new
column, and any query touching `products.seller_id` will fail with
"no such column: products.seller_id".

This script is NOT called automatically by `main.py`, by any test, or by
`create_all()`. It is a manual utility you run yourself, once, only if you
have an existing database you want to keep. It follows the same pattern
as scripts/migrate_add_category_brand.py before it.

WHAT IT DOES
------------
- Adds `seller_id INTEGER` (a nullable foreign key to `users.id`) to
  `products` via `ALTER TABLE ... ADD COLUMN`, if and only if the column
  doesn't already exist.
- Every existing product row gets `seller_id = NULL`. This is
  intentional: products created before ownership existed have no
  historical owner, and this script does NOT invent one (no "assign
  everything to the first admin" heuristic, no guessing). A NULL
  `seller_id` simply means only an ADMIN can manage that particular
  product until someone deliberately assigns it an owner (e.g. via a
  direct `UPDATE products SET seller_id = ... WHERE id = ...`, or a
  future seller-onboarding flow — not implemented in this phase).
- Never drops the table, never deletes rows, never touches any other
  table.
- Safe to run multiple times (idempotent) — it checks `PRAGMA table_info`
  before altering anything.

WHAT TO DO INSTEAD, IF YOU DON'T CARE ABOUT EXISTING DATA
-----------------------------------------------------------
In local development, if the existing SQLite file only has disposable test
data, it is simplest and equally safe to just delete `ecommerce.db` and let
`create_all()` build a fresh schema (including this column) on next
startup. This script exists for the case where you want to KEEP existing
rows.

USAGE
-----
    python scripts/migrate_add_product_seller_id.py

Run this from the project root (so `core.config` is importable), the same
way you would run the app itself.
"""

import sqlite3
import os
import sys
# Make the project root importable when run as `python scripts/<name>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import settings

# Only handles the sqlite:///./something.db form this project actually uses.
_DB_PATH = settings.DATABASE_URL.replace("sqlite:///", "")


def _column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    existing_columns = {row[1] for row in cursor.fetchall()}
    return column in existing_columns


def migrate() -> None:
    conn = sqlite3.connect(_DB_PATH)
    try:
        cursor = conn.cursor()

        if _column_exists(cursor, "products", "seller_id"):
            print("'seller_id' already exists on 'products' — skipping.")
        else:
            cursor.execute("ALTER TABLE products ADD COLUMN seller_id INTEGER")
            conn.commit()
            print(
                "Added column 'seller_id' to 'products' (all existing rows set "
                "to NULL — no owner is inferred for pre-existing products)."
            )
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
