"""
Manual, one-off migration: adds `category` and `brand` to an EXISTING
`products` table.

WHY THIS SCRIPT EXISTS
-----------------------
`Base.metadata.create_all()` (used everywhere else in this project) only
CREATES tables that don't exist yet — it never ALTERs an existing table to
add new columns. If you already have a local `ecommerce.db` with product
rows in it, restarting the app after this change will NOT add the new
columns, and any query touching `category`/`brand` will fail with
"no such column: products.category".

This script is NOT called automatically by `main.py`, by any test, or by
`create_all()`. It is a manual utility you run yourself, once, only if you
have an existing database you want to keep.

WHAT IT DOES
------------
- Adds `category TEXT` and `brand TEXT` to `products` via `ALTER TABLE ...
  ADD COLUMN`, if and only if the column doesn't already exist.
- Never drops the table, never deletes rows, never touches any other table.
- Safe to run multiple times (idempotent) — it checks `PRAGMA table_info`
  before altering anything.

WHAT TO DO INSTEAD, IF YOU DON'T CARE ABOUT EXISTING DATA
-----------------------------------------------------------
In local development, if the existing SQLite file only has disposable test
data, it is simplest and equally safe to just delete `ecommerce.db` and let
`create_all()` build a fresh schema (including the new columns) on next
startup. This script exists for the case where you want to KEEP existing
rows.

USAGE
-----
    python scripts/migrate_add_category_brand.py

Run this from the project root (so `core.config` is importable), the same
way you would run the app itself.
"""

import sqlite3

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

        added = []
        for column in ("category", "brand"):
            if _column_exists(cursor, "products", column):
                print(f"'{column}' already exists on 'products' — skipping.")
                continue
            cursor.execute(f"ALTER TABLE products ADD COLUMN {column} TEXT")
            added.append(column)

        conn.commit()

        if added:
            print(f"Added column(s) to 'products': {', '.join(added)}")
        else:
            print("Nothing to do — 'products' already has category and brand.")
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
