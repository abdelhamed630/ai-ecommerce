"""
SUPERSEDED: the application no longer reads `is_admin` — authorization is
now based on `User.role` (USER / SELLER / ADMIN). If you're setting up a
database from before that change, run this script first (to add the old
column, if you still need it for history) and then
scripts/migrate_is_admin_to_role.py (to add `role` and migrate the data
across). New databases don't need this script at all — `create_all()`
already creates the `role` column directly. This file is kept only for
historical reference.

Manual, one-off migration: adds `is_admin` to an EXISTING `users` table.

WHY THIS SCRIPT EXISTS
-----------------------
`Base.metadata.create_all()` (used everywhere else in this project) only
CREATES tables that don't exist yet — it never ALTERs an existing table to
add new columns. If you already have a local `ecommerce.db` with user rows
in it, restarting the app after this change will NOT add the new column,
and any query touching `users.is_admin` will fail with
"no such column: users.is_admin".

This script is NOT called automatically by `main.py`, by any test, or by
`create_all()`. It is a manual utility you run yourself, once, only if you
have an existing database you want to keep. See
scripts/migrate_add_category_brand.py for the precedent this follows.

WHAT IT DOES
------------
- Adds `is_admin BOOLEAN NOT NULL DEFAULT 0` to `users` via `ALTER TABLE
  ... ADD COLUMN`, if and only if the column doesn't already exist. Every
  existing user row becomes `is_admin = 0` (False) — nobody is silently
  promoted to admin by running this migration.
- Never drops the table, never deletes rows, never touches any other
  table.
- Safe to run multiple times (idempotent) — it checks `PRAGMA table_info`
  before altering anything.

PROMOTING A USER TO ADMIN
--------------------------
There is intentionally no API endpoint for this (self-service admin
promotion would be a security hole). After running this migration,
promote a specific user directly, e.g.:

    UPDATE users SET is_admin = 1 WHERE email = 'someone@example.com';

WHAT TO DO INSTEAD, IF YOU DON'T CARE ABOUT EXISTING DATA
-----------------------------------------------------------
In local development, if the existing SQLite file only has disposable test
data, it is simplest and equally safe to just delete `ecommerce.db` and let
`create_all()` build a fresh schema (including this column) on next
startup. This script exists for the case where you want to KEEP existing
rows.

USAGE
-----
    python scripts/migrate_add_user_is_admin.py

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

        if _column_exists(cursor, "users", "is_admin"):
            print("'is_admin' already exists on 'users' — skipping.")
        else:
            cursor.execute(
                "ALTER TABLE users ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT 0"
            )
            conn.commit()
            print("Added column 'is_admin' to 'users' (all existing rows set to 0/False).")
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
