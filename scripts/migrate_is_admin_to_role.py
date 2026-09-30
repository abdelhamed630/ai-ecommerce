"""
Manual, one-off migration: adds `role` to an EXISTING `users` table and,
if that table still has the old `is_admin` column, migrates every
existing user's authorization from `is_admin` to `role`.

WHY THIS SCRIPT EXISTS
-----------------------
`Base.metadata.create_all()` (used everywhere else in this project) only
CREATES tables that don't exist yet — it never ALTERs an existing table to
add new columns. If you already have a local `ecommerce.db` with user rows
in it (in particular, one that was set up before this change and may still
have `is_admin`), restarting the app after this change will NOT add the
new `role` column, and any query touching `users.role` will fail with
"no such column: users.role".

This script is NOT called automatically by `main.py`, by any test, or by
`create_all()`. It is a manual utility you run yourself, once, only if you
have an existing database you want to keep. It follows the same pattern
as scripts/migrate_add_user_is_admin.py and
scripts/migrate_add_category_brand.py before it.

WHAT IT DOES
------------
1. Adds `role TEXT NOT NULL DEFAULT 'USER'` to `users` via `ALTER TABLE
   ... ADD COLUMN`, if and only if the column doesn't already exist.
   Every existing row starts out as `role = 'USER'`.
2. If the table still has an `is_admin` column, backfills `role` from it:
       is_admin = 1 (true)  -> role = 'ADMIN'
       is_admin = 0 (false) -> role = 'USER'
   No existing user is ever promoted to SELLER by this migration — SELLER
   must always be assigned explicitly (see scripts/set_user_role.py).
3. Never drops any table or column, never deletes any row. `is_admin` (if
   present) is left in place, untouched, as an inert legacy column — the
   application no longer reads it (see models/user.py), so leaving it
   behind is harmless. This script intentionally does NOT attempt
   `ALTER TABLE ... DROP COLUMN is_admin`; dropping it, if you want to,
   is a separate, deliberate decision to make once you've verified the
   migration.
4. Safe to run multiple times (idempotent) — it checks `PRAGMA
   table_info` before altering anything. Step 2 only promotes rows that
   are still `role = 'USER'` while `is_admin = 1`; it never downgrades or
   overwrites any other role, so a SELLER assigned after the migration
   stays a SELLER if the script is run again.

WHAT TO DO INSTEAD, IF YOU DON'T CARE ABOUT EXISTING DATA
-----------------------------------------------------------
In local development, if the existing SQLite file only has disposable test
data, it is simplest and equally safe to just delete `ecommerce.db` and let
`create_all()` build a fresh schema (including this column) on next
startup. This script exists for the case where you want to KEEP existing
rows.

VERIFYING THE MIGRATION
------------------------
After running this script, verify:
    - every user who was `is_admin = 1` now has `role = 'ADMIN'`
    - every user who was `is_admin = 0` now has `role = 'USER'`
    - no user row was deleted (row count unchanged)
    - existing JWT authentication still works (`role` is looked up fresh
      from the database on every request via `get_current_user` /
      `get_current_admin_user` — nothing about login/JWT changes)

USAGE
-----
    python scripts/migrate_is_admin_to_role.py

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

        if _column_exists(cursor, "users", "role"):
            print("'role' already exists on 'users' — skipping column creation.")
        else:
            cursor.execute(
                "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'USER'"
            )
            conn.commit()
            print("Added column 'role' to 'users' (all existing rows set to 'USER').")

        if _column_exists(cursor, "users", "is_admin"):
            # Only ever promotes USER -> ADMIN for rows whose legacy flag is
            # set. is_admin = 0 rows need no write (they are already 'USER'
            # by the column default), and any row that already holds
            # SELLER or a deliberately assigned role is never touched, so
            # re-running this script can never undo a later role change.
            cursor.execute(
                "UPDATE users SET role = 'ADMIN' WHERE is_admin = 1 AND role = 'USER'"
            )
            promoted = cursor.rowcount
            conn.commit()
            print(
                f"Backfilled 'role' from 'is_admin': {promoted} user(s) promoted "
                "to 'ADMIN'; is_admin = 0 users remain 'USER'. 'is_admin' left "
                "in place, unused, on the table."
            )
        else:
            print("No 'is_admin' column found — nothing to backfill.")
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
