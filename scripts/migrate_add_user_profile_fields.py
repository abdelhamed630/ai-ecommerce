"""
Manual, one-off migration: adds the profile columns to an EXISTING `users`
table (first_name, last_name, username, phone, bio, avatar_url, updated_at).

`Base.metadata.create_all()` never ALTERs existing tables, so an existing
`ecommerce.db` (including the one the test suite runs against) must be
migrated before running the app/tests after the profile change. Same
pattern as scripts/migrate_add_category_brand.py.

- Adds each column only if missing (idempotent, checks PRAGMA table_info).
- Creates the UNIQUE index `ix_users_username` if missing (SQLite cannot add
  a UNIQUE column via ALTER TABLE; multiple NULL usernames are allowed).
- Backfills updated_at from created_at for existing rows.
- Never drops/recreates a table, never deletes or rewrites users, passwords,
  ids or roles. No names/usernames are invented for existing users.

USAGE (from the project root):
    python scripts/migrate_add_user_profile_fields.py
"""

import sqlite3
import os
import sys
# Make the project root importable when run as `python scripts/<name>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import settings

_DB_PATH = settings.DATABASE_URL.replace("sqlite:///", "")

_COLUMNS = {
    "first_name": "VARCHAR(50)",
    "last_name": "VARCHAR(50)",
    "username": "VARCHAR(30)",
    "phone": "VARCHAR(20)",
    "bio": "VARCHAR(500)",
    "avatar_url": "VARCHAR(255)",
    "updated_at": "DATETIME",
}


def migrate() -> None:
    conn = sqlite3.connect(_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(users)")
        existing = {row[1] for row in cur.fetchall()}

        for name, ddl in _COLUMNS.items():
            if name in existing:
                print(f"'{name}' already exists on 'users' — skipping.")
            else:
                cur.execute(f"ALTER TABLE users ADD COLUMN {name} {ddl}")
                print(f"Added column '{name}' to 'users'.")

        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_username ON users (username)"
        )
        cur.execute("UPDATE users SET updated_at = created_at WHERE updated_at IS NULL")
        conn.commit()
        print("Done.")
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
