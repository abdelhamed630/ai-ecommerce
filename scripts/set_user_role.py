"""
Manual, explicit role assignment for a single existing user.

WHY THIS SCRIPT EXISTS
-----------------------
There is intentionally no HTTP endpoint that lets a client set their own
(or anyone else's) role to SELLER or ADMIN — self-service privilege
escalation would be a security hole. Public registration (`POST
/auth/register`) always creates a USER, full stop.

For now, promoting a user to SELLER or ADMIN is a deliberate,
out-of-band, database-level operation, the same way admin provisioning
already worked before this change (see the now-superseded
scripts/migrate_add_user_is_admin.py). This script is that operation for
the new three-role system: it requires you to name an exact existing user
and an exact target role — there is no bulk mode and no "promote everyone"
option.

WHAT IT DOES
------------
- Looks up the user by email.
- If found, sets `role` to the given value and prints the before/after
  role in an easy to skim way.
- If not found, prints a clear error and makes no changes.
- Idempotent: running it again with the same email/role is a harmless
  no-op that reports the user already has that role.
- Never creates a user, never deletes a user, never touches any other
  table.

USAGE
-----
    python scripts/set_user_role.py --email someone@example.com --role ADMIN
    python scripts/set_user_role.py --email someone@example.com --role SELLER
    python scripts/set_user_role.py --email someone@example.com --role USER

Run this from the project root (so `core.config`, `database`, and
`models` are importable), the same way you would run the app itself.
"""

import argparse
import os
import sys
# Make the project root importable when run as `python scripts/<name>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from models.user import User, UserRole


def set_user_role(email: str, role: UserRole) -> None:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            print(f"No user found with email '{email}'. No changes made.")
            return

        if user.role == role:
            print(f"User '{email}' already has role {role.value} — nothing to do.")
            return

        previous_role = user.role
        user.role = role
        db.commit()
        print(f"User '{email}': {previous_role.value} -> {role.value}")
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicitly set an existing user's role. Not exposed via any API."
    )
    parser.add_argument(
        "--email", required=True, help="Exact email of the user to update."
    )
    parser.add_argument(
        "--role",
        required=True,
        choices=[role.value for role in UserRole],
        help="Target role.",
    )
    args = parser.parse_args()

    set_user_role(args.email, UserRole(args.role))


if __name__ == "__main__":
    main()
