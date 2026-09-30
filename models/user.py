import enum

from sqlalchemy import Boolean, Column, DateTime, Enum, Integer, String, func

from database.database import Base


class UserRole(str, enum.Enum):
    """The application's single canonical role enum.

    Every authorization check in the codebase (core/security.py,
    api/products.py, api/segmentation.py, scripts/set_user_role.py) is
    written against *this* definition — no other module should declare a
    second, competing role enum.
    """

    USER = "USER"
    SELLER = "SELLER"
    ADMIN = "ADMIN"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    # Legacy single-field name captured at registration. Kept as-is (not
    # duplicated/removed); the profile system adds first_name/last_name
    # below and does not sync the two.
    full_name = Column(String, nullable=True)

    # --- Profile fields (all nullable; existing users have none of them) ---
    first_name = Column(String(50), nullable=True)
    last_name = Column(String(50), nullable=True)
    # Unique, stored lowercase. Nullable because existing users predate it
    # (multiple NULLs are allowed by a UNIQUE index in SQLite).
    username = Column(String(30), unique=True, index=True, nullable=True)
    phone = Column(String(20), nullable=True)
    bio = Column(String(500), nullable=True)
    # A relative URL/path reference (e.g. /media/avatars/user1_<uuid>.png).
    # Image bytes are NEVER stored in the database.
    avatar_url = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    # Authorization role. Every self-registered (public /auth/register)
    # user is USER — there is no API that accepts a client-supplied role,
    # and public registration never trusts a client-provided role/is_admin
    # field. SELLER and ADMIN can only be assigned out-of-band, at the
    # database level, via scripts/set_user_role.py.
    #
    # Replaces the old `is_admin` boolean. If this application's database
    # still has an `is_admin` column from before this change, see
    # scripts/migrate_is_admin_to_role.py to migrate existing rows
    # (is_admin=True -> ADMIN, is_admin=False -> USER) without losing any
    # users. `role` is the single source of truth for authorization from
    # this point on — `is_admin` is no longer read anywhere in the
    # application.
    role = Column(
        Enum(UserRole, native_enum=False, length=20),
        nullable=False,
        default=UserRole.USER,
        server_default=UserRole.USER.value,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
