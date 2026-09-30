import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

from models.user import UserRole

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
_PHONE_RE = re.compile(r"^\+?[0-9][0-9 ()\-]{5,18}[0-9]$")


class UserProfileResponse(BaseModel):
    """A user's own profile. Never includes hashed_password."""

    id: int
    email: EmailStr
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None  # legacy registration field, read-only
    phone: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    role: UserRole
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class UserProfileUpdate(BaseModel):
    """Only legitimate, self-editable profile fields.

    `extra="forbid"`: sending role, is_admin, hashed_password, id, email,
    created_at, updated_at (or anything else unknown) is rejected with 422
    instead of being silently ignored. Email is intentionally NOT editable
    (see docs/profile.md: the JWT `sub` claim is the email).
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    first_name: Optional[str] = Field(default=None, max_length=50)
    last_name: Optional[str] = Field(default=None, max_length=50)
    username: Optional[str] = Field(default=None, max_length=30)
    phone: Optional[str] = Field(default=None, max_length=20)
    bio: Optional[str] = Field(default=None, max_length=500)

    @field_validator("first_name", "last_name", "bio")
    @classmethod
    def _blank_to_none(cls, v):
        # Sending "" or null clears the field.
        return v or None

    @field_validator("username")
    @classmethod
    def _validate_username(cls, v):
        if v is None:
            raise ValueError("username cannot be cleared")
        if not _USERNAME_RE.match(v):
            raise ValueError(
                "username must be 3-30 characters: letters, digits, '.', '_' or '-'"
            )
        return v.lower()

    @field_validator("phone")
    @classmethod
    def _validate_phone(cls, v):
        if not v:
            return None
        if not _PHONE_RE.match(v):
            raise ValueError("phone must be 7-20 characters: digits, spaces, (), -, optional leading +")
        return v


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)
    confirm_new_password: str

    @field_validator("new_password")
    @classmethod
    def _validate_strength(cls, v):
        if len(v.encode("utf-8")) > 72:
            raise ValueError("password must be at most 72 bytes (bcrypt limit)")
        if not (re.search(r"[A-Za-z]", v) and re.search(r"[0-9]", v)):
            raise ValueError("password must contain at least one letter and one digit")
        return v

    @model_validator(mode="after")
    def _passwords_match(self) -> "PasswordChange":
        if self.new_password != self.confirm_new_password:
            raise ValueError("Passwords do not match")
        return self
