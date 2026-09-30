from pydantic import BaseModel, ConfigDict, Field

from models.user import UserRole


class AdminUserOut(BaseModel):
    id: int
    email: str
    full_name: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    username: str | None = None
    phone: str | None = None
    bio: str | None = None
    avatar_url: str | None = None
    is_active: bool
    role: UserRole

    model_config = ConfigDict(from_attributes=True)


class AdminUserRoleUpdate(BaseModel):
    role: UserRole


class AdminUserStatusUpdate(BaseModel):
    is_active: bool