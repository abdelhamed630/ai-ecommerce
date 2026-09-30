from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from core.config import settings
from database.database import get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    from models.user import User  # local import avoids circular import

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
        email: Optional[str] = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user"
        )

    return user


def _require_role(current_user, allowed_roles) -> "User":  # noqa: F821 (str annotation)
    """Shared authorization check used by every role-gated dependency
    below. Deliberately NOT itself a FastAPI dependency (no `Depends`
    default) so it has exactly one implementation, reused instead of
    copied — the dependencies below just decide *which* roles to pass it.
    """
    if current_user.role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to perform this action.",
        )
    return current_user


def require_roles(*allowed_roles):
    """Generic, reusable authorization dependency *factory*.

    Usage: ``Depends(require_roles(UserRole.ADMIN, UserRole.SELLER))``.
    Keeps authentication (JWT -> current user, via `get_current_user`)
    and authorization (current user -> role check) as separate concerns:
    this only adds a role check on top of the existing authentication
    dependency, the same way `get_current_admin_user` always has.
    """

    def _dependency(current_user=Depends(get_current_user)):
        return _require_role(current_user, allowed_roles)

    return _dependency


def get_current_admin_user(current_user=Depends(get_current_user)):
    """Like `get_current_user`, but also requires the ADMIN role.

    Builds directly on `get_current_user` (same JWT decoding, same 401 on
    a missing/invalid/expired token) rather than introducing a second
    authentication mechanism — this only adds one extra check on top of
    the project's existing authentication dependency. Used to gate
    operationally sensitive, non-self-service endpoints such as
    POST /segmentation/run. Equivalent to ``require_roles(UserRole.ADMIN)``;
    kept as its own named dependency since it's used at nearly every call
    site that needs "admin only".
    """
    from models.user import UserRole

    return _require_role(current_user, (UserRole.ADMIN,))


def get_current_seller_user(current_user=Depends(get_current_user)):
    """Like `get_current_user`, but also requires the SELLER role."""
    from models.user import UserRole

    return _require_role(current_user, (UserRole.SELLER,))


def get_current_admin_or_seller_user(current_user=Depends(get_current_user)):
    """Like `get_current_user`, but requires the ADMIN or SELLER role.

    Used for endpoints (e.g. product create/update/delete) that both
    roles may call, where a further per-resource ownership check —
    "SELLER only for their own product" — happens separately, inside the
    endpoint, once the specific resource has been loaded.
    """
    from models.user import UserRole

    return _require_role(current_user, (UserRole.ADMIN, UserRole.SELLER))
