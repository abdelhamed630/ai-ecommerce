from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.security import hash_password, verify_password
from models.user import User
from schemas.user import UserCreate


def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, user: UserCreate) -> User:
    db_user = User(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hash_password(user.password),
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


# --- Profile ---------------------------------------------------------------

class UsernameTakenError(Exception):
    pass


class InvalidCurrentPasswordError(Exception):
    pass


class SamePasswordError(Exception):
    pass


def update_profile(db: Session, user: User, update) -> User:
    """Apply only the fields the client actually sent. `update` is a
    schemas.profile.UserProfileUpdate, which contains no role/is_admin/
    password/email/id field, so none of those can reach this function."""
    data = update.model_dump(exclude_unset=True)

    new_username = data.get("username")
    if new_username and new_username != user.username:
        clash = (
            db.query(User)
            .filter(User.username == new_username, User.id != user.id)
            .first()
        )
        if clash:
            raise UsernameTakenError()

    for field, value in data.items():
        setattr(user, field, value)

    try:
        db.commit()
    except IntegrityError:  # concurrent request grabbed the username first
        db.rollback()
        raise UsernameTakenError()
    db.refresh(user)
    return user


def change_password(db: Session, user: User, current_password: str, new_password: str) -> None:
    if not verify_password(current_password, user.hashed_password):
        raise InvalidCurrentPasswordError()
    if verify_password(new_password, user.hashed_password):
        raise SamePasswordError()
    user.hashed_password = hash_password(new_password)
    db.commit()
