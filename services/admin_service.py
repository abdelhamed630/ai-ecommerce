from sqlalchemy.orm import Session

from models.user import User, UserRole


def list_users(
    db: Session,
    *,
    search: str | None = None,
    role: UserRole | None = None,
    is_active: bool | None = None,
    skip: int = 0,
    limit: int = 50,
):
    query = db.query(User)

    if search:
        search_value = f"%{search.strip()}%"
        query = query.filter(
            (User.email.ilike(search_value))
            | (User.full_name.ilike(search_value))
            | (User.username.ilike(search_value))
        )

    if role is not None:
        query = query.filter(User.role == role)

    if is_active is not None:
        query = query.filter(User.is_active == is_active)

    return (
        query.order_by(User.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).first()


def update_user_role(
    db: Session,
    user: User,
    role: UserRole,
) -> User:
    user.role = role
    db.commit()
    db.refresh(user)
    return user


def update_user_status(
    db: Session,
    user: User,
    is_active: bool,
) -> User:
    user.is_active = is_active
    db.commit()
    db.refresh(user)
    return user