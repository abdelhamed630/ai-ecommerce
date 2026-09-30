from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from core.security import get_current_user
from database.database import get_db
from models.user import User
from schemas.profile import PasswordChange, UserProfileResponse, UserProfileUpdate
from services import avatar_service, user_service

router = APIRouter(prefix="/users", tags=["Users"])

# Every route below operates on the authenticated user ONLY (`/me`). There is
# deliberately no /users/{id} route: reading or editing another user's
# profile is out of scope (admin user management is a separate phase).


@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Get my profile",
    description="Returns the authenticated user's own profile. Available to USER, SELLER and ADMIN.",
)
def get_my_profile(current_user: User = Depends(get_current_user)):
    return current_user


@router.patch(
    "/me",
    response_model=UserProfileResponse,
    summary="Update my profile",
    description=(
        "Updates first_name, last_name, username, phone and/or bio. Unknown "
        "fields (role, is_admin, email, hashed_password, ...) are rejected "
        "with 422. A username already in use returns 400."
    ),
)
def update_my_profile(
    update: UserProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return user_service.update_profile(db, current_user, update)
    except user_service.UsernameTakenError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken")


@router.patch(
    "/me/password",
    summary="Change my password",
    description=(
        "Requires the current password. Existing JWTs are NOT revoked "
        "(tokens are stateless); see docs/profile.md."
    ),
)
def change_my_password(
    payload: PasswordChange,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        user_service.change_password(
            db, current_user, payload.current_password, payload.new_password
        )
    except user_service.InvalidCurrentPasswordError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect")
    except user_service.SamePasswordError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from the current password",
        )
    return {"message": "Password updated successfully"}


@router.post(
    "/me/avatar",
    response_model=UserProfileResponse,
    summary="Upload my avatar",
    description=(
        "multipart/form-data with a `file` part. JPEG, PNG or WEBP, max 2 MB. "
        "The real format is verified from the file's content, not its name."
    ),
)
async def upload_my_avatar(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if file.content_type not in avatar_service.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only JPEG, PNG and WEBP images are allowed",
        )

    data = await file.read(avatar_service.MAX_AVATAR_BYTES + 1)
    if len(data) > avatar_service.MAX_AVATAR_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Avatar must be at most 2 MB",
        )
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    detected = avatar_service.detect_image_type(data)
    if detected is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File content is not a valid JPEG, PNG or WEBP image",
        )

    old_url = current_user.avatar_url
    current_user.avatar_url = avatar_service.save_avatar(current_user.id, data, detected)
    db.commit()
    db.refresh(current_user)
    avatar_service.remove_avatar_file(old_url)  # only after the new one is committed
    return current_user


@router.delete(
    "/me/avatar",
    summary="Delete my avatar",
    description="Removes the avatar reference and stored file. Safe to call when no avatar exists.",
)
def delete_my_avatar(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    old_url = current_user.avatar_url
    if old_url:
        current_user.avatar_url = None
        db.commit()
        avatar_service.remove_avatar_file(old_url)
    return {"message": "Avatar removed"}
