from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.security import get_current_user
from database.database import get_db
from models.interaction import InteractionType
from models.user import User
from schemas.cart import CartItemAdd, CartItemUpdate, CartOut
from services import cart_service, interaction_service

router = APIRouter(prefix="/cart", tags=["Cart"])


@router.get("/", response_model=CartOut)
def read_cart(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    return cart_service.get_cart(db, current_user.id)


@router.post("/items", response_model=CartOut)
def add_item(
    payload: CartItemAdd,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    result = cart_service.add_item_to_cart(db, current_user.id, payload)

    # Best-effort interaction logging — recorded only after the cart write
    # has already succeeded, and never allowed to affect the cart response.
    try:
        interaction_service.record_interaction(
            db, current_user.id, payload.product_id, InteractionType.CART_ADD
        )
    except Exception:
        db.rollback()

    return result


@router.patch("/items/{item_id}", response_model=CartOut)
def update_item(
    item_id: int,
    payload: CartItemUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return cart_service.update_cart_item(db, current_user.id, item_id, payload.quantity)


@router.delete("/items/{item_id}", response_model=CartOut)
def delete_item(
    item_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return cart_service.remove_cart_item(db, current_user.id, item_id)


@router.delete("/", response_model=CartOut)
def clear(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    return cart_service.clear_cart(db, current_user.id)
