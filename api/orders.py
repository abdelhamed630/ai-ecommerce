from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.security import (
    get_current_admin_or_seller_user,
    get_current_user,
)
from database.database import get_db
from models.interaction import InteractionType
from models.user import User
from schemas.order import OrderOut
from services import interaction_service, order_service

router = APIRouter(prefix="/orders", tags=["Orders"])


@router.post("/", response_model=OrderOut)
def create_order(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    order = order_service.create_order_from_cart(db, current_user.id)

    # Best-effort interaction logging — recorded only after the order's own
    # atomic transaction has already committed successfully, and a logging
    # failure here can never roll back or affect the order itself.
    try:
        for item in order.items:
            interaction_service.record_interaction(
                db, current_user.id, item.product_id, InteractionType.PURCHASE
            )
    except Exception:
        db.rollback()

    return order


@router.get("/", response_model=List[OrderOut])
def list_my_orders(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    return order_service.get_orders(db, current_user.id)


@router.get("/{order_id}", response_model=OrderOut)
def get_my_order(
    order_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return order_service.get_order(db, current_user.id, order_id)


@router.patch("/{order_id}/cancel", response_model=OrderOut)
def cancel_order(
    order_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return order_service.cancel_order(db, current_user.id, order_id)


@router.patch("/{order_id}/complete", response_model=OrderOut)
def advance_order(
    order_id: int,
    current_user: User = Depends(get_current_admin_or_seller_user),
    db: Session = Depends(get_db),
):
    return order_service.advance_order(
        db,
        current_user,
        order_id,
    )