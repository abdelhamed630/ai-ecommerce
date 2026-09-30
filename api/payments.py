from fastapi import APIRouter, Depends

from sqlalchemy.orm import Session

from core.security import (
    get_current_admin_user,
    get_current_user,
)

from database.database import get_db

from models.user import User

from schemas.payment import PaymentCreate, PaymentOut

from services import payment_service


router = APIRouter(prefix="/payments", tags=["Payments"])


@router.post("/", response_model=PaymentOut)
def create_payment(
    payload: PaymentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return payment_service.create_payment(
        db,
        current_user.id,
        payload,
    )


@router.get("/order/{order_id}", response_model=PaymentOut)
def get_order_payment(
    order_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return payment_service.get_order_payment(
        db,
        current_user.id,
        order_id,
    )


@router.get("/{payment_id}", response_model=PaymentOut)
def get_payment(
    payment_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return payment_service.get_payment(
        db,
        current_user.id,
        payment_id,
    )


@router.patch("/{payment_id}/pay", response_model=PaymentOut)
def pay_payment(
    payment_id: int,
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    return payment_service.mark_payment_paid(
        db,
        payment_id,
    )


@router.patch("/{payment_id}/fail", response_model=PaymentOut)
def fail_payment(
    payment_id: int,
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    return payment_service.mark_payment_failed(
        db,
        payment_id,
    )


@router.patch("/{payment_id}/refund", response_model=PaymentOut)
def refund_payment(
    payment_id: int,
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    return payment_service.refund_payment(
        db,
        payment_id,
    )