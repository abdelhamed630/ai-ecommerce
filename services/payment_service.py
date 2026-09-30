import uuid

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from models.order import Order, OrderStatus
from models.payment import Payment, PaymentMethod, PaymentStatus
from schemas.payment import PaymentCreate

# Allowed state transitions for payments.
_ALLOWED_TRANSITIONS = {
    ("pay", PaymentStatus.PENDING): PaymentStatus.PAID,
    ("fail", PaymentStatus.PENDING): PaymentStatus.FAILED,
    ("refund", PaymentStatus.PAID): PaymentStatus.REFUNDED,
}


def _get_owned_order_or_404(db: Session, user_id: int, order_id: int) -> Order:
    order = (
        db.query(Order)
        .filter(Order.id == order_id, Order.user_id == user_id)
        .first()
    )
    if not order:
        # Same 404 whether the order doesn't exist or belongs to someone else.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")
    return order


def _get_owned_payment_or_404(db: Session, user_id: int, payment_id: int) -> Payment:
    payment = (
        db.query(Payment)
        .filter(Payment.id == payment_id, Payment.user_id == user_id)
        .first()
    )
    if not payment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Payment not found")
    return payment


def create_payment(db: Session, user_id: int, payload: PaymentCreate) -> Payment:
    order = _get_owned_order_or_404(db, user_id, payload.order_id)

    if order.status == OrderStatus.CANCELLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot create a payment for a cancelled order",
        )

    existing_paid = (
        db.query(Payment)
        .filter(Payment.order_id == order.id, Payment.status == PaymentStatus.PAID)
        .first()
    )
    if existing_paid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This order already has a successful payment",
        )

    try:
        payment = Payment(
            order_id=order.id,
            user_id=user_id,
            # Amount is always derived from the order on the server — never
            # trusted from the client.
            amount=order.total_price,
            status=PaymentStatus.PENDING,
            payment_method=payload.payment_method,
            transaction_reference=None,
        )
        db.add(payment)
        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(payment)
    return payment


def get_payment(db: Session, user_id: int, payment_id: int) -> Payment:
    return _get_owned_payment_or_404(db, user_id, payment_id)


def get_order_payment(db: Session, user_id: int, order_id: int) -> Payment:
    order = _get_owned_order_or_404(db, user_id, order_id)

    payment = db.query(Payment).filter(Payment.order_id == order.id).first()
    if not payment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Payment not found")
    return payment


def _apply_transition(db: Session, payment: Payment, action: str) -> Payment:
    next_status = _ALLOWED_TRANSITIONS.get((action, payment.status))
    if next_status is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Cannot transition payment from '{payment.status.value}' "
                f"using action '{action}'"
            ),
        )

    try:
        payment.status = next_status
        if next_status == PaymentStatus.PAID and not payment.transaction_reference:
            # Simulated provider reference — no real gateway involved yet.
            payment.transaction_reference = f"SIMULATED-{uuid.uuid4().hex[:12]}"
        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(payment)
    return payment


def mark_payment_paid(db: Session, payment_id: int) -> Payment:
    payment = db.query(Payment).filter(Payment.id == payment_id).first()

    if not payment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payment not found",
        )

    return _apply_transition(db, payment, "pay")


def mark_payment_failed(db: Session, payment_id: int) -> Payment:
    payment = db.query(Payment).filter(Payment.id == payment_id).first()

    if not payment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payment not found",
        )

    return _apply_transition(db, payment, "fail")


def refund_payment(db: Session, payment_id: int) -> Payment:
    payment = db.query(Payment).filter(Payment.id == payment_id).first()

    if not payment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payment not found",
        )

    return _apply_transition(db, payment, "refund")