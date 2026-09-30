import enum

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from database.database import Base


class PaymentStatus(str, enum.Enum):
    PENDING = "PENDING"
    PAID = "PAID"
    FAILED = "FAILED"
    REFUNDED = "REFUNDED"


class PaymentMethod(str, enum.Enum):
    CASH_ON_DELIVERY = "CASH_ON_DELIVERY"
    CARD = "CARD"


class Payment(Base):
    __tablename__ = "payments"
    __table_args__ = (
        # A transaction_reference, when present, must be globally unique.
        UniqueConstraint("transaction_reference", name="uq_payment_transaction_reference"),
        CheckConstraint("amount > 0", name="ck_payment_amount_positive"),
    )

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Snapshot of Order.total_price at payment-creation time — never trusted
    # from the client.
    amount = Column(Float, nullable=False)

    status = Column(
        Enum(PaymentStatus, native_enum=False, length=20),
        nullable=False,
        default=PaymentStatus.PENDING,
    )
    payment_method = Column(
        Enum(PaymentMethod, native_enum=False, length=30),
        nullable=False,
    )

    # No sensitive card data is ever stored here — only an opaque reference
    # a real payment provider would hand back (e.g. a charge/intent id).
    transaction_reference = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # One-directional relationships only — nothing added to Order/User models.
    order = relationship("Order")
    user = relationship("User")
