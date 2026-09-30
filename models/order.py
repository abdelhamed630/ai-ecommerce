import enum

from sqlalchemy import Column, DateTime, Enum, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import relationship

from database.database import Base


class OrderStatus(str, enum.Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (
        # Churn and segmentation aggregate over COMPLETED orders within a time
        # window (`status = 'COMPLETED' AND created_at <= cutoff`, then
        # GROUP BY user_id) — see services/churn_service.py and
        # services/segmentation_analysis_service.py.
        Index("ix_orders_status_created_at", "status", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    status = Column(
        Enum(OrderStatus, native_enum=False, length=20),
        nullable=False,
        default=OrderStatus.PENDING,
    )
    # Snapshot of the order's total at creation time — never recomputed from
    # current product prices.
    total_price = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # One-directional relationships only — nothing added to the User/Product models.
    user = relationship("User")
    items = relationship("OrderItem", cascade="all, delete-orphan")


class OrderItem(Base):
    __tablename__ = "order_items"
    __table_args__ = (
        # order_items.order_id is the foreign key used to load an order's items
        # (Order.items) and by the RFM adapter's Order JOIN OrderItem
        # (ml/segmentation/adapters.py). PostgreSQL does not index FKs itself.
        Index("ix_order_items_order_id", "order_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)

    # Snapshot fields — captured at order-creation time so a later change to
    # the product's name/price never affects a past order.
    product_name = Column(String, nullable=False)
    product_price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    subtotal = Column(Float, nullable=False)

    product = relationship("Product")
