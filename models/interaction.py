import enum

from sqlalchemy import Column, DateTime, Enum, ForeignKey, Index, Integer, func
from sqlalchemy.orm import relationship

from database.database import Base


class InteractionType(str, enum.Enum):
    VIEW = "VIEW"
    CART_ADD = "CART_ADD"
    PURCHASE = "PURCHASE"


class ProductInteraction(Base):
    __tablename__ = "product_interactions"
    __table_args__ = (
        # Composite indexes for the query shapes a recommendation pipeline
        # will actually run: "this user's history", "this product's activity",
        # and "all X-type events over time".
        Index("ix_interaction_user_product", "user_id", "product_id"),
        Index("ix_interaction_product_type", "product_id", "interaction_type"),
        Index("ix_interaction_type_created_at", "interaction_type", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    interaction_type = Column(
        Enum(InteractionType, native_enum=False, length=20),
        nullable=False,
        index=True,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # One-directional relationships only — nothing added to User/Product models.
    user = relationship("User")
    product = relationship("Product")
