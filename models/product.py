from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import relationship

from database.database import Base


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    # Indexed for GET /products/?min_price=&max_price= range filters (migration 0003).
    price = Column(Float, nullable=False, index=True)
    stock = Column(Integer, default=0)
    image_url = Column(String, nullable=True)
    # Nullable by design: existing products predate these fields and must
    # not be forced to backfill them. Indexed to support future catalog
    # filtering (e.g. "browse by category" or "browse by brand").
    category = Column(String, nullable=True, index=True)
    brand = Column(String, nullable=True, index=True)
    # The SELLER who owns this product. Set from the authenticated
    # current user at creation time (api/products.py) when that user is a
    # SELLER — never trusted from client input. NULL for two distinct
    # cases: (1) a product created by an ADMIN, since ADMIN is a system
    # administrator role, not automatically a seller/vendor; and (2) a
    # legacy product created before this column existed (see
    # scripts/migrate_add_product_seller_id.py). Either way, a NULL-owner
    # product can only be managed by an ADMIN until explicitly assigned a
    # seller.
    seller_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # One-directional relationship only — nothing added back on User.
    seller = relationship("User")
