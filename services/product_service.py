from typing import Optional

from sqlalchemy.orm import Session

from models.product import Product
from schemas.product import ProductCreate, ProductUpdate


def get_products(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Product).offset(skip).limit(limit).all()


def get_product(db: Session, product_id: int):
    return db.query(Product).filter(Product.id == product_id).first()


def create_product(
    db: Session, product: ProductCreate, seller_id: Optional[int]
) -> Product:
    # seller_id always comes from the authenticated current user's role
    # (see api/products.py) — SELLER -> their own id, ADMIN -> None.
    # ProductCreate has no seller_id field, so a client can never set
    # another user's id as the owner.
    db_product = Product(**product.model_dump(), seller_id=seller_id)
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product


def update_product(db: Session, product_id: int, product: ProductUpdate):
    db_product = get_product(db, product_id)
    if not db_product:
        return None

    update_data = product.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_product, field, value)

    db.commit()
    db.refresh(db_product)
    return db_product


def delete_product(db: Session, product_id: int):
    db_product = get_product(db, product_id)
    if not db_product:
        return None

    db.delete(db_product)
    db.commit()
    return db_product
