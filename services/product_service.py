from typing import Optional

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from core.text_normalization import contains_arabic, normalize_arabic, sql_fold
from models.product import Product
from schemas.product import ProductCreate, ProductUpdate
from services import product_search_service


def get_products(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    q: Optional[str] = None,
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
):
    """Catalogue listing with optional filters, all applied by the database.

    - q: every word must occur (case-insensitive, Arabic spelling variants folded) in
      name, description, category or brand. No word is dropped (see
      `product_search_service.literal_terms`).
    - category: exact, case-insensitive match (Arabic-folded when the value is Arabic).
    - min_price / max_price: inclusive bounds.
    Callers validate the values (api/products.py). Ordered by id so skip/limit pages
    are stable. Only the requested page is loaded.
    """
    query = db.query(Product)
    conditions = []

    for term in product_search_service.literal_terms(q or ""):
        conditions.append(product_search_service.term_condition(term))

    category = (category or "").strip()
    if category:
        if contains_arabic(category):
            conditions.append(sql_fold(func.lower(Product.category)) == normalize_arabic(category).lower())
        else:
            conditions.append(func.lower(Product.category) == category.lower())

    if min_price is not None:
        conditions.append(Product.price >= min_price)
    if max_price is not None:
        conditions.append(Product.price <= max_price)

    if conditions:
        query = query.filter(and_(*conditions))
    return query.order_by(Product.id).offset(skip).limit(limit).all()


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
