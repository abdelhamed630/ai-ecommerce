from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class CartItemAdd(BaseModel):
    product_id: int
    quantity: int = Field(gt=0)


class CartItemUpdate(BaseModel):
    quantity: int = Field(gt=0)


class ProductMini(BaseModel):
    """Minimal product info embedded inside a cart item."""

    id: int
    name: str
    price: float

    model_config = ConfigDict(from_attributes=True)


class CartItemOut(BaseModel):
    id: int
    product_id: int
    quantity: int
    product: ProductMini
    subtotal: float

    model_config = ConfigDict(from_attributes=True)


class CartOut(BaseModel):
    id: int
    user_id: int
    items: List[CartItemOut]
    total_items: int
    total_price: float
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
