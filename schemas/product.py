from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ProductBase(BaseModel):
    name: str = Field(min_length=2, max_length=200)
    description: Optional[str] = None
    price: float = Field(gt=0)
    stock: int = Field(ge=0)
    image_url: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None


class ProductCreate(ProductBase):
    pass


class ProductUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=200)
    description: Optional[str] = None
    price: Optional[float] = Field(default=None, gt=0)
    stock: Optional[int] = Field(default=None, ge=0)
    image_url: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None


class ProductOut(ProductBase):
    id: int
    # The owning seller's user id. None for legacy products created before
    # ownership existed. Never accepted as client input on
    # ProductCreate/ProductUpdate above — always set server-side from the
    # authenticated current user (see api/products.py).
    seller_id: Optional[int] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
