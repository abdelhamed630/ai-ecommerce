from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict

from models.order import OrderStatus


class OrderItemOut(BaseModel):
    id: int
    product_id: int
    product_name: str
    product_price: float
    quantity: int
    subtotal: float

    model_config = ConfigDict(from_attributes=True)


class OrderOut(BaseModel):
    id: int
    user_id: int
    status: OrderStatus
    items: List[OrderItemOut]
    total_price: float
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
