from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from models.payment import PaymentMethod, PaymentStatus


class PaymentCreate(BaseModel):
    order_id: int
    payment_method: PaymentMethod
    # amount is intentionally NOT accepted from the client — it is always
    # derived server-side from Order.total_price.


class PaymentOut(BaseModel):
    id: int
    order_id: int
    user_id: int
    amount: float
    status: PaymentStatus
    payment_method: PaymentMethod
    transaction_reference: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
