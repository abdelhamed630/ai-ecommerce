from datetime import datetime

from pydantic import BaseModel, ConfigDict

from models.interaction import InteractionType


class InteractionCreate(BaseModel):
    product_id: int
    interaction_type: InteractionType
    # user_id is intentionally NOT accepted here — it is always derived
    # from the authenticated JWT user, never trusted from the client.


class InteractionOut(BaseModel):
    id: int
    user_id: int
    product_id: int
    interaction_type: InteractionType
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
