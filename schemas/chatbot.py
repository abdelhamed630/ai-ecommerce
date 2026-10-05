from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChatTurn(BaseModel):
    """One earlier message of the conversation, as kept by the client."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    # Optional conversation memory for follow-ups ("and how much is it?"). The server
    # is stateless: the client sends the previous turns (oldest first) and the server
    # only uses the last CHATBOT_MAX_HISTORY_TURNS of them. Omit it for a fresh chat.
    history: List[ChatTurn] = Field(default_factory=list, max_length=50)


class ChatProduct(BaseModel):
    """Product info exposed to the shopping assistant.

    Deliberately a subset of `ProductOut`: no `seller_id` and no
    `created_at`, since the chatbot response does not need them.
    """

    id: int
    name: str
    description: Optional[str] = None
    price: float
    stock: int
    image_url: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ChatResponse(BaseModel):
    answer: str
    products: List[ChatProduct]
