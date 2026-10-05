from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from core.rate_limit import chatbot_rate_limit
from database.database import get_db
from models.user import User
from schemas.chatbot import ChatMessageRequest, ChatResponse
from services import chatbot_service

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatMessageRequest,
    # Authenticates (401) and rate-limits per user (429), see core/rate_limit.py.
    current_user: User = Depends(chatbot_rate_limit),
    db: Session = Depends(get_db),
):
    return chatbot_service.get_chat_response(
        db, payload.message, history=payload.history, user_id=current_user.id
    )


@router.post("/chat/stream")
def chat_stream(
    payload: ChatMessageRequest,
    # Same dependency as /chat: JWT authentication, then the per-user rate limit (429
    # BEFORE anything is looked up or sent to the LLM). Both endpoints share one counter.
    current_user: User = Depends(chatbot_rate_limit),
    db: Session = Depends(get_db),
):
    """Server-Sent Events version of /chat (see chatbot_service.stream_chat_events).

    The database is read here, before the response starts, so the DB session is never
    held open while streaming. /chat is unchanged.
    """
    context = chatbot_service.build_context(
        db, payload.message, history=payload.history, user_id=current_user.id
    )
    return StreamingResponse(
        chatbot_service.stream_chat_events(context),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # tell nginx not to buffer this response
        },
    )
