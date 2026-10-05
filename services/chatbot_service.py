"""Shopping chatbot service: real products from the database + an LLM answer.

    message (+ optional history) -> intent
        "recommend me something"   -> recommendation engine (personalized, per user)
        anything else              -> product_search_service (PostgreSQL)
                                   -> real Product rows
            -> llm_service (phrases the answer from that data only)
            -> ChatResponse

The `products` in the response always come from the database. The LLM only
writes the natural-language `answer`; it is never a source of product facts.
If the LLM is unavailable the endpoint still succeeds: it returns the real
products with a fixed "temporarily unavailable" answer, in the customer's
language (Arabic or English).

Catalogue questions ("What products do you have?", "ايه المنتجات الموجودة؟")
and specific-product questions are both handled by
`product_search_service.search_products`; no LLM is involved in fetching.

Conversation memory: the server stays stateless. The client may send the
previous turns as `history`; they are (a) passed to the LLM so it can follow
up ("and how much is it?") and (b) used to re-run the search with the
customer's earlier message when the new message alone matches nothing.
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from sqlalchemy.orm import Session

from core.config import settings
from schemas.chatbot import ChatProduct, ChatResponse
from services import llm_service, product_search_service, recommendation_cache_service

logger = logging.getLogger(__name__)

UNAVAILABLE_ANSWER = "The AI assistant is temporarily unavailable. Please try again."
UNAVAILABLE_ANSWER_AR = "المساعد الذكي غير متاح مؤقتًا. من فضلك حاول مرة أخرى."

RECOMMENDATION_LIMIT = 5
# How many earlier customer messages may be re-used to find the product a follow-up is about.
FOLLOW_UP_LOOKBACK = 3


def _clean_history(history: Optional[Sequence[Any]]) -> List[Dict[str, str]]:
    """Plain {"role","content"} dicts, last CHATBOT_MAX_HISTORY_TURNS only."""
    limit = settings.CHATBOT_MAX_HISTORY_TURNS
    if not history or limit <= 0:
        return []
    turns = []
    for turn in history:
        role = turn["role"] if isinstance(turn, Mapping) else turn.role
        content = turn["content"] if isinstance(turn, Mapping) else turn.content
        turns.append({"role": role, "content": content})
    return turns[-limit:]


def _recommended_products(db: Session, user_id: int) -> List[ChatProduct]:
    """Personalized picks for this user; [] if the engine fails (caller falls back to search)."""
    try:
        items = recommendation_cache_service.get_user_recommendations_cached(
            db, user_id, RECOMMENDATION_LIMIT
        )
        return [ChatProduct.model_validate(item["product"]) for item in items]
    except Exception:  # the chatbot must keep answering even if recommendations break
        logger.exception("chatbot: recommendations failed; falling back to search")
        return []


def _search(db: Session, message: str, history: List[Dict[str, str]]) -> List[ChatProduct]:
    rows = product_search_service.search_products(db, message)
    if not rows and history:
        # Follow-up such as "and how much is it?": look for the product the customer
        # talked about in their most recent earlier messages.
        earlier = [t["content"] for t in history if t["role"] == "user"][-FOLLOW_UP_LOOKBACK:]
        for previous in reversed(earlier):
            rows = product_search_service.search_products(db, previous)
            if rows:
                break
    return [ChatProduct.model_validate(row) for row in rows]


@dataclass
class ChatContext:
    """Everything the LLM step needs, already copied out of the database (plain data only)."""

    message: str
    products: List[ChatProduct] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    personalized: bool = False

    def llm_options(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {}
        if self.history:
            options["history"] = self.history
        if self.personalized:
            options["personalized"] = True
        return options

    def product_dicts(self) -> List[Dict[str, Any]]:
        return [product.model_dump() for product in self.products]


def build_context(
    db: Session,
    message: str,
    history: Optional[Sequence[Any]] = None,
    user_id: Optional[int] = None,
) -> ChatContext:
    """Intent detection + data lookup: recommendations OR product search (max 10 products).

    All database work happens here, before any LLM call, and the read transaction is
    ended so the connection returns to the pool during the slow LLM call.
    """
    history_turns = _clean_history(history)

    chat_products: List[ChatProduct] = []
    personalized = False
    if user_id is not None and product_search_service.is_recommendation_request(message):
        chat_products = _recommended_products(db, user_id)
        personalized = bool(chat_products)
    if not personalized:
        chat_products = _search(db, message, history_turns)

    # The rows are copied into plain schema objects; release the connection now.
    db.rollback()
    return ChatContext(message=message, products=chat_products, history=history_turns, personalized=personalized)


def _unavailable_answer(message: str) -> str:
    return UNAVAILABLE_ANSWER_AR if llm_service.detect_language(message) == "ar" else UNAVAILABLE_ANSWER


def get_chat_response(
    db: Session,
    message: str,
    history: Optional[Sequence[Any]] = None,
    user_id: Optional[int] = None,
) -> ChatResponse:
    context = build_context(db, message, history=history, user_id=user_id)

    try:
        answer = llm_service.generate_answer(message, context.product_dicts(), **context.llm_options())
    except llm_service.LLMError:
        # Already logged (without secrets) by llm_service; do not leak details to the client.
        answer = _unavailable_answer(message)

    return ChatResponse(answer=answer, products=context.products)


def _sse(event: str, data: Mapping[str, Any]) -> str:
    # json.dumps never emits a raw newline, so one `data:` line is always enough.
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_chat_events(context: ChatContext) -> Iterator[str]:
    """Server-Sent Events for POST /chatbot/chat/stream (all DB work is already done).

        event: products   {"products": [...]}   real DB rows, sent first
        event: token      {"text": "..."}       answer pieces, in order
        event: done       {}                    finished normally
        event: error      {"message": "..."}    LLM failed (fixed, localized text; no details)

    The stream always ends with `done` or `error`. If the consumer stops early (client
    disconnect) the generator is closed, which closes the provider stream too.
    """
    yield _sse("products", {"products": [product.model_dump() for product in context.products]})

    pieces = llm_service.stream_answer(context.message, context.product_dicts(), **context.llm_options())
    try:
        for piece in pieces:
            yield _sse("token", {"text": piece})
    except llm_service.LLMError:
        # Already logged (without secrets) by llm_service.
        yield _sse("error", {"message": _unavailable_answer(context.message)})
        return
    except Exception:  # never let an unexpected failure become a broken stream with a stack trace
        logger.exception("chatbot: unexpected streaming failure")
        yield _sse("error", {"message": _unavailable_answer(context.message)})
        return
    finally:
        close = getattr(pieces, "close", None)
        if callable(close):
            close()
    yield _sse("done", {})
