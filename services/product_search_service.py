"""Deterministic product search for the shopping chatbot (English + Arabic).

Searches the real `products` table with SQLAlchemy expressions only (every
user-supplied term is bound as a query parameter after LIKE-wildcard
escaping; nothing is interpolated into SQL strings). No LLM, no fake data.

Two kinds of message are recognised:

1. Catalogue query -- "What products do you have?", "ايه المنتجات الموجودة؟",
   "عندكم ايه؟", "What do you sell?" ... The message asks what the store has
   in general and names no specific product. The first MAX_RESULTS products
   are returned (ordered by id), out-of-stock ones included.

2. Specific search -- everything else. How a message is turned into a query:
     1. Lower-case it, strip Arabic diacritics/tatweel and split it into word
        tokens.
     2. Drop common filler words in English and Arabic ("do", "you", "have",
        "show", "me", "هل", "يوجد", "عندكم", "وريني", ...) and generic
        catalogue nouns ("products", "items", "المنتجات", ...).
     3. A trailing plural "s" is removed from longer words ("laptops" ->
        "laptop") so plurals match singular product names.
     4. Every remaining term must appear (case-insensitive substring) in at
        least one of: name, description, category, brand.
   Matching is tolerant of the usual Arabic spelling variants: both the term
   and the stored text are folded (hamza on alef, alef maqsura/yaa, taa
   marbuta/haa) before comparing, so "سماعه" finds "سماعة". The extracted
   terms themselves are not rewritten (see `extract_search_terms`).
3. Recommendation requests ("رشحلي حاجة", "recommend something") that name no
   specific product are detected by `is_recommendation_request`; the chatbot
   answers those from the personalized recommendation engine instead.

Results are ordered by product id and capped at MAX_RESULTS. Out-of-stock
products are returned too (the chatbot must report real stock, including 0).
"""
import re
from typing import FrozenSet, List

from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from core.text_normalization import ARABIC_MARKS_RE, normalize_arabic, sql_fold
from models.product import Product

MAX_RESULTS = 10
MAX_TERMS = 8
MAX_TERM_LENGTH = 50
MIN_TERM_LENGTH = 2

_LIKE_ESCAPE = "/"

_TOKEN_RE = re.compile(r"\w+")

# Arabic marks stripping and letter folding live in core.text_normalization (shared with
# the product catalogue filter). Folding is used ONLY to compare a token with the word
# lists below and to match terms against stored text.
_ARABIC_MARKS_RE = ARABIC_MARKS_RE
_ARABIC_DEFINITE_ARTICLE = "ال"

# Words are written here in their folded form (see _fold): no hamza on alef,
# final yaa (not alef maqsura), final haa (not taa marbuta), no diacritics.
_STOPWORDS_EN = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with",
        "about", "it", "this", "that", "is", "are", "was", "do", "does", "have",
        "has", "got", "there", "any", "some", "i", "me", "my", "we", "you", "your",
        "what", "whats", "how", "much", "many", "which", "where", "can", "could",
        "would", "will", "please", "show", "find", "search", "looking", "look",
        "want", "need", "get", "buy", "sell", "tell", "give", "list", "price",
        "prices", "cost", "available", "stock",
        # added for catalogue queries
        "sells", "selling", "all", "now", "currently",
    }
)

# Generic nouns meaning "the things you sell". They carry no product identity,
# so they are never used as search terms; on their own they make a catalogue query.
_CATALOG_NOUNS_EN = frozenset(
    {"product", "products", "item", "items", "goods", "merchandise", "catalog", "catalogue", "inventory"}
)
_CATALOG_NOUNS_AR = frozenset(
    {
        "منتج", "منتجات", "منتجاتكم", "منتجاتك", "بضاعه", "بضايع", "بضائع",
        "سلع", "سلعه", "اصناف", "صنف", "كتالوج", "كاتالوج",
    }
)

_STOPWORDS_AR = frozenset(
    {
        # questions / grammar
        "ايه", "اية", "اي", "ايش", "شو", "ما", "ماذا", "ماهي", "ماهو", "هي", "هو", "هل",
        # relative pronouns: "الي / اللي / التي / الذي" (the products THAT are available)
        "الي", "اللي", "التي", "الذي", "الذين", "اللتي",
        "في", "فيه", "من", "عن", "كل", "كام", "كم", "بكام", "لو", "ممكن",
        "سمحت", "فضلك", "لي", "لنا", "لدي", "عندي", "عندنا",
        # "do you have / is there"
        "عندكم", "عندكو", "عندكوا", "عندك", "لديكم", "لديك", "يوجد", "توجد", "فيها",
        # "available"
        "موجود", "موجوده", "موجودين", "متوفر", "متوفره", "متوفير", "متوفيره", "متاح", "متاحه",
        # "show me / give me / I want"
        "وريني", "ورينا", "ورني", "اعرض", "اعرضلي", "اعرضوا", "عرض", "هات", "هاتلي",
        "اديني", "اعطني", "اريد", "عايز", "عايزه", "عاوز", "عاوزه", "ابغي", "ابي",
        "محتاج", "محتاجه", "ابحث", "دور", "دورلي",
        # "do you sell"
        "بتبيعوا", "بتبيعو", "بتبيع", "تبيعون", "تبيعوا", "تبيع", "يبيعون", "بيع",
        # price / stock / time
        "سعر", "اسعار", "ثمن", "سعره", "مخزون", "حاليا", "دلوقتي", "دلوقت", "الان",
    }
)

# Signals that, together with NO specific product term, mean "list the catalogue".
_SELL_WORDS = frozenset(
    {"sell", "sells", "selling", "بتبيعوا", "بتبيعو", "بتبيع", "تبيعون", "تبيعوا", "تبيع", "يبيعون", "بيع"}
)
_QUESTION_WORDS = frozenset({"what", "whats", "which", "ايه", "اية", "اي", "ايش", "شو", "ما", "ماذا", "ماهي", "ماهو"})
_HAVE_OR_AVAILABLE_WORDS = frozenset(
    {
        "have", "got", "available", "stock",
        "عندكم", "عندكو", "عندكوا", "عندك", "لديكم", "لديك",
        "موجود", "موجوده", "موجودين", "متوفر", "متوفره", "متوفير", "متوفيره", "متاح", "متاحه",
    }
)

# "recommend / suggest me something" (folded forms, see _fold).
_RECOMMEND_WORDS = frozenset(
    {
        "recommend", "recommended", "recommendation", "recommendations", "suggest", "suggested",
        "suggestion", "suggestions",
        "رشحلي", "رشح", "رشحوا", "ترشحلي", "ترشح", "ترشيح", "ترشيحات", "مرشح", "مرشحه",
        "اقترح", "اقترحلي", "تقترح", "اقتراح", "اقتراحات", "توصيه", "توصيات",
        "انصحني", "نصحني", "تنصحني", "تنصح",
    }
)
# Words that may accompany a recommendation request without naming a product.
_RECOMMEND_FILLER = frozenset(
    {
        "something", "anything", "good", "nice", "best", "great", "new", "else", "other",
        "حاجه", "حاجات", "شي", "شيء", "اي", "احسن", "افضل", "حلو", "حلوه", "جديد", "تاني", "غيرها",
        "ليا", "لينا", "لك", "بيها", "بيه",
    }
)

_STOPWORDS: FrozenSet[str] = _STOPWORDS_EN | _STOPWORDS_AR | _CATALOG_NOUNS_EN | _CATALOG_NOUNS_AR

_SEARCH_COLUMNS = (
    Product.name,
    Product.description,
    Product.category,
    Product.brand,
)


def _tokenize(message: str) -> List[str]:
    """Lower-case, strip Arabic diacritics/tatweel, split into word tokens."""
    return _TOKEN_RE.findall(_ARABIC_MARKS_RE.sub("", message).lower())


def _fold(token: str) -> str:
    """Fold Arabic letter variants (hamza, alef maqsura, taa marbuta) for list lookups."""
    return normalize_arabic(token)


def _forms(token: str) -> List[str]:
    """Spellings of a token to look up in the word lists: as typed, and without "ال"."""
    folded = _fold(token)
    forms = [folded]
    if folded.startswith(_ARABIC_DEFINITE_ARTICLE) and len(folded) > 3:
        forms.append(folded[len(_ARABIC_DEFINITE_ARTICLE):])
    return forms


def _in(token: str, words: FrozenSet[str]) -> bool:
    return any(form in words for form in _forms(token))


def extract_search_terms(message: str) -> List[str]:
    """Turn a free-text message into a short, de-duplicated list of terms."""
    terms: List[str] = []
    seen = set()
    for token in _tokenize(message):
        if _in(token, _STOPWORDS) or _is_recommend_token(token):
            continue  # filler, or "recommend/رشحلي": a request verb, not a product name
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        token = token[:MAX_TERM_LENGTH]
        if len(token) < MIN_TERM_LENGTH or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= MAX_TERMS:
            break
    return terms


def is_catalog_query(message: str) -> bool:
    """True when the customer asks what the store has/sells in general.

    Needs BOTH: no specific product term left after filler words and generic
    nouns are removed ("Do you have iphone?" is NOT a catalogue query), and a
    catalogue signal: a generic noun ("products", "المنتجات"), a "sell" verb
    ("What do you sell?", "بتبيعوا ايه؟"), or a question word together with
    "have"/"available" ("What is available?", "عندكم ايه؟").
    """
    if extract_search_terms(message):
        return False
    tokens = _tokenize(message)
    if any(_in(t, _CATALOG_NOUNS_EN) or _in(t, _CATALOG_NOUNS_AR) or _in(t, _SELL_WORDS) for t in tokens):
        return True
    return any(_in(t, _QUESTION_WORDS) for t in tokens) and any(
        _in(t, _HAVE_OR_AVAILABLE_WORDS) for t in tokens
    )


_RECOMMEND_PREFIXES = ("رشح", "ترشح", "يرشح", "اقترح", "تقترح", "انصح", "تنصح")


def _is_recommend_token(token: str) -> bool:
    """Known recommend word, or a verb form with attached pronouns ("ترشحهولي", "اقترحلنا")."""
    return _in(token, _RECOMMEND_WORDS) or any(f.startswith(_RECOMMEND_PREFIXES) for f in _forms(token))


def is_recommendation_request(message: str) -> bool:
    """True for "recommend me something" with no specific product named.

    "رشحلي حاجة" / "what do you recommend?" -> True (use personalized recommendations).
    "recommend a laptop" / "رشحلي لابتوب"   -> False (a specific search is better).
    """
    tokens = _tokenize(message)
    if not any(_is_recommend_token(t) for t in tokens):
        return False
    remaining = [
        t
        for t in extract_search_terms(message)
        if not _is_recommend_token(t) and not _in(t, _RECOMMEND_FILLER)
    ]
    return not remaining


def _fold_sql(column):
    """SQL expression applying the same letter folding as `_fold` to a column."""
    return sql_fold(column)


def literal_terms(query: str) -> List[str]:
    """Search terms of a free-text catalogue query WITHOUT removing any word.

    Used by GET /products/?q=: unlike `extract_search_terms` (chatbot messages, which
    drops filler words such as "do you have"), every word is kept. Lower-cased,
    Arabic marks stripped, de-duplicated, bounded by MAX_TERMS / MAX_TERM_LENGTH.
    """
    terms: List[str] = []
    for token in _tokenize(query or ""):
        token = token[:MAX_TERM_LENGTH]
        if token and token not in terms:
            terms.append(token)
        if len(terms) >= MAX_TERMS:
            break
    return terms


def term_condition(term: str):
    """SQL condition: `term` occurs (case-insensitive, Arabic-folded) in name/description/category/brand."""
    pattern = f"%{_escape_like(_fold(term))}%"
    return or_(*(_fold_sql(column).ilike(pattern, escape=_LIKE_ESCAPE) for column in _SEARCH_COLUMNS))


def _escape_like(term: str) -> str:
    """Escape LIKE wildcards so the term is matched literally."""
    return (
        term.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
        .replace("%", f"{_LIKE_ESCAPE}%")
        .replace("_", f"{_LIKE_ESCAPE}_")
    )


def list_catalog(db: Session, limit: int = MAX_RESULTS) -> List[Product]:
    """The first `limit` (max MAX_RESULTS) real products, ordered by id, stock 0 included."""
    limit = max(1, min(limit, MAX_RESULTS))
    return db.query(Product).order_by(Product.id).limit(limit).all()


def search_products(db: Session, message: str, limit: int = MAX_RESULTS) -> List[Product]:
    """Real Product rows for `message` (max MAX_RESULTS).

    A catalogue query returns the catalogue (see `list_catalog`). Otherwise the
    rows match ALL terms found in `message`; the result is empty when the
    message contains no searchable term, so a message such as "do you have"
    never returns the whole catalogue.
    """
    if is_catalog_query(message):
        return list_catalog(db, limit)

    terms = extract_search_terms(message)
    if not terms:
        return []

    limit = max(1, min(limit, MAX_RESULTS))

    conditions = [term_condition(term) for term in terms]

    return (
        db.query(Product)
        .filter(and_(*conditions))
        .order_by(Product.id)
        .limit(limit)
        .all()
    )
