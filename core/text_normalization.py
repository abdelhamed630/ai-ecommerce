"""Arabic text normalization shared by the chatbot search and the product catalogue search.

Only Arabic characters are touched: Latin letters, digits and punctuation pass
through unchanged (case is NOT altered here either; callers lower-case or use
ILIKE as they already do). So "iPhone 15" -> "iPhone 15" and
"سماعة iPhone" -> "سماعه iPhone".

Folded variants (the usual spelling differences between what people type and
what is stored):

    أ إ آ ٱ  ->  ا      (hamza / madda / wasla on alef)
    ى        ->  ي      (alef maqsura typed as yaa: مستشفى / مستشفي)
    ة        ->  ه      (taa marbuta typed as haa: سماعة / سماعه)

`strip_marks=True` additionally removes harakat (U+064B-U+065F), superscript alef
(U+0670) and tatweel (U+0640).

Both sides of a comparison must be normalized: the user's term here, and the
stored column in SQL (`sql_fold`, a portable REPLACE chain that PostgreSQL and
SQLite both run). The folding is deliberately one-way and lossy; it is used for
MATCHING only, never to rewrite stored data or text shown to the customer.
"""
import re

ARABIC_FOLD_MAP = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "ى": "ي",
    "ة": "ه",
}
ARABIC_FOLD_TABLE = str.maketrans(ARABIC_FOLD_MAP)
ARABIC_MARKS_RE = re.compile("[\u064b-\u065f\u0670\u0640]")
_ARABIC_LETTER_RE = re.compile("[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")


def contains_arabic(text: str) -> bool:
    return bool(text) and _ARABIC_LETTER_RE.search(text) is not None


def normalize_arabic(text: str, strip_marks: bool = False) -> str:
    """Fold common Arabic spelling variants; non-Arabic text is returned unchanged."""
    if not text:
        return text
    if strip_marks:
        text = ARABIC_MARKS_RE.sub("", text)
    return text.translate(ARABIC_FOLD_TABLE)


def sql_fold(column):
    """SQL expression applying `normalize_arabic` (letter folding) to a column.

    Built from SQLAlchemy `func.replace`, so the text is never interpolated into
    SQL. Wrapping a column in functions means a plain B-tree index on it cannot
    be used for the comparison (a `%term%` ILIKE cannot use one anyway).
    """
    from sqlalchemy import func  # local import: keep this module importable without SQLAlchemy

    expr = column
    for source, target in ARABIC_FOLD_MAP.items():
        expr = func.replace(expr, source, target)
    return expr
