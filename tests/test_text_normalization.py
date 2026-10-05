"""core.text_normalization: Arabic folding shared by chatbot search and product filtering."""
import pytest
from sqlalchemy import literal, select

from core.text_normalization import contains_arabic, normalize_arabic, sql_fold
from database.database import SessionLocal


@pytest.mark.parametrize(
    "typed, expected",
    [
        ("أحذية", "احذيه"),
        ("إسلام", "اسلام"),
        ("آلة", "اله"),
        ("سماعة", "سماعه"),
        ("مستشفى", "مستشفي"),
        ("سماعه", "سماعه"),  # already folded: unchanged
    ],
)
def test_arabic_variants_are_folded(typed, expected):
    assert normalize_arabic(typed) == expected


@pytest.mark.parametrize(
    "a, b",
    [("سماعة", "سماعه"), ("أحذية", "احذيه"), ("مستشفى", "مستشفي"), ("احذيه", "أحذية")],
)
def test_spelling_variants_normalize_to_the_same_text(a, b):
    assert normalize_arabic(a) == normalize_arabic(b)


@pytest.mark.parametrize("text", ["iPhone 15 Pro", "Laptop", "MacBook-Air_13", "a1/b2 %", ""])
def test_english_and_symbols_are_never_altered(text):
    assert normalize_arabic(text) == text


def test_mixed_arabic_english_only_changes_the_arabic_part():
    assert normalize_arabic("سماعة Sony WH-1000XM5") == "سماعه Sony WH-1000XM5"


def test_normalization_is_idempotent():
    once = normalize_arabic("أحذية رياضية ة ى")
    assert normalize_arabic(once) == once


def test_marks_are_stripped_only_when_asked():
    with_marks = "سَمَّاعَة"
    assert normalize_arabic(with_marks) == "سَمَّاعَه"  # folding alone keeps the marks
    assert normalize_arabic(with_marks, strip_marks=True) == "سماعه"
    assert normalize_arabic("كـــتاب", strip_marks=True) == "كتاب"  # tatweel


def test_contains_arabic():
    assert contains_arabic("سماعة") and contains_arabic("phone سماعة")
    assert not contains_arabic("phone") and not contains_arabic("")


def test_sql_fold_matches_the_python_normalizer():
    samples = ["أحذية", "مستشفى", "سماعة", "Laptop", "آلة حاسبة"]
    db = SessionLocal()
    try:
        for sample in samples:
            assert db.execute(select(sql_fold(literal(sample)))).scalar() == normalize_arabic(sample)
    finally:
        db.close()
