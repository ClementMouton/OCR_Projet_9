from datetime import date

from src.date_utils import (
    extract_date_constraint,
    get_next_weekday,
)


REFERENCE_DATE = date(2026, 8, 20)


def test_today():
    result = extract_date_constraint(
        "Quels événements ont lieu aujourd'hui ?",
        reference_date=REFERENCE_DATE,
    )

    assert result == date(2026, 8, 20)


def test_tomorrow():
    result = extract_date_constraint(
        "Que puis-je faire demain ?",
        reference_date=REFERENCE_DATE,
    )

    assert result == date(2026, 8, 21)


def test_this_friday():
    result = extract_date_constraint(
        "Quels événements ont lieu ce vendredi soir ?",
        reference_date=REFERENCE_DATE,
    )

    assert result == date(2026, 8, 21)


def test_next_weekday_across_week():
    result = get_next_weekday(
        target_weekday=0,
        reference_date=REFERENCE_DATE,
    )

    assert result == date(2026, 8, 24)


def test_no_date_constraint():
    result = extract_date_constraint(
        "Je cherche un concert de jazz à Metz.",
        reference_date=REFERENCE_DATE,
    )

    assert result is None