from datetime import date, timedelta
import re


WEEKDAYS = {
    "lundi": 0,
    "mardi": 1,
    "mercredi": 2,
    "jeudi": 3,
    "vendredi": 4,
    "samedi": 5,
    "dimanche": 6,
}


def get_next_weekday(
    target_weekday: int,
    reference_date: date | None = None,
) -> date:
    """
    Retourne la prochaine occurrence d'un jour de la semaine.

    Exemple :
    jeudi 20 août + vendredi -> vendredi 21 août.
    """

    reference_date = reference_date or date.today()

    days_ahead = (
        target_weekday - reference_date.weekday()
    ) % 7

    return reference_date + timedelta(days=days_ahead)


def extract_date_constraint(
    question: str,
    reference_date: date | None = None,
) -> date | None:
    """
    Extrait une contrainte temporelle simple d'une question.

    Gère :
    - aujourd'hui
    - demain
    - ce lundi, ce mardi, ..., ce dimanche
    """

    reference_date = reference_date or date.today()
    question = question.lower()

    if "aujourd'hui" in question or "aujourdhui" in question:
        return reference_date

    if "demain" in question:
        return reference_date + timedelta(days=1)

    for weekday_name, weekday_number in WEEKDAYS.items():
        pattern = rf"\bce\s+{weekday_name}\b"

        if re.search(pattern, question):
            return get_next_weekday(
                weekday_number,
                reference_date,
            )

    return None