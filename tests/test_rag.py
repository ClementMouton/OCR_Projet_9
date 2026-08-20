from datetime import date
from unittest.mock import MagicMock

import pytest

from src.rag import RAGSystem


def main():
    rag = RAGSystem()

    question = (
        "Quels événements festifs ont lieu à Metz ce vendredi soir ?"
    )

    result = rag.ask(question)

    print("\nQUESTION")
    print(result["question"])

    print("\nRÉPONSE")
    print(result["answer"])

    print("\nSOURCES")

    for source in result["sources"]:
        print(
            f"- {source['title']} : "
            f"{source['url']}"
        )


def create_rag_without_init():
    """
    Crée une instance de RAGSystem sans charger FAISS
    ni initialiser Mistral.
    """
    return RAGSystem.__new__(RAGSystem)


def test_empty_question():
    rag = create_rag_without_init()

    with pytest.raises(
        ValueError,
        match="La question ne peut pas être vide",
    ):
        rag.ask("")


def test_event_matches_date():
    rag = create_rag_without_init()

    document = MagicMock()
    document.metadata = {
        "start_date": "2026-08-20T08:00:00+00:00",
        "end_date": "2026-08-22T18:00:00+00:00",
    }

    assert rag._event_matches_date(
        document,
        date(2026, 8, 21),
    )


def test_event_does_not_match_date():
    rag = create_rag_without_init()

    document = MagicMock()
    document.metadata = {
        "start_date": "2026-05-22T08:00:00+00:00",
        "end_date": "2026-05-22T18:00:00+00:00",
    }

    assert not rag._event_matches_date(
        document,
        date(2026, 8, 21),
    )


if __name__ == "__main__":
    main()