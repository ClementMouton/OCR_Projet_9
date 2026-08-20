from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import api.main as api_main


client = TestClient(api_main.app)


def test_health():
    api_main.rag_system = MagicMock()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "rag_loaded": True,
    }


def test_ask():
    mock_rag = MagicMock()

    mock_rag.ask.return_value = {
        "question": "Quels événements sont disponibles ?",
        "answer": "Voici les événements disponibles.",
        "sources": [
            {
                "uid": "123",
                "title": "Événement test",
                "url": "https://example.com/event",
            }
        ],
    }

    api_main.rag_system = mock_rag

    response = client.post(
        "/ask",
        json={
            "question": "Quels événements sont disponibles ?",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == (
        "Quels événements sont disponibles ?"
    )
    assert data["answer"] == (
        "Voici les événements disponibles."
    )
    assert len(data["sources"]) == 1

    mock_rag.ask.assert_called_once_with(
        "Quels événements sont disponibles ?"
    )


def test_ask_empty_question():
    mock_rag = MagicMock()
    mock_rag.ask.side_effect = ValueError(
        "La question ne peut pas être vide."
    )

    api_main.rag_system = mock_rag

    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "La question ne peut pas être vide."
    )