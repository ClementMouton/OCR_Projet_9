from unittest.mock import MagicMock

import pytest
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


def test_ask_valid_question():
    mock_rag = MagicMock()

    mock_rag.ask.return_value = {
        "question": "Quels événements sont disponibles ?",
        "answer": "Voici les événements disponibles.",
        "sources": [
            {
                "uid": "123",
                "title": "Événement test",
                "url": "https://example.com/event",
                "start_date": "2026-08-28",
                "end_date": "2026-08-28",
                "location": "Metz",
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

    assert data["question"] == "Quels événements sont disponibles ?"
    assert data["answer"] == "Voici les événements disponibles."
    assert len(data["sources"]) == 1

    mock_rag.ask.assert_called_once_with(
        "Quels événements sont disponibles ?"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": 123},
        {"question": None},
    ],
)
def test_ask_rejects_invalid_question(payload):
    mock_rag = MagicMock()
    api_main.rag_system = mock_rag

    response = client.post("/ask", json=payload)

    assert response.status_code == 422

    mock_rag.ask.assert_not_called()


def test_ask_rejects_extra_field():
    mock_rag = MagicMock()
    api_main.rag_system = mock_rag

    response = client.post(
        "/ask",
        json={
            "question": "Concert à Metz ?",
            "champ_inconnu": "test",
        },
    )

    assert response.status_code == 422

    mock_rag.ask.assert_not_called()


def test_ask_rejects_non_alphabetic_question():
    mock_rag = MagicMock()
    api_main.rag_system = mock_rag

    response = client.post(
        "/ask",
        json={"question": "123456"},
    )

    assert response.status_code == 422

    mock_rag.ask.assert_not_called()


def test_ask_rejects_too_long_question():
    mock_rag = MagicMock()
    api_main.rag_system = mock_rag

    response = client.post(
        "/ask",
        json={"question": "a" * 501},
    )

    assert response.status_code == 422

    mock_rag.ask.assert_not_called()

def test_rebuild():
    mock_events = MagicMock()
    mock_processed_events = [MagicMock(), MagicMock()]
    mock_documents = [MagicMock(), MagicMock()]
    mock_chunks = [MagicMock(), MagicMock(), MagicMock()]

    mock_vector_store = MagicMock()
    mock_vector_store.index.ntotal = 3

    mock_rag_system = MagicMock()

    with (
        pytest.MonkeyPatch.context() as monkeypatch
    ):
        monkeypatch.setattr(
            api_main,
            "fetch_events",
            lambda city: mock_events,
        )
        monkeypatch.setattr(
            api_main,
            "preprocess_events",
            lambda events: mock_processed_events,
        )
        monkeypatch.setattr(
            api_main,
            "create_documents",
            lambda events: mock_documents,
        )
        monkeypatch.setattr(
            api_main,
            "split_documents",
            lambda documents: mock_chunks,
        )
        monkeypatch.setattr(
            api_main,
            "build_vector_store",
            lambda chunks: mock_vector_store,
        )
        monkeypatch.setattr(
            api_main,
            "RAGSystem",
            lambda: mock_rag_system,
        )

        response = client.post("/rebuild")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Index FAISS reconstruit avec succès.",
        "events_count": 2,
        "chunks_count": 3,
        "vectors_count": 3,
    }

    assert api_main.rag_system is mock_rag_system