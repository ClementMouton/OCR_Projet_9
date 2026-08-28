import sys

import requests


BASE_URL = "http://localhost:8000"
TIMEOUT = 60


def check_health():
    print("1. Test de GET /health")

    response = requests.get(
        f"{BASE_URL}/health",
        timeout=TIMEOUT,
    )

    response.raise_for_status()
    data = response.json()

    assert data["status"] == "ok"
    assert data["rag_loaded"] is True

    print("   OK - API disponible et RAG chargé.")


def check_ask():
    print("\n2. Test de POST /ask")

    question = "Quels concerts sont disponibles à Metz ?"

    response = requests.post(
        f"{BASE_URL}/ask",
        json={"question": question},
        timeout=TIMEOUT,
    )

    response.raise_for_status()
    data = response.json()

    assert data["question"] == question
    assert isinstance(data["answer"], str)
    assert data["answer"].strip()
    assert isinstance(data["sources"], list)

    print("   OK - Le RAG a répondu.")
    print(f"\nQuestion : {data['question']}")
    print(f"\nRéponse : {data['answer']}")

    print("\nSources :")

    if data["sources"]:
        for source in data["sources"]:
            print(
                f"   - {source.get('title', 'Sans titre')} "
                f"({source.get('url', 'URL indisponible')})"
            )
    else:
        print("   Aucune source retournée.")


def main():
    print("=== Test fonctionnel de l'API Puls-Events ===\n")

    try:
        check_health()
        check_ask()

    except requests.exceptions.ConnectionError:
        print(
            "\nERREUR : impossible de contacter l'API."
            "\nVérifie qu'elle est lancée sur http://localhost:8000."
        )
        sys.exit(1)

    except requests.exceptions.RequestException as error:
        print(f"\nERREUR HTTP : {error}")
        sys.exit(1)

    except (AssertionError, KeyError, TypeError) as error:
        print(f"\nERREUR : réponse API invalide : {error}")
        sys.exit(1)

    print("\n=== Test fonctionnel réussi ===")


if __name__ == "__main__":
    main()