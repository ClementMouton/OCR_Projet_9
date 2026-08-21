from src.vector_store import load_vector_store


QUESTIONS = [
    "Quand a lieu la Semaine des Métiers du Tourisme 2026 ?",
    "Quand a lieu l'atelier CV à l'Agence Metz Blida ?",
]


def analyze_question(vector_store, question: str, k: int = 20):
    print("\n" + "=" * 80)
    print(f"QUESTION : {question}")
    print("=" * 80)

    results = vector_store.similarity_search_with_score(
        question,
        k=k,
    )

    for rank, (document, score) in enumerate(results, start=1):
        print(f"\n--- Rang {rank} | Score : {score:.4f} ---")
        print(f"Titre : {document.metadata.get('title')}")
        print(f"Date : {document.metadata.get('start_date')}")
        print(f"Lieu : {document.metadata.get('location')}")
        print(f"UID : {document.metadata.get('uid')}")


def main():
    vector_store = load_vector_store()

    for question in QUESTIONS:
        analyze_question(
            vector_store=vector_store,
            question=question,
            k=50,
        )


if __name__ == "__main__":
    main()