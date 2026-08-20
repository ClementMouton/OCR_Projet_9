from src.vector_store import load_vector_store


def main():
    vector_store = load_vector_store()

    query = (
        "Je cherche un événement autour du recrutement "
        "et de l'emploi à Metz"
    )

    results = vector_store.similarity_search(
        query,
        k=5,
    )

    print(f"Question : {query}")
    print(f"Nombre de résultats : {len(results)}")

    for index, document in enumerate(results, start=1):
        print(f"\n--- Résultat {index} ---")
        print(f"Titre : {document.metadata.get('title')}")
        print(f"Lieu : {document.metadata.get('location')}")
        print(f"Date début : {document.metadata.get('start_date')}")
        print(f"Date fin : {document.metadata.get('end_date')}")
        print(f"URL : {document.metadata.get('url')}")

        print("\nContenu :")
        print(document.page_content[:500])


if __name__ == "__main__":
    main()