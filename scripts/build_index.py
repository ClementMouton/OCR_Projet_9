from pathlib import Path

import pandas as pd

from src.embeddings import create_documents, split_documents
from src.vector_store import build_vector_store

DATA_PATH = Path("data/processed/events_metz.csv")


def main():
    events = pd.read_csv(DATA_PATH)

    documents = create_documents(events)
    chunks = split_documents(documents)

    print(f"Nombre d'événements : {len(events)}")
    print(f"Nombre de documents : {len(documents)}")
    print(f"Nombre de chunks : {len(chunks)}")

    chunk_lengths = pd.Series(
        [len(chunk.page_content) for chunk in chunks]
    )

    print("\nLongueur des chunks :")
    print(
        chunk_lengths.describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
    )

    print("\n--- Exemple de document ---")
    print(documents[0].page_content)

    print("\n--- Métadonnées ---")
    print(documents[0].metadata)

    print("\n--- Premier chunk ---")
    print(chunks[0].page_content)

    print("\nConstruction de l'index FAISS...")

    vector_store = build_vector_store(chunks)

    print("Index FAISS construit et sauvegardé.")
    print(
        f"Nombre de vecteurs dans l'index : "
        f"{vector_store.index.ntotal}"
    )


if __name__ == "__main__":
    main()