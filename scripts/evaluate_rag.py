from pathlib import Path

import pandas as pd

from src.rag import RAGSystem


DATASET_PATH = Path(
    "data/evaluation/test_dataset.csv"
)

OUTPUT_PATH = Path(
    "data/evaluation/rag_results.csv"
)


def evaluate_rag():
    """
    Exécute le système RAG sur le jeu de test annoté
    et sauvegarde les réponses et contextes récupérés.
    """

    dataset = pd.read_csv(DATASET_PATH)

    rag = RAGSystem()

    results = []

    total = len(dataset)

    for index, row in dataset.iterrows():
        question = row["question"]

        print(
            f"\n[{index + 1}/{total}] "
            f"{question}"
        )

        result = rag.ask(
            question,
            include_contexts=True,
        )

        results.append(
            {
                "id": row["id"],
                "category": row["category"],
                "question": question,
                "reference_answer": (
                    row["reference_answer"]
                ),
                "answer": result["answer"],
                "retrieved_contexts": (
                    result["retrieved_contexts"]
                ),
            }
        )

        print("Réponse générée :")
        print(result["answer"])

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        f"\nRésultats enregistrés dans : "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Nombre de questions évaluées : "
        f"{len(results_df)}"
    )

    return results_df


if __name__ == "__main__":
    evaluate_rag()