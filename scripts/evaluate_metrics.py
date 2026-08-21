import ast
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI


INPUT_PATH = Path("data/evaluation/rag_results.csv")
OUTPUT_PATH = Path("data/evaluation/metrics_results.csv")
SUMMARY_PATH = Path("data/evaluation/metrics_summary.json")

JUDGE_MODEL = "mistral-small-latest"
MAX_RETRIES = 3
RETRY_DELAY = 5


def load_results() -> pd.DataFrame:
    """Charge les résultats générés par le système RAG."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_PATH}. "
            "Exécutez d'abord scripts.evaluate_rag."
        )

    return pd.read_csv(INPUT_PATH)


def parse_contexts(value) -> list[str]:
    """Convertit la colonne retrieved_contexts en liste de textes."""
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [str(context) for context in parsed]

    except (ValueError, SyntaxError):
        pass

    return [str(value)]


def exact_match(answer: str, reference: str) -> float:
    """
    Compare strictement la réponse générée et la référence.

    Cette métrique est volontairement sévère et sert de baseline.
    """
    answer_normalized = " ".join(
        str(answer).lower().strip().split()
    )
    reference_normalized = " ".join(
        str(reference).lower().strip().split()
    )

    return float(answer_normalized == reference_normalized)


def build_judge_prompt(
    question: str,
    reference: str,
    answer: str,
    contexts: list[str],
) -> str:
    """Construit le prompt utilisé par Mistral pour évaluer une réponse."""

    context_text = "\n\n--- DOCUMENT ---\n".join(contexts)

    return f"""
Tu es un évaluateur impartial d'un système RAG.

Évalue la réponse générée et les documents récupérés selon exactement
quatre critères.

1. faithfulness
La réponse générée est-elle supportée par les documents récupérés ?
1 = toutes les affirmations importantes sont justifiées par les documents.
0 = les affirmations importantes ne sont pas justifiées.

2. answer_correctness
La réponse générée transmet-elle les mêmes informations essentielles
que la réponse humaine de référence ?
1 = réponse correcte.
0 = réponse incorrecte.

3. answer_relevancy
La réponse générée répond-elle directement et utilement à la question ?
1 = totalement pertinente.
0 = non pertinente.

4. context_recall
Les documents récupérés contiennent-ils les informations nécessaires
pour produire la réponse humaine de référence ?
1 = toutes les informations essentielles de la référence sont présentes
dans les documents récupérés.
0 = aucune des informations nécessaires n'est présente.

Tu peux attribuer n'importe quelle valeur décimale entre 0 et 1.

IMPORTANT :
- Évalue uniquement à partir des éléments fournis.
- N'utilise aucune connaissance externe.
- Une réponse peut être fidèle aux documents mais incorrecte par rapport
  à la référence.
- Une réponse indiquant qu'une information est absente peut être fidèle
  aux documents récupérés tout en étant incorrecte par rapport à la
  référence humaine.
- Pour context_recall, évalue uniquement les documents récupérés par
  rapport à la référence humaine, indépendamment de la réponse générée.
- Pour une question dont la référence indique explicitement qu'aucune
  information n'est disponible, context_recall vaut 1 si les documents
  récupérés ne contiennent pas d'information permettant de contredire
  cette absence.
- Ne pénalise pas une réponse uniquement parce qu'elle est formulée
  différemment de la référence.

QUESTION :
{question}

RÉPONSE HUMAINE DE RÉFÉRENCE :
{reference}

RÉPONSE GÉNÉRÉE :
{answer}

DOCUMENTS RÉCUPÉRÉS :
{context_text}

Retourne UNIQUEMENT un objet JSON valide sous cette forme :

{{
    "faithfulness": 0.0,
    "answer_correctness": 0.0,
    "answer_relevancy": 0.0,
    "context_recall": 0.0
}}
""".strip()


def parse_judge_response(content: str) -> dict:
    """Extrait et valide les métriques retournées par le LLM."""
    content = content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            f"Réponse JSON invalide du juge : {content}"
        )

    scores = json.loads(content[start:end + 1])

    required_metrics = {
        "faithfulness",
        "answer_correctness",
        "answer_relevancy",
        "context_recall"
    }

    if not required_metrics.issubset(scores):
        raise ValueError(
            f"Métriques manquantes : {scores}"
        )

    for metric in required_metrics:
        score = float(scores[metric])

        if not 0 <= score <= 1:
            raise ValueError(
                f"Score invalide pour {metric} : {score}"
            )

        scores[metric] = score

    return scores


def evaluate_with_judge(
    llm,
    question: str,
    reference: str,
    answer: str,
    contexts: list[str],
) -> dict:
    """Évalue une réponse avec gestion des erreurs temporaires."""

    prompt = build_judge_prompt(
        question=question,
        reference=reference,
        answer=answer,
        contexts=contexts,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = llm.invoke(prompt)

            return parse_judge_response(response.content)

        except Exception as error:
            if attempt == MAX_RETRIES:
                raise

            print(
                f"Erreur lors de l'évaluation : {error}"
            )
            print(
                f"Nouvelle tentative dans "
                f"{RETRY_DELAY} secondes..."
            )

            time.sleep(RETRY_DELAY)


def evaluate_metrics():
    """Calcule les métriques automatiques du système RAG."""
    load_dotenv()

    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError(
            "La variable MISTRAL_API_KEY est absente."
        )

    df = load_results()

    llm = ChatMistralAI(
        model=JUDGE_MODEL,
        temperature=0,
    )

    evaluation_rows = []

    total = len(df)

    for index, row in df.iterrows():
        question = str(row["question"])
        reference = str(row["reference_answer"])
        answer = str(row["answer"])
        contexts = parse_contexts(row["retrieved_contexts"])

        print(
            f"\n[{index + 1}/{total}] "
            f"Évaluation : {question}"
        )

        scores = evaluate_with_judge(
            llm=llm,
            question=question,
            reference=reference,
            answer=answer,
            contexts=contexts,
        )

        em_score = exact_match(
            answer=answer,
            reference=reference,
        )

        result = {
            "id": row["id"],
            "category": row["category"],
            "question": question,
            "faithfulness": scores["faithfulness"],
            "answer_correctness": scores["answer_correctness"],
            "answer_relevancy": scores["answer_relevancy"],
            "context_recall": scores["context_recall"],
            "exact_match": em_score,
        }

        evaluation_rows.append(result)

        print(
            f"Faithfulness       : "
            f"{result['faithfulness']:.2f}"
        )
        print(
            f"Answer correctness : "
            f"{result['answer_correctness']:.2f}"
        )
        print(
            f"Answer relevancy   : "
            f"{result['answer_relevancy']:.2f}"
        )
        print(
            f"Context Recall      : "
            f"{result['context_recall']:.2f}"
        )
        print(
            f"Exact Match        : "
            f"{result['exact_match']:.2f}"
        )

    results_df = pd.DataFrame(evaluation_rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    metric_columns = [
        "faithfulness",
        "answer_correctness",
        "answer_relevancy",
        "context_recall",
        "exact_match",
    ]

    summary = {
        "number_of_questions": len(results_df),
        "metrics": {
            metric: round(
                float(results_df[metric].mean()),
                4,
            )
            for metric in metric_columns
        },
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\n==============================")
    print("RÉSULTATS GLOBAUX")
    print("==============================")

    for metric, score in summary["metrics"].items():
        print(
            f"{metric:<20} : {score:.4f}"
        )

    print(
        f"\nRésultats détaillés : {OUTPUT_PATH}"
    )
    print(
        f"Résumé : {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    evaluate_metrics()