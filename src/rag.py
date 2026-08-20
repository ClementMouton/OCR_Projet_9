from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from src.date_utils import extract_date_constraint
from src.vector_store import load_vector_store


LLM_MODEL = "mistral-small-latest"
TOP_K = 5


SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans les événements disponibles à Metz.

Tu dois répondre à la question de l'utilisateur uniquement à partir
des événements présents dans le contexte fourni.

Règles :
- N'invente aucune information absente du contexte.
- Si le contexte ne permet pas de répondre, indique clairement que
  tu ne disposes pas de suffisamment d'informations.
- Privilégie les événements qui répondent directement à la demande.
- Lorsque tu recommandes un événement, indique son titre, sa date
  et son lieu lorsque ces informations sont disponibles.
- Réponds en français.
"""


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
Contexte :
{context}

Question :
{question}
""",
        ),
    ]
)


class RAGSystem:
    """
    Système RAG combinant une recherche FAISS
    et un modèle de génération Mistral.
    """

    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k

        self.vector_store = load_vector_store()

        self.llm = ChatMistralAI(
            model=LLM_MODEL,
            temperature=0,
        )

    def _event_matches_date(
        self,
        document,
        target_date,
    ) -> bool:
        """
        Vérifie si un événement est actif à la date demandée.
        """

        start_date = document.metadata.get("start_date")
        end_date = document.metadata.get("end_date")

        if not start_date or not end_date:
            return False

        try:
            start = datetime.fromisoformat(start_date).date()
            end = datetime.fromisoformat(end_date).date()
        except (ValueError, TypeError):
            return False

        return start <= target_date <= end

    def retrieve(self, question: str):
        """
        Recherche les événements pertinents.

        Sans contrainte temporelle :
        recherche sémantique classique dans FAISS.

        Avec contrainte temporelle :
        la recherche FAISS est filtrée afin de ne conserver
        que les événements actifs à la date demandée.
        """

        target_date = extract_date_constraint(question)

        if target_date is None:
            return self.vector_store.similarity_search(
                question,
                k=self.top_k,
            )

        def date_filter(metadata: dict) -> bool:
            start_date = metadata.get("start_date")
            end_date = metadata.get("end_date")

            if not start_date or not end_date:
                return False

            try:
                start = datetime.fromisoformat(
                    start_date
                ).date()

                end = datetime.fromisoformat(
                    end_date
                ).date()

            except (ValueError, TypeError):
                return False

            return start <= target_date <= end

        return self.vector_store.similarity_search(
            question,
            k=self.top_k,
            filter=date_filter,
            fetch_k=790,
        )

    def _build_context(self, documents) -> str:
        """
        Transforme les documents récupérés en contexte
        exploitable par le LLM.
        """

        contexts = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            metadata = document.metadata

            context = (
                f"Événement {index}\n"
                f"Titre : {metadata.get('title', '')}\n"
                f"Date de début : "
                f"{metadata.get('start_date', '')}\n"
                f"Date de fin : "
                f"{metadata.get('end_date', '')}\n"
                f"Lieu : "
                f"{metadata.get('location', '')}\n"
                f"URL : "
                f"{metadata.get('url', '')}\n"
                f"Informations :\n"
                f"{document.page_content}"
            )

            contexts.append(context)

        return "\n\n---\n\n".join(contexts)

    def ask(
        self,
        question: str,
        include_contexts: bool = False,
    ) -> dict:
        """
        Répond à une question à partir du pipeline RAG.

        include_contexts permet de retourner les chunks
        récupérés pour l'évaluation du système.
        """

        if not question or not question.strip():
            raise ValueError(
                "La question ne peut pas être vide."
            )

        documents = self.retrieve(question)

        if not documents:
            result = {
                "question": question,
                "answer": (
                    "Je n'ai trouvé aucun événement correspondant "
                    "à cette demande dans les données disponibles."
                ),
                "sources": [],
            }

            if include_contexts:
                result["retrieved_contexts"] = []

            return result

        context = self._build_context(documents)

        messages = PROMPT.format_messages(
            context=context,
            question=question,
        )

        response = self.llm.invoke(messages)

        sources = []
        seen_uids = set()

        for document in documents:
            uid = document.metadata.get("uid")

            if uid not in seen_uids:
                sources.append(
                    {
                        "uid": uid,
                        "title": document.metadata.get(
                            "title"
                        ),
                        "url": document.metadata.get(
                            "url"
                        ),
                    }
                )

                seen_uids.add(uid)

        result = {
            "question": question,
            "answer": response.content,
            "sources": sources,
        }

        if include_contexts:
            result["retrieved_contexts"] = [
                document.page_content
                for document in documents
            ]

        return result