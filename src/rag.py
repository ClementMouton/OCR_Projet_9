from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from src.date_utils import extract_date_constraint
from src.vector_store import load_vector_store


LLM_MODEL = "mistral-small-latest"
TOP_K = 5
TEMPORAL_SEARCH_K = 30


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
        except ValueError:
            return False

        return start <= target_date <= end

    def retrieve(self, question: str):
        """
        Recherche les événements pertinents.

        Si une date relative est détectée dans la question,
        la recherche sémantique est suivie d'un filtre temporel.
        """

        target_date = extract_date_constraint(question)

        if target_date is None:
            return self.vector_store.similarity_search(
                question,
                k=self.top_k,
            )

        candidates = self.vector_store.similarity_search(
            question,
            k=TEMPORAL_SEARCH_K,
        )

        filtered_documents = [
            document
            for document in candidates
            if self._event_matches_date(
                document,
                target_date,
            )
        ]

        return filtered_documents[:self.top_k]

    def _build_context(self, documents) -> str:
        contexts = []

        for index, document in enumerate(documents, start=1):
            metadata = document.metadata

            context = (
                f"Événement {index}\n"
                f"Titre : {metadata.get('title', '')}\n"
                f"Date de début : {metadata.get('start_date', '')}\n"
                f"Date de fin : {metadata.get('end_date', '')}\n"
                f"Lieu : {metadata.get('location', '')}\n"
                f"URL : {metadata.get('url', '')}\n"
                f"Informations :\n{document.page_content}"
            )

            contexts.append(context)

        return "\n\n---\n\n".join(contexts)

    def ask(self, question: str) -> dict:
        if not question or not question.strip():
            raise ValueError(
                "La question ne peut pas être vide."
            )

        documents = self.retrieve(question)

        if not documents:
            return {
                "question": question,
                "answer": (
                    "Je n'ai trouvé aucun événement correspondant "
                    "à cette demande dans les données disponibles."
                ),
                "sources": [],
            }

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
                        "title": document.metadata.get("title"),
                        "url": document.metadata.get("url"),
                    }
                )
                seen_uids.add(uid)

        return {
            "question": question,
            "answer": response.content,
            "sources": sources,
        }