import os
import re
import unicodedata
from datetime import date

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

from src.date_utils import extract_date_constraint
from src.vector_store import load_vector_store


load_dotenv()


class RAGSystem:
    def __init__(self):
        self.vector_store = load_vector_store()

        self.llm = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0,
            api_key=os.getenv("MISTRAL_API_KEY"),
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalise un texte pour faciliter
        les comparaisons lexicales.
        """
        text = str(text or "").lower()

        text = unicodedata.normalize("NFD", text)
        text = "".join(
            char
            for char in text
            if unicodedata.category(char) != "Mn"
        )

        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    @classmethod
    def _extract_query_terms(
        cls,
        question: str,
    ) -> set[str]:
        """
        Extrait les termes significatifs
        de la question.
        """
        stopwords = {
            "a",
            "au",
            "aux",
            "avec",
            "ce",
            "ces",
            "dans",
            "de",
            "des",
            "du",
            "en",
            "est",
            "et",
            "il",
            "je",
            "la",
            "le",
            "les",
            "lieu",
            "me",
            "peut",
            "pour",
            "proposer",
            "quand",
            "que",
            "quel",
            "quelle",
            "quels",
            "quelles",
            "qui",
            "se",
            "sur",
            "un",
            "une",
        }

        normalized = cls._normalize_text(question)

        return {
            term
            for term in normalized.split()
            if len(term) >= 3
            and term not in stopwords
        }

    def _rerank_documents(
        self,
        question: str,
        documents: list,
        limit: int = 10,
    ) -> list:
        """
        Rerank les résultats FAISS en favorisant
        les correspondances lexicales avec le titre,
        le lieu et le contenu.

        Le rang FAISS initial est utilisé pour
        départager les documents ayant le même score.
        """
        query_terms = self._extract_query_terms(
            question
        )

        if not query_terms:
            return documents[:limit]

        ranked_documents = []

        for original_rank, document in enumerate(
            documents
        ):
            metadata = document.metadata or {}

            title = self._normalize_text(
                metadata.get("title", "")
            )

            location = self._normalize_text(
                metadata.get("location", "")
            )

            content = self._normalize_text(
                document.page_content
            )

            title_terms = set(title.split())
            location_terms = set(location.split())
            content_terms = set(content.split())

            title_matches = len(
                query_terms & title_terms
            )

            location_matches = len(
                query_terms & location_terms
            )

            content_matches = len(
                query_terms & content_terms
            )

            lexical_score = (
                title_matches * 3
                + location_matches * 2
                + content_matches
            )

            ranked_documents.append(
                (
                    lexical_score,
                    original_rank,
                    document,
                )
            )

        ranked_documents.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        return [
            document
            for _, _, document
            in ranked_documents[:limit]
        ]

    @staticmethod
    def _event_matches_date(
        document,
        target_date: date,
    ) -> bool:
        """
        Vérifie si un événement correspond
        à une date donnée.
        """
        metadata = document.metadata or {}

        start_date = metadata.get(
            "start_date"
        )
        end_date = metadata.get(
            "end_date"
        )

        if not start_date or not end_date:
            return False

        try:
            start = date.fromisoformat(
                str(start_date)[:10]
            )

            end = date.fromisoformat(
                str(end_date)[:10]
            )

        except ValueError:
            return False

        return start <= target_date <= end

    def retrieve(
        self,
        question: str,
    ) -> list:
        """
        Recherche les documents pertinents.

        Étapes :
        1. Recherche vectorielle FAISS large.
        2. Reranking lexical.
        3. Filtrage temporel si nécessaire.
        """
        candidates = (
            self.vector_store.similarity_search(
                question,
                k=30,
            )
        )

        documents = self._rerank_documents(
            question=question,
            documents=candidates,
            limit=10,
        )

        target_date = extract_date_constraint(
            question
        )

        if target_date is not None:
            documents = [
                document
                for document in documents
                if self._event_matches_date(
                    document,
                    target_date,
                )
            ]

        return documents

    @staticmethod
    def _build_context(
        documents: list,
    ) -> str:
        """
        Construit le contexte envoyé au LLM.
        """
        contexts = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            metadata = document.metadata or {}

            context = (
                f"Événement {index}\n"
                f"Titre : "
                f"{metadata.get('title', '')}\n"
                f"Date de début : "
                f"{metadata.get('start_date', '')}\n"
                f"Date de fin : "
                f"{metadata.get('end_date', '')}\n"
                f"Lieu : "
                f"{metadata.get('location', '')}\n"
                f"URL : "
                f"{metadata.get('url', '')}\n"
                f"Contenu :\n"
                f"{document.page_content}"
            )

            contexts.append(context)

        return "\n\n".join(contexts)

    @staticmethod
    def _build_sources(
        documents: list,
    ) -> list:
        """
        Construit la liste des sources
        associées à la réponse.
        """
        sources = []
        seen = set()

        for document in documents:
            metadata = document.metadata or {}

            uid = metadata.get("uid")
            url = metadata.get("url")

            source_id = uid or url

            if source_id in seen:
                continue

            seen.add(source_id)

            sources.append(
                {
                    "uid": uid,
                    "title": metadata.get(
                        "title"
                    ),
                    "url": url,
                    "start_date": metadata.get(
                        "start_date"
                    ),
                    "end_date": metadata.get(
                        "end_date"
                    ),
                    "location": metadata.get(
                        "location"
                    ),
                }
            )

        return sources

    def ask(
        self,
        question: str,
        include_contexts: bool = False,
    ) -> dict:
        """
        Répond à une question à partir
        du pipeline RAG.

        include_contexts permet de retourner
        les chunks récupérés pour l'évaluation.
        """
        if not question or not question.strip():
            raise ValueError(
                "La question ne peut pas être vide"
            )

        question = question.strip()

        documents = self.retrieve(
            question
        )

        sources = self._build_sources(
            documents
        )

        if not documents:
            result = {
                "question": question,
                "answer": (
                    "Je n'ai trouvé aucun événement "
                    "correspondant à cette demande "
                    "dans les données disponibles."
                ),
                "sources": [],
            }

            if include_contexts:
                result[
                    "retrieved_contexts"
                ] = []

            return result

        context = self._build_context(
            documents
        )

        system_prompt = """
Tu es un assistant spécialisé dans la recommandation
d'événements à Metz.

Tu dois répondre uniquement à partir du contexte fourni.

Règles :
- N'invente aucune information.
- Si l'information demandée n'est pas présente dans le
  contexte, indique clairement que tu ne disposes pas de
  suffisamment d'informations.
- Ne présente pas un événement comme pertinent s'il ne
  correspond pas réellement à la demande.
- Utilise uniquement les dates, lieux et horaires présents
  dans le contexte.
- Si plusieurs événements correspondent, présente les plus
  pertinents clairement.
- Si la question concerne une ville autre que Metz et que
  le contexte ne contient pas d'événement dans cette ville,
  indique que tu ne disposes pas de cette information.
- Réponds en français.
"""

        user_prompt = f"""
Contexte :

{context}

Question :

{question}
"""

        messages = [
            (
                "system",
                system_prompt,
            ),
            (
                "human",
                user_prompt,
            ),
        ]

        response = self.llm.invoke(
            messages
        )

        result = {
            "question": question,
            "answer": response.content,
            "sources": sources,
        }

        if include_contexts:
            result[
                "retrieved_contexts"
            ] = [
                document.page_content
                for document in documents
            ]

        return result