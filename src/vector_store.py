from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings


DEFAULT_INDEX_PATH = Path("faiss_index")
EMBEDDING_MODEL = "mistral-embed"


def get_embeddings() -> MistralAIEmbeddings:
    """
    Initialise le modèle d'embeddings Mistral.
    """
    load_dotenv()

    return MistralAIEmbeddings(
        model=EMBEDDING_MODEL,
    )


def build_vector_store(
    documents: list[Document],
    index_path: Path = DEFAULT_INDEX_PATH,
) -> FAISS:
    """
    Vectorise les documents avec Mistral et construit un index FAISS.

    L'index est ensuite sauvegardé localement afin de pouvoir être
    rechargé sans recalculer les embeddings des documents.
    """
    if not documents:
        raise ValueError(
            "Impossible de construire un index à partir d'une liste vide."
        )

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    index_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(
        str(index_path)
    )

    return vector_store


def load_vector_store(
    index_path: Path = DEFAULT_INDEX_PATH,
) -> FAISS:
    """
    Recharge un index FAISS existant depuis le disque.
    """
    embeddings = get_embeddings()

    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )