import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _clean_value(value) -> str:
    """Convertit une valeur en texte exploitable."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def create_documents(events: pd.DataFrame) -> list[Document]:
    """
    Transforme chaque événement OpenAgenda en Document LangChain.

    Le contenu textuel contient les informations utiles à la recherche
    sémantique. Les informations d'identification sont conservées dans
    les métadonnées.
    """

    documents = []

    for _, event in events.iterrows():
        sections = [
            f"Titre : {_clean_value(event.get('title_fr'))}",
            f"Description : {_clean_value(event.get('description_fr'))}",
        ]

        optional_fields = [
            ("Description détaillée", "longdescription_fr"),
            ("Mots-clés", "keywords_fr"),
            ("Date", "daterange_fr"),
            ("Lieu", "location_name"),
            ("Adresse", "location_address"),
            ("Conditions", "conditions_fr"),
            ("Accessibilité", "accessibility_label_fr"),
        ]

        for label, column in optional_fields:
            value = _clean_value(event.get(column))

            if value:
                sections.append(f"{label} : {value}")

        page_content = "\n".join(sections)

        metadata = {
            "uid": _clean_value(event.get("uid")),
            "title": _clean_value(event.get("title_fr")),
            "start_date": _clean_value(event.get("firstdate_begin")),
            "end_date": _clean_value(event.get("lastdate_end")),
            "location": _clean_value(event.get("location_name")),
            "city": _clean_value(event.get("location_city")),
            "url": _clean_value(event.get("canonicalurl")),
        }

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata,
            )
        )

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Découpe les documents en chunks avant vectorisation.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return text_splitter.split_documents(documents)