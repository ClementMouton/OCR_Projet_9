import pandas as pd
from bs4 import BeautifulSoup

COLUMNS_TO_KEEP = [
    "uid",
    "canonicalurl",
    "title_fr",
    "description_fr",
    "longdescription_fr",
    "conditions_fr",
    "keywords_fr",
    "category",
    "daterange_fr",
    "firstdate_begin",
    "lastdate_end",
    "timings",
    "location_name",
    "location_address",
    "location_postalcode",
    "location_city",
    "location_department",
    "location_region",
    "accessibility_label_fr",
    "age_min",
    "age_max",
    "attendancemode",
    "onlineaccesslink",
    "registration",
]

def clean_html(text: str) -> str:
    """Supprime les balises HTML d'un texte."""
    if not text:
        return ""

    return BeautifulSoup(text, "html.parser").get_text(
        separator=" ",
        strip=True,
    )

def preprocess_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et structure les événements OpenAgenda avant indexation.

    - conserve uniquement les champs utiles au système RAG ;
    - supprime les événements sans titre ;
    - supprime les doublons à partir de l'identifiant OpenAgenda ;
    - convertit les dates au format datetime ;
    - nettoie les champs textuels principaux.
    """

    df = events.copy()

    existing_columns = [
        column
        for column in COLUMNS_TO_KEEP
        if column in df.columns
    ]

    df = df[existing_columns]

    # Un événement sans titre n'est pas exploitable.
    df = df.dropna(subset=["title_fr"])

    # L'UID OpenAgenda identifie un événement.
    df = df.drop_duplicates(subset=["uid"])

    # Conversion des dates.
    for column in ["firstdate_begin", "lastdate_end"]:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
                utc=True,
            )

    # Nettoyage des principaux champs textuels.
    text_columns = [
        "title_fr",
        "description_fr",
        "longdescription_fr",
        "conditions_fr",
        "keywords_fr",
        "category",
        "location_name",
        "location_address",
        "location_city",
        "location_department",
        "location_region",
        "accessibility_label_fr",
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    for column in ["description_fr", "longdescription_fr"]:
        if column in df.columns:
            df[column] = df[column].apply(clean_html)

    return df.reset_index(drop=True)