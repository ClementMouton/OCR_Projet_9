from pathlib import Path

from src.data_loader import fetch_events
from src.preprocessing import preprocess_events


RAW_PATH = Path("data/raw/events_metz.csv")
PROCESSED_PATH = Path("data/processed/events_metz.csv")


def main():
    events = fetch_events(city="Metz")

    RAW_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    events.to_csv(
        RAW_PATH,
        index=False,
        encoding="utf-8",
    )

    processed_events = preprocess_events(events)

    PROCESSED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_events.to_csv(
        PROCESSED_PATH,
        index=False,
        encoding="utf-8",
    )

    print(f"Événements récupérés : {len(events)}")
    print(f"Événements après nettoyage : {len(processed_events)}")
    print(f"Nombre de colonnes conservées : {len(processed_events.columns)}")

    print("\nValeurs manquantes :")
    print(processed_events.isna().sum())

    print(f"\nDonnées brutes : {RAW_PATH}")
    print(f"Données nettoyées : {PROCESSED_PATH}")


if __name__ == "__main__":
    main()