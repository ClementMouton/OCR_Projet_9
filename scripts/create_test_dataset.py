from pathlib import Path

import pandas as pd


OUTPUT_PATH = Path("data/evaluation/test_dataset.csv")


TEST_CASES = [
    {
        "id": 1,
        "question": (
            "Où a lieu la Semaine des Métiers du Tourisme 2026 ?"
        ),
        "reference_answer": (
            "La Semaine des Métiers du Tourisme 2026 a lieu "
            "à l'Agence Inspire Metz, 2 place d'Armes."
        ),
        "category": "factual",
    },
    {
        "id": 2,
        "question": (
            "Quand a lieu la Semaine des Métiers du Tourisme 2026 ?"
        ),
        "reference_answer": (
            "La Semaine des Métiers du Tourisme 2026 a lieu "
            "le vendredi 6 février à 14h00."
        ),
        "category": "factual",
    },
    {
        "id": 3,
        "question": (
            "Quel type de poste LIDL recrute-t-il lors de son "
            "événement à Metz ?"
        ),
        "reference_answer": (
            "LIDL recrute des préparateurs de commandes H/F en CDI "
            "pour sa plateforme régionale de Montoy-Flanville."
        ),
        "category": "factual",
    },
    {
        "id": 4,
        "question": (
            "Où a lieu l'exposition Identités régionales et modernité ?"
        ),
        "reference_answer": (
            "L'exposition Identités régionales et modernité a lieu "
            "à l'Hôtel de Ville, place d'Armes à Metz."
        ),
        "category": "factual",
    },
    {
        "id": 5,
        "question": (
            "Quel concert du groupe Ylesia est disponible à Metz ?"
        ),
        "reference_answer": (
            "Ylesia - Concert Metz a lieu au Jardin de l'Esplanade "
            "à Metz le dimanche 21 juin 2026 à 22h00."
        ),
        "category": "factual",
    },
    {
        "id": 6,
        "question": (
            "Je cherche un événement autour du blues à Metz. "
            "Que peux-tu me proposer ?"
        ),
        "reference_answer": (
            "Blues Bazar Revival est un concert prévu au Jardin "
            "de l'Esplanade à Metz le dimanche 21 juin à 20h15."
        ),
        "category": "recommendation",
    },
    {
        "id": 7,
        "question": (
            "Quelle activité à vélo permet de découvrir "
            "le patrimoine messin ?"
        ),
        "reference_answer": (
            "Pédalons vers l'Histoire propose de découvrir "
            "le patrimoine messin à vélo les 20 et 21 septembre 2025."
        ),
        "category": "factual",
    },
    {
        "id": 8,
        "question": (
            "Quel événement de recrutement Chausséa propose-t-il "
            "dans le secteur de la logistique ?"
        ),
        "reference_answer": (
            "Chausséa recrute notamment des préparateurs de commandes "
            "et des caristes pour son site logistique situé à Trémery."
        ),
        "category": "factual",
    },
    {
        "id": 9,
        "question": (
            "Quand peut-on visiter la chapelle Sainte-Blandine ?"
        ),
        "reference_answer": (
            "La visite libre de la chapelle Sainte-Blandine "
            "est proposée les 20 et 21 septembre 2025."
        ),
        "category": "temporal",
    },
    {
        "id": 10,
        "question": (
            "Quand a lieu l'atelier CV à l'Agence Metz Blida ?"
        ),
        "reference_answer": (
            "La Semaine des Métiers du Tourisme 2026 a lieu "
            "du lundi 2 au vendredi 6 février 2026, avec des "
            "créneaux à 10h00 et 14h00 chaque jour."
        ),
        "category": "temporal",
    },
    {
        "id": 11,
        "question": (
            "Quels événements culturels sont disponibles à Nancy ?"
        ),
        "reference_answer": (
            "Le corpus utilisé contient uniquement des événements "
            "situés à Metz et ne permet pas de recommander "
            "des événements à Nancy."
        ),
        "category": "no_answer",
    },
    {
        "id": 12,
        "question": (
            "Quels festivals sont disponibles à Strasbourg ?"
        ),
        "reference_answer": (
            "Le corpus utilisé contient uniquement des événements "
            "situés à Metz et ne permet pas de recommander "
            "des festivals à Strasbourg."
        ),
        "category": "no_answer",
    },
]


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(TEST_CASES)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        f"Jeu d'évaluation créé : {OUTPUT_PATH}"
    )
    print(
        f"Nombre de questions : {len(df)}"
    )
    print("\nRépartition :")
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()