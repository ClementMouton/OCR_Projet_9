# Puls-Events – Système RAG de recommandation d'événements culturels

Projet réalisé dans le cadre du parcours **Data Scientist – Machine Learning** d'OpenClassrooms.

L'objectif du projet est de développer un POC d'assistant capable de rechercher et recommander des événements culturels à Metz à partir des données OpenAgenda.

Le système repose sur une architecture **RAG (Retrieval-Augmented Generation)** combinant :

- les données événementielles d'OpenAgenda ;
- des embeddings Mistral ;
- un index vectoriel FAISS ;
- un mécanisme de recherche et de reranking ;
- un LLM Mistral pour générer les réponses ;
- une API REST développée avec FastAPI ;
- une conteneurisation avec Docker.

---

## 1. Architecture

Le fonctionnement général du système est le suivant :

```text
                  OpenAgenda
                      │
                      ▼
            Collecte des événements
                      │
                      ▼
          Nettoyage / prétraitement
                      │
                      ▼
             Documents LangChain
                      │
                      ▼
        Chunking (1000 / overlap 150)
                      │
                      ▼
            Embeddings Mistral
                      │
                      ▼
                 FAISS
                      │
              ┌───────┴───────┐
              │               │
              ▼               │
      Recherche vectorielle   │
              │               │
              ▼               │
        Reranking lexical     │
              │               │
              ▼               │
       Filtrage temporel      │
              │               │
              ▼               │
        Contexte pertinent    │
              │               │
              ▼               │
       Mistral + LangChain    │
              │               │
              ▼               │
       Réponse + sources      │
              │               │
              ▼               │
          API FastAPI         │
              │               │
              └── /rebuild ───┘
```

L'index FAISS peut être reconstruit à partir des données OpenAgenda via le script d'indexation ou directement depuis l'API.

---

## 2. Données

Les événements proviennent du jeu de données public **OpenAgenda** accessible via l'API OpenDataSoft.

Le POC est limité géographiquement à **Metz**.

Lors de la collecte, seuls les événements dont la date de fin se situe dans une fenêtre d'un an avant la date de récupération ou dans le futur sont conservés.

Les données sont ensuite prétraitées afin de conserver les informations utiles au RAG, notamment :

- titre ;
- descriptions courte et détaillée ;
- dates ;
- lieu et adresse ;
- mots-clés ;
- conditions d'accès ;
- accessibilité ;
- URL de l'événement.

Les balises HTML présentes dans certaines descriptions sont également supprimées avant l'indexation.

Les données brutes et les données prétraitées sont conservées respectivement dans :

```text
data/raw/
data/processed/
```

---

## 3. Construction de l'index vectoriel

Chaque événement est transformé en document LangChain contenant son contenu textuel et ses métadonnées.

Les documents sont découpés avec `RecursiveCharacterTextSplitter` selon les paramètres suivants :

```text
chunk_size = 1000 caractères
chunk_overlap = 150 caractères
```

Ce découpage permet de limiter la taille des documents envoyés au moteur de recherche tout en conservant un chevauchement entre les fragments.

Les métadonnées de l'événement sont préservées pendant le chunking :

```text
uid
title
start_date
end_date
location
city
url
```

Les représentations vectorielles sont générées avec :

```text
mistral-embed
```

puis stockées dans un index **FAISS** local.

L'index peut être construit avec :

```powershell
python scripts/build_index.py
```

---

## 4. Pipeline RAG

Le système RAG est implémenté dans `src/rag.py`.

### Recherche

Pour chaque question, le système effectue d'abord une recherche vectorielle dans FAISS sur un ensemble élargi de documents candidats.

Cette première recherche est complétée par un **reranking lexical** prenant notamment en compte la présence des termes de la question dans :

- le titre ;
- le lieu ;
- le contenu du document.

L'objectif est de conserver les avantages de la recherche sémantique tout en remontant les événements contenant des correspondances lexicales particulièrement pertinentes.

### Gestion des contraintes temporelles

Le système détecte également certaines formulations temporelles telles que :

```text
aujourd'hui
demain
ce vendredi
ce samedi
...
```

Lorsqu'une contrainte temporelle est détectée, les événements sont filtrés à partir de leurs métadonnées de début et de fin.

### Génération

Les documents retenus sont transmis au modèle :

```text
mistral-small-latest
```

avec une température de `0`.

Le prompt système demande au modèle :

- de répondre uniquement à partir du contexte récupéré ;
- de ne pas inventer d'événement ;
- de répondre en français ;
- de signaler lorsqu'aucun événement pertinent n'est disponible.

La réponse de l'API contient également les sources OpenAgenda associées aux événements récupérés.

---

## 5. API REST

Le RAG est exposé avec **FastAPI**.

L'API fournit trois endpoints principaux.

### `GET /health`

Vérifie que l'API fonctionne et que le système RAG est chargé.

Exemple :

```json
{
  "status": "ok",
  "rag_loaded": true
}
```

### `POST /ask`

Interroge le système RAG.

Exemple de requête :

```json
{
  "question": "Quels concerts sont disponibles à Metz ?"
}
```

La réponse contient :

```json
{
  "question": "...",
  "answer": "...",
  "sources": [...]
}
```

Les entrées sont validées avec Pydantic. Une question doit notamment :

- être une chaîne de caractères ;
- contenir entre 3 et 500 caractères ;
- contenir au moins un caractère alphabétique ;
- ne pas être accompagnée de champs non prévus par le schéma.

### `POST /rebuild`

Récupère à nouveau les événements OpenAgenda et reconstruit l'index FAISS utilisé par le RAG.

La documentation Swagger est disponible lorsque l'API est lancée :

```text
http://localhost:8000/docs
```

---

## 6. Évaluation du RAG

Un jeu de test annoté a été créé afin d'évaluer le comportement du système.

Il contient **12 questions** couvrant plusieurs types de requêtes :

- questions factuelles ;
- questions temporelles ;
- recommandations ;
- questions pour lesquelles aucune réponse ne doit être trouvée dans le corpus.

Le dataset se trouve dans :

```text
data/evaluation/test_dataset.csv
```

Les réponses du système sont générées avec :

```powershell
python scripts/evaluate_rag.py
```

puis évaluées avec :

```powershell
python scripts/evaluate_metrics.py
```

### Métriques automatiques

L'évaluation reprend plusieurs dimensions couramment utilisées pour les systèmes RAG :

| Métrique | Résultat |
|---|---:|
| Faithfulness | 95 % |
| Answer Correctness | 75 % |
| Answer Relevancy | 100 % |
| Context Recall | 83,33 % |
| Exact Match | 0 % |

L'Exact Match est particulièrement strict pour un système génératif : une réponse peut être correcte sans reproduire exactement la réponse de référence.

Les autres métriques sont évaluées automatiquement à l'aide d'un LLM juge. Elles doivent donc être interprétées comme des indicateurs et non comme une mesure parfaitement déterministe.

### Évaluation manuelle

Une vérification humaine complémentaire des 12 réponses a donné :

```text
11 réponses correctes : 91,7 %
1 réponse partiellement correcte : 8,3 %
0 réponse incorrecte
```

Cette évaluation permet notamment de compléter les métriques automatiques lorsque le juge LLM pénalise une réponse pourtant correcte ou pertinente.

Les résultats sont disponibles dans :

```text
data/evaluation/
├── test_dataset.csv
├── rag_results.csv
├── metrics_results.csv
├── metrics_summary.json
└── manual_evaluation.csv
```

---

## 7. Tests

Les tests automatisés couvrent :

- récupération et validation des données ;
- gestion des dates ;
- création des documents ;
- chunking ;
- préservation des métadonnées ;
- comportement du RAG ;
- validation des entrées de l'API ;
- endpoints `/health`, `/ask` et `/rebuild`.

Pour lancer la suite :

```powershell
python -m pytest -v
```

État actuel :

```text
31 tests passed
```

Un test fonctionnel séparé permet également de vérifier l'API en conditions réelles :

```powershell
python api_test.py
```

Il appelle successivement `/health` puis `/ask` sur une API en cours d'exécution.

---

## 8. Installation

### Prérequis

- Python 3.11
- une clé API Mistral
- Docker, pour l'exécution conteneurisée

Le projet a été développé avec **Python 3.11.9**.

### Cloner le dépôt

```powershell
git clone https://github.com/ClementMouton/OCR_Projet_9.git
cd OCR_Projet_9
```

### Créer l'environnement virtuel

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Installer les dépendances

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configurer Mistral

Créer un fichier `.env` à partir de `.env.example` :

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

Le fichier contenant la véritable clé API est exclu du versionnement Git.

---

## 9. Initialisation du projet

Après l'installation, récupérer les données :

```powershell
python scripts/fetch_events.py
```

Puis construire l'index FAISS :

```powershell
python scripts/build_index.py
```

Lancer ensuite l'API :

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Swagger est alors accessible à l'adresse :

```text
http://localhost:8000/docs
```

---

## 10. Docker

Construire l'image :

```powershell
docker build -t puls-events-rag .
```

Puis lancer le conteneur en injectant les variables d'environnement :

```powershell
docker run --name puls-events-rag -p 8000:8000 --env-file .env puls-events-rag
```

Si le fichier `.env` est situé dans le dossier parent :

```powershell
docker run --name puls-events-rag -p 8000:8000 --env-file ..\.env puls-events-rag
```

L'API est ensuite accessible sur :

```text
http://localhost:8000
```

et Swagger sur :

```text
http://localhost:8000/docs
```

---

## 11. Structure du projet

```text
OCR_Projet_9/
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── scripts/
│   ├── fetch_events.py
│   ├── build_index.py
│   ├── test_search.py
│   ├── analyze_retrieval.py
│   ├── create_test_dataset.py
│   ├── evaluate_rag.py
│   └── evaluate_metrics.py
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── date_utils.py
│   └── rag.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_date_utils.py
│   ├── test_index.py
│   └── test_rag.py
│
├── api_test.py
├── Dockerfile
├── .dockerignore
├── .env.example
├── requirements.txt
└── README.md
```

---

## 12. Choix techniques

### Mistral

Mistral est utilisé à la fois pour les embeddings et pour la génération afin de disposer d'une stack cohérente et adaptée aux contenus en français.

### FAISS

FAISS fournit une recherche vectorielle rapide et simple à intégrer localement. Il est adapté à un POC ne nécessitant pas encore une base vectorielle distribuée ou managée.

### LangChain

LangChain facilite la représentation des documents, le découpage des contenus, l'intégration avec Mistral et l'orchestration du pipeline RAG.

### FastAPI

FastAPI permet d'exposer simplement le système sous forme d'API REST, avec validation Pydantic et documentation OpenAPI/Swagger automatique.

### Docker

Docker permet de reproduire l'environnement d'exécution de l'application indépendamment de la machine utilisée.

---

## 13. Limites et pistes d'amélioration

Le projet constitue un POC et plusieurs évolutions seraient possibles :

- enrichir le parsing des dates et des expressions temporelles complexes ;
- ajouter des filtres structurés sur le type d'événement, le lieu ou l'accessibilité ;
- améliorer le reranking avec un modèle dédié ;
- augmenter et diversifier le jeu d'évaluation ;
- automatiser davantage l'évaluation continue du RAG ;
- gérer plus précisément les fuseaux horaires ;
- rendre l'index FAISS reconstruit persistant lorsque l'application est exécutée dans Docker ;
- remplacer le stockage local par une base vectorielle persistante pour un déploiement à plus grande échelle ;
- mettre en place une interface utilisateur au-dessus de l'API.

Dans la version Docker actuelle, une reconstruction via `/rebuild` modifie l'index présent dans le système de fichiers du conteneur. Cet index est donc perdu si le conteneur est supprimé. Pour un passage en production, un volume persistant ou un stockage vectoriel externe serait préférable.

---

## 14. Sécurité

La clé API Mistral n'est jamais stockée directement dans le code source.

Elle est fournie via la variable d'environnement :

```text
MISTRAL_API_KEY
```

Les fichiers `.env` sont exclus de Git et du contexte de build Docker. Seul `.env.example`, ne contenant aucun secret, est versionné.