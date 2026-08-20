# OCR Projet 9 – Système RAG de recommandation d'événements culturels

Projet réalisé dans le cadre du parcours **Data Scientist – Machine Learning** d'OpenClassrooms.

L'objectif est de développer un POC d'assistant intelligent capable de répondre à des questions et de recommander des événements culturels à partir de données issues d'OpenAgenda.

Le système reposera sur une architecture **RAG (Retrieval-Augmented Generation)** combinant une recherche vectorielle avec FAISS et la génération de réponses avec un modèle Mistral.

## Technologies

Le projet utilise principalement :

- Python 3.11
- LangChain
- Mistral AI
- FAISS
- Pandas
- FastAPI
- Pytest
- Ragas
- Docker

## Structure du projet

```text
OCR_Projet_9/
│
├── api/                    # API REST
├── data/                   # Données du projet
├── scripts/                # Scripts d'exécution
├── src/                    # Logique métier du système RAG
├── tests/                  # Tests unitaires et fonctionnels
│
├── .env.example            # Exemple de configuration des variables d'environnement
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

La structure sera complétée au fur et à mesure du développement.

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/ClementMouton/OCR_Projet_9.git
cd OCR_Projet_9
```

### 2. Créer un environnement virtuel

Le projet a été développé et testé avec **Python 3.11.9**.

Sous Windows PowerShell :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurer la clé API Mistral

Créer un fichier `.env` à la racine du projet à partir du fichier `.env.example` :

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

Remplacer la valeur par une clé API Mistral valide.

Le fichier `.env` contenant la clé réelle est exclu du versionnement Git.

## Architecture cible

Le fonctionnement général du système sera le suivant :

```text
OpenAgenda
    ↓
Collecte et nettoyage des événements
    ↓
Découpage des contenus en chunks
    ↓
Embeddings Mistral
    ↓
Index vectoriel FAISS
    ↓
Recherche des événements pertinents
    ↓
LangChain + Mistral
    ↓
Réponse augmentée
    ↓
API FastAPI
```

## État du projet

### Étape 1 – Environnement de développement

- [x] Environnement Python 3.11
- [x] Environnement virtuel
- [x] Dépendances reproductibles
- [x] Configuration sécurisée de la clé Mistral
- [x] Test des embeddings Mistral
- [x] Test de génération avec Mistral
- [x] Test de FAISS
- [x] Test d'installation dans un environnement propre

### Étapes suivantes

- [ ] Collecte et préparation des données OpenAgenda
- [ ] Construction de l'index vectoriel FAISS
- [ ] Développement du système RAG
- [ ] Création de l'API REST
- [ ] Évaluation automatique du système
- [ ] Tests unitaires et fonctionnels
- [ ] Conteneurisation Docker

## Sécurité

Les informations sensibles, notamment la clé API Mistral, sont stockées dans des variables d'environnement et ne sont pas versionnées dans le dépôt Git.