from fastapi import FastAPI, HTTPException

from api.schemas import (
    AskRequest,
    AskResponse,
    RebuildResponse,
)
from src.data_loader import fetch_events
from src.embeddings import create_documents, split_documents
from src.preprocessing import preprocess_events
from src.rag import RAGSystem
from src.vector_store import build_vector_store


app = FastAPI(
    title="Puls-Events RAG API",
    description=(
        "API de recommandation d'événements culturels à Metz "
        "basée sur un système RAG avec Mistral et FAISS."
    ),
    version="1.0.0",
)


rag_system: RAGSystem | None = None


@app.on_event("startup")
def startup_event():
    global rag_system

    rag_system = RAGSystem()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_loaded": rag_system is not None,
    }


@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask(request: AskRequest):
    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="Le système RAG n'est pas disponible.",
        )

    try:
        return rag_system.ask(request.question)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.post(
    "/rebuild",
    response_model=RebuildResponse,
)
def rebuild():
    global rag_system

    try:
        events = fetch_events(city="Metz")
        processed_events = preprocess_events(events)

        documents = create_documents(processed_events)
        chunks = split_documents(documents)

        vector_store = build_vector_store(chunks)

        rag_system = RAGSystem()

        return {
            "message": "Index FAISS reconstruit avec succès.",
            "events_count": len(processed_events),
            "chunks_count": len(chunks),
            "vectors_count": vector_store.index.ntotal,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la reconstruction : {error}",
        ) from error