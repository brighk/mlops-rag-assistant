"""
FastAPI application — RAG Assistant API
"""
import os
import time

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="RAG Assistant API",
    description=(
        "Self-hosted document Q&A with local LLM inference. "
        "No external API calls — all inference runs via Ollama on your infrastructure."
    ),
    version="1.0.0",
)

pipeline = None


@app.on_event("startup")
def startup():
    global pipeline
    from src.rag.pipeline import RAGPipeline
    pipeline = RAGPipeline()


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class QueryRequest(BaseModel):
    question: str
    return_sources: bool = True


class SourceItem(BaseModel):
    content: str
    source: str
    relevance_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem] | None = None
    num_sources: int
    latency_ms: int


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health", summary="Health check")
def health():
    """Check liveness of the API, Ollama, and ChromaDB."""
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        r = requests.get(f"{ollama_host}/api/tags", timeout=3)
        ollama_status = "connected" if r.status_code == 200 else "error"
    except Exception:
        ollama_status = "unreachable"

    chromadb_status = "connected"
    try:
        import chromadb
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "data/vector_store")
        chromadb.PersistentClient(path=persist_dir)
    except Exception:
        chromadb_status = "error"

    return {
        "status": "ok",
        "ollama": ollama_status,
        "chromadb": chromadb_status,
    }


@app.post("/query", response_model=QueryResponse, summary="Query the RAG pipeline")
def query(request: QueryRequest):
    """
    Ask a question against the ingested document corpus.
    The answer is grounded in retrieved document chunks — no hallucination from
    training data alone.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    start = time.time()
    result = pipeline.query(request.question, return_sources=request.return_sources)
    latency_ms = int((time.time() - start) * 1000)

    sources = None
    if request.return_sources and result.get("sources"):
        sources = [
            SourceItem(
                content=doc["content"][:400],
                source=doc["metadata"].get("source", "unknown"),
                relevance_score=round(doc["relevance_score"], 4),
            )
            for doc in result["sources"]
        ]

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=sources,
        num_sources=result["num_sources"],
        latency_ms=latency_ms,
    )


@app.post("/ingest", summary="Trigger document ingestion")
def ingest():
    """
    Ingest all documents in `data/raw/` into the ChromaDB vector store.
    Idempotent — re-running overwrites existing embeddings.
    """
    from src.data.ingest_documents import DocumentIngestor
    ingestor = DocumentIngestor()
    ingestor.ingest()
    return {"status": "ok", "message": "Ingestion complete"}


@app.get("/documents", summary="List ingested document chunks")
def list_documents():
    """Return the total number of chunks stored in the vector database."""
    try:
        import chromadb
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "data/vector_store")
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection("documents")
        return {"total_chunks": collection.count()}
    except Exception as e:
        return {"total_chunks": 0, "note": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
