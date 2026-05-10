# System Architecture

## Overview

The RAG Assistant is a fully self-hosted retrieval-augmented generation (RAG) system. It answers natural-language questions against a private document corpus using local LLM inference — no external API calls, no data leaving the host.

---

## Ingestion Flow

```
data/raw/ (PDF / TXT)
    │
    ▼
DocumentIngestor
    │  RecursiveCharacterTextSplitter
    │  chunk_size=500, chunk_overlap=50
    ▼
SentenceTransformers (all-MiniLM-L6-v2)
    │  384-dimensional dense vectors
    │  Runs on CPU or CUDA
    ▼
ChromaDB (PersistentClient)
    │  data/vector_store/
    │  Collection: "documents"
    ▼
Ready for retrieval
```

**Key decisions:**
- Chunk overlap prevents context loss at boundaries.
- `all-MiniLM-L6-v2` is fast, lightweight, and sufficient for English enterprise docs.
- ChromaDB persists to disk — no data loss on restart.

---

## Query Flow

```
User Query (HTTP POST /query)
    │
    ▼
Embed query → 384-dim vector
    │
    ▼
ChromaDB similarity_search_with_score (cosine)
    │  Returns top-k chunks + relevance scores
    ▼
Build context string (Document 1 ... Document k)
    │
    ▼
Prompt template:
    "Use the following context to answer the question..."
    │
    ▼
Ollama HTTP API (POST /api/generate)
    │  Model: phi3 (or any Ollama model)
    │  Runs entirely on-host
    ▼
Answer text
    │
    ▼
MLflow run logged (params, metrics, artifacts)
    │
    ▼
JSON response → client
```

---

## Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `src/data/ingest_documents.py` | Load, chunk, embed, store |
| `src/rag/retriever.py` | Query ChromaDB, return top-k chunks |
| `src/rag/pipeline.py` | Orchestrate retrieval → prompt → generation → MLflow |
| `src/models/llm_ollama.py` | Ollama HTTP client wrapper |
| `src/api/main.py` | FastAPI REST interface |
| `configs/config.yaml` | Central configuration (chunk size, top-k, model, etc.) |

---

## Security Model

### Data Residency
- All documents, embeddings, and LLM inference run on-host.
- No document content, queries, or answers are transmitted externally.
- Ollama communicates only with its local model files.

### Network Exposure
- FastAPI binds to `0.0.0.0:8000` — place behind a reverse proxy (nginx/Caddy) in production.
- Ollama (`11434`) and MLflow (`5000`) should **not** be publicly exposed; use internal networking.
- In Docker Compose, Ollama and MLflow are reachable only from within the Docker network unless explicitly mapped.

### Authentication (Roadmap)
- Current: no authentication (suitable for internal/trusted networks).
- Planned: API key middleware, OAuth2/JWT for multi-user deployments.

### Secrets
- `.env` is gitignored — never commit credentials.
- Use `.env.example` as the template for deployment.

---

## Backup Strategy

### Vector Store
ChromaDB persists to `data/vector_store/`. Back this up like any stateful directory:

```bash
# Simple tar backup
tar -czf backup-$(date +%F).tar.gz data/vector_store/

# Restore
tar -xzf backup-2025-01-01.tar.gz
```

In Docker, the volume is named `rag_vector_store` (via bind mount `./data`). Back up the host path.

### MLflow Runs
MLflow artifacts live in `mlruns/` (or the Docker volume `mlflow_data`). Include this in your backup rotation.

### Models
Ollama model weights live in `~/.ollama/models` (or the `ollama_models` Docker volume). These can be re-pulled with `ollama pull <model>` — no backup required unless bandwidth is constrained.

### Recommended Schedule
| Data | Frequency | Method |
|------|-----------|--------|
| `data/vector_store/` | Daily | tar + S3/MinIO upload |
| `mlruns/` | Daily | tar + S3/MinIO upload |
| `data/raw/` (source docs) | On change | Git or S3 versioning |

---

## Limitations

| Limitation | Detail |
|------------|--------|
| No streaming | Responses are returned as a complete block; streaming via SSE is on the roadmap. |
| Single collection | All documents share one ChromaDB collection; no per-team isolation yet. |
| No authentication | API is open; intended for internal networks. Auth is planned. |
| Chunk-level retrieval | No re-ranking or hybrid BM25+semantic search (planned). |
| English-optimized | `all-MiniLM-L6-v2` works best on English text. Swap embedding model for multilingual use. |
| LLM quality | Answer quality depends on the Ollama model selected. Phi-3 is fast; Llama 3.2 is more accurate. |
| No conversation memory | Each `/query` call is stateless. Chat history is on the roadmap. |

---

## Deployment Notes

### Minimum Requirements
| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 8 cores |
| Disk | 20 GB | 50 GB |
| GPU | None (CPU inference) | NVIDIA GPU for faster generation |

### Docker Compose (Production)
```bash
# Start all services
docker compose up -d

# Pull model (first time only)
docker exec -it rag-ollama ollama pull phi3

# Ingest documents
docker exec -it rag-api python src/data/ingest_documents.py

# View logs
docker compose logs -f app
```

### Reverse Proxy (nginx example)
```nginx
server {
    listen 80;
    server_name rag.internal.company.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Switching LLM Models
Update `.env` — no rebuild needed:
```bash
OLLAMA_MODEL=llama3.2
docker compose up -d
```

Or pull and use any Ollama-compatible model:
```bash
docker exec -it rag-ollama ollama pull mistral
OLLAMA_MODEL=mistral docker compose up -d
```

### Scaling Considerations
- ChromaDB does not support horizontal scaling natively. For high-throughput deployments, migrate to **pgvector** (PostgreSQL extension) — see Enterprise Roadmap.
- Multiple API replicas can run behind a load balancer as long as they share the same `data/vector_store/` mount.
- Ollama does not support concurrent requests; add a request queue or run multiple Ollama instances with a round-robin proxy for parallel query handling.

---

## Enterprise Roadmap

See the main [README.md](../README.md#enterprise-roadmap) for the full feature backlog.
