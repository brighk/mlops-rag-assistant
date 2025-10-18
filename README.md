# MLOps RAG Assistant

A production-ready MLOps project implementing Retrieval-Augmented Generation (RAG) for document question-answering using local LLMs.

## 🎯 Project Overview

This project demonstrates a complete MLOps workflow for a RAG-based AI assistant that can answer questions based on your own documents. It combines:

- **Document Processing**: Ingest and vectorize documents
- **Semantic Search**: ChromaDB vector database for retrieval
- **LLM Integration**: Ollama for local LLM inference
- **Experiment Tracking**: MLflow for logging queries and responses
- **API Service**: FastAPI for serving predictions
- **Full MLOps**: Versioning, monitoring, and reproducibility

## 🏗️ Architecture
```
Documents → Embeddings → Vector DB (ChromaDB)
                              ↓
User Query → Retriever → Top-K Documents → LLM (Ollama) → Answer
                              ↓
                          MLflow Tracking
```

## 🛠️ Tech Stack

- **LLM**: Ollama (Phi-3, Llama3.2, etc.)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB
- **Framework**: LangChain
- **API**: FastAPI
- **Experiment Tracking**: MLflow
- **Data Versioning**: DVC (optional)

## 📊 Features

✅ Local LLM inference (privacy-first, no API costs)  
✅ Document ingestion pipeline  
✅ Semantic search with vector embeddings  
✅ RAG-based question answering  
✅ Experiment tracking with MLflow  
✅ RESTful API for integration  
✅ Comprehensive logging  

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Ollama installed
- 8GB+ RAM (16GB recommended)

### Installation

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/mlops-rag-assistant.git
cd mlops-rag-assistant
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

4. Install and setup Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull phi3

# Verify it's running
ollama list
```

### Usage

#### 1. Add Documents

Place your documents (`.txt` or `.pdf`) in the `data/raw/` directory:
```bash
cp your_documents.txt data/raw/
```

#### 2. Ingest Documents
```bash
python src/data/ingest_documents.py
```

This will:
- Process your documents
- Create embeddings
- Store them in ChromaDB

#### 3. Test RAG System
```bash
python test_rag.py
```

#### 4. Start API Server
```bash
python src/api/main.py
```

Access the API at:
- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

#### 5. Query the API
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "return_sources": true}'
```

#### 6. View Experiments in MLflow
```bash
mlflow ui
```

Open: http://localhost:5000

## 📁 Project Structure
```
mlops-rag-assistant/
├── data/
│   ├── raw/              # Original documents
│   ├── processed/        # Processed documents
│   └── vector_store/     # ChromaDB storage
├── src/
│   ├── data/
│   │   └── ingest_documents.py  # Document processing
│   ├── models/
│   │   ├── llm.py               # Transformers LLM (optional)
│   │   └── llm_ollama.py        # Ollama LLM wrapper
│   ├── rag/
│   │   ├── retriever.py         # Vector retrieval
│   │   └── pipeline.py          # Complete RAG pipeline
│   └── api/
│       └── main.py              # FastAPI application
├── configs/
│   └── config.yaml       # Configuration
├── tests/               # Unit tests
├── notebooks/           # Jupyter notebooks
├── docker/             # Docker files (coming soon)
├── test_rag.py         # Testing script
├── requirements.txt
└── README.md
```

## 🎓 How It Works

### 1. Document Ingestion
- Documents are split into chunks (500 chars with 50 char overlap)
- Each chunk is converted to embeddings using SentenceTransformers
- Embeddings stored in ChromaDB for fast semantic search

### 2. Query Processing
- User query is converted to embedding
- Top-K most similar document chunks retrieved
- Retrieved context + query sent to LLM

### 3. Answer Generation
- LLM (via Ollama) generates answer based on context
- Answer is grounded in your documents (reduces hallucination)
- All interactions logged to MLflow

## 🔧 Configuration

Edit `configs/config.yaml`:
```yaml
llm:
  model_name: "microsoft/Phi-3-mini-4k-instruct"  # Model to use
  max_length: 512                                  # Max response tokens
  temperature: 0.7                                 # Creativity (0-1)

embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

rag:
  chunk_size: 500      # Document chunk size
  chunk_overlap: 50    # Overlap between chunks
  top_k: 3            # Number of chunks to retrieve
```

## 📈 Example Queries

The system can answer questions like:

- "What is machine learning?"
- "Explain the key components of MLOps"
- "What Python libraries are used for data science?"
- "How does supervised learning work?"

Answers are grounded in your uploaded documents!

## 🐳 Docker Deployment (Coming Soon)
```bash
docker-compose up -d
```

## 🧪 Testing
```bash
# Run unit tests
pytest tests/

# Test RAG pipeline
python test_rag.py

# Test API
curl http://localhost:8000/health
```

## 📊 MLOps Features Demonstrated

1. **Experiment Tracking**: All queries logged to MLflow
2. **Reproducibility**: Config-driven, versioned code
3. **Monitoring**: API metrics and logging
4. **Modularity**: Separated concerns (retrieval, generation, API)
5. **Scalability**: Easy to swap components (different LLMs, vector DBs)

## 🔮 Future Enhancements

- [ ] Streaming responses
- [ ] Multi-document support with metadata filtering
- [ ] A/B testing different LLMs
- [ ] Monitoring dashboard (Prometheus + Grafana)
- [ ] CI/CD pipeline
- [ ] Kubernetes deployment
- [ ] Advanced retrieval strategies (hybrid search)
- [ ] Chat history and conversation memory

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License

## 👤 Author

[Your Name] - [GitHub](https://github.com/brighk)

## 🙏 Acknowledgments

- Ollama for local LLM inference
- LangChain for RAG framework
- ChromaDB for vector storage
- MLflow for experiment tracking

---

**⭐ Star this repo if you find it helpful!**
