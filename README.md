# 🎓 ScholarRAG — Production-Grade RAG Pipeline for Research Paper QA & Summarization

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?style=for-the-badge&logo=fastapi)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=for-the-badge&logo=docker)
![LangChain](https://img.shields.io/badge/LangChain-Powered-yellow?style=for-the-badge)

**A production-ready Retrieval-Augmented Generation system for intelligent research paper Q&A and summarization, featuring hybrid search, cross-encoder re-ranking, citation-aware generation, and full observability.**

[Architecture](#-architecture) • [Quick Start](#-quick-start) • [API Docs](#-api-reference) • [Deployment](#-deployment) • [Contributing](#-contributing)

</div>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ScholarRAG Pipeline                          │
└─────────────────────────────────────────────────────────────────────┘

  📄 PDF / Papers
       │
       ▼
  ┌──────────────┐      ┌──────────────────┐      ┌──────────────────┐
  │ Data         │      │  Text Chunking    │      │  Embedding       │
  │ Ingestion    │─────▶│  LangChain        │─────▶│  Generation      │
  │ Unstructured │      │  TextSplitter     │      │  BGE-large / E5  │
  └──────────────┘      └──────────────────┘      └────────┬─────────┘
                                                            │
                        ┌───────────────────────────────────┤
                        ▼                                   ▼
                  ┌─────────────┐                  ┌─────────────────┐
                  │ Vector Store│                  │  Keyword Index  │
                  │  (Qdrant)   │                  │    (BM25)       │
                  └──────┬──────┘                  └────────┬────────┘
                         │                                  │
                         └──────────────┬───────────────────┘
                                        ▼
                              ┌──────────────────┐
                              │ Hybrid Retrieval  │
                              │ Dense + Sparse    │
                              └────────┬──────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │   Re-ranking      │
                              │ Cross-Encoder /   │
                              │ Cohere Rerank     │
                              └────────┬──────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │  Top-K Context   │
                              │    Chunks        │
                              └────────┬──────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ Prompt Builder   │
                              │ Context + Query  │
                              │  + Citations     │
                              └────────┬──────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │   LLM (GPT-4 /  │
                              │   Claude / etc.) │
                              └────────┬──────────┘
                                       │
                              ┌────────▼──────────┐
                              │  Quality Check    │
                              │  (Pass / Fail)    │
                              └────────┬──────────┘
                                       │ Pass
                                       ▼
                              ┌──────────────────┐
                              │ Answer with       │
                              │  Citations        │
                              └──────────────────┘

  Observability: LangSmith + Prometheus + Grafana
  CI/CD: GitHub Actions → Docker Container
  Frontend: Streamlit / React
  Backend: FastAPI
```

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Hybrid Retrieval** | Combines dense (embedding) + sparse (BM25) search for maximum recall |
| 🎯 **Cross-Encoder Re-ranking** | Cohere Rerank / local cross-encoder for precision boosting |
| 📝 **Citation-Aware Generation** | Every answer is grounded with source references |
| ✅ **Quality Gating** | Automatic answer quality validation before delivery |
| 📊 **Full Observability** | LangSmith traces + Prometheus metrics + Grafana dashboards |
| 🐳 **Containerized** | Docker Compose for one-command local setup |
| 🔄 **CI/CD Ready** | GitHub Actions for automated test + deploy pipelines |
| 📄 **Multi-format Ingestion** | PDF, DOCX, HTML via Unstructured.io |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ScholarRAG.git
cd ScholarRAG
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Start Infrastructure (Qdrant + Monitoring)

```bash
docker compose -f infrastructure/docker/docker-compose.yml up -d qdrant prometheus grafana
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Backend

```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Launch the Frontend

```bash
streamlit run frontend/streamlit_app.py
```

### 7. Ingest Your Papers

```bash
python scripts/ingest_papers.py --input ./data/papers/ --collection research
```

Visit:
- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000
- **Qdrant UI**: http://localhost:6333/dashboard

---

## 📁 Project Structure

```
ScholarRAG/
├── .github/workflows/          # CI/CD pipelines
│   ├── ci.yml                  # Lint, test, build
│   └── cd.yml                  # Deploy to production
├── backend/
│   ├── api/                    # FastAPI application
│   │   ├── main.py             # App entrypoint
│   │   ├── routes/             # API route handlers
│   │   └── middleware/         # Logging, auth middleware
│   ├── core/                   # Config, exceptions
│   ├── ingestion/              # Document loading & chunking
│   ├── embeddings/             # BGE-large / E5 encoders
│   ├── retrieval/              # Qdrant, BM25, hybrid, reranker
│   ├── generation/             # Prompt builder, LLM, quality check
│   ├── monitoring/             # LangSmith, Prometheus
│   └── tests/                  # Unit + integration tests
├── frontend/
│   ├── streamlit_app.py        # Streamlit UI
│   └── assets/                 # Static assets
├── infrastructure/
│   ├── docker/                 # Dockerfiles + Compose
│   ├── prometheus/             # Prometheus config
│   └── grafana/                # Grafana dashboards
├── scripts/                    # CLI utilities
├── notebooks/                  # Exploratory notebooks
├── docs/                       # Documentation
├── data/                       # Local paper storage (gitignored)
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🌐 API Reference

### Ingest Papers

```http
POST /api/v1/ingest
Content-Type: multipart/form-data

file: <PDF file>
collection_name: research
```

### Query

```http
POST /api/v1/query
Content-Type: application/json

{
  "question": "What are the main contributions of transformer-based models?",
  "collection_name": "research",
  "top_k": 5,
  "rerank": true
}
```

### Health Check

```http
GET /api/v1/health
```

Full OpenAPI docs available at: `http://localhost:8000/docs`

---

## ⚙️ Configuration

Key environment variables (see `.env.example`):

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for LLM |
| `COHERE_API_KEY` | Cohere API key for reranking |
| `QDRANT_URL` | Qdrant vector DB URL |
| `QDRANT_API_KEY` | Qdrant cloud API key (optional) |
| `LANGCHAIN_API_KEY` | LangSmith API key |
| `EMBEDDING_MODEL` | BGE-large-en-v1.5 or BAAI/bge-large-en |
| `LLM_MODEL` | gpt-4o, claude-3-5-sonnet, etc. |
| `CHUNK_SIZE` | Token chunk size (default: 512) |
| `CHUNK_OVERLAP` | Token overlap between chunks (default: 64) |
| `TOP_K_RETRIEVAL` | Number of retrieved chunks (default: 20) |
| `TOP_K_RERANK` | Final chunks after reranking (default: 5) |

---

## 🐳 Deployment

### Docker Compose (Full Stack)

```bash
docker compose -f infrastructure/docker/docker-compose.yml up --build
```

### Individual Services

```bash
# Backend only
docker build -f infrastructure/docker/Dockerfile.backend -t scholarrag-backend .
docker run -p 8000:8000 --env-file .env scholarrag-backend

# Frontend only
docker build -f infrastructure/docker/Dockerfile.frontend -t scholarrag-frontend .
docker run -p 8501:8501 scholarrag-frontend
```

---

## 🧪 Testing

```bash
# Run all tests
pytest backend/tests/ -v

# Unit tests only
pytest backend/tests/unit/ -v

# Integration tests (requires running services)
pytest backend/tests/integration/ -v

# With coverage
pytest backend/tests/ --cov=backend --cov-report=html
```

---

## 📊 Monitoring

- **LangSmith**: Tracks LLM traces, latency, token usage
- **Prometheus**: Collects API metrics (requests, latency, errors)
- **Grafana**: Pre-built dashboards for pipeline monitoring

Access Grafana at `http://localhost:3000` (admin/admin).

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ for the research community

**ScholarRAG** • Production-Grade RAG for Academic Research

</div>
