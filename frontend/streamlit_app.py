"""
ScholarRAG — Streamlit Frontend
Production-grade UI for the RAG pipeline.
"""
from __future__ import annotations

import os
import time

import httpx
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_BASE = f"{BACKEND_URL}/api/v1"

st.set_page_config(
    page_title="ScholarRAG",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }

    .metric-card {
        background: #1e1e2e;
        border: 1px solid #3c3c5c;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }

    .source-card {
        background: #16213e;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .quality-pass { color: #4ade80; font-weight: 600; }
    .quality-fail { color: #f87171; font-weight: 600; }

    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper functions ──────────────────────────────────────────────────────────

def check_backend() -> bool:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def ingest_arxiv(query: str, max_results: int) -> dict:
    r = httpx.post(
        f"{API_BASE}/ingest/arxiv",
        json={"query": query, "max_results": max_results},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def ask_question(question: str, top_k: int, rerank: bool) -> dict:
    r = httpx.post(
        f"{API_BASE}/query",
        json={"question": question, "top_k": top_k, "rerank": rerank},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def upload_pdf(file_bytes: bytes, filename: str) -> dict:
    r = httpx.post(
        f"{API_BASE}/ingest/pdf",
        files={"file": (filename, file_bytes, "application/pdf")},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🎓 ScholarRAG</h1>
        <p style="font-size: 1.1rem; opacity: 0.9;">
            Production-Grade RAG Pipeline for Research Paper QA & Summarization
        </p>
        <p style="font-size: 0.85rem; opacity: 0.7;">
            Hybrid Retrieval · Cross-Encoder Reranking · Citation-Grounded Answers · RAGAS Evaluation
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Backend status ────────────────────────────────────────────────────────────
backend_ok = check_backend()
if backend_ok:
    st.success("✅ Backend connected")
else:
    st.error(f"❌ Backend unreachable at {BACKEND_URL} — start the API server first")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/graduation-cap.png",
        width=80,
    )
    st.title("⚙️ Settings")

    st.subheader("🔍 Retrieval")
    top_k = st.slider("Top-K results", min_value=1, max_value=15, value=5)
    rerank = st.toggle("Enable Re-ranking", value=True)

    st.subheader("📄 Ingestion")
    ingest_method = st.radio(
        "Ingest method", ["ArXiv Search", "Upload PDF"], horizontal=True
    )

    st.divider()
    st.markdown("**Pipeline Stages:**")
    stages = [
        ("✅", "ArXiv API"),
        ("✅", "Unstructured.io"),
        ("✅", "BGE Embeddings"),
        ("✅", "Qdrant + BM25"),
        ("✅", "Hybrid Retrieval"),
        ("✅", "Cross-Encoder Rerank"),
        ("✅", "GPT-4 / Claude"),
        ("✅", "RAGAS Eval"),
    ]
    for icon, stage in stages:
        st.markdown(f"{icon} {stage}")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_query, tab_ingest, tab_eval = st.tabs(
    ["💬 Query", "📥 Ingest Papers", "📊 Evaluate"]
)

# ─── Query Tab ────────────────────────────────────────────────────────────────
with tab_query:
    st.subheader("Ask a Research Question")

    example_questions = [
        "What is the attention mechanism in transformers?",
        "How does RLHF improve language model alignment?",
        "Explain retrieval-augmented generation.",
        "What are the main contributions of BERT?",
    ]
    selected_example = st.selectbox(
        "📌 Example questions", ["Custom question..."] + example_questions
    )

    if selected_example != "Custom question...":
        default_q = selected_example
    else:
        default_q = ""

    question = st.text_area(
        "Your question",
        value=default_q,
        height=100,
        placeholder="Ask anything about the ingested research papers...",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_btn = st.button("🚀 Ask", use_container_width=True, disabled=not backend_ok)

    if ask_btn and question.strip():
        with st.spinner("Running RAG pipeline..."):
            try:
                result = ask_question(question, top_k, rerank)

                # ── Metrics row ───────────────────────────────────────────────
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("⏱️ Latency", f"{result['latency_seconds']}s")
                m2.metric("🔢 Tokens", result["tokens_used"])
                m3.metric("📚 Sources", len(result["sources"]))
                qpass = "✅ Pass" if result["quality_passed"] else "❌ Fail"
                m4.metric("🎯 Quality", f"{qpass} ({result['quality_score']:.0%})")

                st.divider()

                # ── Answer ────────────────────────────────────────────────────
                st.subheader("💡 Answer")
                st.markdown(result["answer"])

                st.divider()

                # ── Sources ───────────────────────────────────────────────────
                st.subheader("📚 Sources")
                for src in result["sources"]:
                    with st.expander(
                        f"[Source {src['index']}] {src['source'][:60]}... — Score: {src['score']:.3f}"
                    ):
                        st.markdown(f"**Snippet:** {src['snippet']}")
                        if src.get("metadata"):
                            st.json(src["metadata"])

            except httpx.HTTPStatusError as exc:
                st.error(f"API error {exc.response.status_code}: {exc.response.text}")
            except Exception as exc:
                st.error(f"Error: {exc}")

# ─── Ingest Tab ───────────────────────────────────────────────────────────────
with tab_ingest:
    st.subheader("📥 Ingest Research Papers")

    if ingest_method == "ArXiv Search":
        st.info("Searches ArXiv API and downloads PDFs for indexing.")
        arxiv_query = st.text_input(
            "ArXiv search query",
            value="large language models survey",
            placeholder="e.g. 'RAG retrieval augmented generation'",
        )
        max_results = st.slider("Max papers to fetch", 1, 50, 10)

        if st.button("📡 Fetch & Index from ArXiv", disabled=not backend_ok):
            with st.spinner(f"Fetching {max_results} papers from ArXiv..."):
                try:
                    result = ingest_arxiv(arxiv_query, max_results)
                    st.success(
                        f"✅ Indexed **{result['chunks_indexed']} chunks** "
                        f"from **{result['papers_fetched']} papers** "
                        f"into collection `{result['collection']}`"
                    )
                    st.balloons()
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    else:
        st.info("Upload a PDF to extract and index its content.")
        uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

        if uploaded_file and st.button("📤 Upload & Index", disabled=not backend_ok):
            with st.spinner("Processing PDF..."):
                try:
                    result = upload_pdf(uploaded_file.read(), uploaded_file.name)
                    st.success(
                        f"✅ Indexed **{result['chunks_indexed']} chunks** "
                        f"from **{uploaded_file.name}**"
                    )
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

# ─── Evaluation Tab ───────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("📊 RAGAS Evaluation Dashboard")
    st.info(
        "Run the RAGAS evaluator from CLI: `python scripts/run_evaluation.py` "
        "then upload results here."
    )

    uploaded_results = st.file_uploader("Upload eval_results.json", type=["json"])
    if uploaded_results:
        import json

        data = json.load(uploaded_results)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔒 Faithfulness", f"{data.get('avg_faithfulness', 0):.1%}")
        col2.metric("🎯 Relevancy", f"{data.get('avg_answer_relevancy', 0):.1%}")
        col3.metric("📌 Precision", f"{data.get('avg_context_precision', 0):.1%}")
        col4.metric("🔁 Recall", f"{data.get('avg_context_recall', 0):.1%}")

        st.subheader("Per-Sample Results")
        if per_sample := data.get("per_sample"):
            import pandas as pd

            df = pd.DataFrame(per_sample)[
                ["question", "faithfulness", "answer_relevancy", "context_precision"]
            ]
            df.columns = ["Question", "Faithfulness", "Relevancy", "Precision"]
            st.dataframe(df, use_container_width=True)
