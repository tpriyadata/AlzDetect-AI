# AlzDetect AI — Streamlit App ───────────
import streamlit as st
import faiss
import numpy as np
import json
import os
import anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

# ── Page Config ────────────────────────────
st.set_page_config(
    page_title="AlzDetect AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem 0;
    }
 AlzDetect AI   .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1E3A5F;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #e8f4f8;
        border-radius: 4px;
        padding: 0.5rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    .metric-box {
        background-color: #1E3A5F;
        color: white;
        padding: 0.5rem;
        border-radius: 4px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Components (Cached) ───────────────
@st.cache_resource
def load_components():
    """Load all RAG components once"""

    # Paths
    CHUNKS_FILE = 'data/processed/chunks.json'
    FAISS_FILE  = 'vector_store/faiss_index/alzheimer.index'

    # Load chunks
    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)

    # Load FAISS
    index = faiss.read_index(FAISS_FILE)

    # Load model
    model = SentenceTransformer(
        'neuml/pubmedbert-base-embeddings'
    )

    # Load Claude
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    return chunks, index, model, client

# ── RAG Functions ──────────────────────────
def retrieve_chunks(query, chunks, index, model, k=5):
    query_vector = model.encode([query])
    query_vector = query_vector.astype('float32')
    distances, indices = index.search(query_vector, k=k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        chunk = chunks[idx]
        results.append({
            "title":    chunk['title'],
            "authors":  chunk['authors'],
            "year":     chunk['year'],
            "journal":  chunk['journal'],
            "chunk":    chunk['chunk'],
            "pmid":     chunk['pmid'],
            "distance": float(dist)
        })
    return results

def generate_answer(query, retrieved, client):
    context = ""
    for i, r in enumerate(retrieved):
        context += f"""
Source {i+1}:
Title:   {r['title']}
Authors: {r['authors']}
Year:    {r['year']}
Journal: {r['journal']}
Content: {r['chunk']}
---"""

    prompt = f"""You are AlzDetect AI — expert
medical research assistant specializing in
Alzheimer's disease research.

Answer using ONLY these PubMed sources.
Cite as (Source 1), (Source 2) etc.
Be specific with biomarker names and findings.
Use markdown formatting.
Never hallucinate.

SOURCES:
{context}

QUESTION: {query}"""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    return response.content[0].text

# ── Main App ───────────────────────────────
def main():
    # Header
    st.markdown(
        '<div class="main-header">🧠 AlzDetect AI</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Enterprise RAG Chatbot for Alzheimer\'s Early Detection Research<br>Powered by PubMed × PubMedBERT × Claude</div>',
        unsafe_allow_html=True
    )

    # Load components
    with st.spinner("Loading AI components..."):
        chunks, index, model, client = load_components()

    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Pipeline Stats")
        st.markdown(
            '<div class="metric-box">📄 16,281 Papers</div>',
            unsafe_allow_html=True
        )
        st.markdown("")
        st.markdown(
            '<div class="metric-box">🧩 40,995 Chunks</div>',
            unsafe_allow_html=True
        )
        st.markdown("")
        st.markdown(
            '<div class="metric-box">🧠 768-dim PubMedBERT</div>',
            unsafe_allow_html=True
        )
        st.markdown("")
        st.markdown(
            '<div class="metric-box">🔍 FAISS Vector Search</div>',
            unsafe_allow_html=True
        )
        st.markdown("")
        st.markdown(
            '<div class="metric-box">🤖 Claude AI</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown("## 💡 Sample Questions")
        sample_questions = [
            "What blood biomarkers detect Alzheimer's early?",
            "How does diabetes connect to Alzheimer's?",
            "How effective is Lecanemab treatment?",
            "What is the role of APOE4 in dementia?",
            "How does deep learning help diagnose Alzheimer's?"
        ]
        for q in sample_questions:
            if st.button(q, key=q):
                st.session_state.query = q

        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.markdown("""
        AlzDetect AI uses enterprise RAG
        to answer Alzheimer's research
        questions with full citations.

        **Data source:** PubMed
        **Model:** PubMedBERT
        **LLM:** Claude API
        **Built by:** TPriya
        """)

    # Main query input
    query = st.text_input(
        "🔍 Ask a research question:",
        value=st.session_state.get('query', ''),
        placeholder="e.g. What blood biomarkers detect Alzheimer's early?"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        search_btn = st.button(
            "🔍 Search",
            type="primary",
            use_container_width=True
        )

    if search_btn and query:
        # Search
        with st.spinner("🔍 Searching 40,995 research chunks..."):
            retrieved = retrieve_chunks(
                query, chunks, index, model, k=5
            )

        # Generate
        with st.spinner("🤖 Generating cited answer with Claude..."):
            answer = generate_answer(
                query, retrieved, client
            )

        # Display answer
        st.markdown("### 🤖 Answer")
        st.markdown(
            f'<div class="answer-box">{answer}</div>',
            unsafe_allow_html=True
        )

        # Display sources
        st.markdown("### 📚 Sources Retrieved")
        for i, r in enumerate(retrieved):
            with st.expander(
                f"Source {i+1}: {r['title'][:60]}..."
            ):
                st.markdown(f"**Title:** {r['title']}")
                st.markdown(f"**Authors:** {r['authors']}")
                st.markdown(f"**Year:** {r['year']}")
                st.markdown(f"**Journal:** {r['journal']}")
                st.markdown(f"**PMID:** {r['pmid']}")
                st.markdown(f"**Relevance Score:** {r['distance']:.2f}")
                st.markdown("**Abstract Excerpt:**")
                st.markdown(f"_{r['chunk'][:300]}..._")

        # Footer metrics
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Sources Found",
                len(retrieved)
            )
        with col2:
            st.metric(
                "Best Distance",
                f"{retrieved[0]['distance']:.2f}"
            )
        with col3:
            st.metric(
                "Papers Searched",
                "16,281"
            )

    elif search_btn and not query:
        st.warning(
            "⚠️ Please enter a question first!"
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """<div style='text-align:center; color:#666; font-size:0.8rem'>
        AlzDetect AI | Built with PubMed × PubMedBERT × FAISS × Claude
        | For research purposes only — not medical advice
        </div>""",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
