"""
RAG Diagnostics Dashboard

Visualizes the internal steps of the RAG pipeline:
1. Query Analysis & HyDE
2. Hybrid Retrieval (BM25 vs Dense)
3. Fusion & Reranking
4. Final Source Attribution
"""

import streamlit as st
import pandas as pd
import pandas as pd
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import time
from typing import List, Dict, Any

from config import settings
from core.retrieval.retriever import LocalRetriever

def render_rag_diagnostics():
    """Render the RAG Diagnostics dashboard."""
    st.header("🔍 RAG Internals & Diagnostics")
    st.caption("Deep Dive into the Neural/Lexical Retrieval Pipeline")

    # Initialize Retriever
    if 'diag_retriever' not in st.session_state:
        try:
            st.session_state.diag_retriever = LocalRetriever()
        except Exception as e:
            st.error(f"Failed to initialize retriever: {e}")
            return

    retriever = st.session_state.diag_retriever

    # Input Section
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Test Query", placeholder="e.g., What are the PK parameters of drug X?")
    with col2:
        top_k = st.number_input("Top K", min_value=1, max_value=50, value=5)

    if not query:
        st.info("Enter a query to inspect the retrieval pipeline.")
        return

    if st.button("🚀 Run Diagnostic Trace", type="primary"):
        _run_diagnostic_trace(retriever, query, top_k)


def _run_diagnostic_trace(retriever: LocalRetriever, query: str, top_k: int):
    """Run and visualize each step of the retrieval pipeline."""
    
    # 1. HyDE Generation
    st.subheader("1. Cognitive Processing (HyDE)")
    with st.spinner("Generating hypothetical answer..."):
        start_time = time.time()
        hyde_doc = retriever._generate_hypothetical_doc(query)
        hyde_time = time.time() - start_time
    
    with st.expander("Values", expanded=True):
        col_q, col_h = st.columns(2)
        with col_q:
            st.markdown("**Original Query**")
            st.info(query)
        with col_h:
            st.markdown(f"**Hallucinated Answer (HyDE)** ({hyde_time:.2f}s)")
            st.success(hyde_doc)

    st.markdown("---")

    # 2. Hybrid Retrieval (Parallel)
    st.subheader("2. Hybrid Retrieval Layers")
    
    col_bm25, col_dense = st.columns(2)
    
    # BM25
    with col_bm25:
        st.markdown("### 📚 Lexical (BM25)")
        with st.spinner("Running BM25..."):
            try:
                # Ensure index exists
                if retriever._bm25_index is None:
                    retriever._build_bm25_index()
                
                start_time = time.time()
                bm25_results = retriever._bm25_search(query, top_k=top_k*2)
                bm25_time = time.time() - start_time
                st.caption(f"Found {len(bm25_results)} matches in {bm25_time:.3f}s")
                
                for i, res in enumerate(bm25_results[:5]):
                    with st.expander(f"#{i+1} Score: {res['score']:.4f}", expanded=False):
                        st.markdown(f"**Source:** `{res['source']}`")
                        st.text(res['content'][:200] + "...")
            except Exception as e:
                st.error(f"BM25 Error: {e}")
                bm25_results = []

    # Dense
    with col_dense:
        st.markdown("### 🧠 Semantic (Dense)")
        with st.spinner("Running Vector Search..."):
            try:
                start_time = time.time()
                # Search using the HyDE doc if available, or just query
                search_query = hyde_doc if hyde_doc != query else query
                dense_results = retriever._dense_search(search_query, top_k=top_k*2)
                dense_time = time.time() - start_time
                st.caption(f"Found {len(dense_results)} matches in {dense_time:.3f}s")
                
                for i, res in enumerate(dense_results[:5]):
                    with st.expander(f"#{i+1} Score: {res['score']:.4f}", expanded=False):
                        st.markdown(f"**Source:** `{res['source']}`")
                        st.text(res['content'][:200] + "...")
            except Exception as e:
                st.error(f"Dense Error: {e}")
                dense_results = []

    st.markdown("---")

    # 3. Fusion & Reranking
    st.subheader("3. Fusion & Cross-Encoder Reranking")
    
    with st.spinner("Fusing and Reranking..."):
        # Fusion
        combined_results = retriever._reciprocal_rank_fusion(
            [dense_results, bm25_results],
            weights=[retriever.dense_weight, retriever.bm25_weight]
        )
        
        # Rerank
        start_time = time.time()
        final_results = retriever._rerank(query, combined_results, top_k=top_k)
        rerank_time = time.time() - start_time

    st.success(f"Reranking completed in {rerank_time:.3f}s")

    # Visualization of Score Delta
    data = []
    for i, res in enumerate(final_results):
        # Calculate approximate initial rank (if present)
        # This is tricky because of RRF, just showing final scores
        data.append({
            "Rank": i + 1,
            "Document": f"{res['source']} (Chunk {res.get('chunk_index', '?')})",
            "Rerank Score": res.get('rerank_score', 0),
            "Content Preview": res['content'][:50] + "..."
        })
    
    if data:
        df = pd.DataFrame(data)
        st.dataframe(
            df,
            column_config={
                "Rerank Score": st.column_config.ProgressColumn(
                    "Relevance Model Score",
                    help="Cross-Encoder confidence",
                    format="%.4f",
                    min_value=-10,
                    max_value=10,
                ),
            },
            hide_index=True,
            use_container_width=True
        )
