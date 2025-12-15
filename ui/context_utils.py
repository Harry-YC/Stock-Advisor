"""
Context Utilities Module

Shared utilities for checking and displaying context availability.
Deduplicates context checking code from app_lr.py and UI modules.
"""

import streamlit as st
from typing import Tuple, List, Dict


def get_available_context() -> Tuple[List, List, List, Dict]:
    """
    Get all available context from session state.

    Returns:
        Tuple of (papers, indexed_docs, uploaded_docs, discussion)
    """
    papers = []
    if st.session_state.get('search_results'):
        papers = st.session_state.search_results.get('citations', [])

    indexed_docs = st.session_state.get('indexed_documents', [])
    uploaded_docs = st.session_state.get('uploaded_documents', [])
    discussion = st.session_state.get('expert_discussion', {})

    return papers, indexed_docs, uploaded_docs, discussion


def get_context_summary() -> Dict[str, int]:
    """
    Get summary counts of available context.

    Returns:
        Dict with 'papers', 'documents', 'discussion_rounds' keys
    """
    papers, indexed, uploaded, discussion = get_available_context()
    return {
        'papers': len(papers),
        'documents': len(indexed) + len(uploaded),
        'discussion_rounds': len(discussion)
    }


def has_context() -> bool:
    """
    Check if any context is available.

    Returns:
        True if any papers, documents, or discussions exist
    """
    summary = get_context_summary()
    return any(v > 0 for v in summary.values())


def render_context_indicator():
    """
    Render the context indicator bar.

    Shows paper count, document count, and discussion rounds.
    """
    summary = get_context_summary()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"📄 {summary['papers']} papers loaded")
    with col2:
        st.caption(f"📁 {summary['documents']} documents")
    with col3:
        st.caption(f"💬 {summary['discussion_rounds']} discussion rounds")


def render_context_check(tab_name: str, show_toggle: bool = True) -> bool:
    """
    Render context availability check with optional web search toggle.

    Args:
        tab_name: Name of the tab (for messaging)
        show_toggle: Whether to show web search toggle when no context

    Returns:
        True if context is available
    """
    papers, indexed, uploaded, discussion = get_available_context()
    has_ctx = bool(papers or indexed or uploaded or discussion)

    if not has_ctx:
        st.info(f"💡 **No papers or documents loaded.** {tab_name} will use web search and LLM knowledge to answer questions.")
        if show_toggle:
            toggle_key = f"{tab_name.lower().replace(' ', '_')}_web_search_toggle"
            st.checkbox(
                "Enable web search for real-time information",
                value=True,
                key=toggle_key
            )
        return False
    else:
        context_parts = []
        if papers:
            context_parts.append(f"{len(papers)} papers")
        if indexed or uploaded:
            context_parts.append(f"{len(indexed) + len(uploaded)} documents")
        if discussion:
            context_parts.append(f"{len(discussion)} discussion rounds")
        st.success(f"✅ **Context available:** {', '.join(context_parts)}")
        return True


def format_context_status() -> str:
    """
    Format context status as a string.

    Returns:
        Formatted string describing available context
    """
    papers, indexed, uploaded, discussion = get_available_context()

    if not (papers or indexed or uploaded or discussion):
        return "No context loaded"

    parts = []
    if papers:
        parts.append(f"{len(papers)} papers")
    if indexed or uploaded:
        parts.append(f"{len(indexed) + len(uploaded)} documents")
    if discussion:
        parts.append(f"{len(discussion)} discussion rounds")

    return ", ".join(parts)
