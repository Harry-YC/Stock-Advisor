"""
Results Display Module

Search results display with sorting and filtering.
"""

import streamlit as st
from typing import Dict, Any, List, Set

from core.citation_utils import get_attr


def render_results_section(
    results: Dict[str, Any],
    selected_papers: Set[str],
    on_selection_change: callable = None
):
    """
    Render the search results section.

    Args:
        results: Search results dictionary
        selected_papers: Set of selected PMID strings
        on_selection_change: Callback when selection changes
    """
    if not results:
        return

    st.markdown("---")
    st.subheader("Search Results")

    # Query analysis section
    _render_query_analysis(results)

    st.markdown("---")

    # Results metrics
    _render_results_metrics(results, selected_papers)

    # Selection controls
    _render_selection_controls(results, selected_papers, on_selection_change)

    # Citation list
    if results.get('scored_citations'):
        _render_scored_citations(results, selected_papers, on_selection_change)
    elif results.get('citations'):
        _render_basic_citations(results, selected_papers, on_selection_change)


def _render_query_analysis(results: Dict[str, Any]):
    """Render the query analysis section."""
    with st.container():
        st.markdown("### 🔍 Query Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            query_type_display = results.get("query_type", "DIRECT")
            if results.get("is_cached"):
                query_type_display += " (Cached)"
            st.metric("Query Type", query_type_display)

        with col2:
            st.metric("Confidence", results.get("query_confidence", "high"))

        with col3:
            if results.get("used_fallback"):
                st.metric("Status", "Fallback Used", delta="⚠️")
            else:
                st.metric("Status", "Success", delta="✓")

        # Original query
        if results.get("query"):
            st.markdown("**Original Query:**")
            st.code(results["query"], language="text")

            # Optimized query
            if results.get("optimized_query") and results["optimized_query"] != results["query"]:
                st.markdown("**Optimized Query:**")
                st.code(results["optimized_query"], language="text")
                if results.get("query_explanation"):
                    st.caption(f"💡 {results['query_explanation']}")

        # PubMed translation
        if results.get("query_translation"):
            st.markdown("**PubMed Translation:**")
            st.code(results["query_translation"], language="text")

        # Fallback info
        if results.get("used_fallback") and results.get("fallback_from_query"):
            st.warning("ℹ️ AI-optimized query returned 0 results. Used simpler fallback query above.")
            with st.expander("🔍 See original AI-optimized query", expanded=False):
                st.code(results["fallback_from_query"], language="text")

        # Final query with filters
        if results.get("final_query") and results.get("final_query") != results.get("optimized_query"):
            st.markdown("**Final Query (with filters):**")
            st.code(results["final_query"], language="text")

        # Filter info
        filter_info = []
        if results.get("applied_filters"):
            filter_info.extend([f"🏥 {f}" for f in results["applied_filters"]])
        if results.get("ranking_mode"):
            filter_info.append(f"📊 Ranking: {results['ranking_mode']}")
        if filter_info:
            st.caption(" · ".join(filter_info))


def _render_results_metrics(results: Dict[str, Any], selected_papers: Set[str]):
    """Render the results metrics row."""
    col1, col2, col3 = st.columns(3)

    total_count = results.get('total_count', 0)
    retrieved_count = results.get('retrieved_count', len(results.get('citations', [])))

    col1.metric("Total Matches", f"{total_count:,}")
    col2.metric("Retrieved", retrieved_count)
    col3.metric("Selected", len(selected_papers))


def _render_selection_controls(
    results: Dict[str, Any],
    selected_papers: Set[str],
    on_selection_change: callable
):
    """Render selection control buttons."""
    sel_col1, sel_col2, sel_col3 = st.columns([1, 1, 2])

    with sel_col1:
        if st.button("☑️ Select All", use_container_width=True):
            if results.get('scored_citations'):
                all_pmids = {sc.citation.get('pmid') for sc in results['scored_citations']}
            elif results.get('citations'):
                all_pmids = {get_attr(c, 'pmid') for c in results['citations']}
            else:
                all_pmids = set()

            st.session_state.selected_papers = all_pmids
            if on_selection_change:
                on_selection_change()
            st.rerun()

    with sel_col2:
        if st.button("☐ Deselect All", use_container_width=True):
            st.session_state.selected_papers = set()
            if on_selection_change:
                on_selection_change()
            st.rerun()

    with sel_col3:
        retrieved = results.get('retrieved_count', len(results.get('citations', [])))
        st.caption(f"💡 {len(selected_papers)} of {retrieved} selected")


def _render_scored_citations(
    results: Dict[str, Any],
    selected_papers: Set[str],
    on_selection_change: callable
):
    """Render scored citations with sorting options."""
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.caption("💡 Clinical Utility Score combines evidence quality, relevance, and recency")

    with col_right:
        sort_option = st.selectbox(
            "🔀 Sort by",
            [
                "Clinical Utility Score (Evidence + Relevance)",
                "Relevance (PubMed Best Match)",
                "Publication Year (Newest First)",
                "Publication Year (Oldest First)",
                "Evidence Quality (RCTs/SRs First)"
            ],
            index=0,
            key="sort_option"
        )

    # Sort citations
    scored_citations = results['scored_citations'].copy()

    if sort_option == "Relevance (PubMed Best Match)":
        scored_citations.sort(key=lambda x: x.rank_position)
    elif sort_option == "Publication Year (Newest First)":
        scored_citations.sort(
            key=lambda x: int(x.citation.get('year', '0') or '0'),
            reverse=True
        )
    elif sort_option == "Publication Year (Oldest First)":
        scored_citations.sort(
            key=lambda x: int(x.citation.get('year', '0') or '0')
        )
    elif sort_option == "Evidence Quality (RCTs/SRs First)":
        scored_citations.sort(key=lambda x: x.evidence_score, reverse=True)

    st.markdown("### 📚 Citations")
    st.caption(f"Sorted by: {sort_option}")

    # Render each citation
    for idx, scored_cit in enumerate(scored_citations, 1):
        c = scored_cit.citation
        pmid = c['pmid']
        is_selected = pmid in selected_papers

        col1, col2 = st.columns([0.05, 0.95])

        with col1:
            st.write("")  # Vertical alignment
            checkbox_icon = "✅" if is_selected else "☐"
            if st.button(checkbox_icon, key=f"select_{pmid}_{idx}", use_container_width=True):
                if is_selected:
                    st.session_state.selected_papers.discard(pmid)
                else:
                    st.session_state.selected_papers.add(pmid)
                if on_selection_change:
                    on_selection_change()
                st.rerun()

        with col2:
            from ui.literature_search.citation_card import render_scored_citation_card
            render_scored_citation_card(scored_cit, idx)


def _render_basic_citations(
    results: Dict[str, Any],
    selected_papers: Set[str],
    on_selection_change: callable
):
    """Render basic citations without scoring."""
    st.markdown("### Citations")
    st.caption("Click on any citation to view full details")

    citations = results['citations']

    for idx, c in enumerate(citations, 1):
        pmid = get_attr(c, 'pmid')
        is_selected = pmid in selected_papers

        col1, col2 = st.columns([0.05, 0.95])

        with col1:
            st.write("")
            checkbox_icon = "✅" if is_selected else "☐"
            if st.button(checkbox_icon, key=f"select_{pmid}_{idx}", use_container_width=True):
                if is_selected:
                    st.session_state.selected_papers.discard(pmid)
                else:
                    st.session_state.selected_papers.add(pmid)
                if on_selection_change:
                    on_selection_change()
                st.rerun()

        with col2:
            from ui.literature_search.citation_card import render_citation_card
            render_citation_card(c, idx)

    # Manual PMID addition
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        pmid_input = st.text_input(
            "Add PMID to selection",
            placeholder="Enter PMID and press Add"
        )

    with col2:
        if st.button("➕ Add"):
            if pmid_input and pmid_input not in selected_papers:
                st.session_state.selected_papers.add(pmid_input)
                st.success(f"✓ Added {pmid_input}")
                st.rerun()

    with col3:
        if st.button("🗑️ Clear All"):
            st.session_state.selected_papers = set()
            st.success("✓ Cleared selection")
            st.rerun()
