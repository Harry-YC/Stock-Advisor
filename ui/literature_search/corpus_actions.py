"""
Corpus Actions Module

Evidence corpus include/exclude actions for literature search.
"""

import streamlit as st
from typing import Optional

# Evidence corpus imports
try:
    from core.evidence_corpus import get_corpus_from_session, EvidenceCorpus
    EVIDENCE_CORPUS_AVAILABLE = True
except ImportError:
    EVIDENCE_CORPUS_AVAILABLE = False


def render_corpus_status_badge(pmid: str, corpus: Optional['EvidenceCorpus'] = None) -> str:
    """
    Render inclusion/exclusion status badge for a citation.

    Args:
        pmid: The PMID to check
        corpus: EvidenceCorpus object (if None, gets from session)

    Returns:
        HTML string for the status badge, or empty string if not available
    """
    if not EVIDENCE_CORPUS_AVAILABLE:
        return ""

    if corpus is None:
        corpus = get_corpus_from_session()

    if corpus is None:
        return ""

    if pmid in corpus.excluded_pmids:
        reason = corpus.excluded_pmids.get(pmid, "No reason provided")
        return f"""<span style="background: #dc3545; color: white; padding: 2px 8px;
                   border-radius: 4px; font-size: 0.75em; margin-left: 8px;"
                   title="{reason}">❌ Excluded</span>"""

    elif pmid in corpus.included_pmids:
        return """<span style="background: #28a745; color: white; padding: 2px 8px;
                   border-radius: 4px; font-size: 0.75em; margin-left: 8px;">✅ Included</span>"""

    else:
        return """<span style="background: #6c757d; color: white; padding: 2px 8px;
                   border-radius: 4px; font-size: 0.75em; margin-left: 8px;">⏳ Not Screened</span>"""


def render_corpus_action_buttons(pmid: str, idx: int, corpus: Optional['EvidenceCorpus'] = None):
    """
    Render include/exclude buttons for corpus management.

    Args:
        pmid: The PMID to manage
        idx: Index for unique key generation
        corpus: EvidenceCorpus object (if None, gets from session)
    """
    if not EVIDENCE_CORPUS_AVAILABLE:
        return

    if corpus is None:
        corpus = get_corpus_from_session()

    if corpus is None:
        return

    col_inc, col_exc = st.columns(2)

    with col_inc:
        is_included = pmid in corpus.included_pmids
        btn_label = "✅ Included" if is_included else "Include"
        btn_type = "primary" if not is_included else "secondary"

        if st.button(btn_label, key=f"include_{pmid}_{idx}", type=btn_type, use_container_width=True):
            if not is_included:
                corpus.include(pmid, reason="Manual inclusion from search")
                st.session_state.evidence_corpus = corpus
                st.success(f"Included PMID {pmid}")
                st.rerun()

    with col_exc:
        is_excluded = pmid in corpus.excluded_pmids
        btn_label = "❌ Excluded" if is_excluded else "Exclude"

        if st.button(btn_label, key=f"exclude_{pmid}_{idx}", use_container_width=True):
            if not is_excluded:
                # Show reason input
                reason = st.session_state.get(f"exclude_reason_{pmid}", "")
                if reason:
                    corpus.exclude(pmid, reason=reason)
                    st.session_state.evidence_corpus = corpus
                    st.success(f"Excluded PMID {pmid}")
                    st.rerun()
                else:
                    st.session_state[f"show_exclude_reason_{pmid}"] = True
                    st.rerun()

    # Show reason input if needed
    if st.session_state.get(f"show_exclude_reason_{pmid}"):
        reason = st.text_input(
            "Exclusion reason",
            key=f"exclude_reason_input_{pmid}_{idx}",
            placeholder="e.g., Wrong population, Not relevant"
        )
        if st.button("Confirm Exclusion", key=f"confirm_exclude_{pmid}_{idx}"):
            if reason:
                corpus.exclude(pmid, reason=reason)
                st.session_state.evidence_corpus = corpus
                del st.session_state[f"show_exclude_reason_{pmid}"]
                st.success(f"Excluded PMID {pmid}: {reason}")
                st.rerun()
            else:
                st.warning("Please provide a reason for exclusion")


def bulk_include_papers(pmids: list, reason: str = "Bulk inclusion"):
    """
    Include multiple papers in the corpus at once.

    Args:
        pmids: List of PMIDs to include
        reason: Reason for inclusion
    """
    if not EVIDENCE_CORPUS_AVAILABLE:
        return

    corpus = get_corpus_from_session()
    if corpus is None:
        return

    for pmid in pmids:
        if pmid not in corpus.included_pmids:
            corpus.include(pmid, reason=reason)

    st.session_state.evidence_corpus = corpus


def bulk_exclude_papers(pmids: list, reason: str):
    """
    Exclude multiple papers from the corpus at once.

    Args:
        pmids: List of PMIDs to exclude
        reason: Reason for exclusion
    """
    if not EVIDENCE_CORPUS_AVAILABLE:
        return

    corpus = get_corpus_from_session()
    if corpus is None:
        return

    for pmid in pmids:
        if pmid not in corpus.excluded_pmids:
            corpus.exclude(pmid, reason=reason)

    st.session_state.evidence_corpus = corpus


def get_corpus_summary() -> dict:
    """
    Get a summary of the current corpus state.

    Returns:
        Dictionary with included_count, excluded_count, unscreened_count
    """
    if not EVIDENCE_CORPUS_AVAILABLE:
        return {'included_count': 0, 'excluded_count': 0, 'unscreened_count': 0}

    corpus = get_corpus_from_session()
    if corpus is None:
        return {'included_count': 0, 'excluded_count': 0, 'unscreened_count': 0}

    return {
        'included_count': len(corpus.included_pmids),
        'excluded_count': len(corpus.excluded_pmids),
    }
