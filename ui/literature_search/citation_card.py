"""
Citation Card Module

Reusable citation display components.
"""

import streamlit as st
from typing import Any, Optional, Dict

from core.citation_utils import get_attr, format_authors


def render_citation_card(
    citation: Any,
    idx: int,
    show_abstract: bool = True,
    show_corpus_status: bool = True,
    show_corpus_actions: bool = True,
    expanded: bool = False
):
    """
    Render a single citation card.

    Args:
        citation: Citation object or dict
        idx: Index for unique key generation
        show_abstract: Whether to show the abstract
        show_corpus_status: Whether to show corpus inclusion status
        show_corpus_actions: Whether to show include/exclude buttons
        expanded: Whether the expander is initially expanded
    """
    pmid = get_attr(citation, 'pmid', '')
    title = get_attr(citation, 'title', 'No title')
    authors = get_attr(citation, 'authors', [])
    journal = get_attr(citation, 'journal', 'Unknown Journal')
    year = get_attr(citation, 'year', 'N/A')
    abstract = get_attr(citation, 'abstract', '')
    doi = get_attr(citation, 'doi', '')

    # Ensure title is a string
    if not isinstance(title, str):
        title = str(title) if title else 'No title'

    # Build card title
    card_title = f"{idx}. {title[:100]}..." if len(title) > 100 else f"{idx}. {title}"

    with st.expander(card_title, expanded=expanded):
        # Corpus status badge
        if show_corpus_status:
            from ui.literature_search.corpus_actions import render_corpus_status_badge
            corpus_badge = render_corpus_status_badge(pmid)
            if corpus_badge:
                st.markdown(corpus_badge, unsafe_allow_html=True)

        # Full title with link
        title_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        st.markdown(f"**Full Title:** [{title}]({title_link})")

        # Authors
        if authors:
            authors_display = format_authors(authors, max_authors=3)
            st.caption(f"✍️ {authors_display}")

        # Journal and year
        st.caption(f"📖 *{journal}* · 📅 {year}")

        # Abstract
        if show_abstract and abstract:
            st.markdown("---")
            abstract_display = f"_{abstract[:500]}..._" if len(abstract) > 500 else abstract
            st.markdown(abstract_display)

        # Corpus actions
        if show_corpus_actions:
            from ui.literature_search.corpus_actions import render_corpus_action_buttons
            st.markdown("---")
            st.markdown("**Evidence Corpus:**")
            render_corpus_action_buttons(pmid, idx)

        # Badges footer
        st.markdown("---")
        badges = _build_badges(pmid, doi)
        st.markdown(" ".join(badges))


def render_scored_citation_card(
    scored_citation: Any,
    idx: int,
    show_abstract: bool = True,
    show_corpus_status: bool = True,
    show_corpus_actions: bool = True,
    expanded: bool = False
):
    """
    Render a citation card with scoring information.

    Args:
        scored_citation: ScoredCitation object with citation and score data
        idx: Index for unique key generation
        show_abstract: Whether to show the abstract
        show_corpus_status: Whether to show corpus inclusion status
        show_corpus_actions: Whether to show include/exclude buttons
        expanded: Whether the expander is initially expanded
    """
    c = scored_citation.citation
    pmid = c.get('pmid', '') if isinstance(c, dict) else get_attr(c, 'pmid', '')
    title = c.get('title', 'No title') if isinstance(c, dict) else get_attr(c, 'title', 'No title')
    authors = c.get('authors', []) if isinstance(c, dict) else get_attr(c, 'authors', [])
    journal = c.get('journal', 'Unknown Journal') if isinstance(c, dict) else get_attr(c, 'journal', 'Unknown Journal')
    year = c.get('year', 'N/A') if isinstance(c, dict) else get_attr(c, 'year', 'N/A')
    abstract = c.get('abstract', '') if isinstance(c, dict) else get_attr(c, 'abstract', '')
    doi = c.get('doi', '') if isinstance(c, dict) else get_attr(c, 'doi', '')

    # Build card title
    card_title = f"#{idx} · {title[:100]}..." if len(title) > 100 else f"#{idx} · {title}"

    with st.expander(card_title, expanded=expanded):
        # Score badge
        score_badge = f"**Score: {scored_citation.final_score:.3f}**"

        # Add corpus status badge
        if show_corpus_status:
            from ui.literature_search.corpus_actions import render_corpus_status_badge
            corpus_badge = render_corpus_status_badge(pmid)
            if corpus_badge:
                st.markdown(f"{score_badge} {corpus_badge}", unsafe_allow_html=True)
            else:
                st.markdown(score_badge)
        else:
            st.markdown(score_badge)

        # Full title with link
        title_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        st.markdown(f"**Full Title:** [{title}]({title_link})")

        # Explanation chips
        if scored_citation.explanation:
            chips_html = " ".join([
                f"<span style='background-color: var(--primary-soft); color: var(--primary); "
                f"padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-right: 4px; "
                f"border: 1px solid var(--primary);'>{chip}</span>"
                for chip in scored_citation.explanation
            ])
            st.markdown(f"<div style='margin: 8px 0;'>{chips_html}</div>", unsafe_allow_html=True)

        # Authors
        if authors:
            authors_display = format_authors(authors, max_authors=3)
            st.caption(f"✍️ {authors_display}")

        # Journal and year
        st.caption(f"📖 *{journal}* · 📅 {year}")

        # Abstract
        if show_abstract and abstract:
            st.markdown("---")
            abstract_display = f"_{abstract[:500]}..._" if len(abstract) > 500 else abstract
            st.markdown(abstract_display)

        # Corpus actions
        if show_corpus_actions:
            from ui.literature_search.corpus_actions import render_corpus_action_buttons
            st.markdown("---")
            st.markdown("**Evidence Corpus:**")
            render_corpus_action_buttons(pmid, idx)

        # Badges footer
        st.markdown("---")
        badges = _build_badges(pmid, doi)
        st.markdown(" ".join(badges))


def _build_badges(pmid: str, doi: str = '') -> list:
    """
    Build badge markdown for PMID and DOI.

    Args:
        pmid: PubMed ID
        doi: DOI (optional)

    Returns:
        List of badge markdown strings
    """
    badges = []

    if pmid:
        badges.append(
            f"[![PMID](https://img.shields.io/badge/PMID-{pmid}-blue)]"
            f"(https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
        )

    if doi:
        doi_clean = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
        badges.append(
            f"[![DOI](https://img.shields.io/badge/DOI-{doi_clean.replace('-', '--')}-orange)]"
            f"(https://doi.org/{doi_clean})"
        )

    return badges
