"""
Citation Utilities

Provides citation highlighting and expandable citation cards for evidence traceability.

Features:
- Inline citation highlighting ([PMID:12345678], [1], [L1], etc.)
- Expandable citation cards with abstract snippets
- Source type badges (Literature, Web, Clinical Trial)
"""

import re
import streamlit as st
from typing import List, Dict, Optional


# =============================================================================
# CITATION HIGHLIGHTING
# =============================================================================

def highlight_inline_citations(content: str) -> str:
    """
    Parse and highlight citation markers as purple badges.

    Supports:
    - [PMID:12345678] or [PMID: 12345678]
    - [1], [2], [1-3], [1,2,3]
    - [L1] (Literature), [W1] (Web), [C1] (Clinical Trial), [T1] (Trial)

    Args:
        content: Text containing citation markers

    Returns:
        HTML string with highlighted citations
    """
    # Purple gradient badge style
    citation_style = (
        "background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%); "
        "color: white; padding: 1px 6px; border-radius: 4px; font-size: 11px; "
        "font-weight: 600; margin: 0 1px; cursor: help; display: inline-block;"
    )

    # Pattern matches:
    # 1. [PMID:12345678] or [PMID: 12345678]
    # 2. [1], [2], [1-3], [1,2,3], [1, 2]
    # 3. [L1], [W1], [C1], [T1] (prefixed citations)
    citation_pattern = r'\[(?:PMID[:\s]*(\d{7,8})|([LWCT]?\d+(?:[,\s-]+[LWCT]?\d+)*))\]'

    def replace_citation(match):
        pmid = match.group(1)
        ref_nums = match.group(2)

        if pmid:
            # PMID citation - link to PubMed
            return (
                f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank" '
                f'style="{citation_style} text-decoration: none;" '
                f'title="View PMID {pmid} on PubMed">[PMID:{pmid}]</a>'
            )
        else:
            # Reference number(s)
            return (
                f'<span style="{citation_style}" '
                f'title="Citation {ref_nums}">[{ref_nums}]</span>'
            )

    return re.sub(citation_pattern, replace_citation, content)


def highlight_epistemic_tags(content: str) -> str:
    """
    Highlight epistemic tags with appropriate colors.

    Tags:
    - EVIDENCE (PMID: XXXXX) - Green
    - ASSUMPTION: - Yellow
    - OPINION: - Blue
    - EVIDENCE GAP - Red
    """
    # EVIDENCE tag - green
    content = re.sub(
        r'(EVIDENCE\s*\([^)]+\))',
        r'<span style="background: #D1FAE5; color: #065F46; padding: 2px 6px; '
        r'border-radius: 3px; font-weight: 600;">\1</span>',
        content
    )

    # ASSUMPTION tag - yellow
    content = re.sub(
        r'(ASSUMPTION:)',
        r'<span style="background: #FEF3C7; color: #92400E; padding: 2px 6px; '
        r'border-radius: 3px; font-weight: 600;">\1</span>',
        content
    )

    # OPINION tag - blue
    content = re.sub(
        r'(OPINION:)',
        r'<span style="background: #DBEAFE; color: #1E40AF; padding: 2px 6px; '
        r'border-radius: 3px; font-weight: 600;">\1</span>',
        content
    )

    # EVIDENCE GAP tag - red
    content = re.sub(
        r'(EVIDENCE GAP[^\n]*)',
        r'<span style="background: #FEE2E2; color: #991B1B; padding: 2px 6px; '
        r'border-radius: 3px; font-weight: 600;">\1</span>',
        content
    )

    return content


def format_expert_response(content: str, enable_highlighting: bool = True) -> str:
    """
    Format expert response with citation and epistemic tag highlighting.

    Args:
        content: Raw expert response text
        enable_highlighting: Whether to apply highlighting (default True)

    Returns:
        Formatted HTML string
    """
    if not enable_highlighting:
        return content

    # Apply both highlighting functions
    formatted = highlight_inline_citations(content)
    formatted = highlight_epistemic_tags(formatted)

    return formatted


# =============================================================================
# CITATION CARDS
# =============================================================================

def render_citation_cards(
    citations: List[Dict],
    expanded: bool = False,
    max_display: int = 10,
    key_prefix: str = "citation"
) -> None:
    """
    Render expandable citation cards for evidence traceability.

    Args:
        citations: List of citation dicts with pmid, title, abstract, etc.
        expanded: Whether cards should be expanded by default
        max_display: Maximum number of citations to display
        key_prefix: Prefix for unique Streamlit keys
    """
    if not citations:
        st.info("No citations available.")
        return

    # Source type colors
    type_colors = {
        'literature': ('#6366F1', '#EEF2FF'),  # Purple/light purple
        'web': ('#F59E0B', '#FEF3C7'),         # Amber/light amber
        'clinical_trial': ('#10B981', '#D1FAE5'),  # Green/light green
        'preprint': ('#8B5CF6', '#EDE9FE'),    # Violet/light violet
    }

    for idx, citation in enumerate(citations[:max_display]):
        pmid = citation.get('pmid', citation.get('id', ''))
        title = citation.get('title', 'Unknown Source')
        abstract = citation.get('abstract', '')
        authors = citation.get('authors', [])
        year = citation.get('year', citation.get('pub_date', ''))
        source_type = citation.get('source_type', 'literature').lower()
        journal = citation.get('journal', '')

        # Get colors for this source type
        badge_color, bg_color = type_colors.get(source_type, type_colors['literature'])

        # Format authors
        if isinstance(authors, list) and authors:
            author_str = ', '.join(authors[:3])
            if len(authors) > 3:
                author_str += f" et al."
        else:
            author_str = str(authors) if authors else "Unknown authors"

        # Build the expander label
        if pmid:
            label = f"[PMID: {pmid}] {title[:60]}..."
        else:
            label = f"{title[:70]}..."

        with st.expander(label, expanded=expanded):
            # Title with link
            st.markdown(f"**{title}**")

            # Metadata row
            meta_parts = []
            if author_str:
                meta_parts.append(f"{author_str}")
            if year:
                meta_parts.append(f"({year})")
            if journal:
                meta_parts.append(f"*{journal}*")

            if meta_parts:
                st.caption(' '.join(meta_parts))

            # Source type badge
            st.markdown(
                f'<span style="background: {badge_color}; color: white; '
                f'padding: 2px 8px; border-radius: 4px; font-size: 11px; '
                f'font-weight: 600; text-transform: uppercase;">'
                f'{source_type.replace("_", " ")}</span>',
                unsafe_allow_html=True
            )

            # Abstract snippet
            if abstract:
                abstract_display = abstract[:500] + "..." if len(abstract) > 500 else abstract
                st.markdown(
                    f'<div style="background: {bg_color}; border-left: 3px solid {badge_color}; '
                    f'padding: 10px 12px; margin: 8px 0; font-size: 13px; '
                    f'line-height: 1.5; color: #374151;">{abstract_display}</div>',
                    unsafe_allow_html=True
                )

            # Links
            col1, col2 = st.columns(2)
            if pmid:
                with col1:
                    st.markdown(f"[View on PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")

            # DOI link if available
            doi = citation.get('doi', '')
            if doi:
                with col2:
                    st.markdown(f"[View DOI](https://doi.org/{doi})")

    # Show count if truncated
    if len(citations) > max_display:
        st.caption(f"Showing {max_display} of {len(citations)} citations")


def render_inline_citation_list(
    citations: List[Dict],
    compact: bool = True
) -> str:
    """
    Generate an inline HTML list of citations for embedding in text.

    Args:
        citations: List of citation dicts
        compact: If True, show only PMIDs; if False, show titles too

    Returns:
        HTML string of citation badges
    """
    if not citations:
        return ""

    badge_style = (
        "background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%); "
        "color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; "
        "font-weight: 600; margin: 2px; display: inline-block; text-decoration: none;"
    )

    badges = []
    for cit in citations[:10]:  # Limit to 10
        pmid = cit.get('pmid', '')
        title = cit.get('title', 'Source')

        if pmid:
            badge = (
                f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank" '
                f'style="{badge_style}" title="{title}">[PMID:{pmid}]</a>'
            )
        else:
            short_title = title[:30] + "..." if len(title) > 30 else title
            badge = f'<span style="{badge_style}" title="{title}">{short_title}</span>'

        badges.append(badge)

    return ' '.join(badges)


# =============================================================================
# EXPERT RESPONSE RENDERING
# =============================================================================

def render_expert_response_with_citations(
    expert_name: str,
    content: str,
    citations: Optional[List[Dict]] = None,
    show_citation_cards: bool = True,
    key_prefix: str = "expert"
) -> None:
    """
    Render a single expert response with highlighted citations and optional cards.

    Args:
        expert_name: Name of the expert
        content: Response content
        citations: Optional list of citations referenced
        show_citation_cards: Whether to show expandable citation cards
        key_prefix: Prefix for unique keys
    """
    # Expert name header
    st.markdown(f"**{expert_name}**")

    # Format and display content with highlighting
    formatted_content = format_expert_response(content)
    st.markdown(formatted_content, unsafe_allow_html=True)

    # Citation cards if provided
    if show_citation_cards and citations:
        with st.expander(f"View {len(citations)} cited sources", expanded=False):
            render_citation_cards(
                citations,
                expanded=False,
                max_display=5,
                key_prefix=f"{key_prefix}_{expert_name}"
            )

    st.markdown("---")
