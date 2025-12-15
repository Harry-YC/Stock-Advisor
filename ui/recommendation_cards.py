"""
Recommendation Cards UI Component

Enhanced recommendation display with:
- Strength badges (Strong FOR / Conditional FOR / etc.)
- Certainty badges (⊕⊕⊕⊕ / ⊕⊕⊕○ / etc.)
- Citation badges with quality indicators
- Warning badges for missing citations or conflicts
- Edit / Lock controls
"""

import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

from services.recommendation_service import Recommendation
from core.evidence_corpus import EvidenceCorpus, get_corpus_from_session
from core.quality_assessment import QualityRating, format_certainty_badge


def render_recommendation_card(
    recommendation: Recommendation,
    corpus: EvidenceCorpus = None,
    editable: bool = False,
    show_warnings: bool = True
):
    """
    Render an enhanced recommendation card.

    Args:
        recommendation: Recommendation object
        corpus: EvidenceCorpus for validation
        editable: Whether to show edit controls
        show_warnings: Whether to show warning badges
    """
    if corpus is None:
        corpus = get_corpus_from_session()

    # Collect warnings
    warnings = _collect_warnings(recommendation, corpus) if show_warnings else []

    # Card container
    st.markdown("""
    <style>
    .rec-card {
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        # Header row with badges
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(f"### {recommendation.statement}")

        with col2:
            _render_strength_badge(recommendation.strength)

        with col3:
            _render_certainty_badge(recommendation.evidence_quality)

        # Warning badges
        if warnings:
            st.markdown("")
            for warning in warnings:
                _render_warning_badge(warning)

        st.markdown("---")

        # Two-column layout for details
        col_left, col_right = st.columns(2)

        with col_left:
            # Population
            if recommendation.population:
                st.markdown(f"**Population:** {recommendation.population}")

            # Benefits
            if recommendation.benefits:
                st.markdown("**Benefits:**")
                for b in recommendation.benefits[:5]:
                    st.markdown(f"- {b}")

            # Rationale
            if recommendation.rationale:
                with st.expander("Rationale", expanded=False):
                    st.markdown(recommendation.rationale)

        with col_right:
            # Intervention/Comparator
            if recommendation.intervention:
                st.markdown(f"**Intervention:** {recommendation.intervention}")
            if recommendation.comparator:
                st.markdown(f"**Comparator:** {recommendation.comparator}")

            # Harms
            if recommendation.harms:
                st.markdown("**Harms/Burdens:**")
                for h in recommendation.harms[:5]:
                    st.markdown(f"- {h}")

        # Citation badges
        if recommendation.key_citations:
            st.markdown("---")
            st.markdown("**Supporting Evidence:**")
            _render_citation_badges(recommendation.key_citations, corpus)

        # Research gaps
        if recommendation.research_gaps:
            with st.expander("Research Gaps", expanded=False):
                for gap in recommendation.research_gaps:
                    st.markdown(f"- {gap}")

        # Implementation notes
        if recommendation.implementation_notes:
            with st.expander("Implementation Considerations", expanded=False):
                st.markdown(recommendation.implementation_notes)

        # Edit/Lock controls
        if editable:
            st.markdown("---")
            _render_edit_controls(recommendation)


def _render_strength_badge(strength: str):
    """Render the recommendation strength badge."""
    colors = {
        "Strong FOR": ("#28a745", "white", "++"),
        "Conditional FOR": ("#17a2b8", "white", "+?"),
        "Conditional AGAINST": ("#fd7e14", "white", "-?"),
        "Strong AGAINST": ("#dc3545", "white", "--")
    }

    bg_color, text_color, icon = colors.get(strength, ("#6c757d", "white", "?"))

    st.markdown(f"""
    <div style="background: {bg_color}; color: {text_color}; padding: 8px 16px;
                border-radius: 8px; text-align: center; font-weight: bold;">
        <div style="font-size: 1.5rem;">{icon}</div>
        <div style="font-size: 0.8rem;">{strength}</div>
    </div>
    """, unsafe_allow_html=True)


def _render_certainty_badge(certainty: str):
    """Render the evidence certainty badge."""
    symbols = {
        "High": ("⊕⊕⊕⊕", "#28a745"),
        "Moderate": ("⊕⊕⊕○", "#17a2b8"),
        "Low": ("⊕⊕○○", "#ffc107"),
        "Very Low": ("⊕○○○", "#dc3545")
    }

    symbol, color = symbols.get(certainty, ("○○○○", "#6c757d"))

    st.markdown(f"""
    <div style="background: {color}; color: white; padding: 8px 16px;
                border-radius: 8px; text-align: center;">
        <div style="font-size: 1.2rem;">{symbol}</div>
        <div style="font-size: 0.8rem;">{certainty}</div>
    </div>
    """, unsafe_allow_html=True)


def _render_warning_badge(warning: Dict):
    """Render a warning badge."""
    severity_colors = {
        "error": "#dc3545",
        "warning": "#ffc107",
        "info": "#17a2b8"
    }

    severity_icons = {
        "error": "🔴",
        "warning": "🟡",
        "info": "🔵"
    }

    color = severity_colors.get(warning.get("severity", "warning"), "#ffc107")
    icon = severity_icons.get(warning.get("severity", "warning"), "⚠️")

    st.markdown(f"""
    <div style="background: {color}20; border-left: 4px solid {color};
                padding: 8px 12px; margin: 4px 0; border-radius: 4px;">
        {icon} <strong>{warning.get('type', 'Warning')}:</strong> {warning.get('message', '')}
    </div>
    """, unsafe_allow_html=True)


def _render_citation_badges(pmids: List[str], corpus: EvidenceCorpus = None):
    """Render citation badges with quality indicators."""
    cols = st.columns(min(len(pmids), 4))

    for i, pmid in enumerate(pmids[:8]):  # Max 8 citations
        col_idx = i % 4
        with cols[col_idx]:
            # Check if PMID is valid
            is_valid = corpus.can_cite(pmid) if corpus else True

            # Get quality rating if available
            quality = None
            if corpus and pmid in corpus.quality_ratings:
                quality = corpus.quality_ratings[pmid]

            # Determine badge color
            if not is_valid:
                bg_color = "#dc3545"  # Red for invalid
                tooltip = "Not in evidence corpus!"
            elif quality:
                certainty = quality.get('certainty', '') if isinstance(quality, dict) else getattr(quality, 'certainty', '')
                quality_colors = {
                    "High": "#28a745",
                    "Moderate": "#17a2b8",
                    "Low": "#ffc107",
                    "Very Low": "#fd7e14"
                }
                bg_color = quality_colors.get(certainty, "#6c757d")
                tooltip = f"Certainty: {certainty}"
            else:
                bg_color = "#6c757d"  # Gray for no quality rating
                tooltip = "No quality rating"

            st.markdown(f"""
            <a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}" target="_blank"
               title="{tooltip}"
               style="display: inline-block; background: {bg_color}; color: white;
                      padding: 4px 10px; border-radius: 4px; margin: 2px;
                      text-decoration: none; font-size: 0.85rem;">
                PMID: {pmid}
            </a>
            """, unsafe_allow_html=True)


def _collect_warnings(recommendation: Recommendation, corpus: EvidenceCorpus = None) -> List[Dict]:
    """Collect warnings about the recommendation."""
    warnings = []

    # Check for invalid citations
    if corpus and recommendation.key_citations:
        invalid = [p for p in recommendation.key_citations if not corpus.can_cite(p)]
        if invalid:
            warnings.append({
                "type": "Invalid Citations",
                "message": f"PMIDs not in evidence corpus: {', '.join(invalid)}",
                "severity": "error"
            })

    # Check for missing citations
    if not recommendation.key_citations:
        warnings.append({
            "type": "No Citations",
            "message": "Recommendation has no supporting citations",
            "severity": "warning"
        })

    # Check for low certainty
    if recommendation.evidence_quality in ["Low", "Very Low"]:
        warnings.append({
            "type": "Low Certainty",
            "message": f"Evidence certainty is {recommendation.evidence_quality}",
            "severity": "info"
        })

    # Check for conditional recommendation without clear rationale
    if "Conditional" in recommendation.strength and not recommendation.rationale:
        warnings.append({
            "type": "Missing Rationale",
            "message": "Conditional recommendation should include rationale",
            "severity": "warning"
        })

    return warnings


def _render_edit_controls(recommendation: Recommendation):
    """Render edit/lock controls for recommendation."""
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Edit", key=f"edit_rec_{id(recommendation)}"):
            st.session_state[f"editing_rec_{id(recommendation)}"] = True
            st.rerun()

    with col2:
        if st.button("Lock", key=f"lock_rec_{id(recommendation)}", type="primary"):
            st.success("Recommendation locked!")


def render_recommendation_list(
    recommendations: List[Recommendation],
    corpus: EvidenceCorpus = None
):
    """
    Render a list of recommendations.

    Args:
        recommendations: List of Recommendation objects
        corpus: EvidenceCorpus for validation
    """
    if not recommendations:
        st.info("No recommendations generated yet.")
        return

    st.markdown(f"## Recommendations ({len(recommendations)})")

    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"{i}. {rec.statement[:60]}...", expanded=i == 1):
            render_recommendation_card(rec, corpus, editable=False, show_warnings=True)


def render_recommendation_summary_table(recommendations: List[Recommendation]):
    """
    Render recommendations as a summary table.

    Args:
        recommendations: List of Recommendation objects
    """
    if not recommendations:
        return

    st.markdown("### Recommendation Summary")

    # Build table data
    data = []
    for rec in recommendations:
        data.append({
            "Recommendation": rec.statement[:80] + "..." if len(rec.statement) > 80 else rec.statement,
            "Strength": rec.strength,
            "Certainty": rec.evidence_quality,
            "Citations": len(rec.key_citations)
        })

    # Display as dataframe
    import pandas as pd
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_grade_summary_table(recommendations: List[Recommendation]):
    """
    Render GRADE-style summary of findings table.

    Args:
        recommendations: List of Recommendation objects
    """
    st.markdown("### GRADE Summary of Findings")

    st.markdown("""
    | Outcome | Effect | Certainty | Comments |
    |---------|--------|-----------|----------|
    """)

    for rec in recommendations[:5]:
        certainty_symbol = {
            "High": "⊕⊕⊕⊕",
            "Moderate": "⊕⊕⊕○",
            "Low": "⊕⊕○○",
            "Very Low": "⊕○○○"
        }.get(rec.evidence_quality, "○○○○")

        outcomes = ", ".join(rec.outcomes[:3]) if rec.outcomes else "Multiple outcomes"
        effect = rec.rationale[:50] + "..." if rec.rationale else "See details"

        st.markdown(f"| {outcomes} | {effect} | {certainty_symbol} {rec.evidence_quality} | {rec.strength} |")


def export_recommendations_markdown(recommendations: List[Recommendation]) -> str:
    """Export recommendations to markdown format."""
    lines = [
        "# Clinical Recommendations",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d')}*",
        "",
        "---",
        ""
    ]

    for i, rec in enumerate(recommendations, 1):
        lines.append(f"## Recommendation {i}")
        lines.append("")
        lines.append(f"**{rec.statement}**")
        lines.append("")
        lines.append(f"| Strength | Evidence Certainty |")
        lines.append(f"|----------|-------------------|")
        lines.append(f"| {rec.strength} | {rec.evidence_quality} |")
        lines.append("")

        if rec.population:
            lines.append(f"**Population:** {rec.population}")
        if rec.benefits:
            lines.append("**Benefits:**")
            for b in rec.benefits:
                lines.append(f"- {b}")
        if rec.harms:
            lines.append("**Harms:**")
            for h in rec.harms:
                lines.append(f"- {h}")
        if rec.rationale:
            lines.append(f"**Rationale:** {rec.rationale}")
        if rec.key_citations:
            lines.append(f"**Key Citations:** {', '.join(rec.key_citations)}")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
