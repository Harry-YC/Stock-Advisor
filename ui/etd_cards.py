"""
Evidence-to-Decision Card UI Component

Renders the GRADE EtD framework as an interactive card:
- Domain judgments with dropdowns
- AI pre-fill from expert discussion
- Manual editing
- Lock/finalize functionality
- Export to various formats
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Optional, List
import json

from services.etd_service import (
    EtDService,
    EvidenceToDecision,
    DomainJudgment,
    ETD_DOMAINS,
    get_etd_service
)
from core.evidence_corpus import get_corpus_from_session


def render_etd_card(etd: EvidenceToDecision, editable: bool = True):
    """
    Render a complete EtD card.

    Args:
        etd: EvidenceToDecision object
        editable: Whether the card can be edited
    """
    # Header
    st.markdown("## Evidence-to-Decision Framework")

    # Status badge
    status_colors = {
        "draft": "#ffc107",
        "reviewed": "#17a2b8",
        "locked": "#28a745"
    }
    status_color = status_colors.get(etd.status, "#6c757d")
    st.markdown(f"""
    <span style="background: {status_color}; color: white; padding: 4px 12px; border-radius: 4px; font-size: 0.85rem;">
        {etd.status.upper()}
    </span>
    """, unsafe_allow_html=True)

    # PICO header
    with st.expander("PICO Elements", expanded=False):
        if editable and etd.status != "locked":
            etd.population = st.text_input("Population", value=etd.population, key="etd_population")
            etd.intervention = st.text_input("Intervention", value=etd.intervention, key="etd_intervention")
            etd.comparator = st.text_input("Comparator", value=etd.comparator, key="etd_comparator")
        else:
            st.markdown(f"**Population:** {etd.population or 'Not specified'}")
            st.markdown(f"**Intervention:** {etd.intervention or 'Not specified'}")
            st.markdown(f"**Comparator:** {etd.comparator or 'Not specified'}")

    st.markdown("---")

    # Domain judgments in two columns
    col1, col2 = st.columns(2)

    # Left column domains
    left_domains = ["benefits", "harms", "certainty", "values", "balance"]
    with col1:
        for domain_id in left_domains:
            _render_domain_judgment(etd, domain_id, editable)

    # Right column domains
    right_domains = ["resources", "equity", "acceptability", "feasibility"]
    with col2:
        for domain_id in right_domains:
            _render_domain_judgment(etd, domain_id, editable)

    st.markdown("---")

    # Recommendation section
    st.markdown("### Recommendation")

    if editable and etd.status != "locked":
        col_dir, col_str = st.columns(2)
        with col_dir:
            etd.recommendation_direction = st.selectbox(
                "Direction",
                ["For", "Against"],
                index=0 if etd.recommendation_direction == "For" else 1,
                key="etd_direction"
            )
        with col_str:
            etd.recommendation_strength = st.selectbox(
                "Strength",
                ["Strong", "Conditional"],
                index=0 if etd.recommendation_strength == "Strong" else 1,
                key="etd_strength"
            )

        etd.recommendation_statement = st.text_area(
            "Recommendation Statement",
            value=etd.recommendation_statement,
            placeholder="Enter the specific recommendation (without 'We recommend/suggest')",
            key="etd_statement"
        )

        etd.justification = st.text_area(
            "Justification",
            value=etd.justification,
            placeholder="Brief rationale for this recommendation",
            key="etd_justification"
        )
    else:
        # Display recommendation card
        _render_recommendation_badge(etd)

    # Action buttons
    st.markdown("---")
    col_actions = st.columns([1, 1, 1, 2])

    with col_actions[0]:
        if etd.status != "locked":
            if st.button("Save Draft", key="etd_save_draft"):
                etd.status = "draft"
                etd.last_modified = datetime.now().isoformat()
                st.session_state.current_etd = etd
                st.success("Draft saved!")

    with col_actions[1]:
        if etd.status == "draft":
            if st.button("Mark Reviewed", key="etd_mark_reviewed"):
                etd.status = "reviewed"
                etd.last_modified = datetime.now().isoformat()
                st.session_state.current_etd = etd
                st.success("Marked as reviewed!")

    with col_actions[2]:
        if etd.status == "reviewed":
            if st.button("Lock", key="etd_lock", type="primary"):
                etd.status = "locked"
                etd.last_modified = datetime.now().isoformat()
                st.session_state.current_etd = etd
                st.success("EtD card locked!")
        elif etd.status == "locked":
            if st.button("Unlock", key="etd_unlock"):
                etd.status = "reviewed"
                st.session_state.current_etd = etd
                st.info("EtD card unlocked for editing")


def _render_domain_judgment(etd: EvidenceToDecision, domain_id: str, editable: bool):
    """Render a single domain judgment."""
    domain_info = ETD_DOMAINS.get(domain_id, {})
    domain_name = domain_info.get("name", domain_id)
    domain_question = domain_info.get("question", "")
    options = domain_info.get("options", ["Uncertain"])

    judgment = etd.domain_judgments.get(domain_id)
    if not judgment:
        judgment = DomainJudgment(domain_id, domain_name, "Uncertain")
        etd.domain_judgments[domain_id] = judgment

    with st.container():
        st.markdown(f"**{domain_name}**")
        st.caption(domain_question)

        if editable and etd.status != "locked":
            # Dropdown for judgment
            current_index = options.index(judgment.judgment) if judgment.judgment in options else len(options) - 1
            new_judgment = st.selectbox(
                f"{domain_name} judgment",
                options,
                index=current_index,
                key=f"etd_domain_{domain_id}",
                label_visibility="collapsed"
            )
            judgment.judgment = new_judgment

            # Rationale expander
            with st.expander("Add rationale", expanded=False):
                judgment.rationale = st.text_area(
                    "Rationale",
                    value=judgment.rationale,
                    key=f"etd_rationale_{domain_id}",
                    label_visibility="collapsed",
                    height=100
                )
        else:
            # Display only
            _render_judgment_badge(judgment.judgment, domain_id)
            if judgment.rationale:
                st.caption(f"_{judgment.rationale}_")

        st.markdown("")  # Spacing


def _render_judgment_badge(judgment: str, domain_id: str):
    """Render a colored badge for a judgment."""
    # Color mapping based on favorability
    positive_judgments = ["Large", "Yes", "Probably yes", "Favors intervention", "Probably favors intervention", "High", "Moderate", "Increased", "Probably increased"]
    negative_judgments = ["Trivial", "No", "Probably no", "Favors comparator", "Probably favors comparator", "Very Low", "Reduced", "Probably reduced", "Large costs"]
    neutral_judgments = ["Neither", "Does not favor either", "Negligible", "Probably no impact"]

    if judgment in positive_judgments:
        if domain_id in ["harms", "resources"]:  # These are reversed
            color = "#fd7e14"  # Orange for bad harms/costs
        else:
            color = "#28a745"  # Green
    elif judgment in negative_judgments:
        if domain_id in ["harms", "resources"]:  # Reversed
            color = "#28a745"  # Green for low harms
        else:
            color = "#dc3545"  # Red
    elif judgment in neutral_judgments:
        color = "#6c757d"  # Gray
    elif "Uncertain" in judgment or "Varies" in judgment:
        color = "#ffc107"  # Yellow
    else:
        color = "#17a2b8"  # Blue default

    st.markdown(f"""
    <span style="background: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem;">
        {judgment}
    </span>
    """, unsafe_allow_html=True)


def _render_recommendation_badge(etd: EvidenceToDecision):
    """Render the final recommendation as a styled badge."""
    # Determine color based on direction and strength
    if etd.recommendation_direction == "For":
        if etd.recommendation_strength == "Strong":
            bg_color = "#28a745"  # Green
            icon = "++"
        else:
            bg_color = "#17a2b8"  # Teal
            icon = "+?"
    else:
        if etd.recommendation_strength == "Strong":
            bg_color = "#dc3545"  # Red
            icon = "--"
        else:
            bg_color = "#fd7e14"  # Orange
            icon = "-?"

    full_rec = etd.get_full_recommendation_text()

    st.markdown(f"""
    <div style="background: {bg_color}; color: white; padding: 16px; border-radius: 8px; margin: 8px 0;">
        <div style="font-size: 1.5rem; margin-bottom: 8px;">{icon}</div>
        <div style="font-size: 1.1rem; font-weight: 600;">{full_rec}</div>
        <div style="font-size: 0.9rem; margin-top: 8px; opacity: 0.9;">
            {etd.recommendation_strength} {etd.recommendation_direction}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if etd.justification:
        st.markdown(f"**Justification:** {etd.justification}")

    if etd.key_citations:
        st.markdown(f"**Key Citations:** {', '.join(etd.key_citations)}")


def render_etd_generator(
    question: str,
    expert_responses: Dict[str, str],
    evidence_summary: str = "",
    included_pmids: List[str] = None
):
    """
    Render the EtD generation interface.

    Shows button to generate EtD from discussion, then renders the card.

    Args:
        question: Clinical question
        expert_responses: Expert responses from GDG discussion
        evidence_summary: Summary of evidence
        included_pmids: Valid PMIDs for citation
    """
    st.markdown("---")
    st.markdown("### Generate Evidence-to-Decision Card")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.caption("Generate a structured EtD card from the GDG discussion above.")

    with col2:
        if st.button("Generate EtD Card", type="primary", key="generate_etd"):
            with st.spinner("Analyzing discussion and generating EtD..."):
                service = get_etd_service()
                etd = service.generate_etd_from_discussion(
                    question=question,
                    expert_responses=expert_responses,
                    evidence_summary=evidence_summary,
                    included_pmids=included_pmids or []
                )
                # Derive recommendation from judgments
                etd = service.derive_recommendation(etd)
                st.session_state.current_etd = etd
                st.success("EtD card generated!")
                st.rerun()

    # Display existing EtD if available
    if 'current_etd' in st.session_state:
        render_etd_card(st.session_state.current_etd, editable=True)

        # Export options
        with st.expander("Export EtD Card", expanded=False):
            col_export = st.columns(3)
            with col_export[0]:
                service = get_etd_service()
                md_content = service.format_etd_summary(st.session_state.current_etd)
                st.download_button(
                    "Download Markdown",
                    md_content,
                    file_name=f"etd_card_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
            with col_export[1]:
                json_content = json.dumps(st.session_state.current_etd.to_dict(), indent=2)
                st.download_button(
                    "Download JSON",
                    json_content,
                    file_name=f"etd_card_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )


def render_mini_etd_summary(etd: EvidenceToDecision):
    """
    Render a compact EtD summary for display in other views.

    Args:
        etd: EvidenceToDecision object
    """
    if not etd:
        return

    # Get key judgments
    benefits = etd.domain_judgments.get("benefits")
    harms = etd.domain_judgments.get("harms")
    certainty = etd.domain_judgments.get("certainty")
    balance = etd.domain_judgments.get("balance")

    st.markdown("**EtD Summary:**")
    cols = st.columns(4)
    with cols[0]:
        st.caption("Benefits")
        st.markdown(f"**{benefits.judgment if benefits else 'N/A'}**")
    with cols[1]:
        st.caption("Harms")
        st.markdown(f"**{harms.judgment if harms else 'N/A'}**")
    with cols[2]:
        st.caption("Certainty")
        st.markdown(f"**{certainty.judgment if certainty else 'N/A'}**")
    with cols[3]:
        st.caption("Balance")
        st.markdown(f"**{balance.judgment if balance else 'N/A'}**")
