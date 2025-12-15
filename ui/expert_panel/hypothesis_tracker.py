"""
Hypothesis Tracker Module

Extract and track key hypotheses from expert discussions.
"""

import streamlit as st
from typing import Dict, List, Any

from services.analysis_service import AnalysisService


def render_hypothesis_tracker(
    latest_responses: Dict[str, str],
    clinical_question: str,
    latest_round: int
):
    """
    Render the hypothesis tracking interface.

    Args:
        latest_responses: Dictionary of expert name -> response content
        clinical_question: The current research question
        latest_round: The most recent discussion round number
    """
    st.markdown("---")
    st.subheader("📊 Hypothesis Tracker")
    st.caption("Track key claims and their evidence strength across discussion rounds")

    # Extract hypotheses button
    if st.button("Extract Hypotheses from Discussion"):
        with st.spinner("Extracting key hypotheses..."):
            try:
                analysis_service = AnalysisService()
                hypotheses = analysis_service.extract_hypotheses(
                    responses=latest_responses,
                    clinical_question=clinical_question,
                    round_num=latest_round
                )

                # Add new hypotheses (avoid duplicates)
                existing_texts = {
                    h.get('hypothesis', '')
                    for h in st.session_state.tracked_hypotheses
                }
                new_hypotheses = [
                    h for h in hypotheses
                    if h.get('hypothesis') not in existing_texts
                ]

                st.session_state.tracked_hypotheses.extend(new_hypotheses)

                if new_hypotheses:
                    st.success(f"Extracted {len(new_hypotheses)} new hypotheses")
                else:
                    st.info("No new hypotheses found")

            except Exception as e:
                st.error(f"Extraction failed: {e}")

    # Display tracked hypotheses
    _render_hypothesis_list()


def _render_hypothesis_list():
    """Render the list of tracked hypotheses."""
    if not st.session_state.tracked_hypotheses:
        return

    # Sort by evidence strength
    sorted_hyps = sorted(
        st.session_state.tracked_hypotheses,
        key=lambda x: x.get('evidence_strength', 0),
        reverse=True
    )

    for i, hyp in enumerate(sorted_hyps):
        strength = hyp.get('evidence_strength', 3)
        strength_bar = "🟢" * strength + "⚪" * (5 - strength)

        with st.expander(f"{strength_bar} {hyp.get('hypothesis', 'Unknown')[:80]}..."):
            # Metrics row
            col1, col2, col3 = st.columns(3)
            col1.metric("Evidence Strength", f"{strength}/5")
            col2.metric("Type", hyp.get('evidence_type', 'Unknown'))
            col3.metric("Round", hyp.get('round', '?'))

            # Details
            st.markdown(f"**Full hypothesis:** {hyp.get('hypothesis', '')}")

            if hyp.get('supporting_experts'):
                st.markdown(f"**Supported by:** {', '.join(hyp['supporting_experts'])}")

            if hyp.get('key_data') and hyp['key_data'] != 'none':
                st.markdown(f"**Key data:** {hyp['key_data']}")

    # Clear button
    if st.button("🗑️ Clear All Hypotheses"):
        st.session_state.tracked_hypotheses = []
        st.rerun()


def add_hypothesis(hypothesis: Dict[str, Any]):
    """
    Manually add a hypothesis to the tracker.

    Args:
        hypothesis: Hypothesis dictionary with keys:
            - hypothesis: str
            - evidence_strength: int (1-5)
            - evidence_type: str
            - round: int
            - supporting_experts: List[str]
            - key_data: str
    """
    if 'tracked_hypotheses' not in st.session_state:
        st.session_state.tracked_hypotheses = []

    st.session_state.tracked_hypotheses.append(hypothesis)


def get_hypotheses_summary() -> str:
    """
    Get a summary of all tracked hypotheses.

    Returns:
        Markdown formatted summary
    """
    if not st.session_state.tracked_hypotheses:
        return "No hypotheses tracked."

    lines = ["## Tracked Hypotheses", ""]

    sorted_hyps = sorted(
        st.session_state.tracked_hypotheses,
        key=lambda x: x.get('evidence_strength', 0),
        reverse=True
    )

    for i, hyp in enumerate(sorted_hyps, 1):
        strength = hyp.get('evidence_strength', 3)
        lines.append(f"### {i}. {hyp.get('hypothesis', 'Unknown')}")
        lines.append(f"- **Evidence Strength:** {strength}/5")
        lines.append(f"- **Type:** {hyp.get('evidence_type', 'Unknown')}")
        if hyp.get('supporting_experts'):
            lines.append(f"- **Supporters:** {', '.join(hyp['supporting_experts'])}")
        lines.append("")

    return "\n".join(lines)
