"""
Analysis Display Module

Display components for gap analysis and conflict detection results.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any

from services.analysis_service import AnalysisService


def render_gap_analysis(
    latest_responses: Dict[str, str],
    scenario: str
):
    """
    Render the gap analysis section.

    Args:
        latest_responses: Dictionary of expert name -> response content
        scenario: The current scenario/question type
    """
    st.markdown("---")
    st.subheader("Gap Analysis")
    st.caption("Analyzes discussion completeness, evidence quality, and quantification")

    if st.button("Run Gap Analysis", type="secondary"):
        with st.spinner("Analyzing discussion gaps..."):
            try:
                analysis_service = AnalysisService()
                # Convert responses to expected format
                responses_dict = {
                    exp: {'content': content}
                    for exp, content in latest_responses.items()
                }

                gap_result = analysis_service.analyze_gaps(
                    responses=responses_dict,
                    scenario=scenario
                )
                st.session_state.gap_analysis = gap_result

            except Exception as e:
                st.error(f"Gap analysis failed: {e}")

    # Display gap analysis results
    if st.session_state.get('gap_analysis'):
        _render_gap_result(st.session_state.gap_analysis)


def _render_gap_result(gap: Any):
    """
    Render the gap analysis result.

    Args:
        gap: GapAnalysisResult object
    """
    # Coverage and Quantification Scores
    col1, col2 = st.columns(2)

    with col1:
        coverage_pct = int(gap.coverage_score * 100)
        st.metric("Coverage Score", f"{coverage_pct}%")

        if coverage_pct >= 80:
            st.success("Strong coverage")
        elif coverage_pct >= 50:
            st.warning("Moderate coverage")
        else:
            st.error("Significant gaps")

    with col2:
        quant_pct = int(gap.quantification_score * 100)
        st.metric("Quantification Score", f"{quant_pct}%")

        if quant_pct >= 70:
            st.success("Well-quantified")
        elif quant_pct >= 40:
            st.warning("Needs more data")
        else:
            st.error("Lacks specifics")

    # Strengths and Gaps
    col1, col2 = st.columns(2)

    with col1:
        if gap.strengths:
            st.markdown("**Topics Covered:**")
            for s in gap.strengths[:5]:
                st.markdown(f"- {s.replace('_', ' ').title()}")

    with col2:
        if gap.gaps:
            st.markdown("**Missing Topics:**")
            for g in gap.gaps[:5]:
                st.markdown(f"- {g.replace('_', ' ').title()}")

    # Evidence Issues
    if gap.evidence_issues:
        with st.expander(f"Evidence Issues ({len(gap.evidence_issues)})"):
            for issue in gap.evidence_issues[:5]:
                st.markdown(f"**{issue.get('expert', 'Unknown')}** ({issue.get('issue_type', 'unknown')})")
                st.markdown(f"_{issue.get('details', issue.get('excerpt', ''))[:150]}..._")

    # Recommendations
    if gap.recommendations:
        st.markdown("**Recommendations:**")
        for rec in gap.recommendations[:4]:
            st.info(rec)


def render_conflict_detection(
    latest_responses: Dict[str, str]
):
    """
    Render the conflict detection section.

    Args:
        latest_responses: Dictionary of expert name -> response content
    """
    st.markdown("---")
    st.subheader("Conflict Detection")
    st.caption("Identifies disagreements and divergent views between experts")

    col1, col2 = st.columns([3, 1])

    with col1:
        use_adversarial = st.checkbox(
            "Adversarial Mode",
            help="Generate tough reviewer-style questions"
        )

    with col2:
        if st.button("Detect Conflicts", type="secondary"):
            with st.spinner("Detecting conflicts..."):
                try:
                    analysis_service = AnalysisService()
                    responses_dict = {
                        exp: {'content': content}
                        for exp, content in latest_responses.items()
                    }

                    conflict_result = analysis_service.detect_conflicts(
                        responses=responses_dict
                    )
                    st.session_state.conflict_analysis = conflict_result
                    st.session_state.use_adversarial = use_adversarial

                except Exception as e:
                    st.error(f"Conflict detection failed: {e}")

    # Display conflict analysis results
    if st.session_state.get('conflict_analysis'):
        _render_conflict_result(st.session_state.conflict_analysis)


def _render_conflict_result(conflicts: Any):
    """
    Render the conflict detection result.

    Args:
        conflicts: ConflictResult object
    """
    if not conflicts.conflicts:
        st.success("No significant conflicts detected between experts")
    else:
        st.markdown(f"**{len(conflicts.conflicts)} conflict(s) detected**")

        # Group by severity
        critical = [c for c in conflicts.conflicts if c.severity == 'critical']
        moderate = [c for c in conflicts.conflicts if c.severity == 'moderate']
        minor = [c for c in conflicts.conflicts if c.severity == 'minor']

        if critical:
            st.error(f"Critical conflicts: {len(critical)}")
            for c in critical[:3]:
                with st.expander(f"[CRITICAL] {c.metric}"):
                    for expert, value in c.values.items():
                        st.markdown(f"**{expert}**: {value}")
                    st.caption(f"Rationale: {c.rationale}")

        if moderate:
            st.warning(f"Moderate conflicts: {len(moderate)}")
            for c in moderate[:3]:
                with st.expander(f"[MODERATE] {c.metric}"):
                    for expert, value in c.values.items():
                        st.markdown(f"**{expert}**: {value}")
                    st.caption(f"Rationale: {c.rationale}")

        if minor:
            st.info(f"Minor conflicts: {len(minor)}")

    # Clarification prompts
    if conflicts.clarification_needed:
        st.markdown("---")
        st.markdown("**Clarification Questions:**")

        for prompt in conflicts.clarification_needed[:3]:
            st.markdown(prompt)

            if st.button("Use as next round question", key=f"use_q_{hash(prompt)}"):
                st.session_state.expert_clinical_question = prompt.split('\n\n')[-1][:200]
                st.rerun()

    # Decision Memo
    if conflicts.decision_memo:
        with st.expander("Decision Memo", expanded=False):
            st.markdown(conflicts.decision_memo)
            st.download_button(
                "Download Decision Memo",
                conflicts.decision_memo,
                file_name=f"decision_memo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )


def clear_analysis_results():
    """Clear all analysis results from session state."""
    if 'gap_analysis' in st.session_state:
        del st.session_state.gap_analysis
    if 'conflict_analysis' in st.session_state:
        del st.session_state.conflict_analysis
