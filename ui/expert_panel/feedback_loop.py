"""
Feedback Loop Module

Human-in-the-loop feedback collection for steering expert discussions.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, List, Any


# Feedback type definitions
FEEDBACK_TYPES = {
    "correction": "Correct an error or misstatement",
    "deepen": "Request deeper analysis on a topic",
    "redirect": "Shift focus to a different aspect",
    "challenge": "Challenge an assumption or conclusion",
    "add_context": "Add missing context or information"
}

FEEDBACK_ICONS = {
    "correction": "[FIX]",
    "deepen": "[DEEP]",
    "redirect": "[PIVOT]",
    "challenge": "[?]",
    "add_context": "[+]"
}


def render_feedback_loop(
    latest_responses: Dict[str, str],
    latest_round: int,
    clinical_question: str
):
    """
    Render the human feedback loop interface.

    Args:
        latest_responses: Dictionary of expert name -> response content
        latest_round: The most recent discussion round number
        clinical_question: The current research question
    """
    st.markdown("---")
    st.subheader("Human Feedback")
    st.caption("Provide feedback to steer the next discussion round")

    # Feedback input form
    col1, col2 = st.columns([2, 3])

    with col1:
        feedback_type = st.selectbox(
            "Feedback Type",
            options=list(FEEDBACK_TYPES.keys()),
            format_func=lambda x: FEEDBACK_TYPES[x]
        )

    with col2:
        target_expert = st.selectbox(
            "Target Expert (optional)",
            options=["All Experts"] + list(latest_responses.keys())
        )

    feedback_text = st.text_area(
        "Your Feedback",
        placeholder="e.g., 'The DMPK assessment didn't account for food effect on bioavailability' or 'Need more detail on the competitive landscape for this indication'",
        height=80
    )

    if st.button("Add Feedback", type="secondary", disabled=not feedback_text.strip()):
        new_feedback = {
            "type": feedback_type,
            "target": target_expert if target_expert != "All Experts" else None,
            "text": feedback_text.strip(),
            "round": latest_round,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.human_feedback.append(new_feedback)
        st.success("Feedback added! It will be incorporated in the next round.")

    # Display existing feedback
    _render_pending_feedback(clinical_question)


def _render_pending_feedback(clinical_question: str):
    """
    Render the list of pending feedback items.

    Args:
        clinical_question: The current research question
    """
    if not st.session_state.human_feedback:
        return

    st.markdown("**Pending Feedback:**")

    for i, fb in enumerate(st.session_state.human_feedback):
        target_str = f" → {fb['target']}" if fb.get('target') else ""
        type_icon = FEEDBACK_ICONS.get(fb['type'], "")

        with st.expander(f"{type_icon} {fb['text'][:50]}...{target_str}"):
            st.markdown(f"**Type:** {FEEDBACK_TYPES.get(fb['type'], fb['type'])}")

            if fb.get('target'):
                st.markdown(f"**Target:** {fb['target']}")

            st.markdown(f"**Feedback:** {fb['text']}")
            st.caption(f"Added during Round {fb.get('round', '?')}")

            if st.button("Remove", key=f"remove_fb_{i}"):
                st.session_state.human_feedback.pop(i)
                st.rerun()

    # Generate feedback-informed prompt
    if st.button("Generate Feedback-Informed Prompt"):
        feedback_prompt = _generate_feedback_prompt(st.session_state.human_feedback)
        st.code(feedback_prompt, language="markdown")
        st.info("Copy this prompt and add it to the research question for the next round, or use the 'Use as context' button below.")

        if st.button("Use as context for next round"):
            if clinical_question:
                st.session_state.expert_clinical_question = f"{clinical_question}\n\n{feedback_prompt}"
            st.session_state.human_feedback = []  # Clear after use
            st.rerun()


def _generate_feedback_prompt(feedback_items: List[Dict]) -> str:
    """
    Generate a prompt incorporating all feedback items.

    Args:
        feedback_items: List of feedback dictionaries

    Returns:
        Formatted prompt string
    """
    feedback_parts = []

    for fb in feedback_items:
        target_str = f" (specifically for {fb['target']})" if fb.get('target') else ""
        feedback_parts.append(f"- [{fb['type'].upper()}]{target_str}: {fb['text']}")

    return f"""**Human Reviewer Feedback from Previous Round:**

Please address the following feedback in your response:
{chr(10).join(feedback_parts)}

Incorporate this feedback while maintaining your expert perspective."""


def get_feedback_for_expert(expert_name: str, feedback_items: List[Dict]) -> List[Dict]:
    """
    Get feedback items relevant to a specific expert.

    Args:
        expert_name: Name of the expert
        feedback_items: List of all feedback items

    Returns:
        List of feedback items targeting this expert or all experts
    """
    return [
        fb for fb in feedback_items
        if fb.get('target') is None or fb.get('target') == expert_name
    ]


def clear_feedback():
    """Clear all pending feedback."""
    st.session_state.human_feedback = []
