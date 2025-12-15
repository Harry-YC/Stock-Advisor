"""
Debate Mode Module

Collaborative debate interface for resolving trade-offs between experts.
"""

import streamlit as st
from typing import List, Any, Optional

from services.expert_service import ExpertDiscussionService
from config import settings


def render_debate_section(
    clinical_question: str,
    selected_experts: List[str],
    citations: Optional[List[Any]] = None,
    scenario: str = "general"
):
    """
    Render the collaborative debate section.

    Args:
        clinical_question: The current research question
        selected_experts: List of expert names
        citations: List of citations for context
        scenario: The current scenario/question type
    """
    if not st.session_state.expert_discussion:
        return

    st.markdown("---")
    st.subheader("⚔️ Collaborative Debate (DeepMind Co-Scientist)")
    st.caption("Resolves trade-offs by forcing a Proposal -> Challenge -> Mitigation loop between two experts.")

    # Expert selection for debate
    col_deb1, col_deb2, col_deb3 = st.columns(3)

    with col_deb1:
        debate_pro = st.selectbox(
            "Proponent",
            selected_experts,
            key="debate_pro"
        )

    with col_deb2:
        # Challenger must be different from proponent
        challenger_options = [e for e in selected_experts if e != debate_pro]
        debate_con = st.selectbox(
            "Challenger",
            challenger_options,
            key="debate_con"
        )

    with col_deb3:
        debate_topic = st.text_input(
            "Debate Topic",
            placeholder="e.g. Efficacy vs Toxicity"
        )

    # Start debate button
    if st.button("🔥 Start Debate", key="start_debate", type="secondary"):
        if not debate_topic:
            st.warning("Enter a topic")
        elif not debate_con:
            st.warning("Select two different experts")
        else:
            with st.spinner(f"Running debate: {debate_pro} vs {debate_con}..."):
                expert_service = ExpertDiscussionService(api_key=settings.OPENAI_API_KEY)

                debate_res = expert_service.run_debate_round(
                    clinical_question,
                    debate_pro,
                    debate_con,
                    debate_topic,
                    citations,
                    scenario
                )
                st.session_state.debate_result = debate_res

    # Display debate results
    if st.session_state.get('debate_result'):
        _render_debate_result(st.session_state.debate_result)


def _render_debate_result(result: dict):
    """
    Render the debate result in a visual format.

    Args:
        result: Debate result dictionary containing:
            - topic: str
            - pro_expert: str
            - con_expert: str
            - proposal: str
            - challenge: str
            - mitigation: str
            - synthesis: str
    """
    st.markdown(f"### ⚔️ Debate Resolution: {result['topic']}")

    # 1. Proposal Node
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                padding: 16px; border-radius: 8px; margin: 8px 0;
                border-left: 4px solid #28a745;">
        <div style="font-size: 0.8rem; font-weight: 700; color: #155724; margin-bottom: 8px;">
            PROPOSAL ({result['pro_expert']})
        </div>
        <div style="color: #155724;">
            {result['proposal'].replace(chr(10), '<br>')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Challenge Node
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                padding: 16px; border-radius: 8px; margin: 8px 0;
                border-left: 4px solid #ffc107;">
        <div style="font-size: 0.8rem; font-weight: 700; color: #856404; margin-bottom: 8px;">
            CHALLENGE ({result['con_expert']})
        </div>
        <div style="color: #856404;">
            {result['challenge'].replace(chr(10), '<br>')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Mitigation Node
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #cce5ff 0%, #b8daff 100%);
                padding: 16px; border-radius: 8px; margin: 8px 0;
                border-left: 4px solid #17a2b8;">
        <div style="font-size: 0.8rem; font-weight: 700; color: #004085; margin-bottom: 8px;">
            MITIGATION ({result['pro_expert']})
        </div>
        <div style="color: #004085;">
            {result['mitigation'].replace(chr(10), '<br>')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. Synthesis
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                padding: 20px; border-radius: 8px; margin: 16px 0;">
        <h3 style="color: white !important; margin-top: 0;">🏛️ Chairperson Synthesis</h3>
        <div style="color: #E2E8F0;">
            {result['synthesis'].replace(chr(10), '<br>')}
        </div>
    </div>
    """, unsafe_allow_html=True)


def clear_debate_result():
    """Clear the current debate result."""
    if 'debate_result' in st.session_state:
        del st.session_state.debate_result


def export_debate_to_markdown(result: dict) -> str:
    """
    Export debate result to markdown format.

    Args:
        result: Debate result dictionary

    Returns:
        Markdown formatted string
    """
    return f"""# Debate Resolution: {result['topic']}

## Proposal ({result['pro_expert']})
{result['proposal']}

## Challenge ({result['con_expert']})
{result['challenge']}

## Mitigation ({result['pro_expert']})
{result['mitigation']}

## Chairperson Synthesis
{result['synthesis']}
"""
