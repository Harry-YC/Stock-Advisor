"""
Expert Chat Module

Interactive Q&A chat interface for follow-up questions to the expert panel.
"""

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any, Optional

from config import settings


def build_chat_context(citations: List[Any], expert_discussion: Dict) -> str:
    """
    Build context string from papers and prior discussion.

    Args:
        citations: List of Citation objects or dicts
        expert_discussion: Dictionary of expert discussion rounds

    Returns:
        Formatted context string for chat
    """
    from services.chat_service import ChatService
    chat_service = ChatService(api_key=settings.OPENAI_API_KEY)
    return chat_service.build_context(citations, expert_discussion)


def render_expert_chat(
    clinical_question: str,
    citations: Optional[List[Any]] = None,
    selected_experts: Optional[List[str]] = None
):
    """
    Render the interactive Q&A chat interface.

    Args:
        clinical_question: The current research question
        citations: List of citations for context
        selected_experts: List of expert names to chat with
    """
    st.markdown("---")
    st.subheader("💬 Interactive Q&A")
    st.caption("Ask follow-up questions - each selected expert will respond with streaming")

    # Import here to avoid circular imports
    try:
        from gdg import call_expert_stream
    except ImportError:
        st.error("GDG module not available for chat")
        return

    if not st.session_state.expert_discussion:
        st.info("💡 Run an expert discussion round first, then come back here to ask follow-up questions.")
        return

    # Initialize active chat experts if not set
    if not st.session_state.active_chat_experts:
        st.session_state.active_chat_experts = selected_experts or []

    # Display existing messages
    for msg in st.session_state.expert_chat_messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🧬"):
                st.markdown(f"**{msg.get('expert', 'Expert')}**")
                st.markdown(msg["content"])

    # Chat input
    if user_question := st.chat_input("Ask a follow-up question to the expert panel..."):
        # Add user message
        st.session_state.expert_chat_messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        # Build context from citations and discussion
        chat_context = build_chat_context(
            citations or [],
            st.session_state.expert_discussion
        )

        # Get responses from each active expert
        for expert_name in st.session_state.active_chat_experts:
            with st.chat_message("assistant", avatar="🧬"):
                st.markdown(f"**{expert_name}**")
                response_placeholder = st.empty()
                full_response = ""

                try:
                    with st.spinner("Thinking..."):
                        for chunk in call_expert_stream(
                            persona_name=expert_name,
                            clinical_question=user_question,
                            evidence_context=chat_context,
                            round_num=1,
                            previous_responses=None,
                            priors_text=None,
                            model=settings.EXPERT_MODEL,
                            max_completion_tokens=2000
                        ):
                            if chunk.get("type") == "chunk":
                                full_response += chunk.get("content", "")
                                response_placeholder.markdown(full_response + "▌")
                            elif chunk.get("type") == "complete":
                                response_placeholder.markdown(full_response)

                    st.session_state.expert_chat_messages.append({
                        "role": "assistant",
                        "expert": expert_name,
                        "content": full_response
                    })

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    response_placeholder.markdown(error_msg)
                    st.session_state.expert_chat_messages.append({
                        "role": "assistant",
                        "expert": expert_name,
                        "content": error_msg
                    })

        st.rerun()

    # Chat actions
    if st.session_state.expert_chat_messages:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.expert_chat_messages = []
                st.rerun()

        with col2:
            # Export chat
            chat_export = _export_chat_to_markdown(
                clinical_question,
                st.session_state.expert_chat_messages
            )
            st.download_button(
                "📥 Export Chat",
                chat_export,
                file_name=f"expert_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )


def _export_chat_to_markdown(clinical_question: str, messages: List[Dict]) -> str:
    """
    Export chat messages to markdown format.

    Args:
        clinical_question: The research question
        messages: List of chat messages

    Returns:
        Markdown formatted string
    """
    lines = [
        "# Expert Q&A Chat",
        "",
        f"**Research Question:** {clinical_question}",
        "",
        "---",
        ""
    ]

    for msg in messages:
        if msg["role"] == "user":
            lines.append(f"## User Question")
            lines.append(msg['content'])
            lines.append("")
        else:
            lines.append(f"### {msg.get('expert', 'Expert')}")
            lines.append(msg['content'])
            lines.append("")

    return "\n".join(lines)
