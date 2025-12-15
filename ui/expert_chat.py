"""
Expert Chat UI Module

Provides an interactive chat interface for the Research Assistant.
Features:
- ResearchAgent Agentic Workflow (Reasoning + Tools)
- Tool Output Visualization
- Context Integration
"""

import streamlit as st
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from config import settings
from services.chat_service import ResearchAgent
from core.database import DatabaseManager, CitationDAO, SearchHistoryDAO, QueryCacheDAO

# Import RAG retrieval (optional)
try:
    from core.retrieval import LocalRetriever
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    
# Import Tavily web search (optional)
try:
    from integrations.tavily import search_web_for_rag
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

logger = logging.getLogger(__name__)

def get_research_agent():
    """Get or create ResearchAgent instance."""
    if not settings.OPENAI_API_KEY:
        return None
        
    if "research_agent" not in st.session_state:
        st.session_state.research_agent = ResearchAgent(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-5-mini" # or use settings.OPENAI_MODEL
        )
    return st.session_state.research_agent

def build_chat_context_from_session() -> str:
    """Build context from session state (documents + discussions)."""
    context_parts = []

    # Add documents from search results
    if st.session_state.get('search_results') and st.session_state.search_results.get('citations'):
        citations = st.session_state.search_results['citations']
        context_parts.append("= EVIDENCE FROM LITERATURE SEARCH =")
        for i, c in enumerate(citations[:10], 1):
             title = c.title if hasattr(c,'title') else c.get('title','Untitled')
             pmid = c.pmid if hasattr(c,'pmid') else c.get('pmid','N/A')
             context_parts.append(f"[{i}] {title} (PMID: {pmid})")
    
    return "\n".join(context_parts)

def retrieve_rag_context(query: str, top_k: int = 5) -> str:
    """Retrieve context from indexed documents."""
    if not RAG_AVAILABLE or not settings.ENABLE_LOCAL_RAG:
        return ""
        
    try:
        retriever = LocalRetriever()
        results = retriever.retrieve(query, top_k=top_k, project_filter=st.session_state.get('current_project_name'))
        if not results:
            return ""
            
        context = ["= RELEVANT DOCUMENT EXCERPTS ="]
        for i, res in enumerate(results, 1):
            context.append(f"[{i}] {res.get('content','')} (Source: {res.get('source','')})")
        return "\n\n".join(context)
    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return ""

def render_expert_chat(database: DatabaseManager = None):
    """Render the Research Assistant Chat interface."""
    st.title("Intelligent Research Assistant")
    
    if not settings.OPENAI_API_KEY:
        st.error("OpenAI API key not configured.")
        return

    # Check for project
    if not st.session_state.get('current_project_name'):
        st.info("Create a new project in the sidebar to get started")
        return

    agent = get_research_agent()
    
    # Initialize chat history
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
        
    # Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Assistant Settings")
    if st.sidebar.button("Clear Conversation", use_container_width=True):
        st.session_state.agent_messages = []
        st.rerun()

    # Display Chat
    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            # If message results from tool use or reasoning, we can display it nicely?
            # For now, we store flattened messages.
            # But we want to show "Thinking" for past messages? 
            # Ideally we only show thinking for the current turn, or collapse it.
            # Let's keep it simple: History only stores final user/assistant exchange for simplicity
            # UNLESS we want to keep reasoning logs. 
            # Let's store simple content for history to avoid clutter unless we build a complex message object.
            st.markdown(msg["content"])
            
    # Chat Input
    if user_input := st.chat_input("Ask a research question..."):
        # 1. User Message
        st.session_state.agent_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # 2. Assistant Stream
        with st.chat_message("assistant"):
            # Prepare DAOs
            citation_dao = CitationDAO(database) if database else None
            search_dao = SearchHistoryDAO(database) if database else None
            query_cache_dao = QueryCacheDAO(database) if database else None
            
            # Prepare Context
            session_context = build_chat_context_from_session()
            rag_context = retrieve_rag_context(user_input)
            full_context = f"{session_context}\n\n{rag_context}"
            
            # Containers for streaming
            reasoning_container = st.empty()
            response_container = st.empty()
            
            full_response = ""
            current_reasoning = ""
            
            # We use an expander for reasoning logs
            with st.status("Thinking...", expanded=True) as status:
                for event in agent.run_agent_stream(
                    question=user_input,
                    context=full_context,
                    project_id=st.session_state.current_project_id,
                    citation_dao=citation_dao,
                    search_dao=search_dao,
                    query_cache_dao=query_cache_dao,
                    history=st.session_state.agent_messages[:-1] # Exclude just added user msg to avoid double add? run_agent adds it.
                    # run_agent expects history WITHOUT the current new question usually, it appends it.
                ):
                    event_type = event["type"]
                    
                    if event_type == "reasoning":
                        st.write(f"🤔 {event['content']}")

                    elif event_type == "plan":
                        st.markdown(
                            f"""
                            <div class="paper-card research-plan-card">
                                <h3>📋 Strategic Research Plan</h3>
                                <div style="font-size: 0.95rem; color: #334155;">
                                    {event['content'].replace(chr(10), '<br>')}
                                </div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

                    
                    elif event_type == "tool_start":
                        st.write(f"🛠️ **Executing Tool:** `{event['tool']}`")
                        with st.expander("Input"):
                            st.code(event['input'])
                            
                    elif event_type == "tool_end":
                        st.write(f"✅ **Tool Output:** {event['output']}")
                        
                    elif event_type == "chunk":
                        full_response += event["content"]
                        response_container.markdown(full_response + "▌")
                        
                    elif event_type == "error":
                        st.error(event["content"])
                
                status.update(label="Finished thinking", state="complete", expanded=False)
            
            # Finalize
            if full_response:
                response_container.markdown(full_response)
                st.session_state.agent_messages.append({"role": "assistant", "content": full_response})

    # Quick actions
    st.markdown("---")
    if st.button("Export Chat"):
        chat_text = "\n\n".join([f"**{m['role'].upper()}**: {m['content']}" for m in st.session_state.agent_messages])
        st.download_button("Download", chat_text, "research_chat.md")

def render_quick_ask_panel():
    """Simple wrapper for simple questions if needed."""
    pass
