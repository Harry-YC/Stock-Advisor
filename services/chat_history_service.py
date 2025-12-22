"""
Chat History Service

Manages persistent chat history with session support.
"""

import logging
from typing import List, Dict, Optional, Any

from mcp_server.database import (
    FinancialDatabase,
    get_database,
    ChatMessage,
    ChatSession,
)

logger = logging.getLogger(__name__)


class ChatHistoryService:
    """
    Service for managing persistent chat history.

    Features:
    - Save and retrieve chat messages
    - Session management
    - AI-generated session summaries
    """

    def __init__(self, db: Optional[FinancialDatabase] = None):
        self.db = db or get_database()

    def start_session(self, session_id: str, title: str = "") -> ChatSession:
        """
        Start a new chat session.

        Args:
            session_id: Unique session identifier
            title: Optional session title

        Returns:
            ChatSession object
        """
        return self.db.create_session(session_id, title)

    def save_user_message(
        self,
        session_id: str,
        content: str,
        tickers: Optional[List[str]] = None
    ) -> ChatMessage:
        """
        Save a user message.

        Args:
            session_id: Session ID
            content: Message content
            tickers: Optional list of mentioned tickers

        Returns:
            ChatMessage object
        """
        return self.db.save_message(
            session_id=session_id,
            role="user",
            content=content,
            tickers=tickers,
        )

    def save_assistant_message(
        self,
        session_id: str,
        content: str,
        expert_responses: Optional[Dict[str, str]] = None,
        tickers: Optional[List[str]] = None
    ) -> ChatMessage:
        """
        Save an assistant message with expert responses.

        Args:
            session_id: Session ID
            content: Main response content
            expert_responses: Dict mapping expert names to their responses
            tickers: List of analyzed tickers

        Returns:
            ChatMessage object
        """
        return self.db.save_message(
            session_id=session_id,
            role="assistant",
            content=content,
            expert_responses=expert_responses,
            tickers=tickers,
        )

    def get_history(self, session_id: str, limit: int = 50) -> List[ChatMessage]:
        """
        Get chat history for a session.

        Args:
            session_id: Session ID
            limit: Maximum messages to retrieve

        Returns:
            List of ChatMessage objects (oldest first)
        """
        return self.db.get_session_history(session_id, limit)

    def get_history_for_context(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get chat history formatted for LLM context.

        Args:
            session_id: Session ID
            limit: Number of recent messages

        Returns:
            List of {"role": ..., "content": ...} dicts
        """
        messages = self.db.get_session_history(session_id, limit)
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

    def list_recent_sessions(self, limit: int = 10) -> List[ChatSession]:
        """
        List recent chat sessions.

        Args:
            limit: Maximum sessions to return

        Returns:
            List of ChatSession objects
        """
        return self.db.list_sessions(limit)

    def get_session_summary(self, session_id: str) -> Optional[str]:
        """
        Get session summary if available.

        Args:
            session_id: Session ID

        Returns:
            Summary string or None
        """
        sessions = self.db.list_sessions(100)  # Search recent
        for session in sessions:
            if session.id == session_id:
                return session.summary if session.summary else None
        return None

    def generate_session_summary(
        self,
        session_id: str,
        llm_callable: Optional[callable] = None
    ) -> str:
        """
        Generate AI summary of a session.

        Args:
            session_id: Session ID
            llm_callable: Optional LLM function to use

        Returns:
            Generated summary
        """
        history = self.get_history(session_id, limit=20)

        if not history:
            return "Empty session"

        # Extract key info
        tickers = set()
        for msg in history:
            if msg.tickers:
                tickers.update(msg.tickers)

        # Create simple summary if no LLM
        user_messages = [m for m in history if m.role == "user"]
        if not user_messages:
            return "No user messages"

        first_query = user_messages[0].content[:100]
        ticker_str = ", ".join(sorted(tickers)) if tickers else "general"

        summary = f"Discussed {ticker_str}: {first_query}..."

        if llm_callable:
            try:
                # Build context for LLM
                context = "\n".join([
                    f"{m.role}: {m.content[:200]}"
                    for m in history[:10]
                ])
                prompt = f"Summarize this stock analysis conversation in 1-2 sentences:\n\n{context}"
                summary = llm_callable(prompt)
            except Exception as e:
                logger.warning(f"LLM summary failed: {e}")

        # Save the summary
        self.db.update_session_summary(session_id, summary)
        return summary

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its messages.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted
        """
        return self.db.delete_session(session_id)

    def format_sessions_list(self, limit: int = 5) -> str:
        """
        Format recent sessions as markdown for display.

        Args:
            limit: Number of sessions

        Returns:
            Markdown formatted string
        """
        sessions = self.list_recent_sessions(limit)

        if not sessions:
            return "No previous sessions found."

        lines = ["**Recent Sessions:**", ""]
        for i, session in enumerate(sessions, 1):
            date_str = session.updated_at.strftime("%m/%d %H:%M") if session.updated_at else "Unknown"
            title = session.title[:40] + "..." if len(session.title) > 40 else session.title
            lines.append(f"{i}. [{title}] - {date_str}")

        return "\n".join(lines)


# Convenience functions
def get_chat_history_service() -> ChatHistoryService:
    """Get chat history service instance."""
    return ChatHistoryService()
