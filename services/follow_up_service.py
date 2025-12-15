
import logging
from typing import Dict, List, Optional
from core.llm_utils import get_llm_client

logger = logging.getLogger(__name__)

class FollowUpService:
    """Service to handle follow-up questions in the conversational interface."""

    def __init__(self, api_key: str, model: str = "gemini-3-pro-preview"):
        self.api_key = api_key
        self.model = model

    def generate_response(
        self,
        question: str,
        chat_history: List[Dict],
        research_context: Dict
    ) -> str:
        """
        Generate a response to a follow-up question.
        
        Args:
            question: The user's follow-up question
            chat_history: Previous chat messages
            research_context: The original research result (as dict)
            
        Returns:
            The AI response string
        """
        try:
            # Build context from research result
            context_str = f"""
ORIGINAL QUESTION: {research_context.get('question', '')}
RECOMMENDATION: {research_context.get('recommendation', '')}
KEY FINDINGS: {chr(10).join(research_context.get('key_findings', []))}

EVIDENCE SUMMARY:
{self._format_evidence(research_context)}
"""
            
            # Build conversation for LLM
            messages = [
                {"role": "system", "content": f"You are a helpful research assistant. You are answering follow-up questions about a clinical research analysis.\n\nCONTEXT:\n{context_str}"}
            ]
            
            # Add existing history (excluding the current question if it's already there)
            # chat_history usually comes from st.session_state which might have just been updated
            
            # We want to exclude the last message if it IS the current question to avoid duplication in "user" role if we add it manually,
            # OR we just pass the history as is.
            
            # Let's filter history to be safe: valid roles only
            valid_msgs = []
            for msg in chat_history:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role in ['user', 'assistant'] and content:
                    valid_msgs.append({"role": role, "content": content})
            
            # If the last message in history is NOT the current question, add it. 
            # (In ui/home.py we append it before calling, so it SHOULD be there)
            if not valid_msgs or valid_msgs[-1]['content'] != question:
                valid_msgs.append({"role": "user", "content": question})
                
            messages.extend(valid_msgs)
            
            client = get_llm_client(api_key=self.api_key, model=self.model)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content or "I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"Follow-up generation failed: {e}")
            return f"Error generating response: {str(e)}"

    def _format_evidence(self, context: Dict) -> str:
        """Format evidence summary from context."""
        summary = context.get('evidence_summary', {})
        citations = summary.get('citations', [])
        
        lines = []
        for i, cit in enumerate(citations[:5]):
            # citations can be objects or dicts
            title = cit.get('title', 'Unknown') if isinstance(cit, dict) else getattr(cit, 'title', 'Unknown')
            lines.append(f"- {title}")
            
        return "\n".join(lines)
