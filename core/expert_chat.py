"""
Expert Chat Module

Provides interactive chat capabilities with expert personas:
1. Single Expert Mode - Chat with one selected expert
2. Panel Router Mode - System routes questions to best expert(s)

Integrates full context from:
- Ingested documents (Part 1)
- Panel discussions (Part 2)
- Knowledge store (facts + triples)
"""

import os
import re
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI

# Import GDG expert definitions
from gdg import GDG_PERSONAS, COGNITIVE_CONSTRAINTS

# Alias for backward compatibility
PRECLINICAL_EXPERTS = GDG_PERSONAS


# =============================================================================
# CHAT SYSTEM PROMPTS
# =============================================================================

SINGLE_EXPERT_SYSTEM = """You are {role}, an expert in {specialty}.

**YOUR PERSPECTIVE:**
{perspective}

**COMMUNICATION STYLE:**
You are now in a conversational chat mode, not a formal panel discussion.
- Be direct and helpful
- Answer the user's specific questions
- Draw on your expertise and the provided context
- Cite sources when making factual claims: (PMID: XXXX) or (Document: Title)
- Use [EVIDENCE], [ASSUMPTION], or [OPINION] tags for key claims
- Ask clarifying questions if the user's question is ambiguous

**YOUR FOCUS AREAS:** {focus_areas}

**CONTEXT AVAILABLE:**
You have access to:
1. Evidence corpus (research papers, uploaded documents)
2. Previous panel discussions on this project
3. Accumulated knowledge from prior analyses

Use this context to provide informed, evidence-based responses.
"""

PANEL_ROUTER_SYSTEM = """You are a Panel Router coordinating a team of drug development experts.

Your role is to:
1. Analyze the user's question to determine which expert(s) are best suited to answer
2. Route the question to 1-3 relevant experts based on their specialties
3. Synthesize their responses into a cohesive answer

**AVAILABLE EXPERTS:**
{expert_list}

**ROUTING RULES:**
- Match question keywords to expert specialty_keywords
- For broad questions, select 2-3 complementary experts
- For specific technical questions, select the single best expert
- Always explain which expert(s) are responding and why

**CONTEXT AVAILABLE:**
You have access to evidence corpus and previous panel discussions.
Draw on this context to provide informed responses.
"""

CHAT_CONTEXT_TEMPLATE = """
================================================================================
AVAILABLE CONTEXT
================================================================================

{documents_section}

{discussions_section}

{knowledge_section}

================================================================================
"""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ChatMessage:
    """Single message in a chat conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    expert_name: Optional[str] = None  # Which expert responded (for panel mode)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "expert_name": self.expert_name,
            "timestamp": self.timestamp
        }


@dataclass
class ChatConversation:
    """A chat conversation with history."""
    id: str
    mode: str  # "single_expert" or "panel_router"
    expert_name: Optional[str] = None  # For single expert mode
    title: str = "New Conversation"
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def add_message(self, role: str, content: str, expert_name: Optional[str] = None):
        """Add a message to the conversation."""
        self.messages.append(ChatMessage(
            role=role,
            content=content,
            expert_name=expert_name
        ))

    def get_openai_messages(self, system_prompt: str) -> List[Dict[str, str]]:
        """Format conversation for OpenAI API."""
        messages = [{"role": "system", "content": system_prompt}]

        for msg in self.messages:
            if msg.role in ["user", "assistant"]:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        return messages

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.mode,
            "expert_name": self.expert_name,
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at
        }


# =============================================================================
# EXPERT CHAT CLASS
# =============================================================================

class ExpertChat:
    """
    Main class for expert chat functionality.

    Supports:
    - Single expert mode: Chat with one specific expert
    - Panel router mode: System routes to appropriate expert(s)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-5-mini",
        max_tokens: int = 2000
    ):
        """
        Initialize ExpertChat.

        Args:
            openai_api_key: OpenAI API key (uses env var if not provided)
            model: Model to use (recommend gpt-5-mini for chat)
            max_tokens: Max tokens per response
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.client = None

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, timeout=30.0)

    def _get_expert_system_prompt(self, expert_name: str) -> str:
        """Build system prompt for single expert mode."""
        if expert_name not in PRECLINICAL_EXPERTS:
            raise ValueError(f"Unknown expert: {expert_name}")

        expert = PRECLINICAL_EXPERTS[expert_name]
        constraints = COGNITIVE_CONSTRAINTS.get(expert_name, {})

        return SINGLE_EXPERT_SYSTEM.format(
            role=expert["role"],
            specialty=expert["specialty"],
            perspective=expert["perspective"],
            focus_areas=", ".join(constraints.get("focus_areas", []))
        )

    def _get_panel_router_system_prompt(self) -> str:
        """Build system prompt for panel router mode."""
        expert_list = []
        for name, expert in PRECLINICAL_EXPERTS.items():
            if name != "Panel Chair":
                expert_list.append(f"- **{name}**: {expert['specialty']}")

        return PANEL_ROUTER_SYSTEM.format(
            expert_list="\n".join(expert_list)
        )

    def route_question(
        self,
        question: str,
        available_experts: Optional[List[str]] = None,
        max_experts: int = 3
    ) -> List[str]:
        """
        Route a question to the most appropriate expert(s).

        Uses keyword matching against expert specialty_keywords.

        Args:
            question: User's question
            available_experts: Experts to consider (defaults to all)
            max_experts: Maximum experts to return

        Returns:
            List of expert names, sorted by relevance
        """
        if available_experts is None:
            available_experts = [
                name for name in PRECLINICAL_EXPERTS.keys()
                if name != "Panel Chair"
            ]

        question_lower = question.lower()
        scores = {}

        for expert_name in available_experts:
            if expert_name not in PRECLINICAL_EXPERTS:
                continue

            expert = PRECLINICAL_EXPERTS[expert_name]
            keywords = expert.get("specialty_keywords", [])

            score = 0
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    score += 1

            if score > 0:
                scores[expert_name] = score

        # Sort by score descending
        sorted_experts = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top experts
        result = [name for name, _ in sorted_experts[:max_experts]]

        # If no matches, return default expert
        if not result:
            result = ["Bioscience Lead"]

        return result

    def chat_single_expert(
        self,
        expert_name: str,
        message: str,
        conversation: ChatConversation,
        context: str = ""
    ) -> str:
        """
        Send a message to a single expert and get a response.

        Args:
            expert_name: Name of the expert
            message: User's message
            conversation: Conversation object (will be updated)
            context: Formatted context string

        Returns:
            Expert's response text
        """
        if not self.client:
            return "Error: No OpenAI API key configured"

        # Add user message to conversation
        conversation.add_message("user", message)

        # Build system prompt with context
        system_prompt = self._get_expert_system_prompt(expert_name)
        if context:
            system_prompt += f"\n\n{context}"

        # Get conversation history
        messages = conversation.get_openai_messages(system_prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.7
            )

            content = response.choices[0].message.content or ""

            # Add response to conversation
            conversation.add_message("assistant", content, expert_name)

            return content

        except Exception as e:
            error_msg = f"Error calling API: {str(e)}"
            conversation.add_message("assistant", error_msg, expert_name)
            return error_msg

    def chat_single_expert_stream(
        self,
        expert_name: str,
        message: str,
        conversation: ChatConversation,
        context: str = ""
    ) -> Generator[str, None, None]:
        """
        Stream a response from a single expert.

        Args:
            expert_name: Name of the expert
            message: User's message
            conversation: Conversation object (will be updated)
            context: Formatted context string

        Yields:
            Response text chunks
        """
        if not self.client:
            yield "Error: No OpenAI API key configured"
            return

        # Add user message to conversation
        conversation.add_message("user", message)

        # Build system prompt with context
        system_prompt = self._get_expert_system_prompt(expert_name)
        if context:
            system_prompt += f"\n\n{context}"

        # Get conversation history
        messages = conversation.get_openai_messages(system_prompt)

        full_response = ""

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.7,
                stream=True
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    yield text

            # Add complete response to conversation
            conversation.add_message("assistant", full_response, expert_name)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield error_msg
            conversation.add_message("assistant", error_msg, expert_name)

    def chat_panel_router(
        self,
        message: str,
        conversation: ChatConversation,
        context: str = "",
        available_experts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Route a question to appropriate expert(s) and get responses.

        Args:
            message: User's message
            conversation: Conversation object (will be updated)
            context: Formatted context string
            available_experts: Experts to consider

        Returns:
            Dict with:
            - 'routed_to': List of expert names
            - 'responses': Dict[expert_name] -> response
            - 'synthesis': Combined response (if multiple experts)
        """
        if not self.client:
            return {
                "routed_to": [],
                "responses": {},
                "synthesis": "Error: No OpenAI API key configured"
            }

        # Add user message
        conversation.add_message("user", message)

        # Route to experts
        experts = self.route_question(message, available_experts)

        responses = {}

        # Get response from each expert
        for expert_name in experts:
            system_prompt = self._get_expert_system_prompt(expert_name)
            if context:
                system_prompt += f"\n\n{context}"

            # For panel router, we use fresh context per expert
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0.7
                )

                responses[expert_name] = response.choices[0].message.content or ""

            except Exception as e:
                responses[expert_name] = f"Error: {str(e)}"

        # Synthesize if multiple experts
        synthesis = ""
        if len(experts) > 1:
            synthesis = self._synthesize_expert_responses(
                message, responses, context
            )
            conversation.add_message("assistant", synthesis, "Panel")
        else:
            # Single expert response
            expert = experts[0]
            synthesis = responses.get(expert, "")
            conversation.add_message("assistant", synthesis, expert)

        return {
            "routed_to": experts,
            "responses": responses,
            "synthesis": synthesis
        }

    def _synthesize_expert_responses(
        self,
        question: str,
        responses: Dict[str, str],
        context: str = ""
    ) -> str:
        """Synthesize multiple expert responses into one."""
        if not self.client:
            return "Error: No OpenAI API key"

        # Format expert responses
        formatted = []
        for expert, response in responses.items():
            formatted.append(f"**{expert}:**\n{response}")

        system_prompt = """You are synthesizing responses from multiple drug development experts.

Your task:
1. Combine their insights into a cohesive response
2. Note areas of agreement and any differences in perspective
3. Provide a clear, actionable answer to the user's question
4. Cite the expert sources: (per EXPERT_NAME)

Keep the synthesis concise but comprehensive."""

        user_content = f"""User question: {question}

Expert responses:
{chr(10).join(formatted)}

Provide a synthesized response."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=self.max_tokens,
                temperature=0.5
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            return f"Synthesis error: {str(e)}"


# =============================================================================
# CONTEXT ASSEMBLY
# =============================================================================

class ContextAssembler:
    """
    Assembles context from various sources for chat.

    Sources:
    - Documents (uploaded/ingested files)
    - Panel discussions
    - Knowledge store (facts + triples)
    """

    def __init__(self, database=None, knowledge_store=None):
        """
        Initialize ContextAssembler.

        Args:
            database: Database instance for accessing documents/discussions
            knowledge_store: KnowledgeStore instance for facts/triples
        """
        self.database = database
        self.knowledge_store = knowledge_store

    def build_context(
        self,
        project_id: str,
        max_documents: int = 10,
        max_discussions: int = 3,
        include_knowledge: bool = True
    ) -> str:
        """
        Build full context string for chat.

        Args:
            project_id: Project to get context for
            max_documents: Maximum documents to include
            max_discussions: Maximum discussions to include
            include_knowledge: Include knowledge store facts

        Returns:
            Formatted context string
        """
        documents_section = self._format_documents(project_id, max_documents)
        discussions_section = self._format_discussions(project_id, max_discussions)
        knowledge_section = ""

        if include_knowledge and self.knowledge_store:
            knowledge_section = self._format_knowledge(project_id)

        return CHAT_CONTEXT_TEMPLATE.format(
            documents_section=documents_section,
            discussions_section=discussions_section,
            knowledge_section=knowledge_section
        )

    def _format_documents(self, project_id: str, max_docs: int) -> str:
        """Format documents section."""
        if not self.database:
            return "**DOCUMENTS:** No database available"

        # Get documents from database
        try:
            from core.database import DocumentDAO
            doc_dao = DocumentDAO(self.database.conn)
            documents = doc_dao.get_documents_by_project(project_id)
        except Exception:
            documents = []

        if not documents:
            return "**DOCUMENTS:** No documents loaded"

        lines = ["**EVIDENCE DOCUMENTS:**", ""]

        for i, doc in enumerate(documents[:max_docs], 1):
            title = doc.get("title", "Untitled")
            source_type = doc.get("source_type", "unknown")
            content = doc.get("content", "")

            # Truncate content
            if len(content) > 500:
                content = content[:500] + "..."

            lines.append(f"[{i}] **{title}** ({source_type})")
            if doc.get("pmid"):
                lines.append(f"PMID: {doc['pmid']}")
            lines.append(content)
            lines.append("")

        if len(documents) > max_docs:
            lines.append(f"... and {len(documents) - max_docs} more documents")

        return "\n".join(lines)

    def _format_discussions(self, project_id: str, max_discussions: int) -> str:
        """Format panel discussions section."""
        if not self.database:
            return "**PANEL DISCUSSIONS:** No database available"

        # Get discussions from database
        try:
            discussions = self.database.discussion_dao.get_discussions_by_project(project_id)
        except Exception:
            discussions = []

        if not discussions:
            return "**PANEL DISCUSSIONS:** No prior discussions"

        lines = ["**PREVIOUS PANEL DISCUSSIONS:**", ""]

        for disc in discussions[:max_discussions]:
            topic = disc.get("topic", "Untitled discussion")
            created = disc.get("created_at", "")[:10]  # Just date

            lines.append(f"**{topic}** ({created})")

            # Add brief summary of rounds if available
            rounds = disc.get("rounds", [])
            if rounds:
                lines.append(f"Rounds: {len(rounds)} | Experts: {', '.join(r.get('expert', 'Unknown') for r in rounds[:3])}")

            # Add consensus if available
            if disc.get("consensus"):
                consensus = disc["consensus"]
                if len(consensus) > 200:
                    consensus = consensus[:200] + "..."
                lines.append(f"Consensus: {consensus}")

            lines.append("")

        return "\n".join(lines)

    def _format_knowledge(self, project_id: str) -> str:
        """Format knowledge store section."""
        if not self.knowledge_store:
            return "**KNOWLEDGE STORE:** Not available"

        try:
            facts = self.knowledge_store.get_all_facts()
            triples = self.knowledge_store.get_all_triples()
        except Exception:
            return "**KNOWLEDGE STORE:** Error loading"

        if not facts and not triples:
            return "**KNOWLEDGE STORE:** No accumulated knowledge"

        lines = ["**ACCUMULATED KNOWLEDGE:**", ""]

        # Format facts by expert
        if facts:
            lines.append("*Key Facts:*")
            for expert, expert_facts in list(facts.items())[:5]:
                lines.append(f"- {expert}:")
                for fact in expert_facts[:2]:
                    fact_text = fact if isinstance(fact, str) else fact.get("fact", "")
                    if len(fact_text) > 100:
                        fact_text = fact_text[:100] + "..."
                    lines.append(f"  • {fact_text}")
            lines.append("")

        # Format triples (relationships)
        if triples:
            lines.append("*Key Relationships:*")
            for triple in triples[:5]:
                subj = triple.get("subject", "?")
                pred = triple.get("predicate", "?")
                obj = triple.get("object", "?")
                lines.append(f"  • {subj} → {pred} → {obj}")

        return "\n".join(lines)

    def build_minimal_context(
        self,
        documents: List[Dict],
        discussion_summary: Optional[str] = None
    ) -> str:
        """
        Build context from provided data (without database).

        Args:
            documents: List of document dicts
            discussion_summary: Optional summary of prior discussions

        Returns:
            Formatted context string
        """
        lines = ["=" * 60, "CONTEXT", "=" * 60, ""]

        # Documents
        if documents:
            lines.append("**EVIDENCE:**")
            for i, doc in enumerate(documents[:10], 1):
                title = doc.get("title", "Untitled")
                content = doc.get("content", "")[:300]
                lines.append(f"[{i}] {title}")
                lines.append(content)
                lines.append("")

        # Discussion summary
        if discussion_summary:
            lines.append("**PRIOR DISCUSSION:**")
            lines.append(discussion_summary)

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_available_experts() -> List[str]:
    """Get list of available expert names (excluding Panel Chair)."""
    return [
        name for name in PRECLINICAL_EXPERTS.keys()
        if name != "Panel Chair"
    ]


def get_expert_description(expert_name: str) -> str:
    """Get a brief description of an expert's specialty."""
    if expert_name not in PRECLINICAL_EXPERTS:
        return "Unknown expert"

    expert = PRECLINICAL_EXPERTS[expert_name]
    return f"{expert['role']}: {expert['specialty']}"


def create_chat_conversation(
    mode: str = "single_expert",
    expert_name: Optional[str] = None,
    title: Optional[str] = None
) -> ChatConversation:
    """
    Create a new chat conversation.

    Args:
        mode: "single_expert" or "panel_router"
        expert_name: Expert for single expert mode
        title: Conversation title

    Returns:
        New ChatConversation instance
    """
    import uuid

    conv_id = f"chat_{uuid.uuid4().hex[:12]}"

    if not title:
        if mode == "single_expert" and expert_name:
            title = f"Chat with {expert_name}"
        else:
            title = "Panel Discussion"

    return ChatConversation(
        id=conv_id,
        mode=mode,
        expert_name=expert_name,
        title=title
    )
