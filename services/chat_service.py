"""
Research Agent Service

Transforms the chat into an agentic system capable of using tools:
- PubMed Search (via SearchService)
- RAG Retrieval (via LocalRetriever)
- Context Management

Implements a ReAct-style loop or Function Calling loop.
"""

import json
import logging
from typing import Dict, List, Generator, Optional, Any, Union
from datetime import datetime

from config import settings
from services.search_service import SearchService

logger = logging.getLogger(__name__)

class ResearchAgent:
    """
    Intelligent Research Assistant.
    
    Capabilities:
    1. Analyze user queries.
    2. Decide to use tools (Search, Retrieve).
    3. Execute tools and observe outputs.
    4. Synthesize grounded responses.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        search_service: Optional[SearchService] = None
    ):
        from config import settings
        self.api_key = api_key
        self.model = model or settings.EXPERT_MODEL
        self.search_service = search_service or SearchService(openai_api_key=api_key)
        
        # Tools definition for OpenAI
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_pubmed",
                    "description": "Search PubMed for medical literature. Use this when you need to find papers, clinical trials, or verify medical facts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query (e.g., 'immunotherapy for NSCLC', 'comparison of drug A vs B')"
                            },
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def run_agent_stream(
        self,
        question: str,
        context: str,
        project_id: str,
        citation_dao: Any,
        search_dao: Any,
        query_cache_dao: Any,
        history: List[Dict] = None
    ) -> Generator[Dict, None, None]:
        """
        Run the agent loop with Co-Scientist Planning and Active Recall.
        """
        from core.knowledge_store import search_knowledge, format_triples_for_context  # Active Recall
        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=self.api_key)

        # --- Step 0: Active Recall ---
        yield {"type": "reasoning", "content": "Recalling known facts..."}
        known_facts = search_knowledge(question)
        recall_context = ""
        if known_facts:
            recall_context = "\n\nRECALLED KNOWLEDGE:\n" + "\n".join([f"- {f.get('facts', [''])[0]} (Source: {f.get('source')})" for f in known_facts[:3]])
            yield {"type": "reasoning", "content": f"Recalled {len(known_facts)} related facts."}

        # Limit conversation history to prevent token overflow
        history = (history or [])[-10:]  # Keep only last 10 messages
        messages = [
            {"role": "system", "content": self._get_system_prompt(context + recall_context)}
        ]

        for msg in history:
            role = "user" if msg.get("role") == "user" else "assistant"
            # Truncate long messages
            content = msg.get("content", "")[:2000]
            messages.append({"role": role, "content": content})
            
        messages.append({"role": "user", "content": question})
        
        # --- Step 1: Planning ---
        yield {"type": "reasoning", "content": "Formulating research plan..."}
        
        planning_prompt = f"""You are a research planner. 
        Question: {question}
        
        Create a 3-step execution plan to answer this using available tools (search_pubmed).
        Output ONLY the plan as a numbered list."""
        
        plan_resp = client.chat.completions.create(
            model=self.model,
            messages=messages + [{"role": "user", "content": planning_prompt}]
        )
        plan = plan_resp.choices[0].message.content
        yield {"type": "plan", "content": plan}
        
        # Inject plan into context
        messages.append({"role": "system", "content": f"Follow this plan:\n{plan}"})
        
        # --- Step 2: Execution Loop ---
        yield {"type": "reasoning", "content": "Executing plan..."}
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                stream=True
            )
            
            tool_calls = []
            func_call = {"name": None, "arguments": ""}
            current_tool_id = None
            content_accum = "" 
            
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.id:
                            current_tool_id = tc.id
                            func_call = {"name": tc.function.name, "arguments": ""}
                            tool_calls.append({"id": current_tool_id, "function": func_call})
                        elif tc.function.arguments:
                            func_call["arguments"] += tc.function.arguments
                if delta.content:
                    content_accum += delta.content
                    yield {"type": "chunk", "content": delta.content}

            if tool_calls:
                # Limit tool calls to prevent infinite loops
                max_tool_calls = 3
                if len(tool_calls) > max_tool_calls:
                    logger.warning(f"Limiting tool calls from {len(tool_calls)} to {max_tool_calls}")
                    tool_calls = tool_calls[:max_tool_calls]

                assistant_msg = {
                    "role": "assistant",
                    "content": content_accum if content_accum else None,
                    "tool_calls": [
                        {"id": tc["id"], "type": "function", "function": tc["function"]} for tc in tool_calls
                    ]
                }
                messages.append(assistant_msg)

                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    func_args_str = tc["function"]["arguments"]
                    yield {"type": "tool_start", "tool": func_name, "input": func_args_str}
                    yield {"type": "reasoning", "content": f"Executing {func_name}..."}
                    
                    tool_result_content = ""
                    if func_name == "search_pubmed":
                        try:
                            args = json.loads(func_args_str)
                            q = args.get("query")
                            search_res = self.search_service.execute_search(
                                query=q,
                                project_id=project_id,
                                citation_dao=citation_dao,
                                search_dao=search_dao,
                                query_cache_dao=query_cache_dao,
                                max_results=5,
                                ranking_mode="Balanced"
                            )
                            count = search_res["total_count"]
                            citations = search_res["citations"]
                            tool_result_content = f"Found {count} papers. Top {len(citations)}:\n"
                            for i, c in enumerate(citations, 1):
                                tool_result_content += f"{i}. {c.title} (Year: {c.year}, PMID: {c.pmid})\nAbstract: {c.abstract[:200]}...\n\n"
                            yield {"type": "tool_end", "output": f"Found {len(citations)} papers."}
                        except Exception as e:
                            tool_result_content = f"Search failed: {str(e)}"
                            yield {"type": "tool_end", "output": f"Error: {str(e)}"}
                    else:
                        tool_result_content = "Unknown tool."
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result_content
                    })

                yield {"type": "reasoning", "content": "Synthesizing answer..."}
                final_response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True
                )
                for chunk in final_response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield {"type": "chunk", "content": content}

        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            yield {"type": "error", "content": f"An error occurred: {str(e)}"}

    def _get_system_prompt(self, context: str) -> str:
        return f"""You are a Co-Scientist Research Agent.
        
Your goal is to help the user with high-level research tasks.
You have access to tools like `search_pubmed`.

RULES:
1. If the user asks for new information not in the context, USE THE TOOLS.
2. If you use a tool, synthesize the results clearly.
3. ALWAYS cite your sources using [1], [2] format if relevant.
4. Be professional, concise, and rigorous.

CURRENT CONTEXT:
{context}
"""

    def format_chat_export(self, messages: List[Dict], clinical_question: str) -> str:
        # Legacy support
        export = f"# Research Chat: {clinical_question}\n\n"
        for msg in messages:
            role = msg.get("role", "User").capitalize()
            content = msg.get("content", "")
            export += f"**{role}**: {content}\n\n"
        return export


# Backwards compatibility alias
ChatService = ResearchAgent
