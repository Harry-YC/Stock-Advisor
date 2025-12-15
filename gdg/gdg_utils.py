"""
GDG (Guideline Development Group) Utilities for Palliative Surgery

Provides functions for:
- Paper relevance scoring by expert specialty
- Evidence context formatting with epistemic tags
- OpenAI API calls for expert responses
- Discussion export
"""

import logging
import time
from typing import Dict, List, Set, Optional, Any, Callable
from openai import OpenAI, APITimeoutError, RateLimitError, APIConnectionError, APIStatusError
import json
import os

from .gdg_personas import GDG_PERSONAS, get_gdg_prompts, GDG_BASE_CONTEXT, ROUND_INSTRUCTIONS

# Re-export for backward compatibility
try:
    from core.knowledge_extractor import process_discussion_for_knowledge
except ImportError:
    process_discussion_for_knowledge = None

logger = logging.getLogger(__name__)

# API configuration
DEFAULT_TIMEOUT = 120  # seconds
MAX_RETRIES = 5  # Increased for 503 errors
RETRY_DELAY_BASE = 2  # seconds, exponential backoff
RETRYABLE_STATUS_CODES = {503, 429, 500, 502, 504}  # Server errors that may be transient


def _call_with_retry(
    client: OpenAI,
    api_params: Dict,
    max_retries: int = MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT
) -> Any:
    """
    Call OpenAI API with retry logic for transient failures.

    Handles:
    - RateLimitError (429)
    - APITimeoutError
    - APIConnectionError
    - APIStatusError with 503 (model overloaded), 500, 502, 504

    Args:
        client: OpenAI client instance
        api_params: Parameters for chat.completions.create
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds

    Returns:
        API response

    Raises:
        Exception: If all retries exhausted
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            response = client.with_options(timeout=timeout).chat.completions.create(**api_params)
            return response

        except RateLimitError as e:
            last_exception = e
            wait_time = RETRY_DELAY_BASE ** (attempt + 1)
            logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
            time.sleep(wait_time)

        except APITimeoutError as e:
            last_exception = e
            wait_time = RETRY_DELAY_BASE ** attempt
            logger.warning(f"API timeout, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
            time.sleep(wait_time)

        except APIConnectionError as e:
            last_exception = e
            wait_time = RETRY_DELAY_BASE ** attempt
            logger.warning(f"Connection error, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
            time.sleep(wait_time)

        except APIStatusError as e:
            last_exception = e
            if e.status_code == 503:
                wait_time = (RETRY_DELAY_BASE ** (attempt + 1)) * 2 # Wait longer for overloaded
                logger.warning(f"Model overloaded (503), waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
            elif e.status_code in [500, 502, 504]:
                wait_time = RETRY_DELAY_BASE ** (attempt + 1)
                logger.warning(f"Server error ({e.status_code}), waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
            else:
                logger.error(f"Non-retryable API status error ({e.status_code}): {e}")
                raise

        except Exception as e:
            # Check if error message contains retryable indicators
            error_str = str(e).lower()
            if '503' in error_str or 'overloaded' in error_str or 'unavailable' in error_str:
                last_exception = e
                wait_time = (RETRY_DELAY_BASE ** (attempt + 1)) * 2
                logger.warning(f"Transient error detected, waiting {wait_time}s before retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(wait_time)
            else:
                # Non-retryable error
                logger.error(f"Non-retryable API error: {e}")
                raise

    # All retries exhausted
    logger.error(f"All {max_retries} retries exhausted. Last error: {last_exception}")
    raise last_exception


# =============================================================================
# PAPER RELEVANCE SCORING
# =============================================================================

def score_paper_relevance(citation: Dict, expert_role: str) -> float:
    """
    Score a paper's relevance to a specific expert's specialty.

    Uses keyword matching in title and abstract to determine relevance.
    Title matches are weighted higher than abstract matches.

    Args:
        citation: Paper dict with 'title' and 'abstract' keys
        expert_role: Expert name (must match GDG_PERSONAS key)

    Returns:
        Relevance score (0-100)
    """
    if expert_role not in GDG_PERSONAS:
        return 50.0  # Neutral score for unknown expert

    expert = GDG_PERSONAS[expert_role]
    keywords = expert.get('specialty_keywords', [])

    if not keywords:
        return 50.0  # Neutral score if no keywords defined

    title = (citation.get('title') or '').lower()
    abstract = (citation.get('abstract') or '').lower()

    score = 0.0

    for keyword in keywords:
        kw_lower = keyword.lower()
        # Title matches: 15 points
        if kw_lower in title:
            score += 15
        # Abstract matches: 5 points
        if kw_lower in abstract:
            score += 5

    # Cap at 100
    return min(100.0, score)


def auto_select_papers_for_experts(
    citations: List[Dict],
    expert_roles: List[str],
    threshold: float = 30.0
) -> Set[str]:
    """
    Auto-select papers that are relevant to at least one expert.

    Args:
        citations: List of paper dicts with 'pmid' key
        expert_roles: List of expert names to consider
        threshold: Minimum relevance score for selection

    Returns:
        Set of PMIDs to pre-select
    """
    selected = set()

    for citation in citations:
        pmid = citation.get('pmid')
        if not pmid:
            continue

        # Check if paper scores above threshold for any expert
        for expert in expert_roles:
            score = score_paper_relevance(citation, expert)
            if score >= threshold:
                selected.add(pmid)
                break  # No need to check other experts

    return selected


def get_top_papers_for_expert(
    citations: List[Dict],
    expert_role: str,
    max_papers: int = 10
) -> List[Dict]:
    """
    Get the most relevant papers for a specific expert.

    Args:
        citations: List of paper dicts
        expert_role: Expert name
        max_papers: Maximum papers to return

    Returns:
        List of top papers, sorted by relevance
    """
    scored = [(cit, score_paper_relevance(cit, expert_role)) for cit in citations]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [cit for cit, score in scored[:max_papers]]


# =============================================================================
# EVIDENCE CONTEXT FORMATTING
# =============================================================================

def format_evidence_context(
    search_results: Dict,
    persona_role: str,
    max_citations: int = 10,
    priors_text: Optional[str] = None,
    clinical_question: Optional[str] = None
) -> str:
    """
    Format evidence context for an expert's prompt.

    Each expert gets their own personalized evidence context based on
    specialty relevance scoring.

    Args:
        search_results: Dict with 'citations' key containing paper list
        persona_role: Expert name
        max_citations: Maximum citations to include
        priors_text: Optional canonical frameworks text
        clinical_question: Optional research question for context

    Returns:
        Formatted evidence text for prompt
    """
    citations = search_results.get('citations', [])

    if not citations:
        return "**No literature evidence available for this review.**"

    # Score and sort papers for this expert
    scored = [(cit, score_paper_relevance(cit, persona_role)) for cit in citations]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Take top N
    top_citations = [cit for cit, score in scored[:max_citations]]

    # Build context
    output = []

    # Add priors if provided
    if priors_text:
        output.append(priors_text)
        output.append("")

    # Add literature evidence
    output.append("=" * 70)
    output.append("LITERATURE EVIDENCE")
    output.append("=" * 70)
    output.append("")
    output.append(f"**Papers selected for your expertise ({persona_role}):**")
    output.append(f"Showing top {len(top_citations)} of {len(citations)} total papers")
    output.append("")
    output.append("**Key Studies:**")
    output.append("")

    for i, citation in enumerate(top_citations, 1):
        pmid = citation.get('pmid', 'N/A')
        title = citation.get('title', 'No title')
        authors = citation.get('authors', [])
        if isinstance(authors, list):
            first_author = authors[0] if authors else 'Unknown'
        else:
            first_author = authors.split(',')[0] if authors else 'Unknown'

        year = citation.get('year', 'N/A')
        journal = citation.get('journal', 'Unknown journal')
        abstract = citation.get('abstract', '')

        output.append(f"**[{i}] {title}**")
        output.append(f"*{first_author} et al. ({year}) - {journal}*")
        output.append(f"PMID: {pmid}")

        if abstract:
            # Truncate long abstracts
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."
            output.append(f"Abstract: {abstract}")

        output.append("")

    # Add citation guidance
    output.append("=" * 70)
    output.append("CITATION INSTRUCTIONS:")
    output.append("=" * 70)
    output.append("")
    output.append("- Cite papers using: (PMID: XXXXXXXX)")
    output.append("- ONLY cite PMIDs listed above")
    output.append("- Tag claims with [EVIDENCE], [ASSUMPTION], [OPINION], or [EVIDENCE GAP]")
    output.append("")

    return "\n".join(output)


# =============================================================================
# HITL HELPER FUNCTIONS
# =============================================================================

def format_injected_evidence(citations: List[Dict]) -> str:
    """
    Format human-specified citations for injection into expert prompts.

    These are papers the human reviewer wants the expert to specifically address.

    Args:
        citations: List of citation dicts with 'pmid', 'title', 'authors', 'abstract'

    Returns:
        Formatted string to prepend to evidence context
    """
    if not citations:
        return ""

    lines = [
        "",
        "=" * 60,
        "HUMAN-SPECIFIED EVIDENCE (you MUST address these papers)",
        "=" * 60,
        ""
    ]

    for i, c in enumerate(citations, 1):
        pmid = c.get('pmid', 'Unknown')
        title = c.get('title', 'Untitled')
        authors = c.get('authors', [])
        abstract = c.get('abstract', '')

        # Format authors
        if isinstance(authors, list):
            author_str = ', '.join(authors[:3])
            if len(authors) > 3:
                author_str += ' et al.'
        else:
            author_str = str(authors)

        # Truncate abstract
        if len(abstract) > 1500:
            abstract = abstract[:1500] + "..."

        lines.extend([
            f"[{i}] **PMID: {pmid}** - {title}",
            f"    Authors: {author_str}",
            f"    Abstract: {abstract}",
            ""
        ])

    lines.extend([
        "=" * 60,
        "You MUST explicitly discuss each paper above in your response.",
        "Cite them using (PMID: XXXXXXXX) format.",
        "=" * 60,
        ""
    ])

    return "\n".join(lines)


def format_rag_context(rag_results: List[Dict]) -> str:
    """
    Format RAG retrieval results for injection into expert prompts.

    These are document chunks retrieved from uploaded reference documents.

    Args:
        rag_results: List of retrieval results with 'content', 'source', 'score'

    Returns:
        Formatted string to prepend to evidence context
    """
    if not rag_results:
        return ""

    lines = [
        "",
        "=" * 60,
        "REFERENCE DOCUMENTS (retrieved from uploaded files)",
        "=" * 60,
        ""
    ]

    for i, result in enumerate(rag_results, 1):
        source = result.get('source', 'Unknown')
        content = result.get('content', '')
        score = result.get('score', 0)
        rerank_score = result.get('rerank_score')

        # Truncate content if too long
        if len(content) > 2000:
            content = content[:2000] + "..."

        # Format score
        score_str = f"relevance: {score:.2f}"
        if rerank_score:
            score_str = f"relevance: {rerank_score:.2f}"

        lines.extend([
            f"[{i}] Source: {source} ({score_str})",
            f"    {content}",
            ""
        ])

    lines.extend([
        "=" * 60,
        "Use the above reference material to inform your response.",
        "Cite sources using [Source: filename] format where applicable.",
        "=" * 60,
        ""
    ])

    return "\n".join(lines)


# =============================================================================
# OPENAI API CALLS
# =============================================================================

def call_expert(
    persona_name: str,
    clinical_question: str,
    evidence_context: str,
    round_num: int = 1,
    previous_responses: Optional[Dict[str, str]] = None,
    priors_text: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-5-mini",
    max_completion_tokens: int = 4096,
    timeout: float = DEFAULT_TIMEOUT,
    # HITL Enhancement parameters
    rejection_critique: Optional[str] = None,
    injected_evidence: Optional[List[Dict]] = None,
    temperature: Optional[float] = None,
    working_memory: Optional[Any] = None,
    # RAG context from local documents
    rag_context: Optional[List[Dict]] = None,
    # System instruction override for custom prompts (e.g., Pass 1 two-pass mode)
    system_instruction_override: Optional[str] = None,
    # Project knowledge injection parameters
    project_id: Optional[int] = None,
    question_type: Optional[str] = None,
    extracted_entities: Optional[List[str]] = None,
    # Database for SQLite-based corrections
    db = None
) -> Dict[str, Any]:
    """
    Call OpenAI API to generate an expert response with retry logic.

    Args:
        persona_name: Expert name
        clinical_question: Research question
        evidence_context: Formatted evidence for expert
        round_num: Discussion round (1-4)
        previous_responses: Dict of previous expert responses
        priors_text: Optional canonical frameworks
        openai_api_key: OpenAI API key (uses env var if not provided)
        model: Model name (default: gpt-5-mini)
        max_completion_tokens: Max tokens for response
        timeout: API request timeout in seconds
        rejection_critique: If regenerating, why the previous response was rejected
        injected_evidence: List of citation dicts to force-include in prompt
        temperature: Optional temperature (only for models that support it)
        working_memory: WorkingMemory instance for persistent constraints
        rag_context: List of retrieved document chunks for RAG context
        system_instruction_override: If provided, replaces the default system prompt entirely
            (used for Pass 1 in two-pass mode, debate rounds, STORM workflow, etc.)
        project_id: Optional project ID for injecting project-specific learned knowledge
        question_type: Optional question type for query effectiveness hints
        extracted_entities: Optional list of entities extracted from question for knowledge lookup

    Returns:
        Dict with keys:
        - 'content': Response text
        - 'finish_reason': API finish reason
        - 'model': Model used
        - 'tokens': Token usage dict
        - 'raw_response': Full API response (for debugging)
    """
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        logger.error("No OpenAI API key provided")
        return {
            'content': "Error: No OpenAI API key provided",
            'finish_reason': 'error',
            'model': model,
            'tokens': {},
            'raw_response': None
        }

    # Get prompts for this persona
    all_prompts = get_gdg_prompts(bullets_per_role=5)
    if persona_name not in all_prompts:
        logger.error(f"Unknown persona: {persona_name}")
        return {
            'content': f"Error: Unknown persona '{persona_name}'",
            'finish_reason': 'error',
            'model': model,
            'tokens': {},
            'raw_response': None
        }

    context, task = all_prompts[persona_name]

    # Build system prompt from context
    system_prompt_base = context + "\n\n" + task

    # Add round-specific instructions if applicable
    if round_num > 1 and round_num in ROUND_INSTRUCTIONS:
        round_info = ROUND_INSTRUCTIONS[round_num]
        system_prompt_base += f"\n\n## ROUND {round_num}: {round_info['name']}\n{round_info['instruction']}"

    # Add previous responses context for later rounds
    if previous_responses and round_num > 1:
        prev_context = "\n\n## PREVIOUS EXPERT RESPONSES:\n"
        for expert, resp in previous_responses.items():
            truncated = resp[:800] if len(resp) > 800 else resp
            prev_context += f"\n**{expert}:**\n{truncated}\n"
        system_prompt_base += prev_context

    # Use override if provided, otherwise use default system prompt
    if system_instruction_override:
        system_prompt = system_instruction_override
    else:
        system_prompt = system_prompt_base

    # Inject working memory (persistent constraints/facts)
    if working_memory:
        memory_context = working_memory.format_for_prompt()
        if memory_context:
            system_prompt += memory_context

    # Inject project knowledge (learned insights from user feedback and prior sessions)
    if project_id or question_type or extracted_entities:
        try:
            from core.knowledge_store import (
                get_default_store,
                format_learning_for_prompt,
            )
            store = get_default_store()

            # Inject learned insights (corrections, effective queries)
            learning_context = format_learning_for_prompt(
                persona=persona_name,
                question_type=question_type,
                project_id=project_id
            )
            if learning_context:
                system_prompt += "\n\n" + learning_context

            # Inject prior knowledge (facts and triples from previous sessions)
            knowledge_context = store.format_knowledge_for_prompt(
                query=clinical_question,
                persona=persona_name,
                entities=extracted_entities,
                max_facts=5
            )
            if knowledge_context:
                system_prompt += "\n\n" + knowledge_context

            logger.debug(f"Injected project knowledge for {persona_name}")
        except Exception as e:
            logger.warning(f"Failed to inject project knowledge: {e}")

    # Inject SQLite-based corrections (from ExpertCorrectionDAO)
    if db and project_id:
        try:
            from core.database import ExpertCorrectionDAO
            correction_dao = ExpertCorrectionDAO(db)
            correction_context = correction_dao.format_for_prompt(
                expert_name=persona_name,
                project_id=project_id,
                limit=3
            )
            if correction_context:
                system_prompt += "\n\n" + correction_context
                logger.debug(f"Injected SQLite corrections for {persona_name}")
        except Exception as e:
            logger.warning(f"Failed to inject SQLite corrections: {e}")

    # Inject rejection critique (for regeneration)
    if rejection_critique:
        system_prompt += f"""

============================================================
IMPORTANT - PREVIOUS RESPONSE REJECTED
============================================================
Your previous response was rejected for the following reason:

{rejection_critique}

You MUST address this feedback in your new response. Do NOT repeat the same mistakes.
============================================================
"""

    # Build user message with clinical question and evidence
    user_message = f"Clinical Question: {clinical_question}\n\n{evidence_context}"

    # Inject human-specified evidence (must address these papers)
    if injected_evidence:
        injected_text = format_injected_evidence(injected_evidence)
        user_message = injected_text + "\n\n" + user_message

    # Inject RAG context from local documents
    if rag_context:
        rag_text = format_rag_context(rag_context)
        user_message = rag_text + "\n\n" + user_message

    try:
        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=openai_api_key, model=model)

        # Build API params
        api_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_completion_tokens": max_completion_tokens
        }

        # Add temperature if provided (model-aware)
        if temperature is not None:
            api_params["temperature"] = temperature

        logger.info(f"Calling expert {persona_name} (round {round_num}) with model {model}")

        # Use retry wrapper
        response = _call_with_retry(client, api_params, timeout=timeout)

        # Extract response
        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason

        # Get token usage
        usage = response.usage
        tokens = {
            'prompt_tokens': usage.prompt_tokens if usage else 0,
            'completion_tokens': usage.completion_tokens if usage else 0,
            'total_tokens': usage.total_tokens if usage else 0
        }

        # Check for reasoning tokens (available in some models)
        if hasattr(usage, 'completion_tokens_details'):
            details = usage.completion_tokens_details
            if hasattr(details, 'reasoning_tokens'):
                tokens['reasoning_tokens'] = details.reasoning_tokens

        logger.info(f"Expert {persona_name} response complete: {tokens.get('total_tokens', 0)} tokens")

        return {
            'content': content,
            'finish_reason': finish_reason,
            'model': model,
            'tokens': tokens,
            'raw_response': response
        }

    except Exception as e:
        logger.error(f"Expert API call failed for {persona_name}: {e}", exc_info=True)
        return {
            'content': f"Error calling API: {str(e)}",
            'finish_reason': 'error',
            'model': model,
            'tokens': {},
            'raw_response': None
        }


def call_expert_stream(
    persona_name: str,
    clinical_question: str,
    evidence_context: str,
    round_num: int = 1,
    previous_responses: Optional[Dict[str, str]] = None,
    priors_text: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-5-mini",
    max_completion_tokens: int = 4000,
    timeout: float = DEFAULT_TIMEOUT
):
    """
    Stream an expert response for real-time display with error recovery.

    Yields dicts with:
    - {"type": "chunk", "content": "..."} for text chunks
    - {"type": "error", "content": "..."} for errors
    - {"type": "complete", "finish_reason": "...", "model": "..."} when done

    Args:
        persona_name: Expert name
        clinical_question: Research question
        evidence_context: Formatted evidence for expert
        round_num: Discussion round (1-4)
        previous_responses: Dict of previous expert responses
        priors_text: Optional canonical frameworks
        openai_api_key: OpenAI API key
        model: Model name (recommend gpt-5-mini for streaming)
        max_completion_tokens: Max tokens for response
        timeout: API request timeout in seconds

    Yields:
        Dict with type and content
    """
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        logger.error("No OpenAI API key provided for streaming")
        yield {'type': 'error', 'content': "Error: No OpenAI API key provided"}
        yield {'type': 'complete', 'finish_reason': 'error', 'model': model}
        return

    # Get prompts for this persona
    all_prompts = get_gdg_prompts(bullets_per_role=5)
    if persona_name not in all_prompts:
        logger.error(f"Unknown persona: {persona_name}")
        yield {'type': 'error', 'content': f"Error: Unknown persona '{persona_name}'"}
        yield {'type': 'complete', 'finish_reason': 'error', 'model': model}
        return

    context, task = all_prompts[persona_name]

    # Build system prompt from context
    system_prompt = context + "\n\n" + task

    # Add round-specific instructions if applicable
    if round_num > 1 and round_num in ROUND_INSTRUCTIONS:
        round_info = ROUND_INSTRUCTIONS[round_num]
        system_prompt += f"\n\n## ROUND {round_num}: {round_info['name']}\n{round_info['instruction']}"

    # Add previous responses context for later rounds
    if previous_responses and round_num > 1:
        prev_context = "\n\n## PREVIOUS EXPERT RESPONSES:\n"
        for expert, resp in previous_responses.items():
            truncated = resp[:800] if len(resp) > 800 else resp
            prev_context += f"\n**{expert}:**\n{truncated}\n"
        system_prompt += prev_context

    # Build user message with clinical question and evidence
    user_message = f"Clinical Question: {clinical_question}\n\n{evidence_context}"

    try:
        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=openai_api_key, model=model)

        logger.info(f"Starting stream for expert {persona_name}")

        # Use streaming with timeout
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_completion_tokens=max_completion_tokens,
            stream=True
        )

        chunk_count = 0
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunk_count += 1
                yield {'type': 'chunk', 'content': chunk.choices[0].delta.content}

            # Check for finish
            if chunk.choices and chunk.choices[0].finish_reason:
                logger.info(f"Stream complete for {persona_name}: {chunk_count} chunks")
                yield {
                    'type': 'complete',
                    'finish_reason': chunk.choices[0].finish_reason,
                    'model': model
                }
                return  # Explicit termination

        # If we exit the loop without a finish_reason, something went wrong
        logger.warning(f"Stream ended without finish_reason for {persona_name}")
        yield {'type': 'complete', 'finish_reason': 'unknown', 'model': model}

    except APITimeoutError as e:
        logger.error(f"Stream timeout for {persona_name}: {e}")
        yield {'type': 'error', 'content': f"Request timed out after {timeout}s. Please try again."}
        yield {'type': 'complete', 'finish_reason': 'timeout', 'model': model}

    except RateLimitError as e:
        logger.error(f"Rate limit for {persona_name}: {e}")
        yield {'type': 'error', 'content': "Rate limit exceeded. Please wait a moment and try again."}
        yield {'type': 'complete', 'finish_reason': 'rate_limit', 'model': model}

    except APIConnectionError as e:
        logger.error(f"Connection error for {persona_name}: {e}")
        yield {'type': 'error', 'content': "Connection error. Please check your internet and try again."}
        yield {'type': 'complete', 'finish_reason': 'connection_error', 'model': model}

    except Exception as e:
        logger.error(f"Unexpected error streaming {persona_name}: {e}", exc_info=True)
        yield {'type': 'error', 'content': f"Error: {str(e)}"}
        yield {'type': 'complete', 'finish_reason': 'error', 'model': model}


def call_expert_batch(
    expert_names: List[str],
    clinical_question: str,
    search_results: Dict,
    round_num: int = 1,
    previous_responses: Optional[Dict[str, str]] = None,
    priors_manager: Optional[Any] = None,
    scenario: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-5-mini",
    max_citations: int = 10,
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Call multiple experts in sequence.

    Args:
        expert_names: List of expert names to call
        clinical_question: Research question
        search_results: Dict with 'citations' key
        round_num: Discussion round
        previous_responses: Previous round responses
        priors_manager: Optional PriorsManager instance
        scenario: Drug development scenario for priors selection
        openai_api_key: OpenAI API key
        model: Model name
        max_citations: Max papers per expert
        progress_callback: Optional callback(expert_name, current, total)

    Returns:
        Dict mapping expert_name -> response dict
    """
    results = {}
    total = len(expert_names)

    for i, expert_name in enumerate(expert_names):
        # Format priors if manager provided
        priors_text = None
        if priors_manager:
            priors_text = priors_manager.format_priors_for_context(
                scenario=scenario,
                persona=expert_name,
                max_frameworks=3,
                compressed=(round_num > 1)
            )

        # Format evidence context for this expert
        evidence_context = format_evidence_context(
            search_results=search_results,
            persona_role=expert_name,
            max_citations=max_citations,
            priors_text=None,  # Priors added separately
            clinical_question=clinical_question
        )

        # Call expert
        response = call_expert(
            persona_name=expert_name,
            clinical_question=clinical_question,
            evidence_context=evidence_context,
            round_num=round_num,
            previous_responses=previous_responses,
            priors_text=priors_text,
            openai_api_key=openai_api_key,
            model=model
        )

        results[expert_name] = response

        # Progress callback
        if progress_callback:
            progress_callback(expert_name, i + 1, total)
            
        # Add delay between experts to avoid rate limits
        if i < total - 1:
            time.sleep(2.0)

    return results


# =============================================================================
# EXPORT UTILITIES
# =============================================================================

def export_discussion_to_markdown(
    clinical_question: str,
    expert_responses: Dict[str, List[Dict]],
    citations: List[Dict],
    scenario: Optional[str] = None,
    include_citations: bool = True
) -> str:
    """
    Export expert discussion to markdown format.

    Args:
        clinical_question: Research question
        expert_responses: Dict[expert_name] -> List of round responses
        citations: List of paper dicts used in discussion
        scenario: Drug development scenario
        include_citations: Include bibliography

    Returns:
        Markdown formatted string
    """
    output = []

    # Header
    output.append("# Expert Panel Discussion")
    output.append("")
    output.append(f"**Research Question:** {clinical_question}")
    if scenario:
        output.append(f"**Scenario:** {scenario}")
    output.append("")
    output.append("---")
    output.append("")

    # Expert responses by round
    max_rounds = max(len(responses) for responses in expert_responses.values())

    for round_num in range(1, max_rounds + 1):
        output.append(f"## Round {round_num}")
        output.append("")

        for expert_name, responses in expert_responses.items():
            if len(responses) >= round_num:
                response = responses[round_num - 1]
                content = response.get('content', 'No response')

                output.append(f"### {expert_name}")
                output.append("")
                output.append(content)
                output.append("")

        output.append("---")
        output.append("")

    # Bibliography
    if include_citations and citations:
        output.append("## References")
        output.append("")

        for i, cit in enumerate(citations, 1):
            pmid = cit.get('pmid', 'N/A')
            title = cit.get('title', 'No title')
            authors = cit.get('authors', [])
            if isinstance(authors, list):
                author_str = ', '.join(authors[:3])
                if len(authors) > 3:
                    author_str += ' et al.'
            else:
                author_str = authors

            year = cit.get('year', 'N/A')
            journal = cit.get('journal', '')

            output.append(f"{i}. {author_str} ({year}). {title}. *{journal}*. PMID: {pmid}")
            output.append("")

    return "\n".join(output)


def extract_cited_pmids(text: str) -> Set[str]:
    """
    Extract PMIDs cited in text.

    Looks for patterns like (PMID: 12345678) or PMID:12345678

    Args:
        text: Text to search

    Returns:
        Set of PMID strings
    """
    import re

    # Match PMID: followed by digits
    pattern = r'PMID[:\s]*(\d+)'
    matches = re.findall(pattern, text, re.IGNORECASE)

    return set(matches)


# =============================================================================
# META-REVIEW SYNTHESIS
# =============================================================================

# Domain groupings for GDG synthesis - experts within same group can be compared
EXPERT_DOMAINS = {
    "Surgical & Interventional": ["Surgical Oncologist", "Perioperative Medicine Physician", "Interventionalist"],
    "Palliative & Patient Care": ["Palliative Care Physician", "Patient Advocate", "Pain and Symptom-Control Specialist"],
    "Evidence & Methodology": ["GRADE Methodologist", "Clinical Evidence Specialist"],
    "Ethics & Values": ["Medical Ethicist", "Patient Advocate"],
    "Special Populations": ["Geriatric and Frailty Specialist", "Perioperative Medicine Physician"],
    "Health Systems": ["Health Economist", "GDG Chair"],
}


def get_expert_domain(expert_name: str) -> str:
    """Get primary domain for an expert."""
    for domain, experts in EXPERT_DOMAINS.items():
        if expert_name in experts:
            return domain
    return "General"


def synthesize_expert_responses(
    expert_responses: Dict[str, str],
    clinical_question: str,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-5-mini"
) -> Dict[str, Any]:
    """
    Synthesize expert panel responses into an actionable summary.

    Unlike debate, this RESPECTS domain boundaries - it consolidates
    insights by domain rather than having experts argue about things
    outside their expertise.

    Args:
        expert_responses: Dict[expert_name] -> response content
        clinical_question: The research question
        openai_api_key: OpenAI API key
        model: Model to use

    Returns:
        Dict with:
        - 'synthesis': Overall synthesis text
        - 'by_domain': Dict[domain] -> key findings
        - 'consensus_points': List of agreed points
        - 'open_questions': List of unresolved questions
        - 'recommended_actions': Prioritized next steps
    """
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key or not expert_responses:
        return {
            'synthesis': "Unable to synthesize - no API key or responses",
            'by_domain': {},
            'consensus_points': [],
            'open_questions': [],
            'recommended_actions': []
        }

    # Group responses by domain
    domain_responses = {}
    for expert, response in expert_responses.items():
        domain = get_expert_domain(expert)
        if domain not in domain_responses:
            domain_responses[domain] = {}
        domain_responses[domain][expert] = response

    # Format for prompt
    formatted_responses = []
    for domain, experts in domain_responses.items():
        formatted_responses.append(f"\n## {domain} Domain\n")
        for expert, response in experts.items():
            # Truncate long responses
            truncated = response[:2000] + "..." if len(response) > 2000 else response
            formatted_responses.append(f"### {expert}\n{truncated}\n")

    try:
        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=openai_api_key)

        system_prompt = """You are a Meta-Review Agent synthesizing a Guideline Development Group (GDG) discussion on palliative surgery.

Your role is NOT to debate or judge experts against each other - each expert has domain expertise that others lack.

Instead, you should:
1. Extract key findings FROM EACH DOMAIN (Surgical, Palliative Care, Evidence/Methodology, Ethics, Special Populations, Health Systems)
2. Identify points of CONSENSUS across experts
3. Note OPEN QUESTIONS that need more investigation
4. Recommend PRIORITIZED ACTIONS based on the collective input

Be specific and actionable. Use evidence tags [EVIDENCE (PMID:)], [ASSUMPTION], [OPINION], [EVIDENCE GAP] from the expert responses."""

        user_prompt = f"""Research Question: {clinical_question}

Expert Panel Responses (grouped by domain):
{''.join(formatted_responses)}

Provide a synthesis with these sections:

## Executive Summary
[2-3 sentence overview]

## Key Findings by Domain
[For each domain with experts, 2-3 bullet points of key findings]

## Points of Consensus
[List areas where multiple domains agree]

## Open Questions
[List unresolved questions that need investigation]

## Recommended Actions
[Prioritized list of next steps, numbered 1-5]"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )

        synthesis_text = response.choices[0].message.content or ""

        # Parse structured output
        result = {
            'synthesis': synthesis_text,
            'by_domain': {},
            'consensus_points': [],
            'open_questions': [],
            'recommended_actions': []
        }

        # Extract sections (basic parsing)
        import re

        # Extract domain findings
        for domain in EXPERT_DOMAINS.keys():
            if domain in synthesis_text:
                result['by_domain'][domain] = f"See synthesis for {domain} findings"

        # Extract consensus points
        consensus_match = re.search(r'## Points of Consensus\n(.*?)(?=\n## |$)', synthesis_text, re.DOTALL)
        if consensus_match:
            points = re.findall(r'[-•]\s*(.+)', consensus_match.group(1))
            result['consensus_points'] = points[:5]

        # Extract open questions
        questions_match = re.search(r'## Open Questions\n(.*?)(?=\n## |$)', synthesis_text, re.DOTALL)
        if questions_match:
            questions = re.findall(r'[-•]\s*(.+)', questions_match.group(1))
            result['open_questions'] = questions[:5]

        # Extract recommended actions
        actions_match = re.search(r'## Recommended Actions\n(.*?)(?=\n## |$)', synthesis_text, re.DOTALL)
        if actions_match:
            actions = re.findall(r'\d+[.)]\s*(.+)', actions_match.group(1))
            result['recommended_actions'] = actions[:5]

        return result

    except Exception as e:
        return {
            'synthesis': f"Synthesis error: {str(e)}",
            'by_domain': {},
            'consensus_points': [],
            'open_questions': [],
            'recommended_actions': []
        }


def generate_perspective_questions(
    clinical_question: str,
    expert_names: List[str],
    openai_api_key: Optional[str] = None,
    model: str = "gpt-5-mini"
) -> Dict[str, List[str]]:
    """
    Generate domain-specific questions each expert would want answered.

    Inspired by Stanford STORM's perspective-guided questioning.
    These questions can drive the literature search or expert discussion.

    Args:
        clinical_question: The main research question
        expert_names: List of expert personas to generate questions for
        openai_api_key: OpenAI API key
        model: Model to use

    Returns:
        Dict[expert_name] -> List of 3 questions from that perspective
    """
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        return {}

    result = {}

    try:
        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=openai_api_key)

        # Batch all experts into one call for efficiency
        expert_descriptions = []
        for expert in expert_names:
            if expert in GDG_PERSONAS:
                desc = GDG_PERSONAS[expert].get('perspective', expert)
                expert_descriptions.append(f"- {expert}: {desc}")

        system_prompt = """You generate domain-specific questions for Guideline Development Group (GDG) experts on palliative surgery.
Each expert has different concerns based on their specialty.
Generate exactly 3 questions per expert that they would want answered before providing their assessment."""

        user_prompt = f"""Research Question: {clinical_question}

Experts and their focus areas:
{chr(10).join(expert_descriptions)}

For each expert, generate 3 specific questions from their domain perspective.
Format as:
EXPERT_NAME:
1. Question 1
2. Question 2
3. Question 3"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1500,
            temperature=0.5
        )

        content = response.choices[0].message.content or ""

        # Parse response
        current_expert = None
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Check if this is an expert header
            for expert in expert_names:
                if line.startswith(expert) or expert.lower() in line.lower():
                    if ':' in line:
                        current_expert = expert
                        result[expert] = []
                        break

            # Check if this is a question
            if current_expert and re.match(r'^\d+[.)]\s*', line):
                question = re.sub(r'^\d+[.)]\s*', '', line)
                if question and len(result.get(current_expert, [])) < 3:
                    result[current_expert].append(question)

        return result

    except Exception as e:
        print(f"Error generating perspective questions: {e}")
        return {}


def generate_followup_questions(
    clinical_question: str,
    expert_responses: Dict[str, str],
    openai_api_key: Optional[str] = None,
    model: str = "gpt-5-mini",
    max_questions: int = 4
) -> List[str]:
    """
    Generate follow-up questions based on expert discussion.

    These are meant to drive the next round of discussion or
    identify gaps that need more investigation.

    Args:
        clinical_question: Original research question
        expert_responses: Dict[expert_name] -> response content
        openai_api_key: OpenAI API key
        model: Model to use
        max_questions: Maximum questions to return

    Returns:
        List of follow-up question strings
    """
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key or not expert_responses:
        return []

    # Combine responses (truncated)
    combined = ""
    for expert, response in expert_responses.items():
        truncated = response[:800] if len(response) > 800 else response
        combined += f"\n{expert}: {truncated}\n"

    try:
        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=openai_api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """Generate follow-up questions for a palliative surgery Guideline Development Group (GDG) panel.
Focus on:
- Gaps in evidence identified by experts
- Areas of uncertainty or disagreement
- Patient outcomes and quality of life considerations
- Specific data that would strengthen guideline recommendations

Return 4 concise questions, one per line, no numbering."""
                },
                {
                    "role": "user",
                    "content": f"Original question: {clinical_question}\n\nExpert responses:\n{combined[:4000]}"
                }
            ],
            max_tokens=300,
            temperature=0.5
        )

        content = response.choices[0].message.content or ""
        questions = [q.strip() for q in content.split('\n') if q.strip() and len(q.strip()) > 15]

        return questions[:max_questions]

    except Exception as e:
        print(f"Error generating follow-up questions: {e}")
        return []


def extract_hypotheses_from_discussion(
    expert_responses: Dict[str, str],
    clinical_question: str,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-5-mini"
) -> List[Dict[str, Any]]:
    """
    Extract key hypotheses/claims from expert discussion.

    Inspired by Google's AI Co-Scientist approach to tracking and
    evolving hypotheses across discussion rounds.

    Args:
        expert_responses: Dict[expert_name] -> response content
        clinical_question: The research question
        openai_api_key: OpenAI API key
        model: Model to use

    Returns:
        List of hypothesis dicts with:
        - 'hypothesis': The claim/hypothesis text
        - 'evidence_strength': 1-5 rating
        - 'supporting_experts': List of expert names
        - 'evidence_type': EVIDENCE/ASSUMPTION/OPINION
        - 'key_data': Any quantitative data cited
    """
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key or not expert_responses:
        return []

    # Combine responses
    combined = ""
    for expert, response in expert_responses.items():
        truncated = response[:1500] if len(response) > 1500 else response
        combined += f"\n### {expert}\n{truncated}\n"

    try:
        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=openai_api_key)

        system_prompt = """You extract key hypotheses/claims from palliative surgery GDG expert discussions.

For each major claim, extract:
1. The hypothesis (concise statement)
2. Evidence strength (1-5, where 5 = strong RCT/meta-analysis data, 1 = expert opinion)
3. Which experts support it
4. Evidence type: EVIDENCE (cited data), ASSUMPTION, or OPINION
5. Key data points (mortality rates, symptom relief %, survival, etc.)

Format as:
HYPOTHESIS: [statement]
STRENGTH: [1-5]
SUPPORTERS: [expert names]
TYPE: [EVIDENCE/ASSUMPTION/OPINION]
DATA: [key numbers or "none"]
---

Extract 3-5 key hypotheses. Focus on actionable claims about palliative interventions and patient outcomes."""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Research question: {clinical_question}\n\nExpert responses:\n{combined[:6000]}"}
            ],
            max_tokens=1000,
            temperature=0.3
        )

        content = response.choices[0].message.content or ""

        # Parse response
        hypotheses = []
        current = {}

        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith('HYPOTHESIS:'):
                if current.get('hypothesis'):
                    hypotheses.append(current)
                current = {'hypothesis': line.replace('HYPOTHESIS:', '').strip()}
            elif line.startswith('STRENGTH:'):
                try:
                    current['evidence_strength'] = int(line.replace('STRENGTH:', '').strip())
                except ValueError:
                    current['evidence_strength'] = 3
            elif line.startswith('SUPPORTERS:'):
                supporters = line.replace('SUPPORTERS:', '').strip()
                current['supporting_experts'] = [s.strip() for s in supporters.split(',')]
            elif line.startswith('TYPE:'):
                current['evidence_type'] = line.replace('TYPE:', '').strip()
            elif line.startswith('DATA:'):
                current['key_data'] = line.replace('DATA:', '').strip()
            elif line == '---':
                if current.get('hypothesis'):
                    hypotheses.append(current)
                current = {}

        # Don't forget last one
        if current.get('hypothesis'):
            hypotheses.append(current)

        return hypotheses[:5]

    except Exception as e:
        print(f"Error extracting hypotheses: {e}")
        return []
