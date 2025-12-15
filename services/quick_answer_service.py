"""
Quick Answer Service for Palliative Surgery GDG

Provides instant answers (<5s) to clinical questions using a single LLM call
with PubMed search context. This is the "fast path" alternative to running
a full GDG expert panel discussion.

Usage:
    from services.quick_answer_service import get_quick_answer, get_quick_answer_with_search

    # With pre-fetched context
    answer = get_quick_answer(question, citations_context)

    # With automatic PubMed search
    answer = get_quick_answer_with_search(question, scenario="Malignant bowel obstruction")
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class QuickAnswer:
    """Response from quick answer service."""
    answer: str
    sources_used: int
    model: str
    has_context: bool
    citations: List[Dict] = None  # List of {pmid, title, authors}

    def __post_init__(self):
        if self.citations is None:
            self.citations = []


def get_quick_answer(
    question: str,
    context: List[Dict],
    scenario: str = "",
    max_sources: int = 5
) -> QuickAnswer:
    """
    Generate instant answer to a clinical question with citations.

    Args:
        question: The clinical question
        context: List of context dicts with 'pmid', 'title', 'abstract'
        scenario: Optional clinical scenario context
        max_sources: Maximum sources to include in prompt

    Returns:
        QuickAnswer with answer text and metadata
    """
    from services.llm_router import LLMRouter

    # Format context for prompt
    context_items = context[:max_sources] if context else []
    has_context = len(context_items) > 0

    context_text = ""
    citations = []
    if context_items:
        lines = []
        for i, c in enumerate(context_items, 1):
            pmid = c.get('pmid', '')
            title = c.get('title', 'Source')[:100]
            abstract = c.get('abstract', c.get('content', ''))[:400]
            authors = c.get('authors', [])
            author_str = ', '.join(authors[:3]) + (' et al.' if len(authors) > 3 else '') if authors else ''

            lines.append(f"[{i}] {title}")
            if author_str:
                lines.append(f"    Authors: {author_str}")
            if pmid:
                lines.append(f"    PMID: {pmid}")
            lines.append(f"    {abstract}")
            lines.append("")

            citations.append({
                'pmid': pmid,
                'title': title,
                'authors': authors[:3] if authors else []
            })

        context_text = "\n".join(lines)

    # Build prompt
    scenario_context = f"\nClinical Scenario: {scenario}" if scenario else ""

    if has_context:
        prompt = f"""You are an expert palliative surgery consultant. Answer the clinical question concisely and accurately using the provided evidence sources.
{scenario_context}

**Evidence Sources:**
{context_text}

**Question:** {question}

**Instructions:**
- Provide a direct, evidence-based answer (2-3 paragraphs max)
- Cite sources using [PMID:XXXXXXXX] format when referencing specific data
- Also use [1], [2] format to reference the numbered sources above
- Highlight key outcome data (survival, QoL scores, complication rates)
- Note evidence quality limitations (study design, sample size)
- Address risk-benefit considerations for palliative intent procedures
- Be specific to the question asked

**Answer:**"""
    else:
        prompt = f"""You are an expert palliative surgery consultant. Answer the following clinical question based on your knowledge.
{scenario_context}

**Question:** {question}

**Instructions:**
- Provide a direct answer (2-3 paragraphs max)
- Highlight key clinical considerations
- Clearly note that this is based on general knowledge without specific literature citations
- Recommend consulting literature for specific data points
- Be specific to the question asked

**Answer:**"""

    # Get LLM router
    model = getattr(settings, 'QUICK_ANSWER_MODEL', getattr(settings, 'EXPERT_MODEL', 'gemini-3-pro-preview'))
    llm = LLMRouter()

    try:
        answer = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.3,  # Lower temp for factual responses
            max_tokens=1000
        )

        return QuickAnswer(
            answer=answer.strip(),
            sources_used=len(context_items),
            model=model,
            has_context=has_context,
            citations=citations
        )

    except Exception as e:
        logger.error(f"Quick answer generation failed: {e}")
        return QuickAnswer(
            answer=f"Unable to generate answer: {str(e)}",
            sources_used=0,
            model=model,
            has_context=False,
            citations=[]
        )


def get_quick_answer_with_search(
    question: str,
    scenario: str = "",
    max_results: int = 5,
    pubmed_api_key: str = None,
    openai_api_key: str = None,
) -> QuickAnswer:
    """
    Generate instant answer with automatic PubMed search.

    Args:
        question: The clinical question
        scenario: Optional clinical scenario context (used to refine search)
        max_results: Maximum search results to use
        pubmed_api_key: Optional PubMed API key
        openai_api_key: Optional OpenAI API key for query optimization

    Returns:
        QuickAnswer with answer and search context
    """
    context = []

    # Build search query
    search_query = question
    if scenario:
        # Extract key terms from scenario for search
        search_query = f"{scenario} {question}"

    # PubMed search
    try:
        from core.pubmed_client import search_pubmed

        # Use API key from settings if not provided
        api_key = pubmed_api_key or getattr(settings, 'PUBMED_API_KEY', None)

        results = search_pubmed(
            query=search_query,
            max_results=max_results,
            api_key=api_key
        )

        for citation in results:
            context.append({
                'pmid': citation.pmid,
                'title': citation.title,
                'abstract': citation.abstract or '',
                'authors': citation.authors if hasattr(citation, 'authors') else [],
                'year': getattr(citation, 'year', getattr(citation, 'pub_date', '')),
                'journal': getattr(citation, 'journal', ''),
            })

        logger.info(f"Quick answer search found {len(context)} PubMed results")

    except Exception as e:
        logger.warning(f"PubMed search failed: {e}")

    # Fallback: try semantic scholar if PubMed fails
    if not context:
        try:
            from integrations.semantic_scholar import search_papers

            ss_results = search_papers(search_query, limit=max_results)
            for paper in ss_results:
                context.append({
                    'pmid': paper.get('paperId', ''),
                    'title': paper.get('title', ''),
                    'abstract': paper.get('abstract', ''),
                    'authors': [a.get('name', '') for a in paper.get('authors', [])][:3],
                    'year': paper.get('year', ''),
                })

            logger.info(f"Quick answer fallback found {len(context)} Semantic Scholar results")

        except Exception as e:
            logger.warning(f"Semantic Scholar fallback also failed: {e}")

    # Generate answer
    return get_quick_answer(
        question=question,
        context=context,
        scenario=scenario,
        max_sources=max_results
    )


def render_quick_answer_ui(question: str, scenario: str = "") -> Optional[QuickAnswer]:
    """
    Streamlit UI component for quick answer with search.

    Args:
        question: Clinical question
        scenario: Clinical scenario

    Returns:
        QuickAnswer if generated, None otherwise
    """
    import streamlit as st

    # Cache key for this question
    cache_key = f"quick_answer_{hash(question + scenario)}"

    if cache_key not in st.session_state:
        with st.spinner("Searching literature and generating quick answer..."):
            result = get_quick_answer_with_search(
                question=question,
                scenario=scenario,
                max_results=5
            )
            st.session_state[cache_key] = result

    result = st.session_state.get(cache_key)

    if result:
        # Header
        source_label = f"Based on {result.sources_used} sources" if result.has_context else "Based on general knowledge"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #10B981 0%, #059669 100%);
                    padding: 12px 16px; border-radius: 8px; color: white; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600;">Quick Answer</span>
                <span style="font-size: 12px; opacity: 0.9;">{source_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Answer with citation highlighting
        from ui.citation_utils import format_expert_response
        formatted_answer = format_expert_response(result.answer)
        st.markdown(formatted_answer, unsafe_allow_html=True)

        # Show citations if available
        if result.citations:
            with st.expander(f"View {len(result.citations)} sources", expanded=False):
                for i, cit in enumerate(result.citations, 1):
                    pmid = cit.get('pmid', '')
                    title = cit.get('title', 'Unknown')
                    if pmid:
                        st.markdown(f"[{i}] [{title}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/) (PMID: {pmid})")
                    else:
                        st.markdown(f"[{i}] {title}")

        # Option to run full panel
        st.caption(f"Model: {result.model}")

        return result

    return None
