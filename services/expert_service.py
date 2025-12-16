"""
Expert Discussion Service

Handles expert panel discussion orchestration with no UI dependencies.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional, Any, Callable

from config import settings

# Set up logger for this module
logger = logging.getLogger("literature_review.expert_service")


@dataclass
class DiscussionRoundResult:
    """Result of a single discussion round."""
    responses: Dict[str, Dict] = field(default_factory=dict)
    failures: List[tuple] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    knowledge_extracted: Dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of Pass 2 literature validation for a single expert."""
    expert_name: str
    validation_text: str
    citations_used: List[str] = field(default_factory=list)
    claims_supported: int = 0
    claims_contradicted: int = 0
    claims_no_evidence: int = 0
    analysis: str = ""  # Extracted analysis/additional findings section


class ExpertDiscussionService:
    """
    Orchestrates expert panel discussions.

    This service handles all business logic for expert discussions,
    including running rounds, regenerating responses, and extracting knowledge.
    No Streamlit or UI dependencies.
    """

    def __init__(
        self,
        api_key: str,
        model: str = None,
        max_tokens: int = None
    ):
        """
        Initialize the expert discussion service.

        Args:
            api_key: OpenAI API key
            model: Model to use (defaults to settings.EXPERT_MODEL)
            max_tokens: Max completion tokens (defaults to settings.EXPERT_MAX_TOKENS)
        """
        self.api_key = api_key
        self.model = model or settings.EXPERT_MODEL
        self.max_tokens = max_tokens or settings.EXPERT_MAX_TOKENS
        self._embedder = None
        
    def _get_embedder(self):
        """Lazy load embedder for semantic tagging."""
        if self._embedder is None:
            try:
                from core.ingestion.embedder import LocalEmbedder
                self._embedder = LocalEmbedder()
            except Exception as e:
                # Log but continue (semantic search will just be skipped)
                logger.warning(f"Could not load embedder for expert service: {e}")
        return self._embedder

    def auto_search_for_discussion(
        self,
        clinical_question: str,
        project_id: str,
        citation_dao,
        search_dao,
        query_cache_dao,
        max_results: int = 20
    ) -> Dict[str, Any]:
        """
        Auto-search when no citations provided.
        Returns search results dict compatible with existing flow.
        """
        from services.search_service import SearchService
        
        # Initialize SearchService
        search_service = SearchService(openai_api_key=self.api_key)

        # Run optimized search
        results = search_service.execute_search(
            query=clinical_question,
            project_id=project_id,
            citation_dao=citation_dao,
            search_dao=search_dao,
            query_cache_dao=query_cache_dao,
            max_results=max_results,
            ranking_mode="Balanced"
        )
        
        return results

    def start_background_literature_search(
        self,
        clinical_question: str,
        max_results: int = 20
    ):
        """
        Start tiered literature search in background thread.
        Uses component-based query building with fallback tiers.

        Args:
            clinical_question: The research question to search for
            max_results: Maximum papers per query

        Returns:
            Future that resolves to dict with 'citations', 'trials', 'concepts', etc.
        """
        from concurrent.futures import ThreadPoolExecutor

        def _search():
            try:
                from services.tiered_search_service import TieredSearchService

                # Use tiered search with component-based queries
                service = TieredSearchService(api_key=self.api_key)
                # Adjust max results per query
                service.MAX_RESULTS_PER_QUERY = max_results
                result = service.search(clinical_question)

                logger.info(
                    f"Background search complete: {len(result.citations)} papers, "
                    f"{len(result.trials)} trials (tier: {result.tier_used}, filtered {result.filtered_count})"
                )

                return {
                    'citations': result.citations,
                    'trials': result.trials,
                    'concepts': result.concepts.to_dict() if result.concepts else {},
                    'queries_used': result.queries_executed,
                    'queries_executed': result.queries_executed,
                    'tier_used': result.tier_used,
                    'filtered_count': result.filtered_count,
                    'total_before_filter': result.total_before_filter,
                    'query_type': 'TIERED',
                    'original_question': clinical_question
                }
            except Exception as e:
                logger.error(f"Background literature search failed: {e}")
                import traceback
                traceback.print_exc()
                return {
                    'citations': [],
                    'trials': [],
                    'concepts': {},
                    'queries_used': {},
                    'queries_executed': {},
                    'tier_used': 'ERROR',
                    'query_type': 'ERROR',
                    'error': str(e)
                }

        executor = ThreadPoolExecutor(max_workers=1)
        return executor.submit(_search)

    def start_two_channel_search(
        self,
        clinical_question: str,
        max_results_per_channel: int = 30
    ):
        """
        Start two-channel literature search in background thread.
        Searches Clinical + Biology channels in parallel.

        Clinical Channel: Phase 2/3 trials, SoC, competitive landscape
        Biology Channel: Expression, mechanism, safety precedent

        Args:
            clinical_question: The research question to search for
            max_results_per_channel: Maximum papers per channel

        Returns:
            Future that resolves to TwoChannelSearchResult dict
        """
        from concurrent.futures import ThreadPoolExecutor

        def _search():
            try:
                from services.two_channel_search_service import TwoChannelSearchService

                service = TwoChannelSearchService(api_key=self.api_key)
                service.MAX_RESULTS_PER_CHANNEL = max_results_per_channel
                result = service.search(clinical_question)

                logger.info(
                    f"Two-channel search complete: "
                    f"{len(result.clinical.citations)} clinical, "
                    f"{len(result.symptom.citations)} symptom, "
                    f"{len(result.trials)} trials"
                )

                return {
                    'clinical': result.clinical.to_dict(),
                    'symptom': result.symptom.to_dict(),
                    'concepts': result.concepts.to_dict() if result.concepts else {},
                    'trials': result.trials,
                    'total_citations': len(result.all_citations),
                    # Also include combined citations for backward compatibility
                    'citations': result.all_citations,
                    'query_type': 'TWO_CHANNEL',
                    'original_question': clinical_question,
                    # Quality report with target coverage and warnings
                    'quality_report': result.quality_report.to_dict() if result.quality_report else {}
                }
            except Exception as e:
                logger.error(f"Two-channel search failed: {e}")
                import traceback
                traceback.print_exc()
                return {
                    'clinical': {'citations': [], 'queries_executed': []},
                    'symptom': {'citations': [], 'queries_executed': []},
                    'concepts': {},
                    'trials': [],
                    'citations': [],
                    'query_type': 'ERROR',
                    'error': str(e)
                }

        executor = ThreadPoolExecutor(max_workers=1)
        return executor.submit(_search)

    def run_pass1_immediate(
        self,
        clinical_question: str,
        selected_experts: List[str],
        scenario: str,
        web_context: Optional[List[Dict]] = None,
        temperatures: Optional[Dict[str, float]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Dict]:
        """
        Pass 1: Immediate answers using LLM knowledge + web search.
        No literature citations - fast response for immediate user feedback.

        Args:
            clinical_question: The research question
            selected_experts: List of expert names to consult
            scenario: Clinical scenario key
            web_context: Optional web search results for context
            temperatures: Per-expert temperature settings
            progress_callback: Callback(expert_name, current, total)

        Returns:
            Dict mapping expert name to response dict
        """
        from travel.travel_personas import TRAVEL_EXPERTS, call_travel_expert as call_expert
        PRECLINICAL_EXPERTS = TRAVEL_EXPERTS  # Use travel experts

        responses = {}

        def call_single_expert_pass1(expert_name: str) -> tuple:
            """Call a single expert for Pass 1 (no literature)."""

            # Build Web Context (Grounding enabled)
            web_context_str = ""
            
            # Use Google Search grounding pre-fetch if enabled and no context provided
            if not web_context and settings.ENABLE_GOOGLE_SEARCH_GROUNDING:
                try:
                    from integrations.google_search import search_with_grounding
                    _, web_sources = search_with_grounding(
                        question=clinical_question,
                        system_context="Find travel tips, local recommendations, tourist attractions, and destination information",
                        max_sources=5
                    )
                    # Convert to context string
                    if web_sources:
                         web_snippets = []
                         for i, item in enumerate(web_sources, 1):
                            title = item.get('title', 'Web Source')
                            snippet = item.get('snippet', '')[:300]
                            url = item.get('url', '')
                            web_snippets.append(f"[Web {i}] {title}\n{snippet}\nSource: {url}")
                         web_context_str = "\n\n".join(web_snippets)
                except Exception as e:
                    logger.warning(f"Pass 1 grounding pre-fetch failed: {e}")

            elif web_context:
                web_snippets = []
                for i, item in enumerate(web_context[:5], 1):
                    title = item.get('title', 'Web Source')
                    snippet = item.get('snippet', item.get('content', ''))[:300]
                    url = item.get('url', '')
                    web_snippets.append(f"[Web {i}] {title}\n{snippet}\nSource: {url}")
                web_context_str = "\n\n".join(web_snippets)

            # Build Pass 1 specific prompt
            expert_info = PRECLINICAL_EXPERTS.get(expert_name, {})
            expert_role = expert_info.get('role', expert_name)

            system_instruction = f"""You are {expert_name}, a {expert_role}.

IMPORTANT: This is a PRELIMINARY response based on your expert knowledge and available web information.
A literature search is running in the background and will validate your claims shortly.

Please:
1. Answer the question using your domain expertise
2. Be clear about what is well-established vs. your professional opinion
3. Tag claims as [EVIDENCE], [ASSUMPTION], or [OPINION]
4. Note any areas where literature validation would be especially valuable

{"Web Search Context:" + chr(10) + web_context_str if web_context_str else "No web context available - rely on your training knowledge."}
"""

            # Get temperature for this expert
            expert_temp = None
            if temperatures and self.model in [
                "gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.5-flash", "gemini-3-pro-preview",
                "gpt-4o", "gpt-4o-mini",
                "claude-3-5-sonnet", "claude-3-opus"
            ]:
                expert_temp = temperatures.get(expert_name, 0.7)

            try:
                response = call_expert(
                    persona_name=expert_name,
                    clinical_question=clinical_question,
                    evidence_context="",  # No literature context in Pass 1
                    round_num=1,
                    previous_responses=None,
                    priors_text=None,
                    openai_api_key=self.api_key,
                    model=self.model,
                    max_completion_tokens=self.max_tokens,
                    injected_evidence=[],
                    temperature=expert_temp,
                    system_instruction_override=system_instruction
                )
                response['pass'] = 1
                response['has_literature'] = False
                return (expert_name, response, None)
            except Exception as e:
                logger.error(f"Pass 1 expert call failed for {expert_name}: {e}")
                return (expert_name, None, str(e))

        # Run all experts in parallel with thread safety
        responses_lock = Lock()
        max_workers = min(len(selected_experts), 10)  # Limit max workers

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(call_single_expert_pass1, name): name
                for name in selected_experts
            }

            completed = 0
            for future in as_completed(futures):
                expert_name = futures[future]
                completed += 1
                if progress_callback:
                    progress_callback(expert_name, completed, len(selected_experts))

                try:
                    name, response, error = future.result()
                    with responses_lock:
                        if error:
                            responses[name] = {
                                'content': f"Error: {error}",
                                'finish_reason': 'error',
                                'pass': 1
                            }
                        else:
                            responses[name] = response
                except Exception as e:
                    with responses_lock:
                        responses[expert_name] = {
                            'content': f"Error: {str(e)}",
                            'finish_reason': 'error',
                            'pass': 1
                        }

        return responses

    def run_pass2_validation(
        self,
        clinical_question: str,
        pass1_responses: Dict[str, Dict],
        literature_citations: List,
        selected_experts: List[str],
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, ValidationResult]:
        """
        Pass 2: Validate Pass 1 claims against literature.
        Reviews each expert's initial response and checks claims against found papers.

        Args:
            clinical_question: Original research question
            pass1_responses: Dict of expert name -> Pass 1 response
            literature_citations: List of Citation objects from PubMed search
            selected_experts: List of expert names to validate
            progress_callback: Callback(expert_name, current, total)

        Returns:
            Dict mapping expert name to ValidationResult
        """
        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=self.api_key, model=self.model)

        # Format citations for validation prompt
        def format_citations_for_validation(citations, max_citations=10):
            """Format citations into a readable context string."""
            # Handle dict format (convert to list)
            if isinstance(citations, dict):
                citations = list(citations.values())
            # Flatten if nested list
            if citations and isinstance(citations[0], list):
                citations = [c for sublist in citations for c in sublist if c]

            citation_parts = []
            for i, c in enumerate(citations[:max_citations], 1):
                # Handle both Citation objects and dicts
                if hasattr(c, 'pmid'):
                    pmid = c.pmid
                    title = c.title
                    abstract = c.abstract[:500] if c.abstract else "No abstract"
                    year = c.year
                    authors = ', '.join(c.authors[:3]) if c.authors else "Unknown"
                else:
                    pmid = c.get('pmid', 'N/A')
                    title = c.get('title', 'Untitled')
                    abstract = (c.get('abstract', '') or '')[:500] or "No abstract"
                    year = c.get('year', 'N/A')
                    authors = ', '.join(c.get('authors', [])[:3]) if c.get('authors') else "Unknown"

                citation_parts.append(
                    f"[{i}] PMID:{pmid} ({year})\n"
                    f"Title: {title}\n"
                    f"Authors: {authors}\n"
                    f"Abstract: {abstract}..."
                )
            return "\n\n".join(citation_parts)

        citation_context = format_citations_for_validation(literature_citations)

        def validate_single_expert(expert_name: str) -> tuple:
            """Validate a single expert's Pass 1 response."""
            pass1_content = pass1_responses.get(expert_name, {}).get('content', '')

            if not pass1_content or pass1_content.startswith("Error:"):
                return (expert_name, ValidationResult(
                    expert_name=expert_name,
                    validation_text="No valid Pass 1 response to validate.",
                    citations_used=[]
                ), None)

            validation_prompt = f"""You are {expert_name}. Review your previous response and validate it against the literature search results.

**Your Previous Response (Pass 1):**
{pass1_content[:3000]}

**Literature Found ({len(literature_citations)} papers):**
{citation_context}

**Task:**
Review each key claim from your previous response. For each claim, indicate if the literature:
- ✅ SUPPORTS (cite the specific PMID)
- ❌ CONTRADICTS (cite PMID, briefly explain the discrepancy)
- ⚠️ NO EVIDENCE FOUND (this remains an assumption/opinion)

Also add any important findings from the literature that you missed in your initial response.

**Format your response as:**

## Evidence Validation

**Claim 1:** [state the claim from your response]
**Status:** ✅/❌/⚠️ PMID:XXXXXXXX (if applicable)
**Note:** [brief explanation]

**Claim 2:** [next claim]
**Status:** ...
**Note:** ...

[continue for 3-5 key claims]

## Additional Findings from Literature
[Any important information from the papers that you didn't mention in Pass 1]

## Confidence Update
[Has the literature search increased or decreased your confidence in your recommendations? Why?]
"""

            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": f"You are {expert_name}, validating your previous claims against new literature evidence."},
                        {"role": "user", "content": validation_prompt}
                    ],
                    max_completion_tokens=self.max_tokens
                )

                validation_text = response.choices[0].message.content or ""

                # Count validation statuses
                supported = validation_text.count('✅')
                contradicted = validation_text.count('❌')
                no_evidence = validation_text.count('⚠️')

                # Extract analysis section (Additional Findings + Confidence Update)
                analysis = ""
                if "## Additional Findings" in validation_text:
                    parts = validation_text.split("## Additional Findings")
                    if len(parts) > 1:
                        analysis = parts[1].strip()
                elif "## Confidence Update" in validation_text:
                    parts = validation_text.split("## Confidence Update")
                    if len(parts) > 1:
                        analysis = parts[1].strip()
                else:
                    # Fallback: use last 500 chars as summary
                    analysis = validation_text[-500:] if len(validation_text) > 500 else validation_text

                return (expert_name, ValidationResult(
                    expert_name=expert_name,
                    validation_text=validation_text,
                    citations_used=[c.pmid if hasattr(c, 'pmid') else c.get('pmid') for c in literature_citations[:10]],
                    claims_supported=supported,
                    claims_contradicted=contradicted,
                    claims_no_evidence=no_evidence,
                    analysis=analysis
                ), None)

            except Exception as e:
                logger.error(f"Pass 2 validation failed for {expert_name}: {e}")
                return (expert_name, ValidationResult(
                    expert_name=expert_name,
                    validation_text=f"Validation failed: {str(e)}",
                    citations_used=[]
                ), str(e))

        # Run validations in parallel
        validations = {}  # Initialize validations dict
        with ThreadPoolExecutor(max_workers=len(selected_experts)) as executor:
            futures = {
                executor.submit(validate_single_expert, name): name
                for name in selected_experts
            }

            completed = 0
            for future in as_completed(futures):
                expert_name = futures[future]
                completed += 1
                if progress_callback:
                    progress_callback(expert_name, completed, len(selected_experts))

                try:
                    name, validation_result, error = future.result()
                    validations[name] = validation_result
                except Exception as e:
                    validations[expert_name] = ValidationResult(
                        expert_name=expert_name,
                        validation_text=f"Error during validation: {str(e)}",
                        citations_used=[]
                    )

        return validations

    def run_two_channel_validation(
        self,
        clinical_question: str,
        pass1_responses: Dict[str, Dict],
        two_channel_result: Dict,
        selected_experts: List[str],
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Dict]:
        """
        Pass 2 with two-channel evidence: Validate claims against appropriate channel.

        Clinical claims (trials, SoC, efficacy) → Clinical channel evidence
        Biology claims (expression, mechanism, safety) → Biology channel evidence

        Args:
            clinical_question: Original research question
            pass1_responses: Dict of expert name -> Pass 1 response
            two_channel_result: Dict from start_two_channel_search()
            selected_experts: List of expert names to validate
            progress_callback: Callback(expert_name, current, total)

        Returns:
            Dict mapping expert name to validation dict with channel breakdown
        """
        from services.claim_validator import ClaimValidator, ClaimType

        validator = ClaimValidator(api_key=self.api_key)

        # Extract channel citations
        clinical_citations = two_channel_result.get('clinical', {}).get('citations', [])
        biology_citations = two_channel_result.get('biology', {}).get('citations', [])

        def validate_expert_two_channel(expert_name: str) -> tuple:
            """Validate expert response against both channels."""
            pass1_content = pass1_responses.get(expert_name, {}).get('content', '')

            if not pass1_content or pass1_content.startswith("Error:"):
                return (expert_name, {
                    'validation_text': "No valid Pass 1 response to validate.",
                    'clinical_claims': 0,
                    'biology_claims': 0,
                    'claims_supported': 0,
                    'claims_contradicted': 0,
                    'claims_no_evidence': 0
                }, None)

            try:
                # Extract and classify claims
                claims = validator.extract_claims(pass1_content)

                clinical_claims = [c for c, t in claims if t == ClaimType.CLINICAL]
                biology_claims = [c for c, t in claims if t in [ClaimType.BIOLOGY, ClaimType.SAFETY]]

                # Build validation summary
                lines = []
                lines.append(f"## Two-Channel Validation for {expert_name}")
                lines.append("")

                total_supported = 0
                total_contradicted = 0
                total_no_evidence = 0

                # Validate clinical claims against clinical channel
                if clinical_claims:
                    lines.append(f"### Clinical Claims ({len(clinical_claims)})")
                    lines.append(f"*Validated against {len(clinical_citations)} clinical papers*")
                    lines.append("")

                    for claim in clinical_claims[:3]:
                        # Simple keyword matching for demonstration
                        found = self._find_supporting_citation(claim, clinical_citations)
                        if found:
                            lines.append(f"✅ \"{claim[:80]}...\"")
                            lines.append(f"   Supported by PMID: {found}")
                            total_supported += 1
                        else:
                            lines.append(f"⚠️ \"{claim[:80]}...\"")
                            lines.append("   No supporting evidence found")
                            total_no_evidence += 1
                    lines.append("")

                # Validate biology claims against biology channel
                if biology_claims:
                    lines.append(f"### Biology Claims ({len(biology_claims)})")
                    lines.append(f"*Validated against {len(biology_citations)} biology papers*")
                    lines.append("")

                    for claim in biology_claims[:3]:
                        found = self._find_supporting_citation(claim, biology_citations)
                        if found:
                            lines.append(f"✅ \"{claim[:80]}...\"")
                            lines.append(f"   Supported by PMID: {found}")
                            total_supported += 1
                        else:
                            lines.append(f"⚠️ \"{claim[:80]}...\"")
                            lines.append("   No supporting evidence found")
                            total_no_evidence += 1
                    lines.append("")

                # Summary
                lines.append("### Summary")
                lines.append(f"- **Supported:** {total_supported}")
                lines.append(f"- **Contradicted:** {total_contradicted}")
                lines.append(f"- **No Evidence:** {total_no_evidence}")

                return (expert_name, {
                    'validation_text': "\n".join(lines),
                    'clinical_claims': len(clinical_claims),
                    'biology_claims': len(biology_claims),
                    'claims_supported': total_supported,
                    'claims_contradicted': total_contradicted,
                    'claims_no_evidence': total_no_evidence,
                    'citations_used': {
                        'clinical': [c.get('pmid') if isinstance(c, dict) else c.pmid for c in clinical_citations[:5]],
                        'biology': [c.get('pmid') if isinstance(c, dict) else c.pmid for c in biology_citations[:5]]
                    }
                }, None)

            except Exception as e:
                logger.error(f"Two-channel validation failed for {expert_name}: {e}")
                return (expert_name, {
                    'validation_text': f"Validation error: {str(e)}",
                    'claims_supported': 0,
                    'claims_contradicted': 0,
                    'claims_no_evidence': 0
                }, str(e))

        # Run validations in parallel
        validations = {}
        with ThreadPoolExecutor(max_workers=len(selected_experts)) as executor:
            futures = {
                executor.submit(validate_expert_two_channel, name): name
                for name in selected_experts
            }

            completed = 0
            for future in as_completed(futures):
                expert_name = futures[future]
                completed += 1
                if progress_callback:
                    progress_callback(expert_name, completed, len(selected_experts))

                try:
                    name, validation_result, error = future.result()
                    validations[name] = validation_result
                except Exception as e:
                    validations[expert_name] = {
                        'validation_text': f"Error: {str(e)}",
                        'claims_supported': 0,
                        'claims_contradicted': 0,
                        'claims_no_evidence': 0
                    }

        return validations

    def _find_supporting_citation(self, claim: str, citations: List) -> Optional[str]:
        """Find a citation that supports the given claim."""
        # Extract key terms from claim
        import re
        words = re.findall(r'\b[A-Z][a-z]+\b|\b[A-Z]+\b|\d+%?', claim)
        key_terms = [w.lower() for w in words if len(w) >= 2]

        if not key_terms:
            return None

        for cit in citations:
            if isinstance(cit, dict):
                text = f"{cit.get('title', '')} {cit.get('abstract', '')}".lower()
                pmid = cit.get('pmid', 'unknown')
            elif hasattr(cit, 'title'):
                text = f"{cit.title} {cit.abstract}".lower()
                pmid = cit.pmid
            else:
                continue

            # Check for term overlap
            matching = sum(1 for t in key_terms if t in text)
            if matching >= len(key_terms) * 0.3:  # At least 30% overlap
                return pmid

        return None

    def run_discussion_round(
        self,
        round_num: int,
        clinical_question: str,
        selected_experts: List[str],
        citations: List[Dict],
        scenario: str,
        previous_responses: Optional[Dict[str, str]] = None,
        injected_evidence: Optional[List[Dict]] = None,
        temperatures: Optional[Dict[str, float]] = None,
        working_memory: Any = None,
        rag_context: Optional[List] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> DiscussionRoundResult:
        """
        Run a single discussion round.

        Args:
            round_num: Current round number
            clinical_question: The research question
            selected_experts: List of expert names to consult
            citations: List of citation dicts
            scenario: Drug development scenario key
            previous_responses: Responses from previous round
            injected_evidence: HITL injected papers
            temperatures: Per-expert temperature settings
            working_memory: Working memory instance
            rag_context: RAG context from local documents
            progress_callback: Callback(expert_name, current, total)

        Returns:
            DiscussionRoundResult with responses and failures
        """
        from gdg.gdg_utils import format_evidence_context
        from travel.travel_personas import call_travel_expert as call_expert
        from core.priors_manager import PriorsManager

        # Initialize priors manager
        try:
            priors_manager = PriorsManager()
        except Exception:
            priors_manager = None

        # Prepare search results dict
        search_results_dict = {'citations': citations}

        result = DiscussionRoundResult()

        def call_single_expert(expert_name: str) -> tuple:
            """Call a single expert (for parallel execution)."""
            # Get priors text
            priors_text = None
            if priors_manager:
                try:
                    priors_text = priors_manager.format_priors_for_context(
                        scenario=scenario,
                        persona=expert_name,
                        max_frameworks=3,
                        compressed=(round_num > 1),
                        query=clinical_question,
                        embedder=self._get_embedder()
                    )
                except Exception:
                    pass

            # Format evidence context
            evidence_context = format_evidence_context(
                search_results=search_results_dict,
                persona_role=expert_name,
                max_citations=10,
                clinical_question=clinical_question
            )

            # Get temperature for this expert
            expert_temp = None
            if temperatures and self.model in [
                "gemini-2.0-flash", "gemini-1.5-pro", "gemini-3.0-pro-preview",
                "gpt-4o", "gpt-4o-mini",
                "claude-3-5-sonnet", "claude-3-opus"
            ]:
                expert_temp = temperatures.get(expert_name, 0.7)

            try:
                response = call_expert(
                    persona_name=expert_name,
                    clinical_question=clinical_question,
                    evidence_context=evidence_context,
                    round_num=round_num,
                    previous_responses=previous_responses,
                    priors_text=priors_text,
                    openai_api_key=self.api_key,
                    model=self.model,
                    max_completion_tokens=self.max_tokens,
                    injected_evidence=injected_evidence or [],
                    temperature=expert_temp,
                    working_memory=working_memory,
                    rag_context=rag_context
                )
                return (expert_name, response, None)
            except Exception as e:
                return (expert_name, None, str(e))

        # Run all experts in parallel
        with ThreadPoolExecutor(max_workers=len(selected_experts)) as executor:
            futures = {
                executor.submit(call_single_expert, name): name
                for name in selected_experts
            }

            completed = 0
            for future in as_completed(futures):
                expert_name = futures[future]
                completed += 1
                if progress_callback:
                    progress_callback(expert_name, completed, len(selected_experts))

                try:
                    name, response, error = future.result()
                    if error:
                        error_response = {'content': f"Error: {error}", 'finish_reason': 'error'}
                        result.responses[name] = error_response
                        result.failures.append((name, error))
                    else:
                        result.responses[name] = response
                        if response.get('finish_reason') == 'error':
                            result.failures.append((name, response.get('content', 'Unknown error')))
                except Exception as e:
                    error_response = {'content': f"Error: {str(e)}", 'finish_reason': 'error'}
                    result.responses[expert_name] = error_response
                    result.failures.append((expert_name, str(e)))

        return result

    def regenerate_response(
        self,
        expert_name: str,
        round_num: int,
        clinical_question: str,
        citations: List[Dict],
        scenario: str,
        rejection_critique: str,
        previous_responses: Optional[Dict[str, str]] = None,
        injected_evidence: Optional[List[Dict]] = None,
        working_memory: Any = None,
        old_response: Optional[Dict] = None
    ) -> Dict:
        """
        Regenerate a single expert response with feedback.

        Args:
            expert_name: Expert to regenerate
            round_num: Round number
            clinical_question: Research question
            citations: Citation dicts
            scenario: Scenario key
            rejection_critique: Human feedback on why response was rejected
            previous_responses: Previous round responses
            injected_evidence: Injected papers
            working_memory: Working memory instance
            old_response: Optional previous response object to track history

        Returns:
            New response dict with 'regenerated' flag and 'history'
        """
        from gdg.gdg_utils import format_evidence_context
        from travel.travel_personas import call_travel_expert as call_expert
        from core.priors_manager import PriorsManager
        from datetime import datetime

        # Initialize priors manager
        try:
            priors_manager = PriorsManager()
        except Exception:
            priors_manager = None

        search_results_dict = {'citations': citations}

        priors_text = None
        if priors_manager:
            try:
                priors_text = priors_manager.format_priors_for_context(
                    scenario=scenario,
                    persona=expert_name,
                    max_frameworks=3,
                    compressed=(round_num > 1),
                    query=clinical_question,
                    embedder=self._get_embedder()
                )
            except Exception:
                pass

        evidence_context = format_evidence_context(
            search_results=search_results_dict,
            persona_role=expert_name,
            max_citations=10,
            clinical_question=clinical_question
        )

        response = call_expert(
            persona_name=expert_name,
            clinical_question=clinical_question,
            evidence_context=evidence_context,
            round_num=round_num,
            previous_responses=previous_responses,
            priors_text=priors_text,
            openai_api_key=self.api_key,
            model=self.model,
            max_completion_tokens=self.max_tokens,
            rejection_critique=rejection_critique,
            injected_evidence=injected_evidence or [],
            working_memory=working_memory
        )

        response['regenerated'] = True
        
        # Handle Version History
        if old_response:
            history = old_response.get('history', [])
            # Archive the old content
            history_entry = {
                'content': old_response.get('content'),
                'critique': rejection_critique,
                'timestamp': datetime.now().isoformat(),
                'version': len(history) + 1
            }
            response['history'] = history + [history_entry]
            
            # Log this interaction for RLHF
            self.log_feedback_interaction(
                expert_name=expert_name,
                prompt=evidence_context, # Approximation of prompt
                old_response=old_response.get('content'),
                critique=rejection_critique,
                new_response=response.get('content')
            )

        return response

    def log_feedback_interaction(self, expert_name: str, prompt: str, old_response: str, critique: str, new_response: str):
        """
        Log HITL interaction for future RLHF / Model improvement.
        In a real production system, this would write to BigQuery or a dedicated DB.
        """
        import json
        from pathlib import Path
        
        log_entry = {
            "timestamp": "iso_timestamp_here", # simplistic placeholder
            "expert": expert_name,
            "interaction_type": "hitl_regeneration",
            "data": {
                "old_response_len": len(old_response) if old_response else 0,
                "critique": critique,
                "new_response_len": len(new_response) if new_response else 0
            }
        }
        # For now, just print/log to console to avoid file I/O overhead in this demo
        # print(f"RELHF LOG: {json.dumps(log_entry)}")
        pass

    def generate_follow_up_questions(
        self,
        clinical_question: str,
        responses: Dict[str, Dict],
        max_questions: int = 4
    ) -> List[str]:
        """
        Generate follow-up questions based on discussion.

        Args:
            clinical_question: Original question
            responses: Expert responses dict
            max_questions: Maximum questions to generate

        Returns:
            List of follow-up question strings
        """
        try:
            from openai import OpenAI

            all_responses = " ".join([
                resp.get('content', '')[:500]
                for resp in responses.values()
            ])

            if len(all_responses) < 200:
                return []

            from core.llm_utils import get_llm_client
            client = get_llm_client(api_key=self.api_key, model=self.model)
            question_response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate 4 concise follow-up questions based on this expert panel discussion. Return only the questions, one per line, no numbering."
                    },
                    {
                        "role": "user",
                        "content": f"Research question: {clinical_question}\n\nExpert responses:\n{all_responses[:3000]}"
                    }
                ],
                max_completion_tokens=300
            )

            questions = question_response.choices[0].message.content.strip().split('\n')
            questions = [q.strip() for q in questions if q.strip() and len(q.strip()) > 10]
            return questions[:max_questions]

        except Exception:
            return []

    def extract_knowledge(
        self,
        responses: Dict[str, Dict],
        clinical_question: str,
        source_name: str
    ) -> Dict:
        """
        Extract knowledge from discussion responses.

        Args:
            responses: Expert responses
            clinical_question: Research question
            source_name: Name for knowledge source

        Returns:
            Dict with 'facts_count' and 'triples_count'
        """
        try:
            from gdg import process_discussion_for_knowledge

            return process_discussion_for_knowledge(
                discussion_responses=responses,
                clinical_question=clinical_question,
                source_name=source_name,
                openai_api_key=self.api_key
            )
        except Exception:
            return {'facts_count': 0, 'triples_count': 0}

    def generate_perspective_questions(
        self,
        clinical_question: str,
        expert_names: List[str]
    ) -> Dict[str, List[str]]:
        """
        Generate perspective-specific questions for each expert.

        Args:
            clinical_question: Research question
            expert_names: List of experts

        Returns:
            Dict mapping expert name to list of questions
        """
        try:
            from gdg import generate_perspective_questions

            return generate_perspective_questions(
                clinical_question=clinical_question,
                expert_names=expert_names,
                openai_api_key=self.api_key
            )
        except Exception:
            return {}
    def run_debate_round(
        self,
        clinical_question: str,
        pro_expert: str,
        con_expert: str,
        topic: str,
        citations: List[Dict],
        scenario: str,
        working_memory: Any = None
    ) -> Dict[str, Any]:
        """
        Run a collaborative debate round (Proposal -> Challenge -> Mitigation).
        
        Logic:
        1. Proposal: Pro_expert proposes a path.
        2. Challenge: Con_expert crtiiques it based on their domain.
        3. Mitigation: Pro_expert proposes a fix.
        4. Synthesis: Chairperson summarizes.
        """
        from gdg.gdg_utils import format_evidence_context
        from travel.travel_personas import call_travel_expert as call_expert
        from core.priors_manager import PriorsManager

        # 0. Setup Contexts
        evidence_context_pro = format_evidence_context({'citations': citations}, pro_expert, 5, clinical_question)
        evidence_context_con = format_evidence_context({'citations': citations}, con_expert, 5, clinical_question)
        
        # 1. Proposal (Pro-Expert)
        prompt_1 = f"""Discussion Topic: {topic}
        Based on your domain expertise, propose a concrete decision or hypothesis for this topic. 
        Cite evidence [1] where possible."""
        
        resp_1 = call_expert(pro_expert, clinical_question, evidence_context_pro, 1, 
                             injected_evidence=[], system_instruction_override=prompt_1, model=self.model, openai_api_key=self.api_key)
        proposal_text = resp_1.get('content', '')
        
        # 2. Constructive Challenge (Con-Expert)
        prompt_2 = f"""Review the following proposal from the {pro_expert}:
        "{proposal_text}"
        
        Assume their data is correct. However, what SPECIFIC risks or failure modes does this create for YOUR domain goals?
        Be constructive but rigorous."""
        
        resp_2 = call_expert(con_expert, clinical_question, evidence_context_con, 1, 
                             injected_evidence=[], system_instruction_override=prompt_2, model=self.model, openai_api_key=self.api_key)
        challenge_text = resp_2.get('content', '')
        
        # 3. Mitigation (Pro-Expert)
        prompt_3 = f"""The {con_expert} has raised this concern:
        "{challenge_text}"
        
        Propose a modification or mitigation strategy to address this while maintaining your original objective."""
        
        resp_3 = call_expert(pro_expert, clinical_question, evidence_context_pro, 1,
                             injected_evidence=[], system_instruction_override=prompt_3, model=self.model, openai_api_key=self.api_key)
        mitigation_text = resp_3.get('content', '')
        
        # 4. Chairperson Synthesis
        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=self.api_key, model=self.model)
        
        synthesis_prompt = f"""Synthesize this debate into a decision path.
        Topic: {topic}
        Proposal ({pro_expert}): {proposal_text}
        Challenge ({con_expert}): {challenge_text}
        Mitigation ({pro_expert}): {mitigation_text}
        
        Identify:
        1. The Core Conflict (what are they trading off?).
        2. The Consensus Path (if any).
        3. Residual Risks.
        """
        
        synth_resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        synthesis_text = synth_resp.choices[0].message.content
        
        return {
            "topic": topic,
            "pro_expert": pro_expert,
            "con_expert": con_expert,
            "proposal": proposal_text,
            "challenge": challenge_text,
            "mitigation": mitigation_text,
            "synthesis": synthesis_text
        }

    def run_storm_workflow(
        self,
        clinical_question: str,
        expert_name: str,
        citations: List[Dict],
        injected_evidence: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Run a true STORM-style (Synthesis of Topic Outlines through Retrieval and Multi-perspective) workflow.
        
        Stages:
        1. Outline: Generate a structure.
        2. Draft: Write the initial full text.
        3. Critique: Simulated Peer Reviewer identifies gaps/bias.
        4. Revision: Author rewrites for the final version.
        """
        from gdg.gdg_utils import format_evidence_context
        from travel.travel_personas import call_travel_expert as call_expert
        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=self.api_key, model=self.model)

        evidence_context = format_evidence_context({'citations': citations}, expert_name, 10, clinical_question)
        
        # 1. Outline Generation
        outline_prompt = f"""You are {expert_name}. Create a comprehensive outline to answer: "{clinical_question}".
        Focus on structure, key headings, and the logical flow of arguments. Do not write the full text yet."""
        
        resp_outline = call_expert(expert_name, clinical_question, evidence_context, 1, 
                                   injected_evidence=[], system_instruction_override=outline_prompt, 
                                   model=self.model, openai_api_key=self.api_key)
        outline_text = resp_outline.get('content', '')
        
        # 2. Initial Draft
        draft_prompt = f"""You are {expert_name}. Write a FULL DRAFT based on this outline:
        {outline_text[:2000]}
        
        Be comprehensive, cite evidence, and be rigorous."""
        
        resp_draft = call_expert(expert_name, clinical_question, evidence_context, 1,
                                 injected_evidence=[], system_instruction_override=draft_prompt,
                                 model=self.model, openai_api_key=self.api_key)
        draft_text = resp_draft.get('content', '')

        # 3. Peer Review (Critique)
        reviewer_prompt = f"""You are a Critical Scientific Reviewer. Review this draft from {expert_name}:
        
        "{draft_text[:10000]}"
        
        Research Question: {clinical_question}
        
        Identify 3 specific weaknesses:
        1. Ambiguous claims?
        2. Missing evidence citations?
        3. Logical gaps?
        
        Provide specific, constructive criticism."""
        
        resp_critique = client.chat.completions.create(
            model=settings.REASONING_MODEL,  # Use Reasoning Model for Critique (Step 3)
            messages=[{"role": "user", "content": reviewer_prompt}]
        )
        critique_text = resp_critique.choices[0].message.content
        
        # 4. Final Revision
        revision_prompt = f"""You are {expert_name}. REVISE your draft based on this feedback.
        
        Original Draft:
        {draft_text[:5000]}... [truncated]
        
        Reviewer Feedback:
        {critique_text}
        
        Task: Write the FINAL, polished version. Address the feedback. Ensure all [EVIDENCE] tags are preserved."""
        
        resp_final = call_expert(expert_name, clinical_question, evidence_context, 1,
                                 injected_evidence=[], system_instruction_override=revision_prompt,
                                 model=self.model, openai_api_key=self.api_key)
        
        return {
            "method": "STORM-v2",
            "outline": outline_text,
            "draft": draft_text,
            "critique": critique_text,
            "final_response": resp_final.get('content', ''),
            "provenance": resp_final
        }

