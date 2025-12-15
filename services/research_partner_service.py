"""
Research Partner Service

Unified orchestration for conversational research workflow (v3.0):
1. Parse question and detect type
2. Auto-search relevant literature
3. Consult selected experts (Two-Pass mode)
4. Synthesize recommendation
5. Support inline follow-ups

Reuses: ExpertDiscussionService, existing expert infrastructure
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Generator, Any, Callable

from openai import OpenAI

from config import settings
from core.llm_utils import get_llm_client
from core.question_templates import (
    QUESTION_TYPES,
    get_experts_for_question_type,
    get_scenario_for_question_type,
    get_synthesis_template,
    detect_question_type
)
from services.expert_service import ExpertDiscussionService, ValidationResult

# Grounding sources
from integrations.tavily import search_web, format_web_results, TavilyClient
from core.knowledge_store import KnowledgeStore
from services.document_library import get_document_library

# Program context extraction
from services.program_extractor import ProgramExtractor
from core.database import ProgramProfileDAO, TrustedKnowledgeDAO

# Portfolio context removed - no longer used for palliative surgery domain

# Expert-First Evidence Flow: Claim extraction and supporting literature
try:
    from core.claims import extract_claims_from_responses, get_searchable_claims
    from services.supporting_literature_service import (
        SupportingLiteratureService,
        LiteratureSearchResult,
        search_supporting_literature,
        determine_certainty
    )
    EXPERT_FIRST_FLOW_AVAILABLE = True
except ImportError:
    EXPERT_FIRST_FLOW_AVAILABLE = False

logger = logging.getLogger("research_partner.service")


# DEPRECATED: ResponseMode is no longer used. Kept for backwards compatibility.
# The system now always uses Expert-First approach with supporting literature.
class ResponseMode(str, Enum):
    """DEPRECATED: Response mode is no longer used. Always uses Expert-First approach."""
    EXPERT_CONSENSUS = "expert_consensus"
    LITERATURE_VERIFIED = "literature_verified"  # No longer triggers different behavior


@dataclass
class ResearchResult:
    """Complete research result for conversational mode."""
    question: str
    question_type: str
    recommendation: str
    confidence: str  # HIGH, MEDIUM, LOW
    response_mode: str = "expert_consensus"  # Track which mode was used
    key_findings: List[str] = field(default_factory=list)
    evidence_summary: Dict = field(default_factory=dict)
    expert_responses: Dict[str, Dict] = field(default_factory=dict)
    validations: Dict[str, ValidationResult] = field(default_factory=dict)
    follow_up_suggestions: List[str] = field(default_factory=list)
    dissenting_views: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    # Expert-First Flow: Supporting literature matched to expert claims
    supporting_literature: Optional[Any] = None  # LiteratureSearchResult when available

    def to_dict(self) -> Dict:
        """Convert to dictionary for session state storage."""
        result = {
            'question': self.question,
            'question_type': self.question_type,
            'recommendation': self.recommendation,
            'confidence': self.confidence,
            'response_mode': self.response_mode,
            'key_findings': self.key_findings,
            'evidence_summary': self.evidence_summary,
            'expert_responses': self.expert_responses,
            'validations': {k: v.__dict__ if hasattr(v, '__dict__') else v
                           for k, v in self.validations.items()},
            'follow_up_suggestions': self.follow_up_suggestions,
            'dissenting_views': self.dissenting_views,
            'metadata': self.metadata
        }
        # Include supporting literature if available
        if self.supporting_literature:
            if hasattr(self.supporting_literature, 'to_dict'):
                result['supporting_literature'] = self.supporting_literature.to_dict()
            else:
                result['supporting_literature'] = self.supporting_literature
        return result


class ResearchPartnerService:
    """
    Unified orchestration service for conversational research.

    Coordinates:
    - Question parsing and type detection
    - Background literature search
    - Expert consultation (Two-Pass)
    - Response synthesis
    - Inline follow-up handling
    """

    def __init__(self, api_key: str, model: str = None):
        """
        Initialize the Research Partner service.

        Args:
            api_key: OpenAI API key
            model: Model to use (defaults to settings.EXPERT_MODEL)
        """
        self.api_key = api_key
        self.model = model or getattr(settings, 'EXPERT_MODEL', 'gemini-3-pro-preview')

        # Reuse existing expert service
        self.expert_service = ExpertDiscussionService(
            api_key=api_key,
            model=self.model
        )
        
        # Initialize Database connection
        try:
            from core.database import DatabaseManager, CitationDAO, ExpertDiscussionDAO
            from pathlib import Path
            # Use default DB path from settings or fallback
            db_path = Path(getattr(settings, 'DB_PATH', "outputs/literature.db"))
            self.db_manager = DatabaseManager(db_path)
            self.citation_dao = CitationDAO(self.db_manager)
            self.discussion_dao = ExpertDiscussionDAO(self.db_manager)
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
            self.db_manager = None

    # Alias for compatibility
    def run_research_flow(self, *args, **kwargs):
        """Alias for search_and_answer to maintain compatibility."""
        return self.search_and_answer(*args, **kwargs)

    def search_and_answer(
        self,
        question: str,
        question_type: Optional[str] = None,
        additional_context: Optional[str] = None,
        project_id: Optional[int] = None,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
        selected_experts: Optional[List[str]] = None,
        response_mode: Optional[ResponseMode] = None  # Deprecated, ignored
    ) -> Generator[Dict, None, ResearchResult]:
        """
        Execute complete research flow with streaming progress.

        Uses Expert-First approach:
        1. Get expert perspectives (Pass 1)
        2. Extract claims with search hints from responses
        3. Search for supporting case series
        4. Synthesize recommendation with GRADE certainty

        Args:
            question: The user's research question
            question_type: Optional pre-selected question type
            additional_context: Optional user-provided context
            progress_callback: Callback(stage, message, progress_pct)
            selected_experts: Optional list of expert names (overrides auto-selection)
            response_mode: DEPRECATED - ignored, always uses Expert-First approach

        Yields:
            Progress events with stage, message, and data

        Returns:
            Final ResearchResult
        """
        start_time = datetime.now()

        # Stage 1: Parse question and detect type
        yield {"stage": "parsing", "message": "Understanding your question...", "progress": 0.05}

        if not question_type:
            question_type = detect_question_type(question, self.api_key)

        type_info = QUESTION_TYPES.get(question_type, QUESTION_TYPES["general"])

        # Use user-selected experts if provided, otherwise auto-detect
        if selected_experts and len(selected_experts) > 0:
            experts = selected_experts
        else:
            experts = get_experts_for_question_type(question_type)

        yield {
            "stage": "parsing",
            "message": f"Detected: {type_info['name']}",
            "progress": 0.10,
            "question_type": question_type,
            "experts": experts
        }

        # Stage 1b: Auto-fetch - extract concepts, update profile, trigger background fetch
        program_context = None
        prior_knowledge = ""
        if project_id and self.db_manager:
            try:
                from services.auto_fetch_service import AutoFetchService
                auto_fetch = AutoFetchService(api_key=self.api_key, db=self.db_manager)
                program_context = auto_fetch.on_question(project_id, question, background=True)

                if program_context:
                    context_parts = []
                    if program_context.get("target"):
                        context_parts.append(program_context["target"])
                    if program_context.get("indication"):
                        context_parts.append(program_context["indication"])

                    if context_parts:
                        yield {
                            "stage": "context",
                            "message": f"Program context: {' / '.join(context_parts)}",
                            "progress": 0.12,
                            "program_context": program_context
                        }
                        logger.info(f"Auto-fetch updated program context: {program_context}")
            except Exception as e:
                logger.warning(f"Auto-fetch failed: {e}")

            # Get prior knowledge from trusted_knowledge table for injection
            try:
                knowledge_dao = TrustedKnowledgeDAO(self.db_manager)
                prior_knowledge = knowledge_dao.get_context_for_question(project_id, question, max_entries=8)
                if prior_knowledge:
                    logger.info(f"Retrieved prior knowledge ({len(prior_knowledge)} chars) for prompt injection")
            except Exception as e:
                logger.warning(f"Prior knowledge retrieval failed: {e}")

        # Stage 2: Expert-First approach (two-channel search removed)
        # Supporting literature is now fetched in Stage 3b based on expert claims
        yield {"stage": "searching", "message": "Expert-First approach - consulting experts first...", "progress": 0.20}

        # Stage 3: Run Pass 1 (immediate expert responses)
        yield {"stage": "consulting", "message": "Getting expert perspectives...", "progress": 0.25}

        pass1_responses = {}
        scenario = get_scenario_for_question_type(question_type)

        def pass1_progress(expert_name, current, total):
            pct = 0.25 + (0.25 * current / total)
            if progress_callback:
                progress_callback("consulting", f"Consulting {expert_name}...", pct)

        try:
            pass1_responses = self.expert_service.run_pass1_immediate(
                clinical_question=question,
                selected_experts=experts,
                scenario=scenario,
                web_context=None,
                temperatures=None,
                progress_callback=pass1_progress
            )
        except Exception as e:
            logger.error(f"Pass 1 failed: {e}")
            pass1_responses = {exp: {"content": f"Error: {e}", "pass": 1} for exp in experts}

        yield {
            "stage": "consulting",
            "message": f"Gathered perspectives from {len(pass1_responses)} experts",
            "progress": 0.50,
            "pass1_complete": True,
            "responses": pass1_responses
        }

        # Stage 3b: Expert-First Flow - Extract claims and find supporting literature
        supporting_literature_result = None
        if EXPERT_FIRST_FLOW_AVAILABLE:
            yield {"stage": "claims", "message": "Extracting searchable claims from expert responses...", "progress": 0.52}

            try:
                # Extract claims with search hints from expert responses
                claim_extraction = extract_claims_from_responses(pass1_responses)
                searchable_claims = get_searchable_claims(claim_extraction)

                if searchable_claims:
                    logger.info(f"Extracted {len(searchable_claims)} searchable claims from expert responses")
                    yield {
                        "stage": "claims",
                        "message": f"Found {len(searchable_claims)} claims with search hints",
                        "progress": 0.53,
                        "claims_count": len(searchable_claims)
                    }

                    # Search for supporting literature (case series, cohorts)
                    yield {"stage": "supporting_lit", "message": "Finding supporting case series...", "progress": 0.54}

                    supporting_literature_result = search_supporting_literature(
                        claims=searchable_claims,
                        max_per_claim=3
                    )

                    if supporting_literature_result.total_papers > 0:
                        certainty = determine_certainty(supporting_literature_result)
                        logger.info(
                            f"Found {supporting_literature_result.total_papers} supporting papers "
                            f"for {supporting_literature_result.claims_with_support} claims. "
                            f"Certainty: {certainty}"
                        )
                        yield {
                            "stage": "supporting_lit",
                            "message": f"Found {supporting_literature_result.total_papers} supporting papers ({certainty})",
                            "progress": 0.55,
                            "supporting_literature": supporting_literature_result.to_dict(),
                            "certainty": certainty
                        }
                    else:
                        logger.info("No supporting literature found - recommendation based on expert consensus")
                        yield {
                            "stage": "supporting_lit",
                            "message": "No case series found - expert consensus basis",
                            "progress": 0.55
                        }
                else:
                    logger.info("No searchable claims extracted (experts may not have used search hints)")

            except Exception as e:
                logger.warning(f"Expert-first flow failed: {e}")
                # Continue without supporting literature - graceful degradation

        # Stage 4: Expert-First approach - no separate validation pass
        # Supporting literature is already fetched in Stage 3b based on expert claims
        # These variables kept for backwards compatibility with synthesis and results
        literature_citations = []
        search_results = {}
        validations = {}
        quality_report = {}
        clinical_trials = []

        # Extract citations from supporting literature if available
        if supporting_literature_result and hasattr(supporting_literature_result, 'papers'):
            for paper in supporting_literature_result.papers:
                if hasattr(paper, 'pmid') and paper.pmid:
                    literature_citations.append({
                        'pmid': paper.pmid,
                        'title': getattr(paper, 'title', ''),
                        'year': getattr(paper, 'year', ''),
                        'journal': getattr(paper, 'journal', ''),
                        'study_type': getattr(paper, 'study_type', ''),
                        'sample_size': getattr(paper, 'sample_size', None)
                    })

        yield {
            "stage": "evidence",
            "message": f"Expert-First approach complete - {len(literature_citations)} supporting papers",
            "progress": 0.75,
            "citations": literature_citations
        }

        # Stage 5: Synthesize recommendation
        yield {"stage": "synthesizing", "message": "Synthesizing recommendation...", "progress": 0.80}

        synthesis = self._synthesize_recommendation(
            question=question,
            question_type=question_type,
            pass1_responses=pass1_responses,
            validations=validations,
            literature_citations=literature_citations,
            additional_context=additional_context,
            project_id=project_id,
            prior_knowledge=prior_knowledge
        )

        yield {"stage": "synthesizing", "message": "Generating final recommendation...", "progress": 0.90}

        # Build final result
        elapsed = (datetime.now() - start_time).total_seconds()

        result = ResearchResult(
            question=question,
            question_type=question_type,
            recommendation=synthesis['recommendation'],
            confidence=synthesis['confidence'],
            response_mode="expert_first",  # Always Expert-First approach now
            key_findings=synthesis['key_findings'],
            evidence_summary={
                'citations': literature_citations,
                'clinical_trials': clinical_trials,
                'paper_count': len(literature_citations),
                'trial_count': len(clinical_trials),
                'program_context': program_context,
                'approach': 'expert_first',  # Document the approach used
                'supporting_claims': supporting_literature_result.total_papers if supporting_literature_result else 0,
            },
            expert_responses=pass1_responses,
            validations=validations,  # Empty dict in Expert-First approach
            follow_up_suggestions=synthesis.get('follow_ups', []),
            dissenting_views=synthesis.get('dissenting_views', []),
            metadata={
                'timestamp': datetime.now().isoformat(),
                'elapsed_seconds': elapsed,
                'model': self.model,
                'experts_consulted': experts
            },
            # Expert-First Flow: Supporting literature matched to claims
            supporting_literature=supporting_literature_result
        )

        yield {"stage": "complete", "message": "Research complete", "progress": 1.0, "result": result}
        
        # --- Persistence Layer ---
        if self.db_manager and project_id:
            try:
                logger.info(f"Saving results to project {project_id}...")
                
                # 1. Save Citations
                for cit in literature_citations:
                    # Handle object or dict
                    pmid = getattr(cit, 'pmid', cit.get('pmid')) if hasattr(cit, 'pmid') or isinstance(cit, dict) else None
                    if pmid:
                        # Convert object to dict if needed
                        cit_dict = cit.__dict__ if hasattr(cit, '__dict__') else cit
                        self.citation_dao.upsert_citation(cit_dict)
                        self.citation_dao.add_citation_to_project(project_id, pmid)

                # 2. Save Expert Discussion
                # Create discussion record
                disc_id = self.discussion_dao.create_discussion(
                    project_id=project_id,
                    clinical_question=question,
                    scenario=getattr(scenario, 'name', str(scenario)) if scenario else None,
                    experts=experts
                )
                
                # Save Pass 1 entries
                for expert, response in pass1_responses.items():
                    content = response.get('content', '')
                    self.discussion_dao.add_entry(
                        discussion_id=disc_id,
                        round_num=1,
                        expert_name=expert,
                        content=content
                    )
                    
                # Save Pass 2 validations (as Round 2 or distinct entries)
                for expert, val in validations.items():
                    val_content = f"**Validation Analysis**\n\nSupported Claims: {val.claims_supported}\nContradicted Claims: {val.claims_contradicted}\n\nAnalysis:\n{val.analysis}"
                    self.discussion_dao.add_entry(
                        discussion_id=disc_id,
                        round_num=2,
                        expert_name=expert,
                        content=val_content
                    )
                    
                logger.info(f"Saved discussion {disc_id} and citations to project {project_id}")
            except Exception as e:
                logger.error(f"Failed to save results to database: {e}")

        return result

    def _synthesize_recommendation(
        self,
        question: str,
        question_type: str,
        pass1_responses: Dict[str, Dict],
        validations: Dict[str, ValidationResult],
        literature_citations: List,
        additional_context: Optional[str] = None,
        project_id: Optional[int] = None,
        prior_knowledge: Optional[str] = None
    ) -> Dict:
        """
        Generate synthesized recommendation based on expert responses.

        Args:
            question: Original research question
            question_type: Detected question type
            pass1_responses: Expert Pass 1 responses
            validations: Pass 2 validation results
            literature_citations: Found literature
            additional_context: User-provided context
            project_id: Optional project ID for prior conclusions

        Returns:
            Dict with recommendation, confidence, key_findings, follow_ups
        """
        # Route to CDP-specific synthesis for CDP section types
        if question_type == "cdp_section":
            return self._synthesize_cdp_section(
                question=question,
                pass1_responses=pass1_responses,
                validations=validations,
                literature_citations=literature_citations,
                additional_context=additional_context
            )

        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=self.api_key, model=self.model)

        # Collect expert insights
        expert_summaries = []
        for expert_name, response in pass1_responses.items():
            content = response.get('content', '')[:1500]
            validation = validations.get(expert_name)
            
            # Format sequentially: Expert Response -> Validation
            expert_block = f"### {expert_name}\n\n{content}"
            
            if validation:
                val_stats = ""
                if hasattr(validation, 'claims_supported'):
                    val_stats = f" ({validation.claims_supported} supported, {validation.claims_contradicted} contradicted)"
                
                val_analysis = getattr(validation, 'analysis', '')
                if val_analysis:
                    expert_block += f"\n\n**Literature Validation{val_stats}**\n{val_analysis}"
            
            expert_summaries.append(expert_block)

        experts_text = "\n\n".join(expert_summaries)

        # Format literature citations for inline referencing
        citations_text = ""
        citation_count = 0
        if literature_citations:
            citation_list = []
            for i, cit in enumerate(literature_citations[:15], 1):
                if hasattr(cit, 'title'):
                    title = cit.title[:80]
                    pmid = cit.pmid
                    year = cit.year
                elif isinstance(cit, dict):
                    title = cit.get('title', 'Unknown')[:80]
                    pmid = cit.get('pmid', '')
                    year = cit.get('year', '')
                else:
                    continue
                citation_list.append(f"[{i}] {title}... (PMID: {pmid}, {year})")
                citation_count = i
            citations_text = "\n".join(citation_list)

        # Query knowledge store for prior facts
        knowledge_text = ""
        try:
            knowledge_store = KnowledgeStore()
            prior_knowledge = knowledge_store.get_relevant_knowledge_for_query(question)
            if prior_knowledge:
                knowledge_facts = []
                for persona, facts in prior_knowledge.items():
                    if facts:
                        knowledge_facts.append(f"**{persona}** (prior knowledge):")
                        for fact in facts[:3]:  # Limit per persona
                            knowledge_facts.append(f"  - {fact}")
                if knowledge_facts:
                    knowledge_text = "\n".join(knowledge_facts)
                    logger.info(f"Found {len(knowledge_facts)} prior knowledge facts")
        except Exception as e:
            logger.warning(f"Knowledge store query failed: {e}")

        # Query prior program conclusions (institutional memory)
        conclusions_text = ""
        if project_id:
            try:
                conclusions = knowledge_store.get_program_conclusions(project_id, limit=3)
                if conclusions:
                    conclusion_lines = ["**Prior Program Conclusions:**"]
                    for c in conclusions:
                        conclusion_lines.append(f"- Q: {c.get('question', '')[:80]}...")
                        conclusion_lines.append(f"  A: {c.get('conclusion', '')[:150]}...")
                    conclusions_text = "\n".join(conclusion_lines)
                    logger.info(f"Found {len(conclusions)} prior conclusions for project {project_id}")
            except Exception as e:
                logger.warning(f"Prior conclusions query failed: {e}")

        # Google Search Grounding (replacing Tavily)
        web_search_text = ""
        grounding_sources = []

        # Use Google Search Grounding for real-time web context
        if settings.ENABLE_GOOGLE_SEARCH_GROUNDING:
            try:
                from integrations.google_search import GoogleSearchClient
                google_client = GoogleSearchClient()

                if google_client.is_available():
                    # Build a search-optimized query from the question
                    search_query = f"{question} palliative surgery clinical evidence"

                    grounded_response = google_client.generate_with_grounding(
                        prompt=f"Find recent clinical evidence and guidelines relevant to: {question}",
                        system_instruction="You are a medical research assistant. Provide factual summaries of relevant clinical evidence.",
                        model_name="gemini-2.0-flash",
                        dynamic_threshold=settings.GOOGLE_SEARCH_GROUNDING_THRESHOLD
                    )

                    if grounded_response and grounded_response.sources:
                        web_lines = []
                        for i, source in enumerate(grounded_response.sources[:5], 1):
                            title = source.title[:80]
                            snippet = source.snippet[:200] if source.snippet else ""
                            url = source.url
                            web_lines.append(f"[W{i}] **{title}**\n{snippet}\nSource: {url}")
                            grounding_sources.append(source.to_dict())

                        web_search_text = "\n\n".join(web_lines)
                        logger.info(f"Google Search returned {len(grounding_sources)} sources")
                else:
                    logger.warning("Google Search not available (no API key)")
            except Exception as e:
                logger.warning(f"Google Search grounding failed: {e}")

        # Document library search (uploaded docs)
        doc_library_text = ""
        try:
            library = get_document_library()
            if library:
                doc_results = library.search(question, top_k=5)
                if doc_results:
                    doc_lines = []
                    for i, r in enumerate(doc_results, 1):
                        source = r.get('source', 'Unknown')[:40]
                        content = r.get('content', '')[:200]
                        doc_lines.append(f"[D{i}] {source}: {content}...")
                    doc_library_text = "\n".join(doc_lines)
                    logger.info(f"Document library returned {len(doc_results)} results")
        except Exception as e:
            logger.warning(f"Document library search failed: {e}")

        # Get synthesis template
        template = get_synthesis_template(question_type)
        type_info = QUESTION_TYPES.get(question_type, {})

        synthesis_prompt = f"""You are a senior drug development executive synthesizing expert panel input into an actionable recommendation.

**Question Type:** {type_info.get('name', 'Research Question')}
**Research Question:** {question}

**Expert Panel Responses:**
{experts_text}

**Available Literature ({len(literature_citations)} papers):**
{citations_text if citations_text else "No papers found"}

{f"**Web Search Results (real-time):**{chr(10)}{web_search_text}" if web_search_text else ""}

{f"**Prior Knowledge (from knowledge database):**{chr(10)}{knowledge_text}" if knowledge_text else ""}

{f"**Trusted Literature (auto-fetched from prior questions):**{chr(10)}{prior_knowledge}" if prior_knowledge else ""}

{f"**Prior Program Conclusions (institutional memory):**{chr(10)}{conclusions_text}" if conclusions_text else ""}

{f"**Uploaded Documents:**{chr(10)}{doc_library_text}" if doc_library_text else ""}

{f"**Additional Context:** {additional_context}" if additional_context else ""}

**Your Task:**
1. Synthesize the expert perspectives into a clear, actionable recommendation
2. **CITE evidence using [1], [2], etc.** from the Available Literature above
3. Identify the consensus position and any significant dissent
4. Assess overall confidence based on evidence quality and expert agreement
5. Suggest 2-3 follow-up questions that would strengthen the analysis

**IMPORTANT - Citation Requirements:**
- Every factual claim (efficacy %, safety data, prevalence) MUST cite a source
- Use [1], [2] for literature, [W1], [W2] for web sources, [D1], [D2] for uploaded documents
- Use [KB] for prior knowledge, [TL] for trusted literature, [PC] for prior program conclusions
- If no citation supports a claim, mark it as [ASSUMPTION] or [EXPERT OPINION]
- Be explicit about what is evidence-based vs expert judgment

**Output Format:**
{template}

Also include at the end:
## Follow-up Questions
- [2-3 suggested questions]

Be direct and executive-level in your synthesis. Focus on actionable recommendations with clear evidence grounding.
"""

        try:
            # Use generate_with_optional_grounding
            from core.llm_utils import generate_with_optional_grounding
            
            synthesis_text, grounding_sources = generate_with_optional_grounding(
                prompt=synthesis_prompt,
                system_instruction="You are a senior drug development executive providing synthesis and recommendations.",
                model=self.model,
                enable_grounding=settings.ENABLE_GOOGLE_SEARCH_GROUNDING,
                api_key=self.api_key
            )
            
            synthesis_text = synthesis_text or ""

            # Parse synthesis output
            result_dict = self._parse_synthesis(synthesis_text, pass1_responses)
            
            # Attach grounding sources if any
            if grounding_sources:
                 result_dict['grounding_sources'] = grounding_sources
                 
            return result_dict

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return self._fallback_synthesis(pass1_responses, validations)

    def _parse_synthesis(self, synthesis_text: str, pass1_responses: Dict) -> Dict:
        """Parse synthesis text into structured output."""
        import re

        # Extract recommendation (first major section or paragraph)
        recommendation = ""
        if "## Recommendation" in synthesis_text:
            match = re.search(r'## Recommendation\s*(.*?)(?=##|\Z)', synthesis_text, re.DOTALL)
            if match:
                recommendation = match.group(1).strip()
        else:
            # Use first 500 chars as recommendation
            recommendation = synthesis_text[:500].strip()

        # Extract confidence
        confidence = "MEDIUM"
        confidence_match = re.search(r'\b(HIGH|MEDIUM|LOW)\b', synthesis_text.upper())
        if confidence_match:
            confidence = confidence_match.group(1)

        # Extract key findings - improved extraction
        key_findings = self._extract_key_findings(synthesis_text, pass1_responses)

        # Extract follow-ups
        follow_ups = []
        followup_match = re.search(r'Follow-up.*?(?=##|\Z)', synthesis_text, re.DOTALL | re.IGNORECASE)
        if followup_match:
            bullets = re.findall(r'[-•\d.]\s*(.+?)(?=\n[-•\d.]|\n\n|\Z)', followup_match.group(0))
            follow_ups = [b.strip() for b in bullets[:3] if len(b.strip()) > 10]

        # Detect dissenting views - improved detection
        dissenting = self._detect_dissenting_views(pass1_responses)

        return {
            'recommendation': recommendation or "Please review the expert perspectives for detailed analysis.",
            'confidence': confidence,
            'key_findings': key_findings or ["See expert responses for detailed findings"],
            'follow_ups': follow_ups,
            'dissenting_views': dissenting
        }

    def _truncate_at_sentence(self, text: str, max_chars: int = 200) -> str:
        """
        Truncate text at sentence boundary, not mid-sentence.

        Args:
            text: Text to truncate
            max_chars: Maximum characters (soft limit - will extend to end of sentence)

        Returns:
            Truncated text ending at a sentence boundary
        """
        if len(text) <= max_chars:
            return text.strip()

        # Find sentence boundaries
        import re
        sentence_ends = list(re.finditer(r'[.!?](?:\s|$)', text))

        if not sentence_ends:
            # No sentence boundaries - truncate at word boundary
            truncated = text[:max_chars].rsplit(' ', 1)[0]
            return truncated.strip() + '...'

        # Find the last sentence end before or near max_chars
        best_end = 0
        for match in sentence_ends:
            if match.end() <= max_chars + 50:  # Allow 50 char overflow to complete sentence
                best_end = match.end()
            else:
                break

        if best_end > 0:
            return text[:best_end].strip()
        else:
            # First sentence is too long - truncate at word boundary
            truncated = text[:max_chars].rsplit(' ', 1)[0]
            return truncated.strip() + '...'

    def _extract_key_findings(self, synthesis_text: str, pass1_responses: Dict) -> List[str]:
        """
        Extract key findings from synthesis text and expert responses.

        Looks for:
        1. Explicit key findings section in synthesis
        2. Quantitative claims with PMIDs
        3. Clinical statements with key indicators
        """
        import re
        key_findings = []

        def is_valid_sentence(text: str) -> bool:
            """Check if text is a complete sentence (starts capital, ends punctuation)."""
            text = text.strip()
            if len(text) < 30:
                return False
            # Must start with capital letter
            if not text[0].isupper():
                return False
            # Must end with sentence punctuation
            if not text.rstrip().endswith(('.', '!', '?')):
                return False
            return True

        def is_meta_sentence(text: str) -> bool:
            """Check if sentence is meta/filler rather than clinical content."""
            text_lower = text.lower()
            skip_patterns = [
                'as a ', 'i would ', 'based on my ', 'here is ', 'the following',
                'in my opinion', 'from my perspective', 'i recommend that',
                'let me ', 'i should note', 'it is important to',
                'this is a ', 'there are several'
            ]
            return any(text_lower.startswith(p) or p in text_lower[:50] for p in skip_patterns)

        def extract_sentences(text: str) -> List[str]:
            """Extract complete sentences from text."""
            # Split on sentence boundaries but preserve the delimiter
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            valid = []
            for s in sentences:
                s = s.strip()
                if is_valid_sentence(s) and not is_meta_sentence(s):
                    # Ensure it ends with punctuation
                    if not s.endswith(('.', '!', '?')):
                        s = s + '.'
                    valid.append(s)
            return valid

        # Key clinical indicators for findings
        clinical_indicators = [
            'mortality', 'survival', 'recurrence', 'risk', 'recommend',
            'standard of care', 'evidence', 'high-level', 'local control',
            'prognosis', 'outcome', 'complication', 'SBRT', 'radiation',
            'surgical', 'palliative', 'quality of life', '%'
        ]

        # Method 1: Extract from explicit "Key Findings" or similar sections
        findings_patterns = [
            r'(?:Key Findings|Key Evidence|Key Points|Main Findings)[:\s]*\n?(.*?)(?=##|$)',
            r'(?:Supporting Evidence|Evidence Summary)[:\s]*\n?(.*?)(?=##|$)',
            r'(?:Summary|Conclusion)[:\s]*\n?(.*?)(?=##|Follow|$)'
        ]

        for pattern in findings_patterns:
            match = re.search(pattern, synthesis_text, re.DOTALL | re.IGNORECASE)
            if match:
                section = match.group(1)
                bullets = re.findall(r'[-•*]\s*(.+?)(?=\n[-•*]|\n\n|$)', section)
                for b in bullets[:5]:
                    b = b.strip()
                    # Ensure it's a complete sentence
                    if is_valid_sentence(b) and not is_meta_sentence(b):
                        finding = self._truncate_at_sentence(b, max_chars=400)
                        if finding and finding not in key_findings:
                            key_findings.append(finding)
                if key_findings:
                    break

        # Method 2: Extract sentences with clinical indicators from expert responses
        if len(key_findings) < 5:
            for expert, response in pass1_responses.items():
                content = response.get('content', '')
                sentences = extract_sentences(content)

                for sent in sentences:
                    sent_lower = sent.lower()
                    # Check for clinical indicators
                    has_indicator = any(ind in sent_lower for ind in clinical_indicators)
                    # Prefer sentences with PMIDs or percentages
                    has_evidence = re.search(r'PMID[:\s]*\d{7,8}|\d+%|\d+\.\d+', sent)

                    if has_indicator or has_evidence:
                        finding = self._truncate_at_sentence(sent, max_chars=400)
                        if finding and len(finding) > 40:
                            # Check not duplicate
                            is_dup = any(finding[:50].lower() in f.lower() for f in key_findings)
                            if not is_dup:
                                key_findings.append(f"{finding} ({expert})")
                                if len(key_findings) >= 5:
                                    break
                if len(key_findings) >= 5:
                    break

        # Deduplicate and limit
        seen = set()
        unique_findings = []
        for f in key_findings:
            # Use first 50 chars for dedup but keep full finding
            normalized = f.lower()[:50]
            if normalized not in seen:
                seen.add(normalized)
                unique_findings.append(f)

        return unique_findings[:5]

    def _detect_dissenting_views(self, pass1_responses: Dict) -> List[str]:
        """
        Detect TRUE dissenting views - experts who oppose the recommendation.

        Classification:
        - DISSENT: Recommends different action or contradicts key claim
        - ALTERNATIVE: Mentions other options (stent, SEMS) - NOT dissent
        - CAVEAT: Agrees but adds conditions - NOT dissent

        Returns list of formatted strings describing dissenting views.
        """
        import re
        dissenting_raw = []
        alternatives_raw = []  # Track alternatives separately

        # TRUE dissent indicators - expert actually opposes the recommendation
        true_dissent_phrases = [
            'should not proceed', 'do not recommend', 'recommend against',
            'contraindicated', 'inappropriate for this patient', 'would advise against',
            'disagree with the recommendation', 'risks outweigh benefits',
            'not a candidate', 'should avoid', 'futile'
        ]

        # Alternative mentions - NOT dissent, just context
        alternative_phrases = [
            'alternative', 'alternatively', 'versus', 'compared to',
            'stent', 'sems', 'stenting', 'endoscopic', 'non-surgical',
            'conservative management', 'medical management'
        ]

        for expert, response in pass1_responses.items():
            content = response.get('content', '')
            content_lower = content.lower()

            # Check for TRUE dissent - expert actually opposes
            has_true_dissent = False
            dissent_sentence = ""

            for phrase in true_dissent_phrases:
                if phrase in content_lower:
                    # Verify this is a genuine opposition, not just discussing risks
                    # Extract the sentence to check context
                    sentences = re.split(r'[.!?]\s+', content)
                    for sent in sentences:
                        if phrase in sent.lower():
                            # Check it's not negated or hypothetical
                            sent_lower = sent.lower()
                            if not any(neg in sent_lower for neg in ['if ', 'when ', 'unless ', 'would be ', 'could be ']):
                                has_true_dissent = True
                                dissent_sentence = sent.strip()[:500]
                                break
                    if has_true_dissent:
                        break

            if has_true_dissent:
                dissenting_raw.append({
                    'expert': expert,
                    'position': dissent_sentence,
                    'type': 'strong_dissent'
                })
            else:
                # Check if they mention alternatives (for separate display, not as dissent)
                mentions_alternative = any(phrase in content_lower for phrase in alternative_phrases)
                if mentions_alternative:
                    # Extract the alternative sentence
                    sentences = re.split(r'[.!?]\s+', content)
                    for sent in sentences:
                        sent_lower = sent.lower()
                        if any(phrase in sent_lower for phrase in alternative_phrases):
                            alternatives_raw.append({
                                'expert': expert,
                                'position': sent.strip()[:300],
                                'type': 'alternative'
                            })
                            break

        # Check for contradicting recommendations between experts (actual stance conflict)
        recommendations = {}
        for expert, response in pass1_responses.items():
            content = response.get('content', '').lower()
            # Only detect explicit recommendations, not risk discussions
            if any(phrase in content for phrase in ['i recommend', 'we recommend', 'should proceed', 'is appropriate']):
                recommendations[expert] = 'for'
            elif any(phrase in content for phrase in ['recommend against', 'should not proceed', 'is not appropriate']):
                recommendations[expert] = 'against'

        # Only flag if there's a clear minority dissent (not just different risk estimates)
        if recommendations:
            for_count = sum(1 for v in recommendations.values() if v == 'for')
            against_count = sum(1 for v in recommendations.values() if v == 'against')

            # Only flag if there's clear disagreement (minority < 30% of those with explicit stance)
            total_with_stance = for_count + against_count
            if total_with_stance >= 3:  # Need enough data
                minority_stance = 'against' if for_count > against_count else 'for'
                minority_count = min(for_count, against_count)

                if minority_count > 0 and minority_count / total_with_stance < 0.3:
                    for expert, stance in recommendations.items():
                        if stance == minority_stance and not any(d['expert'] == expert for d in dissenting_raw):
                            dissenting_raw.append({
                                'expert': expert,
                                'position': f"Takes minority position: recommends {stance} intervention",
                                'type': 'contradicting_recommendation'
                            })

        # Store alternatives in session state for separate display (not as dissent)
        import streamlit as st
        if alternatives_raw:
            st.session_state['_alternatives_discussed'] = alternatives_raw

        # Format only TRUE dissents
        return self._format_dissenting_views(dissenting_raw)

    def _classify_dissent_type(self, expert_response: str) -> str:
        """
        Classify the type of disagreement in an expert response.

        Returns one of:
        - STRONG_DISSENT: Recommends opposite intervention
        - CONDITIONAL_DISSENT: Agrees but with significant conditions
        - CAVEAT: Agrees, adds nuance (not true dissent)
        - RISK_EMPHASIS: Emphasizes risks but doesn't oppose
        """
        import re
        content_lower = expert_response.lower()

        # Strong dissent indicators - explicitly opposes the intervention
        strong_dissent_phrases = [
            'should not', 'do not recommend', 'recommend against',
            'contraindicated', 'inappropriate', 'futile',
            'disagree with', 'oppose', 'not appropriate'
        ]

        # Conditional dissent - agrees with conditions
        conditional_phrases = [
            'only if', 'only when', 'provided that', 'assuming',
            'would need', 'requires', 'depends on', 'contingent',
            'in selected patients', 'case by case'
        ]

        # Caveat indicators - adds nuance but agrees
        caveat_phrases = [
            'however', 'although', 'while', 'that said',
            'should note', 'important to consider', 'be aware',
            'caveat', 'limitation', 'nuance'
        ]

        # Risk emphasis - focuses on risks but doesn't oppose
        risk_phrases = [
            'high risk', 'significant risk', 'concern about',
            'worry about', 'mortality risk', 'complication rate',
            'careful consideration', 'weigh risks'
        ]

        # Check for strong dissent first (highest priority)
        has_strong_dissent = any(phrase in content_lower for phrase in strong_dissent_phrases)
        if has_strong_dissent:
            # But check if it's a conditional
            has_conditional = any(phrase in content_lower for phrase in conditional_phrases)
            if has_conditional:
                return 'CONDITIONAL_DISSENT'
            return 'STRONG_DISSENT'

        # Check for conditional agreement
        has_conditional = any(phrase in content_lower for phrase in conditional_phrases)
        if has_conditional:
            return 'CONDITIONAL_DISSENT'

        # Check for risk emphasis
        has_risk_emphasis = any(phrase in content_lower for phrase in risk_phrases)
        if has_risk_emphasis:
            # If also has positive language, it's just a caveat
            positive_phrases = ['recommend', 'support', 'favor', 'benefit', 'appropriate']
            has_positive = any(phrase in content_lower for phrase in positive_phrases)
            if has_positive:
                return 'CAVEAT'
            return 'RISK_EMPHASIS'

        # Default to caveat if has any caveat phrases
        has_caveat = any(phrase in content_lower for phrase in caveat_phrases)
        if has_caveat:
            return 'CAVEAT'

        return 'CAVEAT'  # Default

    def _format_dissenting_views(self, dissenting_raw: List[Dict]) -> List[str]:
        """
        Format raw dissenting view dicts into readable prose strings.

        Properly distinguishes:
        - STRONG_DISSENT: "Expert X disagrees: [position]"
        - CONDITIONAL_DISSENT: "Expert X supports with conditions: [position]"
        - CAVEAT: "Expert X notes: [position]" (not true dissent)
        - RISK_EMPHASIS: "Expert X emphasizes risk: [position]"

        Args:
            dissenting_raw: List of dicts with 'expert', 'position', 'type' keys

        Returns:
            List of formatted strings suitable for display
        """
        formatted = []

        # Reclassify each dissent using improved classifier
        for d in dissenting_raw:
            expert = d.get('expert', 'Unknown Expert')
            position = d.get('position', '')
            original_type = d.get('type', 'caution')

            # Reclassify using the new classifier
            if position:
                refined_type = self._classify_dissent_type(position)
            else:
                refined_type = 'CAVEAT'

            # Map old types to new classification
            type_mapping = {
                'strong_dissent': 'STRONG_DISSENT',
                'caution': refined_type,  # Use refined classification
                'contradicting_recommendation': 'CONDITIONAL_DISSENT'
            }

            dissent_type = type_mapping.get(original_type, refined_type)

            # Format based on classification
            type_labels = {
                'STRONG_DISSENT': 'disagrees',
                'CONDITIONAL_DISSENT': 'supports with conditions',
                'CAVEAT': 'notes',
                'RISK_EMPHASIS': 'emphasizes risks'
            }

            label = type_labels.get(dissent_type, 'notes')

            # Format as readable prose - only flag true dissent prominently
            if position:
                if dissent_type == 'STRONG_DISSENT':
                    formatted.append(f"**{expert}** ({label}): {position}")
                elif dissent_type == 'CONDITIONAL_DISSENT':
                    formatted.append(f"**{expert}** ({label}): {position}")
                else:
                    # Caveats are less prominent
                    formatted.append(f"*{expert}* ({label}): {position}")
            else:
                formatted.append(f"**{expert}** ({label})")

        return formatted

    def get_dissent_summary(self, dissenting_views: List[str]) -> Dict[str, any]:
        """
        Summarize dissenting views by classification.

        Returns:
            Dict with counts by type and summary text
        """
        summary = {
            'strong_dissent_count': 0,
            'conditional_count': 0,
            'caveat_count': 0,
            'total_count': len(dissenting_views),
            'has_true_dissent': False,
            'summary_text': ''
        }

        for view in dissenting_views:
            view_lower = view.lower()
            if 'disagrees' in view_lower:
                summary['strong_dissent_count'] += 1
            elif 'conditions' in view_lower:
                summary['conditional_count'] += 1
            else:
                summary['caveat_count'] += 1

        summary['has_true_dissent'] = summary['strong_dissent_count'] > 0

        # Generate summary text
        if summary['strong_dissent_count'] > 0:
            summary['summary_text'] = (
                f"{summary['strong_dissent_count']} expert(s) disagree with the recommendation. "
                "Review their concerns before proceeding."
            )
        elif summary['conditional_count'] > 0:
            summary['summary_text'] = (
                f"Expert(s) support the recommendation with {summary['conditional_count']} condition(s) noted."
            )
        elif summary['caveat_count'] > 0:
            summary['summary_text'] = (
                f"Experts agree with the recommendation. {summary['caveat_count']} caveat(s) noted for context."
            )
        else:
            summary['summary_text'] = "No dissenting views identified."

        return summary

    def _fallback_synthesis(self, pass1_responses: Dict, validations: Dict) -> Dict:
        """Fallback synthesis when LLM synthesis fails."""
        # Count validation results
        total_supported = 0
        total_contradicted = 0

        for v in validations.values():
            if hasattr(v, 'claims_supported'):
                total_supported += v.claims_supported
                total_contradicted += v.claims_contradicted

        # Determine confidence based on validation
        if total_contradicted > total_supported:
            confidence = "LOW"
        elif total_supported > 3:
            confidence = "HIGH"
        else:
            confidence = "MEDIUM"

        # Extract key points from each expert
        key_findings = []
        for expert, response in pass1_responses.items():
            content = response.get('content', '')
            if content and not content.startswith('Error'):
                # Take first sentence
                first_sentence = content.split('.')[0][:150]
                key_findings.append(f"{expert}: {first_sentence}")

        return {
            'recommendation': "Review the expert perspectives below for detailed analysis and recommendations.",
            'confidence': confidence,
            'key_findings': key_findings[:5],
            'follow_ups': [
                "What additional data would strengthen this analysis?",
                "Are there any regulatory considerations we should address?",
                "What is the competitive timeline for this decision?"
            ],
            'dissenting_views': []
        }

    def _synthesize_cdp_section(
        self,
        question: str,
        pass1_responses: Dict[str, Dict],
        validations: Dict[str, ValidationResult],
        literature_citations: List,
        additional_context: Optional[str] = None
    ) -> Dict:
        """
        Generate CDP section synthesis using Clinical Development Strategist persona.

        This method produces structured, executive-friendly CDP narrative that can
        be directly incorporated into a Clinical Development Plan document.

        Args:
            question: Research question (will become section focus)
            pass1_responses: Expert Pass 1 responses
            validations: Pass 2 validation results
            literature_citations: Found literature
            additional_context: User-provided context

        Returns:
            Dict with recommendation (CDP section), confidence, key_findings, follow_ups
        """
        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=self.api_key, model=self.model)
        from core.question_templates import get_synthesis_template

        # Collect expert insights with validation status
        expert_summaries = []
        for expert_name, response in pass1_responses.items():
            content = response.get('content', '')[:1500]
            validation = validations.get(expert_name)
            
            # Format sequentially
            expert_block = f"### {expert_name}\n\n{content}"
            
            if validation:
                val_stats = ""
                if hasattr(validation, 'claims_supported'):
                    val_stats = f" ({validation.claims_supported} supported, {validation.claims_contradicted} contradicted)"
                    
                val_analysis = getattr(validation, 'analysis', '')
                if val_analysis:
                    expert_block += f"\n\n**Literature Validation{val_stats}**\n{val_analysis}"
            
            expert_summaries.append(expert_block)

        experts_text = "\n\n".join(expert_summaries)

        # Build citation summary
        citation_summary = ""
        if literature_citations:
            citation_lines = []
            for i, cit in enumerate(literature_citations[:10]):
                title = getattr(cit, 'title', cit.get('title', '')) if hasattr(cit, 'title') or isinstance(cit, dict) else ''
                pmid = getattr(cit, 'pmid', cit.get('pmid', '')) if hasattr(cit, 'pmid') or isinstance(cit, dict) else ''
                citation_lines.append(f"- {title[:80]}... (PMID: {pmid})")
            citation_summary = "\n".join(citation_lines)

        # Get CDP section template
        template = get_synthesis_template("cdp_section")

        # Clinical Development Strategist synthesis prompt
        cdp_prompt = f"""You are a Clinical Development Strategist with 20+ years experience leading INDs and NDAs.

**YOUR ROLE:** CDP Author & Cross-Functional Integrator
**YOUR TASK:** Synthesize the expert panel input into a polished CDP section.

**WRITING STYLE:**
- Executive-friendly, structured prose suitable for governance review
- Lead with strategic recommendations, support with evidence
- Use tables for complex comparisons
- Flag risks prominently with proposed mitigations
- Include concrete decision points and dependencies
- Every claim must cite its source (expert name or PMID)

**RESEARCH QUESTION (Section Focus):**
{question}

**EXPERT PANEL INPUT:**
{experts_text}

**SUPPORTING LITERATURE ({len(literature_citations)} papers found):**
{citation_summary if citation_summary else "No additional literature found"}

{f"**ADDITIONAL CONTEXT:** {additional_context}" if additional_context else ""}

**OUTPUT FORMAT:**
Generate a complete CDP section following this structure:
{template}

**IMPORTANT:**
- Write in third person, formal CDP style
- Ensure all recommendations are traceable to expert input or literature
- Explicitly state confidence level (HIGH/MEDIUM/LOW) with rationale
- Include specific next steps with owners where possible
"""

        try:
            client = get_llm_client(api_key=self.api_key, model=self.model)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Clinical Development Strategist writing formal CDP sections for drug development programs."
                    },
                    {"role": "user", "content": cdp_prompt}
                ],
                max_completion_tokens=3000
            )

            synthesis_text = response.choices[0].message.content or ""

            # Parse CDP-specific synthesis
            return self._parse_cdp_synthesis(synthesis_text, pass1_responses)

        except Exception as e:
            logger.error(f"CDP synthesis failed: {e}")
            return self._fallback_synthesis(pass1_responses, validations)

    def _parse_cdp_synthesis(self, synthesis_text: str, pass1_responses: Dict) -> Dict:
        """Parse CDP synthesis text into structured output."""
        import re

        # For CDP sections, the entire output IS the recommendation (the section content)
        recommendation = synthesis_text.strip()

        # Extract confidence
        confidence = "MEDIUM"
        confidence_match = re.search(r'(?:Confidence|CONFIDENCE)[:\s]*\[?(HIGH|MEDIUM|LOW)\]?', synthesis_text, re.IGNORECASE)
        if confidence_match:
            confidence = confidence_match.group(1).upper()

        # Extract key findings from the "Key Evidence" or "Strategic Rationale" sections
        key_findings = []
        evidence_match = re.search(r'(?:Key Evidence|Strategic Rationale).*?(?=###|\Z)', synthesis_text, re.DOTALL | re.IGNORECASE)
        if evidence_match:
            bullets = re.findall(r'[-•]\s*(.+?)(?=\n[-•]|\n\n|\n###|\Z)', evidence_match.group(0))
            key_findings = [b.strip()[:200] for b in bullets[:5] if b.strip()]

        # Extract decision points as follow-ups
        follow_ups = []
        decision_match = re.search(r'Decision Points.*?(?=###|\Z)', synthesis_text, re.DOTALL | re.IGNORECASE)
        if decision_match:
            bullets = re.findall(r'[-•\[\]]\s*(.+?)(?=\n[-•\[]|\n\n|\n###|\Z)', decision_match.group(0))
            follow_ups = [b.strip()[:150] for b in bullets[:3] if len(b.strip()) > 10]

        # Detect dissenting views from risk section
        dissenting = []
        for expert, response in pass1_responses.items():
            content = response.get('content', '').lower()
            if any(word in content for word in ['disagree', 'caution', 'significant risk', 'concern', 'oppose', 'insufficient']):
                dissenting.append(expert)

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'key_findings': key_findings or ["See CDP section above for detailed evidence"],
            'follow_ups': follow_ups or ["What additional studies are needed to address evidence gaps?"],
            'dissenting_views': dissenting
        }

    def handle_follow_up(
        self,
        follow_up_question: str,
        previous_result: ResearchResult,
        chat_history: List[Dict]
    ) -> Generator[Dict, None, str]:
        """
        Handle inline follow-up question using existing context.

        Args:
            follow_up_question: The user's follow-up question
            previous_result: The previous ResearchResult
            chat_history: Previous chat messages

        Yields:
            Progress events

        Returns:
            Follow-up response text
        """
        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=self.api_key, model=self.model)

        yield {"stage": "processing", "message": "Processing follow-up..."}

        # Build context from previous result
        context_parts = [
            f"Previous Question: {previous_result.question}",
            f"Question Type: {previous_result.question_type}",
            f"Recommendation: {previous_result.recommendation}",
            f"Confidence: {previous_result.confidence}",
        ]

        # Add expert summaries
        for expert, response in previous_result.expert_responses.items():
            content = response.get('content', '')[:500]
            context_parts.append(f"\n{expert}:\n{content}")

        context = "\n".join(context_parts)

        # Build chat history string
        history_text = ""
        if chat_history:
            for msg in chat_history[-5:]:
                role = msg.get('role', 'user')
                content = msg.get('content', '')[:300]
                history_text += f"\n{role.upper()}: {content}"

        try:
            client = get_llm_client(api_key=self.api_key, model=self.model)

            logger.info(f"[FOLLOW-UP] Calling LLM. Model: {self.model}, Question: {follow_up_question[:100]}")

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a drug development research assistant answering follow-up questions.

Previous Analysis Context:
{context}

Previous Conversation:
{history_text}

Answer the follow-up question based on the analysis context. Be specific and reference the relevant expert perspectives when applicable."""
                    },
                    {"role": "user", "content": follow_up_question}
                ],
                max_tokens=4500  # Larger for detailed follow-up responses
            )

            answer = response.choices[0].message.content if response.choices else None

            if not answer:
                logger.warning(f"LLM returned empty response for follow-up. Model: {self.model}")
                answer = "I couldn't generate a response. Please try rephrasing your question."

            logger.info(f"Follow-up response generated: {len(answer)} chars")
            yield {"stage": "complete", "message": "Response ready", "response": answer}

            return answer

        except Exception as e:
            logger.error(f"Follow-up handling failed: {e}", exc_info=True)
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            yield {"stage": "error", "message": error_msg}
            return error_msg
