"""
Expert Panel UI Module

Handles the Expert Panel Discussion interface, including:
- Research question input
- Expert selection and persona management
- Discussion execution (rounds)
- Meta-synthesis
- Gap analysis and conflict detection
- Hypothesis tracking
- Interactive Q&A chat
"""

import streamlit as st
import json
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

from config import settings
from core.database import ExpertDiscussionDAO, CitationDAO, ExpertCorrectionDAO
from services.expert_service import ExpertDiscussionService
import difflib
from core.priors_manager import PriorsManager
from core.working_memory import WorkingMemory, get_working_memory

# Import expert selector
from ui.expert_selector import render_expert_selector, render_expert_selector_compact

# Import services layer
from services.analysis_service import AnalysisService
from services.chat_service import ChatService
from services.recommendation_service import RecommendationService, get_recommendation_service

# Import analysis modules
try:
    from core.analysis.gap_analyzer import GapAnalyzer, analyze_panel_discussion, EXPECTED_TOPICS
    from core.analysis.conflict_detector import ConflictDetector, DecisionSynthesizer, detect_panel_conflicts
    from core.analysis.expert_enhancement import enhance_expert_for_question, detect_context
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False

# Import Google Search Grounding availability
try:
    from integrations.google_search import GoogleSearchClient
    GOOGLE_SEARCH_AVAILABLE = GoogleSearchClient().is_available()
except ImportError:
    GOOGLE_SEARCH_AVAILABLE = False

# Models that support temperature control
# Note: GPT-5 series uses reasoning_effort instead of temperature
TEMP_SUPPORTED_MODELS = [
    "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
    "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
    "claude-3.5-sonnet", "claude-3.5-haiku",
    "gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash",
    "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro",
    "gemini-3-pro-preview",
]

# Import GDG (Guideline Development Group) modules
try:
    from gdg import (
        GDG_PERSONAS,
        get_gdg_prompts,
        get_all_expert_names,
        get_default_experts,
        CLINICAL_SCENARIOS,
        get_enhanced_expert_prompts,
        format_evidence_context,
        call_expert,
        call_expert_stream,
        auto_select_papers_for_experts,
        export_discussion_to_markdown,
        generate_perspective_questions,
        synthesize_expert_responses,
        extract_hypotheses_from_discussion,
        process_discussion_for_knowledge,
        ResponseValidator,
    )
except ImportError as e:
    # We will handle this gracefully in the render function
    pass

# Import Evidence Corpus for citation validation (GRADE v2.0)
try:
    from core.evidence_corpus import get_corpus_from_session, EvidenceCorpus
    from gdg.response_validator import get_citation_warnings
    EVIDENCE_CORPUS_AVAILABLE = True
except ImportError:
    EVIDENCE_CORPUS_AVAILABLE = False


def _render_hallucination_warnings(response_text: str, expert_name: str, corpus: 'EvidenceCorpus' = None):
    """
    Render hallucination and citation quality warnings for an expert response.

    Args:
        response_text: The expert's response text
        expert_name: Name of the expert
        corpus: EvidenceCorpus for validation (if None, gets from session)
    """
    if not EVIDENCE_CORPUS_AVAILABLE:
        return

    if corpus is None:
        corpus = get_corpus_from_session()

    if corpus is None or not corpus.included_pmids:
        return  # No corpus to validate against

    # Get warnings using response_validator
    warnings = get_citation_warnings(
        responses={expert_name: response_text},
        corpus=corpus
    )

    if not warnings:
        return

    # Group warnings by severity
    errors = [w for w in warnings if w.get('severity') == 'error']
    warnings_list = [w for w in warnings if w.get('severity') == 'warning']
    info_list = [w for w in warnings if w.get('severity') == 'info']

    # Render warning badges
    if errors:
        for err in errors:
            st.markdown(f"""
            <div style="background: #f8d7da; border-left: 4px solid #dc3545;
                        padding: 8px 12px; margin: 4px 0; border-radius: 4px;">
                🔴 <strong>{err.get('type', 'Error')}:</strong> {err.get('message', '')}
            </div>
            """, unsafe_allow_html=True)

    if warnings_list:
        for warn in warnings_list:
            st.markdown(f"""
            <div style="background: #fff3cd; border-left: 4px solid #ffc107;
                        padding: 8px 12px; margin: 4px 0; border-radius: 4px;">
                🟡 <strong>{warn.get('type', 'Warning')}:</strong> {warn.get('message', '')}
            </div>
            """, unsafe_allow_html=True)

    if info_list:
        for info in info_list:
            st.markdown(f"""
            <div style="background: #d1ecf1; border-left: 4px solid #17a2b8;
                        padding: 8px 12px; margin: 4px 0; border-radius: 4px;">
                🔵 <strong>{info.get('type', 'Note')}:</strong> {info.get('message', '')}
            </div>
            """, unsafe_allow_html=True)

def _get_text_from_context(item) -> str:
    """Safely extract text from a context item (dict or object)."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        # Try common keys
        for key in ['text', 'content', 'page_content', 'chunk']:
            if key in item:
                return item[key]
        return str(item)
    if hasattr(item, 'text'):
        return item.text
    if hasattr(item, 'page_content'):
        return item.page_content
    return str(item)


def build_chat_context(citations, expert_discussion):
    """
    Build context string from papers and prior discussion.

    Delegates to ChatService for context building.
    """
    chat_service = ChatService(api_key=settings.OPENAI_API_KEY)
    return chat_service.build_context(citations, expert_discussion)

def render_expert_panel(expert_discussion_dao: ExpertDiscussionDAO, citation_dao: CitationDAO):
    """
    Renders the Expert Panel Discussion tab.

    Args:
        expert_discussion_dao: Data Access Object for expert discussions
        citation_dao: Data Access Object for citations
    """
    st.title("👥 Expert Panel")

    # Expert Selection - ALWAYS visible (before project check)
    # Default to GDG mode for expert selection
    mode_key = st.session_state.get('last_mode', 'cdp')

    with st.expander("👥 Select GDG Experts", expanded=True):
        selected_experts = render_expert_selector(
            key_prefix="panel_expert",
            default_selection=get_default_experts(mode=mode_key),
            max_experts=12,
            min_experts=2,
            show_presets=True
        )

    if not st.session_state.current_project_name:
        st.info("👈 Select a project to begin the discussion")
        return

    # Direct Input for Expert Panel
    st.caption("Background context or data.")

    # Research Mode selector - gated for advanced tools
    if settings.ENABLE_ADVANCED_TOOLS:
        mode_col1, mode_col2 = st.columns([1, 2])
        with mode_col1:
            research_mode = st.radio(
                "Mode",
                ["GDG (Guideline Dev)", "Discovery (Hypothesis)", "Strategy (Planning)"],
                index=0,
                key="research_mode_toggle_top",
                horizontal=True
            )

        # Determine Mode Key
        if "Discovery" in research_mode:
            mode_key = 'discovery'
        elif "Strategy" in research_mode:
            mode_key = 'strategy'
        else:
            mode_key = 'cdp'
    else:
        # Default to GDG mode when advanced tools are disabled
        mode_key = 'cdp'

    st.session_state.last_mode = mode_key



    # Context Input (File Upload + Text)
    with st.expander("📝 Add Context (Files or Text)", expanded=False):
        context_tab1, context_tab2 = st.tabs(["📤 Upload File", "✏️ Paste Text"])
        
        with context_tab1:
            uploaded_file = st.file_uploader(
                "Upload background (PDF/DOCX/TXT)", 
                type=['pdf', 'docx', 'txt'],
                key="panel_context_upload"
            )
            if uploaded_file:
                # Process strictly for this session's context
                try:
                    # Import ingestion if not already available
                    from core.document_ingestion import ingest_file
                    with st.spinner("Reading file..."):
                        doc = ingest_file(uploaded_file, uploaded_file.name)
                        st.session_state.panel_file_context = doc.content
                        st.success(f"Loaded: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Failed to read file: {e}")
        
        with context_tab2:
            direct_context = st.text_area(
                "Context", 
                placeholder="Paste background info, rules, or data...",
                height=150,
                key="panel_direct_context",
                label_visibility="collapsed"
            )

    # Combine sources
    file_context = st.session_state.get('panel_file_context', '')
    combined_context = f"{file_context}\n\n{direct_context}".strip()
    if combined_context:
        st.info("✅ Context loaded")
            
    # Clinical Question Input
    clinical_question_raw = st.text_area(
        "Research Question",
        placeholder="What is the evidence regarding...",
        height=80,
        key="expert_clinical_question",
        max_chars=5000
    )
    
    # Input validation logic (hidden)
    clinical_question = clinical_question_raw.strip()
    
    # Initialize Service
    if 'expert_service' not in st.session_state:
        st.session_state.expert_service = ExpertDiscussionService(api_key=settings.OPENAI_API_KEY)

    # Sidebar controls
    # Sidebar controls
    # col1, col2 = st.columns([1, 2]) -- Removed, replaced by full width layout above
    # Scenario Selection (Chips Layout)
    st.caption("Select Clinical Scenario:")
    scenarios = list(CLINICAL_SCENARIOS.keys())
    if 'selected_scenario' not in st.session_state:
        st.session_state.selected_scenario = scenarios[0]

    # Layout: First 3, then remaining
    row1_cols = st.columns(3)
    for i, sc_key in enumerate(scenarios[:3]):
        with row1_cols[i]:
            is_active = (st.session_state.selected_scenario == sc_key)
            if st.button(
                sc_key,  # Use the key itself as the name
                key=f"sc_chip_{sc_key}",
                type="primary" if is_active else "secondary",
                use_container_width=True
            ):
                st.session_state.selected_scenario = sc_key
                st.rerun()

    if len(scenarios) > 3:
        row2_cols = st.columns(len(scenarios) - 3)
        for i, sc_key in enumerate(scenarios[3:]):
            with row2_cols[i]:
                is_active = (st.session_state.selected_scenario == sc_key)
                if st.button(
                    sc_key,  # Use the key itself as the name
                    key=f"sc_chip_{sc_key}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True
                ):
                    st.session_state.selected_scenario = sc_key
                    st.rerun()
                
    scenario = st.session_state.selected_scenario

    if not settings.OPENAI_API_KEY:
        st.error("Key Missing")
        st.stop()

    # HITL Controls (Simplified)
    project_id = st.session_state.get('current_project_id', 'default')
    working_memory = get_working_memory(str(project_id)) if project_id else None

    with st.expander("🎛️ Mission Control", expanded=False):
        mc_tabs = st.tabs(["📚 Inject Evidence", "🧠 Working Memory", "⚙️ Parameters", "🔍 Retrieval"])
        
        with mc_tabs[0]:
            search_results = st.session_state.get('search_results')
            citations = search_results.get('citations') if search_results else None
            
            if citations:
                 pmids = [c.pmid if hasattr(c,'pmid') else c.get('pmid') for c in citations]
                 selected_pmids = st.multiselect("Force-inject papers", pmids)
                 
                 if selected_pmids:
                     injected = [c for c in citations if (c.pmid if hasattr(c,'pmid') else c.get('pmid')) in selected_pmids]
                     # Convert to dicts.. (omitted for brevity, assume logic remains)
                     st.session_state.injected_evidence = injected 
            else:
                 st.caption("No search results available")

        # TAB 2: Working Memory
        with mc_tabs[1]:
            st.markdown("**Persistent constraints and facts:**")
            st.caption("These are injected into EVERY expert's prompt, across all rounds")

            if working_memory:
                # Add new memory entry
                with st.form("add_memory_form", clear_on_submit=True):
                    mem_content = st.text_input("Add constraint/fact/correction", placeholder="e.g., This drug is IV only, not oral")
                    mem_category = st.selectbox("Type", ["constraint", "fact", "correction"])
                    submit_mem = st.form_submit_button("➕ Add to Memory")
                    if submit_mem and mem_content.strip():
                        working_memory.add(mem_content.strip(), mem_category, "human")
                        st.success(f"Added to working memory!")
                        st.rerun()

                # Display current memory
                if working_memory.entries:
                    st.markdown(f"**Current Memory ({len(working_memory.entries)} items):**")
                    for i, entry in enumerate(working_memory.entries):
                        col_mem, col_del = st.columns([5, 1])
                        with col_mem:
                            category_icon = {"constraint": "🔒", "fact": "📌", "correction": "⚠️"}.get(entry.category, "•")
                            st.caption(f"{category_icon} [{entry.category.upper()}] {entry.content[:80]}{'...' if len(entry.content) > 80 else ''}")
                        with col_del:
                            if st.button("❌", key=f"del_mem_{i}", help="Remove"):
                                working_memory.remove(i)
                                st.rerun()

                    if st.button("🗑️ Clear All Memory", type="secondary"):
                        working_memory.clear()
                        st.success("Working memory cleared")
                        st.rerun()
                else:
                    st.info("No working memory entries. Add constraints or facts above.")
            else:
                st.warning("Working memory not available (no project selected)")

        # TAB 3: Parameters (Temperature)
        with mc_tabs[2]:
            st.markdown("**Expert parameters:**")

            # Model-aware temperature controls
            current_model = settings.EXPERT_MODEL
            if current_model in TEMP_SUPPORTED_MODELS:
                st.caption(f"Temperature controls for {current_model}")
                # expert_temperatures initialized in core/state_manager.py
                
                # Mode-aware default
                default_temp = 0.9 if mode_key == 'discovery' else 0.4
                st.caption(f"Default for {mode_key.title()} Mode: {default_temp}")

                if selected_experts:
                    for expert in selected_experts:
                        current_temp = st.session_state.expert_temperatures.get(expert, default_temp)
                        new_temp = st.slider(
                            f"{expert}",
                            min_value=0.0,
                            max_value=1.0,
                            value=current_temp,
                            step=0.1,
                            key=f"temp_slider_{expert}",
                            help="Lower = more rigorous/conservative, Higher = more creative"
                        )
                        st.session_state.expert_temperatures[expert] = new_temp
                else:
                    st.info("Select experts above to configure their parameters")
            # Hide parameters tab content when model doesn't support temperature

        # TAB 4: Retrieval Settings
        with mc_tabs[3]:
            st.markdown("**Retrieval settings:**")

            # Initialize session state for retrieval toggles
            if 'enable_web_search' not in st.session_state:
                st.session_state.enable_web_search = settings.ENABLE_WEB_FALLBACK
            if 'enable_hyde' not in st.session_state:
                st.session_state.enable_hyde = getattr(settings, 'ENABLE_HYDE', True)
            if 'enable_query_expansion' not in st.session_state:
                st.session_state.enable_query_expansion = getattr(settings, 'ENABLE_QUERY_EXPANSION', True)
            if 'deep_research_mode' not in st.session_state:
                st.session_state.deep_research_mode = False

            # Local RAG status
            st.caption(f"Local RAG: {'✅ Enabled' if settings.ENABLE_LOCAL_RAG else '❌ Disabled'}")

            # Deep Research Mode (STORM) toggle
            st.session_state.deep_research_mode = st.toggle(
                "🌪️ Deep Research Mode (STORM)",
                value=st.session_state.deep_research_mode,
                help="Enable iterative 'Draft -> Clarify -> Revise' loops for deeper insights (slower but higher quality)."
            )

            # Two-Pass Mode (Perplexity-style)
            st.markdown("---")
            st.markdown("**⚡ Two-Pass Mode (Perplexity-style)**")
            st.session_state.two_pass_mode = st.toggle(
                "⚡ Enable Two-Pass Responses",
                value=st.session_state.get('two_pass_mode', True),
                help="Get immediate answers (Pass 1), then literature validation (Pass 2). Faster feedback!"
            )
            if st.session_state.two_pass_mode:
                st.session_state.auto_search_max = st.slider(
                    "Auto-search max papers",
                    min_value=5,
                    max_value=50,
                    value=st.session_state.get('auto_search_max', 20),
                    help="Maximum papers to auto-search when no literature is loaded"
                )
            st.markdown("---")

            # Google Search Grounding toggle
            if GOOGLE_SEARCH_AVAILABLE:
                st.session_state.enable_web_search = st.checkbox(
                    "🌐 Enable Google Search Grounding",
                    value=st.session_state.enable_web_search,
                    help="Ground responses with real-time Google Search results"
                )
            else:
                st.info("🌐 Google Search unavailable (GOOGLE_API_KEY not set)")

            # HyDE toggle
            st.session_state.enable_hyde = st.checkbox(
                "🔮 Enable HyDE (Hypothetical Document Embeddings)",
                value=st.session_state.enable_hyde,
                help="Generate hypothetical answers to improve retrieval accuracy"
            )

            # Query expansion toggle
            st.session_state.enable_query_expansion = st.checkbox(
                "📊 Enable Query Expansion",
                value=st.session_state.enable_query_expansion,
                help="Expand queries with synonyms for better recall"
            )

    st.markdown("---")

    # Perspective-Guided Questions
    if clinical_question and selected_experts:
        with st.expander("🔎 Generate Perspective Questions (STORM-style)", expanded=False):
            st.caption("Generate domain-specific questions each expert would want answered before discussion")
            if st.button("Generate Questions by Expert Perspective"):
                with st.spinner("Generating perspective-specific questions..."):
                    try:
                        perspective_qs = generate_perspective_questions(
                            clinical_question=clinical_question,
                            expert_names=selected_experts,
                            openai_api_key=settings.OPENAI_API_KEY
                        )
                        st.session_state.perspective_questions = perspective_qs
                    except Exception as e:
                        st.error(f"Failed to generate: {e}")

            if st.session_state.perspective_questions:
                for expert, questions in st.session_state.perspective_questions.items():
                    if questions:
                        st.markdown(f"**{expert}:**")
                        for q in questions: st.markdown(f"- {q}")

    st.markdown("---")

    # =========================================================================
    # HIDDEN: Research Tools (Hypothesis Lab + Strategy War Room)
    # These features are commented out to focus on CDP development.
    # Uncomment to restore, or move to a separate "Research Tools" app.
    # =========================================================================

    # # SYSTEM MODE TOGGLE
    # mode_col1, mode_col2 = st.columns([1, 2])
    # with mode_col1:
    #     st.markdown("### 🔭 Research Mode")
    #     research_mode = st.radio(
    #         "Select capability:",
    #         ["Discovery (Hypothesis Gen)", "Strategy (Clinical Planning)"],
    #         index=1 if st.session_state.get('last_mode') == 'strategy' else 0,
    #         key="research_mode_toggle",
    #         help="Discovery: Generate novel scientific hypotheses.\nStrategy: Plan clinical development and INDs."
    #     )
    # st.markdown("---")

    # Hypothesis Lab and Strategy War Room - gated for advanced tools
    if settings.ENABLE_ADVANCED_TOOLS and mode_key == "discovery":  # HYPOTHESIS LAB
        # MODE 1: HYPOTHESIS LAB (Discovery)
        with st.expander("🧪 Hypothesis Lab (DeepMind Co-Scientist)", expanded=True):

            st.caption("Generate novel scientific hypotheses by connecting disparate concepts in the retrieved literature.")

            if st.button("✨ Generate Novel Hypotheses", type="primary", use_container_width=True):
                if not clinical_question:
                    st.warning("Please enter a research question first.")
                else:
                    try:
                        from services.hypothesis_service import HypothesisGenerator

                        with st.spinner("Analyzing literature and generating hypotheses..."):
                            # Get Retrieval Context
                            context_chunks = []
                            if 'rag_context' in st.session_state:
                                context_chunks = [_get_text_from_context(c) for c in st.session_state.rag_context]
                            else:
                                from core.retrieval import LocalRetriever
                                retriever = LocalRetriever()
                                docs = retriever.retrieve(clinical_question, top_k=10)
                                context_chunks = [_get_text_from_context(d) for d in docs]

                            gen_service = HypothesisGenerator(api_key=settings.OPENAI_API_KEY)
                            hypotheses = gen_service.generate_hypotheses(clinical_question, context_chunks)
                            st.session_state.generated_hypotheses = hypotheses
                            st.session_state.last_mode = 'discovery'

                    except Exception as e:
                        st.error(f"Hypothesis generation failed: {e}")

            if st.session_state.get('generated_hypotheses'):
                st.markdown("### Generated Hypotheses")
                for i, h in enumerate(st.session_state.generated_hypotheses):
                    novelty_color = "green" if h.novelty_score > 0.7 else "orange"
                    with st.expander(f"**#{i+1}: {h.title}** (Novelty: :{novelty_color}[{h.novelty_score:.2f}])"):
                        st.markdown(f"**Description:** {h.description}")
                        st.markdown(f"**Reasoning:** _{h.reasoning_chain}_")
                        st.markdown(f"**Proposed Experiment:** {h.experimental_validation}")

                        if st.button("Use as Discussion Topic", key=f"use_hyp_{i}"):
                            st.session_state.expert_clinical_question = f"Evaluate this hypothesis: {h.title}. {h.description}"
                            st.rerun()

    elif settings.ENABLE_ADVANCED_TOOLS and mode_key == "strategy":  # STRATEGY WAR ROOM
        # MODE 2: STRATEGY WAR ROOM (Clinical Planning)
        st.subheader("♟️ Strategy War Room")
        
        tab_strat, tab_plan = st.tabs(["Clinical Strategy", "Regulatory Planner"])
        
        with tab_strat:
            with st.expander("🧠 Strategy Generator", expanded=True):
                st.caption("Synthesize public literature with internal constraints to propose development paths.")
                
                proprietary_context = st.text_area("Internal Data / Constraints (Optional)", placeholder="e.g. 'We have a novel biomarker for Patient Population X' or 'Budget limited to Phase 1b'")
                
                if st.button("Generate Strategic Options", type="primary", use_container_width=True):
                    if not clinical_question:
                        st.warning("Please define the Research Objective (Question) above.")
                    else:
                        try:
                            from services.strategy_service import StrategyGenerator
                            
                            with st.spinner("Synthesizing Strategy..."):
                                # Get Retrieval Context
                                context_chunks = []
                                if 'rag_context' in st.session_state:
                                    context_chunks = [_get_text_from_context(c) for c in st.session_state.rag_context]
                                else:
                                    from core.retrieval import LocalRetriever
                                    retriever = LocalRetriever()
                                    docs = retriever.retrieve(clinical_question, top_k=10)
                                    context_chunks = [_get_text_from_context(d) for d in docs]

                                strat_service = StrategyGenerator()
                                strategies = strat_service.generate_strategies(clinical_question, context_chunks, proprietary_context)
                                st.session_state.generated_strategies = strategies
                                st.session_state.last_mode = 'strategy'
                                
                        except Exception as e:
                            st.error(f"Strategy generation failed: {e}")

                if st.session_state.get('generated_strategies'):
                    st.markdown("### 🏆 Recommended Strategies")
                    for i, s in enumerate(st.session_state.generated_strategies):
                        with st.expander(f"**Option {i+1}: {s.name}**"):
                            st.progress(s.benefit_potential, text=f"Benefit Potential: {int(s.benefit_potential*100)}%")
                            st.progress(s.feasibility, text=f"Feasibility: {int(s.feasibility*100)}%")
                            st.markdown(f"**Rationale:** {s.rationale}")
                            st.markdown(f"**⚠️ Risks:** {s.risk_assessment}")
                            st.markdown("**Next Steps:**")
                            for step in s.next_steps: st.markdown(f"- {step}")

                            # Decision Tree Visualization
                            benefit_status = "Pass" if s.benefit_potential >= 0.7 else "Fail"
                            feasibility_status = "Pass" if s.feasibility >= 0.6 else "Fail"
                            
                            # Interactive Decision Logic
                            tree_md = f"""
                            graph LR
                              Start[Start] --> Efficacy{{Benefit > 0.7?}}
                              Efficacy -- Yes --> Feasibility{{Feasible > 0.6?}}
                              Efficacy -- No --> Kill1[STOP: Low Benefit]
                              Feasibility -- Yes --> Go[GO: {s.name}]
                              Feasibility -- No --> Kill2[STOP: Low Feasibility]
                              
                              style Efficacy fill:{'#90EE90' if benefit_status=='Pass' else '#FFCCCB'}
                              style Feasibility fill:{'#90EE90' if feasibility_status=='Pass' else '#FFCCCB'}
                              style Go fill:#90EE90,stroke:#006400,stroke-width:2px
                            """
                            st.markdown("##### 🌳 Decision Path")
                            st.markdown(f"```mermaid\n{tree_md}\n```")

                            
                            if st.button("Analyze this Path", key=f"strat_btn_{i}"):
                                st.session_state.expert_clinical_question = f"Analyze: {s.name}. {s.description}"
                                st.rerun()

        with tab_plan:
            st.caption("Draft formatted regulatory documents based on the current context.")
            
            from services.planning_service import PlanType
            plan_type_sel = st.selectbox("Document Type", [pt.value for pt in PlanType])
            
            if st.button("📝 Draft Document", type="secondary", use_container_width=True):
                 if not clinical_question:
                    st.warning("Please define the Research Objective.")
                 else:
                    try:
                        from services.planning_service import ClinicalPlanner, PlanType as PT
                        # Map selection back to enum
                        selected_enum = next(pt for pt in PT if pt.value == plan_type_sel)
                        
                        with st.spinner(f"Drafting {plan_type_sel}..."):
                             # Get Retrieval Context
                            context_chunks = []
                            if 'rag_context' in st.session_state:
                                context_chunks = [_get_text_from_context(c) for c in st.session_state.rag_context]
                            else:
                                from core.retrieval import LocalRetriever
                                retriever = LocalRetriever()
                                docs = retriever.retrieve(clinical_question, top_k=15)
                                context_chunks = [_get_text_from_context(d) for d in docs]
                            
                            # Add strategy context if available
                            strat_ctx = ""
                            if st.session_state.get('generated_strategies'):
                                strat_ctx = "\nConsider these strategies:\n" + "\n".join([s.description for s in st.session_state.generated_strategies])
                                context_chunks.append(strat_ctx)

                            planner = ClinicalPlanner(api_key=settings.OPENAI_API_KEY)
                            doc = planner.generate_plan(clinical_question, "\n".join(context_chunks), selected_enum)
                            st.session_state.generated_plan = doc
                            
                    except Exception as e:
                        st.error(f"Planning failed: {e}")

            if st.session_state.get('generated_plan'):
                doc = st.session_state.generated_plan
                st.markdown(f"### 📄 {doc.title}")
                
                # Render Sections
                for title, content in doc.sections.items():
                    st.markdown(f"#### {title}")
                    st.markdown(content)
                
                st.divider()
                st.markdown("**⚠️ Identified Critical Risks:**")
                for r in doc.key_risks: st.markdown(f"- {r}")
                
                st.markdown("**🛑 Missing Data (Gap Analysis):**")
                for m in doc.missing_data: st.markdown(f"- {m}")


    # Note: Session state is initialized centrally in core/state_manager.py

    # Run Expert Discussion
    st.subheader(f"🎯 Round {st.session_state.discussion_round} Discussion")

    # AUTO-SEARCH: If no papers loaded, fetch them automatically
    def ensure_evidence_context(clinical_question: str) -> list:
        """Auto-fetch papers if none are loaded."""
        papers = st.session_state.get('search_results', {}).get('citations', []) if st.session_state.get('search_results') else []
        documents = st.session_state.get('indexed_documents', [])
        uploaded = st.session_state.get('uploaded_documents', [])
        
        if papers or documents or uploaded:
            return papers  # Already have context
        
        # No context - auto-search
        with st.status("🔍 Gathering evidence...", expanded=True) as status:
            st.write("Parsing your question...")
            
            try:
                expert_service = st.session_state.get('expert_service')
                if not expert_service:
                    expert_service = ExpertDiscussionService(api_key=settings.OPENAI_API_KEY)
                
                st.write("Searching PubMed...")
                results = expert_service.auto_search_for_discussion(
                    clinical_question=clinical_question,
                    project_id=st.session_state.get('current_project_id', 'default'),
                    citation_dao=st.session_state.citation_dao,
                    search_dao=st.session_state.search_dao,
                    query_cache_dao=st.session_state.query_cache_dao,
                    max_results=st.session_state.get('auto_search_max', 20)
                )
                
                st.session_state.search_results = results
                papers = results.get('citations', [])
                
                status.update(
                    label=f"✅ Found {len(papers)} papers (ranked by clinical relevance)", 
                    state="complete"
                )
                return papers
                
            except Exception as e:
                status.update(label=f"⚠️ Search failed: {str(e)}", state="error")
                return []

    if st.button(f"🔬 Analyze (Round {st.session_state.discussion_round})", type="primary", use_container_width=True, disabled=not clinical_question or not selected_experts):
        if not clinical_question:
            st.warning("Please enter a research question.")
            st.stop()
        if not selected_experts:
            st.warning("Please select at least one expert.")
            st.stop()

        # Ensure we have evidence (auto-search if needed)
        # Only needed if NOT using two-pass mode (which handles its own search)
        use_two_pass = st.session_state.get('two_pass_mode', True) and (not citations or len(citations) == 0)
        
        if not use_two_pass:
             citations_list = ensure_evidence_context(clinical_question)
             # Refresh papers var
             search_results = st.session_state.get('search_results')
             citations = search_results.get('citations') if search_results else None

        # =====================================================================
        # TWO-PASS MODE: Immediate answers + Background literature validation
        # =====================================================================
        if use_two_pass:
            # Initialize expert service early
            expert_service = ExpertDiscussionService(
                api_key=settings.OPENAI_API_KEY,
                model=settings.EXPERT_MODEL,
                max_tokens=settings.EXPERT_MAX_TOKENS
            )

            # 1. Start background literature search IMMEDIATELY
            st.info("🔍 Literature search running in background...")
            search_future = expert_service.start_background_literature_search(
                clinical_question=clinical_question,
                max_results=st.session_state.get('auto_search_max', 20)
            )

            # 2. Get web context for Pass 1 (if enabled)
            web_context = []
            if st.session_state.get('enable_web_search', False) and GOOGLE_SEARCH_AVAILABLE:
                try:
                    from integrations.google_search import search_with_grounding
                    _, web_context = search_with_grounding(clinical_question, max_sources=5)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Google Search for Pass 1 failed: {e}")

            # 3. Run Pass 1 (Immediate LLM + Web responses)
            with st.status("⚡ Pass 1: Getting immediate expert responses...", expanded=True) as pass1_status:
                def pass1_progress(expert_name, current, total):
                    st.write(f"✓ {expert_name} ({current}/{total})")

                pass1_responses = expert_service.run_pass1_immediate(
                    clinical_question=clinical_question,
                    selected_experts=selected_experts,
                    scenario=scenario,
                    web_context=web_context,
                    temperatures=st.session_state.get('expert_temperatures', {}),
                    progress_callback=pass1_progress
                )
                pass1_status.update(label=f"⚡ Pass 1 Complete - {len(pass1_responses)} experts responded", state="complete")

            # Store Pass 1 responses
            st.session_state.pass1_responses = pass1_responses
            current_round = st.session_state.discussion_round

            # 4. Display Pass 1 results IMMEDIATELY (yellow banner)
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); padding: 12px 16px; border-radius: 8px; margin: 16px 0; border-left: 4px solid #f39c12;">
                <strong>⚡ Initial Response</strong> — Based on expert knowledge + web search. Literature validation pending...
            </div>
            """, unsafe_allow_html=True)

            for expert_name, response in pass1_responses.items():
                content = response.get('content', 'No response')
                finish_reason = response.get('finish_reason', 'unknown')
                with st.expander(f"🧑‍🔬 {expert_name}" + (" ⚠️" if finish_reason == 'error' else ""), expanded=True):
                    st.markdown(content)

            # 5. Wait for literature search and run Pass 2
            with st.status("📚 Pass 2: Validating against literature...", expanded=True) as pass2_status:
                try:
                    st.write("Waiting for PubMed search to complete...")
                    search_results = search_future.result(timeout=45)
                    literature_citations = search_results.get('citations', [])
                    optimized_query = search_results.get('optimized_query', '')

                    st.write(f"Found {len(literature_citations)} papers")
                    if optimized_query:
                        st.caption(f"Query: {optimized_query[:100]}...")

                    # Store search results
                    if literature_citations:
                        st.session_state.search_results = search_results
                        st.session_state.literature_search_results = search_results

                    if literature_citations:
                        st.write("Running literature validation...")

                        def pass2_progress(expert_name, current, total):
                            st.write(f"✓ Validating {expert_name} ({current}/{total})")

                        pass2_validations = expert_service.run_pass2_validation(
                            clinical_question=clinical_question,
                            pass1_responses=pass1_responses,
                            literature_citations=literature_citations,
                            selected_experts=selected_experts,
                            progress_callback=pass2_progress
                        )
                        st.session_state.pass2_validations = pass2_validations
                        pass2_status.update(label=f"📚 Pass 2 Complete - {len(literature_citations)} papers reviewed", state="complete")
                    else:
                        pass2_validations = None
                        pass2_status.update(label="⚠️ No papers found - Pass 1 only", state="complete")

                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Pass 2 failed: {e}")
                    pass2_status.update(label=f"❌ Pass 2 failed: {str(e)[:50]}", state="error")
                    pass2_validations = None
                    literature_citations = []

            # 6. Display Pass 2 validation results (green banner or contradiction alert)
            if pass2_validations:
                # Check for contradictions
                total_contradictions = sum(v.claims_contradicted for v in pass2_validations.values())
                total_supported = sum(v.claims_supported for v in pass2_validations.values())

                if total_contradictions > 0:
                    # CONTRADICTION ALERT - "Great Catch" moment
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 12px 16px; border-radius: 8px; margin: 16px 0; border-left: 4px solid #dc3545;">
                        <strong>⚠️ Evidence Update</strong> — New literature contradicts {total_contradictions} initial claim(s). Review validation below.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); padding: 12px 16px; border-radius: 8px; margin: 16px 0; border-left: 4px solid #28a745;">
                        <strong>📚 Literature Validated</strong> — {len(literature_citations)} papers reviewed. {total_supported} claim(s) supported.
                    </div>
                    """, unsafe_allow_html=True)

                st.subheader("📚 Literature Validation")
                for expert_name, validation in pass2_validations.items():
                    # Color-code based on contradictions
                    has_contradictions = validation.claims_contradicted > 0
                    icon = "🔴" if has_contradictions else "🟢"
                    with st.expander(f"{icon} {expert_name} — Evidence Check", expanded=has_contradictions):
                        # Show stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("✅ Supported", validation.claims_supported)
                        with col2:
                            st.metric("❌ Contradicted", validation.claims_contradicted)
                        with col3:
                            st.metric("⚠️ No Evidence", validation.claims_no_evidence)
                        st.markdown("---")
                        st.markdown(validation.validation_text)

            # Store combined responses for session state (use Pass 1 as base)
            combined_responses = {}
            for expert_name, response in pass1_responses.items():
                combined_responses[expert_name] = response.copy()
                combined_responses[expert_name]['pass1'] = True
                if pass2_validations and expert_name in pass2_validations:
                    combined_responses[expert_name]['validation'] = pass2_validations[expert_name].validation_text

            st.session_state.expert_discussion[current_round] = combined_responses

            # Skip the standard flow below
            st.session_state.discussion_round += 1
            st.success("✅ Two-pass discussion complete!")
            st.rerun()

        # =====================================================================
        # ORIGINAL FLOW: Auto-search (blocking) when two-pass mode is disabled
        # =====================================================================
        elif (not citations or len(citations) == 0):
            # Check user preference (default True)
            if st.session_state.get('enable_auto_search', True):
                with st.status("🔍 Auto-searching literature...", expanded=True) as status:
                    st.write("No literature loaded. Searching PubMed for relevant papers...")

                    try:
                        # Get DAOs from session state or assume initialized
                        query_cache_dao = st.session_state.get('query_cache_dao')
                        search_dao = st.session_state.get('search_dao')

                        auto_results = expert_service.auto_search_for_discussion(
                            clinical_question=clinical_question,
                            project_id=project_id,
                            citation_dao=citation_dao,
                            search_dao=search_dao,
                            query_cache_dao=query_cache_dao,
                            max_results=20
                        )

                        citations = auto_results.get('citations', [])

                        # Update session state so they persist
                        if citations:
                            st.session_state.search_results = auto_results

                        status.update(label=f"✅ Found {len(citations)} papers", state="complete")
                        st.write(f"Query: {auto_results.get('optimized_query', '')[:100]}...")

                    except Exception as e:
                        status.update(label="❌ Auto-search failed", state="error")
                        st.error(f"Auto-search error: {e}")

        # Convert citations to paper dicts (whether from auto-search or manual)
        papers = []
        for c in (citations or []):
            if hasattr(c, 'pmid'):
                papers.append({'pmid': c.pmid, 'title': c.title, 'abstract': c.abstract, 'authors': c.authors, 'journal': c.journal, 'year': c.year})
            else:
                papers.append(c)

        # Get previous responses for multi-round discussions
        previous_responses = None
        if st.session_state.discussion_round > 1:
            prev_round = st.session_state.discussion_round - 1
            if prev_round in st.session_state.expert_discussion:
                previous_responses = {}
                for exp, resp in st.session_state.expert_discussion[prev_round].items():
                    content = resp.get('content', '')
                    if resp.get('human_edited'):
                        content = f"[HUMAN VERIFIED - Treat as ground truth]\n\n{content}"
                    elif resp.get('regenerated'):
                        content = f"[REGENERATED with feedback]\n\n{content}"
                    previous_responses[exp] = content

        current_round = st.session_state.discussion_round

        # UI progress elements
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Retrieve RAG context from local documents (if available)
        rag_context = []
        if settings.ENABLE_LOCAL_RAG:
            try:
                from core.retrieval import LocalRetriever
                status_text.text("Retrieving context from uploaded documents...")
                retriever = LocalRetriever()
                rag_context = retriever.retrieve(
                    query=clinical_question,
                    top_k=5,
                    project_filter=st.session_state.get('current_project_name', 'default'),
                    use_hyde=st.session_state.get('enable_hyde', True),
                    use_query_expansion=st.session_state.get('enable_query_expansion', True)
                )
                if rag_context:
                    st.session_state.rag_context = rag_context
            except (ImportError, Exception) as e:
                import logging
                logging.getLogger(__name__).warning(f"Local RAG retrieval failed: {e}")

        # Web search fallback if enabled and local results insufficient
        if st.session_state.get('enable_web_search', False) and TAVILY_AVAILABLE:
            if len(rag_context) < 2:  # Fallback threshold
                try:
                    from integrations.tavily import search_web_for_rag
                    status_text.text("Searching web for additional context...")
                    web_results = search_web_for_rag(clinical_question, max_results=3)
                    if web_results:
                        rag_context.extend(web_results)
                        st.session_state.rag_context = rag_context
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Web search fallback failed: {e}")

        # Progress callback for UI updates
        def update_progress(expert_name, current, total):
            status_text.text(f"Consulting {expert_name}...")
            progress_bar.progress(current / total)

        # Use ExpertDiscussionService for the main business logic
        expert_service = ExpertDiscussionService(
            api_key=settings.OPENAI_API_KEY,
            model=settings.EXPERT_MODEL,
            max_tokens=settings.EXPERT_MAX_TOKENS
        )

        # Prepare injected evidence (Manual + Direct Context)
        final_injected_evidence = st.session_state.get('injected_evidence', [])[:]
        
        # Add direct context if available
        if direct_context.strip():
            final_injected_evidence.append({
                'title': 'User Provided Context',
                'abstract': direct_context,
                'source': 'Direct Input',
                'pmid': 'MANUAL-CTX'
            })

        if st.session_state.deep_research_mode:
            # STORM Workflow (Sequential per expert)
            status_text.text("🌪️ Running Deep Research Mode (STORM)...")
            
            for i, expert_name in enumerate(selected_experts):
                update_progress(expert_name, i, len(selected_experts))
                try:
                    result = expert_service.run_storm_workflow(
                        clinical_question, expert_name, papers, final_injected_evidence
                    )
                    # Add to session state
                    st.session_state.expert_discussion.setdefault(current_round, {})
                    st.session_state.expert_discussion[current_round][expert_name] = {
                        'content': result['final_response'],
                        'finish_reason': 'stop',
                        'storm_data': result # Save the outline/questions for provenance
                    }
                except Exception as e:
                    st.session_state.expert_discussion.setdefault(current_round, {})
                    st.session_state.expert_discussion[current_round][expert_name] = {
                        'content': f"Error in Deep Research: {str(e)}",
                        'finish_reason': 'error'
                    }
            
            # Create a mock result object for downstream consistency
            class MockResult:
                failures = []
                responses = st.session_state.expert_discussion[current_round]
            result = MockResult()

        else:
            # Standard Parallel Execution
            result = expert_service.run_discussion_round(
                round_num=current_round,
                clinical_question=clinical_question,
                selected_experts=selected_experts,
                citations=papers,
                scenario=scenario,
                previous_responses=previous_responses,
                injected_evidence=final_injected_evidence,
                temperatures=st.session_state.get('expert_temperatures', {}),
                working_memory=working_memory,
                rag_context=rag_context,
                progress_callback=update_progress
            )

            # Update session state with results
            st.session_state.expert_discussion[current_round] = result.responses

        progress_bar.progress(1.0)


        if result.failures:
            failed_names = [f[0] for f in result.failures]
            status_text.text(f"Discussion complete with {len(result.failures)} error(s)")
            st.warning(f"Some experts encountered errors: {', '.join(failed_names)}")
        else:
            status_text.text("Discussion complete!")

        try:
            discussion_id = expert_discussion_dao.create_discussion(project_id=st.session_state.current_project_id, clinical_question=clinical_question, scenario=scenario, experts=selected_experts)
            for expert_name, response in st.session_state.expert_discussion[current_round].items():
                expert_discussion_dao.add_entry(
                    discussion_id=discussion_id, round_num=current_round, expert_name=expert_name,
                    content=response.get('content', ''), raw_response=json.dumps({'finish_reason': response.get('finish_reason'), 'tokens': response.get('tokens', {})})
                )
        except Exception as e:
            st.warning(f"Failed to save discussion to database: {e}")

        st.success("✅ Round complete!")

        # Generate follow-up questions using service
        try:
            questions = expert_service.generate_follow_up_questions(
                clinical_question=clinical_question,
                responses=st.session_state.expert_discussion[current_round]
            )
            if questions:
                st.session_state.suggested_questions = questions
        except Exception:
            pass

        # Extract knowledge using service
        try:
            knowledge_result = expert_service.extract_knowledge(
                responses=st.session_state.expert_discussion[current_round],
                clinical_question=clinical_question,
                source_name=f"Expert Panel: {clinical_question[:50]}..."
            )
            if knowledge_result.get('facts_count', 0) > 0 or knowledge_result.get('triples_count', 0) > 0:
                st.info(f"🧠 Learned {knowledge_result['facts_count']} facts and {knowledge_result['triples_count']} relationships from this discussion")
        except Exception:
            pass

    # =========================================================================
    # COLLABORATIVE DEBATE MODE (Co-Scientist Level)
    # =========================================================================
    if st.session_state.expert_discussion:
        st.markdown("---")
        st.subheader("⚔️ Collaborative Debate (DeepMind Co-Scientist)")
        st.caption("Resolves trade-offs by forcing a Proposal -> Challenge -> Mitigation loop between two experts.")
        
        col_deb1, col_deb2, col_deb3 = st.columns(3)
        with col_deb1:
            debate_pro = st.selectbox("Proponent", selected_experts, key="debate_pro")
        with col_deb2:
            debate_con = st.selectbox("Challenger", [e for e in selected_experts if e != debate_pro], key="debate_con")
        with col_deb3:
            debate_topic = st.text_input("Debate Topic", placeholder="e.g. Efficacy vs Toxicity")
            
        if st.button("🔥 Start Debate", key="start_debate", type="secondary"):
            if not debate_topic:
                st.warning("Enter a topic")
            elif not debate_con:
                st.warning("Select two different experts")
            else:
                with st.spinner(f"Running debate: {debate_pro} vs {debate_con}..."):
                    debate_res = expert_service.run_debate_round(
                        clinical_question, debate_pro, debate_con, debate_topic, citations, scenario
                    )
                    st.session_state.debate_result = debate_res
        
        if st.session_state.get('debate_result'):
            res = st.session_state.debate_result
            st.markdown(f"### ⚔️ Debate Resolution: {res['topic']}")
            
            # 1. Proposal Node
            st.markdown(f"""
            <div class="debate-node-pro">
                <div style="font-size: 0.8rem; font-weight: 700; color: var(--primary); margin-bottom: 4px;">PROPOSAL ({res['pro_expert']})</div>
                {res['proposal'].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            # 2. Challenge Node
            st.markdown(f"""
            <div class="debate-node-con">
                <div style="font-size: 0.8rem; font-weight: 700; color: var(--warning); margin-bottom: 4px;">CHALLENGE ({res['con_expert']})</div>
                {res['challenge'].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            # 3. Mitigation Node
            st.markdown(f"""
            <div class="debate-node-mitigation">
                <div style="font-size: 0.8rem; font-weight: 700; color: var(--success); margin-bottom: 4px;">MITIGATION ({res['pro_expert']})</div>
                {res['mitigation'].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
            
            # 4. Synthesis
            st.markdown(f"""
            <div class="debate-synthesis">
                <h3 style="color: white !important; margin-top: 0;">🏛️ Chairperson Synthesis</h3>
                <div style="color: #E2E8F0;">
                    {res['synthesis'].replace(chr(10), '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)


    # Display Suggested Questions
    if st.session_state.suggested_questions:
        st.markdown("---")
        st.markdown("**💡 Suggested Follow-up Questions:**")
        cols = st.columns(2)
        for i, question in enumerate(st.session_state.suggested_questions[:4]):
            with cols[i % 2]:
                display_q = question[:60] + "..." if len(question) > 60 else question
                if st.button(display_q, key=f"suggested_q_{i}", use_container_width=True):
                    st.session_state.expert_clinical_question = question
                    st.session_state.suggested_questions = []
                    st.rerun()

    # Display Discussion Results
    if st.session_state.expert_discussion:
        st.markdown("---")
        st.subheader("💬 Discussion Results")

        # HITL: Regenerate Modal
        if st.session_state.get('show_regenerate_modal') and st.session_state.get('regenerate_target'):
            regen_round, regen_expert = st.session_state.regenerate_target
            with st.container():
                st.warning(f"🔄 **Regenerate Response for {regen_expert} (Round {regen_round})**")
                critique = st.text_area(
                    "Why was this response rejected? (This feedback will guide the new response)",
                    placeholder="e.g., The response didn't address the PK data, was too vague, or made incorrect assumptions...",
                    key="regenerate_critique"
                )
                col_regen1, col_regen2 = st.columns(2)
                with col_regen1:
                    if st.button("🚀 Regenerate", type="primary", use_container_width=True):
                        if critique.strip():
                            # Get context for regeneration
                            papers = []
                            for c in (citations or []):
                                if hasattr(c, 'pmid'):
                                    papers.append({'pmid': c.pmid, 'title': c.title, 'abstract': c.abstract, 'authors': c.authors, 'journal': c.journal, 'year': c.year})
                                else:
                                    papers.append(c)
                            search_results_dict = {'citations': papers}

                            previous_responses = None
                            if regen_round > 1 and (regen_round - 1) in st.session_state.expert_discussion:
                                previous_responses = {exp: resp.get('content', '') for exp, resp in st.session_state.expert_discussion[regen_round - 1].items()}

                            # Get old response for history
                            old_response_obj = st.session_state.expert_discussion[regen_round].get(regen_expert)

                            with st.spinner(f"Regenerating {regen_expert}'s response..."):
                                try:
                                    # Use Service
                                    new_response = st.session_state.expert_service.regenerate_response(
                                        expert_name=regen_expert,
                                        round_num=regen_round,
                                        clinical_question=clinical_question,
                                        citations=citations,
                                        scenario=scenario,
                                        rejection_critique=critique.strip(),
                                        previous_responses=previous_responses,
                                        injected_evidence=st.session_state.get('injected_evidence', []),
                                        working_memory=working_mem,
                                        old_response=old_response_obj
                                    )
                                    st.session_state.expert_discussion[regen_round][regen_expert] = new_response
                                    st.success(f"✅ {regen_expert}'s response regenerated!")

                                    # Record correction for learning (SQLite)
                                    try:
                                        db = st.session_state.get('db')
                                        if db:
                                            correction_dao = ExpertCorrectionDAO(db)
                                            correction_dao.add_correction(
                                                expert_name=regen_expert,
                                                critique=critique.strip(),
                                                project_id=st.session_state.get('current_project_id'),
                                                question_snippet=clinical_question[:200] if clinical_question else None
                                            )
                                            logger.info(f"Recorded correction for {regen_expert} in SQLite")
                                    except Exception as learn_err:
                                        logger.warning(f"Failed to record correction for learning: {learn_err}")
                                except Exception as e:
                                    st.error(f"Regeneration failed: {e}")

                            st.session_state.show_regenerate_modal = False
                            st.session_state.regenerate_target = None
                            st.rerun()
                        else:
                            st.warning("Please provide feedback on why the response was rejected.")
                with col_regen2:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state.show_regenerate_modal = False
                        st.session_state.regenerate_target = None
                        st.rerun()
                st.markdown("---")

        round_tabs = st.tabs([f"Round {r}" for r in sorted(st.session_state.expert_discussion.keys())])

        for tab, round_num in zip(round_tabs, sorted(st.session_state.expert_discussion.keys())):
            with tab:
                round_data = st.session_state.expert_discussion[round_num]
                for expert_name, response in round_data.items():
                    edit_key = f"editing_{round_num}_{expert_name}"

                    # Check if we're in edit mode for this response
                    if st.session_state.get(edit_key, False):
                        # EDIT MODE
                        st.markdown(f"### ✏️ Editing: {expert_name}")
                        edited_content = st.text_area(
                            f"Edit {expert_name}'s response",
                            value=response.get('content', ''),
                            height=400,
                            key=f"edit_area_{round_num}_{expert_name}"
                        )
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.button("💾 Save Edit", key=f"save_{round_num}_{expert_name}", type="primary", use_container_width=True):
                                st.session_state.expert_discussion[round_num][expert_name]['content'] = edited_content
                                st.session_state.expert_discussion[round_num][expert_name]['human_edited'] = True
                                st.session_state[edit_key] = False
                                st.rerun()
                        with col_cancel:
                            if st.button("Cancel", key=f"cancel_{round_num}_{expert_name}", use_container_width=True):
                                st.session_state[edit_key] = False
                                st.rerun()
                    else:
                        # VIEW MODE with HITL controls
                        with st.expander(f"**{expert_name}**", expanded=True):
                            # Status badges
                            status_parts = []
                            if response.get('human_edited'):
                                status_parts.append("✏️ Human Edited")
                            if response.get('regenerated'):
                                status_parts.append("🔄 Regenerated")

                            if status_parts:
                                st.caption(" | ".join(status_parts))

                            # Hallucination warnings (GRADE v2.0)
                            if EVIDENCE_CORPUS_AVAILABLE:
                                _render_hallucination_warnings(
                                    response_text=response.get('content', ''),
                                    expert_name=expert_name
                                )

                            # Response content
                            with st.chat_message(expert_name):
                                st.markdown(response.get('content', 'No response'))

                            # Footer with stats and HITL buttons
                            col_status, col_edit, col_regen, col_verify = st.columns([3, 1, 1, 1])
                            with col_status:
                                st.caption(f"Status: {response.get('finish_reason', 'unknown')} | Tokens: {response.get('tokens', {}).get('total_tokens', 'N/A')}")
                            with col_edit:
                                if st.button("✏️", key=f"edit_btn_{round_num}_{expert_name}", help="Edit response"):
                                    st.session_state[edit_key] = True
                                    st.rerun()
                            with col_verify:
                                if st.button("🛡️", key=f"verify_{round_num}_{expert_name}", help="Verify Claims"):
                                    # Trigger Verification
                                    try:
                                        from services.verification_service import VerificationService
                                        verifier = VerificationService()
                                        # Use RAG context as evidence
                                        evidence = [_get_text_from_context(c) for c in st.session_state.get('rag_context', [])]
                                        checks = verifier.verify_response(response.get('content', ''), evidence)
                                        st.session_state.expert_discussion[round_num][expert_name]['verification'] = checks
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Verification failed: {e}")

                            with col_regen:
                                if st.button("🔄", key=f"regen_btn_{round_num}_{expert_name}", help="Regenerate response"):
                                    st.session_state.regenerate_target = (round_num, expert_name)
                                    st.session_state.show_regenerate_modal = True
                                    st.rerun()

                            # Display Verification Results
                            if response.get('verification'):
                                with st.expander("🛡️ Verification Report (Epistemic Check)", expanded=True):
                                    for check in response['verification']:
                                        color = "green" if check.status == "SUPPORTED" else "red" if check.status == "CONTRADICTED" else "gray"
                                        icon = "✅" if check.status == "SUPPORTED" else "❌" if check.status == "CONTRADICTED" else "❓"
                                        st.markdown(f"{icon} **{check.status}**: {check.claim_text}")
                                        if check.status != "NEUTRAL":
                                            st.caption(f"Evidence: \"{check.evidence_snippet}\"")
                                        st.divider()

                            # History / Diff View
                            if response.get('history'):
                                with st.expander("📜 Version History & Diffs"):
                                    for idx, entry in enumerate(response['history']):
                                        st.caption(f"Version {entry.get('version', idx+1)} - {entry.get('timestamp', 'Unknown time')}")
                                        if entry.get('critique'):
                                            st.markdown(f"**Critique:** _{entry['critique']}_")
                                        
                                        # Show Diff
                                        if st.checkbox(f"Show Diff vs Current", key=f"diff_{round_num}_{expert_name}_{idx}"):
                                            diff = difflib.unified_diff(
                                                entry['content'].splitlines(),
                                                response['content'].splitlines(),
                                                fromfile=f"Version {entry.get('version')}",
                                                tofile="Current Version",
                                                lineterm=''
                                            )
                                            diff_text = "\n".join(diff)
                                            if diff_text:
                                                st.code(diff_text, language="diff")
                                            else:
                                                st.info("No text changes detected.")
                                        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.session_state.discussion_round > 1:
                if st.button("⬅️ Previous Round"):
                    st.session_state.discussion_round -= 1
                    st.rerun()
        with col2:
            if st.session_state.discussion_round < 4:
                if st.button("➡️ Next Round"):
                    st.session_state.discussion_round += 1
                    st.rerun()
        with col3:
            if st.button("Reset Discussion"):
                st.session_state.expert_discussion = {}
                st.session_state.discussion_round = 1
                st.session_state.meta_synthesis = None
                st.session_state.gap_analysis = None
                st.session_state.conflict_analysis = None
                st.session_state.working_memory = None
                st.session_state.human_feedback = []
                st.rerun()

        # Meta-Review Synthesis
        st.markdown("---")
        st.subheader("🔬 Meta-Review Synthesis")
        st.caption("Consolidates expert insights by domain (not debate - respects expertise boundaries)")

        latest_round = max(st.session_state.expert_discussion.keys())
        latest_responses = {exp: resp.get('content', '') for exp, resp in st.session_state.expert_discussion[latest_round].items()}

        if st.button("🧠 Generate Synthesis", type="primary"):
            with st.spinner("Synthesizing expert insights by domain..."):
                try:
                    analysis_service = AnalysisService()
                    synthesis_result = analysis_service.synthesize_responses(
                        responses=latest_responses,
                        clinical_question=clinical_question
                    )
                    # Convert SynthesisResult to dict for session state
                    st.session_state.meta_synthesis = {
                        'synthesis': synthesis_result.synthesis,
                        'consensus_points': synthesis_result.consensus_points,
                        'open_questions': synthesis_result.open_questions,
                        'recommended_actions': synthesis_result.recommended_actions
                    }
                except Exception as e:
                    st.error(f"Synthesis failed: {e}")

        if st.session_state.get('meta_synthesis'):
            synthesis = st.session_state.meta_synthesis
            st.markdown(synthesis.get('synthesis', ''))
            col1, col2 = st.columns(2)
            with col1:
                if synthesis.get('consensus_points'):
                    st.markdown("**Points of Consensus:**")
                    for point in synthesis['consensus_points'][:5]: st.markdown(f"- {point}")
            with col2:
                if synthesis.get('open_questions'):
                    st.markdown("**Open Questions:**")
                    for q in synthesis['open_questions'][:5]: st.markdown(f"- {q}")
            if synthesis.get('recommended_actions'):
                st.markdown("**Recommended Actions:**")
                for i, action in enumerate(synthesis['recommended_actions'][:5], 1): st.markdown(f"{i}. {action}")

            # Generate GRADE-style recommendation
            st.markdown("---")
            st.markdown("### 📋 Generate GRADE Recommendation")
            if st.button("Generate Structured Recommendation", type="secondary"):
                with st.spinner("Generating GRADE-style recommendation..."):
                    try:
                        rec_service = get_recommendation_service()
                        # Get citations if available
                        included_citations = None
                        if st.session_state.get('search_results'):
                            included_citations = st.session_state.search_results.get('citations', [])

                        recommendation = rec_service.generate_recommendation(
                            question=clinical_question,
                            expert_responses=latest_responses,
                            chair_synthesis=synthesis.get('synthesis', ''),
                            included_citations=included_citations,
                            question_type=scenario
                        )
                        st.session_state.generated_recommendation = recommendation
                    except Exception as e:
                        st.error(f"Recommendation generation failed: {e}")

            if st.session_state.get('generated_recommendation'):
                recommendation = st.session_state.generated_recommendation
                # Display styled recommendation card
                st.markdown(get_recommendation_service().format_recommendation_card(recommendation), unsafe_allow_html=True)

                # Expandable details
                with st.expander("📊 Full Recommendation Details", expanded=False):
                    st.markdown(recommendation.format_full_recommendation())

                # Export recommendation
                col_rec1, col_rec2 = st.columns(2)
                with col_rec1:
                    st.download_button(
                        "📥 Download Recommendation (Markdown)",
                        recommendation.format_full_recommendation(),
                        file_name=f"recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                with col_rec2:
                    import json
                    st.download_button(
                        "📥 Download Recommendation (JSON)",
                        json.dumps(recommendation.to_dict(), indent=2),
                        file_name=f"recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

        # =========================================================================
        # GAP ANALYSIS
        # =========================================================================
        if ANALYSIS_AVAILABLE:
            st.markdown("---")
            st.subheader("Gap Analysis")
            st.caption("Analyzes discussion completeness, evidence quality, and quantification")

            # Map GDG question types to analysis topics
            scenario_mapping = {
                "surgical_candidate": "Surgical Candidacy Assessment",
                "palliative_pathway": "Palliative Care Planning",
                "intervention_choice": "Intervention Comparison",
                "symptom_management": "Symptom Control",
                "prognosis_assessment": "Prognosis & Outcomes",
                "ethics_review": "Ethics & Appropriateness",
                "resource_allocation": "Resource & Implementation",
                "general": "General Discussion"
            }
            mapped_scenario = scenario_mapping.get(scenario, "General Discussion")

            if st.button("Run Gap Analysis", type="secondary"):
                with st.spinner("Analyzing discussion gaps..."):
                    try:
                        # Use AnalysisService for gap analysis
                        analysis_service = AnalysisService()
                        # Convert responses to expected format
                        responses_dict = {exp: {'content': content} for exp, content in latest_responses.items()}
                        gap_result = analysis_service.analyze_gaps(
                            responses=responses_dict,
                            scenario=scenario
                        )
                        st.session_state.gap_analysis = gap_result
                    except Exception as e:
                        st.error(f"Gap analysis failed: {e}")

            if st.session_state.get('gap_analysis'):
                gap = st.session_state.gap_analysis

                # Coverage and Quantification Scores
                col1, col2 = st.columns(2)
                with col1:
                    coverage_pct = int(gap.coverage_score * 100)
                    st.metric("Coverage Score", f"{coverage_pct}%")
                    if coverage_pct >= 80:
                        st.success("Strong coverage")
                    elif coverage_pct >= 50:
                        st.warning("Moderate coverage")
                    else:
                        st.error("Significant gaps")

                with col2:
                    quant_pct = int(gap.quantification_score * 100)
                    st.metric("Quantification Score", f"{quant_pct}%")
                    if quant_pct >= 70:
                        st.success("Well-quantified")
                    elif quant_pct >= 40:
                        st.warning("Needs more data")
                    else:
                        st.error("Lacks specifics")

                # Strengths and Gaps
                col1, col2 = st.columns(2)
                with col1:
                    if gap.strengths:
                        st.markdown("**Topics Covered:**")
                        for s in gap.strengths[:5]:
                            st.markdown(f"- {s.replace('_', ' ').title()}")

                with col2:
                    if gap.gaps:
                        st.markdown("**Missing Topics:**")
                        for g in gap.gaps[:5]:
                            st.markdown(f"- {g.replace('_', ' ').title()}")

                # Evidence Issues
                if gap.evidence_issues:
                    with st.expander(f"Evidence Issues ({len(gap.evidence_issues)})"):
                        for issue in gap.evidence_issues[:5]:
                            severity_color = "red" if issue.get('severity') == 'high' else "orange"
                            st.markdown(f"**{issue.get('expert', 'Unknown')}** ({issue.get('issue_type', 'unknown')})")
                            st.markdown(f"_{issue.get('details', issue.get('excerpt', ''))[:150]}..._")

                # Recommendations
                if gap.recommendations:
                    st.markdown("**Recommendations:**")
                    for rec in gap.recommendations[:4]:
                        st.info(rec)

        # =========================================================================
        # CONFLICT DETECTION
        # =========================================================================
        if ANALYSIS_AVAILABLE:
            st.markdown("---")
            st.subheader("Conflict Detection")
            st.caption("Identifies disagreements and divergent views between experts")

            col1, col2 = st.columns([3, 1])
            with col1:
                use_adversarial = st.checkbox("Adversarial Mode", help="Generate tough reviewer-style questions")
            with col2:
                if st.button("Detect Conflicts", type="secondary"):
                    with st.spinner("Detecting conflicts..."):
                        try:
                            # Use AnalysisService for conflict detection
                            analysis_service = AnalysisService()
                            responses_dict = {exp: {'content': content} for exp, content in latest_responses.items()}
                            conflict_result = analysis_service.detect_conflicts(responses=responses_dict)
                            st.session_state.conflict_analysis = conflict_result
                            st.session_state.use_adversarial = use_adversarial
                        except Exception as e:
                            st.error(f"Conflict detection failed: {e}")

            if st.session_state.get('conflict_analysis'):
                conflicts = st.session_state.conflict_analysis

                if conflicts.conflicts:
                    st.markdown(f"**{len(conflicts.conflicts)} conflict(s) detected**")

                    # Group by severity
                    critical = [c for c in conflicts.conflicts if c.severity == 'critical']
                    moderate = [c for c in conflicts.conflicts if c.severity == 'moderate']
                    minor = [c for c in conflicts.conflicts if c.severity == 'minor']

                    if critical:
                        st.error(f"Critical conflicts: {len(critical)}")
                        for c in critical[:3]:
                            with st.expander(f"[CRITICAL] {c.metric}"):
                                for expert, value in c.values.items():
                                    st.markdown(f"**{expert}**: {value}")
                                st.caption(f"Rationale: {c.rationale}")

                    if moderate:
                        st.warning(f"Moderate conflicts: {len(moderate)}")
                        for c in moderate[:3]:
                            with st.expander(f"[MODERATE] {c.metric}"):
                                for expert, value in c.values.items():
                                    st.markdown(f"**{expert}**: {value}")
                                st.caption(f"Rationale: {c.rationale}")

                    if minor:
                        st.info(f"Minor conflicts: {len(minor)}")

                    # Clarification prompts
                    if conflicts.clarification_needed:
                        st.markdown("---")
                        st.markdown("**Clarification Questions:**")
                        for prompt in conflicts.clarification_needed[:3]:
                            st.markdown(prompt)
                            if st.button("Use as next round question", key=f"use_q_{hash(prompt)}"):
                                st.session_state.expert_clinical_question = prompt.split('\n\n')[-1][:200]
                                st.rerun()
                else:
                    st.success("No significant conflicts detected between experts")

                # Decision Memo
                if conflicts.decision_memo:
                    with st.expander("Decision Memo", expanded=False):
                        st.markdown(conflicts.decision_memo)
                        st.download_button(
                            "Download Decision Memo",
                            conflicts.decision_memo,
                            file_name=f"decision_memo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )

        # Hypothesis Tracking
        st.markdown("---")
        st.subheader("📊 Hypothesis Tracker")
        st.caption("Track key claims and their evidence strength across discussion rounds")

        if st.button("Extract Hypotheses from Discussion"):
            with st.spinner("Extracting key hypotheses..."):
                try:
                    # Use AnalysisService for hypothesis extraction
                    analysis_service = AnalysisService()
                    hypotheses = analysis_service.extract_hypotheses(
                        responses=latest_responses,
                        clinical_question=clinical_question,
                        round_num=latest_round
                    )
                    existing_texts = {h.get('hypothesis', '') for h in st.session_state.tracked_hypotheses}
                    new_hypotheses = [h for h in hypotheses if h.get('hypothesis') not in existing_texts]
                    st.session_state.tracked_hypotheses.extend(new_hypotheses)
                    if new_hypotheses:
                        st.success(f"Extracted {len(new_hypotheses)} new hypotheses")
                    else:
                        st.info("No new hypotheses found")
                except Exception as e:
                    st.error(f"Extraction failed: {e}")

        if st.session_state.tracked_hypotheses:
            sorted_hyps = sorted(st.session_state.tracked_hypotheses, key=lambda x: x.get('evidence_strength', 0), reverse=True)
            for i, hyp in enumerate(sorted_hyps):
                strength = hyp.get('evidence_strength', 3)
                strength_bar = "🟢" * strength + "⚪" * (5 - strength)
                with st.expander(f"{strength_bar} {hyp.get('hypothesis', 'Unknown')[:80]}..."):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Evidence Strength", f"{strength}/5")
                    col2.metric("Type", hyp.get('evidence_type', 'Unknown'))
                    col3.metric("Round", hyp.get('round', '?'))
                    st.markdown(f"**Full hypothesis:** {hyp.get('hypothesis', '')}")
                    if hyp.get('supporting_experts'): st.markdown(f"**Supported by:** {', '.join(hyp['supporting_experts'])}")
                    if hyp.get('key_data') and hyp['key_data'] != 'none': st.markdown(f"**Key data:** {hyp['key_data']}")

            if st.button("🗑️ Clear All Hypotheses"):
                st.session_state.tracked_hypotheses = []
                st.rerun()

        # Export
        st.markdown("---")
        st.subheader("📤 Export Discussion")
        if st.button("📄 Export to Markdown"):
            papers = []
            for c in (citations or []):
                if hasattr(c, 'pmid'):
                    papers.append({'pmid': c.pmid, 'title': c.title, 'authors': c.authors, 'journal': c.journal, 'year': c.year})
                else:
                    papers.append(c)
            expert_responses = {}
            for round_num in sorted(st.session_state.expert_discussion.keys()):
                for expert_name, response in st.session_state.expert_discussion[round_num].items():
                    if expert_name not in expert_responses: expert_responses[expert_name] = []
                    expert_responses[expert_name].append(response)
            markdown = export_discussion_to_markdown(clinical_question=clinical_question, expert_responses=expert_responses, citations=papers, scenario=scenario)
            st.download_button(label="Download Markdown", data=markdown, file_name=f"expert_discussion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", mime="text/markdown")

        # =========================================================================
        # HUMAN FEEDBACK LOOP
        # =========================================================================
        st.markdown("---")
        st.subheader("Human Feedback")
        st.caption("Provide feedback to steer the next discussion round")

        feedback_types = {
            "correction": "Correct an error or misstatement",
            "deepen": "Request deeper analysis on a topic",
            "redirect": "Shift focus to a different aspect",
            "challenge": "Challenge an assumption or conclusion",
            "add_context": "Add missing context or information"
        }

        col1, col2 = st.columns([2, 3])
        with col1:
            feedback_type = st.selectbox(
                "Feedback Type",
                options=list(feedback_types.keys()),
                format_func=lambda x: feedback_types[x]
            )

        with col2:
            target_expert = st.selectbox(
                "Target Expert (optional)",
                options=["All Experts"] + list(latest_responses.keys())
            )

        feedback_text = st.text_area(
            "Your Feedback",
            placeholder="e.g., 'The DMPK assessment didn't account for food effect on bioavailability' or 'Need more detail on the competitive landscape for this indication'",
            height=80
        )

        if st.button("Add Feedback", type="secondary", disabled=not feedback_text.strip()):
            new_feedback = {
                "type": feedback_type,
                "target": target_expert if target_expert != "All Experts" else None,
                "text": feedback_text.strip(),
                "round": latest_round,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.human_feedback.append(new_feedback)
            st.success("Feedback added! It will be incorporated in the next round.")

        # Display existing feedback
        if st.session_state.human_feedback:
            st.markdown("**Pending Feedback:**")
            for i, fb in enumerate(st.session_state.human_feedback):
                target_str = f" → {fb['target']}" if fb.get('target') else ""
                type_icon = {"correction": "[FIX]", "deepen": "[DEEP]", "redirect": "[PIVOT]", "challenge": "[?]", "add_context": "[+]"}.get(fb['type'], "")
                with st.expander(f"{type_icon} {fb['text'][:50]}...{target_str}"):
                    st.markdown(f"**Type:** {feedback_types.get(fb['type'], fb['type'])}")
                    if fb.get('target'):
                        st.markdown(f"**Target:** {fb['target']}")
                    st.markdown(f"**Feedback:** {fb['text']}")
                    st.caption(f"Added during Round {fb.get('round', '?')}")
                    if st.button("Remove", key=f"remove_fb_{i}"):
                        st.session_state.human_feedback.pop(i)
                        st.rerun()

            # Generate feedback-informed prompt for next round
            if st.button("Generate Feedback-Informed Prompt"):
                feedback_prompt_parts = []
                for fb in st.session_state.human_feedback:
                    target_str = f" (specifically for {fb['target']})" if fb.get('target') else ""
                    feedback_prompt_parts.append(f"- [{fb['type'].upper()}]{target_str}: {fb['text']}")

                feedback_prompt = f"""**Human Reviewer Feedback from Previous Round:**

Please address the following feedback in your response:
{chr(10).join(feedback_prompt_parts)}

Incorporate this feedback while maintaining your expert perspective."""
                st.code(feedback_prompt, language="markdown")
                st.info("Copy this prompt and add it to the research question for the next round, or use the 'Use as context' button below.")
                if st.button("Use as context for next round"):
                    if clinical_question:
                        st.session_state.expert_clinical_question = f"{clinical_question}\n\n{feedback_prompt}"
                    st.session_state.human_feedback = []  # Clear after use
                    st.rerun()

    # =========================================================================
    # INTERACTIVE Q&A CHAT
    # =========================================================================
    st.markdown("---")
    st.subheader("💬 Interactive Q&A")
    st.caption("Ask follow-up questions - each selected expert will respond with streaming")

    if st.session_state.expert_discussion:
        if not st.session_state.active_chat_experts:
            st.session_state.active_chat_experts = selected_experts

        for msg in st.session_state.expert_chat_messages:
            if msg["role"] == "user":
                with st.chat_message("user"): st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🧬"):
                    st.markdown(f"**{msg.get('expert', 'Expert')}**")
                    st.markdown(msg["content"])

        if user_question := st.chat_input("Ask a follow-up question to the expert panel..."):
            st.session_state.expert_chat_messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"): st.markdown(user_question)

            chat_context = build_chat_context(citations or [], st.session_state.expert_discussion)

            for expert_name in st.session_state.active_chat_experts:
                with st.chat_message("assistant", avatar="🧬"):
                    st.markdown(f"**{expert_name}**")
                    response_placeholder = st.empty()
                    full_response = ""
                    try:
                        with st.spinner("Thinking..."):
                            for chunk in call_expert_stream(
                                persona_name=expert_name, clinical_question=user_question, evidence_context=chat_context,
                                round_num=1, previous_responses=None, priors_text=None,
                                model=settings.EXPERT_MODEL, max_completion_tokens=2000
                            ):
                                if chunk.get("type") == "chunk":
                                    full_response += chunk.get("content", "")
                                    response_placeholder.markdown(full_response + "▌")
                                elif chunk.get("type") == "complete":
                                    response_placeholder.markdown(full_response)
                        st.session_state.expert_chat_messages.append({"role": "assistant", "expert": expert_name, "content": full_response})
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        response_placeholder.markdown(error_msg)
                        st.session_state.expert_chat_messages.append({"role": "assistant", "expert": expert_name, "content": error_msg})
            st.rerun()

        if st.session_state.expert_chat_messages:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.expert_chat_messages = []
                    st.rerun()
            with col2:
                chat_export = "# Expert Q&A Chat\n\n"
                chat_export += f"**Research Question:** {clinical_question}\n\n---\n\n"
                for msg in st.session_state.expert_chat_messages:
                    if msg["role"] == "user": chat_export += f"## User Question\n{msg['content']}\n\n"
                    else: chat_export += f"### {msg.get('expert', 'Expert')}\n{msg['content']}\n\n"
                st.download_button("📥 Export Chat", chat_export, file_name=f"expert_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", mime="text/markdown", use_container_width=True)
    else:
        st.info("💡 Run an expert discussion round first, then come back here to ask follow-up questions.")
