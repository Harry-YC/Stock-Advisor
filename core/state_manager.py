"""
Session State Manager

Centralized management of all Streamlit session state for the Literature Review platform.
This module is the single source of truth for session state initialization and reset.

Usage:
    from core.state_manager import init_all_session_state, reset_project_state, StateKeys

    # At app startup
    init_all_session_state()

    # When switching projects
    reset_project_state()

    # Access state with constants
    results = get_state(StateKeys.SEARCH_RESULTS)
    set_state(StateKeys.DISCUSSION_ROUND, 1)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import streamlit as st


# =============================================================================
# STATE KEYS - Centralized constants for session state access
# =============================================================================

class StateKeys:
    """
    Centralized session state key constants.

    Use these constants instead of string literals to:
    - Prevent typos
    - Enable IDE autocomplete
    - Make refactoring easier
    - Document available state keys

    Usage:
        from core.state_manager import StateKeys, get_state, set_state

        # Read state
        results = get_state(StateKeys.SEARCH_RESULTS)

        # Write state
        set_state(StateKeys.DISCUSSION_ROUND, 1)
    """

    # Project
    CURRENT_PROJECT_ID = 'current_project_id'
    CURRENT_PROJECT_NAME = 'current_project_name'
    SEARCH_RESULTS = 'search_results'
    SELECTED_PAPERS = 'selected_papers'

    # Discussion
    EXPERT_DISCUSSION = 'expert_discussion'
    DISCUSSION_ROUND = 'discussion_round'
    SUGGESTED_QUESTIONS = 'suggested_questions'
    META_SYNTHESIS = 'meta_synthesis'
    PERSPECTIVE_QUESTIONS = 'perspective_questions'
    TRACKED_HYPOTHESES = 'tracked_hypotheses'
    GAP_ANALYSIS = 'gap_analysis'
    CONFLICT_ANALYSIS = 'conflict_analysis'
    WORKING_MEMORY = 'working_memory'
    HUMAN_FEEDBACK = 'human_feedback'

    # Chat
    CHAT_MODE = 'chat_mode'
    CHAT_EXPERT = 'chat_expert'
    CHAT_CONVERSATION = 'chat_conversation'
    CHAT_HISTORY = 'chat_history'
    EXPERT_CHAT_MESSAGES = 'expert_chat_messages'
    ACTIVE_CHAT_EXPERTS = 'active_chat_experts'
    EXPERT_CHAT_INSTANCE = 'expert_chat_instance'

    # HITL (Human-in-the-Loop)
    INJECTED_EVIDENCE = 'injected_evidence'
    REGENERATE_TARGET = 'regenerate_target'
    SHOW_REGENERATE_MODAL = 'show_regenerate_modal'
    EXPERT_TEMPERATURES = 'expert_temperatures'

    # RAG
    RAG_CONTEXT = 'rag_context'
    INDEXED_DOCUMENTS = 'indexed_documents'
    UPLOADED_DOCUMENTS = 'uploaded_documents'
    INDEXED_DOCUMENTS_DIRTY = 'indexed_documents_dirty'

    # UI Toggles
    PANEL_USE_WEB_SEARCH = 'panel_use_web_search'
    CHAT_USE_WEB_SEARCH = 'chat_use_web_search'

    # Screening
    AI_SCREENING_RESULTS = 'ai_screening_results'

    # Two-Pass Mode
    AUTO_SEARCH_ENABLED = 'auto_search_enabled'
    AUTO_SEARCH_MAX = 'auto_search_max'
    PASS1_RESPONSES = 'pass1_responses'
    PASS2_VALIDATIONS = 'pass2_validations'
    LITERATURE_SEARCH_RESULTS = 'literature_search_results'
    TWO_PASS_MODE = 'two_pass_mode'

    # Conversational Mode (v3.0)
    UI_MODE = 'ui_mode'
    VIEW_STATE = 'view_state'
    CURRENT_QUESTION = 'current_question'
    QUESTION_TYPE = 'question_type'
    AUTO_SELECTED_EXPERTS = 'auto_selected_experts'
    RESEARCH_RESULT = 'research_result'
    EVIDENCE_SUMMARY = 'evidence_summary'
    INLINE_CHAT_HISTORY = 'inline_chat_history'
    SHOW_EVIDENCE_DRAWER = 'show_evidence_drawer'
    PROCESSING_STAGE = 'processing_stage'
    RECENT_QUESTIONS = 'recent_questions'

    # CDP (v4.0)
    CDP_SECTIONS = 'cdp_sections'
    CDP_PROJECT_NAME = 'cdp_project_name'
    CDP_LAST_MODIFIED = 'cdp_last_modified'
    CDP_SHOW_WORKSPACE = 'cdp_show_workspace'

    # Evidence Corpus (GRADE v2.0)
    EVIDENCE_CORPUS = 'evidence_corpus'
    CURRENT_ETD = 'current_etd'
    GENERATED_RECOMMENDATION = 'generated_recommendation'


def get_state(key: str, default: Any = None) -> Any:
    """
    Get session state value with default.

    Args:
        key: State key (use StateKeys constants)
        default: Default value if key not found

    Returns:
        State value or default

    Example:
        results = get_state(StateKeys.SEARCH_RESULTS, {})
    """
    return st.session_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    """
    Set session state value.

    Args:
        key: State key (use StateKeys constants)
        value: Value to set

    Example:
        set_state(StateKeys.DISCUSSION_ROUND, 1)
    """
    st.session_state[key] = value


def has_state(key: str) -> bool:
    """
    Check if session state key exists.

    Args:
        key: State key (use StateKeys constants)

    Returns:
        True if key exists in session state
    """
    return key in st.session_state


def delete_state(key: str) -> None:
    """
    Delete session state key if it exists.

    Args:
        key: State key (use StateKeys constants)
    """
    if key in st.session_state:
        del st.session_state[key]


# =============================================================================
# STATE DATACLASSES - Define structure and defaults
# =============================================================================

@dataclass
class ProjectState:
    """Project-level state variables."""
    current_project_id: Optional[str] = None
    current_project_name: Optional[str] = None
    search_results: Optional[Dict] = None
    selected_papers: Set[str] = field(default_factory=set)


@dataclass
class DiscussionState:
    """Expert panel discussion state."""
    expert_discussion: Dict = field(default_factory=dict)  # {round_num: {expert_name: response}}
    discussion_round: int = 1
    suggested_questions: List = field(default_factory=list)
    meta_synthesis: Optional[Dict] = None
    perspective_questions: Dict = field(default_factory=dict)  # {expert_name: [questions]}
    tracked_hypotheses: List = field(default_factory=list)
    # Analysis results
    gap_analysis: Optional[Dict] = None
    conflict_analysis: Optional[Dict] = None
    working_memory: Any = None
    human_feedback: List = field(default_factory=list)


@dataclass
class ChatState:
    """Expert chat state."""
    chat_mode: str = "single_expert"  # "single_expert" or "panel_router"
    chat_expert: str = "Bioscience Lead"
    chat_conversation: Any = None
    chat_history: List = field(default_factory=list)
    expert_chat_messages: List = field(default_factory=list)  # [{role, content, expert}]
    active_chat_experts: List = field(default_factory=list)
    expert_chat_instance: Any = None


@dataclass
class HITLState:
    """Human-in-the-loop mission control state."""
    injected_evidence: List = field(default_factory=list)  # Papers to force-include
    regenerate_target: Optional[tuple] = None  # (round_num, expert_name)
    show_regenerate_modal: bool = False
    expert_temperatures: Dict = field(default_factory=dict)  # {expert_name: float}


@dataclass
class RAGState:
    """Local RAG (Retrieval-Augmented Generation) state."""
    rag_context: List = field(default_factory=list)  # Retrieved document chunks
    indexed_documents: List = field(default_factory=list)  # Document metadata
    uploaded_documents: List = field(default_factory=list)  # User-uploaded docs
    indexed_documents_dirty: bool = False  # Refresh flag


@dataclass
class UIToggleState:
    """UI toggle states."""
    panel_use_web_search: bool = True
    chat_use_web_search: bool = True


@dataclass
class ScreeningState:
    """AI screening state."""
    ai_screening_results: Optional[List] = None


@dataclass
class TwoPassState:
    """Two-pass expert panel state (Perplexity-style)."""
    auto_search_enabled: bool = True  # Enable background literature search
    auto_search_max: int = 20  # Max papers to fetch
    pass1_responses: Optional[Dict] = None  # {expert_name: response}
    pass2_validations: Optional[Dict] = None  # {expert_name: ValidationResult}
    literature_search_results: Optional[Dict] = None  # Background search results
    two_pass_mode: bool = True  # Whether to use two-pass mode


@dataclass
class ConversationalState:
    """Conversational mode state for Research Partner v3.0."""
    ui_mode: str = "conversational"  # "conversational" or "advanced"
    view_state: str = "home"  # "home" | "processing" | "answer"
    current_question: Optional[str] = None
    question_type: Optional[str] = None  # "surgical_candidate", "general", etc.
    auto_selected_experts: List = field(default_factory=list)
    research_result: Optional[Dict] = None  # Final synthesized ResearchResult
    evidence_summary: Optional[Dict] = None  # Citations + validation results
    inline_chat_history: List = field(default_factory=list)  # Follow-up messages
    show_evidence_drawer: bool = False
    processing_stage: str = "idle"  # idle, parsing, searching, consulting, validating, synthesizing, complete
    recent_questions: List = field(default_factory=list)  # History: [{question, question_type, timestamp, result_summary}]


@dataclass
class CDPState:
    """Clinical Development Plan workspace state (v4.0)."""
    cdp_sections: Dict = field(default_factory=dict)  # {section_name: {content, question, timestamp, citations}}
    cdp_project_name: str = ""
    cdp_last_modified: Optional[str] = None
    cdp_show_workspace: bool = False


@dataclass
class EvidenceCorpusState:
    """Evidence Corpus state for GRADE methodology (v2.0)."""
    evidence_corpus: Any = None  # EvidenceCorpus object - single source of truth
    current_etd: Any = None  # Current Evidence-to-Decision card
    generated_recommendation: Any = None  # Generated GRADE recommendation


# =============================================================================
# INITIALIZATION FUNCTIONS
# =============================================================================

def _init_from_dataclass(defaults: Any) -> None:
    """
    Initialize session state from a dataclass instance.
    Only sets values that don't already exist in session state.
    """
    for key, value in defaults.__dict__.items():
        if key not in st.session_state:
            # Handle mutable defaults properly
            if isinstance(value, (dict, list, set)):
                st.session_state[key] = type(value)(value)
            else:
                st.session_state[key] = value


def init_all_session_state() -> None:
    """
    Initialize all session state variables.

    This is the SINGLE SOURCE OF TRUTH for session state initialization.
    Call this once at app startup before any UI rendering.
    """
    # Initialize all state groups
    _init_from_dataclass(ProjectState())
    _init_from_dataclass(DiscussionState())
    _init_from_dataclass(ChatState())
    _init_from_dataclass(HITLState())
    _init_from_dataclass(RAGState())
    _init_from_dataclass(UIToggleState())
    _init_from_dataclass(ScreeningState())
    _init_from_dataclass(TwoPassState())
    _init_from_dataclass(ConversationalState())
    _init_from_dataclass(CDPState())
    _init_from_dataclass(EvidenceCorpusState())

    # Handle migration: selected_papers list -> set
    if isinstance(st.session_state.get('selected_papers'), list):
        st.session_state.selected_papers = set(st.session_state.selected_papers)


def reset_project_state() -> None:
    """
    Reset state when switching projects.
    Clears project-specific data while preserving app-level settings.
    """
    # Clear project data
    st.session_state.search_results = None
    st.session_state.selected_papers = set()

    # Clear discussion data
    st.session_state.expert_discussion = {}
    st.session_state.discussion_round = 1
    st.session_state.suggested_questions = []
    st.session_state.meta_synthesis = None
    st.session_state.perspective_questions = {}
    st.session_state.tracked_hypotheses = []
    st.session_state.gap_analysis = None
    st.session_state.conflict_analysis = None
    st.session_state.working_memory = None
    st.session_state.human_feedback = []

    # Clear chat data
    st.session_state.chat_conversation = None
    st.session_state.expert_chat_messages = []
    st.session_state.active_chat_experts = []
    st.session_state.expert_chat_instance = None

    # Clear HITL data
    st.session_state.injected_evidence = []
    st.session_state.regenerate_target = None
    st.session_state.show_regenerate_modal = False
    st.session_state.expert_temperatures = {}

    # Clear RAG data
    st.session_state.rag_context = []
    st.session_state.indexed_documents = []
    st.session_state.uploaded_documents = []
    st.session_state.indexed_documents_dirty = False

    # Clear screening data
    st.session_state.ai_screening_results = None

    # Clear conversational mode data (but preserve ui_mode preference)
    st.session_state.current_question = None
    st.session_state.question_type = None
    st.session_state.auto_selected_experts = []
    st.session_state.research_result = None
    st.session_state.evidence_summary = None
    st.session_state.inline_chat_history = []
    st.session_state.show_evidence_drawer = False
    st.session_state.processing_stage = "idle"


def reset_conversational_state() -> None:
    """Reset conversational mode state (for 'New Question' button)."""
    st.session_state.view_state = "home"
    st.session_state.current_question = None
    st.session_state.question_type = None
    st.session_state.auto_selected_experts = []
    st.session_state.research_result = None
    st.session_state.evidence_summary = None
    st.session_state.inline_chat_history = []
    st.session_state.show_evidence_drawer = False
    st.session_state.processing_stage = "idle"
    # Note: recent_questions is NOT reset - it persists across questions


def reset_discussion_state() -> None:
    """Reset only discussion-related state (for "New Discussion" button)."""
    st.session_state.expert_discussion = {}
    st.session_state.discussion_round = 1
    st.session_state.suggested_questions = []
    st.session_state.meta_synthesis = None
    st.session_state.gap_analysis = None
    st.session_state.conflict_analysis = None
    st.session_state.expert_chat_messages = []
    st.session_state.human_feedback = []
    st.session_state.tracked_hypotheses = []
    # Reset two-pass state
    st.session_state.pass1_responses = None
    st.session_state.pass2_validations = None
    st.session_state.literature_search_results = None


def reset_chat_state() -> None:
    """Reset chat state (for "New Conversation" button)."""
    st.session_state.chat_conversation = None
    st.session_state.expert_chat_messages = []


# =============================================================================
# STATE ACCESSORS - Type-safe getters
# =============================================================================

def get_citations() -> List:
    """Get citations from search results, if available."""
    results = st.session_state.get('search_results')
    if results and isinstance(results, dict):
        return results.get('citations', [])
    return []


def get_context_summary() -> Dict[str, int]:
    """Get summary of available context."""
    return {
        'papers': len(get_citations()),
        'documents': len(st.session_state.get('indexed_documents', [])) +
                    len(st.session_state.get('uploaded_documents', [])),
        'discussions': len(st.session_state.get('expert_discussion', {}))
    }


def has_context() -> bool:
    """Check if any context is available."""
    summary = get_context_summary()
    return any(v > 0 for v in summary.values())


# =============================================================================
# STATE KEYS - For documentation and validation
# =============================================================================

ALL_STATE_KEYS = [
    # Project
    'current_project_id', 'current_project_name', 'search_results', 'selected_papers',
    # Discussion
    'expert_discussion', 'discussion_round', 'suggested_questions', 'meta_synthesis',
    'perspective_questions', 'tracked_hypotheses', 'gap_analysis', 'conflict_analysis',
    'working_memory', 'human_feedback',
    # Chat
    'chat_mode', 'chat_expert', 'chat_conversation', 'chat_history',
    'expert_chat_messages', 'active_chat_experts', 'expert_chat_instance',
    # HITL
    'injected_evidence', 'regenerate_target', 'show_regenerate_modal', 'expert_temperatures',
    # RAG
    'rag_context', 'indexed_documents', 'uploaded_documents', 'indexed_documents_dirty',
    # UI Toggles
    'panel_use_web_search', 'chat_use_web_search',
    # Screening
    'ai_screening_results',
    # Two-Pass Mode
    'auto_search_enabled', 'auto_search_max', 'pass1_responses', 'pass2_validations',
    'literature_search_results', 'two_pass_mode',
    # Conversational Mode (v3.0)
    'ui_mode', 'view_state', 'current_question', 'question_type', 'auto_selected_experts',
    'research_result', 'evidence_summary', 'inline_chat_history',
    'show_evidence_drawer', 'processing_stage', 'recent_questions',
    # CDP (v4.0)
    'cdp_sections', 'cdp_project_name', 'cdp_last_modified', 'cdp_show_workspace',
    # Evidence Corpus (GRADE v2.0)
    'evidence_corpus', 'current_etd', 'generated_recommendation',
]
