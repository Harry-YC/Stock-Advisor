# API Reference - Literature Review Platform

## Core Modules

### PubMed Client (`core/pubmed_client.py`)

#### Citation Dataclass
```python
@dataclass
class Citation:
    pmid: str
    title: str
    authors: List[str]
    journal: str
    year: str
    abstract: str
    doi: str = ""
    fetched_at: str = ""
    publication_types: List[str] = None
    keywords: List[str] = None
```

#### PubMedClient Class
```python
class PubMedClient:
    def __init__(self, email: str, api_key: str = None):
        """Initialize PubMed client."""

    def search(self, query: str, max_results: int = 100, filters: dict = None) -> dict:
        """
        Search PubMed.

        Args:
            query: PubMed query string
            max_results: Maximum results to return
            filters: Optional filters (date_from, date_to, article_types)

        Returns:
            Dict with 'pmids', 'count', 'query_translation'
        """

    def fetch_citations(self, pmids: List[str]) -> Tuple[List[Citation], List[Dict]]:
        """Fetch full citation details for PMIDs.

        Returns:
            Tuple of (citations, failed_batches) where failed_batches contains
            dicts with 'start', 'end', 'error' keys for any failed batch requests.
        """

    def fetch_by_doi(self, dois: List[str]) -> Tuple[List[Citation], List[str]]:
        """Fetch citations by DOI. Returns (found, not_found)."""
```

---

### Database (`core/database.py`)

#### DatabaseManager
```python
class DatabaseManager:
    def __init__(self, db_path: Path):
        """Initialize SQLite database connection."""

    def get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
```

#### ProjectDAO
```python
class ProjectDAO:
    def create_project(self, name: str, description: str = "") -> int:
        """Create new project. Returns project_id."""

    def get_project(self, project_id: int) -> Optional[dict]:
        """Get project by ID."""

    def get_all_projects(self) -> List[dict]:
        """Get all projects."""

    def delete_project(self, project_id: int) -> bool:
        """Delete project and all associated data."""
```

#### CitationDAO
```python
class CitationDAO:
    def upsert_citation(self, citation: dict) -> None:
        """Insert or update citation."""

    def get_citations_batch(self, pmids: List[str]) -> Dict[str, dict]:
        """Get multiple citations by PMID."""

    def add_citation_to_project(self, project_id: int, pmid: str) -> None:
        """Associate citation with project."""
```

#### ScreeningDAO
```python
class ScreeningDAO:
    def save_decision(self, project_id: int, pmid: str, decision: str,
                      confidence: float = None, reasoning: str = None) -> None:
        """Save screening decision."""

    def get_decisions(self, project_id: int) -> List[dict]:
        """Get all screening decisions for project."""
```

---

### AI Screener (`core/ai_screener.py`)

```python
def screen_papers_batch(
    papers: List[dict],
    criteria: dict,
    api_key: str,
    model: str = "gpt-5-mini",
    max_workers: int = 5
) -> List[dict]:
    """
    Screen multiple papers against inclusion criteria.

    Args:
        papers: List of paper dicts with 'pmid', 'title', 'abstract'
        criteria: Dict with 'inclusion', 'exclusion' criteria lists
        api_key: OpenAI API key
        model: Model to use
        max_workers: Parallel workers

    Returns:
        List of screening results with 'pmid', 'decision', 'confidence', 'reasoning'
    """

def get_screening_summary(decisions: List[dict]) -> dict:
    """Get summary statistics from screening decisions."""
```

---

### Ranking (`core/ranking.py`)

```python
@dataclass
class RankingWeights:
    relevance: float = 0.25
    evidence: float = 0.25
    recency: float = 0.25
    influence: float = 0.25

RANKING_PRESETS = {
    "balanced": RankingWeights(0.25, 0.25, 0.25, 0.25),
    "clinical_appraisal": RankingWeights(0.2, 0.5, 0.15, 0.15),
    "discovery": RankingWeights(0.5, 0.15, 0.2, 0.15),
    "preclinical": RankingWeights(0.3, 0.3, 0.2, 0.2)
}

def rank_citations(
    citations: List[dict],
    weights: RankingWeights,
    original_query: str,
    use_ai_relevance: bool = False,
    openai_api_key: str = None,
    model: str = "gpt-5-mini"
) -> List[ScoredCitation]:
    """
    Rank citations by clinical utility score.

    Returns list of ScoredCitation objects with:
    - citation: Original citation dict
    - final_score: Combined score (0-1)
    - relevance_score, evidence_score, recency_score, influence_score
    - explanation: List of string tags
    - rank_position: Original position
    """
```

---

## Services Layer

### ExpertDiscussionService (`services/expert_service.py`)

```python
class ExpertDiscussionService:
    def __init__(self, api_key: str, model: str = None, max_tokens: int = None):
        """Initialize expert discussion service."""

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
        progress_callback: Optional[Callable] = None
    ) -> DiscussionRoundResult:
        """
        Run a single discussion round.

        Returns:
            DiscussionRoundResult with responses, failures, follow_up_questions
        """

    def regenerate_response(
        self,
        expert_name: str,
        round_num: int,
        clinical_question: str,
        citations: List[Dict],
        scenario: str,
        rejection_critique: str,
        ...
    ) -> Dict:
        """Regenerate single expert response with feedback."""

    def generate_follow_up_questions(
        self,
        clinical_question: str,
        responses: Dict[str, Dict],
        max_questions: int = 4
    ) -> List[str]:
        """Generate follow-up questions based on discussion."""

    def extract_knowledge(
        self,
        responses: Dict[str, Dict],
        clinical_question: str,
        source_name: str
    ) -> Dict:
        """Extract knowledge from discussion responses."""
```

---

### AnalysisService (`services/analysis_service.py`)

```python
class AnalysisService:
    def __init__(self, api_key: str):
        """Initialize analysis service."""

    def analyze_gaps(
        self,
        responses: Dict[str, Dict],
        scenario: str
    ) -> GapAnalysisResult:
        """
        Run gap analysis on discussion.

        Returns:
            GapAnalysisResult with coverage_score, quantification_score,
            strengths, gaps, evidence_issues, recommendations
        """

    def detect_conflicts(
        self,
        responses: Dict[str, Dict]
    ) -> ConflictAnalysisResult:
        """
        Detect conflicts in expert responses.

        Returns:
            ConflictAnalysisResult with conflicts list, clarification_needed
        """

    def synthesize_responses(
        self,
        responses: Dict[str, str],
        clinical_question: str,
        model: str = "gpt-5-mini"
    ) -> SynthesisResult:
        """
        Generate meta-synthesis of expert responses.

        Returns:
            SynthesisResult with synthesis, consensus_points,
            open_questions, recommended_actions
        """

    def extract_hypotheses(
        self,
        responses: Dict[str, str],
        clinical_question: str,
        round_num: int
    ) -> List[Dict]:
        """Extract hypotheses from expert discussion."""
```

---

### ChatService (`services/chat_service.py`)

```python
class ChatService:
    def __init__(self, api_key: str, model: str = "gpt-5-mini"):
        """Initialize chat service."""

    def build_context(
        self,
        citations: List,
        discussion: Dict,
        uploaded_docs: Optional[List] = None,
        max_papers: int = 10,
        max_abstract_chars: int = 500
    ) -> str:
        """Build context string from available sources."""

    def get_expert_response_stream(
        self,
        expert_name: str,
        question: str,
        context: str,
        max_tokens: int = 2000
    ) -> Generator[Dict, None, None]:
        """
        Stream expert response chunks.

        Yields:
            Dicts with 'type' ('chunk' or 'complete') and 'content'
        """
```

---

## UI Modules

### Literature Search (`ui/literature_search.py`)

```python
def render_literature_search(
    citation_dao: CitationDAO,
    search_dao: SearchHistoryDAO,
    query_cache_dao: QueryCacheDAO
) -> None:
    """
    Render the Literature Search tab.

    Features:
    - Query-based PubMed search with AI optimization
    - Identifier search (PMID/DOI)
    - File import (RIS, BibTeX)
    - Document upload for RAG
    - Visualization (timeline, network, table)
    """
```

---

### Expert Panel (`ui/expert_panel.py`)

```python
def render_expert_panel(
    expert_discussion_dao: ExpertDiscussionDAO,
    citation_dao: CitationDAO
) -> None:
    """
    Render the Expert Panel Discussion tab.

    Features:
    - Research question input
    - Expert selection (8+ drug development personas)
    - Multi-round discussions
    - HITL controls (inject evidence, working memory, temperature)
    - Meta-synthesis
    - Gap analysis and conflict detection
    - Hypothesis tracking
    - Interactive Q&A chat
    """
```

---

### Sidebar (`ui/sidebar.py`)

```python
def render_sidebar(
    project_dao: ProjectDAO,
    citation_dao: CitationDAO,
    screening_dao: ScreeningDAO,
    zotero_client_class=None
) -> None:
    """
    Render the sidebar.

    Features:
    - Project manager (create, select, delete)
    - Data management
    - Knowledge store display
    - Export panel (CSV, RIS, BibTeX)
    - Configuration warnings
    """
```

---

### Context Utilities (`ui/context_utils.py`)

```python
def get_available_context() -> Tuple[List, List, List, Dict]:
    """Get all available context from session state."""

def render_context_indicator() -> None:
    """Render the context indicator bar."""

def has_context() -> bool:
    """Check if any context is available."""
```

---

## State Management

### State Manager (`core/state_manager.py`)

```python
def init_all_session_state() -> None:
    """Initialize all session state from centralized definitions."""

def reset_project_state() -> None:
    """Reset state when project changes."""

# Session State Keys:
# - current_project_id, current_project_name
# - search_results, selected_papers
# - expert_discussion, discussion_round
# - gap_analysis, conflict_analysis, meta_synthesis
# - suggested_questions, tracked_hypotheses
# - working_memory, human_feedback
# - expert_chat_messages, active_chat_experts
# - injected_evidence, expert_temperatures
# - rag_context, indexed_documents, uploaded_documents
```

---

## Configuration

### Settings (`config/settings.py`)

```python
# Environment Variables:
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "user@example.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
EXPERT_MODEL = os.getenv("EXPERT_MODEL", "gpt-5-mini")
EXPERT_MAX_TOKENS = int(os.getenv("EXPERT_MAX_TOKENS", "4000"))
ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")

# Feature Flags:
ENABLE_LOCAL_RAG = bool(os.getenv("ENABLE_LOCAL_RAG", "false").lower() == "true")

# Paths:
OUTPUTS_DIR = Path("outputs")
QDRANT_STORAGE_PATH = OUTPUTS_DIR / "qdrant"
```
