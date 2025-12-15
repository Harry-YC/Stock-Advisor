# CLAUDE.md - AI Assistant Context

## Project Overview

**Palliative Surgery GDG** is a guideline development platform for palliative surgery clinical questions. The core workflow is:
1. Search for evidence (PubMed + preprints)
2. Visualize results (timeline, citation network, sortable tables)
3. Consult with AI-powered GDG expert panel (12 experts)
4. Generate evidence-based recommendations with epistemic tagging
5. Build clinical guidelines in the Guideline Workspace

**Focus**: Palliative surgery guideline development following GRADE methodology.

## Key Design Decisions

- **Target users**: Surgical oncologists, palliative care physicians, guideline developers
- **Scale**: Optimized for 200-1000 papers per session
- **AI experts**: 12 GDG personas (Surgical Oncologist, Palliative Care, GRADE Methodologist, etc.)
- **Persistence**: SQLite for projects, JSON for knowledge store (atomic writes)
- **Default AI model**: `gemini-3-pro-preview` for expert discussions

## AI Models (VALIDATED)

The following AI models are used throughout the codebase and are **VALIDATED**:

| Model | Provider | Purpose |
|-------|----------|---------|
| `gpt-5-mini` | OpenAI | Fast queries, screening, parsing |
| `gemini-3-pro-preview` | Google | Expert discussions, synthesis, complex analysis |

**IMPORTANT**: These are the correct model names. Do NOT change them to older model names.

## Architecture

```
app_gl.py                    # Main Streamlit app (entry point)
├── config/
│   ├── settings.py          # Configuration including ENABLE_ADVANCED_TOOLS flag
│   ├── domain_config.py     # Pluggable domain configuration (v2)
│   └── palliative_surgery_vocabulary.py  # Legacy vocabulary (v1)
├── core/                    # Business logic (no UI dependencies)
│   ├── pubmed_client.py     # PubMed API - Citation dataclass
│   ├── citation_network.py  # NetworkX graphs, Plotly viz
│   ├── data_extractor.py    # GPT evidence extraction
│   ├── knowledge_store.py   # Persistent facts + triples (atomic writes)
│   ├── ai_screener.py       # AI-powered paper screening
│   ├── query_parser.py      # AI query optimization
│   ├── ranking.py           # Evidence-based scoring
│   ├── state_manager.py     # Session state initialization
│   ├── question_templates.py # GDG question types with expert mappings
│   └── database.py          # SQLite DAOs
├── integrations/            # External APIs (stateless clients)
│   ├── semantic_scholar.py  # Citation data
│   ├── clinicaltrials.py    # ClinicalTrials.gov
│   └── biorxiv.py           # Preprints
├── services/                # Business logic services
│   ├── expert_service.py    # Two-Pass Expert Discussion engine
│   ├── research_partner_service.py  # Unified research orchestration
│   ├── llm_router.py        # Unified LLM interface (Gemini/OpenAI routing)
│   ├── recommendation_service.py  # GRADE-style recommendation generation
│   ├── challenger_service.py # Red Team challenge generation (P3)
│   ├── quick_answer_service.py # Fast Q&A with citations (P4)
│   └── feedback_service.py  # User feedback/marks persistence (P6)
├── gdg/                     # Guideline Development Group domain
│   ├── gdg_personas.py      # 12 GDG expert definitions
│   ├── gdg_utils.py         # LLM calling, evidence formatting
│   └── response_validator.py # Epistemic tag validation
├── tests/                   # Test suite
│   ├── test_two_pass_playwright.py  # E2E tests for Two-Pass mode
│   ├── test_conversational_mode_playwright.py  # Conversational mode tests
│   ├── test_domain_search.py  # Domain search v2 tests (31 tests)
│   └── e2e/                 # Comprehensive E2E feature tests
│       ├── conftest.py      # Shared fixtures and helpers
│       ├── test_questions.py # Test case definitions
│       ├── test_citation_highlighting.py  # P1 feature tests
│       ├── test_quick_answer.py  # P4 feature tests
│       ├── test_challenger.py  # P3 feature tests
│       ├── test_smart_suggestions.py  # P5 feature tests
│       ├── test_mark_pen.py  # P6 feature tests
│       └── test_full_flow.py  # Integration tests
└── ui/                      # Streamlit components (4 main tabs)
    ├── home.py              # Tab 1: "Ask the GDG" conversational mode
    ├── literature_search.py # Tab 2: "Evidence Library"
    ├── expert_panel.py      # Tab 3: "GDG Discussion"
    ├── export.py            # Tab 4: "Export" recommendations/discussion
    ├── answer_view.py       # Synthesized recommendation display + suggestions (P5)
    ├── evidence_drawer.py   # Collapsible evidence panel
    ├── sidebar.py           # Navigation and project management
    ├── citation_utils.py    # Citation highlighting + epistemic tags (P1)
    └── mark_pen.py          # User feedback annotation UI (P6)
```

## UI Tabs (v1.0)

| Tab | Purpose |
|-----|---------|
| 💬 Ask the GDG | Conversational mode - quick questions, auto-search |
| 📚 Evidence Library | Search PubMed, screen papers, build corpus |
| 👥 GDG Discussion | Multi-round expert panel deliberation |
| 📄 Export | Download recommendations, discussion, evidence |

## Feature Gating

Set `ENABLE_ADVANCED_TOOLS=true` to enable advanced features:
- Hypothesis Lab (discovery mode)
- Strategy War Room (planning mode)
- Knowledge Store UI
- Mode switching between advanced modes

When disabled (default), the UI is simplified for guideline development only.

## GDG Expert Personas

12 expert personas for comprehensive guideline development:

| Expert | Focus Area |
|--------|------------|
| Surgical Oncologist | Technical feasibility, operative mortality/morbidity |
| Perioperative Medicine Physician | Risk stratification, ASA, frailty, ICU trajectory |
| Interventionalist | Stents, embolization, EUS-GE (non-surgical alternatives) |
| Palliative Care Physician | QoL, goals of care, patient perspective |
| Patient Advocate | Patient values, days at home, treatment burden |
| GRADE Methodologist | Study design, risk of bias, GRADE certainty |
| Clinical Evidence Specialist | Data extraction, comparative effectiveness |
| Medical Ethicist | Proportionality, informed consent, appropriateness |
| Pain and Symptom-Control Specialist | Intractable pain, palliative sedation |
| Geriatric and Frailty Specialist | Frailty scores, CGA, geriatric syndromes |
| Health Economist | ICER, QALY, resource utilization, implementation |
| GDG Chair | Synthesis only (no evidence review) |

## Epistemic Tagging

All expert responses must use epistemic tags:

```
• EVIDENCE (PMID: XXXXX) - Numbers/data must have PMID citation
• ASSUMPTION: - Extrapolations beyond available data
• OPINION: - Clinical judgment, values-based decisions
• EVIDENCE GAP → - Missing data that would change recommendation
```

## GDG Question Types

| Type | Icon | Description |
|------|------|-------------|
| `surgical_candidate` | 🔪 | Assess surgical candidacy |
| `palliative_pathway` | 🏥 | Design palliative care approach |
| `intervention_choice` | ⚖️ | Compare surgical vs non-surgical |
| `symptom_management` | 💊 | Address intractable symptoms |
| `prognosis_assessment` | 📊 | Evaluate evidence on outcomes |
| `ethics_review` | ⚖️ | Ethical considerations |
| `resource_allocation` | 💰 | Cost-effectiveness, implementation |
| `general` | ❓ | General GDG question |

## Clinical Scenarios

Pre-defined scenarios from the palliative surgery domain:

- **Malignant Bowel Obstruction**: Ovarian/colorectal with obstruction
- **Pathologic Fracture**: Long bone metastatic fracture
- **Malignant Airway Obstruction**: Central airway tumor
- **Bleeding Control**: Uncontrolled tumor hemorrhage
- **Custom Scenario**: User-defined

## Coding Patterns

### Expert Response Format
```python
from gdg import GDG_PERSONAS, get_gdg_prompts, call_expert

# Get all expert names
experts = list(GDG_PERSONAS.keys())

# Get prompts for discussion
prompts = get_gdg_prompts(bullets_per_role=5)

# Call an expert
response = call_expert(
    persona_name="Surgical Oncologist",
    clinical_question="Should this patient undergo palliative surgery?",
    evidence_context="[formatted citations]",
    openai_api_key=api_key
)
```

### Response Validation
```python
from gdg import ResponseValidator

validator = ResponseValidator(corpus_pmids={'12345678', '87654321'})
result = validator.validate(response_text)

if not result.is_valid:
    print("Errors:", result.errors)
print(f"Compliance: {result.compliance_score:.0%}")
```

### OpenAI Client Usage
- **ALWAYS** include timeout when creating OpenAI client
- Default timeout: 120 seconds (configured in `settings.OPENAI_TIMEOUT`)
- Use `gemini-3-pro-preview` for expert discussions

```python
from openai import OpenAI
from config import settings

# Correct - always include timeout
client = OpenAI(api_key=api_key, timeout=settings.OPENAI_TIMEOUT)
```

## Two-Pass Discussion Flow

**Pass 1 - Immediate Response:**
- Experts respond using LLM knowledge + optional web search
- Fast response time (~5-10 seconds)

**Pass 2 - Literature Validation:**
- Background thread searches PubMed for evidence
- Validates/refutes claims from Pass 1
- Adds citations with PMIDs

## 4-Round GDG Discussion (Optional Expansion)

When user clicks "Run Full GDG Discussion":

1. **Round 1: Evidence Review** - Extract findings with PMIDs
2. **Round 2: Conflict Resolution** - Address discrepancies
3. **Round 3: Decision Framework** - WHO BENEFITS/DOES NOT/THRESHOLDS
4. **Round 4: Synthesis** - FOR/AGAINST/CONDITIONAL recommendations

## Running the App

```bash
# Start the Palliative Surgery GDG app
streamlit run app_gl.py
```

## Running Tests

```bash
# Run all unit tests (fast, no browser)
pytest tests/e2e/ -v -k "unit or service or detection"

# Run Playwright E2E browser tests (requires running app)
TEST_URL=http://localhost:8501 pytest tests/e2e/ -v

# Run specific feature tests
TEST_URL=http://localhost:8501 pytest tests/e2e/test_citation_highlighting.py -v  # P1
TEST_URL=http://localhost:8501 pytest tests/e2e/test_quick_answer.py -v           # P4
TEST_URL=http://localhost:8501 pytest tests/e2e/test_challenger.py -v             # P3
TEST_URL=http://localhost:8501 pytest tests/e2e/test_smart_suggestions.py -v      # P5
TEST_URL=http://localhost:8501 pytest tests/e2e/test_mark_pen.py -v               # P6
TEST_URL=http://localhost:8501 pytest tests/e2e/test_full_flow.py -v              # Integration

# Run with visible browser (debugging)
TEST_URL=http://localhost:8501 HEADLESS=false pytest tests/e2e/ -v

# Legacy E2E tests
TEST_URL=http://localhost:8501 pytest tests/test_two_pass_playwright.py -v
TEST_URL=http://localhost:8501 pytest tests/test_conversational_mode_playwright.py -v
```

## Environment Variables

```bash
PUBMED_API_KEY      # Increases rate limit 3→10 req/sec
OPENAI_API_KEY      # Required for AI features
GOOGLE_API_KEY      # Required for Gemini models
```

## Guideline Workspace

The Guideline Workspace (accessible via sidebar) collects:
- Recommendation statements from GDG discussions
- Evidence profiles (GRADE certainty)
- Implementation notes

Use "Add to Guideline" button in answer view to save recommendations.

## Common Tasks

### Adding a new GDG expert
1. Edit `gdg/gdg_personas.py`
2. Add entry to `GDG_PERSONAS` dict with:
   - `role`, `specialty`, `perspective`
   - `search_queries`, `topics`, `specialty_keywords`
3. Add entry to `COGNITIVE_CONSTRAINTS` dict
4. Update `get_default_experts()` if needed

### Adding evidence extraction fields
1. Edit `core/data_extractor.py`
2. Update `EXTRACTION_PROMPT` with new fields
3. Add fields to `ExtractedEvidence` dataclass

## Domain Configuration System (v2)

The search system uses a pluggable domain configuration to ensure relevance for palliative surgery literature.

### DomainConfig Dataclass

```python
from config.domain_config import DomainConfig, get_domain_config, PALLIATIVE_SURGERY_CONFIG

# Get config by name
config = get_domain_config("palliative_surgery")

# Access vocabulary
union_filter = config.get_union_filter()       # OR-based PubMed filter
exclusion_filter = config.get_exclusion_filter() # NOT filter

# Keywords for scoring
high_relevance = config.high_relevance_keywords  # +0.08 each, max +0.4
procedure_kw = config.procedure_keywords          # +0.05 each, max +0.15
negative_kw = config.negative_keywords            # -0.12 each, max -0.35
```

### Domain Fields

| Field | Purpose |
|-------|---------|
| `verified_mesh` | Confirmed MeSH headings (e.g., "Palliative Care[MeSH]") |
| `candidate_mesh` | Unverified terms, used as [tiab] |
| `high_relevance_keywords` | Core domain terms (+0.08 boost each) |
| `procedure_keywords` | Procedure-specific terms (+0.05 each) |
| `outcome_keywords` | Outcome measures (+0.03 each) |
| `negative_keywords` | Off-topic indicators (-0.12 penalty each) |
| `llm_context` | Prompt context for LLM relevance scoring |

### Two-Pass Ranking

```python
from core.ranking import rank_citations_with_domain, RankingWeights

scored = rank_citations_with_domain(
    citations=citations,
    weights=RankingWeights(relevance=0.4, evidence=0.6),
    domain="palliative_surgery",
    original_query=query,
    use_llm_rerank=True,     # LLM scoring on top-N
    top_n_for_llm=50,        # Only top 50 get LLM scoring
    domain_weight=0.3,       # 30% domain, 70% relevance
    composite_threshold=0.3  # Filter below 0.3 composite
)

# Each ScoredCitation has:
# - domain_score: 0.0-1.0 domain relevance
# - matched_keywords: ["palliative", "symptom control", ...]
# - penalty_keywords: ["curative", "adjuvant", ...]
```

### SearchService Integration

```python
from services.search_service import SearchService

service = SearchService(openai_api_key=api_key)
result = service.execute_search(
    query="malignant bowel obstruction treatment",
    project_id=project_id,
    citation_dao=citation_dao,
    search_dao=search_dao,
    query_cache_dao=query_cache_dao,
    # v2 domain parameters
    domain="palliative_surgery",      # Domain name
    use_two_pass=True,                # Two-pass ranking
    domain_weight=0.3,                # Domain score weight
    use_union_filter=True,            # Apply union filter
    composite_threshold=0.3,          # Minimum score threshold
    top_n_for_llm=50                  # Top-N for LLM reranking
)
```

### Adding a New Domain

1. Create a new `DomainConfig` in `config/domain_config.py`:
```python
ONCOLOGY_CONFIG = DomainConfig(
    name="oncology",
    display_name="General Oncology",
    verified_mesh=["Neoplasms[MeSH]", ...],
    high_relevance_keywords=["cancer", "tumor", ...],
    # ... other fields
)
```

2. Register it in `DOMAIN_REGISTRY`:
```python
DOMAIN_REGISTRY: Dict[str, DomainConfig] = {
    "palliative_surgery": PALLIATIVE_SURGERY_CONFIG,
    "oncology": ONCOLOGY_CONFIG,  # New
}
```

### Running Domain Tests

```bash
# Run all 31 domain search tests
pytest tests/test_domain_search.py -v
```

## Implemented Features (P1-P6)

### P1: Citation Highlighting + Traceability

**File:** `ui/citation_utils.py`

Highlights inline citations with purple gradient badges and epistemic tags with color coding.

```python
from ui.citation_utils import format_expert_response, highlight_inline_citations

# Format expert response with all highlighting
formatted = format_expert_response(response_text)

# Citation patterns handled:
# - [PMID:12345678] → PubMed link with purple badge
# - [1], [2], [1-3] → Styled reference badges
# - [L1], [W1], [C1] → Literature/Web/Clinical trial badges

# Epistemic tag colors:
# - EVIDENCE → Green (#D1FAE5)
# - ASSUMPTION → Yellow (#FEF3C7)
# - OPINION → Blue (#DBEAFE)
# - EVIDENCE GAP → Red (#FEE2E2)
```

### P3: Red Team Challenger

**File:** `services/challenger_service.py`

Generates challenging questions to stress-test GDG recommendations.

```python
from services.challenger_service import ChallengerService

challenger = ChallengerService(google_api_key=api_key)
challenges = challenger.generate_challenges(
    recommendation="We recommend prophylactic fixation...",
    expert_responses={"Surgical Oncologist": {...}},
    conflicts=["Expert A disagrees with Expert B on..."],
    evidence_gaps=["No RCT data for Mirels 9"],
    max_questions=5
)

# Returns list of challenges with categories:
# - assumption, evidence, patient_selection, threshold, risk, feasibility
```

### P4: Quick Q&A Mode

**File:** `services/quick_answer_service.py`

Fast single-LLM response with top PubMed citations.

```python
from services.quick_answer_service import QuickAnswerService

qa_service = QuickAnswerService(
    openai_api_key=openai_key,
    google_api_key=google_key
)
result = qa_service.answer(
    question="What is the role of palliative surgery in MBO?",
    top_n_sources=5
)

# Returns QuickAnswer with:
# - answer: str (with [PMID:...] citations)
# - sources: List[Dict] (title, pmid, relevance)
# - confidence: float
```

### P5: Smart Suggestions

**File:** `ui/answer_view.py` (integrated)

Generates contextual follow-up questions based on response content.

Detection triggers:
- Safety concerns → "Deep dive on safety profile"
- Expert disagreement → "Resolve [topic] debate"
- Evidence gaps → "Find [missing] data"
- Conditional recommendations → "Clarify conditions"

### P6: Mark Pen (User Feedback)

**Files:** `services/feedback_service.py`, `ui/mark_pen.py`

Allows users to annotate expert responses with typed marks.

```python
from services.feedback_service import FeedbackService, MarkType

feedback = FeedbackService(project_id="my_project")

# Add a mark
feedback.add_mark(
    text="30% survival rate at 1 year",
    mark_type=MarkType.IMPORTANT_DATA,
    source="Surgical Oncologist",
    context={"question_id": "q1"}
)

# Mark types available:
# - IMPORTANT_DATA (📊)
# - KEY_FINDING (⭐)
# - EVIDENCE_GAP (🔍)
# - CITATION_USEFUL (📚)
# - DISAGREE (❌)
# - AGREE (✓)

# Retrieve marks
marks = feedback.get_marks(mark_type=MarkType.KEY_FINDING)
```

## Gotchas

- `Citation.authors` is a `List[str]`, not a string
- **Model names**: Use `gpt-5-mini` (OpenAI) and `gemini-3-pro-preview` (Google) - these are VALIDATED
- **OpenAI timeout**: Always set timeout=120.0 to prevent hanging
- **Thread safety**: Services use `threading.Lock()` for thread-safe operations
- PMIDs must be 7-8 digits (e.g., `PMID: 12345678`)
- GDG Chair does NOT review evidence - synthesis only
- **Domain scoring**: `domain_score` is 0-1 (0.5 neutral); `domain_relevance_boost` is deprecated
- **Cache keys**: Include domain settings in cache key to prevent stale results

## Expert Selection Feature (Porting from Virtual_Team)

### Overview

The expert selection feature provides a sophisticated UI for selecting GDG experts with:
- **Category-based organization**: Experts grouped by specialty area
- **Stage-aware presets**: Pre-configured expert combinations for common scenarios
- **Visual cards**: Color-coded cards with stance badges and metadata
- **Min/max constraints**: Enforce 2-12 expert selection

### Source Files (Virtual_Team)

| File | Purpose |
|------|---------|
| `ui/simulation/persona_selector.py` | Main selector UI component |
| `preclinical/advisory_personas.py` | Persona definitions and presets |

### Key Components

#### 1. AdvisoryPersona Dataclass

```python
@dataclass
class AdvisoryPersona:
    name: str                          # Display name
    role: str                          # Professional role
    affiliation: str                   # Institution
    expertise: List[str]               # Areas of expertise
    perspective: List[str]             # Approach/viewpoint
    speaking_style: str                # Communication style
    required_deliverables: List[str]   # Must-provide outputs
    deal_breakers: List[str]           # No-go triggers
    signature_questions: List[str]     # Questions they always ask
    default_stance: str                # Conservative | Moderate | Aggressive
    evidence_threshold: str            # RCT required | Phase 2 acceptable | Mechanistic OK
    quant_mode: str                    # numeric_required | measurable_required | qualitative_ok
```

#### 2. Category Organization

```python
# Categories for visual grouping
CATEGORY_COLORS = {
    "Core Clinical": {"border": "#2196F3", "bg": "#E3F2FD", "header": "#1565C0"},
    "Translational": {"border": "#9C27B0", "bg": "#F3E5F5", "header": "#7B1FA2"},
    "Regulatory & Access": {"border": "#4CAF50", "bg": "#E8F5E9", "header": "#2E7D32"},
    "Stakeholders": {"border": "#FF9800", "bg": "#FFF3E0", "header": "#EF6C00"},
    "Support": {"border": "#607D8B", "bg": "#ECEFF1", "header": "#455A64"},
}

# GDG-specific categories (to be adapted)
def get_personas_by_category() -> Dict[str, List[str]]:
    return {
        "Core Clinical": ["Surgical Oncologist", "Palliative Care Physician", ...],
        "Evidence & Methods": ["GRADE Methodologist", "Clinical Evidence Specialist"],
        "Patient Focus": ["Patient Advocate", "Pain Specialist"],
        ...
    }
```

#### 3. Board Presets (Stage-Aware)

```python
BOARD_PRESETS = {
    "Surgical Decision": {
        "personas": ["Surgical Oncologist", "Palliative Care Physician", ...],
        "focus": "Should this patient undergo palliative surgery?",
        "key_questions": [...]
    },
    "Symptom Management": {...},
    "Ethics Review": {...},
    ...
}
```

#### 4. render_persona_selector() Function

```python
def render_persona_selector(
    key_prefix: str = "persona",
    default_selection: Optional[List[str]] = None,
    max_personas: int = 12,
    min_personas: int = 2
) -> List[str]:
    """
    Render category-organized grid with preset buttons.
    Returns list of selected persona names.
    """
```

### Adaptation for Palliative Surgery GDG

The Virtual_Team version is designed for drug development advisory boards. For Palliative Surgery GDG:

1. **Rename concepts**: "Persona" → "Expert", "Advisory Board" → "GDG"
2. **Use existing GDG experts** from `gdg/gdg_personas.py`:
   - Surgical Oncologist
   - Perioperative Medicine Physician
   - Interventionalist
   - Palliative Care Physician
   - Patient Advocate
   - GRADE Methodologist
   - Clinical Evidence Specialist
   - Medical Ethicist
   - Pain and Symptom-Control Specialist
   - Geriatric and Frailty Specialist
   - Health Economist
   - GDG Chair

3. **Define GDG-specific presets**:
   - `Surgical Candidacy` - Assess if surgery appropriate
   - `Intervention Choice` - Surgery vs non-surgical options
   - `Symptom Management` - Pain/symptom control focus
   - `Ethics Review` - Ethical considerations
   - `Full Panel` - All 12 experts

4. **Maintain existing GDG_PERSONAS structure** in `gdg/gdg_personas.py`

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `ui/expert_selector.py` | CREATE | Persona selector adapted for GDG |
| `gdg/gdg_personas.py` | MODIFY | Add category mappings and presets |
| `ui/expert_panel.py` | MODIFY | Replace multiselect with selector |
| `ui/home.py` | MODIFY | Add selector to Ask the GDG tab |

### UI Integration Points

1. **GDG Discussion tab** (`ui/expert_panel.py`): Replace `st.multiselect()` with `render_expert_selector()`
2. **Ask the GDG tab** (`ui/home.py`): Add collapsible expert selection
3. **Question type detection**: Auto-select relevant experts based on question type

### Implementation Notes

- Use session state key prefix to avoid conflicts: `gdg_expert_`
- Persist selections across tab switches
- Default to scenario-appropriate preset
- Allow manual override of preset selections

---

## Web Search Integration (December 2024)

### Google Search Grounding (Recommended)

The app uses **Gemini's native Google Search grounding** for real-time web context, replacing Tavily.

**Configuration:**
```python
# config/settings.py
ENABLE_GOOGLE_SEARCH_GROUNDING = True
GOOGLE_SEARCH_GROUNDING_THRESHOLD = 0.3  # 0.0-1.0, lower = more grounding
```

**When triggered:**
- During synthesis in `research_partner_service.py` when generating recommendations
- During Pass 1 of expert consultation in `expert_service.py`

**Key files:**
| File | Purpose |
|------|---------|
| `integrations/google_search.py` | GoogleSearchClient with grounding |
| `integrations/tavily.py` | Legacy Tavily client (fallback) |
| `services/research_partner_service.py` | Synthesis with web context |
| `services/expert_service.py` | Expert consultation with grounding |

**Advantages over Tavily:**
- No extra API key (uses existing `GOOGLE_API_KEY`)
- Better medical/scientific coverage (Google Scholar, PubMed)
- Built-in source→text segment mapping
- Lower latency (search + reasoning in one call)

**Environment variables:**
```bash
GOOGLE_API_KEY=your_gemini_api_key
ENABLE_GOOGLE_SEARCH_GROUNDING=true
```

### Model Name Mapping (Native SDK Compatibility)

**Issue Fixed (December 2024):** The model name `gemini-3-pro-preview` works with Google's OpenAI-compatible endpoint but fails with the native `google-generativeai` SDK used for Google Search grounding.

**Solution:** Added model name mapping in `integrations/google_search.py`:

```python
# Model name mapping: OpenAI-compatible names → Native SDK names
MODEL_NAME_MAPPING = {
    "gemini-3-pro-preview": "gemini-2.0-flash",
    "gemini-3-pro": "gemini-2.0-flash",
    "gemini-3.0-pro-preview": "gemini-2.0-flash",
    "gemini-3.0-pro": "gemini-2.0-flash",
}
```

**Why this matters:**
- Expert calls use OpenAI-compatible endpoint → accepts `gemini-3-pro-preview`
- Grounding uses native SDK → requires valid SDK model names like `gemini-2.0-flash`
- The mapping ensures synthesis with grounding works correctly

**Symptom if broken:** "An unexpected error occurred. Please try again. (while researching question)" during the synthesis stage after experts respond successfully.
