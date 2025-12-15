# Architecture - Literature Review Platform

## Overview

The Literature Review Platform is a Streamlit-based web application for AI-powered literature review in drug development. It features PubMed search, AI screening, expert panel discussions, and knowledge extraction.

## Directory Structure

```
literature_review/
├── app_lr.py                    # Main entry point (~400 lines)
├── config/
│   └── settings.py              # Configuration and environment variables
├── core/                        # Business logic (no UI dependencies)
│   ├── pubmed_client.py         # PubMed API client
│   ├── database.py              # SQLite DAOs
│   ├── ai_screener.py           # AI paper screening
│   ├── ranking.py               # Citation ranking
│   ├── query_parser.py          # AI query optimization
│   ├── state_manager.py         # Session state management
│   ├── knowledge_store.py       # Persistent knowledge
│   ├── working_memory.py        # Working memory per project
│   ├── priors_manager.py        # Prior knowledge frameworks
│   ├── citation_network.py      # NetworkX citation graphs
│   ├── data_extractor.py        # GPT evidence extraction
│   └── analysis/
│       ├── gap_analyzer.py      # Discussion gap analysis
│       └── conflict_detector.py # Expert conflict detection
├── services/                    # Business logic layer
│   ├── expert_service.py        # Expert discussion orchestration
│   ├── analysis_service.py      # Gap/conflict/synthesis
│   └── chat_service.py          # Chat context and streaming
├── preclinical/                 # Drug development domain
│   ├── expert_personas.py       # Expert definitions
│   └── expert_utils.py          # LLM calling functions
├── integrations/                # External APIs
│   ├── semantic_scholar.py      # Citation data
│   ├── clinicaltrials.py        # ClinicalTrials.gov
│   ├── open_targets.py          # Target-disease
│   ├── chembl.py                # Compound bioactivity
│   └── biorxiv.py               # Preprints
├── ui/                          # Streamlit components
│   ├── sidebar.py               # Project manager, export
│   ├── literature_search.py     # Search interface
│   ├── expert_panel.py          # Expert discussions
│   ├── expert_chat.py           # Q&A chat
│   ├── ai_screening.py          # Screening interface
│   └── context_utils.py         # Shared context utilities
├── tests/                       # Playwright tests
│   ├── conftest.py              # Test fixtures
│   ├── test_tabs_structure.py   # Tab navigation tests
│   └── test_comprehensive_ui.py # Full UI tests
├── docs/                        # Documentation
│   ├── API_REFERENCE.md         # API documentation
│   └── ARCHITECTURE.md          # This file
└── outputs/                     # Generated files
    ├── literature_review.db     # SQLite database
    └── qdrant/                  # Vector DB storage
```

## Architectural Layers

### 1. Entry Point (`app_lr.py`)
- Initializes Streamlit page config
- Creates database connections (cached)
- Initializes session state
- Renders sidebar and main tabs
- ~400 lines (down from 739)

### 2. Core Layer (`core/`)
- Pure Python business logic
- No Streamlit dependencies
- Testable in isolation
- Key modules:
  - `pubmed_client.py`: PubMed API wrapper
  - `database.py`: SQLite DAOs
  - `state_manager.py`: Session state definitions
  - `ranking.py`: Clinical utility scoring

### 3. Services Layer (`services/`)
- Orchestrates core modules
- No UI dependencies
- Handles complex workflows:
  - `expert_service.py`: Multi-round discussions
  - `analysis_service.py`: Gap/conflict analysis
  - `chat_service.py`: Context building

### 4. UI Layer (`ui/`)
- Streamlit components only
- Delegates to services
- Renders user interface
- Handles user interactions

### 5. Domain Layer (`preclinical/`)
- Drug development specifics
- Expert persona definitions
- Scenario templates
- Evidence tagging formats

## Data Flow

```
User Input
    ↓
UI Layer (ui/*.py)
    ↓
Services Layer (services/*.py)
    ↓
Core Layer (core/*.py)
    ↓
External APIs / Database
    ↓
    ↑
Results flow back up
```

## Session State Management

All session state is centralized in `core/state_manager.py`:

```python
# Project State
- current_project_id: int
- current_project_name: str
- search_results: dict
- selected_papers: set

# Discussion State
- expert_discussion: dict  # Round -> Expert -> Response
- discussion_round: int
- gap_analysis: dict
- conflict_analysis: dict
- meta_synthesis: dict
- tracked_hypotheses: list

# Chat State
- expert_chat_messages: list
- active_chat_experts: list

# HITL State
- injected_evidence: list
- expert_temperatures: dict
- working_memory: WorkingMemory
- human_feedback: list

# RAG State
- rag_context: list
- indexed_documents: list
- uploaded_documents: list
```

## Database Schema

### Projects Table
```sql
CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### Citations Table
```sql
CREATE TABLE citations (
    pmid TEXT PRIMARY KEY,
    title TEXT,
    authors TEXT,  -- JSON array
    journal TEXT,
    year INTEGER,
    abstract TEXT,
    doi TEXT,
    publication_types TEXT,  -- JSON array
    keywords TEXT,  -- JSON array
    fetched_at TIMESTAMP
);
```

### Project Citations (Junction)
```sql
CREATE TABLE project_citations (
    project_id INTEGER REFERENCES projects(id),
    pmid TEXT REFERENCES citations(pmid),
    PRIMARY KEY (project_id, pmid)
);
```

### Screening Decisions
```sql
CREATE TABLE screening_decisions (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    pmid TEXT REFERENCES citations(pmid),
    decision TEXT,  -- 'include', 'exclude', 'review'
    confidence REAL,
    reasoning TEXT,
    is_override BOOLEAN,
    created_at TIMESTAMP
);
```

### Expert Discussions
```sql
CREATE TABLE expert_discussions (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    clinical_question TEXT,
    scenario TEXT,
    experts TEXT,  -- JSON array
    created_at TIMESTAMP
);

CREATE TABLE expert_discussion_entries (
    id INTEGER PRIMARY KEY,
    discussion_id INTEGER REFERENCES expert_discussions(id),
    round_num INTEGER,
    expert_name TEXT,
    content TEXT,
    raw_response TEXT,  -- JSON
    created_at TIMESTAMP
);
```

## Expert Personas

8+ drug development experts defined in `preclinical/expert_personas.py`:

| Expert | Role |
|--------|------|
| Bioscience Lead | Target biology, mechanism of action |
| DMPK Scientist | Pharmacokinetics, drug metabolism |
| Toxicology Expert | Safety assessment, risk evaluation |
| Clinical Pharmacologist | Human PK/PD, dose selection |
| Medical Reviewer | Clinical trial design, endpoints |
| Regulatory Expert | FDA/EMA requirements, submissions |
| Biomarker Scientist | Translational biomarkers |
| Commercial Lead | Market analysis, competitive landscape |

## Key Design Patterns

### 1. DAO Pattern
All database access through Data Access Objects:
```python
class CitationDAO:
    def upsert_citation(self, citation: dict) -> None
    def get_citations_batch(self, pmids: List[str]) -> Dict[str, dict]
```

### 2. Service Layer Pattern
Business logic in services, not UI:
```python
class ExpertDiscussionService:
    def run_discussion_round(...) -> DiscussionRoundResult
```

### 3. Callback Pattern
Progress updates without UI coupling:
```python
def progress_callback(expert_name: str, current: int, total: int):
    status_text.text(f"Consulting {expert_name}...")
    progress_bar.progress(current / total)
```

### 4. Cached Resources
Expensive resources cached for app lifecycle:
```python
@st.cache_resource
def get_database():
    return DatabaseManager(db_path)
```

## Performance Considerations

1. **Citation Caching**: Previously fetched citations cached in DB
2. **Query Caching**: Optimized queries cached by hash
3. **Batch Operations**: Citations fetched in batches
4. **Streaming Responses**: Expert responses streamed to UI
5. **Parallel Processing**: AI screening uses ThreadPoolExecutor

## Security Notes

1. **API Keys**: Stored in environment variables, never in code
2. **Input Validation**: Question length limits, character restrictions
3. **SQL Injection**: All queries use parameterized statements
4. **XSS Prevention**: Streamlit handles HTML escaping
5. **Rate Limiting**: PubMed rate limits respected

## Testing Strategy

### Playwright E2E Tests
- Tab navigation
- Component visibility
- User workflows
- Error handling

### Unit Tests (Future)
- Service layer methods
- Core module functions
- DAO operations

## Deployment

### Local Development
```bash
streamlit run app_lr.py
```

### Environment Variables
```bash
PUBMED_API_KEY=your_key
PUBMED_EMAIL=your@email.com
OPENAI_API_KEY=sk-...
EXPERT_MODEL=gpt-5-mini
```

### Dependencies
- Python 3.9+
- Streamlit
- OpenAI SDK
- Playwright (testing)
- pandas, plotly, networkx (visualization)
