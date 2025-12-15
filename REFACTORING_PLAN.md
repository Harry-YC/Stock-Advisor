# Refactoring Plan: Palliative Surgery GDG v2.0

**Created:** 2025-12-12
**Status:** Proposed
**Priority:** Medium (non-blocking, incremental)

---

## Executive Summary

The codebase is functional but has accumulated technical debt from rapid feature development. Key issues:
- Monolithic UI files (expert_panel.py: 1,832 LOC, literature_search.py: 1,100 LOC)
- Session state accessed directly throughout codebase
- Duplicate citation handling patterns
- Mixed concerns in utility modules

**Approach:** Incremental refactoring - improve as you touch files, don't block feature work.

---

## Phase 1: Quick Wins (1-2 days)

### 1.1 Create Shared Citation Utilities

**Problem:** Citation attribute extraction duplicated in 3+ locations.

**Current state:**
```python
# Pattern repeated in literature_search.py, expert_panel.py, gdg_utils.py
if hasattr(citation, 'pmid'):
    pmid = citation.pmid
else:
    pmid = citation.get('pmid')
```

**Solution:** Create `core/citation_utils.py`

```python
# core/citation_utils.py
from typing import Any, List, Optional
from dataclasses import dataclass

def get_citation_attr(citation: Any, attr: str, default: Any = None) -> Any:
    """Unified citation attribute getter - works with Citation objects or dicts."""
    if hasattr(citation, attr):
        return getattr(citation, attr, default)
    elif isinstance(citation, dict):
        return citation.get(attr, default)
    return default

def citation_to_dict(citation: Any) -> dict:
    """Convert Citation object or dict to normalized dict."""
    if isinstance(citation, dict):
        return citation
    return {
        'pmid': get_citation_attr(citation, 'pmid'),
        'title': get_citation_attr(citation, 'title', 'No title'),
        'abstract': get_citation_attr(citation, 'abstract', ''),
        'authors': get_citation_attr(citation, 'authors', []),
        'journal': get_citation_attr(citation, 'journal', ''),
        'year': get_citation_attr(citation, 'year', ''),
        'doi': get_citation_attr(citation, 'doi', ''),
    }

def format_authors(authors: List[str], max_authors: int = 3) -> str:
    """Format author list for display."""
    if not authors:
        return "Unknown"
    if len(authors) <= max_authors:
        return ", ".join(authors)
    return f"{', '.join(authors[:max_authors])}, et al. ({len(authors)} authors)"
```

**Files to update:**
- [ ] `core/utils.py` - move existing `get_citation_attr` here or create new file
- [ ] `ui/literature_search.py` - import from shared location
- [ ] `ui/expert_panel.py` - import from shared location
- [ ] `gdg/gdg_utils.py` - import from shared location

**Effort:** 2 hours

---

### 1.2 Centralize Session State Keys

**Problem:** Session state keys are string literals scattered throughout codebase.

**Current state:**
```python
# Repeated everywhere
st.session_state.get('search_results')
st.session_state.expert_discussion
st.session_state['current_project_id']
```

**Solution:** Create constants in `core/state_manager.py`

```python
# Add to core/state_manager.py

class StateKeys:
    """Centralized session state key constants."""
    # Project
    CURRENT_PROJECT_ID = 'current_project_id'
    CURRENT_PROJECT_NAME = 'current_project_name'
    SEARCH_RESULTS = 'search_results'
    SELECTED_PAPERS = 'selected_papers'

    # Discussion
    EXPERT_DISCUSSION = 'expert_discussion'
    DISCUSSION_ROUND = 'discussion_round'
    META_SYNTHESIS = 'meta_synthesis'

    # Evidence Corpus (v2.0)
    EVIDENCE_CORPUS = 'evidence_corpus'
    CURRENT_ETD = 'current_etd'

    # Chat
    EXPERT_CHAT_MESSAGES = 'expert_chat_messages'
    ACTIVE_CHAT_EXPERTS = 'active_chat_experts'


# Helper functions
def get_state(key: str, default: Any = None) -> Any:
    """Get session state value with default."""
    return st.session_state.get(key, default)

def set_state(key: str, value: Any) -> None:
    """Set session state value."""
    st.session_state[key] = value
```

**Effort:** 1 hour (define constants), then update files incrementally

---

### 1.3 Remove Backup Files

**Problem:** ~500KB of `.backup*` files in repository.

**Action:**
```bash
# Run from palliative_surgery2 directory
find . -name "*.backup*" -type f -delete
find . -name "*.bak" -type f -delete
```

**Effort:** 5 minutes

---

## Phase 2: UI Module Splitting (3-5 days)

### 2.1 Split expert_panel.py (1,832 LOC)

**Current structure:** One massive file handling:
- Expert selection UI
- Discussion execution
- Two-pass mode logic
- Debate mode
- Gap analysis display
- Conflict detection display
- Hypothesis tracking
- Human feedback loop
- Interactive Q&A chat
- Export functionality

**Proposed structure:**

```
ui/
├── expert_panel.py              # Main orchestrator (reduced to ~300 LOC)
├── expert_panel/
│   ├── __init__.py
│   ├── expert_selection.py      # Expert multiselect + scenario chips
│   ├── discussion_runner.py     # Run discussion button + progress
│   ├── two_pass_mode.py         # Two-pass specific UI
│   ├── debate_mode.py           # Collaborative debate UI
│   ├── analysis_display.py      # Gap analysis + conflict detection
│   ├── hypothesis_tracker.py    # Hypothesis extraction + display
│   ├── feedback_loop.py         # Human feedback UI
│   └── expert_chat.py           # Interactive Q&A
```

**Extraction order:**
1. [ ] `expert_chat.py` - Self-contained, lines 1768-1832
2. [ ] `feedback_loop.py` - Self-contained, lines 1687-1766
3. [ ] `hypothesis_tracker.py` - Lines 1624-1666
4. [ ] `debate_mode.py` - Lines 1076-1139
5. [ ] `analysis_display.py` - Gap + Conflict sections
6. [ ] `two_pass_mode.py` - Two-pass specific logic

**Effort:** 4 hours per component, 1-2 days total

---

### 2.2 Split literature_search.py (1,100 LOC)

**Proposed structure:**

```
ui/
├── literature_search.py         # Main orchestrator (reduced to ~200 LOC)
├── literature_search/
│   ├── __init__.py
│   ├── search_tabs.py           # Tab 1-4 definitions
│   ├── results_display.py       # Citation cards + table
│   ├── visualizations.py        # Timeline, network, table viz
│   ├── corpus_actions.py        # Include/exclude buttons (v2.0)
│   └── export_results.py        # CSV/Word export
```

**Effort:** 3 hours per component, 1 day total

---

## Phase 3: Service Layer Consolidation (2-3 days)

### 3.1 Create Discussion Orchestration Service

**Problem:** Discussion logic spread across `expert_panel.py` and `services/expert_service.py`.

**Solution:** Ensure `services/expert_service.py` handles ALL discussion logic.

```python
# services/expert_service.py should expose:
class ExpertDiscussionService:
    def run_discussion_round(...)      # Standard mode
    def run_two_pass_discussion(...)   # Two-pass mode
    def run_debate(...)                # Debate mode
    def regenerate_response(...)       # HITL regeneration
    def generate_follow_ups(...)       # Follow-up questions
```

**UI should only:**
- Collect user input
- Call service methods
- Display results

**Effort:** 4 hours

---

### 3.2 Create Analysis Orchestration Service

**Problem:** Analysis functions called directly from UI.

**Solution:** Consolidate in `services/analysis_service.py`

```python
# services/analysis_service.py should expose:
class AnalysisService:
    def analyze_gaps(responses, scenario) -> GapAnalysisResult
    def detect_conflicts(responses) -> ConflictResult
    def extract_hypotheses(responses, question) -> List[Hypothesis]
    def synthesize_responses(responses, question) -> SynthesisResult
```

**Effort:** 2 hours (may already exist, verify and consolidate)

---

## Phase 4: Configuration Cleanup (1 day)

### 4.1 Consolidate Settings Access

**Problem:** Mixed patterns for configuration access.

**Current state:**
```python
# Some files use:
from config import settings
settings.OPENAI_API_KEY

# Others use:
import os
os.getenv("OPENAI_API_KEY")
```

**Solution:** All config access through `config/settings.py`

**Files to audit:**
- [ ] `app_gl.py`
- [ ] `services/*.py`
- [ ] `gdg/*.py`
- [ ] `ui/*.py`

**Effort:** 2 hours

---

### 4.2 Create Feature Flags Module

**Problem:** Feature flags scattered and inconsistent.

**Solution:** Create `config/features.py`

```python
# config/features.py
from config import settings

class Features:
    """Feature flags for the application."""

    # Core features
    ADVANCED_TOOLS = settings.ENABLE_ADVANCED_TOOLS
    LOCAL_RAG = settings.ENABLE_LOCAL_RAG
    WEB_FALLBACK = settings.ENABLE_WEB_FALLBACK

    # v2.0 GRADE features
    EVIDENCE_CORPUS = True  # Always on in v2.0
    ETD_FRAMEWORK = True
    QUALITY_ASSESSMENT = True

    # Experimental
    DEEP_RESEARCH_MODE = True
    TWO_PASS_MODE = True

    @classmethod
    def is_enabled(cls, feature: str) -> bool:
        return getattr(cls, feature, False)
```

**Effort:** 1 hour

---

## Phase 5: Testing Infrastructure (2-3 days)

### 5.1 Add Unit Tests for Core Modules

**Priority test targets:**
1. [ ] `core/evidence_corpus.py` - Critical for GRADE methodology
2. [ ] `core/quality_assessment.py` - GRADE certainty calculations
3. [ ] `services/etd_service.py` - EtD framework logic
4. [ ] `gdg/response_validator.py` - Citation validation

**Test structure:**
```
tests/
├── unit/
│   ├── test_evidence_corpus.py
│   ├── test_quality_assessment.py
│   ├── test_etd_service.py
│   └── test_response_validator.py
├── integration/
│   └── test_discussion_flow.py
└── e2e/
    └── test_playwright.py  # Existing
```

**Effort:** 4 hours per module

---

### 5.2 Add Integration Tests

**Key flows to test:**
1. Search → Select papers → Include in corpus → Run discussion
2. Discussion → Generate EtD → Generate recommendation
3. Two-pass mode end-to-end

**Effort:** 1 day

---

## Implementation Schedule

| Week | Phase | Tasks | Effort |
|------|-------|-------|--------|
| 1 | Phase 1 | Quick wins (citation utils, state keys, cleanup) | 4 hours |
| 1-2 | Phase 2.1 | Split expert_panel.py | 8 hours |
| 2 | Phase 2.2 | Split literature_search.py | 4 hours |
| 2-3 | Phase 3 | Service layer consolidation | 6 hours |
| 3 | Phase 4 | Configuration cleanup | 3 hours |
| 3-4 | Phase 5 | Testing infrastructure | 12 hours |

**Total estimated effort:** 37 hours (~1 week of focused work, or 2-3 weeks incremental)

---

## Refactoring Principles

1. **Don't break working code** - Refactor incrementally, test after each change
2. **Follow existing patterns** - New v2.0 modules are well-structured, follow them
3. **Extract, don't rewrite** - Move code to new locations, don't reimagine it
4. **One concern per file** - Each module should have a single responsibility
5. **UI calls services** - UI modules should only handle display, services handle logic

---

## Success Criteria

- [ ] No file exceeds 500 LOC (except data definitions)
- [ ] All session state access through StateKeys constants
- [ ] Citation handling uses shared utilities
- [ ] UI modules only handle display logic
- [ ] Unit test coverage for core modules > 70%
- [ ] No duplicate code patterns > 10 lines

---

## Not In Scope (Defer)

- Database schema changes
- API redesign
- Major feature changes
- Performance optimization
- UI/UX redesign

---

## Notes

- Refactoring should not block feature development
- When touching a file for features, apply relevant refactoring
- New code should follow refactored patterns immediately
- Mark completed items in this document as you go
