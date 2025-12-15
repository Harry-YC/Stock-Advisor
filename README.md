# Research Partner

A research platform designed for drug development scientists. Optimized for the **search → expert discussion** workflow with support for large datasets (200-1000 papers).

## Features

### Core Search & Export
- **PubMed Literature Search** - Boolean operators, MeSH terms, date filtering, rate limiting
- **Preprint Search** - bioRxiv/medRxiv integration for early access to research
- **Multi-Format Export** - CSV, JSON, RIS, BibTeX, EndNote XML
- **Project Management** - SQLite-based persistence with session management

### Drug Development Intelligence
- **Citation Network Analysis** - Visualize paper relationships with Plotly/NetworkX
  - Identify hub papers and research clusters
  - PageRank and betweenness centrality metrics
  - Timeline view of citation propagation
- **ClinicalTrials.gov Integration** - Search and track competitor trials
  - Filter by phase, status, sponsor
  - Competitive landscape dashboards
- **Drug Knowledge Graph** - Open Targets + ChEMBL integration
  - Target-disease associations
  - Compound-target bioactivity data
  - ADMET properties

### Evidence Synthesis
- **Automated Evidence Tables** - GPT-powered extraction from abstracts
  - Study design, sample size, endpoints
  - PK parameters (Cmax, AUC, t1/2, clearance)
  - Efficacy results with p-values and CIs
  - Safety/adverse events
  - Export to Excel with multiple sheets
- **AI Expert Panel** - 8+ drug development personas
  - Bioscience Lead, DMPK Scientist, Toxicology Expert, etc.
  - Multi-round discussions with knowledge persistence
- **Expert Debate Mode** - Structured multi-round debates
  - Cross-examination and rebuttals
  - Consensus tracking with dissent documentation

### Reference Management
- **Zotero Integration** - Bidirectional sync
  - Import from Zotero collections
  - Fetch PDF annotations and notes
  - Tag synchronization

## Quick Start

### Installation

```bash
cd Research_Partner
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```bash
# Required for PubMed (increases rate limit)
PUBMED_EMAIL=your.email@domain.com
PUBMED_API_KEY=your_api_key  # Get from https://www.ncbi.nlm.nih.gov/account/settings/

# Required for AI features
OPENAI_API_KEY=sk-...

# Optional - Zotero
ZOTERO_API_KEY=your_zotero_key
ZOTERO_USER_ID=your_user_id
```

### Run the App

```bash
streamlit run app_gl.py
```

## Project Structure

```
Research_Partner/
├── app_gl.py                    # Main Streamlit application
├── requirements.txt             # Python dependencies
│
├── config/
│   └── settings.py              # Configuration and feature flags
│
├── core/
│   ├── pubmed_client.py         # PubMed API client
│   ├── citation_network.py      # Citation network analysis
│   ├── data_extractor.py        # Evidence table extraction
│   ├── knowledge_store.py       # Persistent knowledge graph
│   ├── ranking.py               # Evidence-based ranking
│   └── database.py              # SQLite data layer
│
├── integrations/
│   ├── semantic_scholar.py      # Citation data API
│   ├── clinicaltrials.py        # ClinicalTrials.gov API
│   ├── open_targets.py          # Target-disease associations
│   ├── chembl.py                # Compound bioactivity data
│   └── biorxiv.py               # Preprint search
│
├── preclinical/
│   ├── expert_personas.py       # Expert definitions + debate mode
│   └── expert_utils.py          # Expert calling functions
│
├── ui/
│   ├── literature_search.py     # Search interface
│   ├── expert_panel.py          # Expert discussion UI
│   └── ai_screening.py          # AI screening interface
│
└── outputs/
    ├── sessions/                # Saved project sessions
    ├── exports/                 # Exported data files
    └── reports/                 # Generated reports
```

## New Integrations

### Citation Network
```python
from core.citation_network import CitationNetworkBuilder, create_network_visualization

builder = CitationNetworkBuilder()
network = builder.build_network(pmids=["12345678", "87654321"])
fig = create_network_visualization(network, color_by="cluster")
```

### ClinicalTrials.gov
```python
from integrations.clinicaltrials import ClinicalTrialsClient, get_competitive_landscape

client = ClinicalTrialsClient()
trials = client.search_by_target("KRAS", active_only=True)
landscape = get_competitive_landscape("non-small cell lung cancer")
```

### Evidence Tables
```python
from core.data_extractor import EvidenceExtractor, export_evidence_to_excel

extractor = EvidenceExtractor()
evidence = extractor.batch_extract(papers)
export_evidence_to_excel(evidence, "evidence_table.xlsx")
```

### Expert Debate Mode
```python
from preclinical.expert_personas import get_debate_prompts, analyze_debate_consensus

prompts = get_debate_prompts(
    persona_name="DMPK Scientist",
    clinical_question="Is compound X suitable for oral dosing?",
    debate_topic="Bioavailability concerns",
    round_num=2,
    previous_responses=round_1_responses
)
```

## Configuration

### AI Model

This application uses **Gemini 3.0 Pro** (Google, via OpenAI Adapter) for all AI-powered features:
- Evidence extraction from abstracts
- AI-assisted paper screening
- Expert panel discussions
- Query optimization
- Conflict detection and gap analysis

The model is configured in `config/settings.py`:

```python
OPENAI_MODEL = "gemini-3-pro-preview"      # Default for all tasks
EXPERT_MODEL = "gemini-3-pro-preview"      # Expert panel discussions
SCREENING_MODEL = "gemini-3-pro-preview"   # AI screening
```

Override via environment variables if needed:
```bash
export OPENAI_MODEL=gpt-5
export EXPERT_MODEL=gpt-5
```

### Feature Flags (config/settings.py)

| Flag | Description |
|------|-------------|
| `ENABLE_AI_FEATURES` | GPT-powered features (requires OPENAI_API_KEY) |
| `ENABLE_ZOTERO` | Zotero integration |
| `ENABLE_EXPERT_PANEL` | AI expert discussions |
| `ENABLE_SEMANTIC_SEARCH` | BioBERT-based ranking |

## Dependencies

Core dependencies:
- `streamlit` - UI framework
- `requests` - API clients
- `networkx` - Citation network analysis
- `plotly` - Interactive visualizations
- `pandas` - Data manipulation
- `openai` - GPT integration

Optional:
- `sentence-transformers` - Semantic search
- `openpyxl` - Excel export

## Testing

This project includes a comprehensive test suite using `pytest` and `playwright`.

### 1. Run Unit & Integration Tests
```bash
pytest tests/test_rag_integration.py
```

### 2. Run Automated RAG Evaluation (DeepMind Quality)
Requires `OPENAI_API_KEY`.
```bash
pytest tests/evaluation/test_rag_quality.py
```

### 3. Run UI E2E Tests (Playwright)
To test the full application flow including the Agentic Chat:
```bash
# Install browses first: playwright install
pytest tests/test_comprehensive_ui.py tests/test_agentic_chat.py
```

## References

- **PubMed E-utilities**: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- **Semantic Scholar API**: https://api.semanticscholar.org/
- **ClinicalTrials.gov API**: https://clinicaltrials.gov/data-api/api
- **Open Targets Platform**: https://platform.opentargets.org/
- **ChEMBL**: https://www.ebi.ac.uk/chembl/

---

## Changelog

### v2.1 (December 2025)
- **Upgraded to Gemini 3 Pro** - All AI features now use Google's Gemini 3 Pro model for improved accuracy and reasoning
- Unified model configuration across all modules
- Updated documentation

### v2.0 (December 2024)
- Initial release with GPT-4o-mini support
- Expert panel with 8+ drug development personas
- Citation network analysis
- Evidence table extraction

---

**Version**: 2.1
**Last Updated**: December 2025
