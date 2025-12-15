"""
Configuration settings for Palliative Surgery GDG App

Centralizes all app configuration including:
- API keys and credentials
- App behavior settings
- UI configuration
- File paths
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

VERSION = "1.0"
ENV = os.getenv("APP_ENV", "dev")

# =============================================================================
# PATHS
# =============================================================================

APP_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = APP_ROOT / "outputs"
SESSIONS_DIR = OUTPUTS_DIR / "sessions"
EXPORTS_DIR = OUTPUTS_DIR / "exports"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure directories exist
for dir_path in [OUTPUTS_DIR, SESSIONS_DIR, EXPORTS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# API CONFIGURATION
# =============================================================================

# PubMed E-utilities
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "user@example.com")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")  # Optional, increases rate limit

# OpenAI (for AI features)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "120"))  # API timeout in seconds (120s for complex LLM generation)

# Google Gemini (OpenAI-compatible endpoint)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

# Crossref (for retraction checking - no key required)
CROSSREF_USER_AGENT = os.getenv("CROSSREF_USER_AGENT", "LiteratureReviewApp/1.0 (mailto:user@example.com)")

# Zotero (for reference manager integration)
ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")

# =============================================================================
# APP BEHAVIOR
# =============================================================================

# Search settings
DEFAULT_MAX_RESULTS = 100
SEARCH_TIMEOUT_SECONDS = 30

# Active learning settings
ACTIVE_LEARNING_MIN_TRAINING = 10  # Min papers to train model
ACTIVE_LEARNING_CERTAINTY_THRESHOLD = 0.95  # When to suggest stopping

# Session management
AUTO_SAVE_INTERVAL_MINUTES = 5
MAX_SESSION_AGE_DAYS = 90

# =============================================================================
# UI CONFIGURATION
# =============================================================================

APP_TITLE = "Palliative Surgery GDG"
APP_ICON = "🏥"
SIDEBAR_STATE = "expanded"  # or "collapsed"

# Theme colors (for custom styling)
PRIMARY_COLOR = "#1F77B4"  # Blue
SECONDARY_COLOR = "#FF7F0E"  # Orange
SUCCESS_COLOR = "#2CA02C"  # Green
WARNING_COLOR = "#FFD700"  # Gold
ERROR_COLOR = "#D62728"  # Red

# Page configuration
PAGE_LAYOUT = "wide"  # or "centered"
INITIAL_SIDEBAR_STATE = "expanded"

# =============================================================================
# EXPERT PANEL CONFIGURATION
# =============================================================================

# Expert panel settings
EXPERT_MAX_CITATIONS = int(os.getenv("EXPERT_MAX_CITATIONS", "10"))
EXPERT_MODEL = os.getenv("EXPERT_MODEL", "gemini-3-pro-preview")
# Reasoning Model (High-computation tasks: Critique, Strategy, Hypothesis)
REASONING_MODEL = os.getenv("REASONING_MODEL", "gemini-3-pro-preview") 

EXPERT_MAX_TOKENS = int(os.getenv("EXPERT_MAX_TOKENS", "6000"))

# Priors configuration (canonical frameworks for GDG)
PRIORS_CONFIG_PATH = APP_ROOT / "config" / "priors" / "gdg_priors.yaml"

# AI Screening settings
SCREENING_CONFIDENCE_THRESHOLD = int(os.getenv("SCREENING_CONFIDENCE_THRESHOLD", "80"))
SCREENING_MODEL = os.getenv("SCREENING_MODEL", "gpt-5-mini")

# Semantic search settings
SEMANTIC_MODEL = os.getenv("SEMANTIC_MODEL", "dmis-lab/biobert-base-cased-v1.2")
ENABLE_SEMANTIC_SEARCH = True

# =============================================================================
# LOCAL RAG CONFIGURATION
# =============================================================================

# Vector Database
QDRANT_STORAGE_PATH = APP_ROOT / "data" / "vector_storage"
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "literature_review_v1")
QDRANT_URL = os.getenv("QDRANT_URL")  # Optional: Qdrant Cloud endpoint
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Optional: Qdrant Cloud API key
VECTOR_SIZE = 768  # nomic-embed-text-v1.5 dimension

# Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")  # auto, cpu, cuda, mps
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# Chunking
CHUNK_PARENT_SIZE = int(os.getenv("CHUNK_PARENT_SIZE", "2000"))
CHUNK_CHILD_SIZE = int(os.getenv("CHUNK_CHILD_SIZE", "200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
USE_RERANKING = os.getenv("USE_RERANKING", "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.5"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))

# Advanced Retrieval Features
ENABLE_HYDE = os.getenv("ENABLE_HYDE", "true").lower() == "true"  # Hypothetical Document Embeddings
ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"  # Query expansion for better recall

# Web Search Fallback (optional)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
ENABLE_WEB_FALLBACK = bool(TAVILY_API_KEY)

# Google Search Grounding (Native)
ENABLE_GOOGLE_SEARCH_GROUNDING = True
GOOGLE_SEARCH_GROUNDING_THRESHOLD = 0.3  # Dynamic retrieval threshold (0.0-1.0)

# Ensure vector storage directory exists
QDRANT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FEATURE FLAGS
# =============================================================================

# Feature gating - set to True to enable advanced/drug-dev specific features
# When False, the UI is simplified for guideline development only
ENABLE_ADVANCED_TOOLS = os.getenv("ENABLE_ADVANCED_TOOLS", "false").lower() == "true"

ENABLE_AI_FEATURES = bool(OPENAI_API_KEY)  # GPT-powered features
ENABLE_ZOTERO = bool(ZOTERO_API_KEY and ZOTERO_USER_ID)  # Zotero integration
ENABLE_RETRACTION_CHECK = True  # Retraction checking via Crossref
ENABLE_ACTIVE_LEARNING = True  # ML-powered screening
ENABLE_EXPERT_PANEL = bool(OPENAI_API_KEY)  # Expert panel discussions
ENABLE_AI_SCREENING = bool(OPENAI_API_KEY)  # AI paper screening
ENABLE_PRIORS = PRIORS_CONFIG_PATH.exists()  # Prior knowledge frameworks
ENABLE_LOCAL_RAG = True  # Local document RAG pipeline
ENABLE_CITATION_VERIFICATION = True  # Post-hoc citation verification

# =============================================================================
# CITATION FORMATS
# =============================================================================

SUPPORTED_EXPORT_FORMATS = [
    "RIS",      # Reference Manager
    "BibTeX",   # LaTeX
    "EndNote",  # EndNote XML
    "CSV",      # Spreadsheet
    "JSON",     # Programmatic access
    "Markdown", # Plain text
]

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate configuration and warn about missing keys"""
    warnings = []

    if not OPENAI_API_KEY:
        warnings.append("⚠️ OPENAI_API_KEY not set - AI features will be disabled")

    if not PUBMED_API_KEY:
        warnings.append("⚠️ PUBMED_API_KEY not set - using lower rate limit (3 req/sec instead of 10)")

    if not ZOTERO_API_KEY or not ZOTERO_USER_ID:
        warnings.append("⚠️ Zotero credentials not set - Zotero integration disabled")

    return warnings

if __name__ == "__main__":
    print("Palliative Surgery GDG App - Configuration")
    print("=" * 70)
    print(f"\nApp Root: {APP_ROOT}")
    print(f"Outputs: {OUTPUTS_DIR}")
    print(f"\nFeature Flags:")
    print(f"  AI Features: {ENABLE_AI_FEATURES}")
    print(f"  Zotero: {ENABLE_ZOTERO}")
    print(f"  Retraction Check: {ENABLE_RETRACTION_CHECK}")
    print(f"  Active Learning: {ENABLE_ACTIVE_LEARNING}")

    warnings = validate_config()
    if warnings:
        print(f"\nWarnings:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("\n✓ All optional features configured")
