"""
Palliative Surgery GDG - Main Application

A guideline development platform for palliative surgery clinical questions.

Core Workflow:
1. Search for evidence (PubMed + preprints)
2. Visualize results (timeline, citation network, sortable tables)
3. Consult with AI-powered GDG expert panel (12 experts)
4. Generate evidence-based recommendations with epistemic tagging
5. Build clinical guidelines in the Guideline Workspace
"""

import streamlit as st
import pandas as pd
import json
import hashlib
from datetime import datetime
from pathlib import Path

# UI Setup (must be first Streamlit command)
st.set_page_config(
    page_title="Palliative Surgery GDG",
    page_icon="🏥",
    layout="wide"
)

# Password Protection
import os

def check_password():
    """Returns `True` if the user has entered the correct password."""
    # Skip password on local development (Railway sets RAILWAY_ENVIRONMENT)
    if not os.environ.get("RAILWAY_ENVIRONMENT"):
        return True

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        entered = st.session_state.get("password", "")
        correct_password = os.environ.get("APP_PASSWORD")
        # Require APP_PASSWORD env var in production
        if not correct_password:
            st.session_state["password_correct"] = False
            return
        if entered == correct_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    # Show password input
    st.markdown("### 🔐 Palliative Surgery GDG")
    st.text_input(
        "Password", type="password", key="password", on_change=password_entered
    )
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Incorrect password")
    return False

if not check_password():
    st.stop()

# Load custom CSS
def load_css():
    """Load custom CSS styles from assets directory."""
    css_file = Path(__file__).parent / "assets" / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Import local modules
from config import settings
from core.pubmed_client import PubMedClient, Citation
from core.database import (
    DatabaseManager,
    ProjectDAO,
    CitationDAO,
    ScreeningDAO,
    SearchHistoryDAO,
    QueryCacheDAO,
    TagDAO,
    AIScreeningDAO,
    ExpertDiscussionDAO,
    CdpDAO,
    SearchContextDAO
)
from core.query_parser import AdaptiveQueryParser
# CitationExporter and EXPORT_FORMATS moved to ui/sidebar.py
from core.zotero_client import ZoteroClient
from core.validators import validate_identifiers, parse_identifier_input
from core.ranking import rank_citations, RankingWeights, RANKING_PRESETS, local_semantic_relevance, SEMANTIC_SEARCH_AVAILABLE
from core.ai_screener import screen_papers_batch, get_screening_summary
from core.priors_manager import PriorsManager
# Knowledge store functions moved to ui/sidebar.py
from core.knowledge_store import KnowledgeStore
from core.knowledge_extractor import process_discussion_for_knowledge, extract_entities_from_question

# Import UI Modules
from ui.literature_search import render_literature_search
from ui.ai_screening import render_ai_screening
from ui.expert_panel import render_expert_panel
from ui.sidebar import render_sidebar
from ui.home import render_home  # Conversational Mode (Tab 1)
from ui.cdp_workspace import render_cdp_workspace  # Guideline Workspace Mode

# Import centralized state manager
from core.state_manager import init_all_session_state, reset_project_state

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

from core.utils import get_citation_attr, extract_simple_query



# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

@st.cache_resource
def get_database():
    """Initialize database connection (cached for app lifecycle)"""
    db_path = settings.OUTPUTS_DIR / "literature_review.db"
    return DatabaseManager(db_path)

@st.cache_resource
def get_pubmed_client():
    """Initialize PubMed client (cached)"""
    return PubMedClient(email=settings.PUBMED_EMAIL, api_key=settings.PUBMED_API_KEY)

@st.cache_resource
def get_priors_manager():
    """Initialize PriorsManager (cached)"""
    try:
        return PriorsManager()
    except Exception:
        return None

# Initialize database and DAOs
db = get_database()
project_dao = ProjectDAO(db)
citation_dao = CitationDAO(db)
screening_dao = ScreeningDAO(db)
search_dao = SearchHistoryDAO(db)
query_cache_dao = QueryCacheDAO(db)
tag_dao = TagDAO(db)
ai_screening_dao = AIScreeningDAO(db)
expert_discussion_dao = ExpertDiscussionDAO(db)
cdp_dao = CdpDAO(db)
search_context_dao = SearchContextDAO(db)

# Initialize cached resources
pubmed_client = get_pubmed_client()
priors_manager = get_priors_manager()


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

# Initialize all session state from centralized state manager
# Initialize all session state from centralized state manager
init_all_session_state()

# Inject DAOs into session state for service access (Dependency Injection)
st.session_state.citation_dao = citation_dao
st.session_state.search_dao = search_dao
st.session_state.query_cache_dao = query_cache_dao
st.session_state.expert_discussion_dao = expert_discussion_dao
st.session_state.project_dao = project_dao
st.session_state.cdp_dao = cdp_dao
st.session_state.search_context_dao = search_context_dao


# =============================================================================
# SIDEBAR - PROJECT MANAGER & NAVIGATION
# =============================================================================

# Render sidebar using extracted module
render_sidebar(
    project_dao=project_dao,
    citation_dao=citation_dao,
    screening_dao=screening_dao,
    zotero_client_class=ZoteroClient
)


# =============================================================================
# MAIN CONTENT - SINGLE PAGE (Like app_sab.py conversational mode)
# =============================================================================

# Get current UI mode
ui_mode = st.session_state.get('ui_mode', 'conversational')

# Handle Guideline Workspace mode separately (full-page takeover)
if ui_mode == 'cdp_workspace':
    render_cdp_workspace()
    st.stop()

# =============================================================================
# SINGLE PAGE INTERFACE - Ask the GDG
# =============================================================================

# Render the single-page home interface
render_home()

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption("Palliative Surgery GDG v1.0 | Built with Streamlit")
