"""
Literature Search UI Module

Handles the literature search interface, including query-based search,
identifier search (PMID/DOI), and file import.
"""

import streamlit as st
import hashlib
from datetime import datetime
import re

from config import settings
from core.pubmed_client import PubMedClient, Citation
from core.validators import validate_identifiers
from core.query_parser import AdaptiveQueryParser
from core.ranking import rank_citations, RANKING_PRESETS

# Visualization imports
try:
    import plotly.express as px
    import pandas as pd
    from core.citation_network import CitationNetworkBuilder, create_network_visualization, create_timeline_visualization
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Import DAOs from main app context or pass them in
# Ideally, we should import them here if they are singletons or pass them as args
# For now, we'll assume they are available via st.session_state or we re-instantiate/import
# To keep it clean, let's import the classes and instantiate if needed, or rely on the main app to pass them.
# But `render_literature_search` usually takes dependencies.
# However, the current app structure uses global `db` and DAOs.
# We will import the DAO classes and use `st.cache_resource` or similar if we want to be independent,
# OR we can just import the global instances if we move them to a shared module.
# Given the current structure, let's import the classes and use the `get_database` pattern if possible,
# or simply accept them as arguments.
# To minimize refactoring friction, let's accept them as arguments or import the getter functions.

from core.database import (
    DatabaseManager,
    ProjectDAO,
    CitationDAO,
    SearchHistoryDAO,
    QueryCacheDAO,
    SearchContextDAO,
    PaperSignalDAO
)
from core.utils import get_citation_attr, extract_simple_query

from services.search_service import SearchService

import logging
logger = logging.getLogger(__name__)

# Evidence corpus imports for inclusion/exclusion visibility
try:
    from core.evidence_corpus import get_corpus_from_session, EvidenceCorpus
    EVIDENCE_CORPUS_AVAILABLE = True
except ImportError:
    EVIDENCE_CORPUS_AVAILABLE = False


def _render_corpus_status_badge(pmid: str, corpus: 'EvidenceCorpus' = None) -> str:
    """
    Render inclusion/exclusion status badge for a citation.

    Args:
        pmid: The PMID to check
        corpus: EvidenceCorpus object (if None, gets from session)

    Returns:
        HTML string for the status badge
    """
    if not EVIDENCE_CORPUS_AVAILABLE:
        return ""

    if corpus is None:
        corpus = get_corpus_from_session()

    if corpus is None:
        return ""

    if pmid in corpus.excluded_pmids:
        reason = corpus.excluded_pmids.get(pmid, "No reason provided")
        return f"""<span style="background: #dc3545; color: white; padding: 2px 8px;
                   border-radius: 4px; font-size: 0.75em; margin-left: 8px;"
                   title="{reason}">❌ Excluded</span>"""
    elif pmid in corpus.included_pmids:
        return """<span style="background: #28a745; color: white; padding: 2px 8px;
                   border-radius: 4px; font-size: 0.75em; margin-left: 8px;">✅ Included</span>"""
    else:
        return """<span style="background: #6c757d; color: white; padding: 2px 8px;
                   border-radius: 4px; font-size: 0.75em; margin-left: 8px;">⏳ Not Screened</span>"""


def _render_corpus_action_buttons(pmid: str, idx: int, corpus: 'EvidenceCorpus' = None):
    """
    Render include/exclude buttons for corpus management.

    Args:
        pmid: The PMID to manage
        idx: Index for unique key generation
        corpus: EvidenceCorpus object (if None, gets from session)
    """
    if not EVIDENCE_CORPUS_AVAILABLE:
        return

    if corpus is None:
        corpus = get_corpus_from_session()

    if corpus is None:
        return

    col_inc, col_exc = st.columns(2)

    with col_inc:
        is_included = pmid in corpus.included_pmids
        btn_label = "✅ Included" if is_included else "Include"
        btn_type = "primary" if not is_included else "secondary"

        if st.button(btn_label, key=f"include_{pmid}_{idx}", type=btn_type, use_container_width=True):
            if not is_included:
                corpus.include(pmid, reason="Manual inclusion from search")
                st.session_state.evidence_corpus = corpus
                st.success(f"Included PMID {pmid}")
                st.rerun()

    with col_exc:
        is_excluded = pmid in corpus.excluded_pmids
        btn_label = "❌ Excluded" if is_excluded else "Exclude"

        if st.button(btn_label, key=f"exclude_{pmid}_{idx}", use_container_width=True):
            if not is_excluded:
                # Show reason input
                reason = st.session_state.get(f"exclude_reason_{pmid}", "")
                if reason:
                    corpus.exclude(pmid, reason=reason)
                    st.session_state.evidence_corpus = corpus
                    st.success(f"Excluded PMID {pmid}")
                    st.rerun()
                else:
                    st.session_state[f"show_exclude_reason_{pmid}"] = True
                    st.rerun()

    # Show reason input if needed
    if st.session_state.get(f"show_exclude_reason_{pmid}"):
        reason = st.text_input(
            "Exclusion reason",
            key=f"exclude_reason_input_{pmid}_{idx}",
            placeholder="e.g., Wrong population, Not relevant"
        )
        if st.button("Confirm Exclusion", key=f"confirm_exclude_{pmid}_{idx}"):
            if reason:
                corpus.exclude(pmid, reason=reason)
                st.session_state.evidence_corpus = corpus
                del st.session_state[f"show_exclude_reason_{pmid}"]
                st.success(f"Excluded PMID {pmid}: {reason}")
                st.rerun()
            else:
                st.warning("Please provide a reason for exclusion")


def _persist_selected_papers():
    """Persist selected papers to database and record signals when user toggles selection."""
    context_id = st.session_state.get('active_search_context_id')
    search_context_dao = st.session_state.get('search_context_dao')
    project_id = st.session_state.get('current_project_id')
    db = st.session_state.get('db')

    if context_id and search_context_dao:
        try:
            pmids = list(st.session_state.get('selected_papers', set()))
            search_context_dao.update_selected_pmids(context_id, pmids)
            logger.debug(f"Persisted {len(pmids)} selected papers to context {context_id}")
        except Exception as e:
            logger.warning(f"Failed to persist selected papers: {e}")

    # Record paper signals for learning
    if project_id and db:
        try:
            paper_signal_dao = PaperSignalDAO(db)
            current_selection = st.session_state.get('selected_papers', set())
            previous_selection = st.session_state.get('_previous_selection', set())

            # Find newly selected and deselected papers
            newly_selected = current_selection - previous_selection
            deselected = previous_selection - current_selection

            # Record signals for newly selected papers
            for pmid in newly_selected:
                paper_signal_dao.record_signal(project_id, pmid, 'selected')
                logger.debug(f"Recorded 'selected' signal for PMID {pmid}")

            # Remove signals for deselected papers
            for pmid in deselected:
                paper_signal_dao.remove_signal(project_id, pmid, 'selected')
                logger.debug(f"Removed 'selected' signal for PMID {pmid}")

            # Update previous selection state
            st.session_state._previous_selection = current_selection.copy()

        except Exception as e:
            logger.warning(f"Failed to record paper signals: {e}")



def render_literature_search(
    citation_dao: CitationDAO,
    search_dao: SearchHistoryDAO,
    query_cache_dao: QueryCacheDAO,
    search_context_dao: SearchContextDAO = None
):
    """
    Renders the Literature Search tab.

    Args:
        citation_dao: Data Access Object for citations
        search_dao: Data Access Object for search history
        query_cache_dao: Data Access Object for query caching
        search_context_dao: Optional DAO for persisting search context
    """
    # Hero empty state when no project
    if not st.session_state.current_project_name:
        st.markdown("""
        <div class="lr-hero">
            <div class="lr-logo">Literature Review</div>
            <div class="lr-tagline">Systematic review platform</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("👈 Create or select a project to begin")
        return

    st.title("🔍 Literature Search")

    # Search Interface with Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "🔢 Identifiers", "📥 Import", "📄 Files"])

    # =============================================================================
    # TAB 1: QUERY-BASED SEARCH
    # =============================================================================
    with tab1:
        with st.form(key="query_search_form"):
            search_query = st.text_area(
                "Search Query",
                placeholder='Enter research question or keywords...',
                height=80,
                key="search_query_input",
                label_visibility="collapsed"
            )

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                max_results = st.number_input("Max", value=100, step=10)
            with col2:
                ranking_mode = st.selectbox("Ranking", ["Balanced", "Clinical", "Discovery"])
            with col3:
                 pub_types = st.multiselect("Types", ["Clinical Trial", "Review", "Meta-Analysis"], placeholder="All types")

            # Minimal Filters Expander
            with st.expander("Advanced Filters"):
                 c1, c2 = st.columns(2)
                 with c1:
                      date_from = st.text_input("From Year", placeholder="2020")
                 with c2:
                      quality_gate = st.checkbox("Strict Quality Gate", value=True)

                 # Domain filtering (ON by default for palliative surgery)
                 st.markdown("---")
                 st.markdown("**Domain Filtering**")
                 c3, c4 = st.columns(2)
                 with c3:
                      domain_filter_enabled = st.checkbox(
                          "Apply Palliative Surgery Domain Filter",
                          value=True,
                          help="Boost relevance for palliative surgery papers and penalize off-topic results"
                      )
                 with c4:
                      relevance_threshold = st.slider(
                          "Min Relevance Score",
                          min_value=0.0,
                          max_value=0.5,
                          value=0.3,
                          step=0.05,
                          help="Filter out papers below this relevance threshold (0 = no filtering)"
                      )

            if st.form_submit_button("Search PubMed", type="primary", use_container_width=True):
                 # Set default values for variables removed from UI but needed for logic
                 clinical_category = "None" 
                 clinical_scope = "Broad"
                 date_to = ""
                 query_search_clicked = True
            else:
                 query_search_clicked = False

    # =============================================================================
    # TAB 2: IDENTIFIER SEARCH
    # =============================================================================
    with tab2:
        st.caption("Directly fetch citations using identifiers.")
        
        # Vertical layout for guaranteed visibility
        pmid_input = st.text_area(
            "Enter PubMed IDs (PMIDs)", 
            placeholder="12345678, 98765432...", 
            height=120, 
            key="pmid_input"
        )
        
        doi_input = st.text_area(
            "Enter DOIs", 
            placeholder="10.1038/...", 
            height=120, 
            key="doi_input"
        )

        if pmid_input or doi_input:
            st.caption(f"Status: {len(pmid_input.split()) + len(doi_input.split())} potential items")

        identifier_search_clicked = st.button("Fetch Citations", type="primary", use_container_width=True, key="identifier_search_btn")

    # =============================================================================
    # TAB 3: IMPORT
    # =============================================================================
    with tab3:
        st.caption("Import from text or files.")
        
        col_paste, col_upload = st.columns(2)
        
        with col_paste:
            st.markdown("###### 📝 Paste Text")
            pasted_citations = st.text_area(
                "Paste Citations", 
                placeholder="Paste (RIS, BibTeX, PMIDs)...", 
                height=150, 
                key="paste_citations_input",
                label_visibility="collapsed"
            )
            if st.button("Import Text", type="primary", use_container_width=True, key="import_paste_btn"):
                 # Logic to handle pasted text
                if pasted_citations:
                    pmid_pattern = r'\\bPMID[:\\s]*(\\d{7,8})\\b'
                    found_pmids = re.findall(pmid_pattern, pasted_citations, re.IGNORECASE)
                    number_pattern = r'^(\\d{7,8})$'
                    lines = pasted_citations.strip().split('\\n')
                    for line in lines:
                        line = line.strip().replace(',', '')
                        if re.match(number_pattern, line):
                            found_pmids.append(line)
                    found_pmids = list(set(found_pmids))

                    if found_pmids:
                        with st.spinner(f"Fetching {len(found_pmids)} citations..."):
                            try:
                                client = PubMedClient(email=settings.PUBMED_EMAIL, api_key=settings.PUBMED_API_KEY)
                                citations, failed_batches = client.fetch_citations(found_pmids)
                                if failed_batches:
                                    st.warning(f"⚠️ {len(failed_batches)} batch(es) failed to fetch")
                                if citations:
                                    for c in citations:
                                        citation_dict = {
                                            "pmid": c.pmid, "title": c.title, "authors": c.authors,
                                            "journal": c.journal, "year": c.year, "abstract": c.abstract,
                                            "doi": c.doi, "publication_types": None, "keywords": None
                                        }
                                        citation_dao.upsert_citation(citation_dict)
                                        citation_dao.add_citation_to_project(st.session_state.current_project_id, c.pmid)
                                    
                                    st.session_state.search_results = {
                                        "query": f"Imported {len(citations)} items",
                                        "optimized_query": "Import",
                                        "citations": citations,
                                        "total_count": len(citations),
                                        "retrieved_count": len(citations),
                                        "search_date": datetime.now().isoformat()
                                    }
                                    st.success(f"✓ Imported {len(citations)} citations")
                                    st.rerun()
                                else:
                                    st.warning("No valid PMIDs found")
                            except Exception as e:
                                st.error(f"Import failed: {str(e)}")
                    else:
                        st.warning("No identifiers found in text")
                else:
                    st.warning("Paste text first")

        with col_upload:
            st.markdown("###### 📄 Upload File")
            uploaded_file = st.file_uploader("Upload File", type=["ris", "bib", "nbib", "txt"], key="upload_citation_file", label_visibility="collapsed")
            if uploaded_file and st.button("Import File", type="primary", use_container_width=True, key="import_file_btn"):
                    try:
                        content = uploaded_file.read().decode('utf-8')
                        pmid_pattern = r'\\bPMID[:\\s-]*(\\d{7,8})\\b'
                        found_pmids = re.findall(pmid_pattern, content, re.IGNORECASE)
                        pm_pattern = r'^PM\\s*-\\s*(\\d{7,8})'
                        found_pmids.extend(re.findall(pm_pattern, content, re.MULTILINE))
                        found_pmids = list(set(found_pmids))

                        if found_pmids:
                            with st.spinner(f"🔍 Fetching {len(found_pmids)} citation(s)..."):
                                client = PubMedClient(email=settings.PUBMED_EMAIL, api_key=settings.PUBMED_API_KEY)
                                citations, failed_batches = client.fetch_citations(found_pmids)
                                if failed_batches:
                                    st.warning(f"⚠️ {len(failed_batches)} batch(es) failed to fetch")
                                if citations:
                                    for c in citations:
                                        citation_dict = {
                                            "pmid": c.pmid, "title": c.title, "authors": c.authors,
                                            "journal": c.journal, "year": c.year, "abstract": c.abstract,
                                            "doi": c.doi, "publication_types": None, "keywords": None
                                        }
                                        citation_dao.upsert_citation(citation_dict)
                                        citation_dao.add_citation_to_project(st.session_state.current_project_id, c.pmid)

                                    st.session_state.search_results = {
                                        "query": f"Imported from {uploaded_file.name}",
                                        "optimized_query": f"Imported from {uploaded_file.name}",
                                        "used_fallback": False,
                                        "total_count": len(citations),
                                        "retrieved_count": len(citations),
                                        "query_translation": "File import",
                                        "citations": citations,
                                        "search_date": datetime.now().isoformat()
                                    }
                                    st.success(f"✓ Imported {len(citations)} citation(s) from {uploaded_file.name}")
                                    st.rerun()
                                else:
                                    st.warning("No valid citations fetched from PMIDs in file")
                        else:
                            st.warning("No PMIDs found in uploaded file. Make sure file contains PMID identifiers.")
                    except Exception as e:
                        st.error(f"Import failed: {str(e)}")

    # =============================================================================
    # TAB 4: DOCUMENT UPLOAD (Local RAG)
    # =============================================================================
    with tab4:
        if settings.ENABLE_LOCAL_RAG:
            st.markdown("**Upload reference documents** for AI-powered retrieval. Uploaded documents will be indexed and used to enhance expert panel discussions.")

            # Document upload
            uploaded_docs = st.file_uploader(
                "Drop files here or click to upload",
                type=["pdf", "docx", "pptx", "txt"],
                accept_multiple_files=True,
                help="Supported: PDF, Word (DOCX), PowerPoint (PPTX), Text files",
                key="upload_reference_docs"
            )

            if uploaded_docs:
                if st.button("📥 Index Documents", type="primary", use_container_width=True, key="index_docs_btn"):
                    with st.spinner("Processing documents..."):
                        try:
                            from core.ingestion import run_ingestion

                            results = []
                            for doc in uploaded_docs:
                                result = run_ingestion(
                                    file_input=doc,
                                    project_id=st.session_state.current_project_name or "default",
                                    filename=doc.name
                                )
                                results.append(result)

                                if result['success']:
                                    st.success(f"✅ **{doc.name}**: {result['chunks']} chunks indexed")
                                else:
                                    st.error(f"❌ **{doc.name}**: {result.get('error', 'Unknown error')}")

                            # Refresh indexed documents list
                            st.session_state.indexed_documents_dirty = True
                            st.rerun()

                        except ImportError as e:
                            st.error(f"❌ RAG dependencies not installed: {e}")
                        except Exception as e:
                            st.error(f"❌ Indexing failed: {str(e)}")

            # Show indexed documents
            st.markdown("---")
            st.subheader("📚 Indexed Documents")

            try:
                from core.ingestion import list_documents

                docs = list_documents(project_id=st.session_state.current_project_name or "default")

                if docs:
                    total_chunks = sum(d['chunk_count'] for d in docs)
                    st.caption(f"**{len(docs)} documents** | **{total_chunks} chunks**")

                    for doc in docs:
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.markdown(f"📄 **{doc['source']}** ({doc['chunk_count']} chunks)")
                        with col2:
                            if st.button("🗑️", key=f"del_doc_{doc['source']}", help="Delete document"):
                                try:
                                    from core.ingestion.pipeline import delete_document
                                    if delete_document(doc['source']):
                                        st.success(f"Deleted {doc['source']}")
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Delete failed: {e}")

                    if st.button("🗑️ Clear All Documents", type="secondary", key="clear_all_docs"):
                        try:
                            from core.ingestion import VectorDB
                            from config import settings as cfg

                            config = {
                                "storage_path": str(cfg.QDRANT_STORAGE_PATH),
                                "collection_name": cfg.QDRANT_COLLECTION_NAME,
                                "vector_size": cfg.VECTOR_SIZE,
                            }
                            vector_db = VectorDB(config)
                            if vector_db.clear_all():
                                st.success("All documents cleared")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Clear failed: {e}")
                else:
                    st.info("No documents indexed yet. Upload PDF, DOCX, or PPTX files above.")

            except ImportError:
                st.warning("⚠️ RAG dependencies not installed. Run: `pip install qdrant-client sentence-transformers`")
            except Exception as e:
                st.error(f"Failed to list documents: {e}")
        else:
            st.info("📄 Local RAG is disabled. Enable it in settings to upload and index documents.")

    # =============================================================================
    # IDENTIFIER SEARCH EXECUTION
    # =============================================================================
    if 'identifier_search_clicked' in locals() and identifier_search_clicked:
        valid_pmids = []
        valid_dois = []
        if pmid_input: valid_pmids, _ = validate_identifiers(pmid_input, "pmid")
        if doi_input: valid_dois, _ = validate_identifiers(doi_input, "doi")

        if not valid_pmids and not valid_dois:
            st.error("⚠️ Please enter at least one valid PMID or DOI")
            st.stop()

        with st.spinner(f"🔍 Fetching {len(valid_pmids) + len(valid_dois)} citation(s)..."):
            try:
                client = PubMedClient(email=settings.PUBMED_EMAIL, api_key=settings.PUBMED_API_KEY)
                all_citations = []
                not_found = []

                if valid_pmids:
                    pmid_citations, failed_batches = client.fetch_citations(valid_pmids)
                    if failed_batches:
                        st.warning(f"⚠️ {len(failed_batches)} batch(es) failed to fetch")
                    all_citations.extend(pmid_citations)
                    fetched_pmids = {c.pmid for c in pmid_citations}
                    not_found.extend([p for p in valid_pmids if p not in fetched_pmids])

                if valid_dois:
                    doi_citations, doi_not_found = client.fetch_by_doi(valid_dois)
                    all_citations.extend(doi_citations)
                    not_found.extend(doi_not_found)

                if all_citations:
                    for c in all_citations:
                        citation_dict = {
                            "pmid": c.pmid, "title": c.title, "authors": c.authors,
                            "journal": c.journal, "year": c.year, "abstract": c.abstract,
                            "doi": c.doi, "publication_types": None, "keywords": None
                        }
                        citation_dao.upsert_citation(citation_dict)
                        citation_dao.add_citation_to_project(st.session_state.current_project_id, c.pmid)

                    identifier_query = f"Identifiers: PMIDs={len(valid_pmids)}, DOIs={len(valid_dois)}"
                    search_dao.add_search(
                        project_id=st.session_state.current_project_id,
                        query=identifier_query,
                        filters={},
                        total_results=len(all_citations),
                        retrieved_count=len(all_citations)
                    )

                    st.session_state.search_results = {
                        "query": identifier_query,
                        "optimized_query": identifier_query,
                        "used_fallback": False,
                        "total_count": len(all_citations),
                        "retrieved_count": len(all_citations),
                        "query_translation": "Direct identifier fetch",
                        "citations": all_citations,
                        "search_date": datetime.now().isoformat()
                    }

                    st.success(f"✓ Fetched {len(all_citations)} citation(s)")
                    if not_found: st.warning(f"⚠️ {len(not_found)} identifier(s) not found: {', '.join(not_found[:5])}")
                    st.success(f"💾 Saved to database")
                    st.rerun()
                else:
                    st.error("❌ No citations found for the provided identifiers")
            except Exception as e:
                st.error(f"❌ Fetch failed: {str(e)}")

    # =============================================================================
    # QUERY SEARCH EXECUTION
    # =============================================================================
    if 'query_search_clicked' in locals() and query_search_clicked:
        if not search_query.strip():
            st.error("⚠️ Please enter a search query")
            st.stop()

        with st.spinner("🔍 Searching PubMed (utilizing optimized Search Service)..."):
            try:
                # Prepare filters
                search_filters = {
                    "date_from": date_from,
                    "date_to": date_to,
                    "pub_types": pub_types,
                    "clinical_category": clinical_category,
                    "clinical_scope": clinical_scope,
                    "quality_gate": quality_gate
                }

                # Initialize Service
                search_service = SearchService(
                    openai_api_key=settings.OPENAI_API_KEY, 
                    model=settings.OPENAI_MODEL
                )

                # Execute Search
                results = search_service.execute_search(
                    query=search_query,
                    project_id=st.session_state.current_project_id,
                    citation_dao=citation_dao,
                    search_dao=search_dao,
                    query_cache_dao=query_cache_dao,
                    search_context_dao=search_context_dao,  # For persisting search metadata
                    max_results=max_results,
                    ranking_mode=ranking_mode,
                    filters=search_filters,
                    domain="palliative_surgery" if domain_filter_enabled else None,
                    relevance_threshold=relevance_threshold
                )

                # Update Session State
                st.session_state.search_results = results

                # Store context ID for persisting selected papers
                if results.get('context_id'):
                    st.session_state.active_search_context_id = results['context_id']

                # Success Messages
                st.success(f"✓ Found {results['total_count']} results, retrieved {results['retrieved_count']} citations")
                if results.get('domain_filter_applied'):
                    filtered_msg = f"🎯 Domain filter: **Palliative Surgery**"
                    if results.get('filtered_count', 0) > 0:
                        filtered_msg += f" ({results['filtered_count']} low-relevance papers filtered)"
                    st.info(filtered_msg)
                st.success(f"💾 Saved to database")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")
                logger.error(f"Search failed: {e}", exc_info=True)

    # Display search results
    if st.session_state.search_results:
        st.markdown("---")
        st.subheader("Search Results")
        results = st.session_state.search_results

        with st.container():
            st.markdown("### 🔍 Query Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                query_type_display = results.get("query_type", "DIRECT")
                if results.get("is_cached"): query_type_display += " (Cached)"
                st.metric("Query Type", query_type_display)
            with col2: st.metric("Confidence", results.get("query_confidence", "high"))
            with col3:
                if results.get("used_fallback"): st.metric("Status", "Fallback Used", delta="⚠️")
                else: st.metric("Status", "Success", delta="✓")

            if results.get("query"):
                st.markdown("**Original Query:**")
                st.code(results["query"], language="text")

                if results.get("optimized_query") and results["optimized_query"] != results["query"]:
                    st.markdown("**Optimized Query:**")
                    st.code(results["optimized_query"], language="text")
                    if results.get("query_explanation"): st.caption(f"💡 {results['query_explanation']}")

            if results.get("query_translation"):
                st.markdown("**PubMed Translation:**")
                st.code(results["query_translation"], language="text")

            if results.get("used_fallback") and results.get("fallback_from_query"):
                st.warning("ℹ️ AI-optimized query returned 0 results. Used simpler fallback query above.")
                with st.expander("🔍 See original AI-optimized query", expanded=False):
                    st.code(results["fallback_from_query"], language="text")

            if results.get("final_query") and results.get("final_query") != results.get("optimized_query"):
                st.markdown("**Final Query (with filters):**")
                st.code(results["final_query"], language="text")

            filter_info = []
            if results.get("applied_filters"): filter_info.extend([f"🏥 {f}" for f in results["applied_filters"]])
            if results.get("ranking_mode"): filter_info.append(f"📊 Ranking: {results['ranking_mode']}")
            if filter_info: st.caption(" · ".join(filter_info))

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        total_count = results.get('total_count', 0)
        retrieved_count = results.get('retrieved_count', len(results.get('citations', [])))
        col1.metric("Total Matches", f"{total_count:,}")
        col2.metric("Retrieved", retrieved_count)
        col3.metric("Selected", len(st.session_state.selected_papers))

        # =============================================================================
        # VISUALIZATION SECTION
        # =============================================================================
        if VISUALIZATION_AVAILABLE and results.get('citations'):
            with st.expander("📊 Visualizations", expanded=False):
                viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📅 Timeline", "🔗 Citation Network", "📋 Table View"])

                # Get citations for visualization
                citations_list = results.get('citations', [])

                # --- Timeline Chart ---
                with viz_tab1:
                    st.subheader("Publication Timeline")

                    # Extract years from citations
                    years = []
                    for c in citations_list:
                        year = c.year if hasattr(c, 'year') else c.get('year')
                        if year:
                            try:
                                years.append(int(year))
                            except (ValueError, TypeError):
                                pass

                    if years:
                        fig = px.histogram(
                            x=years,
                            nbins=min(20, len(set(years))),
                            title="Publications by Year",
                            labels={'x': 'Year', 'y': 'Number of Papers'}
                        )
                        fig.update_layout(
                            bargap=0.1,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Year Range", f"{min(years)} - {max(years)}")
                        col2.metric("Median Year", int(sorted(years)[len(years)//2]))
                        col3.metric("Most Recent", max(years))
                    else:
                        st.info("No year data available for timeline")

                # --- Citation Network ---
                with viz_tab2:
                    st.subheader("Citation Network")
                    st.caption("Build a network showing citation relationships between papers")

                    # Get PMIDs for network
                    pmids = []
                    for c in citations_list:
                        pmid = c.pmid if hasattr(c, 'pmid') else c.get('pmid')
                        if pmid:
                            pmids.append(pmid)

                    if len(pmids) >= 3:
                        col1, col2 = st.columns(2)
                        with col1:
                            network_mode = st.radio(
                                "Network Mode",
                                ["Minimal (corpus only)", "Extended (include citing papers)"],
                                help="Minimal: Only show connections within your search results. Extended: Include papers that cite your results."
                            )
                        with col2:
                            max_papers_network = st.slider("Max papers to analyze", 10, min(100, len(pmids)), min(30, len(pmids)))

                        if st.button("🔗 Build Citation Network", type="primary"):
                            with st.spinner("Building citation network (this may take 30-60 seconds)..."):
                                try:
                                    builder = CitationNetworkBuilder()
                                    if network_mode == "Minimal (corpus only)":
                                        network = builder.build_network_minimal(pmids[:max_papers_network])
                                    else:
                                        network = builder.build_network(
                                            pmids[:max_papers_network],
                                            max_citations_per_paper=20,
                                            max_references_per_paper=20,
                                            include_external=True
                                        )

                                    if network.num_papers > 0:
                                        # Display network stats
                                        col1, col2, col3, col4 = st.columns(4)
                                        col1.metric("Papers", network.num_papers)
                                        col2.metric("Citations", network.num_citations)
                                        col3.metric("Clusters", len(network.clusters))
                                        col4.metric("Hub Papers", len(network.hub_papers))

                                        # Create visualization
                                        fig = create_network_visualization(
                                            network,
                                            color_by="cluster",
                                            size_by="citation_count",
                                            corpus_only=(network_mode == "Minimal (corpus only)")
                                        )
                                        if fig:
                                            st.plotly_chart(fig, use_container_width=True, key="network_viz")

                                        # Show hub papers
                                        if network.hub_papers:
                                            st.markdown("**Hub Papers** (highly cited within network):")
                                            for paper_id in network.hub_papers[:5]:
                                                node = network.nodes.get(paper_id)
                                                if node:
                                                    st.markdown(f"- [{node.title[:80]}...](https://pubmed.ncbi.nlm.nih.gov/{node.pmid}/) ({node.citation_count} citations)")
                                    else:
                                        st.warning("Could not build network - no citation data found")
                                except Exception as e:
                                    st.error(f"Failed to build network: {str(e)}")
                    else:
                        st.info("Need at least 3 papers to build a citation network")

                # --- Table View ---
                with viz_tab3:
                    st.subheader("Results Table")
                    st.caption("Sortable and filterable view of all results")

                    # Build dataframe from citations
                    table_data = []
                    for idx, c in enumerate(citations_list, 1):
                        pmid = c.pmid if hasattr(c, 'pmid') else c.get('pmid')
                        title = c.title if hasattr(c, 'title') else c.get('title', 'No title')
                        authors = c.authors if hasattr(c, 'authors') else c.get('authors', [])
                        journal = c.journal if hasattr(c, 'journal') else c.get('journal', '')
                        year = c.year if hasattr(c, 'year') else c.get('year', '')

                        # Get score if available
                        score = None
                        if results.get('scored_citations'):
                            for sc in results['scored_citations']:
                                if sc.citation.get('pmid') == pmid:
                                    score = sc.final_score
                                    break

                        first_author = authors[0] if authors else 'Unknown'
                        if isinstance(first_author, list):
                            first_author = first_author[0] if first_author else 'Unknown'

                        table_data.append({
                            'Rank': idx,
                            'Score': f"{score:.3f}" if score else "-",
                            'Title': title[:100] + '...' if len(title) > 100 else title,
                            'First Author': first_author,
                            'Journal': journal,
                            'Year': year,
                            'PMID': pmid
                        })

                    if table_data:
                        df = pd.DataFrame(table_data)
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "PMID": st.column_config.LinkColumn(
                                    "PMID",
                                    help="Click to view on PubMed",
                                    display_text=r"(\d+)",
                                    validate=r"^\d+$"
                                ),
                                "Score": st.column_config.TextColumn("Score", help="Clinical Utility Score"),
                                "Title": st.column_config.TextColumn("Title", width="large"),
                            }
                        )

                        # Export options
                        csv = df.to_csv(index=False)
                        col_csv, col_word = st.columns(2)
                        with col_csv:
                            st.download_button(
                                "📥 Download CSV",
                                csv,
                                f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv",
                                use_container_width=True
                            )
                        with col_word:
                            try:
                                from core.report_export import generate_literature_report
                                citations_list = results.get('citations', [])
                                report_buffer = generate_literature_report(
                                    citations=citations_list,
                                    query=results.get('query'),
                                    project_name=st.session_state.get('current_project_name'),
                                    include_abstracts=False
                                )
                                st.download_button(
                                    label="📄 Download Word",
                                    data=report_buffer,
                                    file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    key="search_export_word_btn",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.button("📄 Download Word", disabled=True, help=f"Export unavailable: {e}",
                                          key="search_export_word_btn_disabled", use_container_width=True)

        # =============================================================================
        # SUGGESTED REFINEMENT QUESTIONS
        # =============================================================================
        if settings.OPENAI_API_KEY and results.get('query'):
            with st.expander("💡 Suggested Refinements", expanded=False):
                st.caption("AI-generated questions to refine your search")

                if st.button("Generate Suggestions", key="gen_suggestions"):
                    with st.spinner("Generating refinement suggestions..."):
                        try:
                            # Use new SearchService
                            search_service = SearchService(api_key=settings.OPENAI_API_KEY)
                            suggestions = search_service.generate_search_refinements(
                                query=results.get('query', ''),
                                total_results=results.get('total_count', 0),
                                citations=results.get('citations', [])[:5]
                            )
                            st.session_state.search_suggestions = suggestions
                        except Exception as e:
                            st.error(f"Failed to generate suggestions: {e}")

                if st.session_state.get('search_suggestions'):
                    for i, suggestion in enumerate(st.session_state.search_suggestions, 1):
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.markdown(f"**{i}.** {suggestion}")
                        with col2:
                            if st.button("Use", key=f"use_suggestion_{i}"):
                                st.session_state.suggested_query = suggestion
                                st.rerun()

        sel_col1, sel_col2, sel_col3 = st.columns([1, 1, 2])
        with sel_col1:
            if st.button("☑️ Select All", use_container_width=True):
                if results.get('scored_citations'): all_pmids = {sc.citation.get('pmid') for sc in results['scored_citations']}
                elif results.get('citations'): all_pmids = {get_citation_attr(c, 'pmid') for c in results['citations']}
                else: all_pmids = set()
                st.session_state.selected_papers = all_pmids
                st.rerun()
        with sel_col2:
            if st.button("☐ Deselect All", use_container_width=True):
                st.session_state.selected_papers = set()
                st.rerun()
        with sel_col3:
            retrieved = results.get('retrieved_count', len(results.get('citations', [])))
            st.caption(f"💡 {len(st.session_state.selected_papers)} of {retrieved} selected")

        if results.get('scored_citations'):
            col_left, col_right = st.columns([2, 1])
            with col_left: st.caption("💡 Clinical Utility Score combines evidence quality, relevance, and recency")
            with col_right:
                sort_option = st.selectbox(
                    "🔀 Sort by",
                    ["Clinical Utility Score (Evidence + Relevance)", "Relevance (PubMed Best Match)", "Publication Year (Newest First)", "Publication Year (Oldest First)", "Evidence Quality (RCTs/SRs First)"],
                    index=0, key="sort_option"
                )

            scored_citations = results['scored_citations'].copy()
            if sort_option == "Relevance (PubMed Best Match)": scored_citations.sort(key=lambda x: x.rank_position)
            elif sort_option == "Publication Year (Newest First)": scored_citations.sort(key=lambda x: int(x.citation.get('year', '0') or '0'), reverse=True)
            elif sort_option == "Publication Year (Oldest First)": scored_citations.sort(key=lambda x: int(x.citation.get('year', '0') or '0'))
            elif sort_option == "Evidence Quality (RCTs/SRs First)": scored_citations.sort(key=lambda x: x.evidence_score, reverse=True)

            st.markdown("### 📚 Citations")
            st.caption(f"Sorted by: {sort_option}")

            for idx, scored_cit in enumerate(scored_citations, 1):
                c = scored_cit.citation
                pmid = c['pmid']
                is_selected = pmid in st.session_state.selected_papers

                col1, col2 = st.columns([0.05, 0.95])
                with col1:
                    # Vertical alignment hack
                    st.write("") 
                    checkbox_icon = "✅" if is_selected else "☐"
                    if st.button(checkbox_icon, key=f"select_{pmid}_{idx}", use_container_width=True):
                        if is_selected: st.session_state.selected_papers.discard(pmid)
                        else: st.session_state.selected_papers.add(pmid)
                        # Persist selection to database
                        _persist_selected_papers()
                        st.rerun()

                with col2:
                    # Use expander as the "Card" container
                    card_title = f"#{idx} · {c['title'][:100]}..." if len(c['title']) > 100 else f"#{idx} · {c['title']}"
                    with st.expander(card_title, expanded=False):
                        # Header details inside card
                        score_badge = f"**Score: {scored_cit.final_score:.3f}**"

                        # Add corpus status badge
                        corpus_badge = _render_corpus_status_badge(pmid)
                        if corpus_badge:
                            st.markdown(f"{score_badge} {corpus_badge}", unsafe_allow_html=True)
                        else:
                            st.markdown(score_badge)

                        title_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        st.markdown(f"**Full Title:** [{c['title']}]({title_link})")

                        if scored_cit.explanation:
                            chips_html = " ".join([f"<span style='background-color: var(--primary-soft); color: var(--primary); padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-right: 4px; border: 1px solid var(--primary);'>{chip}</span>" for chip in scored_cit.explanation])
                            st.markdown(f"<div style='margin: 8px 0;'>{chips_html}</div>", unsafe_allow_html=True)

                        if c.get('authors'):
                            authors_display = ', '.join(c['authors'][:3])
                            if len(c['authors']) > 3: authors_display += f", et al. ({len(c['authors'])} authors)"
                            st.caption(f"✍️ {authors_display}")

                        journal_info = c.get('journal', 'Unknown Journal')
                        year_info = c.get('year', 'N/A')
                        st.caption(f"📖 *{journal_info}* · 📅 {year_info}")
                        
                        # Abstract
                        if c.get('abstract'):
                            st.markdown("---")
                            st.markdown(f"_{c['abstract'][:500]}..._" if len(c['abstract']) > 500 else c['abstract'])

                        # Evidence Corpus Actions
                        if EVIDENCE_CORPUS_AVAILABLE:
                            st.markdown("---")
                            st.markdown("**Evidence Corpus:**")
                            _render_corpus_action_buttons(pmid, idx)

                        # Badges footer
                        st.markdown("---")
                        badges = []
                        badges.append(f"[![PMID](https://img.shields.io/badge/PMID-{pmid}-blue)](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                        if c.get('doi'):
                            doi_clean = c['doi'].replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
                            badges.append(f"[![DOI](https://img.shields.io/badge/DOI-{doi_clean.replace('-', '--')}-orange)](https://doi.org/{doi_clean})")
                        st.markdown(" ".join(badges))

        elif results['citations']:
            st.markdown("### Citations")
            st.caption(f"Click on any citation to view full details")

            for idx, c in enumerate(results['citations'], 1):
                pmid = get_citation_attr(c, 'pmid')
                is_selected = pmid in st.session_state.selected_papers

                col1, col2 = st.columns([0.05, 0.95])
                with col1:
                    st.write("")
                    checkbox_icon = "✅" if is_selected else "☐"
                    if st.button(checkbox_icon, key=f"select_{pmid}_{idx}", use_container_width=True):
                        if is_selected: st.session_state.selected_papers.discard(pmid)
                        else: st.session_state.selected_papers.add(pmid)
                        # Persist selection to database
                        _persist_selected_papers()
                        st.rerun()

                with col2:
                    title = get_citation_attr(c, 'title', 'No title')
                    # Ensure title is a string (some citation types may return methods)
                    if not isinstance(title, str):
                        title = str(title) if title else 'No title'
                    card_title = f"{idx}. {title[:100]}..." if len(title) > 100 else f"{idx}. {title}"

                    with st.expander(card_title, expanded=False):
                        # Add corpus status badge
                        corpus_badge = _render_corpus_status_badge(pmid)
                        if corpus_badge:
                            st.markdown(corpus_badge, unsafe_allow_html=True)

                        title_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        st.markdown(f"**Full Title:** [{title}]({title_link})")
                        
                        authors = get_citation_attr(c, 'authors', [])
                        if authors:
                            authors_display = ', '.join(authors[:3])
                            if len(authors) > 3: authors_display += f", et al. ({len(authors)} authors)"
                            st.caption(f"✍️ {authors_display}")

                        journal = get_citation_attr(c, 'journal', 'Unknown Journal')
                        year = get_citation_attr(c, 'year', 'N/A')
                        st.caption(f"📖 *{journal}* · 📅 {year}")

                        abstract = get_citation_attr(c, 'abstract', '')
                        if abstract:
                             st.markdown("---")
                             st.markdown(f"_{abstract[:500]}..._" if len(abstract) > 500 else abstract)

                        # Evidence Corpus Actions
                        if EVIDENCE_CORPUS_AVAILABLE:
                            st.markdown("---")
                            st.markdown("**Evidence Corpus:**")
                            _render_corpus_action_buttons(pmid, idx)

                        doi = get_citation_attr(c, 'doi', '')
                        st.markdown("---")
                        badges = []
                        badges.append(f"[![PMID](https://img.shields.io/badge/PMID-{pmid}-blue)](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                        if doi:
                            doi_clean = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
                            badges.append(f"[![DOI](https://img.shields.io/badge/DOI-{doi_clean.replace('-', '--')}-orange)](https://doi.org/{doi_clean})")
                        st.markdown(" ".join(badges))

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                pmid_input = st.text_input("Add PMID to selection", placeholder="Enter PMID and press Add")
            with col2:
                if st.button("➕ Add"):
                    if pmid_input and pmid_input not in st.session_state.selected_papers:
                        st.session_state.selected_papers.add(pmid_input)
                        st.success(f"✓ Added {pmid_input}")
                        st.rerun()
            with col3:
                if st.button("🗑️ Clear All"):
                    st.session_state.selected_papers = set()
                    st.success("✓ Cleared selection")
                    st.rerun()
