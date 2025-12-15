"""
Sidebar UI Module

Handles the sidebar rendering including:
- Project Manager (select/create projects)
- Data Management popover
- Knowledge Store popover
- Export & Zotero integration
- Configuration warnings
"""

import streamlit as st
from datetime import datetime
from config import settings
from core.utils import get_citation_attr
from core.export_formats import CitationExporter, EXPORT_FORMATS
from core.knowledge_store import (
    get_default_store,
    get_knowledge_summary,
    get_all_knowledge,
    delete_knowledge,
    get_all_triples
)
from core.database import ProgramProfileDAO
from ui.mark_pen import render_mark_sidebar


def render_sidebar(
    project_dao,
    citation_dao,
    screening_dao,
    zotero_client_class=None
):
    """
    Render the complete sidebar.

    Args:
        project_dao: ProjectDAO instance for project operations
        citation_dao: CitationDAO instance for citation operations
        screening_dao: ScreeningDAO instance for screening operations
        zotero_client_class: Optional ZoteroClient class for Zotero integration
    """
    with st.sidebar:
        # Minimal Header
        st.title(f"{settings.APP_ICON} Palliative Surgery GDG")
        
        # 1. Project Selector (Compact)
        _render_project_manager(project_dao, citation_dao, screening_dao)

        # 1b. Program Profile (auto-populated from questions)
        _render_program_profile()

        # 2. Main Navigation / Tools (Grouped)
        st.markdown("### Toolkit")
        _render_document_library()
        _render_knowledge_store()
        _render_cdp_workspace_link()
        _render_export_panel(zotero_client_class)

        # 2b. User Marks (My Clippings)
        render_mark_sidebar(project_id=st.session_state.get('current_project_id'))

        # 3. Mode Switch (v3.0)
        st.markdown("---")
        _render_mode_switch()

        # 4. LLM Transparency
        _render_llm_info()

        # 5. Status (Minimal)
        if not settings.OPENAI_API_KEY:
            st.error("🔑 API Key Missing")

def _render_project_manager(project_dao, citation_dao, screening_dao):
    """Render minimalist project selection."""
    # Get all projects from database
    all_projects = project_dao.get_all_projects()
    project_names = [p.name for p in all_projects]
    
    current_idx = 0
    if st.session_state.current_project_name in project_names:
        current_idx = project_names.index(st.session_state.current_project_name) + 1

    selected_project = st.selectbox(
        "Project",
        options=["-- New --"] + project_names,
        index=current_idx,
        label_visibility="collapsed"
    )

    if selected_project == "-- New --":
        new_name = st.text_input("Name", placeholder="New Project Name", label_visibility="collapsed", key="new_project_name_input")
        if st.button("Create", use_container_width=True, key="create_project_btn"):
            if new_name and new_name.strip():
                # Check if project already exists
                existing = project_dao.get_project_by_name(new_name.strip())
                if existing:
                    st.warning(f"Project '{new_name}' already exists")
                else:
                    try:
                        pid = project_dao.create_project(new_name.strip(), "")
                        st.session_state.current_project_id = pid
                        st.session_state.current_project_name = new_name.strip()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create project: {e}")
    elif selected_project != st.session_state.current_project_name:
        project = project_dao.get_project_by_name(selected_project)
        if project:
            st.session_state.current_project_id = project.id
            st.session_state.current_project_name = project.name

            # Load search context for this project
            search_context_dao = st.session_state.get('search_context_dao')
            if search_context_dao:
                try:
                    active_context = search_context_dao.get_active_context(project.id)
                    if active_context:
                        # Restore selected papers from saved context
                        st.session_state.selected_papers = set(active_context.get('selected_pmids', []))
                        st.session_state.active_search_context_id = active_context.get('id')
                        # Store context metadata for display
                        st.session_state.search_context = active_context
                    else:
                        # Clear search state for new project
                        st.session_state.selected_papers = set()
                        st.session_state.active_search_context_id = None
                        st.session_state.search_context = None
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Failed to load search context: {e}")

            st.rerun()

    # Data Stats (Minimal)
    if st.session_state.current_project_id:
        count = len(citation_dao.get_citations_by_project(st.session_state.current_project_id))
        st.caption(f"{count} papers loaded")
        with st.popover("Manage Data", use_container_width=True):
             _render_manage_data_popover_content(citation_dao, st.session_state.current_project_id)


def _render_manage_data_popover_content(citation_dao, project_id):
    project_citations = citation_dao.get_citations_by_project(project_id)
    if st.button("Clear All Data", type="primary"):
         for cit in project_citations:
             pmid = cit.pmid if hasattr(cit, 'pmid') else cit.get('pmid')
             citation_dao.remove_citation_from_project(project_id, pmid)
         st.rerun()
    st.markdown("---")
    for cit in project_citations[:5]:
        pmid = cit.pmid if hasattr(cit, 'pmid') else cit.get('pmid')
        title = cit.title if hasattr(cit, 'title') else cit.get('title', 'No title')
        c1, c2 = st.columns([5,1])
        c1.caption(title[:40]+"...")
        if c2.button("x", key=f"del_{pmid}"):
             citation_dao.remove_citation_from_project(project_id, pmid)
             st.rerun()


def _render_program_profile():
    """Render program profile expander with auto-detected context."""
    project_id = st.session_state.get('current_project_id')
    db = st.session_state.get('db')

    if not project_id or not db:
        return

    try:
        profile_dao = ProgramProfileDAO(db)
        profile = profile_dao.get(project_id) or {}
    except Exception:
        return

    # Only show if we have some data or user wants to see it
    if not profile.get("target") and not profile.get("indication"):
        # Show collapsed hint
        with st.expander("🏥 Clinical Context", expanded=False):
            st.caption("Auto-populated from your questions")
            st.info("Ask a clinical question to auto-detect context (scenario, cancer type, intervention).")
        return

    with st.expander("🏥 Clinical Context", expanded=False):
        st.caption("Auto-populated from your questions")

        # Display current profile - adapted for palliative surgery
        target = st.text_input(
            "Scenario",
            value=profile.get("target", ""),
            placeholder="e.g., Malignant Bowel Obstruction",
            key="profile_target_input"
        )

        indication = st.text_input(
            "Cancer Type",
            value=profile.get("indication", ""),
            placeholder="e.g., Ovarian cancer, Colorectal cancer",
            key="profile_indication_input"
        )

        drugs_str = ", ".join(profile.get("drug_names", []))
        drugs = st.text_input(
            "Interventions",
            value=drugs_str,
            placeholder="e.g., Surgery, Stent, Conservative",
            key="profile_drugs_input"
        )

        competitors_str = ", ".join(profile.get("competitors", []))
        competitors = st.text_input(
            "Comparators",
            value=competitors_str,
            placeholder="e.g., Medical management, Hospice",
            key="profile_competitors_input"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save", key="save_profile_btn", use_container_width=True):
                try:
                    updated_profile = {
                        "target": target.strip() if target else None,
                        "indication": indication.strip() if indication else None,
                        "drug_names": [d.strip() for d in drugs.split(",") if d.strip()],
                        "competitors": [c.strip() for c in competitors.split(",") if c.strip()],
                        "mechanism": profile.get("mechanism"),
                        "therapeutic_area": profile.get("therapeutic_area"),
                        "development_stage": profile.get("development_stage")
                    }
                    profile_dao.upsert(project_id, updated_profile)
                    st.success("Saved!")
                except Exception as e:
                    st.error(f"Failed to save: {e}")

        with col2:
            if st.button("🗑️ Clear", key="clear_profile_btn", use_container_width=True):
                try:
                    profile_dao.delete(project_id)
                    st.success("Cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear: {e}")


def _render_document_library():
    """Render Document Library with upload and management."""
    try:
        from services.document_library import get_document_library

        library = get_document_library()
        if not library:
            st.caption("Document Library (unavailable)")
            return

        stats = library.get_stats()
        doc_count = stats.get("document_count", 0)

        with st.popover(f"📚 Documents ({doc_count})", use_container_width=True):
            # Upload section
            uploaded = st.file_uploader(
                "Add to Library",
                type=['pdf', 'txt', 'docx', 'md'],
                key="library_upload",
                label_visibility="collapsed"
            )

            if uploaded:
                with st.spinner(f"Processing {uploaded.name}..."):
                    result = library.add_document(uploaded, uploaded.name)
                    if result["status"] == "success":
                        st.success(f"Added: {result['title']} ({result['chunk_count']} chunks)")
                    else:
                        st.error(f"Failed: {result.get('message', 'Unknown error')}")

            # List documents
            docs = library.list_documents()
            if docs:
                st.markdown("**Stored Documents:**")
                for doc in docs[:10]:
                    col1, col2 = st.columns([4, 1])
                    col1.caption(f"{doc['source'][:30]}... ({doc['chunk_count']} chunks)")
                    if col2.button("x", key=f"del_doc_{doc['source'][:10]}"):
                        library.delete_document(doc['source'])
                        st.rerun()

            # Clear all
            if docs and st.button("Clear All Documents", type="secondary"):
                library.clear_all()
                st.rerun()

    except Exception as e:
        st.caption(f"Document Library (error)")


def _render_knowledge_store():
    """Render minimalist Knowledge Store - gated for advanced tools."""
    # Only show Knowledge Store when advanced tools are enabled
    if not settings.ENABLE_ADVANCED_TOOLS:
        return

    summary = get_knowledge_summary()
    total = sum(summary.values())
    if total > 0:
        with st.popover(f"Knowledge ({total})", use_container_width=True):
            if st.button("Clear Memory"):
                 store = get_default_store()
                 store.clear_all_triples()
                 st.rerun()


def _render_cdp_workspace_link():
    """Render Guideline Workspace link/button."""
    cdp_sections = st.session_state.get('cdp_sections', {})
    section_count = len(cdp_sections)

    if section_count > 0:
        if st.button(f"📋 Guideline Workspace ({section_count})", use_container_width=True, type="secondary"):
            st.session_state.cdp_show_workspace = True
            st.session_state.ui_mode = 'cdp_workspace'  # Switch to Guideline workspace view
            st.rerun()
    else:
        # Show disabled/hint state
        st.caption("📋 Guideline Workspace (empty)")


def _render_export_panel(zotero_client_class=None):
    """Render minimalist Export panel with working download."""
    citations = st.session_state.get('search_results', [])

    # Also check for research result citations
    if not citations:
        result = st.session_state.get('research_result')
        if result:
            evidence = result.get('evidence_summary', {}) if isinstance(result, dict) else getattr(result, 'evidence_summary', {})
            citations = evidence.get('citations', []) if evidence else []

    if not citations:
        return

    with st.popover("📤 Export", use_container_width=True):
        st.markdown(f"**{len(citations)} citations available**")
        export_format = st.selectbox("Format", list(EXPORT_FORMATS.keys()), key="sidebar_export_format_select")

        # Generate export data
        exporter = CitationExporter()

        if export_format == "RIS":
            data = exporter.to_ris(citations)
            filename = "citations.ris"
            mime = "application/x-research-info-systems"
        elif export_format == "BibTeX":
            data = exporter.to_bibtex(citations)
            filename = "citations.bib"
            mime = "application/x-bibtex"
        elif export_format == "CSV":
            data = exporter.to_csv(citations)
            filename = "citations.csv"
            mime = "text/csv"
        else:  # JSON
            import json
            data = json.dumps([c.__dict__ if hasattr(c, '__dict__') else c for c in citations], indent=2)
            filename = "citations.json"
            mime = "application/json"

        st.download_button(
            label=f"⬇️ Download {export_format}",
            data=data,
            file_name=filename,
            mime=mime,
            use_container_width=True
        )

        # Zotero direct upload
        if settings.ENABLE_ZOTERO:
            st.markdown("---")
            _render_zotero_upload(citations)

def _render_zotero_upload(citations):
    """Render Zotero upload UI within export panel."""
    from core.zotero_client import ZoteroClient
    from core.pubmed_client import Citation

    # Collection name input
    collection_name = st.text_input(
        "Collection name",
        value=f"Palliative Surgery GDG - {datetime.now().strftime('%Y-%m-%d')}",
        key="zotero_collection_name",
        help="Creates a new collection in Zotero"
    )

    if st.button("📚 Upload to Zotero", use_container_width=True, type="primary"):
        try:
            # Initialize client
            client = ZoteroClient(
                api_key=settings.ZOTERO_API_KEY,
                user_id=settings.ZOTERO_USER_ID
            )

            # Test connection
            if not client.test_connection():
                st.error("Failed to connect to Zotero. Check your API key and user ID.")
                return

            with st.spinner("Creating collection..."):
                # Create collection
                collection = client.create_collection(collection_name)
                collection_key = collection.get('key')

            # Convert citations to Citation objects if needed
            citation_objects = []
            for c in citations:
                if isinstance(c, Citation):
                    citation_objects.append(c)
                elif isinstance(c, dict):
                    citation_objects.append(Citation(
                        pmid=c.get('pmid', ''),
                        title=c.get('title', ''),
                        authors=c.get('authors', []),
                        journal=c.get('journal', ''),
                        year=c.get('year', ''),
                        abstract=c.get('abstract', ''),
                        doi=c.get('doi', '')
                    ))
                elif hasattr(c, 'pmid'):
                    # Object with attributes
                    citation_objects.append(Citation(
                        pmid=getattr(c, 'pmid', ''),
                        title=getattr(c, 'title', ''),
                        authors=getattr(c, 'authors', []),
                        journal=getattr(c, 'journal', ''),
                        year=getattr(c, 'year', ''),
                        abstract=getattr(c, 'abstract', ''),
                        doi=getattr(c, 'doi', '')
                    ))

            with st.spinner(f"Uploading {len(citation_objects)} citations..."):
                # Upload citations
                result = client.add_citations(citation_objects, collection_key=collection_key)

            st.success(f"✅ Uploaded {result['successful']}/{result['total']} citations to '{collection_name}'")
            if result['failed'] > 0:
                st.warning(f"⚠️ {result['failed']} citations failed to upload")

        except Exception as e:
            st.error(f"Zotero upload failed: {str(e)}")


def _render_config_warnings():
    pass


def _render_mode_switch():
    """Render mode switch - simplified for GDG v1.0."""
    ui_mode = st.session_state.get('ui_mode', 'conversational')

    # Display current mode indicator
    if ui_mode == 'cdp_workspace':
        st.caption("Mode: Guideline Workspace")
        if st.button("Back to GDG", use_container_width=True, type="primary"):
            st.session_state.ui_mode = 'conversational'
            st.rerun()
    else:
        # Only show Advanced mode switch when advanced tools are enabled
        if settings.ENABLE_ADVANCED_TOOLS:
            st.caption("Mode: GDG Discussion")
            # No longer showing mode switch - the 4-tab interface handles navigation


def _render_llm_info():
    """Render LLM model info in sidebar."""
    with st.expander("🤖 LLM Models", expanded=False):
        st.caption("Which models power each feature:")

        st.markdown(f"""
**Expert Panel:** `{settings.EXPERT_MODEL}`

**Reasoning:** `{settings.REASONING_MODEL}`

**Screening:** `{settings.SCREENING_MODEL}`
        """)

        st.caption("PubMed & web searches always use current data.")
