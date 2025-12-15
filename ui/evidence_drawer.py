"""
Evidence Drawer Component

Collapsible panel showing:
- Literature citations with validation status
- Expert validation details
- Search query used
"""

import streamlit as st
from typing import Union, Dict, List
from datetime import datetime

from config import settings
from services.research_partner_service import ResearchResult
from services.expert_service import ValidationResult


def render_evidence_drawer(result: Union[ResearchResult, Dict]):
    """
    Render the collapsible evidence drawer.

    Args:
        result: ResearchResult object or dict with result data
    """
    # Handle both ResearchResult and dict
    if isinstance(result, dict):
        evidence_summary = result.get('evidence_summary', {})
        validations = result.get('validations', {})
    else:
        evidence_summary = result.evidence_summary
        validations = result.validations

    citations = evidence_summary.get('citations', [])
    clinical_trials = evidence_summary.get('clinical_trials', [])
    search_query = evidence_summary.get('search_query', '')
    paper_count = evidence_summary.get('paper_count', len(citations))
    trial_count = evidence_summary.get('trial_count', len(clinical_trials))

    # Evidence drawer toggle
    drawer_expanded = st.session_state.get('show_evidence_drawer', False)

    # Build title
    title_parts = []
    if paper_count > 0:
        title_parts.append(f"{paper_count} papers")
    if trial_count > 0:
        title_parts.append(f"{trial_count} trials")
    title = "Evidence (" + ", ".join(title_parts) + ")" if title_parts else "Evidence"

    with st.expander(title, expanded=drawer_expanded):
        # Search debug info (new tiered search metadata)
        concepts = evidence_summary.get('concepts', {})
        queries_executed = evidence_summary.get('queries_executed', {})
        tier_used = evidence_summary.get('tier_used', '')
        total_before_filter = evidence_summary.get('total_before_filter', 0)
        filtered_count = evidence_summary.get('filtered_count', 0)

        # Show search debug when no results or always useful for understanding
        if concepts or queries_executed or not citations:
            with st.expander("Search Debug", expanded=not citations):
                if concepts:
                    st.markdown("**Extracted Concepts:**")
                    cols = st.columns(2)
                    with cols[0]:
                        if concepts.get('targets'):
                            st.markdown(f"**Targets:** {', '.join(concepts['targets'])}")
                        if concepts.get('indications'):
                            st.markdown(f"**Indications:** {', '.join(concepts['indications'])}")
                    with cols[1]:
                        if concepts.get('mechanisms'):
                            st.markdown(f"**Mechanisms:** {', '.join(concepts['mechanisms'])}")
                        if concepts.get('modalities'):
                            st.markdown(f"**Modalities:** {', '.join(concepts['modalities'])}")

                if tier_used:
                    tier_display = {
                        "tier1": "Tier 1 (strict queries)",
                        "tier2": "Tier 2 (relaxed queries)",
                        "tier3": "Tier 3 (broad queries)",
                        "ERROR": "Error during search"
                    }
                    st.markdown(f"**Search Tier Used:** {tier_display.get(tier_used, tier_used)}")

                if total_before_filter > 0:
                    st.markdown(f"**Papers found:** {total_before_filter} total, {filtered_count} filtered out")

                if queries_executed:
                    st.markdown("**Queries Executed:**")
                    for tier_name, tier_queries in queries_executed.items():
                        if tier_queries:
                            with st.expander(f"{tier_name} queries ({len(tier_queries)})"):
                                for q in tier_queries:
                                    st.code(q, language="text")

                if not citations:
                    st.warning("""
                    **No relevant literature found.**

                    This can happen when:
                    - The drug concept is novel/hypothetical (no papers discuss it yet)
                    - The combination of concepts is too specific

                    The expert analysis above is based on general scientific knowledge.
                    """)

        # Search query info (legacy)
        if search_query and not queries_executed:
            st.caption(f"Search query: `{search_query}`")

        # Validation summary metrics
        if validations:
            _render_validation_summary(validations)
            st.markdown("---")

        # Citation list
        if citations:
            _render_citation_list(citations)

            # Zotero export button
            if settings.ENABLE_ZOTERO:
                st.markdown("---")
                _render_zotero_export(citations)
        elif not concepts and not queries_executed:
            st.info("No literature citations found for this query.")
            
        # Google Search Grounding Sources
        grounding_sources = evidence_summary.get('grounding_sources', [])
        if grounding_sources:
             st.markdown("---")
             _render_grounding_sources(grounding_sources)

        # Clinical trials list
        if clinical_trials:
            st.markdown("---")
            _render_clinical_trials(clinical_trials)

        # Validation details per expert
        if validations:
            st.markdown("---")
            _render_validation_details(validations)


def _render_validation_summary(validations: Dict):
    """Render aggregate validation metrics."""
    total_supported = 0
    total_contradicted = 0
    total_no_evidence = 0

    for v in validations.values():
        if isinstance(v, dict):
            total_supported += v.get('claims_supported', 0)
            total_contradicted += v.get('claims_contradicted', 0)
            total_no_evidence += v.get('claims_no_evidence', 0)
        elif hasattr(v, 'claims_supported'):
            total_supported += v.claims_supported
            total_contradicted += v.claims_contradicted
            total_no_evidence += v.claims_no_evidence

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Supported",
            total_supported,
            help="Claims supported by literature evidence"
        )
    with col2:
        delta_color = "inverse" if total_contradicted > 0 else "off"
        st.metric(
            "Contradicted",
            total_contradicted,
            delta=f"-{total_contradicted}" if total_contradicted > 0 else None,
            delta_color=delta_color,
            help="Claims contradicted by literature"
        )
    with col3:
        st.metric(
            "Unverified",
            total_no_evidence,
            help="Claims with no supporting/contradicting evidence found"
        )


def _render_citation_list(citations, max_display: int = 10):
    """Render the list of citations."""
    st.subheader("Literature Citations")

    # Handle dict format (convert to list)
    if isinstance(citations, dict):
        citations = list(citations.values())

    # Ensure we have a list
    if not isinstance(citations, list):
        st.warning("Unable to display citations in unexpected format.")
        return

    # Flatten if nested list (e.g., [[citation1], [citation2]])
    if citations and isinstance(citations[0], list):
        citations = [c for sublist in citations for c in sublist if c]

    displayed = 0
    for citation in citations:
        if displayed >= max_display:
            break

        # Handle Citation objects (with attributes)
        if hasattr(citation, 'pmid'):
            pmid = citation.pmid
            title = citation.title
            year = citation.year
            authors = citation.authors
            abstract = citation.abstract
            journal = getattr(citation, 'journal', '')
        # Handle dict objects
        elif isinstance(citation, dict):
            pmid = citation.get('pmid', 'N/A')
            title = citation.get('title', 'Untitled')
            year = citation.get('year', 'N/A')
            authors = citation.get('authors', [])
            abstract = citation.get('abstract', '')
            journal = citation.get('journal', '')
        # Skip non-citation items (lists, strings, etc.)
        else:
            continue

        displayed += 1

        # Format authors
        if authors:
            if len(authors) > 3:
                author_str = f"{authors[0]} et al."
            else:
                author_str = ", ".join(authors)
        else:
            author_str = "Unknown authors"

        # Citation card
        with st.container():
            st.markdown(f"**[{displayed}] {title}**")
            st.caption(f"{author_str} | {journal} ({year}) | PMID: {pmid}")

            # Show abstract on toggle
            if abstract:
                with st.expander("Show abstract", expanded=False):
                    st.write(abstract)

    # Show remaining count
    if len(citations) > max_display:
        st.caption(f"... and {len(citations) - max_display} more papers")


def _render_clinical_trials(trials: List, max_display: int = 5):
    """Render the list of clinical trials."""
    st.subheader("Clinical Trials")

    displayed = 0
    for trial in trials:
        if displayed >= max_display:
            break

        # Handle ClinicalTrial objects (with attributes)
        if hasattr(trial, 'nct_id'):
            nct_id = trial.nct_id
            title = trial.title
            phase = trial.phase
            status = trial.status
            sponsor = trial.sponsor
            conditions = trial.conditions
            interventions = trial.interventions
            url = trial.url
        # Handle dict objects
        elif isinstance(trial, dict):
            nct_id = trial.get('nct_id', 'N/A')
            title = trial.get('title', 'Untitled')
            phase = trial.get('phase', 'N/A')
            status = trial.get('status', 'N/A')
            sponsor = trial.get('sponsor', 'N/A')
            conditions = trial.get('conditions', [])
            interventions = trial.get('interventions', [])
            url = trial.get('url', f"https://clinicaltrials.gov/study/{nct_id}")
        else:
            continue

        displayed += 1

        # Status badge color
        status_colors = {
            "Recruiting": "#28a745",
            "Active, not recruiting": "#17a2b8",
            "Completed": "#6c757d",
            "Terminated": "#dc3545"
        }
        status_color = status_colors.get(status, "#6c757d")

        # Trial card
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**🏥 [{displayed}] {title[:80]}{'...' if len(title) > 80 else ''}**")
            with col2:
                st.markdown(f"<span style='background: {status_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;'>{phase}</span>", unsafe_allow_html=True)

            # Details
            details = f"**{nct_id}** | {sponsor}"
            if conditions:
                details += f" | {', '.join(conditions[:2])}"
            st.caption(details)

            # Expandable details
            with st.expander("Details", expanded=False):
                st.markdown(f"**Status:** {status}")
                if interventions:
                    st.markdown(f"**Interventions:** {', '.join(interventions[:3])}")
                st.markdown(f"[View on ClinicalTrials.gov]({url})")

    # Show remaining count
    if len(trials) > max_display:
        st.caption(f"... and {len(trials) - max_display} more trials")


def _render_grounding_sources(sources: List[Dict], title: str = "Web Sources (Grounding)"):
    """Render Google Search grounding sources."""
    if not sources:
        return

    st.subheader(f"{title} ({len(sources)})")
    for i, source in enumerate(sources, 1):
        title = source.get('title', 'Web Source')[:80]
        url = source.get('url', '')
        snippet = source.get('snippet', '')
        
        with st.container():
            col1, col2 = st.columns([0.05, 0.95])
            with col1:
                 st.markdown(f"**[{i}]**")
            with col2:
                 st.markdown(f"[{title}]({url})")
                 if snippet:
                     st.caption(snippet[:200] + "...")


def _render_validation_details(validations: Dict):
    """Render per-expert validation details."""
    st.subheader("Expert Validation Details")

    for expert_name, validation in validations.items():
        # Handle both ValidationResult objects and dicts
        if isinstance(validation, dict):
            validation_text = validation.get('validation_text', '')
            claims_supported = validation.get('claims_supported', 0)
            claims_contradicted = validation.get('claims_contradicted', 0)
        elif hasattr(validation, 'validation_text'):
            validation_text = validation.validation_text
            claims_supported = validation.claims_supported
            claims_contradicted = validation.claims_contradicted
        else:
            continue

        # Skip empty validations
        if not validation_text or validation_text.startswith("No valid"):
            continue

        # Summary badge
        badge_color = "#28a745" if claims_contradicted == 0 else "#ffc107"
        badge_text = f"{claims_supported} supported"
        if claims_contradicted > 0:
            badge_text += f", {claims_contradicted} contradicted"

        with st.expander(f"{expert_name} - Validation ({badge_text})"):
            st.markdown(validation_text)


def render_citation_card(citation, index: int = 0):
    """
    Render a single citation card.

    Args:
        citation: Citation object or dict
        index: Display index
    """
    if hasattr(citation, 'pmid'):
        pmid = citation.pmid
        title = citation.title
        year = citation.year
        authors = citation.authors[:3] if citation.authors else []
        abstract = citation.abstract
    else:
        pmid = citation.get('pmid', 'N/A')
        title = citation.get('title', 'Untitled')
        year = citation.get('year', 'N/A')
        authors = citation.get('authors', [])[:3]
        abstract = citation.get('abstract', '')

    author_str = ", ".join(authors) if authors else "Unknown"

    st.markdown(f"""
    <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem;">
        <div style="font-weight: bold; margin-bottom: 0.25rem;">[{index}] {title}</div>
        <div style="color: #666; font-size: 0.85rem;">
            {author_str} | {year} | PMID: {pmid}
        </div>
    </div>
    """, unsafe_allow_html=True)

    return abstract


def _render_zotero_export(citations):
    """Render Zotero export button within evidence drawer."""
    from core.zotero_client import ZoteroClient
    from core.pubmed_client import Citation

    col1, col2 = st.columns([3, 1])
    with col1:
        collection_name = st.text_input(
            "Zotero collection",
            value=f"Research Partner - {datetime.now().strftime('%Y-%m-%d')}",
            key="evidence_zotero_collection",
            label_visibility="collapsed",
            placeholder="Collection name..."
        )
    with col2:
        upload_clicked = st.button("📚 Save to Zotero", key="evidence_zotero_btn", use_container_width=True)

    if upload_clicked:
        try:
            client = ZoteroClient(
                api_key=settings.ZOTERO_API_KEY,
                user_id=settings.ZOTERO_USER_ID
            )

            if not client.test_connection():
                st.error("Failed to connect to Zotero. Check your API key and user ID.")
                return

            with st.spinner("Creating collection..."):
                collection = client.create_collection(collection_name)
                collection_key = collection.get('key')

            # Convert citations to Citation objects
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
                result = client.add_citations(citation_objects, collection_key=collection_key)

            st.success(f"✅ Uploaded {result['successful']}/{result['total']} citations to '{collection_name}'")
            if result['failed'] > 0:
                st.warning(f"⚠️ {result['failed']} citations failed")

        except Exception as e:
            st.error(f"Zotero upload failed: {str(e)}")


def render_evidence_summary_inline(evidence_summary: Dict):
    """
    Render a compact inline evidence summary.

    Args:
        evidence_summary: Dict with citations, search_query, paper_count
    """
    paper_count = evidence_summary.get('paper_count', 0)
    search_query = evidence_summary.get('search_query', '')

    if paper_count > 0:
        st.markdown(f"*Based on {paper_count} papers*")
        if search_query:
            st.caption(f"Query: {search_query}")
    else:
        st.caption("No literature evidence retrieved")
