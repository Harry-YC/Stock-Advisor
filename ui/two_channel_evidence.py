"""
Two-Channel Evidence Display Component

Displays search results organized by channel:
- Clinical Channel: Trial results, SoC, competitive landscape
- Biology Channel: Expression, mechanism, safety data

Shows claim validation status with channel-appropriate evidence.
"""

import streamlit as st
from typing import Dict, List, Optional, Union

from services.two_channel_search_service import TwoChannelSearchResult, ChannelResult
from services.claim_validator import ValidationResult, ValidatedClaim, ValidationStatus, ClaimType
from core.pubmed_client import Citation


def render_two_channel_evidence(
    search_result: Union[TwoChannelSearchResult, Dict],
    validation_result: Optional[Union[ValidationResult, Dict]] = None,
    expanded: bool = False
):
    """
    Render two-channel evidence drawer with optional validation.

    Args:
        search_result: TwoChannelSearchResult or dict
        validation_result: Optional ValidationResult from claim validation
        expanded: Whether to expand the drawer by default
    """
    # Handle dict input
    if isinstance(search_result, dict):
        clinical_data = search_result.get('clinical', {})
        biology_data = search_result.get('biology', {})
        concepts = search_result.get('concepts', {})
        trials = search_result.get('trials', [])
    else:
        clinical_data = search_result.clinical.to_dict() if search_result.clinical else {}
        biology_data = search_result.biology.to_dict() if search_result.biology else {}
        concepts = search_result.concepts.to_dict() if search_result.concepts else {}
        trials = search_result.trials

    # Build title
    clinical_count = clinical_data.get('citation_count', len(clinical_data.get('citations', [])))
    biology_count = biology_data.get('citation_count', len(biology_data.get('citations', [])))
    trial_count = len(trials)

    title_parts = []
    if clinical_count > 0:
        title_parts.append(f"{clinical_count} clinical")
    if biology_count > 0:
        title_parts.append(f"{biology_count} biology")
    if trial_count > 0:
        title_parts.append(f"{trial_count} trials")

    title = "Evidence (" + ", ".join(title_parts) + ")" if title_parts else "Evidence"

    with st.expander(title, expanded=expanded):
        # Validation summary (if available)
        if validation_result:
            _render_validation_summary(validation_result)
            st.markdown("---")

        # Search concepts
        if concepts:
            _render_concepts(concepts)
            st.markdown("---")

        # Channel tabs
        tab_clinical, tab_biology, tab_trials = st.tabs([
            f"Clinical ({clinical_count})",
            f"Biology ({biology_count})",
            f"Trials ({trial_count})"
        ])

        with tab_clinical:
            _render_channel_evidence(
                clinical_data,
                "Clinical",
                description="Phase 2/3 trials, standard of care, competitive landscape"
            )

        with tab_biology:
            _render_channel_evidence(
                biology_data,
                "Biology",
                description="Expression studies, mechanism, biomarker validation"
            )

        with tab_trials:
            _render_trials(trials)


def _render_validation_summary(validation: Union[ValidationResult, Dict]):
    """Render claim validation summary."""
    if isinstance(validation, dict):
        supported = validation.get('claims_supported', 0)
        contradicted = validation.get('claims_contradicted', 0)
        no_evidence = validation.get('claims_no_evidence', 0)
        total = validation.get('total_claims', 0)
    else:
        supported = validation.claims_supported
        contradicted = validation.claims_contradicted
        no_evidence = validation.claims_no_evidence
        total = validation.total_claims

    if total == 0:
        return

    st.markdown("**Claim Validation**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Supported",
            supported,
            help="Claims supported by literature evidence"
        )

    with col2:
        delta_color = "inverse" if contradicted > 0 else "off"
        st.metric(
            "Contradicted",
            contradicted,
            delta=f"-{contradicted}" if contradicted > 0 else None,
            delta_color=delta_color,
            help="Claims contradicted by literature"
        )

    with col3:
        st.metric(
            "Unverified",
            no_evidence,
            help="Claims with no supporting/contradicting evidence found"
        )


def _render_concepts(concepts: Dict):
    """Render extracted concepts."""
    with st.expander("Extracted Concepts", expanded=False):
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


def _render_channel_evidence(
    channel_data: Dict,
    channel_name: str,
    description: str = "",
    max_display: int = 10
):
    """Render evidence for a single channel."""
    citations = channel_data.get('citations', [])
    queries = channel_data.get('queries_executed', [])
    purposes = channel_data.get('query_purposes', [])

    if description:
        st.caption(description)

    # Show queries executed
    if purposes:
        with st.expander(f"Queries ({len(queries)})", expanded=False):
            for i, purpose in enumerate(purposes[:6]):
                st.markdown(f"- {purpose}")
            if queries:
                st.markdown("---")
                st.markdown("**Raw queries:**")
                for q in queries[:3]:
                    st.code(q[:100] + "..." if len(q) > 100 else q, language="text")

    # Show citations
    if not citations:
        st.info(f"No {channel_name.lower()} evidence found for this query.")
        return

    st.markdown(f"**{len(citations)} papers found:**")

    displayed = 0
    for citation in citations:
        if displayed >= max_display:
            break

        # Handle dict or Citation object
        if isinstance(citation, dict):
            pmid = citation.get('pmid', 'N/A')
            title = citation.get('title', 'Untitled')
            year = citation.get('year', 'N/A')
            authors = citation.get('authors', [])
            abstract = citation.get('abstract', '')
            journal = citation.get('journal', '')
        elif hasattr(citation, 'pmid'):
            pmid = citation.pmid
            title = citation.title
            year = citation.year
            authors = citation.authors
            abstract = citation.abstract
            journal = getattr(citation, 'journal', '')
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

            if abstract:
                with st.expander("Show abstract", expanded=False):
                    st.write(abstract)

    if len(citations) > max_display:
        st.caption(f"... and {len(citations) - max_display} more papers")


def _render_trials(trials: List[Dict], max_display: int = 5):
    """Render clinical trials list."""
    if not trials:
        st.info("No clinical trials found for this query.")
        return

    displayed = 0
    for trial in trials:
        if displayed >= max_display:
            break

        nct_id = trial.get('nct_id', 'N/A')
        title = trial.get('title', 'Untitled')
        phase = trial.get('phase', [])
        status = trial.get('status', 'N/A')
        sponsor = trial.get('sponsor', 'N/A')
        url = trial.get('url', f"https://clinicaltrials.gov/study/{nct_id}")

        displayed += 1

        # Phase formatting
        if isinstance(phase, list):
            phase_str = ", ".join(phase) if phase else "N/A"
        else:
            phase_str = str(phase)

        # Status badge color
        status_colors = {
            "Recruiting": "#28a745",
            "Active, not recruiting": "#17a2b8",
            "Completed": "#6c757d",
            "Terminated": "#dc3545"
        }
        status_color = status_colors.get(status, "#6c757d")

        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                title_display = title[:80] + '...' if len(title) > 80 else title
                st.markdown(f"**[{displayed}] {title_display}**")
            with col2:
                st.markdown(
                    f"<span style='background: {status_color}; color: white; "
                    f"padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;'>"
                    f"{phase_str}</span>",
                    unsafe_allow_html=True
                )

            st.caption(f"**{nct_id}** | {sponsor} | Status: {status}")

            with st.expander("Details", expanded=False):
                st.markdown(f"[View on ClinicalTrials.gov]({url})")

    if len(trials) > max_display:
        st.caption(f"... and {len(trials) - max_display} more trials")


def render_validated_claims(
    validation_result: Union[ValidationResult, Dict],
    show_citations: bool = True
):
    """
    Render detailed claim validation results.

    Args:
        validation_result: ValidationResult or dict
        show_citations: Whether to show supporting citations
    """
    if isinstance(validation_result, dict):
        claims = validation_result.get('claims', [])
    else:
        claims = [c.to_dict() for c in validation_result.claims]

    if not claims:
        st.info("No verifiable claims identified in response.")
        return

    st.markdown("**Claim-by-Claim Validation**")

    for i, claim in enumerate(claims):
        claim_text = claim.get('claim_text', '')
        claim_type = claim.get('claim_type', 'unknown')
        status = claim.get('status', 'no_evidence')
        confidence = claim.get('confidence', 0)
        supporting = claim.get('supporting_pmids', [])
        contradicting = claim.get('contradicting_pmids', [])

        # Status icon
        status_icons = {
            'supported': '✅',
            'contradicted': '⚠️',
            'partial': '🔶',
            'no_evidence': '❓'
        }
        icon = status_icons.get(status, '❓')

        # Claim type badge
        type_colors = {
            'clinical': '#0066cc',
            'biology': '#28a745',
            'safety': '#dc3545'
        }
        type_color = type_colors.get(claim_type, '#6c757d')

        with st.container():
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown(f"### {icon}")
            with col2:
                st.markdown(
                    f"<span style='background: {type_color}; color: white; "
                    f"padding: 2px 6px; border-radius: 3px; font-size: 0.7rem;'>"
                    f"{claim_type.upper()}</span>",
                    unsafe_allow_html=True
                )
                st.markdown(f'"{claim_text[:150]}{"..." if len(claim_text) > 150 else ""}"')

                if show_citations and (supporting or contradicting):
                    if supporting:
                        st.caption(f"Supporting: PMID {', '.join(supporting[:3])}")
                    if contradicting:
                        st.caption(f"Contradicting: PMID {', '.join(contradicting[:2])}")

            st.markdown("---")


def render_channel_comparison(search_result: Union[TwoChannelSearchResult, Dict]):
    """
    Render side-by-side channel comparison.

    Useful for seeing what type of evidence was found vs. what's missing.
    """
    if isinstance(search_result, dict):
        clinical = search_result.get('clinical', {})
        biology = search_result.get('biology', {})
    else:
        clinical = search_result.clinical.to_dict()
        biology = search_result.biology.to_dict()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Clinical Channel")
        clinical_count = clinical.get('citation_count', 0)
        if clinical_count > 0:
            st.success(f"Found {clinical_count} clinical papers")
            purposes = clinical.get('query_purposes', [])
            for p in purposes[:3]:
                st.markdown(f"- {p}")
        else:
            st.warning("No clinical evidence found")
            st.caption("Consider: Is this a novel target with no trials yet?")

    with col2:
        st.markdown("### Biology Channel")
        biology_count = biology.get('citation_count', 0)
        if biology_count > 0:
            st.success(f"Found {biology_count} biology papers")
            purposes = biology.get('query_purposes', [])
            for p in purposes[:3]:
                st.markdown(f"- {p}")
        else:
            st.warning("No biology evidence found")
            st.caption("Consider: Are target/mechanism terms correct?")


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

def render_evidence_with_validation(
    search_result: Union[TwoChannelSearchResult, Dict],
    expert_response: str = None,
    api_key: str = None,
    expanded: bool = False
):
    """
    Render evidence drawer with automatic claim validation.

    Args:
        search_result: TwoChannelSearchResult or dict
        expert_response: Optional expert response to validate
        api_key: API key for validation
        expanded: Whether drawer is expanded
    """
    validation_result = None

    if expert_response and search_result:
        from services.claim_validator import ClaimValidator

        try:
            validator = ClaimValidator(api_key=api_key)
            validation_result = validator.validate_claims(
                expert_response,
                search_result
            )
        except Exception as e:
            import logging
            logging.warning(f"Claim validation failed: {e}")

    render_two_channel_evidence(
        search_result,
        validation_result=validation_result,
        expanded=expanded
    )
