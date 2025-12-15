"""
Answer View Component

Displays the synthesized recommendation with progressive disclosure:
- Top-line recommendation (always visible)
- Confidence indicator (two-axis: consensus + evidence)
- Key findings (expandable)
- Claim ledger (evidence transparency)
- Expert perspectives summary (delta-only when possible)
- Guideline integration (add to Guideline Workspace)
"""

import re
import streamlit as st
from datetime import datetime
from typing import Union, Dict, Optional, List
from dataclasses import dataclass

from core.question_templates import QUESTION_TYPES, get_cdp_section_name
from services.research_partner_service import ResearchResult
from core.knowledge_store import add_program_conclusion
from ui.citation_utils import (
    format_expert_response,
    highlight_inline_citations,
    render_citation_cards,
    render_expert_response_with_citations
)
from ui.mark_pen import render_mark_button, render_marks_panel

# Import claim extraction and supporting literature services
try:
    from core.claims import extract_claims_from_responses, get_searchable_claims
    from services.supporting_literature_service import (
        SupportingPaper,
        LiteratureSearchResult,
        search_supporting_literature,
        determine_certainty
    )
    CLAIMS_SUPPORT_AVAILABLE = True
except ImportError:
    CLAIMS_SUPPORT_AVAILABLE = False


# =============================================================================
# TWO-AXIS CONFIDENCE SCORING
# =============================================================================

@dataclass
class ConfidenceScore:
    """Two-axis confidence scoring for recommendations."""
    consensus: float  # 0-1: proportion of experts agreeing
    evidence: float   # 0-1: proportion of claims with PMID support
    headline: str     # HIGH / MODERATE / LOW
    limiting_factor: str  # What's driving the lower score

    @classmethod
    def compute(cls, expert_responses: Dict, claim_ledger: Optional[Dict] = None) -> 'ConfidenceScore':
        """Compute confidence from expert responses and claim validation."""
        # Consensus: measure agreement (simplified - check for dissent keywords)
        consensus_score = cls._measure_expert_agreement(expert_responses)

        # Evidence: measure citation coverage
        if claim_ledger and claim_ledger.get('total', 0) > 0:
            evidence_score = claim_ledger.get('support_rate', 0.5)
        else:
            # Fallback: count PMIDs in responses
            evidence_score = cls._estimate_evidence_coverage(expert_responses)

        # Headline: based on minimum of both axes
        min_score = min(consensus_score, evidence_score)
        if min_score >= 0.7:
            headline = "HIGH"
        elif min_score >= 0.4:
            headline = "MODERATE"
        else:
            headline = "LOW"

        # Limiting factor
        if consensus_score < evidence_score:
            limiting_factor = f"Expert disagreement ({consensus_score:.0%} consensus)"
        else:
            limiting_factor = f"Evidence gaps ({evidence_score:.0%} claims supported)"

        return cls(
            consensus=consensus_score,
            evidence=evidence_score,
            headline=headline,
            limiting_factor=limiting_factor
        )

    @staticmethod
    def _measure_expert_agreement(expert_responses: Dict) -> float:
        """Estimate consensus from expert responses."""
        if not expert_responses:
            return 0.5

        # Check for dissent indicators
        dissent_keywords = ['disagree', 'caution', 'concern', 'however', 'contrary', 'oppose']
        dissent_count = 0

        for expert, response in expert_responses.items():
            content = response.get('content', '') if isinstance(response, dict) else str(response)
            content_lower = content.lower()
            if any(kw in content_lower for kw in dissent_keywords):
                dissent_count += 1

        total = len(expert_responses)
        if total == 0:
            return 0.5

        agreement_ratio = 1 - (dissent_count / total)
        return max(0.3, agreement_ratio)  # Floor at 0.3

    @staticmethod
    def _estimate_evidence_coverage(expert_responses: Dict) -> float:
        """Estimate evidence coverage from PMID mentions."""
        import re
        total_claims = 0
        pmid_claims = 0

        for expert, response in expert_responses.items():
            content = response.get('content', '') if isinstance(response, dict) else str(response)
            # Count sentences with numbers (potential claims)
            sentences = re.split(r'[.!?]', content)
            for sent in sentences:
                if re.search(r'\d+%|\d+\.\d+', sent):
                    total_claims += 1
                    if re.search(r'PMID[:\s]*\d{7,8}', sent, re.IGNORECASE):
                        pmid_claims += 1

        if total_claims == 0:
            return 0.5  # No quantitative claims, neutral

        return pmid_claims / total_claims


def render_confidence_badge(confidence: ConfidenceScore):
    """Render transparent two-axis confidence indicator."""
    color_map = {"HIGH": "#28a745", "MODERATE": "#ffc107", "LOW": "#dc3545"}
    color = color_map.get(confidence.headline, "#666")

    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 16px; padding: 12px;
                background: #f8f9fa; border-radius: 8px; margin: 16px 0; border-left: 4px solid {color};">
        <div style="font-size: 24px; font-weight: bold; color: {color};">
            {confidence.headline}
        </div>
        <div style="font-size: 14px; color: #666;">
            <div>Consensus: {confidence.consensus:.0%} of experts agree</div>
            <div>Evidence: {confidence.evidence:.0%} of claims supported</div>
            <div style="font-style: italic; color: #999;">Limiting factor: {confidence.limiting_factor}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# GRADE CERTAINTY FOR LOW-EVIDENCE DOMAINS
# =============================================================================

def compute_grade_certainty(
    expert_responses: Dict,
    citations: List,
    validations: Optional[Dict] = None
) -> Dict:
    """
    Compute GRADE certainty rating appropriate for low-evidence domains.

    In palliative surgery:
    - Case series = Very Low certainty (but valid evidence)
    - Retrospective cohorts = Low certainty
    - Expert consensus = appropriate basis when RCTs don't exist

    Returns:
        Dict with grade, basis, evidence_type, and gaps
    """
    # Count evidence types
    case_series_count = 0
    retrospective_count = 0
    prospective_count = 0
    rct_count = 0

    case_series_terms = [
        "case series", "consecutive patients", "institutional experience",
        "single-center", "our experience", "review of patients"
    ]
    retrospective_terms = ["retrospective", "chart review", "medical records"]
    prospective_terms = ["prospective", "cohort study"]
    rct_terms = ["randomized", "rct", "randomised", "controlled trial"]

    for cit in citations:
        if hasattr(cit, 'title'):
            text = f"{cit.title} {getattr(cit, 'abstract', '')}".lower()
        elif isinstance(cit, dict):
            text = f"{cit.get('title', '')} {cit.get('abstract', '')}".lower()
        else:
            continue

        if any(term in text for term in rct_terms):
            rct_count += 1
        elif any(term in text for term in prospective_terms):
            prospective_count += 1
        elif any(term in text for term in retrospective_terms):
            retrospective_count += 1
        elif any(term in text for term in case_series_terms):
            case_series_count += 1

    # Determine consensus level
    expert_count = len(expert_responses)
    agreeing_experts = expert_count  # Assume agreement unless dissent detected
    dissent_keywords = ['disagree', 'recommend against', 'should not', 'oppose']
    for expert, response in expert_responses.items():
        content = response.get('content', '') if isinstance(response, dict) else str(response)
        if any(kw in content.lower() for kw in dissent_keywords):
            agreeing_experts -= 1

    # Determine GRADE certainty
    if rct_count >= 2:
        grade = "Moderate"
        evidence_type = f"{rct_count} RCT(s)"
    elif rct_count == 1 or prospective_count >= 2:
        grade = "Low"
        evidence_type = f"{rct_count} RCT(s), {prospective_count} prospective"
    elif retrospective_count >= 2 or (retrospective_count >= 1 and case_series_count >= 2):
        grade = "Very Low"
        evidence_type = f"{retrospective_count} retrospective, {case_series_count} case series"
    elif case_series_count >= 1:
        grade = "Very Low"
        evidence_type = f"{case_series_count} case series"
    else:
        grade = "Very Low"
        evidence_type = "Expert consensus only"

    # Build basis string
    if agreeing_experts == expert_count:
        consensus_str = f"Expert consensus ({expert_count}/{expert_count} agree)"
    else:
        consensus_str = f"Expert majority ({agreeing_experts}/{expert_count} agree)"

    total_papers = len(citations)
    if total_papers > 0:
        basis = f"{consensus_str} + {total_papers} supporting papers"
    else:
        basis = consensus_str

    # Identify evidence gaps
    gaps = []
    if rct_count == 0:
        gaps.append("No RCTs exist (typical for palliative surgery)")
    if total_papers < 5:
        gaps.append(f"Limited published data ({total_papers} papers)")
    if retrospective_count == 0 and prospective_count == 0:
        gaps.append("No comparative studies available")

    return {
        "grade": grade,
        "basis": basis,
        "evidence_type": evidence_type,
        "gaps": gaps,
        "expert_count": expert_count,
        "agreeing_experts": agreeing_experts,
        "paper_count": total_papers,
        "case_series_count": case_series_count,
        "retrospective_count": retrospective_count,
        "prospective_count": prospective_count,
        "rct_count": rct_count
    }


def render_grade_certainty_panel(grade_info: Dict):
    """
    Render GRADE certainty panel for low-evidence domains.

    Shows:
    - GRADE certainty level with appropriate color
    - Basis for recommendation
    - Evidence gaps as explicit feature (not failure)
    """
    # Color coding for GRADE levels
    grade_colors = {
        "High": "#28a745",
        "Moderate": "#17a2b8",
        "Low": "#ffc107",
        "Very Low": "#6c757d"
    }
    grade = grade_info.get("grade", "Very Low")
    color = grade_colors.get(grade, "#6c757d")

    basis = grade_info.get("basis", "Expert consensus")
    gaps = grade_info.get("gaps", [])

    # Main panel
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 16px; border-radius: 12px; margin: 16px 0;
                border-left: 4px solid {color};">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div style="font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px;">
                    GRADE Certainty
                </div>
                <div style="font-size: 20px; font-weight: bold; color: {color}; margin-top: 4px;">
                    {grade}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px;">
                    Basis
                </div>
                <div style="font-size: 14px; color: #333; margin-top: 4px;">
                    {basis}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Evidence gaps as explicit feature (not warning)
    if gaps:
        with st.expander("📋 Evidence Landscape (What Data Exists)", expanded=False):
            st.markdown("**This is a low-evidence domain.** The following reflects what evidence is available:")

            # Show what EXISTS
            paper_count = grade_info.get("paper_count", 0)
            case_series = grade_info.get("case_series_count", 0)
            retrospective = grade_info.get("retrospective_count", 0)
            rct = grade_info.get("rct_count", 0)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Available Evidence:**")
                if case_series > 0:
                    st.markdown(f"- ✅ {case_series} case series")
                if retrospective > 0:
                    st.markdown(f"- ✅ {retrospective} retrospective studies")
                if rct > 0:
                    st.markdown(f"- ✅ {rct} RCT(s)")
                if paper_count == 0:
                    st.markdown("- Expert consensus (clinical experience)")

            with col2:
                st.markdown("**Evidence Gaps:**")
                for gap in gaps:
                    st.markdown(f"- 📌 {gap}")

            st.caption("In palliative surgery, case series and expert consensus ARE appropriate evidence sources.")


# =============================================================================
# SUPPORTING LITERATURE FOR EXPERT CLAIMS
# =============================================================================

def render_supporting_literature_section(
    literature_result: 'LiteratureSearchResult',
    expert_responses: Dict
):
    """
    Render supporting literature matched to expert claims.

    This displays case series and cohorts that ILLUSTRATE expert claims,
    not "validate" them. Key insight for low-evidence domains.

    Args:
        literature_result: Result from supporting literature search
        expert_responses: Original expert responses (for context)
    """
    if not CLAIMS_SUPPORT_AVAILABLE or not literature_result:
        return

    # Get all papers and claim matches
    claim_matches = literature_result.claim_matches
    total_papers = literature_result.total_papers

    if total_papers == 0:
        st.info("📚 **Supporting Literature**: No case series found. "
                "Recommendation based on expert consensus (appropriate for this domain).")
        return

    # Summary stats
    claims_supported = literature_result.claims_with_support
    claims_total = claims_supported + literature_result.claims_without_support

    # Determine overall certainty
    certainty = determine_certainty(literature_result)

    # Header panel
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                padding: 16px; border-radius: 12px; margin: 16px 0;
                border-left: 4px solid #0284c7;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 12px; color: #0369a1; text-transform: uppercase; letter-spacing: 0.5px;">
                    Supporting Literature
                </div>
                <div style="font-size: 16px; font-weight: 600; color: #0c4a6e; margin-top: 4px;">
                    {total_papers} papers illustrate {claims_supported}/{claims_total} claims
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 12px; color: #0369a1;">GRADE Certainty</div>
                <div style="font-size: 14px; font-weight: 600; color: #0c4a6e;">{certainty}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Study type breakdown
    study_types = literature_result.study_types
    if study_types:
        type_labels = []
        for stype, count in study_types.items():
            label = stype.replace("_", " ").title()
            type_labels.append(f"{count} {label}")
        st.caption(f"📊 Study types: {', '.join(type_labels)}")

    # Expandable claim-by-claim breakdown
    with st.expander(f"📖 View Claims with Supporting Papers ({claims_supported} claims)", expanded=False):
        for claim_text, papers in claim_matches.items():
            if papers:
                render_claim_match_card(claim_text, papers)
            else:
                # Show claims without support
                st.markdown(f"""
                <div style="background: #fef3c7; padding: 12px; border-radius: 8px;
                            margin-bottom: 12px; border-left: 3px solid #f59e0b;">
                    <div style="font-size: 13px; color: #92400e; font-style: italic;">
                        {claim_text[:200]}{'...' if len(claim_text) > 200 else ''}
                    </div>
                    <div style="font-size: 11px; color: #b45309; margin-top: 8px;">
                        📌 No published case series found - based on expert experience
                    </div>
                </div>
                """, unsafe_allow_html=True)


def render_claim_match_card(claim_text: str, papers: List['SupportingPaper']):
    """
    Render a claim with its matched supporting papers.

    Args:
        claim_text: The expert claim
        papers: List of SupportingPaper objects that support this claim
    """
    # Truncate claim for display
    display_claim = claim_text[:250] + "..." if len(claim_text) > 250 else claim_text

    st.markdown(f"""
    <div style="background: #f0fdf4; padding: 12px; border-radius: 8px;
                margin-bottom: 16px; border-left: 3px solid #22c55e;">
        <div style="font-size: 13px; color: #166534; font-weight: 500;">
            {display_claim}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Papers table
    if papers:
        # Create compact table
        table_html = """
        <table style="width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 16px;">
            <thead>
                <tr style="background: #f1f5f9; border-bottom: 2px solid #e2e8f0;">
                    <th style="padding: 8px; text-align: left; color: #475569;">Study</th>
                    <th style="padding: 8px; text-align: center; color: #475569; width: 80px;">Type</th>
                    <th style="padding: 8px; text-align: center; color: #475569; width: 60px;">N</th>
                    <th style="padding: 8px; text-align: center; color: #475569; width: 80px;">Relevance</th>
                </tr>
            </thead>
            <tbody>
        """

        for paper in papers[:5]:  # Max 5 papers per claim
            # Type badge color
            type_colors = {
                "RCT": "#22c55e",
                "systematic_review": "#3b82f6",
                "prospective_cohort": "#8b5cf6",
                "retrospective": "#f59e0b",
                "case_series": "#6b7280",
                "unknown": "#9ca3af"
            }
            type_color = type_colors.get(paper.study_type, "#9ca3af")
            type_label = paper.study_type.replace("_", " ").title()

            # Relevance badge
            rel_colors = {"HIGH": "#22c55e", "MEDIUM": "#f59e0b", "LOW": "#ef4444"}
            rel_color = rel_colors.get(paper.relevance, "#9ca3af")

            # Sample size display
            n_display = str(paper.sample_size) if paper.sample_size else "—"

            # Title with link
            title_short = paper.title[:60] + "..." if len(paper.title) > 60 else paper.title
            pmid_link = f'<a href="https://pubmed.ncbi.nlm.nih.gov/{paper.pmid}/" target="_blank" style="color: #0369a1; text-decoration: none;">{title_short}</a>'

            table_html += f"""
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 8px;">{pmid_link}</td>
                    <td style="padding: 8px; text-align: center;">
                        <span style="background: {type_color}20; color: {type_color}; padding: 2px 6px;
                                     border-radius: 4px; font-size: 10px; font-weight: 500;">
                            {type_label}
                        </span>
                    </td>
                    <td style="padding: 8px; text-align: center; font-weight: 500;">{n_display}</td>
                    <td style="padding: 8px; text-align: center;">
                        <span style="background: {rel_color}20; color: {rel_color}; padding: 2px 6px;
                                     border-radius: 4px; font-size: 10px; font-weight: 500;">
                            {paper.relevance}
                        </span>
                    </td>
                </tr>
            """

        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)


# =============================================================================
# CLAIM LEDGER RENDERING
# =============================================================================

def render_claim_ledger(claim_ledger: Dict):
    """Render transparent claim validation table."""
    if not claim_ledger or claim_ledger.get('total', 0) == 0:
        st.info("No verifiable claims extracted from expert responses.")
        return

    # Summary stats
    supported = claim_ledger.get('supported', 0)
    unclear = claim_ledger.get('unclear', 0)
    contradicted = claim_ledger.get('contradicted', 0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Supported", supported, help="Claims with valid PMID citations")
    col2.metric("Unclear", unclear, help="Claims without citation or unverified PMID")
    col3.metric("Contradicted", contradicted, help="Claims contradicted by literature")

    # Detailed table
    claims = claim_ledger.get('claims', [])
    if claims:
        st.markdown("#### Claim Details")

        for claim in claims[:15]:  # Limit display
            icon = {"SUPPORTED": "", "UNCLEAR": "", "CONTRADICTED": ""}.get(claim.get('status', ''), '')
            status = claim.get('status', 'UNKNOWN')
            claim_text = claim.get('claim', '')[:100]
            citations = claim.get('citations', [])
            reason = claim.get('reason', '')
            expert = claim.get('source_expert', '')

            # Status-specific styling
            if status == "SUPPORTED":
                st.success(f"**{claim_text}**...")
                if citations:
                    st.caption(f"Citations: {', '.join(citations)} | Source: {expert}")
            elif status == "CONTRADICTED":
                st.error(f"**{claim_text}**...")
                if reason:
                    st.caption(f"{reason} | Source: {expert}")
            else:
                st.warning(f"**{claim_text}**...")
                if reason:
                    st.caption(f"{reason} | Source: {expert}")


# =============================================================================
# DELTA-ONLY EXPERT VIEW
# =============================================================================

def _render_delta_expert_view(expert_responses: Dict):
    """
    Render expert responses showing only unique contributions (deltas).

    Reduces redundancy by:
    1. Extracting key points from each expert
    2. Identifying overlapping/consensus points
    3. Showing only unique perspective from each expert
    """
    import re

    # Extract key points from each expert
    expert_points = {}
    for expert, response in expert_responses.items():
        content = response.get('content', '') if isinstance(response, dict) else str(response)
        points = _extract_key_points(content)
        expert_points[expert] = points

    # Find common themes (mentioned by 3+ experts)
    all_points = []
    for points in expert_points.values():
        all_points.extend(points)

    # Group similar points
    consensus_themes = _identify_consensus_themes(all_points, threshold=2)

    # Display consensus first
    if consensus_themes:
        st.markdown("### 🤝 Consensus Points")
        st.caption("Themes shared by multiple experts")
        for theme in consensus_themes[:5]:  # Top 5 consensus themes
            st.markdown(f"- {theme}")
        st.markdown("---")

    # Display unique contributions per expert
    st.markdown("### 🎯 Unique Expert Contributions")
    st.caption("Points distinctive to each expert's perspective")

    for expert, points in expert_points.items():
        unique_points = _get_unique_points(points, consensus_themes)

        if unique_points:
            with st.container():
                st.markdown(f"**{expert}**")
                for point in unique_points[:3]:  # Top 3 unique points per expert
                    st.markdown(f"- {point}")
                st.markdown("")


def _extract_key_points(text: str) -> List[str]:
    """Extract key points/sentences from expert response text."""
    import re

    points = []

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sent in sentences:
        sent = sent.strip()
        # Skip very short or very long sentences
        if len(sent) < 30 or len(sent) > 300:
            continue
        # Skip questions
        if '?' in sent:
            continue
        # Skip meta-commentary
        if any(skip in sent.lower() for skip in ['i recommend', 'in summary', 'in conclusion', 'overall']):
            continue

        # Prioritize sentences with:
        # - Numbers/percentages (quantitative claims)
        # - Epistemic tags
        # - Keywords like "should", "must", "key", "critical", "important"
        priority = 0
        if re.search(r'\d+%|\d+\.\d+', sent):
            priority += 2
        if re.search(r'EVIDENCE|PMID|ASSUMPTION|OPINION', sent, re.IGNORECASE):
            priority += 1
        if re.search(r'should|must|key|critical|important|essential', sent, re.IGNORECASE):
            priority += 1

        if priority > 0:
            points.append(sent)

    return points[:10]  # Limit to 10 points per expert


def _identify_consensus_themes(all_points: List[str], threshold: int = 2) -> List[str]:
    """Identify themes that appear across multiple experts."""
    import re

    # Extract key terms from each point
    term_counts = {}
    term_to_points = {}

    for point in all_points:
        # Extract significant terms (capitalized words, medical terms)
        terms = set()
        for word in point.split():
            clean = re.sub(r'[^\w]', '', word)
            if len(clean) >= 4:
                lower = clean.lower()
                # Skip common words
                if lower not in ['with', 'that', 'this', 'from', 'have', 'been', 'would', 'should', 'could']:
                    terms.add(lower)

        for term in terms:
            term_counts[term] = term_counts.get(term, 0) + 1
            if term not in term_to_points:
                term_to_points[term] = []
            term_to_points[term].append(point)

    # Find terms appearing >= threshold times
    consensus_terms = [t for t, c in term_counts.items() if c >= threshold]

    # For each consensus term, pick the best representative point
    consensus_themes = []
    used_points = set()

    for term in sorted(consensus_terms, key=lambda t: term_counts[t], reverse=True):
        for point in term_to_points[term]:
            if point not in used_points:
                # Shorten if needed
                if len(point) > 300:
                    point = point[:300] + "..."
                consensus_themes.append(point)
                used_points.add(point)
                break

    return consensus_themes[:5]


def _get_unique_points(expert_points: List[str], consensus_themes: List[str]) -> List[str]:
    """Get points from an expert that are NOT in the consensus themes."""
    unique = []

    # Create a set of normalized consensus content for comparison
    consensus_lower = set()
    for theme in consensus_themes:
        # Extract key content words
        words = set(w.lower() for w in theme.split() if len(w) >= 4)
        consensus_lower.update(words)

    for point in expert_points:
        point_words = set(w.lower() for w in point.split() if len(w) >= 4)

        # Check overlap with consensus
        overlap = len(point_words & consensus_lower) / max(len(point_words), 1)

        # If less than 50% overlap, consider unique
        if overlap < 0.5:
            # Shorten if needed
            if len(point) > 300:
                point = point[:300] + "..."
            unique.append(point)

    return unique[:5]


def add_to_cdp_workspace(
    result: Dict,
    question_type: str,
    section_name: Optional[str] = None
) -> bool:
    """
    Add a research result to the CDP workspace.

    Args:
        result: Dict with research result data
        question_type: The question type (e.g., "surgical_candidate", "general")
        section_name: Optional custom section name (uses default if not provided)

    Returns:
        True if successfully added, False otherwise
    """
    try:
        # Get or create CDP sections dict
        if 'cdp_sections' not in st.session_state:
            st.session_state.cdp_sections = {}

        # Determine section name
        if not section_name:
            section_name = get_cdp_section_name(question_type)

        # Create section key (slugified section name)
        section_key = section_name.lower().replace(" ", "_")

        # Extract citations from evidence summary
        citations = []
        evidence_summary = result.get('evidence_summary', {})
        if evidence_summary:
            raw_citations = evidence_summary.get('citations', [])
            for cit in raw_citations[:10]:  # Limit to 10 citations
                if hasattr(cit, 'pmid'):
                    citations.append({
                        'pmid': cit.pmid,
                        'title': cit.title,
                        'authors': cit.authors[:3] if cit.authors else []
                    })
                elif isinstance(cit, dict):
                    citations.append({
                        'pmid': cit.get('pmid', ''),
                        'title': cit.get('title', ''),
                        'authors': cit.get('authors', [])[:3]
                    })

        # Build CDP section entry
        cdp_section = {
            'section_key': section_key,
            'title': section_name,
            'content': result.get('recommendation', ''),
            'source_question': result.get('question', ''),
            'question_type': question_type,
            'confidence': result.get('confidence', 'MEDIUM'),
            'key_findings': result.get('key_findings', []),
            'expert_contributors': list(result.get('expert_responses', {}).keys()),
            'citations': citations,
            'status': 'draft',
            'created_at': datetime.now().isoformat(),
            'last_edited': datetime.now().isoformat()
        }

        # Add to CDP sections
        st.session_state.cdp_sections[section_key] = cdp_section

        # Update CDP metadata
        st.session_state.cdp_last_modified = datetime.now().isoformat()

        # Persist to Database immediately
        if st.session_state.get('current_project_id') and st.session_state.get('cdp_dao'):
            st.session_state.cdp_dao.save_cdp(
                st.session_state.current_project_id,
                st.session_state.cdp_sections
            )

        return True

    except Exception as e:
        import logging
        logging.error(f"Failed to add to CDP workspace: {e}")
        return False


def render_answer_view(result: Union[ResearchResult, Dict]):
    """
    Render the synthesized answer view.

    Args:
        result: ResearchResult object or dict with result data
    """
    # Handle both ResearchResult and dict
    if isinstance(result, dict):
        question = result.get('question', '')
        question_type = result.get('question_type', 'general')
        recommendation = result.get('recommendation', '')
        confidence = result.get('confidence', 'MEDIUM')
        response_mode = result.get('response_mode', 'expert_consensus')
        key_findings = result.get('key_findings', [])
        expert_responses = result.get('expert_responses', {})
        dissenting_views = result.get('dissenting_views', [])
        metadata = result.get('metadata', {})
    else:
        question = result.question
        question_type = result.question_type
        recommendation = result.recommendation
        confidence = result.confidence
        response_mode = getattr(result, 'response_mode', 'expert_consensus')
        key_findings = result.key_findings
        expert_responses = result.expert_responses
        dissenting_views = result.dissenting_views
        metadata = result.metadata

    # Get question type info
    type_info = QUESTION_TYPES.get(question_type, {})
    type_icon = type_info.get('icon', '')
    type_name = type_info.get('name', 'Research Question')

    # Header with question type badge
    st.markdown(f"""
    <div style="display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1rem;">
        <span style="font-size: 2rem;">{type_icon}</span>
        <div style="flex: 1;">
            <span style="background: #e3f2fd; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.85rem; color: #1565c0;">
                {type_name}
            </span>
            <h3 style="margin: 0.5rem 0 0 0; font-size: 1.25rem; line-height: 1.4;">{question}</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Response Mode badge - Always Expert-First now
    mode_text = "Expert-First" if response_mode == "expert_first" else "Expert Consensus"
    mode_color = "#6366F1"  # Consistent color for expert-driven approach
    mode_icon = "🎯"  # Target icon for Expert-First approach
    st.markdown(f"""
    <div style="display: inline-flex; align-items: center; gap: 6px; background: {mode_color}15;
                padding: 4px 12px; border-radius: 16px; border: 1px solid {mode_color}40; margin-bottom: 12px;">
        <span>{mode_icon}</span>
        <span style="font-size: 0.85rem; color: {mode_color}; font-weight: 500;">{mode_text}</span>
    </div>
    """, unsafe_allow_html=True)

    # Two-axis confidence scoring
    # Build claim ledger for evidence scoring
    claim_ledger_data = None
    try:
        from services.claim_validator import build_claim_ledger
        evidence_summary = result.get('evidence_summary', {}) if isinstance(result, dict) else getattr(result, 'evidence_summary', {})
        citations = evidence_summary.get('citations', []) if evidence_summary else []
        ledger = build_claim_ledger(expert_responses, citations)
        claim_ledger_data = ledger.to_dict()
    except Exception:
        pass

    # Compute two-axis confidence
    confidence_obj = ConfidenceScore.compute(expert_responses, claim_ledger_data)

    # Get citations for GRADE certainty
    evidence_summary_for_grade = result.get('evidence_summary', {}) if isinstance(result, dict) else getattr(result, 'evidence_summary', {})
    citations_for_grade = evidence_summary_for_grade.get('citations', []) if evidence_summary_for_grade else []

    # Compute GRADE certainty (for low-evidence domain)
    grade_info = compute_grade_certainty(expert_responses, citations_for_grade)

    # Metadata row with two-axis confidence
    elapsed = metadata.get('elapsed_seconds', 0)
    experts_count = len(expert_responses)

    # Render the two-axis confidence badge (pass full object, not just headline)
    render_confidence_badge(confidence_obj)

    # Render GRADE certainty panel (low-evidence domain feature)
    render_grade_certainty_panel(grade_info)

    # Supporting Literature Section (Expert-First Flow)
    # This shows case series matched to expert claims
    supporting_literature = result.get('supporting_literature', None) if isinstance(result, dict) else getattr(result, 'supporting_literature', None)
    if supporting_literature and CLAIMS_SUPPORT_AVAILABLE:
        render_supporting_literature_section(supporting_literature, expert_responses)

    # Additional metadata
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background: #f5f5f5; padding: 0.5rem 1rem; border-radius: 0.5rem; text-align: center;">
            <div style="font-size: 0.75rem; color: #666;">Experts Consulted</div>
            <div style="font-size: 1.25rem; font-weight: bold;">{experts_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background: #f5f5f5; padding: 0.5rem 1rem; border-radius: 0.5rem; text-align: center;">
            <div style="font-size: 0.75rem; color: #666;">Analysis Time</div>
            <div style="font-size: 1.25rem; font-weight: bold;">{elapsed:.0f}s</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")  # Spacing

    # Main recommendation box with citation highlighting
    highlighted_recommendation = highlight_inline_citations(recommendation)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;">
        <h4 style="margin: 0 0 0.75rem 0; color: white; font-size: 1rem; opacity: 0.9;">Recommendation</h4>
        <div style="font-size: 1rem; line-height: 1.6;">{highlighted_recommendation}</div>
    </div>
    """, unsafe_allow_html=True)

    # Literature Validation Status (Two-Pass)
    validations = result.get('validations', {}) if isinstance(result, dict) else getattr(result, 'validations', {})
    if validations:
        total_supported = 0
        total_contradicted = 0
        total_no_evidence = 0

        for val in validations.values():
            if hasattr(val, 'claims_supported'):
                total_supported += val.claims_supported
                total_contradicted += val.claims_contradicted
                total_no_evidence += val.claims_no_evidence
            elif isinstance(val, dict):
                total_supported += val.get('claims_supported', 0)
                total_contradicted += val.get('claims_contradicted', 0)
                total_no_evidence += val.get('claims_no_evidence', 0)

        # Display validation summary
        if total_contradicted > 0:
            st.warning(f"⚠️ **Evidence Update**: {total_contradicted} claim(s) contradicted by literature. {total_supported} claim(s) supported.")
        elif total_supported > 0:
            st.success(f"✅ **Literature Validated**: {total_supported} claim(s) supported by evidence. {total_no_evidence} claim(s) could not be verified.")
        elif total_no_evidence > 0:
            st.info(f"📚 **Validation Complete**: {total_no_evidence} claim(s) could not be verified against available literature.")

    # Google Search Grounding Sources
    evidence_summary = result.get('evidence_summary', {}) if isinstance(result, dict) else getattr(result, 'evidence_summary', {})
    grounding_sources = result.get('grounding_sources', []) if isinstance(result, dict) else getattr(result, 'grounding_sources', [])
    # Also check evidence summary for grounding sources (as moved there in service)
    if not grounding_sources and 'grounding_sources' in evidence_summary:
        grounding_sources = evidence_summary['grounding_sources']
        
    if grounding_sources:
        from ui.evidence_drawer import _render_grounding_sources
        with st.expander(f"Web Sources (Grounding) ({len(grounding_sources)})", expanded=False):
            _render_grounding_sources(grounding_sources, title="")

    # Quality Report Warnings (Two-Channel Search)
    evidence_summary = result.get('evidence_summary', {}) if isinstance(result, dict) else getattr(result, 'evidence_summary', {})
    quality_report = evidence_summary.get('quality_report', {}) if evidence_summary else {}
    quality_warnings = quality_report.get('warnings', []) if quality_report else evidence_summary.get('warnings', [])
    target_coverage = quality_report.get('target_coverage', {}) if quality_report else evidence_summary.get('target_coverage', {})

    # Quality Report - Reframed for low-evidence domain
    if quality_warnings:
        with st.expander("📋 Search Quality Report", expanded=False):
            st.markdown("**Literature Search Context** (Low-Evidence Domain)")
            for warning in quality_warnings:
                # Reframe warnings as informational for low-evidence domains
                if "LOW-EVIDENCE DOMAIN" in warning or "typical for this field" in warning:
                    st.info(warning)
                elif "Supporting literature" in warning:
                    st.info(warning)
                else:
                    st.warning(warning)
            st.caption("In palliative surgery, limited literature is expected. Expert consensus provides appropriate guidance.")

    # Target coverage display - reframed as "Supporting Literature"
    if target_coverage:
        with st.expander("📊 Supporting Literature Coverage", expanded=False):
            st.markdown("**Papers found for key concepts** (not validation, supporting examples)")
            for target, count in target_coverage.items():
                if count > 0:
                    st.success(f"✅ **{target}**: {count} supporting papers")
                else:
                    st.info(f"📌 **{target}**: No published data - relies on expert experience")

    # Action buttons - Export and Add to Guideline
    cdp_section_name = get_cdp_section_name(question_type)

    col_export, col_guideline = st.columns(2)

    # Export Report button
    with col_export:
        try:
            from core.report_export import generate_research_report
            result_dict = result if isinstance(result, dict) else result.to_dict() if hasattr(result, 'to_dict') else result
            report_buffer = generate_research_report(
                result_dict,
                project_name=st.session_state.get('current_project_name', 'Research Partner')
            )
            st.download_button(
                label="📄 Export Report",
                data=report_buffer,
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="export_report_btn",
                type="secondary",
                use_container_width=True
            )
        except Exception as e:
            st.button("📄 Export Report", disabled=True, help=f"Export unavailable: {e}",
                      use_container_width=True, type="secondary", key="export_report_btn_disabled")

    with col_guideline:
        if st.button("Add to Guideline", key="add_to_guideline_btn", type="secondary", use_container_width=True):
            success = add_to_cdp_workspace(
                result if isinstance(result, dict) else result.to_dict(),
                question_type,
                cdp_section_name
            )
            if success:
                st.success(f"Added to Guideline: {cdp_section_name}")
            else:
                st.error("Failed to add to Guideline")

    # Claim Ledger (expandable) - Expert-First approach
    if claim_ledger_data and claim_ledger_data.get('total', 0) > 0:
        total_claims = claim_ledger_data.get('total', 0)
        experts_count = len(expert_responses) if expert_responses else 0
        supported = claim_ledger_data.get('supported', 0)

        # Show expert analysis with supporting literature if available
        with st.expander(f"🎯 Expert Analysis ({experts_count} experts, {total_claims} claims, {supported} with supporting literature)", expanded=False):
            st.info("**Expert-First Approach**: Expert clinical reasoning with supporting case series where available.")
            render_claim_ledger(claim_ledger_data)

    # Cited Sources - Expandable Citation Cards
    evidence_summary_for_cards = result.get('evidence_summary', {}) if isinstance(result, dict) else getattr(result, 'evidence_summary', {})
    included_citations = result.get('included_citations', []) if isinstance(result, dict) else getattr(result, 'included_citations', [])

    # Try to get citations from evidence_summary if included_citations is empty
    if not included_citations and evidence_summary_for_cards:
        raw_citations = evidence_summary_for_cards.get('citations', [])
        if raw_citations:
            # Convert to list of dicts if needed
            included_citations = []
            for cit in raw_citations[:15]:
                if hasattr(cit, 'pmid'):
                    included_citations.append({
                        'pmid': cit.pmid,
                        'title': cit.title,
                        'abstract': getattr(cit, 'abstract', ''),
                        'authors': cit.authors if hasattr(cit, 'authors') else [],
                        'year': getattr(cit, 'year', getattr(cit, 'pub_date', '')),
                        'journal': getattr(cit, 'journal', ''),
                        'source_type': 'literature'
                    })
                elif isinstance(cit, dict):
                    included_citations.append(cit)

    if included_citations:
        with st.expander(f"📖 Cited Sources ({len(included_citations)})", expanded=False):
            render_citation_cards(
                included_citations,
                expanded=False,
                max_display=10,
                key_prefix="answer_view_citation"
            )
    else:
        # Expert-First approach message when no supporting literature found
        st.info("🎯 **Expert-First Approach**: Recommendations based on expert clinical reasoning. "
                "Supporting literature may be sparse in this low-evidence domain.")

    # Key findings (expandable, default open) with citation highlighting
    if key_findings:
        with st.expander("Key Findings", expanded=True):
            for i, finding in enumerate(key_findings):
                # Extract source attribution if present (format: "text (Expert Name)")
                source_match = re.search(r'\(([^)]+(?:Specialist|Physician|Oncologist|Methodologist|Chair|Advocate|Ethicist|Economist|Interventionalist))\)\s*$', finding)

                # Layout: finding text | mark button
                col_finding, col_mark = st.columns([14, 1])

                with col_finding:
                    if source_match:
                        source = source_match.group(1)
                        finding_text = finding[:source_match.start()].strip()
                        highlighted_finding = format_expert_response(finding_text)
                        st.markdown(f"• {highlighted_finding}", unsafe_allow_html=True)
                        st.caption(f"— {source}")
                    else:
                        highlighted_finding = format_expert_response(finding)
                        st.markdown(f"• {highlighted_finding}", unsafe_allow_html=True)

                with col_mark:
                    # Mark button for key finding
                    render_mark_button(
                        text=finding,
                        source_type="key_finding",
                        source_id=f"finding_{i}",
                        question_context=question,
                        key_suffix=f"kf_{i}",
                        project_id=st.session_state.get('current_project_id'),
                        compact=True,
                        inline=True
                    )

    # Dissenting views and alternatives
    alternatives_discussed = st.session_state.get('_alternatives_discussed', [])

    if dissenting_views:
        with st.expander("Dissenting Views", expanded=False):
            st.warning("The following experts raised concerns or dissenting opinions:")
            for dissent in dissenting_views:
                # Handle both old format (string) and new format (dict)
                if isinstance(dissent, dict):
                    expert = dissent.get('expert', 'Unknown')
                    position = dissent.get('position', '')
                    dissent_type = dissent.get('type', 'concern')

                    # Type-specific formatting
                    if dissent_type == 'strong_dissent':
                        st.error(f"**{expert}** (Strong Dissent)")
                    elif dissent_type == 'contradicting_recommendation':
                        st.warning(f"**{expert}** (Minority Position)")
                    else:
                        st.info(f"**{expert}** (Caution)")

                    if position:
                        st.markdown(f"> {position}")

                    # Show full context from expert response if available
                    if expert in expert_responses:
                        with st.expander(f"View {expert}'s full response", expanded=False):
                            content = expert_responses[expert].get('content', '')
                            # Format first, then show - avoid truncating before formatting to prevent broken markdown
                            formatted = format_expert_response(content)
                            st.markdown(formatted, unsafe_allow_html=True)
                else:
                    # Old format: just expert name string
                    expert = dissent
                    st.markdown(f"- **{expert}**")
                    if expert in expert_responses:
                        content = expert_responses[expert].get('content', '')
                        # Truncate at word boundary to avoid breaking markdown patterns
                        if len(content) > 800:
                            truncated = content[:800].rsplit(' ', 1)[0] + "..."
                        else:
                            truncated = content
                        formatted = format_expert_response(truncated)
                        st.markdown(f"<small>{formatted}</small>", unsafe_allow_html=True)
    else:
        # No dissenting views - show consensus message
        experts_count = len(expert_responses) if expert_responses else 0
        if experts_count > 0:
            st.success(f"**Consensus**: All {experts_count} experts agree on the core recommendation.")

    # Show supporting context from experts (separate from dissent)
    if alternatives_discussed:
        with st.expander("📎 Additional Expert Context", expanded=False):
            st.info("Experts provided additional context and considerations supporting the recommendation:")
            # Group by expert
            for alt in alternatives_discussed:
                expert = alt.get('expert', 'Unknown')
                position = alt.get('position', '')
                if position:
                    st.markdown(f"- **{expert}**: {position}")

    # Expert perspectives (expandable, default closed)
    # Implement delta-only display to reduce redundancy
    if expert_responses:
        with st.expander(f"Expert Perspectives ({len(expert_responses)})", expanded=False):
            # Option to toggle between delta and full view
            show_full = st.checkbox("Show full responses", value=False, key="show_full_expert_responses")

            if show_full:
                # Original full view with citation highlighting and mark buttons
                for expert_name, response in expert_responses.items():
                    content = response.get('content', '') if isinstance(response, dict) else str(response)

                    # Expert name header
                    st.markdown(f"**{expert_name}**")

                    # Apply citation and epistemic tag highlighting
                    formatted_content = format_expert_response(content)
                    st.markdown(formatted_content, unsafe_allow_html=True)

                    # Mark button at end of response (non-intrusive placement)
                    col_spacer, col_mark = st.columns([12, 1])
                    with col_mark:
                        render_mark_button(
                            text=content[:500],  # Mark first 500 chars as representative
                            source_type="expert_response",
                            source_id=expert_name,
                            question_context=question,
                            key_suffix=f"expert_{expert_name}",
                            project_id=st.session_state.get('current_project_id'),
                            compact=True,
                            inline=True
                        )
                    st.markdown("---")
            else:
                # Delta-only view - show unique contributions
                _render_delta_expert_view(expert_responses)

    # Smart Suggestions - contextual follow-up recommendations
    render_smart_suggestions(
        recommendation=recommendation,
        question=question,
        expert_responses=expert_responses,
        dissenting_views=dissenting_views,
        key_findings=key_findings,
    )

    # Marks Panel - show user's marked passages if requested
    render_marks_panel(project_id=st.session_state.get('current_project_id'))


def get_confidence_badge_html(confidence: str) -> str:
    """Return HTML for a confidence badge (string version)."""
    colors = {
        "HIGH": "#28a745",
        "MEDIUM": "#ffc107",
        "LOW": "#dc3545"
    }
    color = colors.get(confidence.upper(), "#666")
    return f'<span style="background: {color}; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem;">{confidence}</span>'


def render_mini_answer_card(result: Union[ResearchResult, Dict]):
    """
    Render a compact answer card (for history or previews).

    Args:
        result: ResearchResult object or dict
    """
    if isinstance(result, dict):
        question = result.get('question', '')[:80]
        question_type = result.get('question_type', 'general')
        confidence = result.get('confidence', 'MEDIUM')
        recommendation = result.get('recommendation', '')[:150]
    else:
        question = result.question[:80]
        question_type = result.question_type
        confidence = result.confidence
        recommendation = result.recommendation[:150]

    type_info = QUESTION_TYPES.get(question_type, {})
    icon = type_info.get('icon', '')

    confidence_colors = {"HIGH": "#28a745", "MEDIUM": "#ffc107", "LOW": "#dc3545"}
    conf_color = confidence_colors.get(confidence.upper(), "#666")

    st.markdown(f"""
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span>{icon} {question}...</span>
            <span style="background: {conf_color}; color: white; padding: 0.15rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem;">{confidence}</span>
        </div>
        <div style="color: #666; font-size: 0.9rem;">{recommendation}...</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# RED TEAM CHALLENGE SECTION
# =============================================================================

def _render_challenge_section(
    recommendation: str,
    question: str,
    expert_responses: Dict,
    dissenting_views: List,
    key_findings: List[str],
):
    """
    Render the Red Team Challenge section with generated challenging questions.

    Args:
        recommendation: The panel's recommendation
        question: Original clinical question
        expert_responses: Dict of expert responses
        dissenting_views: List of dissenting views (used as conflicts)
        key_findings: Key findings from the panel
    """
    st.markdown("---")
    st.markdown("### 🤔 Red Team Challenge")
    st.caption("Critical questions to stress-test this recommendation before guideline finalization")

    # Check if we already have challenges cached
    cache_key = f"challenges_{hash(recommendation[:100])}"

    if cache_key not in st.session_state:
        # Generate challenges
        with st.spinner("Generating challenging questions..."):
            try:
                from services.challenger_service import generate_challenges

                # Extract evidence gaps from quality warnings if available
                evidence_gaps = []
                if st.session_state.get('last_quality_warnings'):
                    evidence_gaps = st.session_state['last_quality_warnings']

                result = generate_challenges(
                    recommendation=recommendation,
                    question=question,
                    expert_responses=expert_responses,
                    conflicts=dissenting_views,
                    evidence_gaps=evidence_gaps,
                    key_findings=key_findings,
                )

                st.session_state[cache_key] = result

            except Exception as e:
                st.error(f"Failed to generate challenges: {e}")
                return

    # Display cached challenges
    challenge_result = st.session_state.get(cache_key)
    if not challenge_result:
        st.warning("No challenges generated.")
        return

    # Analysis summary
    if challenge_result.analysis:
        st.markdown(f"""
        <div style="background: #FEF3C7; border-left: 4px solid #F59E0B;
                    padding: 12px 16px; border-radius: 4px; margin-bottom: 16px;">
            <strong>Weak Points Identified:</strong><br/>
            {challenge_result.analysis}
        </div>
        """, unsafe_allow_html=True)

    # Challenge questions
    category_icons = {
        'assumption': '💭',
        'evidence': '📚',
        'patient_selection': '👥',
        'threshold': '📊',
        'risk': '⚠️',
        'feasibility': '🔧',
    }

    category_colors = {
        'assumption': ('#8B5CF6', '#EDE9FE'),     # Purple
        'evidence': ('#6366F1', '#EEF2FF'),       # Indigo
        'patient_selection': ('#10B981', '#D1FAE5'),  # Green
        'threshold': ('#F59E0B', '#FEF3C7'),      # Amber
        'risk': ('#EF4444', '#FEE2E2'),           # Red
        'feasibility': ('#3B82F6', '#DBEAFE'),    # Blue
    }

    for i, q in enumerate(challenge_result.questions, 1):
        icon = category_icons.get(q.category, '❓')
        badge_color, bg_color = category_colors.get(q.category, ('#666', '#F3F4F6'))

        st.markdown(f"""
        <div style="background: {bg_color}; border-radius: 8px; padding: 16px;
                    margin-bottom: 12px; border-left: 4px solid {badge_color};">
            <div style="display: flex; align-items: flex-start; gap: 12px;">
                <span style="font-size: 24px;">{icon}</span>
                <div style="flex: 1;">
                    <div style="font-weight: 600; font-size: 15px; color: #1F2937; margin-bottom: 8px;">
                        {q.question}
                    </div>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px;">
                        <span style="background: {badge_color}; color: white; padding: 2px 8px;
                                     border-radius: 4px; font-size: 11px; font-weight: 600;
                                     text-transform: uppercase;">
                            {q.category.replace('_', ' ')}
                        </span>
                    </div>
                    <div style="font-size: 13px; color: #6B7280;">
                        <strong>Targets:</strong> {q.targets}<br/>
                        <strong>Why it matters:</strong> {q.rationale}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Metadata footer
    st.caption(f"Generated by {challenge_result.model_used} | {challenge_result.conflicts_count} conflicts, {challenge_result.evidence_gaps_count} gaps analyzed")

    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Regenerate Challenges", key="regenerate_challenges_btn"):
            # Clear cache and trigger regeneration
            if cache_key in st.session_state:
                del st.session_state[cache_key]
            st.rerun()

    with col2:
        if st.button("✖️ Close Challenges", key="close_challenges_btn"):
            st.session_state['show_challenge'] = False
            st.rerun()


# =============================================================================
# SMART SUGGESTIONS
# =============================================================================

def _detect_suggestion_triggers(
    recommendation: str,
    expert_responses: Dict,
    dissenting_views: List,
    key_findings: List[str],
) -> List[Dict]:
    """
    Detect patterns in recommendation and generate contextual suggestions.

    Returns list of suggestions with:
    - icon: Display icon
    - category: Type of suggestion
    - title: Short title
    - description: Why this is suggested
    - query: Pre-filled query to run
    """
    suggestions = []
    rec_lower = recommendation.lower()

    # 1. Safety concerns - look for safety keywords
    safety_keywords = ['complication', 'mortality', 'morbidity', 'risk', 'adverse',
                       'toxicity', 'death', 'bleeding', 'infection', 'fistula',
                       'leak', 'wound', 'sepsis', 'failure']
    safety_found = [kw for kw in safety_keywords if kw in rec_lower]

    if safety_found:
        suggestions.append({
            'icon': '⚠️',
            'category': 'safety',
            'title': 'Deep dive on complications',
            'description': f'Safety concerns mentioned: {", ".join(safety_found[:3])}',
            'query': f'What are the specific complication rates and risk factors for {safety_found[0]} in this procedure?'
        })

    # 2. Expert disagreements
    if dissenting_views and len(dissenting_views) > 0:
        # Extract topic from first dissent
        first_dissent = dissenting_views[0]
        if isinstance(first_dissent, dict):
            expert = first_dissent.get('expert', 'Expert')
            position = first_dissent.get('position', '')[:50]
        else:
            expert = str(first_dissent)
            position = ''

        suggestions.append({
            'icon': '⚔️',
            'category': 'conflict',
            'title': f'Resolve {expert} concern',
            'description': f'Expert disagreement: {position}...' if position else 'Panel members disagree',
            'query': f'What evidence addresses {expert}\'s concerns about this recommendation?'
        })

    # 3. Conditional recommendations
    conditional_keywords = ['conditional', 'selected patients', 'certain patients',
                           'only if', 'provided that', 'when', 'in cases where']
    conditional_found = any(kw in rec_lower for kw in conditional_keywords)

    if conditional_found:
        suggestions.append({
            'icon': '🎯',
            'category': 'selection',
            'title': 'Clarify patient selection',
            'description': 'Recommendation is conditional - clarify criteria',
            'query': 'What are the specific patient selection criteria and contraindications for this intervention?'
        })

    # 4. Prognosis uncertainty
    prognosis_keywords = ['life expectancy', 'prognosis', 'survival', 'months to live',
                         'terminal', 'limited life', 'expected survival']
    prognosis_found = any(kw in rec_lower for kw in prognosis_keywords)

    if prognosis_found:
        suggestions.append({
            'icon': '📊',
            'category': 'prognosis',
            'title': 'Refine prognosis thresholds',
            'description': 'Prognosis mentioned - define decision thresholds',
            'query': 'At what life expectancy threshold does the intervention become inappropriate?'
        })

    # 5. Missing evidence (evidence gaps in key findings)
    gap_indicators = ['limited evidence', 'no direct evidence', 'insufficient data',
                     'extrapolated', 'expert opinion', 'evidence gap', 'low quality']

    for finding in (key_findings or []):
        finding_lower = finding.lower()
        if any(gap in finding_lower for gap in gap_indicators):
            suggestions.append({
                'icon': '🔍',
                'category': 'evidence_gap',
                'title': 'Address evidence gap',
                'description': f'{finding[:60]}...',
                'query': f'What additional evidence would strengthen this recommendation?'
            })
            break  # Only one evidence gap suggestion

    # 6. Alternatives mentioned
    alternative_keywords = ['alternative', 'versus', 'vs', 'compared to', 'instead of',
                           'non-surgical', 'conservative', 'stent', 'endoscopic']
    if any(kw in rec_lower for kw in alternative_keywords):
        suggestions.append({
            'icon': '⚖️',
            'category': 'comparison',
            'title': 'Compare alternatives',
            'description': 'Alternative approaches mentioned',
            'query': 'How do surgical and non-surgical approaches compare in outcomes and quality of life?'
        })

    # 7. Quality of life focus
    qol_keywords = ['quality of life', 'qol', 'palliation', 'symptom', 'comfort',
                   'functional status', 'days at home']
    if any(kw in rec_lower for kw in qol_keywords):
        suggestions.append({
            'icon': '💚',
            'category': 'qol',
            'title': 'QoL outcome data',
            'description': 'Quality of life is key - get specific data',
            'query': 'What are the quality of life outcomes and patient-reported measures for this intervention?'
        })

    # Limit to top 4 suggestions
    return suggestions[:4]


def render_smart_suggestions(
    recommendation: str,
    question: str,
    expert_responses: Dict,
    dissenting_views: List,
    key_findings: List[str],
):
    """
    Render smart follow-up suggestions based on the recommendation.

    Args:
        recommendation: The panel's recommendation
        question: Original question
        expert_responses: Expert responses dict
        dissenting_views: List of dissenting views
        key_findings: Key findings from the panel
    """
    suggestions = _detect_suggestion_triggers(
        recommendation=recommendation,
        expert_responses=expert_responses,
        dissenting_views=dissenting_views,
        key_findings=key_findings,
    )

    if not suggestions:
        return

    st.markdown("---")
    st.markdown("### 💡 Suggested Next Steps")
    st.caption("Based on your recommendation findings")

    # Display suggestions as clickable cards
    for i, sugg in enumerate(suggestions):
        icon = sugg['icon']
        title = sugg['title']
        description = sugg['description']
        query = sugg['query']
        category = sugg['category']

        # Category colors
        category_colors = {
            'safety': ('#EF4444', '#FEE2E2'),
            'conflict': ('#F59E0B', '#FEF3C7'),
            'selection': ('#10B981', '#D1FAE5'),
            'prognosis': ('#8B5CF6', '#EDE9FE'),
            'evidence_gap': ('#6366F1', '#EEF2FF'),
            'comparison': ('#3B82F6', '#DBEAFE'),
            'qol': ('#10B981', '#D1FAE5'),
        }
        badge_color, bg_color = category_colors.get(category, ('#666', '#F3F4F6'))

        col1, col2 = st.columns([5, 1])

        with col1:
            st.markdown(f"""
            <div style="background: {bg_color}; border-radius: 8px; padding: 12px 16px;
                        margin-bottom: 8px; border-left: 4px solid {badge_color};">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                    <span style="font-size: 18px;">{icon}</span>
                    <span style="font-weight: 600; color: #1F2937;">{title}</span>
                </div>
                <div style="font-size: 13px; color: #6B7280;">{description}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if st.button("Ask →", key=f"suggestion_btn_{i}", use_container_width=True):
                # Store suggestion query and trigger quick answer
                st.session_state['suggestion_query'] = query
                st.session_state['trigger_suggestion'] = True
                st.rerun()

    # Handle suggestion trigger
    if st.session_state.get('trigger_suggestion'):
        query = st.session_state.get('suggestion_query', '')
        if query:
            st.session_state['trigger_suggestion'] = False

            # Show the query being asked
            st.info(f"**Follow-up:** {query}")

            # Execute quick answer for the suggestion
            try:
                from services.quick_answer_service import get_quick_answer_with_search
                from ui.citation_utils import format_expert_response

                with st.spinner("Getting answer..."):
                    result = get_quick_answer_with_search(
                        question=query,
                        max_results=5
                    )

                # Display inline answer
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #10B981 0%, #059669 100%);
                            padding: 12px 16px; border-radius: 8px; color: white; margin: 8px 0;">
                    <span style="font-size: 12px; text-transform: uppercase; opacity: 0.9;">Quick Answer</span>
                </div>
                """, unsafe_allow_html=True)

                formatted = format_expert_response(result.answer)
                st.markdown(formatted, unsafe_allow_html=True)

                if result.citations:
                    with st.expander(f"View {len(result.citations)} sources"):
                        for j, cit in enumerate(result.citations, 1):
                            pmid = cit.get('pmid', '')
                            title = cit.get('title', '')
                            if pmid:
                                st.markdown(f"[{j}] [{title}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")

            except Exception as e:
                st.error(f"Could not get answer: {e}")
