"""
Quality Assessment Module for GRADE-based Evidence Evaluation

Provides risk of bias assessment, evidence level classification,
and certainty ratings following GRADE methodology.

Key components:
- Study design classification (Oxford CEBM levels)
- Risk of bias domains by study type
- Overall certainty of evidence (High/Moderate/Low/Very Low)
- Downgrade/upgrade reasons
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================

class StudyDesign(Enum):
    """Study design classifications."""
    SYSTEMATIC_REVIEW = "Systematic Review"
    META_ANALYSIS = "Meta-Analysis"
    RCT = "RCT"
    COHORT_PROSPECTIVE = "Prospective Cohort"
    COHORT_RETROSPECTIVE = "Retrospective Cohort"
    CASE_CONTROL = "Case-Control"
    CASE_SERIES = "Case Series"
    CASE_REPORT = "Case Report"
    EXPERT_OPINION = "Expert Opinion"
    OTHER = "Other"


class EvidenceLevel(Enum):
    """Oxford CEBM Evidence Levels (2011)."""
    LEVEL_I = "I"      # Systematic review of RCTs
    LEVEL_II = "II"    # RCT or observational with dramatic effect
    LEVEL_III = "III"  # Non-randomized controlled cohort
    LEVEL_IV = "IV"    # Case-series, case-control
    LEVEL_V = "V"      # Expert opinion


class RiskOfBias(Enum):
    """Risk of bias categories."""
    LOW = "Low"
    SOME_CONCERNS = "Some Concerns"
    HIGH = "High"
    CRITICAL = "Critical"


class Certainty(Enum):
    """GRADE certainty of evidence."""
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    VERY_LOW = "Very Low"


# Evidence level mapping by study design
DESIGN_TO_LEVEL = {
    StudyDesign.SYSTEMATIC_REVIEW.value: EvidenceLevel.LEVEL_I.value,
    StudyDesign.META_ANALYSIS.value: EvidenceLevel.LEVEL_I.value,
    StudyDesign.RCT.value: EvidenceLevel.LEVEL_II.value,
    StudyDesign.COHORT_PROSPECTIVE.value: EvidenceLevel.LEVEL_III.value,
    StudyDesign.COHORT_RETROSPECTIVE.value: EvidenceLevel.LEVEL_III.value,
    StudyDesign.CASE_CONTROL.value: EvidenceLevel.LEVEL_IV.value,
    StudyDesign.CASE_SERIES.value: EvidenceLevel.LEVEL_IV.value,
    StudyDesign.CASE_REPORT.value: EvidenceLevel.LEVEL_V.value,
    StudyDesign.EXPERT_OPINION.value: EvidenceLevel.LEVEL_V.value,
    StudyDesign.OTHER.value: EvidenceLevel.LEVEL_V.value,
}

# Risk of bias domains by study type
ROB_DOMAINS = {
    "RCT": [
        ("randomization", "Randomization process"),
        ("deviations", "Deviations from intended interventions"),
        ("missing_data", "Missing outcome data"),
        ("outcome_measurement", "Measurement of the outcome"),
        ("selective_reporting", "Selection of the reported result"),
    ],
    "Cohort": [
        ("confounding", "Confounding"),
        ("selection", "Selection of participants"),
        ("classification", "Classification of interventions"),
        ("deviations", "Deviations from intended interventions"),
        ("missing_data", "Missing data"),
        ("outcome_measurement", "Measurement of outcomes"),
        ("selective_reporting", "Selection of the reported result"),
    ],
    "Case Series": [
        ("selection", "Selection of cases"),
        ("ascertainment", "Ascertainment of exposure/outcome"),
        ("causality", "Assessment of causality"),
        ("reporting", "Completeness of reporting"),
    ],
    "Case-Control": [
        ("selection", "Selection of cases and controls"),
        ("definition", "Definition of cases and controls"),
        ("comparability", "Comparability of groups"),
        ("exposure", "Ascertainment of exposure"),
    ],
}

# GRADE downgrade reasons
DOWNGRADE_REASONS = [
    ("risk_of_bias", "Risk of bias", "Serious limitations in study design or execution"),
    ("inconsistency", "Inconsistency", "Heterogeneity in results across studies"),
    ("indirectness", "Indirectness", "Evidence not directly applicable to question"),
    ("imprecision", "Imprecision", "Wide confidence intervals or small sample size"),
    ("publication_bias", "Publication bias", "Suspected selective publication of results"),
]

# GRADE upgrade reasons (for observational studies)
UPGRADE_REASONS = [
    ("large_effect", "Large effect", "Large magnitude of effect (RR > 2 or < 0.5)"),
    ("dose_response", "Dose-response gradient", "Evidence of dose-response relationship"),
    ("confounding", "All plausible confounding would reduce effect", "Confounders would bias toward null"),
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DomainRating:
    """Rating for a single risk of bias domain."""
    domain_id: str
    domain_name: str
    rating: str  # Low, Some Concerns, High, Critical
    rationale: str = ""
    supporting_quotes: List[str] = field(default_factory=list)


@dataclass
class QualityRating:
    """
    Complete quality assessment for a single study.

    Includes study design, evidence level, risk of bias by domain,
    and overall certainty rating with downgrade reasons.
    """
    pmid: str
    study_design: str
    evidence_level: str  # I, II, III, IV, V

    # Risk of bias assessment
    risk_of_bias: str  # Overall: Low, Some Concerns, High, Critical
    domain_ratings: List[DomainRating] = field(default_factory=list)

    # GRADE certainty
    certainty: str = "Low"  # High, Moderate, Low, Very Low
    downgrade_reasons: List[str] = field(default_factory=list)
    upgrade_reasons: List[str] = field(default_factory=list)

    # Notes
    assessor_notes: str = ""
    assessed_by: str = "ai"  # "ai" or "human"
    assessed_at: str = ""

    def __post_init__(self):
        if not self.assessed_at:
            self.assessed_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['domain_ratings'] = [asdict(d) for d in self.domain_ratings]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityRating':
        """Create from dictionary."""
        domain_ratings = [
            DomainRating(**d) for d in data.pop('domain_ratings', [])
        ]
        return cls(domain_ratings=domain_ratings, **data)

    def get_certainty_symbol(self) -> str:
        """Get GRADE certainty symbol (⊕⊕⊕⊕)."""
        symbols = {
            "High": "⊕⊕⊕⊕",
            "Moderate": "⊕⊕⊕○",
            "Low": "⊕⊕○○",
            "Very Low": "⊕○○○"
        }
        return symbols.get(self.certainty, "○○○○")

    def get_rob_color(self) -> str:
        """Get color code for risk of bias display."""
        colors = {
            "Low": "#28a745",  # Green
            "Some Concerns": "#ffc107",  # Yellow
            "High": "#fd7e14",  # Orange
            "Critical": "#dc3545"  # Red
        }
        return colors.get(self.risk_of_bias, "#6c757d")


# =============================================================================
# QUALITY ASSESSMENT FUNCTIONS
# =============================================================================

def get_evidence_level(study_design: str) -> str:
    """
    Get Oxford CEBM evidence level for a study design.

    Args:
        study_design: Study design string

    Returns:
        Evidence level (I, II, III, IV, V)
    """
    # Normalize design string
    design_lower = study_design.lower()

    if 'systematic review' in design_lower or 'meta-analysis' in design_lower:
        return EvidenceLevel.LEVEL_I.value
    elif 'rct' in design_lower or 'randomized' in design_lower:
        return EvidenceLevel.LEVEL_II.value
    elif 'cohort' in design_lower:
        return EvidenceLevel.LEVEL_III.value
    elif 'case-control' in design_lower or 'case control' in design_lower:
        return EvidenceLevel.LEVEL_IV.value
    elif 'case series' in design_lower or 'case report' in design_lower:
        return EvidenceLevel.LEVEL_IV.value
    else:
        return EvidenceLevel.LEVEL_V.value


def get_rob_domains(study_design: str) -> List[tuple]:
    """
    Get risk of bias domains appropriate for study design.

    Args:
        study_design: Study design string

    Returns:
        List of (domain_id, domain_name) tuples
    """
    design_lower = study_design.lower()

    if 'rct' in design_lower or 'randomized' in design_lower:
        return ROB_DOMAINS["RCT"]
    elif 'cohort' in design_lower:
        return ROB_DOMAINS["Cohort"]
    elif 'case-control' in design_lower or 'case control' in design_lower:
        return ROB_DOMAINS["Case-Control"]
    elif 'case series' in design_lower or 'case report' in design_lower:
        return ROB_DOMAINS["Case Series"]
    else:
        return ROB_DOMAINS["Case Series"]  # Default to simplest


def calculate_overall_rob(domain_ratings: List[DomainRating]) -> str:
    """
    Calculate overall risk of bias from domain ratings.

    Rules (similar to RoB 2.0):
    - Critical in any domain → Critical overall
    - High in any domain → High overall
    - Some Concerns in multiple domains → High overall
    - Some Concerns in one domain → Some Concerns overall
    - Low in all domains → Low overall

    Args:
        domain_ratings: List of domain ratings

    Returns:
        Overall risk of bias: Low, Some Concerns, High, Critical
    """
    if not domain_ratings:
        return "High"  # No assessment = assume high risk

    ratings = [d.rating for d in domain_ratings]

    if "Critical" in ratings:
        return "Critical"

    high_count = ratings.count("High")
    concerns_count = ratings.count("Some Concerns")

    if high_count > 0:
        return "High"
    elif concerns_count >= 2:
        return "High"
    elif concerns_count == 1:
        return "Some Concerns"
    else:
        return "Low"


def calculate_certainty(
    study_design: str,
    risk_of_bias: str,
    downgrade_reasons: List[str],
    upgrade_reasons: List[str] = None
) -> str:
    """
    Calculate GRADE certainty of evidence.

    Starting certainty based on study design:
    - RCTs start at High
    - Observational studies start at Low

    Then apply downgrades (max 2 levels) and upgrades (max 2 levels for observational).

    Args:
        study_design: Study design string
        risk_of_bias: Overall risk of bias
        downgrade_reasons: List of reasons to downgrade
        upgrade_reasons: List of reasons to upgrade (observational only)

    Returns:
        Certainty: High, Moderate, Low, Very Low
    """
    certainty_levels = ["Very Low", "Low", "Moderate", "High"]

    # Starting level
    design_lower = study_design.lower()
    if 'rct' in design_lower or 'randomized' in design_lower or 'systematic review' in design_lower:
        level_index = 3  # High
    else:
        level_index = 1  # Low for observational

    # Downgrade for risk of bias (if not already in downgrade_reasons)
    if risk_of_bias in ["High", "Critical"] and "risk_of_bias" not in downgrade_reasons:
        level_index = max(0, level_index - 1)
    elif risk_of_bias == "Some Concerns" and "risk_of_bias" not in downgrade_reasons:
        level_index = max(0, level_index - 1)

    # Apply explicit downgrades
    for reason in downgrade_reasons:
        level_index = max(0, level_index - 1)

    # Apply upgrades (only for observational studies)
    if upgrade_reasons and level_index < 3:
        is_observational = 'rct' not in design_lower and 'randomized' not in design_lower
        if is_observational:
            for reason in upgrade_reasons[:2]:  # Max 2 upgrades
                level_index = min(3, level_index + 1)

    return certainty_levels[level_index]


def create_quality_rating(
    pmid: str,
    study_design: str,
    domain_assessments: Dict[str, str] = None,
    downgrade_reasons: List[str] = None,
    upgrade_reasons: List[str] = None,
    assessor_notes: str = "",
    assessed_by: str = "ai"
) -> QualityRating:
    """
    Create a complete quality rating for a study.

    Args:
        pmid: PubMed ID
        study_design: Study design string
        domain_assessments: Dict mapping domain_id -> rating (Low/Some Concerns/High/Critical)
        downgrade_reasons: List of GRADE downgrade reasons
        upgrade_reasons: List of GRADE upgrade reasons
        assessor_notes: Free text notes
        assessed_by: "ai" or "human"

    Returns:
        Complete QualityRating object
    """
    evidence_level = get_evidence_level(study_design)
    domains = get_rob_domains(study_design)

    # Create domain ratings
    domain_ratings = []
    if domain_assessments:
        for domain_id, domain_name in domains:
            rating = domain_assessments.get(domain_id, "Some Concerns")
            domain_ratings.append(DomainRating(
                domain_id=domain_id,
                domain_name=domain_name,
                rating=rating
            ))
    else:
        # Default all to Some Concerns if no assessment provided
        for domain_id, domain_name in domains:
            domain_ratings.append(DomainRating(
                domain_id=domain_id,
                domain_name=domain_name,
                rating="Some Concerns"
            ))

    # Calculate overall ROB
    risk_of_bias = calculate_overall_rob(domain_ratings)

    # Calculate certainty
    certainty = calculate_certainty(
        study_design=study_design,
        risk_of_bias=risk_of_bias,
        downgrade_reasons=downgrade_reasons or [],
        upgrade_reasons=upgrade_reasons
    )

    return QualityRating(
        pmid=pmid,
        study_design=study_design,
        evidence_level=evidence_level,
        risk_of_bias=risk_of_bias,
        domain_ratings=domain_ratings,
        certainty=certainty,
        downgrade_reasons=downgrade_reasons or [],
        upgrade_reasons=upgrade_reasons or [],
        assessor_notes=assessor_notes,
        assessed_by=assessed_by
    )


def assess_quality_with_ai(
    pmid: str,
    title: str,
    abstract: str,
    study_design: str,
    api_key: str = None
) -> Optional[QualityRating]:
    """
    Use AI to assess quality of a study based on abstract.

    Args:
        pmid: PubMed ID
        title: Study title
        abstract: Study abstract
        study_design: Identified study design
        api_key: OpenAI API key

    Returns:
        QualityRating or None if assessment fails
    """
    try:
        from core.llm_utils import get_llm_client
        import json

        client = get_llm_client(api_key=api_key)
        domains = get_rob_domains(study_design)

        domain_list = "\n".join([f"- {d[0]}: {d[1]}" for d in domains])

        prompt = f"""Assess the risk of bias for this study based on the abstract.

**Title:** {title}

**Abstract:** {abstract}

**Study Design:** {study_design}

**Risk of Bias Domains to Assess:**
{domain_list}

**Instructions:**
For each domain, rate as: "Low", "Some Concerns", "High", or "Critical"
Also identify any GRADE downgrade reasons:
- risk_of_bias: Serious methodological limitations
- inconsistency: N/A for single study
- indirectness: Not directly applicable to our population/intervention
- imprecision: Wide CIs or small sample size
- publication_bias: N/A for single study

**Response Format (JSON only):**
{{
    "domain_ratings": {{
        "domain_id": "rating",
        ...
    }},
    "downgrade_reasons": ["reason1", "reason2"],
    "notes": "Brief assessment notes"
}}

Respond with ONLY the JSON object."""

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a GRADE methodologist assessing risk of bias. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        return create_quality_rating(
            pmid=pmid,
            study_design=study_design,
            domain_assessments=result.get("domain_ratings", {}),
            downgrade_reasons=result.get("downgrade_reasons", []),
            assessor_notes=result.get("notes", ""),
            assessed_by="ai"
        )

    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"AI quality assessment failed: {e}")
        return None


# =============================================================================
# AGGREGATE FUNCTIONS
# =============================================================================

def summarize_evidence_quality(ratings: List[QualityRating]) -> Dict[str, Any]:
    """
    Summarize quality across multiple studies.

    Args:
        ratings: List of QualityRating objects

    Returns:
        Summary dict with counts and overall assessment
    """
    if not ratings:
        return {
            "total_studies": 0,
            "by_certainty": {},
            "by_design": {},
            "by_rob": {},
            "overall_certainty": "Very Low"
        }

    by_certainty = {}
    by_design = {}
    by_rob = {}

    for r in ratings:
        by_certainty[r.certainty] = by_certainty.get(r.certainty, 0) + 1
        by_design[r.study_design] = by_design.get(r.study_design, 0) + 1
        by_rob[r.risk_of_bias] = by_rob.get(r.risk_of_bias, 0) + 1

    # Overall certainty is driven by the highest quality evidence available
    certainty_priority = ["High", "Moderate", "Low", "Very Low"]
    overall = "Very Low"
    for cert in certainty_priority:
        if by_certainty.get(cert, 0) > 0:
            overall = cert
            break

    return {
        "total_studies": len(ratings),
        "by_certainty": by_certainty,
        "by_design": by_design,
        "by_rob": by_rob,
        "overall_certainty": overall
    }


def format_certainty_badge(certainty: str) -> str:
    """Format certainty as colored badge HTML."""
    colors = {
        "High": "#28a745",
        "Moderate": "#17a2b8",
        "Low": "#ffc107",
        "Very Low": "#dc3545"
    }
    symbols = {
        "High": "⊕⊕⊕⊕",
        "Moderate": "⊕⊕⊕○",
        "Low": "⊕⊕○○",
        "Very Low": "⊕○○○"
    }
    color = colors.get(certainty, "#6c757d")
    symbol = symbols.get(certainty, "○○○○")

    return f'<span style="background: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem;">{symbol} {certainty}</span>'
