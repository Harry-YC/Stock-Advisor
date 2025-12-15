"""
Evidence-to-Decision (EtD) Framework Service

Implements the GRADE Evidence-to-Decision framework for generating
structured recommendations from GDG discussions.

EtD Framework Domains:
1. Problem priority
2. Benefits
3. Harms
4. Certainty of evidence
5. Values and preferences
6. Balance of effects
7. Resources required
8. Equity
9. Acceptability
10. Feasibility

Each domain is assessed and contributes to the final recommendation.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class Judgment(Enum):
    """Standard EtD judgments for most domains."""
    FAVORS_INTERVENTION = "Favors intervention"
    FAVORS_COMPARATOR = "Favors comparator"
    NEITHER = "Neither favors intervention nor comparator"
    VARIES = "Varies"
    UNCERTAIN = "Uncertain"
    NOT_APPLICABLE = "Not applicable"


class CertaintyLevel(Enum):
    """GRADE certainty levels."""
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    VERY_LOW = "Very Low"


class RecommendationStrength(Enum):
    """GRADE recommendation strength."""
    STRONG_FOR = "Strong FOR"
    CONDITIONAL_FOR = "Conditional FOR"
    CONDITIONAL_AGAINST = "Conditional AGAINST"
    STRONG_AGAINST = "Strong AGAINST"


# EtD domain definitions
ETD_DOMAINS = {
    "problem_priority": {
        "name": "Problem Priority",
        "question": "Is the problem a priority?",
        "options": ["Yes", "Probably yes", "Probably no", "No", "Varies", "Uncertain"]
    },
    "benefits": {
        "name": "Desirable Effects",
        "question": "How substantial are the desirable anticipated effects?",
        "options": ["Large", "Moderate", "Small", "Trivial", "Varies", "Uncertain"]
    },
    "harms": {
        "name": "Undesirable Effects",
        "question": "How substantial are the undesirable anticipated effects?",
        "options": ["Large", "Moderate", "Small", "Trivial", "Varies", "Uncertain"]
    },
    "certainty": {
        "name": "Certainty of Evidence",
        "question": "What is the overall certainty of the evidence of effects?",
        "options": ["High", "Moderate", "Low", "Very Low", "No included studies"]
    },
    "values": {
        "name": "Values and Preferences",
        "question": "Is there important uncertainty about or variability in how much people value the main outcomes?",
        "options": ["Important uncertainty or variability", "Possibly important", "Probably no important", "No important uncertainty"]
    },
    "balance": {
        "name": "Balance of Effects",
        "question": "Does the balance between desirable and undesirable effects favor the intervention or the comparison?",
        "options": ["Favors intervention", "Probably favors intervention", "Does not favor either", "Probably favors comparator", "Favors comparator", "Varies", "Uncertain"]
    },
    "resources": {
        "name": "Resources Required",
        "question": "How large are the resource requirements (costs)?",
        "options": ["Large costs", "Moderate costs", "Negligible costs/savings", "Moderate savings", "Large savings", "Varies", "Uncertain"]
    },
    "equity": {
        "name": "Equity",
        "question": "What would be the impact on health equity?",
        "options": ["Reduced", "Probably reduced", "Probably no impact", "Probably increased", "Increased", "Varies", "Uncertain"]
    },
    "acceptability": {
        "name": "Acceptability",
        "question": "Is the intervention acceptable to key stakeholders?",
        "options": ["Yes", "Probably yes", "Probably no", "No", "Varies", "Uncertain"]
    },
    "feasibility": {
        "name": "Feasibility",
        "question": "Is the intervention feasible to implement?",
        "options": ["Yes", "Probably yes", "Probably no", "No", "Varies", "Uncertain"]
    }
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DomainJudgment:
    """Judgment for a single EtD domain."""
    domain_id: str
    domain_name: str
    judgment: str
    rationale: str = ""
    supporting_evidence: List[str] = field(default_factory=list)  # PMIDs
    expert_sources: List[str] = field(default_factory=list)  # Expert names who informed this


@dataclass
class EvidenceToDecision:
    """
    Complete Evidence-to-Decision framework card.

    Captures all domain judgments and generates a recommendation.
    """
    # Identifiers
    question: str
    question_type: str = ""
    population: str = ""
    intervention: str = ""
    comparator: str = ""

    # Domain judgments
    domain_judgments: Dict[str, DomainJudgment] = field(default_factory=dict)

    # Recommendation outputs
    recommendation_direction: str = ""  # For, Against
    recommendation_strength: str = ""  # Strong, Conditional
    recommendation_statement: str = ""
    key_citations: List[str] = field(default_factory=list)

    # Justification
    justification: str = ""
    implementation_considerations: str = ""
    monitoring_evaluation: str = ""
    research_priorities: List[str] = field(default_factory=list)

    # Metadata
    status: str = "draft"  # draft, reviewed, locked
    created_at: str = ""
    last_modified: str = ""
    created_by: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.last_modified = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['domain_judgments'] = {
            k: asdict(v) for k, v in self.domain_judgments.items()
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvidenceToDecision':
        """Create from dictionary."""
        domain_judgments = {}
        for k, v in data.pop('domain_judgments', {}).items():
            domain_judgments[k] = DomainJudgment(**v)
        return cls(domain_judgments=domain_judgments, **data)

    def get_full_recommendation_text(self) -> str:
        """Get the complete recommendation statement with strength."""
        strength_text = {
            "Strong FOR": "We recommend",
            "Conditional FOR": "We suggest",
            "Conditional AGAINST": "We suggest against",
            "Strong AGAINST": "We recommend against"
        }.get(f"{self.recommendation_strength} {self.recommendation_direction}", "We suggest")

        return f"{strength_text} {self.recommendation_statement}"

    def is_complete(self) -> bool:
        """Check if all required domains have judgments."""
        required = ["benefits", "harms", "certainty", "balance"]
        return all(d in self.domain_judgments for d in required)


# =============================================================================
# ETD SERVICE CLASS
# =============================================================================

class EtDService:
    """
    Service for generating and managing Evidence-to-Decision cards.

    Usage:
        service = EtDService()
        etd = service.generate_etd_from_discussion(
            question="Should patients with MBO undergo surgical bypass?",
            expert_responses={"Surgical Oncologist": "...", ...},
            evidence_summary="..."
        )
    """

    def __init__(self, llm_router=None):
        """
        Initialize EtD service.

        Args:
            llm_router: Optional LLM router for AI-powered generation
        """
        self.router = llm_router

    def create_empty_etd(
        self,
        question: str,
        population: str = "",
        intervention: str = "",
        comparator: str = ""
    ) -> EvidenceToDecision:
        """Create an empty EtD card ready for domain assessment."""
        etd = EvidenceToDecision(
            question=question,
            population=population,
            intervention=intervention,
            comparator=comparator
        )

        # Initialize all domains with empty judgments
        for domain_id, domain_info in ETD_DOMAINS.items():
            etd.domain_judgments[domain_id] = DomainJudgment(
                domain_id=domain_id,
                domain_name=domain_info["name"],
                judgment="Uncertain"
            )

        return etd

    def generate_etd_from_discussion(
        self,
        question: str,
        expert_responses: Dict[str, str],
        evidence_summary: str = "",
        included_pmids: List[str] = None,
        question_type: str = ""
    ) -> EvidenceToDecision:
        """
        Generate EtD card from GDG expert discussion.

        Uses AI to extract domain judgments from expert responses.

        Args:
            question: Clinical question
            expert_responses: Dict mapping expert name to response text
            evidence_summary: Summary of included evidence
            included_pmids: List of valid PMIDs for citation
            question_type: Type of question (for context)

        Returns:
            Populated EvidenceToDecision object
        """
        etd = EvidenceToDecision(
            question=question,
            question_type=question_type,
            key_citations=included_pmids or []
        )

        # Try AI extraction if router available
        if self.router or settings.OPENAI_API_KEY:
            try:
                ai_result = self._extract_etd_with_ai(
                    question=question,
                    expert_responses=expert_responses,
                    evidence_summary=evidence_summary,
                    included_pmids=included_pmids
                )
                if ai_result:
                    return ai_result
            except Exception as e:
                logger.error(f"AI EtD extraction failed: {e}")

        # Fallback: Create from keywords
        return self._extract_etd_from_keywords(
            etd=etd,
            expert_responses=expert_responses
        )

    def _extract_etd_with_ai(
        self,
        question: str,
        expert_responses: Dict[str, str],
        evidence_summary: str,
        included_pmids: List[str]
    ) -> Optional[EvidenceToDecision]:
        """Use AI to extract EtD judgments from discussion."""
        try:
            from core.llm_utils import get_llm_client

            client = get_llm_client()

            # Build expert response summary
            responses_text = "\n\n".join([
                f"**{name}:**\n{text[:1000]}"
                for name, text in expert_responses.items()
            ])

            prompt = f"""Analyze this GDG (Guideline Development Group) discussion and extract Evidence-to-Decision framework judgments.

**Clinical Question:** {question}

**Evidence Summary:**
{evidence_summary[:2000] if evidence_summary else "Not provided"}

**Expert Responses:**
{responses_text}

**Available PMIDs for citation:** {', '.join(included_pmids[:10]) if included_pmids else "None specified"}

**Extract judgments for these EtD domains:**

1. **Benefits** (desirable effects): Large / Moderate / Small / Trivial / Uncertain
2. **Harms** (undesirable effects): Large / Moderate / Small / Trivial / Uncertain
3. **Certainty of evidence**: High / Moderate / Low / Very Low
4. **Values and preferences**: Important uncertainty / Probably no important uncertainty
5. **Balance of effects**: Favors intervention / Does not favor either / Favors comparator / Uncertain
6. **Resources required**: Large costs / Moderate costs / Negligible / Moderate savings / Large savings
7. **Acceptability**: Yes / Probably yes / Probably no / No / Uncertain
8. **Feasibility**: Yes / Probably yes / Probably no / No / Uncertain

**Then generate:**
- recommendation_direction: "For" or "Against"
- recommendation_strength: "Strong" or "Conditional"
- recommendation_statement: Clear actionable statement (WITHOUT "We recommend/suggest" - just the action)
- justification: Brief rationale for the recommendation

**Response Format (JSON only):**
{{
    "population": "extracted population",
    "intervention": "extracted intervention",
    "comparator": "extracted comparator",
    "domain_judgments": {{
        "benefits": {{"judgment": "...", "rationale": "..."}},
        "harms": {{"judgment": "...", "rationale": "..."}},
        "certainty": {{"judgment": "...", "rationale": "..."}},
        "values": {{"judgment": "...", "rationale": "..."}},
        "balance": {{"judgment": "...", "rationale": "..."}},
        "resources": {{"judgment": "...", "rationale": "..."}},
        "acceptability": {{"judgment": "...", "rationale": "..."}},
        "feasibility": {{"judgment": "...", "rationale": "..."}}
    }},
    "recommendation_direction": "For or Against",
    "recommendation_strength": "Strong or Conditional",
    "recommendation_statement": "the specific recommendation",
    "justification": "brief rationale",
    "key_pmids": ["pmid1", "pmid2"]
}}

Respond with ONLY the JSON object."""

            response = client.chat.completions.create(
                model=settings.REASONING_MODEL,
                messages=[
                    {"role": "system", "content": "You are a GRADE methodologist extracting Evidence-to-Decision framework judgments. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Build EtD object from AI result
            etd = EvidenceToDecision(
                question=question,
                population=result.get("population", ""),
                intervention=result.get("intervention", ""),
                comparator=result.get("comparator", ""),
                recommendation_direction=result.get("recommendation_direction", "For"),
                recommendation_strength=result.get("recommendation_strength", "Conditional"),
                recommendation_statement=result.get("recommendation_statement", ""),
                justification=result.get("justification", ""),
                key_citations=result.get("key_pmids", [])
            )

            # Populate domain judgments
            for domain_id, judgment_data in result.get("domain_judgments", {}).items():
                if domain_id in ETD_DOMAINS:
                    etd.domain_judgments[domain_id] = DomainJudgment(
                        domain_id=domain_id,
                        domain_name=ETD_DOMAINS[domain_id]["name"],
                        judgment=judgment_data.get("judgment", "Uncertain"),
                        rationale=judgment_data.get("rationale", "")
                    )

            # Fill in missing domains
            for domain_id, domain_info in ETD_DOMAINS.items():
                if domain_id not in etd.domain_judgments:
                    etd.domain_judgments[domain_id] = DomainJudgment(
                        domain_id=domain_id,
                        domain_name=domain_info["name"],
                        judgment="Uncertain"
                    )

            return etd

        except Exception as e:
            logger.error(f"AI EtD extraction error: {e}")
            return None

    def _extract_etd_from_keywords(
        self,
        etd: EvidenceToDecision,
        expert_responses: Dict[str, str]
    ) -> EvidenceToDecision:
        """Fallback: Extract EtD judgments using keyword matching."""
        full_text = " ".join(expert_responses.values()).lower()

        # Benefits
        if any(w in full_text for w in ["significant benefit", "major improvement", "substantial"]):
            etd.domain_judgments["benefits"] = DomainJudgment("benefits", "Desirable Effects", "Large")
        elif any(w in full_text for w in ["some benefit", "moderate improvement"]):
            etd.domain_judgments["benefits"] = DomainJudgment("benefits", "Desirable Effects", "Moderate")
        else:
            etd.domain_judgments["benefits"] = DomainJudgment("benefits", "Desirable Effects", "Uncertain")

        # Harms
        if any(w in full_text for w in ["high risk", "significant complication", "major adverse"]):
            etd.domain_judgments["harms"] = DomainJudgment("harms", "Undesirable Effects", "Large")
        elif any(w in full_text for w in ["some risk", "moderate complication"]):
            etd.domain_judgments["harms"] = DomainJudgment("harms", "Undesirable Effects", "Moderate")
        else:
            etd.domain_judgments["harms"] = DomainJudgment("harms", "Undesirable Effects", "Uncertain")

        # Certainty
        if any(w in full_text for w in ["high quality", "strong evidence", "well established"]):
            etd.domain_judgments["certainty"] = DomainJudgment("certainty", "Certainty of Evidence", "Moderate")
        elif any(w in full_text for w in ["low quality", "limited evidence", "poor quality"]):
            etd.domain_judgments["certainty"] = DomainJudgment("certainty", "Certainty of Evidence", "Low")
        else:
            etd.domain_judgments["certainty"] = DomainJudgment("certainty", "Certainty of Evidence", "Very Low")

        # Fill remaining domains
        for domain_id, domain_info in ETD_DOMAINS.items():
            if domain_id not in etd.domain_judgments:
                etd.domain_judgments[domain_id] = DomainJudgment(
                    domain_id=domain_id,
                    domain_name=domain_info["name"],
                    judgment="Uncertain"
                )

        return etd

    def derive_recommendation(self, etd: EvidenceToDecision) -> EvidenceToDecision:
        """
        Derive recommendation direction and strength from EtD judgments.

        Rules based on GRADE methodology:
        - Strong FOR: Benefits >> Harms, high certainty, acceptable, feasible
        - Conditional FOR: Benefits > Harms but with uncertainty
        - Conditional AGAINST: Harms > Benefits or high uncertainty
        - Strong AGAINST: Harms >> Benefits, clear evidence

        Args:
            etd: EvidenceToDecision with domain judgments

        Returns:
            Updated EtD with recommendation
        """
        benefits = etd.domain_judgments.get("benefits", DomainJudgment("benefits", "", "Uncertain")).judgment
        harms = etd.domain_judgments.get("harms", DomainJudgment("harms", "", "Uncertain")).judgment
        certainty = etd.domain_judgments.get("certainty", DomainJudgment("certainty", "", "Very Low")).judgment
        balance = etd.domain_judgments.get("balance", DomainJudgment("balance", "", "Uncertain")).judgment

        # Determine direction
        if "Favors intervention" in balance or "Probably favors intervention" in balance:
            direction = "For"
        elif "Favors comparator" in balance or "Probably favors comparator" in balance:
            direction = "Against"
        elif benefits in ["Large", "Moderate"] and harms in ["Small", "Trivial"]:
            direction = "For"
        elif harms in ["Large", "Moderate"] and benefits in ["Small", "Trivial"]:
            direction = "Against"
        else:
            direction = "For"  # Default to conditional for

        # Determine strength
        if certainty in ["High", "Moderate"]:
            if direction == "For" and benefits in ["Large", "Moderate"] and harms in ["Small", "Trivial"]:
                strength = "Strong"
            elif direction == "Against" and harms in ["Large", "Moderate"] and benefits in ["Small", "Trivial"]:
                strength = "Strong"
            else:
                strength = "Conditional"
        else:
            strength = "Conditional"  # Low/Very Low certainty = always conditional

        etd.recommendation_direction = direction
        etd.recommendation_strength = strength

        return etd

    def format_etd_summary(self, etd: EvidenceToDecision) -> str:
        """Format EtD as markdown summary."""
        lines = [
            f"# Evidence-to-Decision Framework",
            f"",
            f"**Question:** {etd.question}",
            f"",
            f"**Population:** {etd.population or 'Not specified'}",
            f"**Intervention:** {etd.intervention or 'Not specified'}",
            f"**Comparator:** {etd.comparator or 'Not specified'}",
            f"",
            f"## Domain Judgments",
            f""
        ]

        for domain_id, judgment in etd.domain_judgments.items():
            lines.append(f"### {judgment.domain_name}")
            lines.append(f"**Judgment:** {judgment.judgment}")
            if judgment.rationale:
                lines.append(f"**Rationale:** {judgment.rationale}")
            lines.append("")

        lines.extend([
            f"## Recommendation",
            f"",
            f"**Direction:** {etd.recommendation_direction}",
            f"**Strength:** {etd.recommendation_strength}",
            f"",
            f"**{etd.get_full_recommendation_text()}**",
            f"",
            f"**Justification:** {etd.justification}",
        ])

        if etd.key_citations:
            lines.extend([
                f"",
                f"**Key Citations:** {', '.join(etd.key_citations)}"
            ])

        return "\n".join(lines)


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_service_instance: Optional[EtDService] = None


def get_etd_service() -> EtDService:
    """Get or create the singleton EtDService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = EtDService()
    return _service_instance
