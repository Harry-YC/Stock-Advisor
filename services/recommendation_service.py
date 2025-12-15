"""
Recommendation Service for Palliative Surgery GDG

Generates structured GRADE-style recommendations from GDG discussions.

Outputs:
- Recommendation statement
- Strength (Strong FOR/AGAINST, Conditional FOR/AGAINST)
- Evidence quality (High/Moderate/Low/Very Low)
- Benefits/Harms balance
- Key supporting PMIDs
- Applicability notes
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

from config import settings
from services.llm_router import LLMRouter, get_llm_router

logger = logging.getLogger(__name__)


@dataclass
class ClaimTier:
    """A single claim tier with its evidence basis."""
    claim: str  # The claim statement
    confidence: str  # HIGH | MODERATE | LOW | VERY LOW
    evidence_type: str  # "direct" | "indirect" | "extrapolated" | "expert_opinion"
    supporting_pmids: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_display(self) -> str:
        """Format for display in recommendation output."""
        pmid_str = ", ".join(self.supporting_pmids[:5]) if self.supporting_pmids else "None"
        evidence_label = {
            "direct": "Direct evidence",
            "indirect": "Indirect evidence",
            "extrapolated": "Extrapolated",
            "expert_opinion": "Expert opinion"
        }.get(self.evidence_type, self.evidence_type)

        return f"**{self.claim}**\n\n*Confidence: {self.confidence} ({evidence_label})*\n*PMIDs: {pmid_str}*"


@dataclass
class TieredRecommendation:
    """
    Multi-tiered recommendation that separates proven claims from extrapolations.

    This structure distinguishes:
    - Primary claim: What we're confident about (direct evidence)
    - Secondary claim: What's extrapolated (indirect evidence)
    - Caveats: Important limitations and conditions
    """
    primary_claim: ClaimTier  # What we're most confident about
    secondary_claims: List[ClaimTier] = field(default_factory=list)  # Extrapolated/indirect
    caveats: List[str] = field(default_factory=list)  # Important limitations
    comparator_match: Optional[Dict] = None  # Result from ComparatorMatcher
    overall_confidence: str = "MODERATE"  # Summary confidence level

    def to_dict(self) -> Dict[str, Any]:
        return {
            'primary_claim': self.primary_claim.to_dict(),
            'secondary_claims': [c.to_dict() for c in self.secondary_claims],
            'caveats': self.caveats,
            'comparator_match': self.comparator_match,
            'overall_confidence': self.overall_confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TieredRecommendation':
        """Create from dictionary."""
        primary = ClaimTier(**data['primary_claim'])
        secondary = [ClaimTier(**c) for c in data.get('secondary_claims', [])]
        return cls(
            primary_claim=primary,
            secondary_claims=secondary,
            caveats=data.get('caveats', []),
            comparator_match=data.get('comparator_match'),
            overall_confidence=data.get('overall_confidence', 'MODERATE')
        )

    def format_full_recommendation(self) -> str:
        """Format as readable tiered recommendation."""
        lines = []

        # Overall confidence badge
        confidence_emoji = {
            "HIGH": "🟢",
            "MODERATE": "🟡",
            "LOW": "🟠",
            "VERY LOW": "🔴"
        }.get(self.overall_confidence, "⚪")

        lines.append(f"## Recommendation {confidence_emoji} ({self.overall_confidence} confidence)")
        lines.append("")

        # Primary claim
        lines.append("### Primary Claim (Direct Evidence)")
        lines.append(self.primary_claim.format_display())
        lines.append("")

        # Secondary claims
        if self.secondary_claims:
            lines.append("### Secondary Claims (Indirect/Extrapolated)")
            for i, claim in enumerate(self.secondary_claims, 1):
                lines.append(f"**{i}.** {claim.format_display()}")
                lines.append("")

        # Caveats
        if self.caveats:
            lines.append("### Caveats & Limitations")
            for caveat in self.caveats:
                lines.append(f"- {caveat}")
            lines.append("")

        # Comparator match warning
        if self.comparator_match and self.comparator_match.get('warning_message'):
            lines.append("### ⚠️ Evidence Directness Warning")
            lines.append(self.comparator_match['warning_message'])
            lines.append("")

        return "\n".join(lines)


@dataclass
class Recommendation:
    """
    Structured GRADE-style recommendation.

    Following GRADE Working Group methodology:
    - Clear actionable statement
    - Strength based on confidence in effect estimate and values/preferences
    - Evidence quality based on study design and risk of bias
    """
    statement: str  # The actual recommendation text
    strength: str  # "Strong FOR" | "Conditional FOR" | "Conditional AGAINST" | "Strong AGAINST"
    evidence_quality: str  # "High" | "Moderate" | "Low" | "Very Low"
    benefits: List[str] = field(default_factory=list)
    harms: List[str] = field(default_factory=list)
    key_citations: List[str] = field(default_factory=list)  # PMIDs
    rationale: str = ""
    population: Optional[str] = None  # Who this applies to
    intervention: Optional[str] = None  # What is being recommended
    comparator: Optional[str] = None  # What it's being compared to
    outcomes: List[str] = field(default_factory=list)  # Key outcomes considered
    alternatives: Optional[List[str]] = None  # Other options considered
    implementation_notes: Optional[str] = None  # Practical guidance
    research_gaps: Optional[List[str]] = None  # Evidence gaps identified
    # New: Tiered recommendation structure
    tiered: Optional[TieredRecommendation] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Recommendation':
        """Create from dictionary."""
        return cls(**data)

    def format_grade_summary(self) -> str:
        """Format as GRADE summary table row."""
        return f"""| {self.population or 'General'} | {self.intervention or 'Intervention'} | {self.statement[:50]}... | {self.strength} | {self.evidence_quality} |"""

    def format_full_recommendation(self) -> str:
        """Format as full recommendation text."""
        lines = []

        # Header with strength badge
        strength_emoji = {
            "Strong FOR": "++",
            "Conditional FOR": "+?",
            "Conditional AGAINST": "-?",
            "Strong AGAINST": "--"
        }.get(self.strength, "?")

        lines.append(f"## Recommendation [{strength_emoji}]")
        lines.append("")
        lines.append(f"**{self.statement}**")
        lines.append("")

        # GRADE panel
        lines.append(f"| Strength | Evidence Quality |")
        lines.append(f"|----------|------------------|")
        lines.append(f"| {self.strength} | {self.evidence_quality} |")
        lines.append("")

        # Benefits and harms
        if self.benefits:
            lines.append("### Benefits")
            for b in self.benefits:
                lines.append(f"- {b}")
            lines.append("")

        if self.harms:
            lines.append("### Harms/Burdens")
            for h in self.harms:
                lines.append(f"- {h}")
            lines.append("")

        # Rationale
        if self.rationale:
            lines.append("### Rationale")
            lines.append(self.rationale)
            lines.append("")

        # Key evidence
        if self.key_citations:
            lines.append("### Key Evidence")
            lines.append(f"PMIDs: {', '.join(self.key_citations)}")
            lines.append("")

        # Implementation
        if self.implementation_notes:
            lines.append("### Implementation Considerations")
            lines.append(self.implementation_notes)
            lines.append("")

        # Research gaps
        if self.research_gaps:
            lines.append("### Research Gaps")
            for gap in self.research_gaps:
                lines.append(f"- {gap}")
            lines.append("")

        return "\n".join(lines)


# System prompt for recommendation extraction
RECOMMENDATION_SYSTEM_PROMPT = """You are a GRADE methodologist synthesizing a GDG (Guideline Development Group) discussion into a formal clinical recommendation.

Your task is to extract a structured recommendation following GRADE methodology:

1. **Recommendation Statement**: Clear, actionable statement starting with "We recommend..." or "We suggest..."
   - "We recommend" = Strong recommendation
   - "We suggest" = Conditional recommendation

2. **Strength of Recommendation**:
   - "Strong FOR": Benefits clearly outweigh harms, most patients should receive intervention
   - "Conditional FOR": Benefits probably outweigh harms, most patients would want intervention but many would not
   - "Conditional AGAINST": Harms probably outweigh benefits
   - "Strong AGAINST": Harms clearly outweigh benefits

3. **Evidence Quality** (GRADE certainty):
   - "High": Very confident the true effect is close to estimate
   - "Moderate": Moderately confident; true effect likely close to estimate
   - "Low": Limited confidence; true effect may be substantially different
   - "Very Low": Very little confidence; true effect likely substantially different

4. **Benefits**: List desirable outcomes supported by evidence
5. **Harms**: List undesirable outcomes, burdens, costs
6. **Key Citations**: Extract PMIDs mentioned that support the recommendation
7. **Rationale**: Brief explanation of why this recommendation was made
8. **Population**: Who this applies to
9. **Alternatives**: Other options considered
10. **Research Gaps**: What evidence is missing

Output JSON in this exact format:
{
    "statement": "We recommend/suggest...",
    "strength": "Strong FOR|Conditional FOR|Conditional AGAINST|Strong AGAINST",
    "evidence_quality": "High|Moderate|Low|Very Low",
    "benefits": ["benefit 1", "benefit 2"],
    "harms": ["harm 1", "harm 2"],
    "key_citations": ["12345678", "87654321"],
    "rationale": "Brief rationale...",
    "population": "Patients with...",
    "intervention": "The intervention being recommended",
    "comparator": "What it's compared to",
    "outcomes": ["Key outcome 1", "Key outcome 2"],
    "alternatives": ["Alternative 1", "Alternative 2"],
    "implementation_notes": "Practical guidance...",
    "research_gaps": ["Gap 1", "Gap 2"]
}"""


class RecommendationService:
    """
    Service for generating structured GRADE-style recommendations.

    Usage:
        service = RecommendationService()
        recommendation = service.generate_recommendation(
            question="Should we...",
            expert_responses={"Expert A": "...", "Expert B": "..."},
            chair_synthesis="The GDG concludes..."
        )
    """

    def __init__(self, llm_router: Optional[LLMRouter] = None):
        """
        Initialize the recommendation service.

        Args:
            llm_router: LLM Router instance (uses default if not provided)
        """
        self.router = llm_router or get_llm_router()

    def generate_recommendation(
        self,
        question: str,
        expert_responses: Dict[str, str],
        chair_synthesis: Optional[str] = None,
        included_citations: Optional[List[Dict]] = None,
        question_type: Optional[str] = None
    ) -> Recommendation:
        """
        Generate a structured GRADE-style recommendation from GDG discussion.

        Args:
            question: The clinical question being addressed
            expert_responses: Dict mapping expert name to their response text
            chair_synthesis: Optional GDG Chair synthesis (if available)
            included_citations: List of citation dicts with PMID, title, etc.
            question_type: Type of question (surgical_candidate, palliative_pathway, etc.)

        Returns:
            Recommendation dataclass with structured output
        """
        # Build the extraction prompt
        prompt = self._build_extraction_prompt(
            question=question,
            expert_responses=expert_responses,
            chair_synthesis=chair_synthesis,
            included_citations=included_citations,
            question_type=question_type
        )

        try:
            # Call LLM for extraction
            response = self.router.call_synthesis(
                prompt=prompt,
                system=RECOMMENDATION_SYSTEM_PROMPT,
                temperature=0.3,  # Lower temperature for structured extraction
                max_tokens=2000
            )

            # Parse the response
            return self._parse_recommendation(response.content, included_citations)

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            # Return a minimal recommendation on error
            return Recommendation(
                statement=f"Unable to generate recommendation: {str(e)}",
                strength="Conditional FOR",
                evidence_quality="Very Low",
                rationale="Recommendation extraction failed"
            )

    def _build_extraction_prompt(
        self,
        question: str,
        expert_responses: Dict[str, str],
        chair_synthesis: Optional[str],
        included_citations: Optional[List[Dict]],
        question_type: Optional[str]
    ) -> str:
        """Build the prompt for recommendation extraction."""
        sections = []

        # Clinical question
        sections.append(f"## Clinical Question\n{question}")

        # Question type context
        if question_type:
            type_context = {
                "surgical_candidate": "This is a surgical candidacy assessment question.",
                "palliative_pathway": "This is a palliative care pathway design question.",
                "intervention_choice": "This is a comparative intervention question (surgery vs non-surgical).",
                "symptom_management": "This is a symptom management question.",
                "prognosis_assessment": "This is a prognosis/outcomes assessment question.",
                "ethics_review": "This is an ethics and appropriateness question.",
                "resource_allocation": "This is a resource allocation/implementation question.",
                "general": "This is a general GDG question."
            }.get(question_type, "")
            if type_context:
                sections.append(f"## Question Type\n{type_context}")

        # Expert responses
        sections.append("## Expert Responses")
        for expert_name, response in expert_responses.items():
            # Truncate very long responses
            truncated = response[:3000] if len(response) > 3000 else response
            sections.append(f"### {expert_name}\n{truncated}")

        # Chair synthesis (if available)
        if chair_synthesis:
            sections.append(f"## GDG Chair Synthesis\n{chair_synthesis}")

        # Available citations
        if included_citations:
            sections.append("## Available Evidence")
            for cit in included_citations[:10]:  # Limit to 10
                pmid = cit.get('pmid', cit.pmid if hasattr(cit, 'pmid') else 'N/A')
                title = cit.get('title', cit.title if hasattr(cit, 'title') else 'Untitled')
                sections.append(f"- PMID {pmid}: {title[:100]}")

        # Instruction
        sections.append("""
## Task
Based on the GDG discussion above, extract a structured GRADE-style recommendation.
Focus on:
1. What the experts agree on
2. The balance of benefits vs harms
3. The quality of supporting evidence
4. Any conditions or qualifications

Output ONLY valid JSON matching the specified format.""")

        return "\n\n".join(sections)

    def _parse_recommendation(
        self,
        response_text: str,
        included_citations: Optional[List[Dict]]
    ) -> Recommendation:
        """Parse LLM response into Recommendation dataclass."""
        try:
            # Try to extract JSON from response
            # Handle cases where JSON is wrapped in markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")

            data = json.loads(json_str)

            # Validate and normalize strength
            valid_strengths = ["Strong FOR", "Conditional FOR", "Conditional AGAINST", "Strong AGAINST"]
            strength = data.get('strength', 'Conditional FOR')
            if strength not in valid_strengths:
                # Try to map common variations
                strength_lower = strength.lower()
                if 'strong' in strength_lower and 'for' in strength_lower:
                    strength = "Strong FOR"
                elif 'strong' in strength_lower and 'against' in strength_lower:
                    strength = "Strong AGAINST"
                elif 'conditional' in strength_lower and 'against' in strength_lower:
                    strength = "Conditional AGAINST"
                else:
                    strength = "Conditional FOR"

            # Validate evidence quality
            valid_qualities = ["High", "Moderate", "Low", "Very Low"]
            quality = data.get('evidence_quality', 'Low')
            if quality not in valid_qualities:
                quality_lower = quality.lower()
                if 'high' in quality_lower:
                    quality = "High"
                elif 'moderate' in quality_lower:
                    quality = "Moderate"
                elif 'very' in quality_lower:
                    quality = "Very Low"
                else:
                    quality = "Low"

            # Validate PMIDs against available citations
            key_citations = data.get('key_citations', [])
            if included_citations:
                available_pmids = {
                    str(c.get('pmid', c.pmid if hasattr(c, 'pmid') else ''))
                    for c in included_citations
                }
                # Filter to only valid PMIDs
                key_citations = [p for p in key_citations if str(p) in available_pmids]

            return Recommendation(
                statement=data.get('statement', 'No recommendation statement generated'),
                strength=strength,
                evidence_quality=quality,
                benefits=data.get('benefits', []),
                harms=data.get('harms', []),
                key_citations=key_citations,
                rationale=data.get('rationale', ''),
                population=data.get('population'),
                intervention=data.get('intervention'),
                comparator=data.get('comparator'),
                outcomes=data.get('outcomes', []),
                alternatives=data.get('alternatives'),
                implementation_notes=data.get('implementation_notes'),
                research_gaps=data.get('research_gaps')
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse recommendation JSON: {e}")
            # Create a basic recommendation from the text
            return Recommendation(
                statement=response_text[:500] if response_text else "Unable to parse recommendation",
                strength="Conditional FOR",
                evidence_quality="Low",
                rationale="Recommendation was not in expected JSON format"
            )

    def format_recommendation_card(self, recommendation: Recommendation) -> str:
        """
        Format recommendation as a styled card for Streamlit display.

        Returns HTML/Markdown suitable for st.markdown with unsafe_allow_html=True.
        """
        # Strength color coding
        strength_colors = {
            "Strong FOR": ("#28a745", "white"),  # Green
            "Conditional FOR": ("#ffc107", "black"),  # Yellow
            "Conditional AGAINST": ("#fd7e14", "white"),  # Orange
            "Strong AGAINST": ("#dc3545", "white")  # Red
        }
        bg_color, text_color = strength_colors.get(recommendation.strength, ("#6c757d", "white"))

        # Evidence quality badge
        quality_colors = {
            "High": "#28a745",
            "Moderate": "#17a2b8",
            "Low": "#ffc107",
            "Very Low": "#dc3545"
        }
        quality_color = quality_colors.get(recommendation.evidence_quality, "#6c757d")

        html = f"""
<div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px 0; background: #f8f9fa;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <span style="background: {bg_color}; color: {text_color}; padding: 4px 12px; border-radius: 4px; font-weight: bold;">
            {recommendation.strength}
        </span>
        <span style="background: {quality_color}; color: white; padding: 4px 12px; border-radius: 4px; font-size: 0.85rem;">
            Evidence: {recommendation.evidence_quality}
        </span>
    </div>
    <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 12px;">
        {recommendation.statement}
    </div>
    <div style="font-size: 0.9rem; color: #666;">
        {recommendation.rationale[:200] + '...' if len(recommendation.rationale) > 200 else recommendation.rationale}
    </div>
</div>
"""
        return html


# Singleton instance for convenience
_service_instance: Optional[RecommendationService] = None


def get_recommendation_service() -> RecommendationService:
    """Get or create the singleton RecommendationService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = RecommendationService()
    return _service_instance
