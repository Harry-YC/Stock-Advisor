"""
Gap Analysis Engine for GDG Expert Panel Discussions

Analyzes discussion completeness, evidence quality, and quantification.
Adapted for Palliative Surgery Guideline Development Group (GDG) context.
"""

import re
import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GapAnalysis:
    """Results of gap analysis"""
    strengths: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    evidence_issues: List[Dict] = field(default_factory=list)
    coverage_score: float = 0.0
    quantification_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strengths": self.strengths,
            "gaps": self.gaps,
            "evidence_issues": self.evidence_issues,
            "coverage_score": self.coverage_score,
            "quantification_score": self.quantification_score,
            "recommendations": self.recommendations
        }


# Expected topics by discussion scenario (Palliative Surgery GDG)
EXPECTED_TOPICS = {
    "surgical_candidate": [
        "patient_selection_criteria",
        "performance_status",
        "frailty_assessment",
        "prognosis_estimation",
        "goals_of_care",
        "operative_risk",
        "expected_benefit",
        "alternative_options"
    ],
    "palliative_pathway": [
        "symptom_assessment",
        "quality_of_life",
        "care_coordination",
        "goals_of_care",
        "treatment_options",
        "expected_outcomes",
        "patient_preferences",
        "caregiver_burden"
    ],
    "intervention_choice": [
        "surgical_outcomes",
        "non_surgical_alternatives",
        "comparative_effectiveness",
        "complication_rates",
        "recovery_time",
        "reintervention_risk",
        "cost_effectiveness",
        "patient_selection"
    ],
    "symptom_management": [
        "symptom_etiology",
        "severity_assessment",
        "treatment_options",
        "evidence_base",
        "expected_response",
        "side_effects",
        "quality_of_life",
        "goals_alignment"
    ],
    "prognosis_assessment": [
        "prognostic_factors",
        "survival_estimates",
        "functional_trajectory",
        "evidence_quality",
        "uncertainty_acknowledgment",
        "patient_communication",
        "decision_thresholds",
        "outcome_metrics"
    ],
    "ethics_review": [
        "patient_autonomy",
        "beneficence",
        "non_maleficence",
        "proportionality",
        "informed_consent",
        "surrogate_decision_making",
        "resource_allocation",
        "appropriateness"
    ],
    "resource_allocation": [
        "cost_effectiveness",
        "resource_utilization",
        "implementation_barriers",
        "healthcare_system_impact",
        "equity_considerations",
        "opportunity_cost",
        "sustainability",
        "outcome_metrics"
    ],
    "General Discussion": [
        "key_findings",
        "evidence_gaps",
        "recommendations",
        "implementation_notes",
        "research_priorities"
    ]
}


class GapAnalyzer:
    """Analyzes expert panel discussions for completeness and quality"""

    def __init__(self, llm_client=None, model: str = "gpt-5-mini"):
        """
        Initialize GapAnalyzer.

        Args:
            llm_client: OpenAI client (optional, for AI-powered analysis)
            model: Model to use for AI analysis
        """
        self.client = llm_client
        self.model = model

    def analyze_discussion(
        self,
        scenario_type: str,
        discussion_data: List[Dict],
        use_ai: bool = False
    ) -> GapAnalysis:
        """
        Run comprehensive gap analysis on panel discussion.

        Args:
            scenario_type: Type of discussion scenario
            discussion_data: List of expert responses
            use_ai: Whether to use AI for deeper analysis

        Returns:
            GapAnalysis with strengths, gaps, and recommendations
        """
        # 1. Coverage check
        coverage = self._check_topic_coverage(scenario_type, discussion_data, use_ai)

        # 2. Quantification check
        quant_check = self._check_quantification(discussion_data)

        # 3. Evidence quality check
        evidence_issues = self._check_evidence_quality(discussion_data)

        # Add quantification issues to evidence_issues
        for vague in quant_check.get('vague_responses', []):
            evidence_issues.append({
                'expert': vague['expert'],
                'excerpt': vague['excerpt'],
                'issue_type': 'lacks_quantification',
                'severity': 'high',
                'details': ', '.join(vague['issues'])
            })

        # 4. Generate recommendations
        recommendations = self._generate_recommendations(
            coverage, evidence_issues, scenario_type, quant_check
        )

        # Calculate combined score (60% coverage + 40% quantification)
        combined_score = (coverage['score'] * 0.6 + quant_check['score'] * 0.4)

        return GapAnalysis(
            strengths=coverage['covered'],
            gaps=coverage['missing'],
            evidence_issues=evidence_issues,
            coverage_score=combined_score,
            quantification_score=quant_check['score'],
            recommendations=recommendations
        )

    def _check_topic_coverage(
        self,
        scenario_type: str,
        discussion_data: List[Dict],
        use_ai: bool = False
    ) -> Dict:
        """Check if expected topics were addressed"""

        expected = set(EXPECTED_TOPICS.get(scenario_type, EXPECTED_TOPICS["General Discussion"]))

        if not expected:
            return {'covered': [], 'missing': [], 'partial': [], 'score': 1.0}

        # Concatenate all expert responses
        full_discussion = "\n".join([
            f"{item.get('expert', 'Unknown')}: {item.get('content', '')}"
            for item in discussion_data
        ])

        if use_ai and self.client:
            return self._ai_coverage_check(expected, full_discussion)

        # Keyword-based coverage check (fallback)
        return self._keyword_coverage_check(expected, full_discussion)

    def _keyword_coverage_check(self, expected: set, discussion_text: str) -> Dict:
        """Simple keyword-based coverage check"""
        covered = []
        missing = []
        partial = []

        # Map topics to keywords (Palliative Surgery GDG context)
        topic_keywords = {
            # Surgical candidate assessment
            "patient_selection_criteria": ["selection", "criteria", "candidate", "appropriate", "eligible"],
            "performance_status": ["performance", "ecog", "karnofsky", "kps", "functional status"],
            "frailty_assessment": ["frailty", "frail", "cga", "geriatric", "weakness"],
            "prognosis_estimation": ["prognosis", "survival", "life expectancy", "months", "weeks"],
            "goals_of_care": ["goals", "care", "patient wishes", "preferences", "advance"],
            "operative_risk": ["operative", "surgical risk", "mortality", "morbidity", "asa"],
            "expected_benefit": ["benefit", "symptom relief", "palliation", "improvement"],
            "alternative_options": ["alternative", "non-surgical", "conservative", "stent", "medical"],

            # Palliative pathway
            "symptom_assessment": ["symptom", "pain", "obstruction", "bleeding", "assessment"],
            "quality_of_life": ["quality of life", "qol", "hrqol", "functional", "wellbeing"],
            "care_coordination": ["coordination", "multidisciplinary", "palliative care", "hospice"],
            "treatment_options": ["treatment", "option", "approach", "intervention", "management"],
            "expected_outcomes": ["outcome", "result", "success", "resolution", "improvement"],
            "patient_preferences": ["preference", "patient wishes", "autonomy", "values"],
            "caregiver_burden": ["caregiver", "family", "burden", "support", "days at home"],

            # Intervention choice
            "surgical_outcomes": ["surgical outcome", "operative", "postoperative", "30-day"],
            "non_surgical_alternatives": ["stent", "embolization", "conservative", "medical management"],
            "comparative_effectiveness": ["comparison", "versus", "vs", "comparative", "head-to-head"],
            "complication_rates": ["complication", "adverse", "morbidity", "leak", "infection"],
            "recovery_time": ["recovery", "hospital stay", "los", "discharge", "return"],
            "reintervention_risk": ["reintervention", "revision", "reoperation", "recurrence"],
            "cost_effectiveness": ["cost", "economic", "icer", "qaly", "resource"],

            # Symptom management
            "symptom_etiology": ["etiology", "cause", "mechanism", "pathophysiology"],
            "severity_assessment": ["severity", "grade", "score", "scale", "intensity"],
            "evidence_base": ["evidence", "study", "trial", "pmid", "meta-analysis"],
            "expected_response": ["response", "relief", "resolution", "improvement"],
            "side_effects": ["side effect", "adverse", "toxicity", "complication"],
            "goals_alignment": ["aligned", "appropriate", "proportionate", "consistent"],

            # Prognosis assessment
            "prognostic_factors": ["prognostic", "predictor", "factor", "variable"],
            "survival_estimates": ["survival", "median", "months", "weeks", "prognosis"],
            "functional_trajectory": ["trajectory", "decline", "functional", "course"],
            "evidence_quality": ["evidence quality", "grade", "certainty", "moderate", "low"],
            "uncertainty_acknowledgment": ["uncertainty", "unclear", "limited", "insufficient"],
            "patient_communication": ["communication", "discussion", "inform", "shared"],
            "decision_thresholds": ["threshold", "cutoff", "criteria", "indication"],
            "outcome_metrics": ["outcome", "endpoint", "measure", "metric"],

            # Ethics review
            "patient_autonomy": ["autonomy", "self-determination", "consent", "wishes"],
            "beneficence": ["beneficence", "benefit", "best interest", "help"],
            "non_maleficence": ["non-maleficence", "harm", "do no harm", "avoid"],
            "proportionality": ["proportionate", "proportionality", "balance", "appropriate"],
            "informed_consent": ["informed consent", "consent", "disclosure", "understanding"],
            "surrogate_decision_making": ["surrogate", "proxy", "family", "decision maker"],
            "appropriateness": ["appropriate", "indication", "justified", "warranted"],

            # Resource allocation
            "resource_utilization": ["resource", "utilization", "capacity", "availability"],
            "implementation_barriers": ["barrier", "implementation", "challenge", "feasibility"],
            "healthcare_system_impact": ["system", "healthcare", "hospital", "service"],
            "equity_considerations": ["equity", "access", "disparity", "underserved"],
            "opportunity_cost": ["opportunity cost", "trade-off", "alternative use"],
            "sustainability": ["sustainable", "long-term", "maintenance", "ongoing"],

            # General
            "key_findings": ["finding", "result", "key", "main", "conclusion"],
            "evidence_gaps": ["gap", "missing", "need", "unknown", "unclear"],
            "recommendations": ["recommend", "suggest", "advise", "guidance"],
            "implementation_notes": ["implementation", "practical", "clinical practice"],
            "research_priorities": ["research", "study needed", "future", "investigate"]
        }

        discussion_lower = discussion_text.lower()

        for topic in expected:
            keywords = topic_keywords.get(topic, [topic.replace("_", " ")])
            matches = sum(1 for kw in keywords if kw in discussion_lower)

            if matches >= 2:
                covered.append(topic)
            elif matches == 1:
                partial.append(topic)
            else:
                missing.append(topic)

        total = len(expected)
        score = (len(covered) + 0.5 * len(partial)) / total if total > 0 else 0

        return {
            'covered': covered,
            'missing': missing,
            'partial': partial,
            'score': score
        }

    def _ai_coverage_check(self, expected: set, discussion_text: str) -> Dict:
        """AI-powered coverage check using LLM"""
        system_prompt = """You are analyzing an expert panel discussion.
Given a list of expected topics and the full discussion transcript,
identify which topics were substantially addressed.

Return ONLY a valid JSON object with this exact format:
{
    "covered": ["topic1", "topic2"],
    "missing": ["topic3"],
    "partial": ["topic4"]
}"""

        user_prompt = f"""Expected topics: {list(expected)}

Discussion transcript (first 4000 chars):
{discussion_text[:4000]}

Which topics were covered? Return JSON only."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )

            result = response.choices[0].message.content.strip()

            # Clean JSON from markdown
            if '```json' in result:
                result = result.split('```json')[1].split('```')[0].strip()
            elif '```' in result:
                result = result.split('```')[1].split('```')[0].strip()

            coverage_data = json.loads(result)

            covered_count = len(coverage_data.get('covered', []))
            partial_count = len(coverage_data.get('partial', []))
            total_count = len(expected)
            score = (covered_count + 0.5 * partial_count) / total_count if total_count > 0 else 0

            return {
                'covered': coverage_data.get('covered', []),
                'missing': coverage_data.get('missing', []),
                'partial': coverage_data.get('partial', []),
                'score': score
            }

        except Exception as e:
            logger.error(f"AI coverage analysis failed: {e}")
            return self._keyword_coverage_check(expected, discussion_text)

    def _check_quantification(self, discussion_data: List[Dict]) -> Dict:
        """Check if responses contain actual numbers vs vague statements"""

        quantification_score = 0
        total_responses = 0
        vague_responses = []

        for item in discussion_data:
            total_responses += 1
            content = item.get('content', '')

            # Check for quantitative elements
            has_numbers = bool(re.search(
                r'\d+\.?\d*\s*%|\d+\s*(patients?|subjects?|mg|µg|mcg|years?|months?|weeks?)',
                content,
                re.IGNORECASE
            ))

            has_citations = bool(re.search(
                r'(Study|Trial|et al|PMID|Phase \d|[A-Z][a-z]+\s+\d{4})',
                content
            ))

            has_comparators = bool(re.search(
                r'vs\.?|compared to|versus|benchmark|relative to',
                content,
                re.IGNORECASE
            ))

            score = sum([has_numbers, has_citations, has_comparators])
            quantification_score += score

            if score < 2:
                issues = []
                if not has_numbers:
                    issues.append('No quantitative data')
                if not has_citations:
                    issues.append('No specific sources/citations')
                if not has_comparators:
                    issues.append('No benchmarks/comparators')

                vague_responses.append({
                    'expert': item.get('expert', 'Unknown'),
                    'excerpt': content[:200] + '...' if len(content) > 200 else content,
                    'issues': issues
                })

        return {
            'score': quantification_score / (total_responses * 3) if total_responses > 0 else 0,
            'vague_responses': vague_responses
        }

    def _check_evidence_quality(self, discussion_data: List[Dict]) -> List[Dict]:
        """Check for unsupported claims or weak evidence"""
        issues = []

        # Pattern for vague claims
        vague_patterns = [
            (r'(may|might|could|should)\s+(be|have|show)', 'speculative_language'),
            (r'(probably|likely|possibly|potentially)', 'uncertain_claim'),
            (r'(some|many|several)\s+(studies|reports|data)', 'vague_reference'),
            (r'(generally|typically|usually|often)', 'non_specific'),
        ]

        for item in discussion_data:
            content = item.get('content', '')
            expert = item.get('expert', 'Unknown')

            for pattern, issue_type in vague_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if len(matches) >= 2:  # Multiple instances suggest pattern
                    issues.append({
                        'expert': expert,
                        'statement': content[:150] + '...',
                        'issue_type': issue_type,
                        'severity': 'moderate'
                    })
                    break  # One issue per expert max

        return issues

    def _generate_recommendations(
        self,
        coverage: Dict,
        evidence_issues: List[Dict],
        scenario_type: str,
        quant_check: Dict
    ) -> List[str]:
        """Generate specific recommendations for improving the discussion"""
        recommendations = []

        # Critical: Low quantification
        if quant_check['score'] < 0.5:
            recommendations.append(
                f"CRITICAL: Only {quant_check['score']:.0%} of responses include specific "
                f"numbers/citations. Request experts provide quantitative data."
            )

        # Recommendations for missing topics
        for missing_topic in coverage.get('missing', [])[:3]:
            topic_readable = missing_topic.replace('_', ' ').title()
            recommendations.append(f"Address missing topic: {topic_readable}")

        # Recommendations for partial coverage
        for partial_topic in coverage.get('partial', [])[:2]:
            topic_readable = partial_topic.replace('_', ' ').title()
            recommendations.append(f"Expand discussion on: {topic_readable}")

        # Evidence quality recommendations
        high_severity = [i for i in evidence_issues if i.get('severity') == 'high']
        if high_severity:
            recommendations.append(
                f"Request supporting data for {len(high_severity)} vague claims"
            )

        # Coverage-based recommendations
        if coverage['score'] < 0.5:
            recommendations.append(
                "Consider adding more expert perspectives to cover missing areas"
            )
        elif coverage['score'] > 0.8:
            recommendations.append(
                "Strong coverage achieved - focus on deepening key topics in next round"
            )

        if not recommendations:
            recommendations.append("Discussion shows good coverage - ready for synthesis")

        return recommendations


def analyze_panel_discussion(
    discussion_data: List[Dict],
    scenario_type: str = "General Discussion",
    llm_client=None
) -> GapAnalysis:
    """
    Convenience function to analyze a panel discussion.

    Args:
        discussion_data: List of dicts with 'expert' and 'content' keys
        scenario_type: Type of discussion scenario
        llm_client: Optional OpenAI client for AI-powered analysis

    Returns:
        GapAnalysis object with results
    """
    analyzer = GapAnalyzer(llm_client=llm_client)
    return analyzer.analyze_discussion(
        scenario_type=scenario_type,
        discussion_data=discussion_data,
        use_ai=llm_client is not None
    )
