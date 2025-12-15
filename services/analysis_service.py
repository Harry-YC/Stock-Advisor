"""
Analysis Service

Handles gap analysis, conflict detection, synthesis, and hypothesis extraction.
No UI dependencies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from config import settings


@dataclass
class SynthesisResult:
    """Result of meta-synthesis."""
    synthesis: str = ""
    consensus_points: List[str] = None
    open_questions: List[str] = None
    recommended_actions: List[str] = None

    def __post_init__(self):
        if self.consensus_points is None:
            self.consensus_points = []
        if self.open_questions is None:
            self.open_questions = []
        if self.recommended_actions is None:
            self.recommended_actions = []


class AnalysisService:
    """
    Analysis operations for expert panel discussions.

    Handles gap analysis, conflict detection, synthesis, and
    hypothesis extraction. No UI dependencies.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the analysis service.

        Args:
            api_key: OpenAI API key (optional)
        """
        self.api_key = api_key

    def analyze_gaps(
        self,
        responses: Dict[str, Dict],
        scenario: str
    ) -> Any:
        """
        Run gap analysis on discussion.

        Args:
            responses: Expert responses dict
            scenario: Scenario key (will be mapped to expected topics)

        Returns:
            GapAnalysisResult from gap_analyzer module
        """
        try:
            from core.analysis.gap_analyzer import analyze_panel_discussion
            from core.analysis.gap_analyzer import analyze_panel_discussion

            # Map GDG question types to analysis types
            scenario_mapping = {
                "surgical_candidate": "Surgical Candidacy Assessment",
                "palliative_pathway": "Palliative Care Planning",
                "intervention_choice": "Intervention Comparison",
                "symptom_management": "Symptom Control",
                "prognosis_assessment": "Prognosis & Outcomes",
                "ethics_review": "Ethics & Appropriateness",
                "resource_allocation": "Resource & Implementation",
                "general": "General Discussion"
            }
            mapped_scenario = scenario_mapping.get(scenario, "General Discussion")

            # Prepare discussion data
            discussion_list = [
                {'expert': exp, 'content': resp.get('content', '')}
                for exp, resp in responses.items()
            ]

            # Get OpenAI client
            from core.llm_utils import get_llm_client
            client = get_llm_client(api_key=self.api_key)

            return analyze_panel_discussion(
                discussion_data=discussion_list,
                scenario_type=mapped_scenario,
                llm_client=client
            )

        except ImportError:
            return None
        except Exception as e:
            raise RuntimeError(f"Gap analysis failed: {e}") from e

    def detect_conflicts(
        self,
        responses: Dict[str, Dict]
    ) -> Any:
        """
        Detect conflicts in expert responses.

        Args:
            responses: Expert responses dict

        Returns:
            ConflictAnalysisResult from conflict_detector module
        """
        try:
            from core.analysis.conflict_detector import detect_panel_conflicts
            from core.analysis.conflict_detector import detect_panel_conflicts

            discussion_list = [
                {'expert': exp, 'content': resp.get('content', '')}
                for exp, resp in responses.items()
            ]

            from core.llm_utils import get_llm_client
            client = get_llm_client(api_key=self.api_key)

            return detect_panel_conflicts(
                discussion_data=discussion_list,
                llm_client=client
            )

        except ImportError:
            return None
        except Exception as e:
            raise RuntimeError(f"Conflict detection failed: {e}") from e

    def synthesize_responses(
        self,
        responses: Dict[str, str],
        clinical_question: str,
        model: str = None
    ) -> SynthesisResult:
        """
        Generate meta-synthesis of expert responses.
        
        Args:
            responses: Dict mapping expert name to response content
            clinical_question: Research question
            model: Model to use for synthesis (defaults to REASONING_MODEL)
        """
        from config import settings
        model = model or settings.REASONING_MODEL

        try:
            from gdg import synthesize_expert_responses

            result = synthesize_expert_responses(
                expert_responses=responses,
                clinical_question=clinical_question,
                openai_api_key=self.api_key,
                model=model
            )

            return SynthesisResult(
                synthesis=result.get('synthesis', ''),
                consensus_points=result.get('consensus_points', []),
                open_questions=result.get('open_questions', []),
                recommended_actions=result.get('recommended_actions', [])
            )

        except Exception as e:
            raise RuntimeError(f"Synthesis failed: {e}") from e

    def extract_hypotheses(
        self,
        responses: Dict[str, str],
        clinical_question: str,
        round_num: int
    ) -> List[Dict]:
        """
        Extract hypotheses from expert discussion.

        Args:
            responses: Dict mapping expert name to response content
            clinical_question: Research question
            round_num: Current round number

        Returns:
            List of hypothesis dicts with 'hypothesis', 'evidence_strength',
            'evidence_type', 'supporting_experts', 'key_data', 'round'
        """
        try:
            from gdg import extract_hypotheses_from_discussion

            hypotheses = extract_hypotheses_from_discussion(
                expert_responses=responses,
                clinical_question=clinical_question,
                openai_api_key=self.api_key
            )

            # Add round number to each hypothesis
            for h in hypotheses:
                h['round'] = round_num

            return hypotheses

        except Exception as e:
            raise RuntimeError(f"Hypothesis extraction failed: {e}") from e

    def get_gap_coverage_score(self, gap_result: Any) -> float:
        """Get coverage score from gap analysis result."""
        if gap_result and hasattr(gap_result, 'coverage_score'):
            return gap_result.coverage_score
        return 0.0

    def get_gap_quantification_score(self, gap_result: Any) -> float:
        """Get quantification score from gap analysis result."""
        if gap_result and hasattr(gap_result, 'quantification_score'):
            return gap_result.quantification_score
        return 0.0

    def get_conflict_count(self, conflict_result: Any) -> int:
        """Get number of conflicts from conflict analysis result."""
        if conflict_result and hasattr(conflict_result, 'conflicts'):
            return len(conflict_result.conflicts)
        return 0

    def get_conflicts_by_severity(self, conflict_result: Any) -> Dict[str, List]:
        """
        Get conflicts grouped by severity.

        Returns:
            Dict with 'critical', 'moderate', 'minor' keys
        """
        result = {'critical': [], 'moderate': [], 'minor': []}

        if not conflict_result or not hasattr(conflict_result, 'conflicts'):
            return result

        for c in conflict_result.conflicts:
            severity = getattr(c, 'severity', 'minor')
            if severity in result:
                result[severity].append(c)

        return result
