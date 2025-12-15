"""
Conflict Detection Module for Expert Panel Discussions

Identifies contradictions, divergent views, and disagreements between experts.
Creates decision memos that preserve expert divergence for transparency.
"""

import re
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Conflict:
    """Represents a detected conflict between experts"""
    metric: str  # What they disagree on
    values: Dict[str, str]  # expert_name: their position
    rationale: str  # Why they differ (if detectable)
    severity: str  # 'critical', 'moderate', 'minor'
    category: str  # 'efficacy', 'safety', 'design', 'regulatory', 'timeline', 'commercial'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "values": self.values,
            "rationale": self.rationale,
            "severity": self.severity,
            "category": self.category
        }


@dataclass
class ConflictAnalysis:
    """Results of conflict detection"""
    conflicts: List[Conflict] = field(default_factory=list)
    agreement_areas: List[str] = field(default_factory=list)
    clarification_needed: List[str] = field(default_factory=list)
    decision_memo: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflicts": [c.to_dict() for c in self.conflicts],
            "agreement_areas": self.agreement_areas,
            "clarification_needed": self.clarification_needed,
            "decision_memo": self.decision_memo
        }


class ConflictDetector:
    """Detects and analyzes conflicts in expert panel discussions"""

    # Categories for classifying conflicts
    CONFLICT_CATEGORIES = {
        'efficacy': ['orr', 'response', 'efficacy', 'pfs', 'os', 'survival', 'benefit'],
        'safety': ['safety', 'toxicity', 'adverse', 'risk', 'dlt', 'tolerability'],
        'design': ['design', 'endpoint', 'sample size', 'arm', 'control', 'randomization'],
        'regulatory': ['fda', 'ema', 'approval', 'regulatory', 'pathway', 'accelerated'],
        'timeline': ['timeline', 'months', 'years', 'duration', 'delay'],
        'commercial': ['market', 'commercial', 'pricing', 'competition', 'differentiation']
    }

    def __init__(self, llm_client=None, model: str = None):
        """
        Initialize ConflictDetector.

        Args:
            llm_client: OpenAI client (optional, for AI-powered analysis)
            model: Model to use for AI analysis
        """
        from config import settings
        self.client = llm_client
        self.model = model or settings.REASONING_MODEL

    def detect_conflicts(
        self,
        discussion_data: List[Dict],
        use_ai: bool = False
    ) -> List[Conflict]:
        """
        Detect conflicts between expert positions.

        Args:
            discussion_data: List of expert responses
            use_ai: Whether to use AI for detection

        Returns:
            List of detected Conflict objects
        """
        if len(discussion_data) < 2:
            return []

        if use_ai and self.client:
            return self._ai_conflict_detection(discussion_data)

        return self._pattern_conflict_detection(discussion_data)

    def _pattern_conflict_detection(self, discussion_data: List[Dict]) -> List[Conflict]:
        """Rule-based conflict detection using patterns"""
        conflicts = []

        # Extract quantitative claims from each expert
        expert_claims = {}
        for item in discussion_data:
            expert = item.get('expert', 'Unknown')
            content = item.get('content', '')
            claims = self._extract_quantitative_claims(content)
            expert_claims[expert] = claims

        # Compare claims across experts
        all_metrics = set()
        for claims in expert_claims.values():
            all_metrics.update(claims.keys())

        for metric in all_metrics:
            values = {}
            for expert, claims in expert_claims.items():
                if metric in claims:
                    values[expert] = claims[metric]

            if len(values) >= 2:
                # Check if values conflict
                conflict = self._check_value_conflict(metric, values)
                if conflict:
                    conflicts.append(conflict)

        # Look for explicit disagreement language
        disagreement_conflicts = self._detect_disagreement_language(discussion_data)
        conflicts.extend(disagreement_conflicts)

        return conflicts

    def _extract_quantitative_claims(self, content: str) -> Dict[str, str]:
        """Extract quantitative claims from text"""
        claims = {}

        # ORR patterns
        orr_match = re.search(r'ORR[:\s]+(\d+[-–]?\d*%?)', content, re.IGNORECASE)
        if orr_match:
            claims['ORR'] = orr_match.group(1)

        # PFS patterns
        pfs_match = re.search(r'PFS[:\s]+(\d+\.?\d*\s*months?)', content, re.IGNORECASE)
        if pfs_match:
            claims['PFS'] = pfs_match.group(1)

        # OS patterns
        os_match = re.search(r'OS[:\s]+(\d+\.?\d*\s*months?)', content, re.IGNORECASE)
        if os_match:
            claims['OS'] = os_match.group(1)

        # Sample size patterns
        n_match = re.search(r'[Nn][=:]\s*(\d+)', content)
        if n_match:
            claims['sample_size'] = n_match.group(1)

        # Timeline patterns
        timeline_match = re.search(r'(\d+[-–]\d+)\s*(months?|years?)', content, re.IGNORECASE)
        if timeline_match:
            claims['timeline'] = f"{timeline_match.group(1)} {timeline_match.group(2)}"

        # Probability/confidence patterns
        prob_match = re.search(r'(\d+[-–]?\d*%)\s*(probability|chance|likelihood|confidence)', content, re.IGNORECASE)
        if prob_match:
            claims['probability'] = prob_match.group(1)

        return claims

    def _check_value_conflict(self, metric: str, values: Dict[str, str]) -> Optional[Conflict]:
        """Check if values represent a meaningful conflict"""
        # Extract numeric values for comparison
        numeric_values = {}
        for expert, value in values.items():
            numbers = re.findall(r'\d+\.?\d*', value)
            if numbers:
                numeric_values[expert] = float(numbers[0])

        if len(numeric_values) < 2:
            return None

        # Check if values differ by more than 20%
        vals = list(numeric_values.values())
        max_val = max(vals)
        min_val = min(vals)

        if max_val > 0 and (max_val - min_val) / max_val > 0.2:
            # Determine severity
            diff_pct = (max_val - min_val) / max_val
            if diff_pct > 0.5:
                severity = 'critical'
            elif diff_pct > 0.3:
                severity = 'moderate'
            else:
                severity = 'minor'

            # Determine category
            category = self._categorize_metric(metric)

            return Conflict(
                metric=metric,
                values=values,
                rationale=f"Values differ by {diff_pct:.0%}",
                severity=severity,
                category=category
            )

        return None

    def _categorize_metric(self, metric: str) -> str:
        """Categorize a metric into conflict categories"""
        metric_lower = metric.lower()
        for category, keywords in self.CONFLICT_CATEGORIES.items():
            if any(kw in metric_lower for kw in keywords):
                return category
        return 'other'

    def _detect_disagreement_language(self, discussion_data: List[Dict]) -> List[Conflict]:
        """Detect explicit disagreement between experts"""
        conflicts = []

        disagreement_patterns = [
            r'(disagree|differ|contrary|however|but|although).*?(\w+)\s+(said|mentioned|suggested|proposed)',
            r'(unlike|in contrast to|opposed to).*?(\w+)',
            r'I\s+(would|must)\s+(disagree|differ|challenge)',
        ]

        for item in discussion_data:
            content = item.get('content', '')
            expert = item.get('expert', 'Unknown')

            for pattern in disagreement_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    # Extract the context around disagreement
                    start = max(0, match.start() - 50)
                    end = min(len(content), match.end() + 100)
                    context = content[start:end]

                    conflicts.append(Conflict(
                        metric="Position disagreement",
                        values={expert: context},
                        rationale="Explicit disagreement expressed",
                        severity='moderate',
                        category='other'
                    ))
                    break

        return conflicts

    def _ai_conflict_detection(self, discussion_data: List[Dict]) -> List[Conflict]:
        """AI-powered conflict detection"""
        formatted_responses = self._format_responses(discussion_data)

        system_prompt = """You are analyzing expert panel responses for contradictions and divergent views.

Identify metrics where experts give different values or opposing positions. For each conflict:
1. What is the metric/topic
2. What value/position did each expert state
3. Brief reason for difference (if mentioned)
4. Severity: critical (blocks decision), moderate (needs resolution), minor (acceptable range)
5. Category: efficacy, safety, design, regulatory, timeline, or commercial

Return ONLY valid JSON array:
[{
    "metric": "ORR target",
    "values": {"Dr. Smith": "55%", "Dr. Jones": "60-65%"},
    "rationale": "different study benchmarks",
    "severity": "moderate",
    "category": "efficacy"
}]

Return empty array [] if no conflicts found."""

        user_prompt = f"""Analyze these expert panel responses for contradictions:

{formatted_responses}

Identify conflicts and return JSON array."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )

            result = response.choices[0].message.content.strip()

            # Clean JSON from markdown
            if '```json' in result:
                result = result.split('```json')[1].split('```')[0].strip()
            elif '```' in result:
                result = result.split('```')[1].split('```')[0].strip()

            conflicts_data = json.loads(result)

            return [Conflict(**c) for c in conflicts_data if isinstance(c, dict)]

        except Exception as e:
            logger.error(f"AI conflict detection failed: {e}")
            return self._pattern_conflict_detection(discussion_data)

    def _format_responses(self, discussion_data: List[Dict]) -> str:
        """Format responses for analysis"""
        formatted = []
        for item in discussion_data:
            expert = item.get('expert', 'Unknown')
            role = item.get('role', '')
            content = item.get('content', '')[:800]
            formatted.append(f"**{expert}** ({role}):\n{content}\n")
        return "\n".join(formatted)

    def categorize_conflicts(self, conflicts: List[Conflict]) -> Dict[str, List[Conflict]]:
        """Group conflicts by severity"""
        categorized = {
            'critical': [],
            'moderate': [],
            'minor': []
        }
        for conflict in conflicts:
            if conflict.severity in categorized:
                categorized[conflict.severity].append(conflict)
        return categorized

    def generate_clarification_prompts(
        self,
        conflicts: List[Conflict],
        top_n: int = 3,
        adversarial: bool = False
    ) -> List[str]:
        """
        Generate targeted questions for clarification round.

        Args:
            conflicts: List of detected conflicts
            top_n: Number of conflicts to address
            adversarial: If True, generate tough reviewer-style questions
        """
        if not conflicts:
            return []

        # Prioritize critical and moderate conflicts
        critical = [c for c in conflicts if c.severity == 'critical']
        moderate = [c for c in conflicts if c.severity == 'moderate']
        prioritized = (critical + moderate)[:top_n]

        if adversarial:
            return self._generate_adversarial_prompts(prioritized)
        else:
            return self._generate_standard_prompts(prioritized)

    def _generate_standard_prompts(self, conflicts: List[Conflict]) -> List[str]:
        """Generate standard clarification prompts"""
        prompts = []
        for conflict in conflicts:
            values_str = "\n".join([f"  • {k}: {v}" for k, v in conflict.values.items()])
            prompt = f"""**Clarification needed on {conflict.metric}:**

The panel provided different estimates:
{values_str}

Rationale: {conflict.rationale}

Each expert who addressed this: briefly explain your rationale and key assumptions. Where is the greatest uncertainty?"""
            prompts.append(prompt)
        return prompts

    def _generate_adversarial_prompts(self, conflicts: List[Conflict]) -> List[str]:
        """Generate adversarial reviewer-style tough questions"""
        if not self.client:
            return self._generate_standard_prompts(conflicts)

        conflict_summary = []
        for c in conflicts:
            values_str = ", ".join([f"{k}: {v}" for k, v in c.values.items()])
            conflict_summary.append(f"• {c.metric}: {values_str} (Reason: {c.rationale})")

        summary_text = "\n".join(conflict_summary)

        prompt = f"""You are a skeptical session chair (think: tough FDA reviewer or critical investor).

The panel showed these divergences:
{summary_text}

Generate 2-3 direct, challenging questions that:
- Expose hidden assumptions ("You're assuming X - what if that's wrong?")
- Demand specific data sources ("Which study? What was the N?")
- Question feasibility ("How will you achieve this timeline?")
- Force commitment ("At what threshold do you change your recommendation?")

Style: Skeptical but professional. Like an FDA reviewer who's seen too many failed programs.
Format: Return ONLY the questions, one per line, <20 words each."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a tough but fair session chair."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.5
            )

            result = response.choices[0].message.content.strip()
            questions = [q.strip() for q in result.split('\n') if q.strip() and len(q.strip()) > 10]

            if questions:
                return [f"**🔍 Session Chair Challenge:**\n\n" + "\n".join(questions[:3])]

        except Exception as e:
            logger.error(f"Adversarial prompt generation failed: {e}")

        return self._generate_standard_prompts(conflicts)


class DecisionSynthesizer:
    """Creates decision memos that preserve expert divergence"""

    def __init__(self, llm_client=None, model: str = None):
        from config import settings
        self.client = llm_client
        self.model = model or settings.REASONING_MODEL
        self.detector = ConflictDetector(llm_client, self.model)

    def synthesize(
        self,
        discussion_data: List[Dict],
        topic: str,
        question: str,
        conflicts: Optional[List[Conflict]] = None
    ) -> str:
        """
        Create decision memo from panel discussion.

        Args:
            discussion_data: List of expert responses
            topic: Discussion topic/product
            question: Clinical question
            conflicts: Pre-detected conflicts (optional)

        Returns:
            Formatted decision memo as markdown
        """
        if conflicts is None:
            conflicts = self.detector.detect_conflicts(discussion_data, use_ai=self.client is not None)

        if self.client:
            return self._ai_synthesis(discussion_data, topic, question, conflicts)

        return self._fallback_synthesis(discussion_data, topic, conflicts)

    def _ai_synthesis(
        self,
        discussion_data: List[Dict],
        topic: str,
        question: str,
        conflicts: List[Conflict]
    ) -> str:
        """AI-powered decision memo synthesis"""
        discussion_summary = self._summarize_discussion(discussion_data)
        conflict_summary = self._format_conflicts(conflicts)

        prompt = f"""Create a decision-ready memo from this expert panel discussion.

Topic: {topic}
Question: {question}

Discussion Summary:
{discussion_summary}

STRUCTURE YOUR MEMO:

## 1. Areas of Agreement
[List topics where experts aligned, with the consensus position]

## 2. Key Divergences
{conflict_summary}

For each divergence:
- Range of expert positions (with names)
- Rationale for different views
- What data/evidence would resolve the uncertainty

## 3. Critical Gaps
[Topics not adequately addressed that need attention]

## 4. Recommendations
[Specific next steps: data to gather, analyses to run, decisions to make]

## 5. Risk Assessment
[What could go wrong if divergences aren't resolved]

Keep it concise (400-600 words). Use bullet points. Be specific about WHO said WHAT."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You synthesize expert panel discussions into actionable decision memos."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.5
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"AI synthesis failed: {e}")
            return self._fallback_synthesis(discussion_data, topic, conflicts)

    def _summarize_discussion(self, discussion_data: List[Dict]) -> str:
        """Create brief summary of key points"""
        summary_parts = []
        for item in discussion_data[:10]:
            expert = item.get('expert', 'Unknown')
            content = item.get('content', '')[:200]
            summary_parts.append(f"• {expert}: {content}...")
        return "\n".join(summary_parts)

    def _format_conflicts(self, conflicts: List[Conflict]) -> str:
        """Format conflicts for synthesis prompt"""
        if not conflicts:
            return "[No major divergences detected]"

        lines = []
        for c in conflicts[:5]:
            values_str = ", ".join([f"{k}: {v}" for k, v in c.values.items()])
            lines.append(f"- {c.metric}: {values_str} ({c.rationale})")
        return "\n".join(lines)

    def _fallback_synthesis(
        self,
        discussion_data: List[Dict],
        topic: str,
        conflicts: List[Conflict]
    ) -> str:
        """Simple fallback if AI synthesis fails"""
        parts = [
            f"# Decision Memo: {topic}",
            "",
            "## Discussion Summary",
            ""
        ]

        for item in discussion_data[:5]:
            expert = item.get('expert', 'Unknown')
            content = item.get('content', '')[:150]
            parts.append(f"**{expert}**: {content}...")
            parts.append("")

        if conflicts:
            parts.extend([
                "## Key Divergences",
                ""
            ])
            for c in conflicts[:5]:
                parts.append(f"**{c.metric}** ({c.severity}):")
                for expert, value in c.values.items():
                    parts.append(f"- {expert}: {value}")
                parts.append("")

        parts.extend([
            "## Recommendations",
            "- Review conflicting positions with internal team",
            "- Gather additional data to resolve uncertainties",
            "- Consider follow-up round to clarify key points"
        ])

        return "\n".join(parts)


def detect_panel_conflicts(
    discussion_data: List[Dict],
    llm_client=None
) -> ConflictAnalysis:
    """
    Convenience function to detect conflicts in a panel discussion.

    Args:
        discussion_data: List of dicts with 'expert' and 'content' keys
        llm_client: Optional OpenAI client for AI-powered analysis

    Returns:
        ConflictAnalysis object with results
    """
    detector = ConflictDetector(llm_client=llm_client)
    conflicts = detector.detect_conflicts(
        discussion_data,
        use_ai=llm_client is not None
    )

    categorized = detector.categorize_conflicts(conflicts)
    clarification_prompts = detector.generate_clarification_prompts(conflicts)

    # Generate decision memo
    synthesizer = DecisionSynthesizer(llm_client=llm_client)
    memo = synthesizer.synthesize(
        discussion_data,
        topic="Panel Discussion",
        question="",
        conflicts=conflicts
    )

    return ConflictAnalysis(
        conflicts=conflicts,
        agreement_areas=[],  # Would need more analysis
        clarification_needed=clarification_prompts,
        decision_memo=memo
    )
