
"""
Strategy Generation Service (formerly Hypothesis Gen)

This module implements the "Strategic Recommendation Engine".
It ingests public literature + (simulated) proprietary data to propose 
optimized clinical/preclinical strategies.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

from config import settings

logger = logging.getLogger("literature_review.strategy_service")

@dataclass
class StrategicOption:
    """A proposed clinical/preclinical strategy."""
    name: str # e.g. "Combination Therapy with PD-1"
    description: str # The core strategy
    rationale: str # Scientific backing
    risk_assessment: str # Potential failure modes
    benefit_potential: float # 0.0 - 1.0
    feasibility: float # 0.0 - 1.0
    key_evidence: List[str] # Citations supporting this path
    next_steps: List[str] # Concrete actions (e.g. "Run assay X")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StrategyGenerator:
    """
    Synthesizes strategies from literature and internal context.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key
        self.model = model or settings.REASONING_MODEL

    def generate_strategies(
        self,
        objective: str,
        context_documents: List[str],
        proprietary_context: Optional[str] = None
    ) -> List[StrategicOption]:
        """
        Generate strategic options.

        Args:
            objective: User's goal (e.g. "Best Indication for Target X").
            context_documents: Public literature context.
            proprietary_context: Internal data/constrains.

        Returns:
            List of StrategicOption objects.
        """
        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=self.api_key, model=self.model)

        combined_context = "\n\n".join(context_documents[:15])
        
        internal_data = ""
        if proprietary_context:
            internal_data = f"\n\nPROPRIETARY DATA (Confidential):\n{proprietary_context}"

        system_prompt = """You are a Senior Drug Development Strategist.
Your goal is to synthesize public literature and internal data to recommend the optimal clinical/preclinical strategy.

Rules:
1. Be pragmatic. Focus on "Probability of Success" (PoS).
2. Weigh risks explicitly (Toxicity, Competition, IP).
3. Connect mechanisms to clinical outcomes.
4. Output strict JSON."""

        user_prompt = f"""Objective: {objective}

Literature Context:
{combined_context[:25000]}

{internal_data}

Task: Propose 3 distinct strategic options (e.g., Aggressive, Conservative, Novel).
Return JSON object with key "strategies" containing list of:
- name (string)
- description (string)
- rationale (string)
- risk_assessment (string): Critical flaws/risks
- benefit_potential (float 0-1)
- feasibility (float 0-1)
- key_evidence (list of strings)
- next_steps (list of strings): 3 concrete next actions
"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7 
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            strategies = []
            for s_data in data.get("strategies", []):
                strategies.append(StrategicOption(
                    name=s_data.get("name", "Unnamed Strategy"),
                    description=s_data.get("description", ""),
                    rationale=s_data.get("rationale", ""),
                    risk_assessment=s_data.get("risk_assessment", ""),
                    benefit_potential=s_data.get("benefit_potential", 0.5),
                    feasibility=s_data.get("feasibility", 0.5),
                    key_evidence=s_data.get("key_evidence", []),
                    next_steps=s_data.get("next_steps", [])
                ))
            
            return strategies

        except Exception as e:
            logger.error(f"Failed to generate strategies: {e}")
            return []
