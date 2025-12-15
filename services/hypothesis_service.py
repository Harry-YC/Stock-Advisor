
"""
Hypothesis Generation Service (Restored)

This module implements the "Hypothesis Generation" engine for the "Discovery Mode"
(Academic/Scientific use case).

It uses "Combinatorial Creativity" to:
1. Ingest retrieved literature context.
2. Identify disparate concepts.
3. Propose novel connections (hypotheses).
4. Score them based on Novelty, Feasibility, and Evidence Support.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

from config import settings

logger = logging.getLogger("literature_review.hypothesis_service")

@dataclass
class Hypothesis:
    """A generated scientific hypothesis."""
    title: str
    description: str
    novelty_score: float  # 0.0 to 1.0 (1.0 = highly novel)
    feasibility_score: float  # 0.0 to 1.0 (1.0 = highly feasible)
    evidence_support: List[str]  # Citations/Facts that inspired this
    experimental_validation: str  # Proposed experiment to test it
    reasoning_chain: str  # How the AI derived this

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HypothesisGenerator:
    """
    Generates novel hypotheses from scientific literature context.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key
        self.model = model or settings.REASONING_MODEL

    def generate_hypotheses(
        self,
        research_question: str,
        context_documents: List[str],
        previous_hypotheses: List[str] = None
    ) -> List[Hypothesis]:
        """
        Generate hypotheses based on the provided context.
        """
        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=self.api_key, model=self.model)

        # 1. Prepare Context
        combined_context = "\n\n".join(context_documents[:10])  # Top 10 documents
        prev_context = ""
        if previous_hypotheses:
            prev_context = "Avoid these previously discussed ideas:\n" + "\n".join(f"- {h}" for h in previous_hypotheses)

        # 2. Construct Prompt (Combinatorial Creativity)
        system_prompt = """You are an advanced Scientific Hypothesis Generation Engine (similar to DeepMind Co-Scientist).
Your goal is to perform "Combinatorial Creativity": connect disparate facts in the literature to propose potentially NOVEL hypotheses.

Rules:
1. Don't just summarize. Propose A -> C relationships where A->B and B->C are known.
2. Be specific about mechanisms.
3. Score your own hypotheses honestly.
4. Output strictly valid JSON."""

        user_prompt = f"""Research Question: {research_question}

Literature Context:
{combined_context[:20000]}  # Truncate if huge

{prev_context}

Task: Generate 3 novel scientific hypotheses.
Return a JSON object with key "hypotheses" containing a list of objects with these fields:
- title (string)
- description (string): Detailed scientific explanation
- novelty_score (float 0-1): How surprising is this given the context?
- feasibility_score (float 0-1): Can we test this with current tech?
- evidence_support (list of strings): Which snippets from context support this?
- experimental_validation (string): A concrete experiment (e.g., "Knockdown X in cell line Y...")
- reasoning_chain (string): Step-by-step logic (A->B, B->C, therefore A->C).
"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.8 # Higher temp for creativity
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            hypotheses = []
            for h_data in data.get("hypotheses", []):
                hypotheses.append(Hypothesis(
                    title=h_data.get("title", "Untitled"),
                    description=h_data.get("description", ""),
                    novelty_score=h_data.get("novelty_score", 0.5),
                    feasibility_score=h_data.get("feasibility_score", 0.5),
                    evidence_support=h_data.get("evidence_support", []),
                    experimental_validation=h_data.get("experimental_validation", ""),
                    reasoning_chain=h_data.get("reasoning_chain", "")
                ))
            
            return hypotheses

        except Exception as e:
            logger.error(f"Failed to generate hypotheses: {e}")
            return []
