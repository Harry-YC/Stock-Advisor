
"""
Verification Service (Epistemic Checking)

This module implements "On-Demand Claim Verification".
It parses a response into atomic claims and checks them against retrieved evidence
using an NLI (Natural Language Inference) approach.
"""

import json
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from config import settings

logger = logging.getLogger("literature_review.verification_service")

@dataclass
class ClaimCheck:
    claim_text: str
    status: str  # "SUPPORTED", "CONTRADICTED", "NEUTRAL"
    confidence: float
    evidence_snippet: str
    reasoning: str

class VerificationService:
    """
    Verifies scientific claims against a corpus of documents.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key
        self.model = model or settings.OPENAI_MODEL

    def verify_response(
        self,
        response_text: str,
        context_documents: List[str]
    ) -> List[ClaimCheck]:
        """
        Decompose response into claims and verify each against context.

        Args:
            response_text: The text to verify (e.g. Expert response)
            context_documents: The ground truth text chunks.

        Returns:
            List of ClaimCheck objects.
        """
        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=self.api_key, model=self.model)

        # 1. Prepare Evidence Context
        evidence_text = "\n\n".join(context_documents[:5]) # Top 5 chunks for verification

        # 2. Prompt for NLI (Claim Extraction + Verification)
        system_prompt = """You are a Fact-Checking Engine for scientific literature.
Your job is to:
1. Extract core atomic claims from the 'Candidate Text'.
2. Verify each claim against the 'Evidence'.
3. Label as SUPPORTED, CONTRADICTED, or NEUTRAL (if not found).

Output JSON only."""

        user_prompt = f"""Evidence:
{evidence_text[:15000]}

Candidate Text:
{response_text[:4000]}

Task: Verify claims.
Return JSON object with key "checks", a list of objects:
- claim_text (string)
- status (enum: SUPPORTED, CONTRADICTED, NEUTRAL)
- confidence (float 0-1)
- evidence_snippet (string direct quote from Evidence)
- reasoning (string explanation)
"""
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0 # Low temp for factual accuracy
            )

            content = response.choices[0].message.content
            data = json.loads(content)
            
            checks = []
            for item in data.get("checks", []):
                checks.append(ClaimCheck(
                    claim_text=item.get("claim_text", ""),
                    status=item.get("status", "NEUTRAL").upper(),
                    confidence=item.get("confidence", 0.0),
                    evidence_snippet=item.get("evidence_snippet", ""),
                    reasoning=item.get("reasoning", "")
                ))
            
            return checks

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return []
