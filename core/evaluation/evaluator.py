"""
RAG Evaluator Module

Implements LLM-as-a-Judge to evaluate the quality of RAG responses.
Metrics:
- Faithfulness: Is the answer grounded in the retrieved context?
- Relevance: Does the answer directly address the user's question?
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from config import settings
from core.llm_utils import get_llm_client

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    faithfulness_score: float  # 0.0 to 1.0
    relevance_score: float     # 0.0 to 1.0
    reasoning: str

class RAGEvaluator:
    """Evaluates RAG pipeline outputs using an LLM judge."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GOOGLE_API_KEY or settings.OPENAI_API_KEY
        self.model = settings.EXPERT_MODEL
        if not self.api_key:
            logger.warning("No API key found. RAGEvaluator will not function.")
        self.client = get_llm_client(api_key=self.api_key, model=self.model) if self.api_key else None

    def evaluate_response(
        self, 
        question: str, 
        response: str, 
        context: str
    ) -> EvaluationResult:
        """
        Evaluate a RAG response for faithfulness and relevance.
        
        Args:
            question: The user's query.
            response: The generated answer.
            context: The text chunks used as context.
            
        Returns:
            EvaluationResult containing scores and reasoning.
        """
        if not self.client:
            return EvaluationResult(0.0, 0.0, "Evaluator not initialized")

        prompt = f"""
You are an expert AI Judge. Evaluate the following RAG response based on the provided Context and Question.

QUESTION: {question}

CONTEXT:
{context}

RESPONSE:
{response}

---
CRITERIA:

1. FAITHFULNESS (0-1):
- 1.0: The response contains ONLY information derived from the Context.
- 0.0: The response contains hallucinations or external knowledge not found in the Context.

2. RELEVANCE (0-1):
- 1.0: The response directly and completely answers the Question.
- 0.0: The response is irrelevant or fails to address the core of the Question.

OUTPUT FORMAT (JSON):
{{
    "faithfulness_score": <float between 0.0 and 1.0>,
    "relevance_score": <float between 0.0 and 1.0>,
    "reasoning": "<concise explanation of scores>"
}}
"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strict evaluator of AI systems."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            import json
            result_json = json.loads(completion.choices[0].message.content)
            
            return EvaluationResult(
                faithfulness_score=result_json.get("faithfulness_score", 0.0),
                relevance_score=result_json.get("relevance_score", 0.0),
                reasoning=result_json.get("reasoning", "No reasoning provided")
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return EvaluationResult(0.0, 0.0, f"Error: {str(e)}")
