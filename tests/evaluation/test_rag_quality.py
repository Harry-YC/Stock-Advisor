"""
Automated RAG Quality Evaluation

This test suite uses the RAGEvaluator to benchmark the system's performance.
It runs a set of "Golden Queries" and asserts that:
1. Faithfulness > 0.8
2. Relevance > 0.8
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.evaluation.evaluator import RAGEvaluator

# Mock data for testing without a live vector DB setup
GOLDEN_DATASET = [
    {
        "question": "What is the primary mechanism of action of Pembrolizumab?",
        "context": """
        Pembrolizumab is a programmed death receptor-1 (PD-1)-blocking antibody. 
        It binds to the PD-1 receptor on T cells, preventing its interaction with PD-L1 and PD-L2 ligands. 
        This blockade releases the PD-1 pathway-mediated inhibition of the immune response, including the anti-tumor immune response.
        """,
        "generated_answer": "Pembrolizumab works by blocking the PD-1 receptor on T cells. This prevents PD-1 from binding to PD-L1 and PD-L2, thereby reactivating the immune system to attack tumor cells."
    },
    {
        "question": "Can Aspirin cure cancer?",
        "context": """
        Aspirin is a nonsteroidal anti-inflammatory drug (NSAID). 
        While some studies suggest daily low-dose aspirin may reduce the risk of developing certain cancers (like colorectal cancer), it is not a cure for established cancer. 
        It is primarily used for pain relief and cardiovascular protection.
        """,
        "generated_answer": "No, Aspirin is not a cure for cancer. However, it may reduce the risk of developing colorectal cancer."
    }
]

@pytest.fixture
def evaluator():
    """Initialize the RAGEvaluator."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OpenAI API key not found. Skipping evaluation tests.")
    return RAGEvaluator(api_key=api_key)

class TestRAGQuality:
    """Test suite for RAG quality metrics."""

    def test_faithfulness_and_relevance(self, evaluator):
        """Test that golden examples achieve high scores."""
        
        for case in GOLDEN_DATASET:
            result = evaluator.evaluate_response(
                question=case["question"],
                response=case["generated_answer"],
                context=case["context"]
            )
            
            print(f"\nQuestion: {case['question']}")
            print(f"Scores -> Faithfulness: {result.faithfulness_score}, Relevance: {result.relevance_score}")
            print(f"Reasoning: {result.reasoning}")
            
            # Assertions for "DeepMind Quality" bar
            assert result.faithfulness_score >= 0.8, f"Low faithfulness for: {case['question']}"
            assert result.relevance_score >= 0.8, f"Low relevance for: {case['question']}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
