
"""
Tests for Clinical Planning Capabilities
- Clinical Planner (IND/CDP)
- Strategy Service (Strategy Generation)
- STORM Workflow (Iterative Loop)
"""
import pytest
from unittest.mock import MagicMock, patch

from services.planning_service import ClinicalPlanner, PlanType
from services.strategy_service import StrategyGenerator
from services.expert_service import ExpertDiscussionService

@pytest.fixture
def mock_openai():
    with patch("openai.OpenAI") as mock:
        yield mock

def test_clinical_planner_ind(mock_openai):
    # Setup Mock
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Mock response
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = """
    {
        "title": "IND Module 2.4 - Asset X",
        "doc_type": "IND Module 2.4 (Nonclinical Overview)",
        "sections": {
            "Pharmacology": "Potent antagonist...",
            "Toxicology": "No NOAEL observed..."
        },
        "key_risks": ["Solubility"],
        "missing_data": ["hERG assay"]
    }
    """
    mock_client.chat.completions.create.return_value = mock_completion
    
    planner = ClinicalPlanner(api_key="fake")
    doc = planner.generate_plan("Asset X", "Context", PlanType.IND_MODULE_2_4)
    
    assert doc.title == "IND Module 2.4 - Asset X"
    assert "Pharmacology" in doc.sections
    assert doc.key_risks[0] == "Solubility"

def test_strategy_generation(mock_openai):
    # Setup Mock
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = """
    {
        "strategies": [
            {
                "name": "Strategy A",
                "description": "Aggressive Phase 2",
                "rationale": "High unmet need",
                "risk_assessment": "Safety signal",
                "benefit_potential": 0.9,
                "feasibility": 0.6,
                "key_evidence": ["Paper 1"],
                "next_steps": ["Run Tox"]
            }
        ]
    }
    """
    mock_client.chat.completions.create.return_value = mock_completion
    
    generator = StrategyGenerator(api_key="fake")
    strategies = generator.generate_strategies("Goal", ["Context"])
    
    assert len(strategies) == 1
    assert strategies[0].name == "Strategy A"
    assert strategies[0].benefit_potential == 0.9

@patch("preclinical.expert_utils.call_expert")
def test_storm_loop(mock_call_expert, mock_openai):
    # Setup Mocks
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # OpenAI mock for Critique (Reviewer) step - this is the 3rd step
    mock_critique_completion = MagicMock()
    mock_critique_completion.choices[0].message.content = "Critique: Too vague."
    mock_client.chat.completions.create.return_value = mock_critique_completion
    
    # Call Expert mocks (Outline, Draft, Revision)
    # The service calls call_expert 3 times:
    # 1. Outline
    # 2. Draft
    # 3. Revision
    mock_call_expert.side_effect = [
        {"content": "Outline Text"},   # 1. Outline
        {"content": "Draft Text"},     # 2. Draft
        {"content": "Final Text"}      # 4. Revision (after critique)
    ]
    
    service = ExpertDiscussionService(api_key="fake")
    result = service.run_storm_workflow("Question", "Expert", [{"text": "cit"}])
    
    assert result["method"] == "STORM-v2"
    assert result["outline"] == "Outline Text"
    assert result["draft"] == "Draft Text"
    assert result["critique"] == "Critique: Too vague."
    assert result["final_response"] == "Final Text"
