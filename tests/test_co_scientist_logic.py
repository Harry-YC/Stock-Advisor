"""
Verification tests for Google Co-Scientist Logic upgrades.

Tests:
1. Collaborative Debate (ExpertDiscussionService.run_debate_round)
2. Agent Planning (ResearchAgent planning phase)
3. Active Recall (ResearchAgent knowledge retrieval)
"""

import pytest
from unittest.mock import MagicMock, patch, ANY
from services.expert_service import ExpertDiscussionService
from services.chat_service import ResearchAgent

@pytest.fixture
def mock_deps():
    return {
        "api_key": "fake-key",
        "model": "gpt-5-mini"
    }

class TestCoScientistLogic:

    @patch("preclinical.expert_utils.call_expert")
    @patch("openai.OpenAI")
    def test_run_debate_round_collaborative(self, mock_openai, mock_call_expert, mock_deps):
        """Verify the Proposal -> Challenge -> Mitigation chain."""
        service = ExpertDiscussionService(mock_deps["api_key"])
        
        # Setup mocks for 3 calls: Proposal, Challenge, Mitigation
        mock_call_expert.side_effect = [
            {"content": "PROPOSAL: Use high dose.", "finish_reason": "stop"},
            {"content": "CHALLENGE: Risk of toxicity.", "finish_reason": "stop"},
            {"content": "MITIGATION: Monitor liver enzymes.", "finish_reason": "stop"}
        ]
        
        # Mock synthesis
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices[0].message.content = "SYNTHESIS: Proceed with monitoring."
        
        result = service.run_debate_round(
            clinical_question="Q",
            pro_expert="Clinician",
            con_expert="Toxicologist",
            topic="Dosing",
            citations=[],
            scenario="chem_selection"
        )
        
        # Verify strict sequence
        assert result['proposal'] == "PROPOSAL: Use high dose."
        assert result['challenge'] == "CHALLENGE: Risk of toxicity."
        assert result['mitigation'] == "MITIGATION: Monitor liver enzymes."
        assert result['synthesis'] == "SYNTHESIS: Proceed with monitoring."
        
        # Verify 3 expert calls + 1 LLM synthesis
        assert mock_call_expert.call_count == 3
        mock_client.chat.completions.create.assert_called_once()

    @patch("core.knowledge_store.search_knowledge") 
    @patch("openai.OpenAI")
    def test_agent_active_recall_and_planning(self, mock_openai, mock_search_knowledge, mock_deps):
        """Verify ResearchAgent actively recalls facts and generates a plan."""
        # Setup Mock active recall
        mock_search_knowledge.return_value = [{"facts": ["Fact 1"], "source": "Prev Session"}]
        
        agent = ResearchAgent(mock_deps["api_key"])
        
        # Setup OpenAI mocks for:
        # 1. Planning call
        # 2. Execution tool call (e.g. search)
        # 3. Final synthesis
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock responses
        mock_plan_msg = MagicMock()
        mock_plan_msg.choices[0].message.content = "1. Plan Step A\n2. Plan Step B"
        
        mock_exec_stream = [
            [MagicMock(choices=[MagicMock(delta=MagicMock(content=None, tool_calls=[
                MagicMock(id="call_1", function=MagicMock(name="search_pubmed", arguments='{"query": "test"}'))
            ]))])],
            [MagicMock(choices=[MagicMock(delta=MagicMock(content="Found it.", tool_calls=None))])] # Final answer stream
        ]
        
        mock_client.chat.completions.create.side_effect = [
            mock_plan_msg,      # 1. Planning
            mock_exec_stream[0], # 2. Tool Call
            mock_exec_stream[1]  # 3. Final Synthesis
        ]
        
        # Mock Search Service execution
        agent.search_service = MagicMock()
        agent.search_service.execute_search.return_value = {"total_count": 1, "citations": []}
        
        # Run generator
        events = list(agent.run_agent_stream(
            question="Research Question",
            context="Base Context",
            project_id="pid",
            citation_dao=None, search_dao=None, query_cache_dao=None
        ))
        
        # Verify Active Recall happened
        mock_search_knowledge.assert_called_once_with("Research Question")
        
        # Verify Planning Event yielded
        plan_events = [e for e in events if e['type'] == 'plan']
        assert len(plan_events) == 1
        assert "1. Plan Step A" in plan_events[0]['content']
        
        # Verify System Prompt contains Plan
        # We can check the messages passed to the second call (Execution)
        call_args = mock_client.chat.completions.create.call_args_list
        assert len(call_args) == 3 # Plan, Execute, Synthesize
        
        # Check execution call (index 1) has the plan injected
        exec_messages = call_args[1].kwargs['messages']
        # The plan should be in one of the system messages
        plan_found = any("Follow this plan" in m.get('content', '') for m in exec_messages)
        assert plan_found, "Plan was not injected into context"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
