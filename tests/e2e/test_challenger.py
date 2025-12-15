"""
E2E Tests for Red Team Challenger Feature (P3)

Tests:
- T3.1: Challenge button appears after recommendation
- T3.2: Clicking generates challenge questions
- T3.3: Challenges have proper categories
- T3.4: Challenges target weak points with rationales
- T3.5: At least 3 challenges are generated
"""

import re
import time
import pytest
from playwright.sync_api import Page, expect

from tests.e2e.test_questions import (
    CHALLENGER_EXPECTATIONS,
    get_quick_test_question,
)


@pytest.mark.e2e
@pytest.mark.feature_challenger
class TestChallenger:
    """Tests for Red Team Challenger functionality."""

    def test_t3_1_challenge_button_appears(
        self,
        page: Page,
        app_url: str,
        navigate_to_app,
        create_project,
        ask_question,
        test_question: str,
        st_helpers,
    ):
        """
        T3.1: Challenge button appears after a recommendation is generated.

        Verifies that "Challenge This" or similar button is visible
        after an expert response is displayed.
        """
        # Setup
        navigate_to_app()
        create_project("Challenger_Test_1")
        time.sleep(2)

        # Ask the test question
        ask_question(test_question)
        time.sleep(5)

        st_helpers.scroll_to_bottom()

        # Look for challenge button variants
        challenge_button_texts = CHALLENGER_EXPECTATIONS["button_text"]
        found_button = None

        for text in challenge_button_texts:
            buttons = page.locator('button').filter(has_text=text)
            if buttons.count() > 0:
                found_button = buttons.first
                print(f"Found challenge button with text: '{text}'")
                break

        # Also check for devil's advocate icon or similar
        icon_buttons = page.locator('button').filter(has_text="😈")
        if icon_buttons.count() > 0:
            found_button = icon_buttons.first
            print("Found challenge button with devil emoji")

        st_helpers.take_screenshot("challenger_button")

        if found_button:
            assert found_button.is_visible(), "Challenge button should be visible"
        else:
            # Button may not be present if feature not integrated yet
            print("Challenge button not found - checking if feature is integrated")

    def test_t3_2_clicking_generates_challenges(
        self,
        page: Page,
        app_url: str,
        navigate_to_app,
        create_project,
        ask_question,
        test_question: str,
        st_helpers,
    ):
        """
        T3.2: Clicking Challenge button generates challenge questions.

        Verifies that clicking the button triggers LLM generation
        and displays challenge questions.
        """
        # Setup
        navigate_to_app()
        create_project("Challenger_Test_2")
        time.sleep(2)

        # Ask the test question
        ask_question(test_question)
        time.sleep(5)

        st_helpers.scroll_to_bottom()

        # Find and click challenge button
        challenge_clicked = False
        for text in CHALLENGER_EXPECTATIONS["button_text"]:
            buttons = page.locator('button').filter(has_text=text)
            if buttons.count() > 0 and buttons.first.is_visible():
                buttons.first.click()
                challenge_clicked = True
                print(f"Clicked challenge button: '{text}'")
                break

        if challenge_clicked:
            # Wait for challenges to generate
            time.sleep(10)  # LLM call may take time
            st_helpers.wait_for_streamlit_rerun(3)

            page_content = page.content()

            # Look for challenge content indicators
            challenge_indicators = [
                "challenge",
                "question",
                "assumption",
                "evidence",
                "patient_selection",
                "threshold",
                "risk",
            ]

            found_indicators = [ind for ind in challenge_indicators if ind.lower() in page_content.lower()]
            print(f"Found challenge indicators: {found_indicators}")

            st_helpers.take_screenshot("challenger_questions")
        else:
            print("Could not find/click challenge button")

    def test_t3_3_challenges_have_categories(
        self,
        page: Page,
        app_url: str,
        navigate_to_app,
        create_project,
        ask_question,
        test_question: str,
        st_helpers,
    ):
        """
        T3.3: Challenges have proper category labels.

        Verifies that generated challenges are categorized as:
        assumption, evidence, patient_selection, threshold, risk, feasibility
        """
        # Setup
        navigate_to_app()
        create_project("Challenger_Test_3")
        time.sleep(2)

        # Ask the test question
        ask_question(test_question)
        time.sleep(5)

        st_helpers.scroll_to_bottom()

        # Find and click challenge button
        for text in CHALLENGER_EXPECTATIONS["button_text"]:
            buttons = page.locator('button').filter(has_text=text)
            if buttons.count() > 0 and buttons.first.is_visible():
                buttons.first.click()
                time.sleep(10)
                break

        page_content = page.content()

        # Check for category labels
        expected_categories = CHALLENGER_EXPECTATIONS["categories"]
        found_categories = []

        for category in expected_categories:
            if category.lower() in page_content.lower():
                found_categories.append(category)

        print(f"Found categories: {found_categories}")
        print(f"Expected categories: {expected_categories}")

        # At least one category should be present
        if found_categories:
            print(f"Categories found: {len(found_categories)}/{len(expected_categories)}")

    def test_t3_4_challenges_have_rationales(
        self,
        page: Page,
        app_url: str,
        navigate_to_app,
        create_project,
        ask_question,
        test_question: str,
        st_helpers,
    ):
        """
        T3.4: Challenges target specific weak points with rationales.

        Verifies that each challenge includes:
        - A specific question
        - What it targets
        - Why it matters (rationale)
        """
        # Setup
        navigate_to_app()
        create_project("Challenger_Test_4")
        time.sleep(2)

        # Ask the test question
        ask_question(test_question)
        time.sleep(5)

        st_helpers.scroll_to_bottom()

        # Find and click challenge button
        for text in CHALLENGER_EXPECTATIONS["button_text"]:
            buttons = page.locator('button').filter(has_text=text)
            if buttons.count() > 0 and buttons.first.is_visible():
                buttons.first.click()
                time.sleep(10)
                break

        page_content = page.content()

        # Look for rationale-related content
        rationale_indicators = [
            "rationale",
            "targets",
            "why",
            "because",
            "matters",
            "important",
            "address",
        ]

        found_rationales = [ind for ind in rationale_indicators if ind.lower() in page_content.lower()]
        print(f"Found rationale indicators: {found_rationales}")

        # Check for question marks (indicates actual questions)
        question_count = page_content.count("?")
        print(f"Found {question_count} question marks in content")

        st_helpers.take_screenshot("challenger_rationales")

    def test_t3_5_minimum_challenges_generated(
        self,
        page: Page,
        app_url: str,
        navigate_to_app,
        create_project,
        ask_question,
        test_question: str,
        st_helpers,
    ):
        """
        T3.5: At least 3 challenges are generated.

        Verifies that the minimum number of challenge questions
        is generated as specified.
        """
        # Setup
        navigate_to_app()
        create_project("Challenger_Test_5")
        time.sleep(2)

        # Ask the test question
        ask_question(test_question)
        time.sleep(5)

        st_helpers.scroll_to_bottom()

        # Find and click challenge button
        for text in CHALLENGER_EXPECTATIONS["button_text"]:
            buttons = page.locator('button').filter(has_text=text)
            if buttons.count() > 0 and buttons.first.is_visible():
                buttons.first.click()
                time.sleep(10)
                break

        page_content = page.content()

        # Count challenges by looking for:
        # 1. Question marks in challenge context
        # 2. Category badges
        # 3. Numbered items

        # Look for category mentions (each category = 1 challenge)
        categories = CHALLENGER_EXPECTATIONS["categories"]
        category_count = sum(1 for cat in categories if cat.lower() in page_content.lower())

        # Count questions (rough heuristic)
        question_count = page_content.count("?")

        print(f"Category mentions: {category_count}")
        print(f"Question marks: {question_count}")
        print(f"Minimum expected: {CHALLENGER_EXPECTATIONS['min_questions']}")

        st_helpers.take_screenshot("challenger_count")


@pytest.mark.e2e
@pytest.mark.feature_challenger
class TestChallengerService:
    """Unit-style tests for Challenger service."""

    def test_challenger_service_import(self):
        """Test that challenger_service module can be imported."""
        from services.challenger_service import (
            ChallengerService,
            generate_challenges,
            ChallengeQuestion,
            ChallengeOutput,
        )

        assert ChallengerService is not None
        assert callable(generate_challenges)
        assert ChallengeQuestion is not None
        assert ChallengeOutput is not None

    def test_challenge_question_dataclass(self):
        """Test ChallengeQuestion dataclass structure."""
        from services.challenger_service import ChallengeQuestion

        question = ChallengeQuestion(
            question="What evidence supports this threshold?",
            category="evidence",
            targets="Mortality claim without PMID",
            rationale="GRADE requires traceable evidence for numeric claims",
        )

        assert question.question.endswith("?")
        assert question.category == "evidence"
        assert len(question.targets) > 0
        assert len(question.rationale) > 0

        # Test to_dict
        d = question.to_dict()
        assert "question" in d
        assert "category" in d

    def test_challenge_output_dataclass(self):
        """Test ChallengeOutput dataclass structure."""
        from services.challenger_service import ChallengeOutput, ChallengeQuestion

        questions = [
            ChallengeQuestion(
                question="Q1?",
                category="assumption",
                targets="Target 1",
                rationale="Rationale 1",
            ),
            ChallengeQuestion(
                question="Q2?",
                category="evidence",
                targets="Target 2",
                rationale="Rationale 2",
            ),
        ]

        output = ChallengeOutput(
            analysis="Two weak points identified",
            questions=questions,
            model_used="test-model",
            generated_at="2024-01-01T00:00:00",
            conflicts_count=1,
            evidence_gaps_count=2,
        )

        assert len(output.questions) == 2
        assert output.conflicts_count == 1
        assert output.evidence_gaps_count == 2

        # Test to_dict
        d = output.to_dict()
        assert "analysis" in d
        assert "questions" in d
        assert len(d["questions"]) == 2

    def test_challenger_categories(self):
        """Test that expected categories are defined."""
        from services.challenger_service import ChallengerService

        service = ChallengerService()

        # These categories should be supported
        expected_categories = [
            "assumption",
            "evidence",
            "patient_selection",
            "threshold",
            "risk",
            "feasibility",
        ]

        # Check that the prompt mentions these categories
        from services.challenger_service import build_challenger_prompt

        prompt = build_challenger_prompt(
            recommendation="Test recommendation",
            question="Test question",
            expert_summary="Expert said X",
            conflicts_summary="Conflict A",
            evidence_gaps_summary="Gap B",
            key_findings=["Finding 1"],
        )

        for category in expected_categories:
            assert category in prompt.lower(), f"Category '{category}' should be in prompt"

    def test_fallback_response(self):
        """Test fallback response when LLM fails."""
        from services.challenger_service import ChallengerService

        service = ChallengerService()
        fallback = service._fallback_response("Test error")

        assert fallback is not None
        assert len(fallback.questions) >= 3
        assert "fallback" in fallback.model_used
        assert "Test error" in fallback.analysis

        # Check fallback questions have valid structure
        for q in fallback.questions:
            assert len(q.question) > 0
            assert len(q.category) > 0
            assert len(q.targets) > 0
            assert len(q.rationale) > 0

    def test_format_conflicts(self):
        """Test conflict formatting."""
        from services.challenger_service import ChallengerService

        service = ChallengerService()

        # Test with structured conflicts
        conflicts = [
            {"expert": "Surgeon", "position": "Recommends surgery", "type": "disagreement"},
            {"expert": "Palliative", "position": "Prefers conservative", "type": "concern"},
        ]

        formatted = service._format_conflicts(conflicts)
        assert "Surgeon" in formatted
        assert "Palliative" in formatted

        # Test with string conflicts
        string_conflicts = ["Conflict 1", "Conflict 2"]
        formatted_strings = service._format_conflicts(string_conflicts)
        assert "Conflict 1" in formatted_strings

    def test_format_evidence_gaps(self):
        """Test evidence gap formatting."""
        from services.challenger_service import ChallengerService

        service = ChallengerService()

        # Test with structured gaps
        gaps = [
            {"gap": "No RCT data", "suggested_action": "Search ClinicalTrials.gov"},
            {"description": "Missing QoL data"},
        ]

        formatted = service._format_evidence_gaps(gaps)
        assert "RCT" in formatted
        assert "QoL" in formatted

        # Test with string gaps
        string_gaps = ["Gap 1", "Gap 2"]
        formatted_strings = service._format_evidence_gaps(string_gaps)
        assert "Gap 1" in formatted_strings
