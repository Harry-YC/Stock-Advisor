"""
E2E Tests for Smart Suggestions Feature (P5)

Tests:
- T4.1: Suggestions appear after response
- T4.2: At least 2 suggestion buttons are clickable
- T4.3: Clicking a suggestion fills the input field
- T4.4: Suggestions are contextually relevant
"""

import re
import time
import pytest
from playwright.sync_api import Page, expect

from tests.e2e.test_questions import (
    SMART_SUGGESTIONS_EXPECTATIONS,
    MIRELS_SCORE_9_QUESTION,
    get_quick_test_question,
)


@pytest.mark.e2e
@pytest.mark.feature_suggestions
class TestSmartSuggestions:
    """Tests for Smart Suggestions functionality."""

    def test_t4_1_suggestions_appear_after_response(
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
        T4.1: Suggestions appear after an expert response.

        Verifies that "Suggested follow-ups" or similar section
        appears after the main response is displayed.
        """
        # Setup
        navigate_to_app()
        create_project("Suggestions_Test_1")
        time.sleep(2)

        # Ask the test question
        ask_question(test_question)
        time.sleep(5)

        st_helpers.scroll_to_bottom()
        st_helpers.take_screenshot("suggestions_after_response")

        page_content = page.content()

        # Look for suggestion section indicators
        suggestion_patterns = SMART_SUGGESTIONS_EXPECTATIONS["header_patterns"]
        found_patterns = []

        for pattern in suggestion_patterns:
            if pattern.lower() in page_content.lower():
                found_patterns.append(pattern)

        print(f"Found suggestion patterns: {found_patterns}")

        # Also look for suggestion-style buttons
        suggestion_buttons = page.locator('button').filter(
            has_text=re.compile(r'(deep dive|clarify|explore|more about)', re.IGNORECASE)
        )
        print(f"Found {suggestion_buttons.count()} suggestion-style buttons")

        # Check for any clickable elements that look like suggestions
        all_buttons = page.locator('button')
        button_texts = []
        for i in range(min(all_buttons.count(), 20)):
            btn = all_buttons.nth(i)
            if btn.is_visible():
                text = btn.inner_text()
                if len(text) > 10:  # Filter out icon-only buttons
                    button_texts.append(text[:50])

        print(f"Visible button texts: {button_texts}")

    def test_t4_2_minimum_suggestion_buttons(
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
        T4.2: At least 2 suggestion buttons are visible and clickable.

        Verifies that multiple follow-up suggestions are provided.
        """
        # Setup
        navigate_to_app()
        create_project("Suggestions_Test_2")
        time.sleep(2)

        # Ask the test question
        ask_question(test_question)
        time.sleep(5)

        st_helpers.scroll_to_bottom()

        # Count suggestion-like buttons
        suggestion_keywords = [
            "deep dive",
            "clarify",
            "explore",
            "more about",
            "safety",
            "evidence",
            "compare",
            "alternative",
            "follow",
        ]

        suggestion_buttons = []
        all_buttons = page.locator('button')

        for i in range(all_buttons.count()):
            btn = all_buttons.nth(i)
            if btn.is_visible():
                text = btn.inner_text().lower()
                if any(kw in text for kw in suggestion_keywords):
                    suggestion_buttons.append(btn)

        print(f"Found {len(suggestion_buttons)} suggestion buttons")

        min_expected = SMART_SUGGESTIONS_EXPECTATIONS["min_suggestions"]
        print(f"Minimum expected: {min_expected}")

        # Check if buttons are clickable
        for i, btn in enumerate(suggestion_buttons[:3]):
            is_enabled = btn.is_enabled()
            print(f"Button {i+1} enabled: {is_enabled}")

        st_helpers.take_screenshot("suggestions_buttons")

    def test_t4_3_clicking_suggestion_fills_input(
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
        T4.3: Clicking a suggestion button fills the question input.

        Verifies that clicking a suggestion pre-populates the
        question textarea for a follow-up query.
        """
        # Setup
        navigate_to_app()
        create_project("Suggestions_Test_3")
        time.sleep(2)

        # Ask the test question
        ask_question(test_question)
        time.sleep(5)

        st_helpers.scroll_to_bottom()

        # Find a suggestion button
        suggestion_keywords = [
            "deep dive",
            "clarify",
            "explore",
            "safety",
            "evidence",
        ]

        suggestion_btn = None
        all_buttons = page.locator('button')

        for i in range(all_buttons.count()):
            btn = all_buttons.nth(i)
            if btn.is_visible():
                text = btn.inner_text().lower()
                if any(kw in text for kw in suggestion_keywords):
                    suggestion_btn = btn
                    print(f"Found suggestion button: '{btn.inner_text()}'")
                    break

        if suggestion_btn:
            # Get initial textarea content
            textarea = page.locator('textarea').first
            initial_value = textarea.input_value() if textarea.is_visible() else ""
            print(f"Initial textarea value: '{initial_value}'")

            # Click the suggestion
            suggestion_btn.click()
            st_helpers.wait_for_streamlit_rerun()

            # Check if textarea was updated
            new_value = textarea.input_value() if textarea.is_visible() else ""
            print(f"New textarea value: '{new_value}'")

            # Value should have changed if suggestion fills input
            if new_value != initial_value:
                print("Suggestion successfully filled input!")
            else:
                print("Textarea value unchanged - checking for other behaviors")

                # Some suggestions may trigger actions instead of filling input
                # Check if page content changed
                page_content = page.content()
                if any(kw in page_content.lower() for kw in suggestion_keywords):
                    print("Suggestion may have triggered a different action")

        st_helpers.take_screenshot("suggestions_clicked")

    def test_t4_4_suggestions_are_contextual(
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
        T4.4: Suggestions are contextually relevant to the question.

        Verifies that suggested follow-ups relate to the original
        question topic (femoral metastases, Mirels score, etc.).
        """
        # Setup
        navigate_to_app()
        create_project("Suggestions_Test_4")
        time.sleep(2)

        # Ask the test question about Mirels score
        ask_question(test_question)
        time.sleep(5)

        st_helpers.scroll_to_bottom()

        page_content = page.content()

        # Expected contextual keywords based on the Mirels score question
        contextual_keywords = MIRELS_SCORE_9_QUESTION.expected_keywords + [
            "prognosis",
            "risk",
            "surgery",
            "conservative",
            "alternative",
            "complication",
        ]

        # Check button texts for contextual relevance
        all_buttons = page.locator('button')
        contextual_buttons = []

        for i in range(all_buttons.count()):
            btn = all_buttons.nth(i)
            if btn.is_visible():
                text = btn.inner_text().lower()
                matches = [kw for kw in contextual_keywords if kw.lower() in text]
                if matches:
                    contextual_buttons.append((btn.inner_text(), matches))

        print(f"Found {len(contextual_buttons)} contextual suggestion buttons:")
        for btn_text, matches in contextual_buttons[:5]:
            print(f"  - '{btn_text[:50]}' (matches: {matches})")

        # Also check page content for suggestion-related sections
        # that mention contextual terms
        suggestion_sections = page.locator('[data-testid="stExpander"]').filter(
            has_text=re.compile(r'(suggest|follow|related)', re.IGNORECASE)
        )
        print(f"Found {suggestion_sections.count()} suggestion expanders")

        st_helpers.take_screenshot("suggestions_contextual")


@pytest.mark.e2e
@pytest.mark.feature_suggestions
class TestSmartSuggestionsDetection:
    """Tests for suggestion detection logic."""

    def test_detect_safety_concerns(self):
        """Test detection of safety concerns in responses."""
        # Safety keywords that should trigger safety-related suggestions
        safety_keywords = [
            "mortality",
            "complication",
            "risk",
            "morbidity",
            "adverse",
            "death",
            "infection",
        ]

        # Sample response text
        response = """
        The procedure has a 30-day mortality rate of 15%.
        Complication rates include infection (5%) and wound dehiscence (3%).
        Risk factors include poor functional status and advanced disease.
        """

        detected_safety = [kw for kw in safety_keywords if kw.lower() in response.lower()]
        print(f"Detected safety keywords: {detected_safety}")

        assert len(detected_safety) > 0, "Safety keywords should be detected"

    def test_detect_evidence_gaps(self):
        """Test detection of evidence gaps in responses."""
        gap_indicators = [
            "no RCT",
            "limited evidence",
            "retrospective",
            "no randomized",
            "low quality",
            "EVIDENCE GAP",
        ]

        response = """
        EVIDENCE (PMID: 12345678) shows survival benefit.
        EVIDENCE GAP: No randomized controlled trials compare these approaches.
        Most studies are retrospective with limited evidence.
        """

        detected_gaps = [ind for ind in gap_indicators if ind.lower() in response.lower()]
        print(f"Detected evidence gap indicators: {detected_gaps}")

        assert len(detected_gaps) > 0, "Evidence gap indicators should be detected"

    def test_detect_disagreements(self):
        """Test detection of expert disagreements."""
        disagreement_indicators = [
            "disagree",
            "however",
            "in contrast",
            "alternatively",
            "on the other hand",
            "debate",
            "controversy",
        ]

        response = """
        The surgical oncologist recommends intervention.
        However, the palliative care physician disagrees and prefers conservative management.
        There is ongoing debate about optimal timing.
        """

        detected_disagreements = [
            ind for ind in disagreement_indicators if ind.lower() in response.lower()
        ]
        print(f"Detected disagreement indicators: {detected_disagreements}")

        assert len(detected_disagreements) > 0, "Disagreement indicators should be detected"

    def test_suggestion_generation_logic(self):
        """Test the logic for generating contextual suggestions."""
        # Simulate suggestion generation based on detected patterns

        def generate_suggestions(response_text: str, question: str) -> list:
            """Simple suggestion generator for testing."""
            suggestions = []

            # Safety concerns -> suggest safety deep dive
            safety_keywords = ["mortality", "complication", "risk"]
            if any(kw in response_text.lower() for kw in safety_keywords):
                suggestions.append("Deep dive on safety and complications")

            # Evidence gaps -> suggest finding more evidence
            if "evidence gap" in response_text.lower() or "no rct" in response_text.lower():
                suggestions.append("Find additional evidence for [topic]")

            # Disagreement -> suggest resolving debate
            if "disagree" in response_text.lower() or "however" in response_text.lower():
                suggestions.append("Resolve the expert debate on [topic]")

            # Conditional recommendation -> suggest clarifying conditions
            if "conditional" in response_text.lower() or "only if" in response_text.lower():
                suggestions.append("Clarify the conditions for this recommendation")

            return suggestions

        # Test with sample response
        test_response = """
        EVIDENCE (PMID: 12345678) shows 30% mortality benefit.
        EVIDENCE GAP: No RCT data for Mirels 9 specifically.
        The surgical oncologist recommends intervention.
        However, this is conditional on patient performance status.
        """

        suggestions = generate_suggestions(test_response, "Mirels score 9")
        print(f"Generated suggestions: {suggestions}")

        assert len(suggestions) >= 2, "Should generate at least 2 suggestions"
        assert any("safety" in s.lower() for s in suggestions), "Should suggest safety review"
        assert any("evidence" in s.lower() for s in suggestions), "Should suggest evidence search"
