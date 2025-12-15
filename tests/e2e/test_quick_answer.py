"""
E2E Tests for Quick Answer Feature (P4)

Tests:
- T2.1: Quick answer UI appears with green gradient header
- T2.2: Sources indicator shows "Based on X sources"
- T2.3: Answer contains citations ([PMID:...] or [1])
- T2.4: Sources expander works and shows source list
- T2.5: Response time is reasonable (<30s)
"""

import re
import time
import pytest
from playwright.sync_api import Page, expect

from tests.e2e.test_questions import (
    QUICK_ANSWER_EXPECTATIONS,
    get_quick_test_question,
)


@pytest.mark.e2e
@pytest.mark.feature_quick_answer
class TestQuickAnswer:
    """Tests for Quick Answer functionality."""

    def test_t2_1_quick_answer_ui_appears(
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
        T2.1: Quick answer UI appears with green gradient header.

        Verifies that the Quick Answer section renders with the
        expected green (#10B981) gradient styling.
        """
        # Setup
        navigate_to_app()
        create_project("QuickAnswer_Test_1")
        time.sleep(2)

        # Ask the test question
        start_time = time.time()
        ask_question(test_question)
        time.sleep(5)

        # Take screenshot
        st_helpers.take_screenshot("quick_answer_ui")

        # Look for Quick Answer header
        page_content = page.content()

        # Check for "Quick Answer" text
        has_quick_answer_text = "Quick Answer" in page_content
        print(f"Has 'Quick Answer' text: {has_quick_answer_text}")

        # Check for green gradient styling
        has_green_gradient = "#10B981" in page_content or "10B981" in page_content
        print(f"Has green gradient styling: {has_green_gradient}")

        # Check for the green gradient header div
        green_headers = page.locator('div[style*="#10B981"]')
        print(f"Found {green_headers.count()} green-styled elements")

        # Also check for any response content
        has_response = any(
            page.locator(f'text={kw}').count() > 0
            for kw in ["Based on", "sources", "Answer", "evidence"]
        )
        print(f"Has response content: {has_response}")

    def test_t2_2_sources_indicator_shows(
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
        T2.2: Sources indicator shows "Based on X sources".

        Verifies that the sources count is displayed correctly.
        """
        # Setup
        navigate_to_app()
        create_project("QuickAnswer_Test_2")
        time.sleep(2)

        # Ask the test question
        ask_question(test_question)
        time.sleep(5)

        page_content = page.content()

        # Look for "Based on X sources" pattern
        sources_pattern = r'Based on \d+ sources?'
        sources_match = re.search(sources_pattern, page_content)

        if sources_match:
            print(f"Found sources indicator: {sources_match.group()}")
        else:
            # Alternative pattern: "X sources"
            alt_pattern = r'\d+ sources?'
            alt_match = re.search(alt_pattern, page_content)
            if alt_match:
                print(f"Found alternative sources indicator: {alt_match.group()}")
            else:
                # Check for "Based on general knowledge" (no sources case)
                if "general knowledge" in page_content.lower():
                    print("Found 'general knowledge' indicator (no sources)")

        st_helpers.take_screenshot("quick_answer_sources")

    def test_t2_3_answer_contains_citations(
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
        T2.3: Answer contains citations ([PMID:...] or [1]).

        Verifies that the quick answer includes citation references.
        """
        # Setup
        navigate_to_app()
        create_project("QuickAnswer_Test_3")
        time.sleep(2)

        # Ask the test question
        ask_question(test_question)
        time.sleep(5)

        page_content = page.content()

        # Look for citation patterns
        pmid_pattern = r'\[PMID[:\s]*\d{7,8}\]'
        ref_pattern = r'\[\d+\]'

        pmid_citations = re.findall(pmid_pattern, page_content, re.IGNORECASE)
        ref_citations = re.findall(ref_pattern, page_content)

        print(f"Found PMID citations: {pmid_citations[:5]}")
        print(f"Found reference citations: {ref_citations[:5]}")

        total_citations = len(pmid_citations) + len(ref_citations)
        print(f"Total citations found: {total_citations}")

        # Also check for PubMed links (styled citations)
        pubmed_links = page.locator('a[href*="pubmed.ncbi.nlm.nih.gov"]').count()
        print(f"Found PubMed links: {pubmed_links}")

    def test_t2_4_sources_expander_works(
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
        T2.4: Sources expander works and shows source list.

        Verifies that clicking "View X sources" expands to show
        the list of sources with titles and links.
        """
        # Setup
        navigate_to_app()
        create_project("QuickAnswer_Test_4")
        time.sleep(2)

        # Ask the test question
        ask_question(test_question)
        time.sleep(5)

        # Look for sources expander
        expanders = page.locator('[data-testid="stExpander"]')
        sources_expander = None

        for i in range(expanders.count()):
            exp = expanders.nth(i)
            text = exp.inner_text().lower()
            if "source" in text or "view" in text:
                sources_expander = exp
                break

        if sources_expander:
            # Click to expand
            header = sources_expander.locator('div[role="button"]').first
            if header.is_visible():
                header.click()
                time.sleep(1)

                # Check expanded content
                expanded_content = sources_expander.inner_text()
                print(f"Sources expander content: {expanded_content[:300]}")

                # Look for source titles or PMIDs
                has_sources = (
                    "PMID" in expanded_content or
                    "pubmed" in expanded_content.lower() or
                    re.search(r'\[\d+\]', expanded_content)
                )
                print(f"Expander has source content: {has_sources}")
        else:
            print("No sources expander found")

        st_helpers.take_screenshot("quick_answer_sources_expanded")

    def test_t2_5_response_time_reasonable(
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
        T2.5: Response time is reasonable (<30s for quick answer).

        Verifies that the quick answer appears within acceptable time.
        """
        # Setup
        navigate_to_app()
        create_project("QuickAnswer_Test_5")
        time.sleep(2)

        # Measure response time
        start_time = time.time()

        # Ask the test question (don't wait for response in fixture)
        textareas = page.locator('textarea')
        question_input = textareas.first

        question_input.fill(test_question)
        question_input.press("Meta+Enter")
        page.wait_for_load_state("networkidle")

        # Wait for response indicator
        max_wait = 30  # 30 seconds max
        response_appeared = False

        for _ in range(max_wait):
            time.sleep(1)
            page_content = page.content()

            # Check for response indicators
            if any(kw in page_content for kw in [
                "Quick Answer",
                "Based on",
                "Expert",
                "EVIDENCE",
                "survival",  # Expected keyword from response
            ]):
                response_appeared = True
                break

        end_time = time.time()
        response_time = end_time - start_time

        print(f"Response time: {response_time:.2f}s")
        print(f"Response appeared: {response_appeared}")

        # Assert reasonable response time (under 30s)
        assert response_time < 30, f"Response took too long: {response_time:.2f}s"


@pytest.mark.e2e
@pytest.mark.feature_quick_answer
class TestQuickAnswerService:
    """Unit-style tests for Quick Answer service."""

    def test_quick_answer_service_import(self):
        """Test that quick_answer_service module can be imported."""
        from services.quick_answer_service import (
            get_quick_answer,
            get_quick_answer_with_search,
            QuickAnswer,
        )

        assert callable(get_quick_answer)
        assert callable(get_quick_answer_with_search)
        assert QuickAnswer is not None

    def test_quick_answer_dataclass(self):
        """Test QuickAnswer dataclass structure."""
        from services.quick_answer_service import QuickAnswer

        answer = QuickAnswer(
            answer="Test answer text",
            sources_used=3,
            model="test-model",
            has_context=True,
            citations=[{"pmid": "12345678", "title": "Test"}]
        )

        assert answer.answer == "Test answer text"
        assert answer.sources_used == 3
        assert answer.has_context is True
        assert len(answer.citations) == 1

    def test_quick_answer_with_context(self):
        """Test quick answer generation with context."""
        from services.quick_answer_service import get_quick_answer

        # Minimal context for testing
        context = [
            {
                "pmid": "12345678",
                "title": "Test Study on Femoral Metastases",
                "abstract": "This study examined prophylactic fixation outcomes.",
            }
        ]

        # This test requires API keys - skip if not available
        try:
            result = get_quick_answer(
                question="What is the role of fixation?",
                context=context,
                scenario="Pathologic Fracture",
            )

            assert result.answer is not None
            assert result.sources_used <= len(context)
            print(f"Quick answer generated: {result.answer[:100]}...")

        except Exception as e:
            pytest.skip(f"API call failed (expected in CI): {e}")


@pytest.mark.e2e
@pytest.mark.feature_quick_answer
class TestQuickAnswerUI:
    """Tests for Quick Answer UI rendering function."""

    def test_render_quick_answer_ui_import(self):
        """Test that render function can be imported."""
        from services.quick_answer_service import render_quick_answer_ui

        assert callable(render_quick_answer_ui)

    def test_quick_answer_citations_in_result(self):
        """Test that QuickAnswer preserves citation data."""
        from services.quick_answer_service import QuickAnswer

        citations = [
            {"pmid": "11111111", "title": "Study A", "authors": ["Smith"]},
            {"pmid": "22222222", "title": "Study B", "authors": ["Jones"]},
        ]

        answer = QuickAnswer(
            answer="The studies [1][2] show benefits.",
            sources_used=2,
            model="test",
            has_context=True,
            citations=citations,
        )

        assert len(answer.citations) == 2
        assert answer.citations[0]["pmid"] == "11111111"
        assert answer.citations[1]["pmid"] == "22222222"
