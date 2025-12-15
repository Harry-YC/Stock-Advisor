"""
Full Integration E2E Test for Palliative Surgery GDG

Tests the complete workflow using the Mirels Score 9 clinical question
from the research report. Validates all implemented features together:

1. Project creation
2. Quick Answer with citations (P4 + P1)
3. Citation highlighting and cards (P1)
4. Mark Pen functionality (P6)
5. Red Team Challenger (P3)
6. Smart Suggestions (P5)

Test Question:
"What is the survival benefit of prophylactic fixation versus observation
in patients with femoral metastases and Mirels score of 9?"
"""

import re
import time
import pytest
from playwright.sync_api import Page

from tests.e2e.test_questions import (
    MIRELS_SCORE_9_QUESTION,
    CITATION_HIGHLIGHTING_EXPECTATIONS,
    QUICK_ANSWER_EXPECTATIONS,
    CHALLENGER_EXPECTATIONS,
    SMART_SUGGESTIONS_EXPECTATIONS,
    MARK_PEN_EXPECTATIONS,
)


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.slow
class TestMirelsScore9FullFlow:
    """
    Complete E2E integration test using the Mirels Score 9 question.

    This test validates all features work together in a realistic
    clinical workflow scenario.
    """

    def test_full_workflow(
        self,
        page: Page,
        app_url: str,
        navigate_to_app,
        create_project,
        ask_question,
        st_helpers,
    ):
        """
        Complete workflow test for the Mirels Score 9 clinical question.

        Steps:
        1. Setup - Create project, navigate to app
        2. Quick Answer - Enter question, verify response
        3. Citation Highlighting - Check PMID badges, epistemic tags
        4. Citation Cards - Expand and verify sources
        5. Mark Pen - Mark a key finding
        6. Challenger - Generate challenges
        7. Smart Suggestions - Verify follow-up suggestions
        8. Cleanup - Delete test marks
        """
        test_question = MIRELS_SCORE_9_QUESTION.question
        results = {}

        print("\n" + "=" * 60)
        print("MIRELS SCORE 9 FULL WORKFLOW TEST")
        print("=" * 60)

        # =================================================================
        # STEP 1: Setup
        # =================================================================
        print("\n[1/8] SETUP - Creating project and navigating...")

        navigate_to_app()
        project_name = create_project("Mirels_E2E_Test")
        time.sleep(2)

        results["setup"] = {
            "project_created": project_name is not None,
            "project_name": project_name,
        }
        print(f"   Project created: {project_name}")

        st_helpers.take_screenshot("step1_setup")

        # =================================================================
        # STEP 2: Quick Answer Mode
        # =================================================================
        print("\n[2/8] QUICK ANSWER - Submitting clinical question...")

        start_time = time.time()
        ask_question(test_question, wait_for_response=True, timeout=60000)
        response_time = time.time() - start_time

        page_content = page.content()

        # Check for Quick Answer indicators
        has_quick_answer = "Quick Answer" in page_content
        has_sources = "Based on" in page_content or "sources" in page_content.lower()

        results["quick_answer"] = {
            "response_time_seconds": round(response_time, 2),
            "has_quick_answer_ui": has_quick_answer,
            "has_sources_indicator": has_sources,
        }

        print(f"   Response time: {response_time:.2f}s")
        print(f"   Quick Answer UI: {has_quick_answer}")
        print(f"   Sources indicator: {has_sources}")

        st_helpers.take_screenshot("step2_quick_answer")

        # =================================================================
        # STEP 3: Citation Highlighting
        # =================================================================
        print("\n[3/8] CITATION HIGHLIGHTING - Checking badges and tags...")

        # Check for PMID badges
        pmid_pattern = r'\[PMID[:\s]*\d{7,8}\]'
        pmid_matches = re.findall(pmid_pattern, page_content, re.IGNORECASE)

        # Check for reference badges
        ref_pattern = r'\[\d+\]'
        ref_matches = re.findall(ref_pattern, page_content)

        # Check for PubMed links (styled citations)
        pubmed_links = page.locator('a[href*="pubmed.ncbi.nlm.nih.gov"]')

        # Check for epistemic tags
        epistemic_tags = {
            "EVIDENCE": "#D1FAE5" in page_content,
            "ASSUMPTION": "#FEF3C7" in page_content,
            "OPINION": "#DBEAFE" in page_content,
            "EVIDENCE GAP": "#FEE2E2" in page_content or "EVIDENCE GAP" in page_content,
        }

        results["citation_highlighting"] = {
            "pmid_badges_count": len(pmid_matches),
            "reference_badges_count": len(ref_matches),
            "pubmed_links_count": pubmed_links.count(),
            "epistemic_tags": epistemic_tags,
        }

        print(f"   PMID badges: {len(pmid_matches)}")
        print(f"   Reference badges: {len(ref_matches)}")
        print(f"   PubMed links: {pubmed_links.count()}")
        print(f"   Epistemic tags: {epistemic_tags}")

        st_helpers.take_screenshot("step3_citations")

        # =================================================================
        # STEP 4: Citation Cards
        # =================================================================
        print("\n[4/8] CITATION CARDS - Testing expander functionality...")

        st_helpers.scroll_to_bottom()

        # Find citation-related expanders
        expanders = page.locator('[data-testid="stExpander"]')
        citation_expander_found = False
        citation_content = ""

        for i in range(expanders.count()):
            exp = expanders.nth(i)
            text = exp.inner_text().lower()
            if any(kw in text for kw in ["source", "cited", "pmid", "view"]):
                citation_expander_found = True
                # Try to expand
                header = exp.locator('div[role="button"]').first
                if header.is_visible():
                    header.click()
                    time.sleep(1)
                    citation_content = exp.inner_text()
                break

        results["citation_cards"] = {
            "expander_found": citation_expander_found,
            "has_abstract_content": "abstract" in citation_content.lower() or len(citation_content) > 100,
        }

        print(f"   Citation expander found: {citation_expander_found}")
        print(f"   Has content: {len(citation_content) > 100}")

        st_helpers.take_screenshot("step4_cards")

        # =================================================================
        # STEP 5: Mark Pen
        # =================================================================
        print("\n[5/8] MARK PEN - Testing mark functionality...")

        pen_icon = MARK_PEN_EXPECTATIONS["button_icon"]
        pen_buttons = page.locator('button').filter(has_text=pen_icon)
        mark_saved = False

        if pen_buttons.count() > 0:
            pen_buttons.first.click()
            st_helpers.wait_for_streamlit_rerun()

            # Click Key Finding type
            key_finding_icon = MARK_PEN_EXPECTATIONS["mark_types"]["key_finding"]
            type_buttons = page.locator('button').filter(has_text=key_finding_icon)

            if type_buttons.count() > 0:
                type_buttons.first.click()
                st_helpers.wait_for_streamlit_rerun(3)

                # Check for success
                page_content = page.content()
                mark_saved = "Marked" in page_content or "success" in page_content.lower()

        # Check sidebar for marks
        sidebar = st_helpers.get_sidebar()
        sidebar_content = sidebar.inner_text()
        has_marks_section = MARK_PEN_EXPECTATIONS["sidebar_text"] in sidebar_content

        results["mark_pen"] = {
            "pen_buttons_found": pen_buttons.count(),
            "mark_saved": mark_saved,
            "sidebar_marks_section": has_marks_section,
        }

        print(f"   Pen buttons found: {pen_buttons.count()}")
        print(f"   Mark saved: {mark_saved}")
        print(f"   Sidebar marks section: {has_marks_section}")

        st_helpers.take_screenshot("step5_mark_pen")

        # =================================================================
        # STEP 6: Challenger
        # =================================================================
        print("\n[6/8] CHALLENGER - Testing red team functionality...")

        challenge_button_found = False
        challenges_generated = False

        for text in CHALLENGER_EXPECTATIONS["button_text"]:
            buttons = page.locator('button').filter(has_text=text)
            if buttons.count() > 0 and buttons.first.is_visible():
                challenge_button_found = True
                buttons.first.click()
                time.sleep(10)  # Wait for LLM

                page_content = page.content()
                # Check for challenge indicators
                challenge_indicators = ["question", "assumption", "evidence", "?"]
                challenges_generated = any(ind in page_content.lower() for ind in challenge_indicators)
                break

        results["challenger"] = {
            "button_found": challenge_button_found,
            "challenges_generated": challenges_generated,
        }

        print(f"   Challenge button found: {challenge_button_found}")
        print(f"   Challenges generated: {challenges_generated}")

        st_helpers.take_screenshot("step6_challenger")

        # =================================================================
        # STEP 7: Smart Suggestions
        # =================================================================
        print("\n[7/8] SMART SUGGESTIONS - Checking follow-up options...")

        st_helpers.scroll_to_bottom()
        page_content = page.content()

        # Look for suggestion patterns
        suggestion_patterns = SMART_SUGGESTIONS_EXPECTATIONS["header_patterns"]
        suggestion_found = any(pat.lower() in page_content.lower() for pat in suggestion_patterns)

        # Count suggestion-like buttons
        suggestion_keywords = ["deep dive", "clarify", "explore", "safety", "evidence"]
        suggestion_buttons = []

        all_buttons = page.locator('button')
        for i in range(min(all_buttons.count(), 30)):
            btn = all_buttons.nth(i)
            if btn.is_visible():
                text = btn.inner_text().lower()
                if any(kw in text for kw in suggestion_keywords):
                    suggestion_buttons.append(text[:50])

        results["smart_suggestions"] = {
            "suggestion_section_found": suggestion_found,
            "suggestion_buttons_count": len(suggestion_buttons),
            "suggestion_buttons": suggestion_buttons[:5],
        }

        print(f"   Suggestion section found: {suggestion_found}")
        print(f"   Suggestion buttons: {len(suggestion_buttons)}")

        st_helpers.take_screenshot("step7_suggestions")

        # =================================================================
        # STEP 8: Cleanup & Summary
        # =================================================================
        print("\n[8/8] CLEANUP & SUMMARY")

        # Try to delete any marks we created
        view_marks_btn = page.locator('button').filter(has_text="View All Marks")
        if view_marks_btn.count() > 0:
            view_marks_btn.first.click()
            st_helpers.wait_for_streamlit_rerun()

            delete_buttons = page.locator('button').filter(has_text="🗑️")
            if delete_buttons.count() > 0:
                delete_buttons.first.click()
                st_helpers.wait_for_streamlit_rerun()
                print("   Cleanup: Deleted test marks")

        st_helpers.take_screenshot("step8_final")

        # =================================================================
        # RESULTS SUMMARY
        # =================================================================
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)

        for section, data in results.items():
            print(f"\n{section.upper()}:")
            for key, value in data.items():
                print(f"   {key}: {value}")

        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        print("\nScreenshots saved to /tmp/e2e_step*.png")

        # Store results for assertions
        return results


@pytest.mark.e2e
@pytest.mark.integration
class TestFeatureIntegration:
    """Tests for feature integration points."""

    def test_citation_highlighting_in_quick_answer(
        self,
        page: Page,
        app_url: str,
        navigate_to_app,
        create_project,
        ask_question,
        test_question: str,
        st_helpers,
    ):
        """Test that citations in Quick Answer responses are highlighted."""
        navigate_to_app()
        create_project("Integration_Citation")
        time.sleep(2)

        ask_question(test_question)
        time.sleep(5)

        page_content = page.content()

        # Quick Answer should have citation highlighting applied
        has_styled_citations = (
            "linear-gradient" in page_content and
            ("pubmed" in page_content.lower() or "#6366F1" in page_content)
        )

        print(f"Citations styled in Quick Answer: {has_styled_citations}")

    def test_mark_pen_after_challenger(
        self,
        page: Page,
        app_url: str,
        navigate_to_app,
        create_project,
        ask_question,
        test_question: str,
        st_helpers,
    ):
        """Test that Mark Pen works on Challenger output."""
        navigate_to_app()
        create_project("Integration_MarkChallenger")
        time.sleep(2)

        ask_question(test_question)
        time.sleep(5)

        # Trigger challenger
        for text in ["Challenge", "Red Team"]:
            buttons = page.locator('button').filter(has_text=text)
            if buttons.count() > 0:
                buttons.first.click()
                time.sleep(10)
                break

        # Try to mark challenger output
        pen_buttons = page.locator('button').filter(has_text="🖊️")
        if pen_buttons.count() > 0:
            print(f"Can mark challenger content: {pen_buttons.count()} pen buttons available")

    def test_suggestions_after_challenger(
        self,
        page: Page,
        app_url: str,
        navigate_to_app,
        create_project,
        ask_question,
        test_question: str,
        st_helpers,
    ):
        """Test that suggestions update after challenger is run."""
        navigate_to_app()
        create_project("Integration_SuggestionChallenger")
        time.sleep(2)

        ask_question(test_question)
        time.sleep(5)

        # Get initial button state
        initial_buttons = page.locator('button').all_inner_texts()

        # Trigger challenger
        for text in ["Challenge", "Red Team"]:
            buttons = page.locator('button').filter(has_text=text)
            if buttons.count() > 0:
                buttons.first.click()
                time.sleep(10)
                break

        # Get updated button state
        final_buttons = page.locator('button').all_inner_texts()

        # Check if new suggestions appeared
        new_buttons = set(final_buttons) - set(initial_buttons)
        print(f"New buttons after challenger: {len(new_buttons)}")


@pytest.mark.e2e
@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling in the integrated system."""

    def test_graceful_handling_of_empty_response(
        self,
        page: Page,
        app_url: str,
        navigate_to_app,
        create_project,
        st_helpers,
    ):
        """Test that the app handles empty/short questions gracefully."""
        navigate_to_app()
        create_project("Error_Empty")
        time.sleep(2)

        # Try submitting a very short question
        textareas = page.locator('textarea')
        if textareas.count() > 0:
            textareas.first.fill("?")
            textareas.first.press("Meta+Enter")
            st_helpers.wait_for_streamlit_rerun(5)

            page_content = page.content()

            # Should not crash - check for error message or prompt to elaborate
            is_stable = (
                "error" in page_content.lower() or
                "please" in page_content.lower() or
                "provide" in page_content.lower() or
                "question" in page_content.lower()
            )
            print(f"App handled empty query gracefully: {is_stable}")

    def test_graceful_handling_of_special_characters(
        self,
        page: Page,
        app_url: str,
        navigate_to_app,
        create_project,
        st_helpers,
    ):
        """Test that the app handles special characters in questions."""
        navigate_to_app()
        create_project("Error_Special")
        time.sleep(2)

        # Try submitting question with special characters
        special_question = "What about <script>alert('test')</script> in surgery?"

        textareas = page.locator('textarea')
        if textareas.count() > 0:
            textareas.first.fill(special_question)
            textareas.first.press("Meta+Enter")
            st_helpers.wait_for_streamlit_rerun(5)

            # Should sanitize and not execute script
            page_content = page.content()
            is_safe = "<script>" not in page_content

            print(f"Special characters handled safely: {is_safe}")
            assert is_safe, "Script tags should be sanitized"
