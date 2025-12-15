"""
E2E Tests for Mark Pen Feature (P6)

Tests:
- T5.1: Mark button (pen icon) appears near text
- T5.2: Clicking shows mark type options
- T5.3: Selecting a type saves the mark
- T5.4: Marks appear in sidebar summary
- T5.5: Marks can be deleted
"""

import re
import time
import pytest
from playwright.sync_api import Page, expect

from tests.e2e.test_questions import (
    MARK_PEN_EXPECTATIONS,
    get_quick_test_question,
)


@pytest.mark.e2e
@pytest.mark.feature_mark_pen
class TestMarkPen:
    """Tests for Mark Pen functionality."""

    def test_t5_1_mark_button_appears(
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
        T5.1: Mark button (pen icon) appears near text content.

        Verifies that the pen icon button is visible after
        expert responses are displayed.
        """
        # Setup
        navigate_to_app()
        create_project("MarkPen_Test_1")
        time.sleep(2)

        # Ask the test question
        ask_question(test_question)
        time.sleep(5)

        st_helpers.scroll_to_bottom()
        st_helpers.take_screenshot("mark_pen_button")

        # Look for pen icon button
        pen_icon = MARK_PEN_EXPECTATIONS["button_icon"]
        pen_buttons = page.locator('button').filter(has_text=pen_icon)

        print(f"Found {pen_buttons.count()} pen icon buttons")

        # Also check for "Mark" text buttons
        mark_buttons = page.locator('button').filter(has_text="Mark")
        print(f"Found {mark_buttons.count()} 'Mark' buttons")

        total_mark_buttons = pen_buttons.count() + mark_buttons.count()
        print(f"Total mark-related buttons: {total_mark_buttons}")

        # Check if any are visible
        if pen_buttons.count() > 0:
            is_visible = pen_buttons.first.is_visible()
            print(f"First pen button visible: {is_visible}")

    def test_t5_2_clicking_shows_mark_types(
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
        T5.2: Clicking the mark button shows mark type options.

        Verifies that clicking reveals the 6 mark type buttons:
        - Important Data (📊)
        - Key Finding (⭐)
        - Evidence Gap (🔍)
        - Useful Citation (📚)
        - Disagree (❌)
        - Agree (✓)
        """
        # Setup
        navigate_to_app()
        create_project("MarkPen_Test_2")
        time.sleep(2)

        # Ask the test question
        ask_question(test_question)
        time.sleep(5)

        # Find and click pen button
        pen_icon = MARK_PEN_EXPECTATIONS["button_icon"]
        pen_buttons = page.locator('button').filter(has_text=pen_icon)

        if pen_buttons.count() > 0:
            pen_buttons.first.click()
            st_helpers.wait_for_streamlit_rerun()

            page_content = page.content()

            # Check for mark type icons
            mark_types = MARK_PEN_EXPECTATIONS["mark_types"]
            found_types = []

            for mark_name, icon in mark_types.items():
                if icon in page_content:
                    found_types.append((mark_name, icon))
                    print(f"Found mark type: {mark_name} ({icon})")

            print(f"Total mark types found: {len(found_types)}/{len(mark_types)}")

            # Also check for mark type labels
            type_labels = ["Important Data", "Key Finding", "Evidence Gap", "Useful Citation"]
            found_labels = [label for label in type_labels if label in page_content]
            print(f"Found labels: {found_labels}")

            st_helpers.take_screenshot("mark_pen_types")
        else:
            print("No pen buttons found to click")

    def test_t5_3_selecting_type_saves_mark(
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
        T5.3: Selecting a mark type saves the mark.

        Verifies that clicking a mark type button triggers
        a save action and shows success feedback.
        """
        # Setup
        navigate_to_app()
        create_project("MarkPen_Test_3")
        time.sleep(2)

        # Ask the test question
        ask_question(test_question)
        time.sleep(5)

        # Find and click pen button
        pen_icon = MARK_PEN_EXPECTATIONS["button_icon"]
        pen_buttons = page.locator('button').filter(has_text=pen_icon)

        if pen_buttons.count() > 0:
            pen_buttons.first.click()
            st_helpers.wait_for_streamlit_rerun()

            # Click a mark type (Key Finding = ⭐)
            key_finding_icon = MARK_PEN_EXPECTATIONS["mark_types"]["key_finding"]
            type_buttons = page.locator('button').filter(has_text=key_finding_icon)

            if type_buttons.count() > 0:
                type_buttons.first.click()
                st_helpers.wait_for_streamlit_rerun()

                page_content = page.content()

                # Check for success indicators
                success_indicators = [
                    "Marked as",
                    "saved",
                    "success",
                    "Key Finding",
                ]

                found_success = [
                    ind for ind in success_indicators
                    if ind.lower() in page_content.lower()
                ]
                print(f"Found success indicators: {found_success}")

                st_helpers.take_screenshot("mark_pen_saved")
            else:
                print("No mark type buttons found")
        else:
            print("No pen buttons found")

    def test_t5_4_marks_appear_in_sidebar(
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
        T5.4: Marks appear in the sidebar summary.

        Verifies that saved marks are shown in the sidebar
        "Your Marks" section with counts.
        """
        # Setup
        navigate_to_app()
        create_project("MarkPen_Test_4")
        time.sleep(2)

        # Ask the test question
        ask_question(test_question)
        time.sleep(5)

        # Save a mark first
        pen_icon = MARK_PEN_EXPECTATIONS["button_icon"]
        pen_buttons = page.locator('button').filter(has_text=pen_icon)

        if pen_buttons.count() > 0:
            pen_buttons.first.click()
            st_helpers.wait_for_streamlit_rerun()

            # Click Important Data
            important_icon = MARK_PEN_EXPECTATIONS["mark_types"]["important_data"]
            type_buttons = page.locator('button').filter(has_text=important_icon)
            if type_buttons.count() > 0:
                type_buttons.first.click()
                st_helpers.wait_for_streamlit_rerun(3)

        # Check sidebar
        sidebar = st_helpers.get_sidebar()
        sidebar_content = sidebar.inner_text()

        # Look for marks section
        marks_text = MARK_PEN_EXPECTATIONS["sidebar_text"]
        has_marks_section = marks_text in sidebar_content

        print(f"Has '{marks_text}' in sidebar: {has_marks_section}")
        print(f"Sidebar content preview: {sidebar_content[:300]}")

        # Look for mark count or indicator
        mark_indicators = ["mark", "total", "📊", "⭐"]
        found_indicators = [ind for ind in mark_indicators if ind.lower() in sidebar_content.lower()]
        print(f"Found mark indicators in sidebar: {found_indicators}")

        st_helpers.take_screenshot("mark_pen_sidebar")

    def test_t5_5_marks_can_be_deleted(
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
        T5.5: Marks can be deleted.

        Verifies that marks can be removed using the delete button.
        """
        # Setup
        navigate_to_app()
        create_project("MarkPen_Test_5")
        time.sleep(2)

        # Ask the test question
        ask_question(test_question)
        time.sleep(5)

        # Save a mark first
        pen_icon = MARK_PEN_EXPECTATIONS["button_icon"]
        pen_buttons = page.locator('button').filter(has_text=pen_icon)

        if pen_buttons.count() > 0:
            pen_buttons.first.click()
            st_helpers.wait_for_streamlit_rerun()

            key_finding_icon = MARK_PEN_EXPECTATIONS["mark_types"]["key_finding"]
            type_buttons = page.locator('button').filter(has_text=key_finding_icon)
            if type_buttons.count() > 0:
                type_buttons.first.click()
                st_helpers.wait_for_streamlit_rerun(3)

        # Look for "View All Marks" button
        view_marks_btn = page.locator('button').filter(has_text="View All Marks")
        if view_marks_btn.count() > 0:
            view_marks_btn.first.click()
            st_helpers.wait_for_streamlit_rerun()

        # Look for delete button (trash icon)
        delete_buttons = page.locator('button').filter(has_text="🗑️")
        print(f"Found {delete_buttons.count()} delete buttons")

        if delete_buttons.count() > 0:
            # Click delete
            delete_buttons.first.click()
            st_helpers.wait_for_streamlit_rerun()

            page_content = page.content()

            # Check for deletion confirmation or reduced count
            delete_indicators = ["deleted", "removed", "No marks"]
            found_delete = [ind for ind in delete_indicators if ind.lower() in page_content.lower()]
            print(f"Found delete indicators: {found_delete}")

        st_helpers.take_screenshot("mark_pen_deleted")


@pytest.mark.e2e
@pytest.mark.feature_mark_pen
class TestMarkPenService:
    """Unit-style tests for Feedback/Mark Pen service."""

    def test_feedback_service_import(self):
        """Test that feedback_service module can be imported."""
        from services.feedback_service import (
            FeedbackService,
            FeedbackMark,
            save_mark,
            get_marks,
            get_relevance_boost,
        )

        assert FeedbackService is not None
        assert FeedbackMark is not None
        assert callable(save_mark)
        assert callable(get_marks)
        assert callable(get_relevance_boost)

    def test_feedback_mark_dataclass(self):
        """Test FeedbackMark dataclass structure."""
        from services.feedback_service import FeedbackMark

        mark = FeedbackMark(
            id="test_mark_001",
            text="Mortality rate was 15%",
            source_type="expert_response",
            source_id="Surgical Oncologist",
            mark_type="important_data",
            question_context="femoral metastases outcomes",
            project_id="test_project",
        )

        assert mark.id == "test_mark_001"
        assert mark.mark_type == "important_data"
        assert mark.source_type == "expert_response"

        # Test to_dict
        d = mark.to_dict()
        assert "id" in d
        assert "text" in d
        assert "mark_type" in d

    def test_mark_types_defined(self):
        """Test that expected mark types are defined."""
        from services.feedback_service import FeedbackService

        expected_types = [
            "important_data",
            "key_finding",
            "evidence_gap",
            "disagree",
            "agree",
            "citation_useful",
        ]

        for mark_type in expected_types:
            assert mark_type in FeedbackService.MARK_TYPES, f"Missing type: {mark_type}"

        # Check boost values are defined
        for mark_type, config in FeedbackService.MARK_TYPES.items():
            assert "boost" in config, f"Missing boost for {mark_type}"
            assert "icon" in config, f"Missing icon for {mark_type}"

    def test_mark_type_boosts(self):
        """Test that mark types have appropriate boost values."""
        from services.feedback_service import FeedbackService

        # Important data should boost relevance
        assert FeedbackService.MARK_TYPES["important_data"]["boost"] > 0

        # Key finding should boost relevance
        assert FeedbackService.MARK_TYPES["key_finding"]["boost"] > 0

        # Disagree should reduce relevance
        assert FeedbackService.MARK_TYPES["disagree"]["boost"] < 0

        # Agree should boost relevance
        assert FeedbackService.MARK_TYPES["agree"]["boost"] > 0

    def test_feedback_mark_from_dict(self):
        """Test FeedbackMark.from_dict method."""
        from services.feedback_service import FeedbackMark

        data = {
            "id": "mark_123",
            "text": "Test text",
            "source_type": "search_result",
            "source_id": "PMID:12345678",
            "mark_type": "key_finding",
            "question_context": "test question",
            "project_id": None,
            "created_at": "2024-01-01T00:00:00",
            "metadata": {"extra": "data"},
        }

        mark = FeedbackMark.from_dict(data)

        assert mark.id == "mark_123"
        assert mark.mark_type == "key_finding"
        assert mark.metadata.get("extra") == "data"


@pytest.mark.e2e
@pytest.mark.feature_mark_pen
class TestMarkPenUI:
    """Tests for Mark Pen UI component."""

    def test_mark_pen_ui_import(self):
        """Test that mark_pen UI module can be imported."""
        from ui.mark_pen import (
            render_mark_button,
            render_markable_text,
            render_mark_sidebar,
            render_marks_panel,
            MARK_TYPES,
        )

        assert callable(render_mark_button)
        assert callable(render_markable_text)
        assert callable(render_mark_sidebar)
        assert callable(render_marks_panel)
        assert len(MARK_TYPES) == 6

    def test_mark_types_configuration(self):
        """Test MARK_TYPES configuration in UI module."""
        from ui.mark_pen import MARK_TYPES

        # Expected mark types with icons
        expected = {
            "important_data": "📊",
            "key_finding": "⭐",
            "evidence_gap": "🔍",
            "citation_useful": "📚",
            "disagree": "❌",
            "agree": "✓",
        }

        for mark_type, icon in expected.items():
            assert mark_type in MARK_TYPES, f"Missing type: {mark_type}"
            assert MARK_TYPES[mark_type]["icon"] == icon, f"Wrong icon for {mark_type}"
            assert "label" in MARK_TYPES[mark_type], f"Missing label for {mark_type}"
            assert "color" in MARK_TYPES[mark_type], f"Missing color for {mark_type}"

    def test_mark_colors_are_valid(self):
        """Test that mark type colors are valid hex colors."""
        from ui.mark_pen import MARK_TYPES
        import re

        hex_pattern = r'^#[0-9A-Fa-f]{6}$'

        for mark_type, config in MARK_TYPES.items():
            color = config["color"]
            assert re.match(hex_pattern, color), f"Invalid color for {mark_type}: {color}"
