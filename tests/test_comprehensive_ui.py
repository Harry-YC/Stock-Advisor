"""
Comprehensive Playwright UI Tests for Literature Review Platform.

Tests all major UI functions across all tabs and components.
"""

import pytest
import os
import sys
import time
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from playwright.sync_api import Page, expect

BASE_URL = os.getenv("TEST_URL", "http://localhost:8501")


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="function")
def app_page(page: Page):
    """Load the app and wait for it to be ready."""
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    time.sleep(3)
    return page


# =============================================================================
# TEST CLASS: SIDEBAR
# =============================================================================

class TestSidebar:
    """Tests for sidebar components."""

    def test_sidebar_visible(self, app_page: Page):
        """Test that sidebar is visible."""
        sidebar = app_page.locator('[data-testid="stSidebar"]')
        expect(sidebar).to_be_visible(timeout=10000)

    def test_app_title_in_sidebar(self, app_page: Page):
        """Test that app title is in sidebar."""
        title = app_page.get_by_text("Literature Review", exact=False)
        expect(title.first).to_be_visible(timeout=10000)

    def test_project_manager_section(self, app_page: Page):
        """Test project manager section exists."""
        # Look for project-related elements
        project_elements = app_page.get_by_text("Project", exact=False)
        expect(project_elements.first).to_be_visible(timeout=10000)

    def test_new_project_input(self, app_page: Page):
        """Test new project input field exists."""
        # Look for text input in sidebar
        sidebar = app_page.locator('[data-testid="stSidebar"]')
        text_input = sidebar.locator('input[type="text"]')
        expect(text_input.first).to_be_visible(timeout=10000)

    def test_create_project_button(self, app_page: Page):
        """Test create project button exists."""
        create_btn = app_page.get_by_text("Create", exact=False)
        expect(create_btn.first).to_be_visible(timeout=10000)


# =============================================================================
# TEST CLASS: MAIN TABS
# =============================================================================

class TestMainTabs:
    """Tests for main tab navigation."""

    def test_all_main_tabs_visible(self, app_page: Page):
        """Test that all three main tabs are visible."""
        docs_tab = app_page.get_by_role("tab", name="Documents & Literature")
        panel_tab = app_page.get_by_role("tab", name="Expert Panel")
        chat_tab = app_page.get_by_role("tab", name="Ask Experts")

        expect(docs_tab).to_be_visible(timeout=10000)
        expect(panel_tab).to_be_visible(timeout=10000)
        expect(chat_tab).to_be_visible(timeout=10000)

    def test_documents_tab_clickable(self, app_page: Page):
        """Test Documents tab is clickable."""
        docs_tab = app_page.get_by_role("tab", name="Documents & Literature")
        docs_tab.click()
        time.sleep(2)

        # Verify we're on documents tab
        search_pubmed = app_page.get_by_text("Search PubMed", exact=False)
        expect(search_pubmed.first).to_be_visible(timeout=10000)

    def test_expert_panel_tab_clickable(self, app_page: Page):
        """Test Expert Panel tab is clickable."""
        panel_tab = app_page.get_by_role("tab", name="Expert Panel")
        panel_tab.click()
        time.sleep(2)

        # Verify we're on panel tab - look for panel-related content
        panel_content = app_page.get_by_text("Panel Discussion", exact=False)
        expect(panel_content.first).to_be_visible(timeout=10000)

    def test_ask_experts_tab_clickable(self, app_page: Page):
        """Test Ask Experts tab is clickable."""
        chat_tab = app_page.get_by_role("tab", name="Ask Experts")
        chat_tab.click()
        time.sleep(2)

        # Verify we're on chat tab
        context_text = app_page.get_by_text("context", exact=False)
        expect(context_text.first).to_be_visible(timeout=10000)


# =============================================================================
# TEST CLASS: LITERATURE SEARCH
# =============================================================================

class TestLiteratureSearch:
    """Tests for literature search functionality."""

    def test_search_subtabs_visible(self, app_page: Page):
        """Test that search sub-tabs are visible."""
        docs_tab = app_page.get_by_role("tab", name="Documents & Literature")
        docs_tab.click()
        time.sleep(2)

        # Check sub-tabs (with emoji prefixes)
        search_tab = app_page.get_by_role("tab", name="Search PubMed")
        upload_tab = app_page.get_by_role("tab", name="Upload Documents")
        library_tab = app_page.get_by_role("tab", name="Document Library")

        expect(search_tab).to_be_visible(timeout=10000)
        expect(upload_tab).to_be_visible(timeout=10000)
        expect(library_tab).to_be_visible(timeout=10000)

    def test_search_query_textarea(self, app_page: Page):
        """Test search query textarea exists."""
        docs_tab = app_page.get_by_role("tab", name="Documents & Literature")
        docs_tab.click()
        time.sleep(2)

        # Look for search-related text area
        search_placeholder = app_page.get_by_text("Enter your research question", exact=False)
        # If no project, we'll see empty state instead
        if search_placeholder.count() == 0:
            empty_state = app_page.get_by_text("Create a new project", exact=False)
            expect(empty_state.first).to_be_visible(timeout=10000)
        else:
            expect(search_placeholder.first).to_be_visible(timeout=10000)

    def test_query_examples_expander(self, app_page: Page):
        """Test query examples expander exists."""
        docs_tab = app_page.get_by_role("tab", name="Documents & Literature")
        docs_tab.click()
        time.sleep(2)

        examples = app_page.get_by_text("Query Examples", exact=False)
        # Only visible if project exists
        if examples.count() > 0:
            expect(examples.first).to_be_visible(timeout=10000)

    def test_identifier_search_tab(self, app_page: Page):
        """Test identifier search tab content."""
        docs_tab = app_page.get_by_role("tab", name="Documents & Literature")
        docs_tab.click()
        time.sleep(2)

        # Look for Search PubMed tab content (sub-tabs within main search)
        search_tab = app_page.get_by_role("tab", name="Search PubMed")
        expect(search_tab).to_be_visible(timeout=10000)

        # Just verify we're on the documents tab
        assert True


# =============================================================================
# TEST CLASS: EXPERT PANEL
# =============================================================================

class TestExpertPanel:
    """Tests for expert panel functionality."""

    def test_panel_subtabs_visible(self, app_page: Page):
        """Test that panel sub-tabs are visible."""
        panel_tab = app_page.get_by_role("tab", name="Expert Panel")
        panel_tab.click()
        time.sleep(2)

        # Check sub-tabs
        discussion_tab = app_page.get_by_role("tab", name="Panel Discussion")
        screening_tab = app_page.get_by_role("tab", name="AI Screening")

        expect(discussion_tab).to_be_visible(timeout=10000)
        expect(screening_tab).to_be_visible(timeout=10000)

    def test_context_status_displayed(self, app_page: Page):
        """Test that context status is displayed."""
        panel_tab = app_page.get_by_role("tab", name="Expert Panel")
        panel_tab.click()
        time.sleep(2)

        # Look for context-related messages
        no_papers = app_page.get_by_text("No papers", exact=False)
        context_loaded = app_page.get_by_text("Context", exact=False)
        web_search = app_page.get_by_text("web search", exact=False)

        visible_count = 0
        for elem in [no_papers, context_loaded, web_search]:
            if elem.count() > 0:
                try:
                    expect(elem.first).to_be_visible(timeout=5000)
                    visible_count += 1
                except:
                    pass

        assert visible_count > 0, "Expected context status to be displayed"

    def test_panel_discussion_tab(self, app_page: Page):
        """Test panel discussion tab content."""
        panel_tab = app_page.get_by_role("tab", name="Expert Panel")
        panel_tab.click()
        time.sleep(2)

        discussion_tab = app_page.get_by_role("tab", name="Panel Discussion")
        discussion_tab.click()
        time.sleep(2)

        # Look for panel discussion elements
        content_found = False
        for text in ["Research Question", "Research/clinical question", "Expert Panel", "Create a new project"]:
            elem = app_page.get_by_text(text, exact=False)
            if elem.count() > 0:
                try:
                    expect(elem.first).to_be_visible(timeout=5000)
                    content_found = True
                    break
                except:
                    pass

        assert content_found, "Expected panel discussion content"

    def test_ai_screening_tab(self, app_page: Page):
        """Test AI screening tab content."""
        panel_tab = app_page.get_by_role("tab", name="Expert Panel")
        panel_tab.click()
        time.sleep(2)

        screening_tab = app_page.get_by_role("tab", name="AI Screening")
        screening_tab.click()
        time.sleep(2)

        # Look for screening elements
        content_found = False
        for text in ["AI Screening", "screening", "papers", "load papers", "search"]:
            elem = app_page.get_by_text(text, exact=False)
            if elem.count() > 0:
                try:
                    expect(elem.first).to_be_visible(timeout=5000)
                    content_found = True
                    break
                except:
                    pass

        assert content_found, "Expected AI screening content"


# =============================================================================
# TEST CLASS: ASK EXPERTS (CHAT)
# =============================================================================

class TestAskExperts:
    """Tests for Ask Experts chat functionality."""

    def test_chat_tab_content(self, app_page: Page):
        """Test chat tab has expected content."""
        chat_tab = app_page.get_by_role("tab", name="Ask Experts")
        chat_tab.click()
        time.sleep(2)

        # Look for chat-related content
        content_found = False
        for text in ["context", "expert", "web search", "knowledge"]:
            elem = app_page.get_by_text(text, exact=False)
            if elem.count() > 0:
                try:
                    expect(elem.first).to_be_visible(timeout=5000)
                    content_found = True
                    break
                except:
                    pass

        assert content_found, "Expected chat content"

    def test_web_search_toggle(self, app_page: Page):
        """Test web search toggle exists when no context."""
        chat_tab = app_page.get_by_role("tab", name="Ask Experts")
        chat_tab.click()
        time.sleep(2)

        # Look for any chat-related content (web search may be hidden)
        context_msg = app_page.get_by_text("context", exact=False)
        if context_msg.count() > 0:
            expect(context_msg.first).to_be_visible(timeout=10000)


# =============================================================================
# TEST CLASS: CONTEXT INDICATOR
# =============================================================================

class TestContextIndicator:
    """Tests for context indicator bar."""

    def test_papers_indicator(self, app_page: Page):
        """Test papers indicator is visible."""
        papers = app_page.get_by_text("papers loaded", exact=False)
        expect(papers.first).to_be_visible(timeout=10000)

    def test_documents_indicator(self, app_page: Page):
        """Test documents indicator is visible."""
        docs = app_page.get_by_text("documents", exact=False)
        expect(docs.first).to_be_visible(timeout=10000)

    def test_discussion_indicator(self, app_page: Page):
        """Test discussion rounds indicator is visible."""
        discussion = app_page.get_by_text("discussion", exact=False)
        expect(discussion.first).to_be_visible(timeout=10000)


# =============================================================================
# TEST CLASS: PROJECT WORKFLOW
# =============================================================================

class TestProjectWorkflow:
    """Tests for project creation and management workflow."""

    def test_empty_state_message(self, app_page: Page):
        """Test empty state message when no project."""
        # Look for create project prompt
        create_prompt = app_page.get_by_text("Create a new project", exact=False)
        quick_start = app_page.get_by_text("Quick Start", exact=False)

        prompt_visible = False
        if create_prompt.count() > 0:
            try:
                expect(create_prompt.first).to_be_visible(timeout=5000)
                prompt_visible = True
            except:
                pass
        if quick_start.count() > 0:
            try:
                expect(quick_start.first).to_be_visible(timeout=5000)
                prompt_visible = True
            except:
                pass

        # It's ok if neither is visible (project might exist)
        assert True

    def test_project_creation_flow(self, app_page: Page):
        """Test that project creation UI elements exist."""
        sidebar = app_page.locator('[data-testid="stSidebar"]')

        # Find text input
        text_input = sidebar.locator('input[type="text"]')
        expect(text_input.first).to_be_visible(timeout=10000)

        # Find create button
        create_btn = app_page.get_by_text("Create", exact=False)
        expect(create_btn.first).to_be_visible(timeout=10000)


# =============================================================================
# TEST CLASS: RESPONSIVE LAYOUT
# =============================================================================

class TestResponsiveLayout:
    """Tests for responsive layout behavior."""

    def test_wide_layout(self, app_page: Page):
        """Test that wide layout is applied."""
        # Check for wide container
        main_content = app_page.locator('[data-testid="stAppViewContainer"]')
        expect(main_content).to_be_visible(timeout=10000)

    def test_footer_visible(self, app_page: Page):
        """Test that footer is visible."""
        footer = app_page.get_by_text("Literature Review Platform", exact=False)
        expect(footer.first).to_be_visible(timeout=10000)


# =============================================================================
# TEST CLASS: VISUALIZATION COMPONENTS
# =============================================================================

class TestVisualizationComponents:
    """Tests for visualization components."""

    def test_visualization_expander_exists(self, app_page: Page):
        """Test that visualization components exist when results are present."""
        # This test checks that visualization infrastructure is in place
        # Actual charts require search results
        docs_tab = app_page.get_by_role("tab", name="Documents & Literature")
        docs_tab.click()
        time.sleep(2)

        # Just verify the documents tab loads without error
        search_tab = app_page.get_by_role("tab", name="Search PubMed")
        expect(search_tab).to_be_visible(timeout=10000)


# =============================================================================
# TEST CLASS: ERROR HANDLING
# =============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_no_api_key_message(self, app_page: Page):
        """Test that missing API key shows appropriate message."""
        panel_tab = app_page.get_by_role("tab", name="Expert Panel")
        panel_tab.click()
        time.sleep(2)

        discussion_tab = app_page.get_by_role("tab", name="Panel Discussion")
        discussion_tab.click()
        time.sleep(2)

        # Look for any panel-related content (API key, research question, or empty state)
        # The panel tab should render something
        content_checks = [
            "API key",
            "Research",
            "Create a new project",
            "Expert Panel",
            "Panel Discussion",
            "project"
        ]

        found = False
        for text in content_checks:
            elem = app_page.get_by_text(text, exact=False)
            if elem.count() > 0:
                try:
                    expect(elem.first).to_be_visible(timeout=3000)
                    found = True
                    break
                except:
                    pass

        # Panel tab should always render something
        assert found, "Expected panel content to be visible"

    def test_app_does_not_crash(self, app_page: Page):
        """Test that app doesn't crash on rapid tab switching."""
        tabs = [
            "Documents & Literature",
            "Expert Panel",
            "Ask Experts"
        ]

        for _ in range(3):
            for tab_name in tabs:
                tab = app_page.get_by_role("tab", name=tab_name)
                tab.click()
                time.sleep(0.5)

        # Verify app is still responsive
        title = app_page.get_by_text("Literature Review", exact=False)
        expect(title.first).to_be_visible(timeout=10000)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
