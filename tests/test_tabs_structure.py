"""
Test the new three-tab structure.
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


class TestThreeTabStructure:
    """Tests for the new three-tab UI structure."""

    def test_main_tabs_visible(self, page: Page):
        """Test that the three main tabs are visible."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        # Check for three main tabs
        docs_tab = page.get_by_role("tab", name="Documents & Literature")
        panel_tab = page.get_by_role("tab", name="Expert Panel")
        chat_tab = page.get_by_role("tab", name="Ask Experts")

        expect(docs_tab).to_be_visible(timeout=10000)
        expect(panel_tab).to_be_visible(timeout=10000)
        expect(chat_tab).to_be_visible(timeout=10000)

    def test_documents_tab_has_subtabs(self, page: Page):
        """Test that Documents tab has sub-tabs."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        # Click on Documents tab
        docs_tab = page.get_by_role("tab", name="Documents & Literature")
        docs_tab.click()
        time.sleep(2)

        # Check for sub-tabs
        search_tab = page.get_by_role("tab", name="Search PubMed")
        upload_tab = page.get_by_role("tab", name="Upload Documents")
        library_tab = page.get_by_role("tab", name="Document Library")

        expect(search_tab).to_be_visible(timeout=10000)
        expect(upload_tab).to_be_visible(timeout=10000)
        expect(library_tab).to_be_visible(timeout=10000)

    def test_expert_panel_tab_accessible(self, page: Page):
        """Test that Expert Panel tab is accessible without papers."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        # Click on Expert Panel tab (with emoji prefix)
        panel_tab = page.get_by_role("tab", name="Expert Panel")
        panel_tab.click()
        time.sleep(3)  # Give time for tab content to render

        # Check for any panel-related content visible on this tab
        # Sub-tabs have emoji prefixes: "👥 Panel Discussion", "🤖 AI Screening"
        no_papers_msg = page.get_by_text("No papers or documents loaded", exact=False)
        context_loaded = page.get_by_text("Context loaded", exact=False)
        web_search_checkbox = page.get_by_text("web search for real-time", exact=False)

        # Any of these indicates we're on the Expert Panel tab
        visible_count = 0
        if no_papers_msg.count() > 0:
            try:
                expect(no_papers_msg.first).to_be_visible(timeout=5000)
                visible_count += 1
            except:
                pass
        if context_loaded.count() > 0:
            try:
                expect(context_loaded.first).to_be_visible(timeout=5000)
                visible_count += 1
            except:
                pass
        if web_search_checkbox.count() > 0:
            try:
                expect(web_search_checkbox.first).to_be_visible(timeout=5000)
                visible_count += 1
            except:
                pass

        # At least one panel-related element should be visible
        assert visible_count > 0, "Expected to find panel-related content"

    def test_chat_tab_accessible(self, page: Page):
        """Test that Chat tab is accessible without context."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        # Click on Chat tab
        chat_tab = page.get_by_role("tab", name="Ask Experts")
        chat_tab.click()
        time.sleep(3)  # Give time for tab content to render

        # Check for any chat-related content visible on this tab
        no_context_msg = page.get_by_text("No context loaded", exact=False)
        context_available = page.get_by_text("Context available", exact=False)
        expert_knowledge = page.get_by_text("expert knowledge", exact=False)
        chat_conversation = page.get_by_text("conversation", exact=False)

        # Any of these indicates we're on the Chat tab
        visible_count = 0
        if no_context_msg.count() > 0:
            try:
                expect(no_context_msg.first).to_be_visible(timeout=5000)
                visible_count += 1
            except:
                pass
        if context_available.count() > 0:
            try:
                expect(context_available.first).to_be_visible(timeout=5000)
                visible_count += 1
            except:
                pass
        if expert_knowledge.count() > 0:
            try:
                expect(expert_knowledge.first).to_be_visible(timeout=5000)
                visible_count += 1
            except:
                pass
        if chat_conversation.count() > 0:
            try:
                expect(chat_conversation.first).to_be_visible(timeout=5000)
                visible_count += 1
            except:
                pass

        # At least one chat-related element should be visible
        assert visible_count > 0, "Expected to find chat-related content"

    def test_context_indicator_visible(self, page: Page):
        """Test that context indicator bar is visible."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        # Check for context indicators
        papers_indicator = page.get_by_text("papers loaded", exact=False)
        docs_indicator = page.get_by_text("documents", exact=False)

        expect(papers_indicator.first).to_be_visible(timeout=10000)
        expect(docs_indicator.first).to_be_visible(timeout=10000)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
