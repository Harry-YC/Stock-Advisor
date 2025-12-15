
import pytest
import os
import sys
import time
from pathlib import Path
from playwright.sync_api import Page, expect

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_URL = os.getenv("TEST_URL", "http://localhost:8501")

@pytest.fixture(scope="function")
def app_page(page: Page):
    """Load the app and wait for it to be ready."""
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    time.sleep(3)
    return page

class TestGeminiIntegration:
    """
    UI Verification for Gemini Integration.
    Ensures that the Expert Panel and Chat interface load correctly
    after the refactoring of underlying services.
    """

    def test_expert_panel_loads_without_error(self, app_page: Page):
        """Verify Expert Panel tab loads and services don't crash the UI."""
        # Navigate to Expert Panel
        app_page.get_by_role("tab", name="Expert Panel").click()
        time.sleep(2)

        # Check for sub-tabs (indicates successful load of the module)
        expect(app_page.get_by_role("tab", name="Panel Discussion")).to_be_visible()
        expect(app_page.get_by_role("tab", name="AI Screening")).to_be_visible()

        # Check for specific UI elements that rely on imports
        # The 'Strategy' or 'Conflict' options might be visible or hidden depending on state,
        # but the main container should be there.
        expect(app_page.get_by_text("Expert Panel", exact=False).first).to_be_visible()

    def test_chat_interface_loads(self, app_page: Page):
        """Verify Ask Experts (Chat) tab loads."""
        app_page.get_by_role("tab", name="Ask Experts").click()
        time.sleep(2)
        
        # This tab uses ChatService which was refactored.
        # Check for input or context message
        expect(app_page.get_by_text("context", exact=False).first).to_be_visible()
