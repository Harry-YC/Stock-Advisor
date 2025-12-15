"""
Pytest configuration and fixtures for Palliative Surgery GDG E2E tests.

Provides shared fixtures for Playwright browser automation testing
of the Streamlit application features.
"""

import os
import time
import pytest
from typing import Generator
from playwright.sync_api import sync_playwright, Browser, Page, BrowserContext


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_URL = "http://localhost:8501"
DEFAULT_TIMEOUT = 30000  # 30 seconds
SLOW_MO = 100  # Slow down by 100ms for stability


# =============================================================================
# BROWSER FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def browser() -> Generator[Browser, None, None]:
    """
    Launch browser for test session.

    Set HEADLESS=false environment variable for visual debugging.
    """
    headless = os.environ.get("HEADLESS", "true").lower() != "false"

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            slow_mo=SLOW_MO,
            args=["--disable-gpu", "--no-sandbox"]
        )
        yield browser
        browser.close()


@pytest.fixture(scope="function")
def context(browser: Browser) -> Generator[BrowserContext, None, None]:
    """Create new browser context for each test."""
    context = browser.new_context(
        viewport={"width": 1280, "height": 720},
        ignore_https_errors=True,
    )
    yield context
    context.close()


@pytest.fixture(scope="function")
def page(context: BrowserContext) -> Generator[Page, None, None]:
    """Create new page for each test with default timeout."""
    page = context.new_page()
    page.set_default_timeout(DEFAULT_TIMEOUT)
    yield page
    page.close()


# =============================================================================
# APP FIXTURES
# =============================================================================

@pytest.fixture
def app_url() -> str:
    """Get app URL from environment or default."""
    return os.environ.get("TEST_URL", DEFAULT_URL)


@pytest.fixture
def test_question() -> str:
    """The primary test question from research report."""
    return (
        "What is the survival benefit of prophylactic fixation versus "
        "observation in patients with femoral metastases and Mirels score of 9?"
    )


@pytest.fixture
def short_test_question() -> str:
    """A shorter question for faster tests."""
    return "What is the role of palliative surgery in malignant bowel obstruction?"


# =============================================================================
# HELPER FIXTURES
# =============================================================================

@pytest.fixture
def navigate_to_app(page: Page, app_url: str):
    """Navigate to the app and wait for load."""
    def _navigate():
        page.goto(app_url, timeout=30000)
        page.wait_for_load_state("networkidle")
        time.sleep(2)  # Streamlit initialization buffer
        return page
    return _navigate


@pytest.fixture
def create_project(page: Page):
    """Create a test project."""
    def _create_project(name: str = None):
        import random
        project_name = name or f"E2E_Test_{random.randint(1000, 9999)}"

        sidebar = page.locator('[data-testid="stSidebar"]')

        # Find project name input
        project_input = sidebar.locator('input[type="text"]').first
        if project_input.is_visible(timeout=5000):
            project_input.fill(project_name)
            time.sleep(0.5)

            # Click Create button
            create_btn = sidebar.get_by_role("button", name="Create")
            if create_btn.is_visible(timeout=5000):
                create_btn.click()
                page.wait_for_load_state("networkidle")
                time.sleep(2)
                return project_name

        return None

    return _create_project


@pytest.fixture
def ask_question(page: Page):
    """Submit a question in the Ask the GDG interface."""
    def _ask_question(question: str, wait_for_response: bool = True, timeout: int = 60000):
        # Wait for page to be ready
        page.wait_for_load_state("networkidle")
        time.sleep(1)

        # Find the textarea - try multiple strategies
        question_input = None

        # Strategy 1: Look for textarea with relevant placeholder
        textareas = page.locator('textarea')
        for i in range(textareas.count()):
            ta = textareas.nth(i)
            try:
                if ta.is_visible(timeout=2000):
                    placeholder = ta.get_attribute("placeholder") or ""
                    # Look for the main question input
                    if any(kw in placeholder.lower() for kw in ["question", "ask", "clinical", "type"]):
                        question_input = ta
                        break
            except Exception:
                continue

        # Strategy 2: Fallback to first visible textarea
        if not question_input:
            try:
                first_textarea = textareas.first
                if first_textarea.is_visible(timeout=5000):
                    question_input = first_textarea
            except Exception:
                pass

        # Strategy 3: Look for chat input specifically
        if not question_input:
            try:
                chat_input = page.locator('[data-testid="stChatInput"] textarea')
                if chat_input.is_visible(timeout=3000):
                    question_input = chat_input
            except Exception:
                pass

        if not question_input:
            raise Exception("Could not find question input textarea")

        # Fill and submit
        question_input.fill(question)
        question_input.press("Meta+Enter")  # Trigger Streamlit update
        page.wait_for_load_state("networkidle")
        time.sleep(2)

        if wait_for_response:
            # Wait for response to appear
            try:
                # Look for various response indicators
                page.wait_for_selector(
                    'text=Quick Answer, text=Expert, text=Based on',
                    timeout=timeout
                )
            except Exception:
                pass  # Response may appear differently

            time.sleep(3)  # Buffer for full render

        return True

    return _ask_question


# =============================================================================
# STREAMLIT-SPECIFIC HELPERS
# =============================================================================

class StreamlitHelpers:
    """Helper methods for Streamlit-specific interactions."""

    def __init__(self, page: Page):
        self.page = page

    def wait_for_streamlit_rerun(self, delay: float = 2.0):
        """Wait for Streamlit to complete a rerun cycle."""
        self.page.wait_for_load_state("networkidle")
        time.sleep(delay)

    def click_button(self, text: str, exact: bool = False):
        """Click a button by text content."""
        if exact:
            btn = self.page.locator(f'button:text-is("{text}")')
        else:
            btn = self.page.locator('button').filter(has_text=text)

        if btn.count() > 0 and btn.first.is_visible(timeout=5000):
            btn.first.click()
            self.wait_for_streamlit_rerun()
            return True
        return False

    def expand_expander(self, text: str):
        """Expand a Streamlit expander by its label text."""
        expander = self.page.locator('[data-testid="stExpander"]').filter(has_text=text)
        if expander.count() > 0:
            # Click the expander header to expand
            header = expander.first.locator('div[role="button"]').first
            if not self._is_expanded(expander.first):
                header.click()
                self.wait_for_streamlit_rerun(0.5)
            return True
        return False

    def _is_expanded(self, expander_element) -> bool:
        """Check if an expander is currently expanded."""
        try:
            # Check aria-expanded attribute
            expanded = expander_element.get_attribute("aria-expanded")
            return expanded == "true"
        except Exception:
            return False

    def get_sidebar(self):
        """Get the sidebar element."""
        return self.page.locator('[data-testid="stSidebar"]')

    def scroll_to_bottom(self):
        """Scroll to the bottom of the page."""
        self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(0.5)

    def take_screenshot(self, name: str):
        """Take a screenshot with the given name."""
        path = f"/tmp/e2e_{name}.png"
        self.page.screenshot(path=path, full_page=True)
        return path


@pytest.fixture
def st_helpers(page: Page) -> StreamlitHelpers:
    """Provide Streamlit-specific helper methods."""
    return StreamlitHelpers(page)


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "feature_citation: tests for citation highlighting feature"
    )
    config.addinivalue_line(
        "markers", "feature_challenger: tests for red team challenger feature"
    )
    config.addinivalue_line(
        "markers", "feature_quick_answer: tests for quick answer feature"
    )
    config.addinivalue_line(
        "markers", "feature_suggestions: tests for smart suggestions feature"
    )
    config.addinivalue_line(
        "markers", "feature_mark_pen: tests for mark pen feature"
    )
