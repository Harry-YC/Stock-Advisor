"""
Pytest configuration and fixtures for Literature Review Platform tests.
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Playwright fixtures
pytest_plugins = ["pytest_playwright"]


@pytest.fixture(scope="session")
def browser_type_launch_args():
    """Configure browser launch arguments."""
    return {
        "headless": True,
        "args": ["--disable-gpu", "--no-sandbox"]
    }


@pytest.fixture(scope="session")
def browser_context_args():
    """Configure browser context arguments."""
    return {
        "viewport": {"width": 1280, "height": 720},
        "ignore_https_errors": True
    }


@pytest.fixture(scope="function")
def page_timeout(page):
    """Set default timeout for page operations."""
    page.set_default_timeout(30000)
    return page


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, no external dependencies)"
    )
