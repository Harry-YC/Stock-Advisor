"""
Pytest configuration and fixtures for Travel Planner tests.

Provides Playwright fixtures and common test utilities.
"""

import pytest
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Generator, AsyncGenerator

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Async Event Loop Fixture
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Playwright Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context for Chainlit testing."""
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
        "ignore_https_errors": True,
    }


@pytest.fixture
def chainlit_base_url() -> str:
    """Base URL for Chainlit app - uses env var or default."""
    return os.environ.get("CHAINLIT_URL", "http://localhost:8000")


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Create a minimal valid PDF for testing."""
    # Minimal valid PDF structure
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer
<< /Size 4 /Root 1 0 R >>
startxref
196
%%EOF"""


@pytest.fixture
def large_file_content() -> bytes:
    """Create content larger than 10MB limit for testing."""
    # 11MB of data
    return b"x" * (11 * 1024 * 1024)


@pytest.fixture
def sample_trip_config() -> dict:
    """Sample trip configuration for testing."""
    return {
        "destination": "Barcelona, Spain",
        "start_date": "2025-03-15",
        "end_date": "2025-03-22",
        "travelers": "Group (4+)",
        "budget": "moderate",
        "interests": ["food", "culture", "architecture"],
        "special_requirements": ""
    }


@pytest.fixture
def sample_expert_responses() -> dict:
    """Sample expert responses for testing."""
    return {
        "Food & Dining Expert": {
            "content": """Here are my top restaurant recommendations for Barcelona:

**La Boqueria Market** - Famous food market with fresh tapas
**Bar Cañete** - Excellent seafood and traditional Catalan cuisine
**Tickets** - Creative tapas by the Adrià brothers
**Cal Pep** - Counter-style dining with amazing seafood

I recommend making reservations at least 2 weeks in advance for **Tickets** and **Cal Pep**.
"""
        },
        "Budget Advisor": {
            "content": """Budget breakdown for your Barcelona trip:

- Accommodation: €120/night x 7 nights = €840
- Food: €60/day x 7 days = €420
- Activities: €200
- Transportation: €100

Total estimated: €1,560 for 4 people
"""
        },
        "Accommodation Specialist": {
            "content": """For your group of 4, I recommend:

**Hotel Arts Barcelona** - Beachfront luxury
**Mandarin Oriental** - Gothic Quarter location
**Casa Camper** - Boutique in El Raval

The Gothic Quarter offers the best walking access to attractions.
"""
        }
    }


# =============================================================================
# Mock Services
# =============================================================================

@pytest.fixture
def mock_places_response() -> dict:
    """Mock Google Places API response."""
    return {
        "places": [
            {
                "id": "ChIJ_test123",
                "displayName": {"text": "La Boqueria Market"},
                "rating": 4.5,
                "userRatingCount": 15234,
                "priceLevel": "PRICE_LEVEL_MODERATE",
                "formattedAddress": "La Rambla, 91, Barcelona",
                "types": ["restaurant", "food"],
                "businessStatus": "OPERATIONAL"
            }
        ]
    }


# =============================================================================
# Chainlit Test Helpers
# =============================================================================

class ChainlitTestHelper:
    """Helper class for Chainlit UI testing."""

    def __init__(self, page):
        self.page = page

    async def wait_for_chat_ready(self, timeout: int = 30000):
        """Wait for Chainlit chat interface to be ready."""
        await self.page.wait_for_selector('[data-testid="chat-input"]', timeout=timeout)

    async def send_message(self, message: str):
        """Send a message in the chat."""
        input_field = self.page.locator('[data-testid="chat-input"]')
        await input_field.fill(message)
        await input_field.press("Enter")

    async def wait_for_response(self, timeout: int = 60000):
        """Wait for a response message to appear."""
        await self.page.wait_for_selector('.message-author', timeout=timeout)

    async def get_last_message_content(self) -> str:
        """Get the content of the last message."""
        messages = self.page.locator('.message-content')
        count = await messages.count()
        if count > 0:
            return await messages.nth(count - 1).text_content()
        return ""

    async def upload_file(self, file_path: str):
        """Upload a file via the file input."""
        file_input = self.page.locator('input[type="file"]')
        await file_input.set_input_files(file_path)


@pytest.fixture
def chainlit_helper(page) -> ChainlitTestHelper:
    """Provide a Chainlit test helper."""
    return ChainlitTestHelper(page)
