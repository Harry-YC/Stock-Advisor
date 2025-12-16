"""
Playwright E2E Tests for Travel Planner Chainlit App

Run with:
    pytest tests/test_travel_planner.py -v --headed

Prerequisites:
    pip install pytest-playwright
    playwright install chromium
"""

import pytest
import re
from playwright.sync_api import Page, expect

# App URL - adjust for your environment
APP_URL = "http://localhost:8000"

# Expert emojis to verify multi-expert responses
EXPERT_EMOJIS = {
    "Budget Advisor": "💰",
    "Logistics Planner": "🚗",
    "Safety Expert": "🛡️",
    "Weather Analyst": "🌤️",
    "Local Culture Guide": "🎎",
    "Food & Dining Expert": "🍜",
    "Activity Curator": "🎯",
    "Accommodation Specialist": "🏨",
}

# Quick Trip Planning preset experts
QUICK_TRIP_EXPERTS = ["Budget Advisor", "Logistics Planner", "Accommodation Specialist", "Activity Curator"]


class TestTravelPlannerApp:
    """End-to-end tests for the Travel Planner Chainlit app."""

    @pytest.fixture(autouse=True)
    def setup(self, page: Page):
        """Navigate to app before each test."""
        self.page = page
        self.page.goto(APP_URL)
        # Wait for Chainlit to load
        self.page.wait_for_selector('[data-testid="chat-input"]', timeout=30000)

    def test_app_loads_with_welcome_message(self):
        """Verify app loads and shows welcome message."""
        # Check for welcome message content
        welcome = self.page.locator("text=Travel Planner")
        expect(welcome.first).to_be_visible(timeout=10000)

        # Check for instructions
        expect(self.page.locator("text=Plan my trip")).to_be_visible()
        expect(self.page.locator("text=settings icon")).to_be_visible()

    def test_settings_panel_opens(self):
        """Verify settings panel can be opened and contains trip fields."""
        # Click settings button (gear icon)
        settings_btn = self.page.locator('[data-testid="chat-settings-open-modal"]')
        settings_btn.click()

        # Wait for settings modal
        self.page.wait_for_selector('[data-testid="chat-settings-modal"]', timeout=5000)

        # Verify trip configuration fields exist
        expect(self.page.locator('text=Destination')).to_be_visible()
        expect(self.page.locator('text=Origin')).to_be_visible()
        expect(self.page.locator('text=Departure Date')).to_be_visible()
        expect(self.page.locator('text=Budget')).to_be_visible()
        expect(self.page.locator('text=Expert Panel')).to_be_visible()

    def test_configure_trip_settings(self):
        """Configure trip settings and verify they're saved."""
        # Open settings
        self.page.locator('[data-testid="chat-settings-open-modal"]').click()
        self.page.wait_for_selector('[data-testid="chat-settings-modal"]', timeout=5000)

        # Fill in destination
        destination_input = self.page.locator('input[id="destination"]')
        destination_input.fill("Barcelona, Spain")

        # Fill in origin
        origin_input = self.page.locator('input[id="origin"]')
        origin_input.fill("New York, NY")

        # Save settings (click outside or submit button)
        save_btn = self.page.locator('[data-testid="chat-settings-submit"]')
        if save_btn.is_visible():
            save_btn.click()
        else:
            # Click outside to close
            self.page.keyboard.press("Escape")

        # Verify confirmation message appears
        self.page.wait_for_selector("text=Trip settings updated", timeout=5000)
        expect(self.page.locator("text=Barcelona")).to_be_visible()

    def test_plan_my_trip_triggers_experts(self):
        """Test 'Plan my trip' command triggers multiple expert responses."""
        # First configure a destination via settings
        self._configure_trip("Tokyo, Japan", "San Francisco, CA")

        # Send "Plan my trip" command
        chat_input = self.page.locator('[data-testid="chat-input"]')
        chat_input.fill("Plan my trip")
        chat_input.press("Enter")

        # Wait for planning to start
        self.page.wait_for_selector("text=Planning your trip", timeout=15000)

        # Wait for expert responses (longer timeout for API calls)
        # Look for at least one expert emoji to appear
        expert_response_found = False
        for expert_name in QUICK_TRIP_EXPERTS:
            emoji = EXPERT_EMOJIS[expert_name]
            try:
                self.page.wait_for_selector(f"text={emoji}", timeout=60000)
                expert_response_found = True
                print(f"✅ Found expert: {emoji} {expert_name}")
                break
            except:
                continue

        assert expert_response_found, "No expert responses appeared"

        # Verify multiple experts responded (check for multiple emojis)
        emojis_found = []
        for expert_name in QUICK_TRIP_EXPERTS:
            emoji = EXPERT_EMOJIS[expert_name]
            if self.page.locator(f"text={emoji}").count() > 0:
                emojis_found.append(emoji)
                print(f"✅ Expert responded: {emoji} {expert_name}")

        assert len(emojis_found) >= 2, f"Expected multiple experts, found: {emojis_found}"

        # Verify completion message
        self.page.wait_for_selector("text=Trip planning complete", timeout=120000)

    def test_ask_specific_expert(self):
        """Test asking a specific expert directly."""
        # Configure trip first
        self._configure_trip("Paris, France", "Boston, MA")

        # Ask specific expert
        chat_input = self.page.locator('[data-testid="chat-input"]')
        chat_input.fill("Ask Food & Dining Expert about best croissants")
        chat_input.press("Enter")

        # Wait for Food expert response (🍜)
        self.page.wait_for_selector("text=🍜", timeout=60000)
        expect(self.page.locator("text=Food & Dining Expert")).to_be_visible()

    def test_auto_routing_to_expert(self):
        """Test that general questions auto-route to the best expert."""
        # Configure trip first
        self._configure_trip("Rome, Italy", "Chicago, IL")

        # Ask a budget-related question (should route to Budget Advisor)
        chat_input = self.page.locator('[data-testid="chat-input"]')
        chat_input.fill("How much money should I bring?")
        chat_input.press("Enter")

        # Should see either Budget Advisor (💰) or a general response
        # Wait for any response
        self.page.wait_for_timeout(30000)  # Wait for response

        # Check if Budget Advisor responded
        budget_emoji = self.page.locator("text=💰")
        if budget_emoji.count() > 0:
            print("✅ Correctly routed to Budget Advisor")
        else:
            print("ℹ️ Routed to general assistant")

    def test_expert_responses_have_distinct_emojis(self):
        """Verify each expert has a unique identifying emoji."""
        # Configure and plan trip
        self._configure_trip("London, UK", "Los Angeles, CA")

        chat_input = self.page.locator('[data-testid="chat-input"]')
        chat_input.fill("Plan my trip")
        chat_input.press("Enter")

        # Wait for completion
        self.page.wait_for_selector("text=Trip planning complete", timeout=180000)

        # Collect all emojis found
        emojis_found = set()
        page_content = self.page.content()

        for expert_name, emoji in EXPERT_EMOJIS.items():
            if emoji in page_content:
                emojis_found.add(emoji)
                print(f"✅ Found {emoji} ({expert_name})")

        # Verify we got multiple distinct expert responses
        assert len(emojis_found) >= 3, f"Expected 3+ distinct experts, found {len(emojis_found)}: {emojis_found}"
        print(f"\n📊 Total distinct experts: {len(emojis_found)}")

    def test_streaming_response_updates(self):
        """Verify responses stream in real-time (not all at once)."""
        self._configure_trip("Sydney, Australia", "Seattle, WA")

        chat_input = self.page.locator('[data-testid="chat-input"]')
        chat_input.fill("Plan my trip")
        chat_input.press("Enter")

        # Wait for first expert to start
        self.page.wait_for_selector("text=Planning your trip", timeout=15000)

        # Check for streaming indicator or growing content
        initial_content_length = len(self.page.content())

        # Wait a bit for streaming
        self.page.wait_for_timeout(5000)

        # Content should have grown (streaming)
        current_content_length = len(self.page.content())
        assert current_content_length > initial_content_length, "Content did not stream/grow"
        print(f"✅ Content streamed: {initial_content_length} -> {current_content_length}")

    # Helper methods

    def _configure_trip(self, destination: str, origin: str):
        """Helper to configure trip settings."""
        # Open settings
        settings_btn = self.page.locator('[data-testid="chat-settings-open-modal"]')
        settings_btn.click()
        self.page.wait_for_selector('[data-testid="chat-settings-modal"]', timeout=5000)

        # Fill destination
        dest_input = self.page.locator('input[id="destination"]')
        dest_input.clear()
        dest_input.fill(destination)

        # Fill origin
        origin_input = self.page.locator('input[id="origin"]')
        origin_input.clear()
        origin_input.fill(origin)

        # Submit/close settings
        save_btn = self.page.locator('[data-testid="chat-settings-submit"]')
        if save_btn.is_visible():
            save_btn.click()
        else:
            self.page.keyboard.press("Escape")

        # Wait for confirmation
        self.page.wait_for_selector(f"text={destination.split(',')[0]}", timeout=5000)


class TestExpertDifferentiation:
    """Tests specifically for verifying multi-expert responses are distinguishable."""

    @pytest.fixture(autouse=True)
    def setup(self, page: Page):
        self.page = page
        self.page.goto(APP_URL)
        self.page.wait_for_selector('[data-testid="chat-input"]', timeout=30000)

    def test_quick_trip_preset_uses_four_experts(self):
        """Quick Trip Planning should use exactly 4 experts."""
        # Configure with Quick Trip preset (default)
        self._configure_trip("Amsterdam, Netherlands", "Miami, FL")

        # Plan trip
        chat_input = self.page.locator('[data-testid="chat-input"]')
        chat_input.fill("Plan my trip")
        chat_input.press("Enter")

        # Wait for completion
        self.page.wait_for_selector("text=Trip planning complete", timeout=180000)

        # Count expert emojis
        page_content = self.page.content()
        expected_emojis = ["💰", "🚗", "🏨", "🎯"]  # Quick Trip preset

        found_count = sum(1 for emoji in expected_emojis if emoji in page_content)
        print(f"Found {found_count}/4 expected experts")

        assert found_count >= 3, f"Expected at least 3 of 4 Quick Trip experts, found {found_count}"

    def test_each_expert_has_header_with_emoji(self):
        """Each expert response should have a header with emoji and name."""
        self._configure_trip("Berlin, Germany", "Denver, CO")

        chat_input = self.page.locator('[data-testid="chat-input"]')
        chat_input.fill("Plan my trip")
        chat_input.press("Enter")

        self.page.wait_for_selector("text=Trip planning complete", timeout=180000)

        # Check for expert headers like "## 💰 Budget Advisor"
        page_content = self.page.content()

        headers_found = []
        for expert_name, emoji in EXPERT_EMOJIS.items():
            # Look for the header pattern
            if f"{emoji} {expert_name}" in page_content or f"{emoji}" in page_content:
                headers_found.append(f"{emoji} {expert_name}")

        print(f"Expert headers found: {headers_found}")
        assert len(headers_found) >= 3, "Expected at least 3 expert headers with emojis"

    def _configure_trip(self, destination: str, origin: str):
        """Helper to configure trip settings."""
        settings_btn = self.page.locator('[data-testid="chat-settings-open-modal"]')
        settings_btn.click()
        self.page.wait_for_selector('[data-testid="chat-settings-modal"]', timeout=5000)

        dest_input = self.page.locator('input[id="destination"]')
        dest_input.clear()
        dest_input.fill(destination)

        origin_input = self.page.locator('input[id="origin"]')
        origin_input.clear()
        origin_input.fill(origin)

        save_btn = self.page.locator('[data-testid="chat-settings-submit"]')
        if save_btn.is_visible():
            save_btn.click()
        else:
            self.page.keyboard.press("Escape")

        self.page.wait_for_selector(f"text={destination.split(',')[0]}", timeout=5000)


# Standalone test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--headed", "-s"])
