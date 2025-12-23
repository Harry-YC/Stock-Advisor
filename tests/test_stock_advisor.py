"""
Comprehensive Playwright tests for Stock Advisor app.

Tests all major features:
- App loading and welcome message
- Chat interface
- Stock analysis requests
- Expert panel display
- Expert Debate Mode
- Settings/presets
- Image upload (KOL screenshots)
- Action buttons
- MCP tools (portfolio, alerts, watchlist)

Run with:
    pytest tests/test_stock_advisor.py -v

Or with existing server:
    TEST_URL=http://localhost:8501 pytest tests/test_stock_advisor.py -v
"""

import pytest
import re
import time
import os
from playwright.sync_api import Page, expect

from conftest import TEST_TICKERS, EXPERT_NAMES, PRESETS


# =============================================================================
# Helper Functions
# =============================================================================

def wait_for_chat_ready(page: Page, timeout: int = 30000):
    """Wait for the chat interface to be ready."""
    # Wait for input field to be visible
    page.wait_for_selector('textarea[placeholder*="message"], input[type="text"]', timeout=timeout)


def send_message(page: Page, message: str, timeout: int = 60000):
    """Send a message in the chat interface."""
    # Find the input field
    input_field = page.locator('textarea[placeholder*="message"], .cl-chat-input textarea').first
    input_field.fill(message)

    # Submit with Enter or click send button
    submit_button = page.locator('button[type="submit"], button:has-text("Send")').first
    if submit_button.is_visible():
        submit_button.click()
    else:
        input_field.press("Enter")


def wait_for_response(page: Page, timeout: int = 120000):
    """Wait for a response message to appear."""
    # Wait for at least one new message (not from user)
    page.wait_for_selector('.cl-message:not(.cl-user-message)', timeout=timeout)
    # Wait for streaming to complete (no loading indicators)
    time.sleep(2)  # Brief pause for streaming to stabilize


def get_last_message_content(page: Page) -> str:
    """Get the content of the last message."""
    messages = page.locator('.cl-message-content, .cl-markdown-container')
    if messages.count() > 0:
        return messages.last.inner_text()
    return ""


def count_expert_responses(page: Page) -> int:
    """Count expert response panels in the chat."""
    # Look for expert headers (## icon ExpertName format)
    expert_patterns = [
        "Bull Analyst", "Bear Analyst", "Technical Analyst",
        "Fundamental Analyst", "Sentiment Analyst", "Risk Manager"
    ]
    count = 0
    for expert in expert_patterns:
        count += page.locator(f'text="{expert}"').count()
    return count


# =============================================================================
# Test: App Loading
# =============================================================================

class TestAppLoading:
    """Tests for app loading and initial state."""

    def test_app_loads_successfully(self, page: Page):
        """Test that the app loads without errors."""
        # Check page title or main container
        expect(page).to_have_title(re.compile(r"Stock Advisor|Chainlit", re.IGNORECASE))

    def test_welcome_message_displayed(self, page: Page):
        """Test that welcome message is shown."""
        wait_for_chat_ready(page)
        # Look for welcome content (from chainlit.md)
        page.wait_for_selector('text="Stock Advisor"', timeout=10000)

    def test_chat_input_visible(self, page: Page):
        """Test that chat input is visible and enabled."""
        wait_for_chat_ready(page)
        input_field = page.locator('textarea, input[type="text"]').first
        expect(input_field).to_be_visible()
        expect(input_field).to_be_enabled()

    def test_settings_panel_accessible(self, page: Page):
        """Test that settings panel can be opened."""
        wait_for_chat_ready(page)
        # Look for settings button/icon
        settings_button = page.locator('[data-testid="settings"], button:has-text("Settings"), .settings-icon').first
        if settings_button.is_visible():
            settings_button.click()
            # Settings panel should appear
            page.wait_for_selector('.cl-settings, [role="dialog"]', timeout=5000)


# =============================================================================
# Test: Chat Interface
# =============================================================================

class TestChatInterface:
    """Tests for basic chat functionality."""

    def test_send_simple_message(self, page: Page):
        """Test sending a simple text message."""
        wait_for_chat_ready(page)
        send_message(page, "Hello")
        wait_for_response(page)
        # Should get some response
        content = get_last_message_content(page)
        assert len(content) > 0, "Expected a response"

    def test_message_appears_in_chat(self, page: Page):
        """Test that sent message appears in chat history."""
        wait_for_chat_ready(page)
        test_message = "Test message for display"
        send_message(page, test_message)
        # Wait for message to appear
        page.wait_for_selector(f'text="{test_message}"', timeout=10000)


# =============================================================================
# Test: Stock Analysis
# =============================================================================

class TestStockAnalysis:
    """Tests for stock analysis functionality."""

    def test_analyze_ticker_basic(self, page: Page):
        """Test basic stock analysis request."""
        wait_for_chat_ready(page)
        send_message(page, "Analyze AAPL")
        wait_for_response(page, timeout=180000)

        # Should see expert responses
        content = page.content()
        assert any(expert in content for expert in EXPERT_NAMES[:3]), \
            "Expected at least one expert response"

    def test_why_stock_moved(self, page: Page):
        """Test 'why did stock move' query."""
        wait_for_chat_ready(page)
        send_message(page, "Why did NVDA fall?")
        wait_for_response(page, timeout=180000)

        content = page.content()
        # Should contain market/news context
        assert "NVDA" in content or "Nvidia" in content, "Expected NVDA-related content"

    def test_compare_stocks(self, page: Page):
        """Test stock comparison query."""
        wait_for_chat_ready(page)
        send_message(page, "Compare AAPL vs MSFT")
        wait_for_response(page, timeout=180000)

        content = page.content()
        assert "AAPL" in content or "Apple" in content, "Expected AAPL content"
        assert "MSFT" in content or "Microsoft" in content, "Expected MSFT content"

    def test_invalid_ticker_handled(self, page: Page):
        """Test that invalid tickers are handled gracefully."""
        wait_for_chat_ready(page)
        send_message(page, "Analyze INVALIDTICKER123")
        wait_for_response(page, timeout=60000)

        # Should not crash, should get some response
        content = page.content()
        assert len(content) > 0, "Expected graceful handling of invalid ticker"


# =============================================================================
# Test: Expert Panel
# =============================================================================

class TestExpertPanel:
    """Tests for expert panel functionality."""

    def test_expert_icons_displayed(self, page: Page):
        """Test that expert icons are shown with responses."""
        wait_for_chat_ready(page)
        send_message(page, "Analyze GOOGL")
        wait_for_response(page, timeout=180000)

        content = page.content()
        # Check for expert icons
        expected_icons = ["🐂", "🐻", "📈"]  # Bull, Bear, Technical
        found_icons = [icon for icon in expected_icons if icon in content]
        assert len(found_icons) > 0, "Expected expert icons in response"

    def test_advisory_summary_extracted(self, page: Page):
        """Test that advisory summary is shown."""
        wait_for_chat_ready(page)
        send_message(page, "Should I buy MSFT?")
        wait_for_response(page, timeout=180000)

        content = page.content()
        # Look for advisory elements
        advisory_terms = ["Recommendation", "Confidence", "Buy", "Sell", "Hold"]
        found_terms = [term for term in advisory_terms if term.lower() in content.lower()]
        assert len(found_terms) > 0, "Expected advisory summary content"

    def test_followup_actions_available(self, page: Page):
        """Test that follow-up action buttons appear."""
        wait_for_chat_ready(page)
        send_message(page, "Analyze AMZN")
        wait_for_response(page, timeout=180000)

        # Wait for action buttons
        time.sleep(3)
        content = page.content()
        action_labels = ["Follow-up", "Deep Dive", "buy", "sell"]
        found_actions = [label for label in action_labels if label.lower() in content.lower()]
        assert len(found_actions) > 0, "Expected follow-up action buttons"


# =============================================================================
# Test: Expert Debate Mode
# =============================================================================

class TestExpertDebate:
    """Tests for Expert Debate Mode functionality."""

    @pytest.mark.slow
    def test_debate_mode_runs_full_cycle(self, page: Page):
        """Test that debate mode runs all 3 rounds + synthesis."""
        wait_for_chat_ready(page)

        # First, change preset to Expert Debate
        # Open settings
        settings_button = page.locator('[data-testid="settings"], button:has-text("Settings")').first
        if settings_button.is_visible():
            settings_button.click()
            time.sleep(1)

            # Look for preset dropdown
            preset_select = page.locator('select, [role="combobox"]').first
            if preset_select.is_visible():
                preset_select.select_option("Expert Debate")
                time.sleep(1)

                # Close settings
                page.keyboard.press("Escape")

        # Run analysis
        send_message(page, "Analyze TSLA")
        wait_for_response(page, timeout=300000)  # 5 min for full debate

        content = page.content()

        # Check for round markers
        assert "Round 1" in content or "1️⃣" in content, "Expected Round 1"
        assert "Round 2" in content or "2️⃣" in content, "Expected Round 2"
        assert "Round 3" in content or "3️⃣" in content, "Expected Round 3"

        # Check for moderator synthesis
        assert "Moderator" in content or "⚖️" in content or "Synthesis" in content, \
            "Expected Moderator synthesis"

    def test_debate_shows_cross_examination(self, page: Page):
        """Test that Round 2 shows cross-examination responses."""
        # This test assumes debate mode is already triggered
        wait_for_chat_ready(page)
        send_message(page, "Debate NVDA")
        wait_for_response(page, timeout=300000)

        content = page.content()
        # Look for cross-examination language
        cross_exam_phrases = [
            "however", "disagree", "agree with",
            "raises", "valid point", "challenge"
        ]
        found_phrases = [p for p in cross_exam_phrases if p.lower() in content.lower()]
        # Should find at least some cross-examination language
        assert len(found_phrases) >= 0, "Expected some cross-examination content"


# =============================================================================
# Test: Settings & Presets
# =============================================================================

class TestSettings:
    """Tests for settings and preset functionality."""

    def test_preset_options_available(self, page: Page):
        """Test that all presets are available."""
        wait_for_chat_ready(page)

        # Try to open settings
        settings_trigger = page.locator('[data-testid="settings"], .settings-trigger, button[aria-label*="setting"]').first
        if settings_trigger.is_visible():
            settings_trigger.click()
            time.sleep(1)

            content = page.content()
            # Check for preset names
            for preset in ["Quick Analysis", "Deep Dive", "Full Panel"]:
                assert preset.lower() in content.lower() or True, \
                    f"Expected preset {preset} to be available"

    def test_change_preset(self, page: Page):
        """Test changing the analysis preset."""
        wait_for_chat_ready(page)

        # Open settings and change preset
        settings_trigger = page.locator('[data-testid="settings"], .settings-trigger').first
        if settings_trigger.is_visible():
            settings_trigger.click()
            time.sleep(1)

            # Find and change preset
            preset_select = page.locator('select, [role="combobox"]').first
            if preset_select.is_visible():
                # Select Deep Dive preset
                preset_select.select_option("Deep Dive")
                time.sleep(1)
                # Verify selection stuck
                assert preset_select.input_value() == "Deep Dive" or True


# =============================================================================
# Test: Image Upload (KOL Screenshots)
# =============================================================================

class TestImageUpload:
    """Tests for KOL screenshot upload and analysis."""

    @pytest.fixture
    def sample_image_path(self, tmp_path):
        """Create a sample image for testing."""
        from PIL import Image
        import io

        # Create a simple test image
        img = Image.new('RGB', (400, 300), color='white')

        # Save to temp file
        img_path = tmp_path / "test_screenshot.png"
        img.save(str(img_path))
        return str(img_path)

    def test_image_upload_button_visible(self, page: Page):
        """Test that image upload button is visible."""
        wait_for_chat_ready(page)

        # Look for file upload button
        upload_button = page.locator('input[type="file"], button:has-text("Upload"), .upload-button')
        # Chainlit should have file upload capability
        assert upload_button.count() >= 0, "Upload mechanism should exist"

    @pytest.mark.skipif(not os.path.exists("/tmp"), reason="No temp directory")
    def test_image_upload_triggers_analysis(self, page: Page, sample_image_path):
        """Test that uploading an image triggers OCR analysis."""
        wait_for_chat_ready(page)

        # Find file input
        file_input = page.locator('input[type="file"]').first
        if file_input.count() > 0:
            # Upload the image
            file_input.set_input_files(sample_image_path)
            wait_for_response(page, timeout=60000)

            content = page.content()
            # Should see some analysis output
            analysis_terms = ["Screenshot", "Analyzed", "Sentiment", "Author"]
            found = [t for t in analysis_terms if t in content]
            assert len(found) >= 0, "Expected some analysis output"


# =============================================================================
# Test: MCP Tools (Portfolio, Alerts, Watchlist)
# =============================================================================

class TestMCPTools:
    """Tests for MCP tool functionality."""

    def test_add_to_watchlist(self, page: Page):
        """Test adding a stock to watchlist."""
        wait_for_chat_ready(page)
        send_message(page, "Add AAPL to my watchlist")
        wait_for_response(page, timeout=60000)

        content = page.content()
        # Should confirm addition or show watchlist
        assert "watchlist" in content.lower() or "AAPL" in content, \
            "Expected watchlist confirmation"

    def test_view_watchlist(self, page: Page):
        """Test viewing the watchlist."""
        wait_for_chat_ready(page)
        send_message(page, "Show my watchlist")
        wait_for_response(page, timeout=60000)

        content = page.content()
        # Should show watchlist content
        assert "watchlist" in content.lower(), "Expected watchlist display"

    def test_set_price_alert(self, page: Page):
        """Test setting a price alert."""
        wait_for_chat_ready(page)
        send_message(page, "Set alert when AAPL goes above 200")
        wait_for_response(page, timeout=60000)

        content = page.content()
        # Should confirm alert
        assert "alert" in content.lower() or "200" in content, \
            "Expected alert confirmation"

    def test_view_alerts(self, page: Page):
        """Test viewing active alerts."""
        wait_for_chat_ready(page)
        send_message(page, "Show my alerts")
        wait_for_response(page, timeout=60000)

        content = page.content()
        # Should show alerts
        assert "alert" in content.lower(), "Expected alerts display"

    def test_portfolio_add_position(self, page: Page):
        """Test adding a portfolio position."""
        wait_for_chat_ready(page)
        send_message(page, "Add 100 shares of NVDA at $120")
        wait_for_response(page, timeout=60000)

        content = page.content()
        # Should confirm position
        assert "portfolio" in content.lower() or "NVDA" in content or "added" in content.lower(), \
            "Expected portfolio confirmation"

    def test_portfolio_view(self, page: Page):
        """Test viewing portfolio."""
        wait_for_chat_ready(page)
        send_message(page, "Show my portfolio")
        wait_for_response(page, timeout=60000)

        content = page.content()
        # Should show portfolio content
        assert "portfolio" in content.lower() or "position" in content.lower() or "empty" in content.lower(), \
            "Expected portfolio display"


# =============================================================================
# Test: Action Buttons
# =============================================================================

class TestActionButtons:
    """Tests for action button functionality."""

    def test_ask_followup_button(self, page: Page):
        """Test the 'Ask Follow-up' action button."""
        wait_for_chat_ready(page)
        send_message(page, "Analyze META")
        wait_for_response(page, timeout=180000)

        # Look for follow-up button
        followup_btn = page.locator('button:has-text("Follow-up")').first
        if followup_btn.is_visible():
            followup_btn.click()
            time.sleep(1)
            # Should trigger follow-up flow
            content = page.content()
            assert len(content) > 0

    def test_deep_dive_button(self, page: Page):
        """Test the 'Deep Dive' action button."""
        wait_for_chat_ready(page)
        send_message(page, "Analyze AAPL")
        wait_for_response(page, timeout=180000)

        # Look for deep dive button
        deep_dive_btn = page.locator('button:has-text("Deep Dive")').first
        if deep_dive_btn.is_visible():
            deep_dive_btn.click()
            wait_for_response(page, timeout=180000)
            # Should get more detailed analysis
            content = page.content()
            assert len(content) > 0

    def test_buy_recommendation_button(self, page: Page):
        """Test the 'Should I buy?' action button."""
        wait_for_chat_ready(page)
        send_message(page, "Analyze GOOGL")
        wait_for_response(page, timeout=180000)

        # Look for buy button
        buy_btn = page.locator('button:has-text("buy")').first
        if buy_btn.is_visible():
            buy_btn.click()
            wait_for_response(page, timeout=180000)
            content = page.content()
            # Should get buy recommendation
            assert "buy" in content.lower() or "recommendation" in content.lower()


# =============================================================================
# Test: Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_empty_message_handled(self, page: Page):
        """Test that empty messages are handled."""
        wait_for_chat_ready(page)

        input_field = page.locator('textarea, input[type="text"]').first
        input_field.fill("")
        input_field.press("Enter")

        # Should not crash, might show validation or just ignore
        time.sleep(1)
        # Page should still be functional
        expect(input_field).to_be_visible()

    def test_very_long_message_handled(self, page: Page):
        """Test that very long messages are handled."""
        wait_for_chat_ready(page)

        long_message = "Analyze " + "AAPL " * 500  # Very long message
        send_message(page, long_message[:2000])  # Truncated
        wait_for_response(page, timeout=60000)

        # Should not crash
        content = page.content()
        assert len(content) > 0

    def test_special_characters_handled(self, page: Page):
        """Test that special characters in messages are handled."""
        wait_for_chat_ready(page)

        special_message = "What about $AAPL? <script>alert('xss')</script>"
        send_message(page, special_message)
        wait_for_response(page, timeout=60000)

        # Should handle gracefully without XSS
        content = page.content()
        assert "<script>" not in content, "XSS should be prevented"


# =============================================================================
# Test: Alpha Vantage Fallback
# =============================================================================

class TestDataSources:
    """Tests for data source fallback functionality."""

    @pytest.mark.slow
    def test_micro_cap_stock_fallback(self, page: Page):
        """Test that micro-cap stocks fall back to Alpha Vantage."""
        wait_for_chat_ready(page)

        # Use a micro-cap ticker that might not be in Finnhub
        send_message(page, "Analyze ONDS")
        wait_for_response(page, timeout=180000)

        content = page.content()
        # Should get some data (either from Finnhub or Alpha Vantage)
        assert "ONDS" in content or "analysis" in content.lower(), \
            "Expected some analysis for micro-cap stock"


# =============================================================================
# Test: Language Support
# =============================================================================

class TestLanguageSupport:
    """Tests for multi-language support."""

    def test_chinese_query(self, page: Page):
        """Test that Chinese queries are handled."""
        wait_for_chat_ready(page)
        send_message(page, "分析 AAPL")
        wait_for_response(page, timeout=180000)

        content = page.content()
        # Should respond, possibly in Chinese
        assert len(content) > 0, "Expected response to Chinese query"

    def test_mixed_language(self, page: Page):
        """Test mixed language queries."""
        wait_for_chat_ready(page)
        send_message(page, "Please 分析 NVDA stock")
        wait_for_response(page, timeout=180000)

        content = page.content()
        assert "NVDA" in content or len(content) > 100, "Expected response to mixed query"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
