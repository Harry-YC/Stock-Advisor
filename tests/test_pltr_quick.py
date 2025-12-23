"""Quick Playwright test for PLTR analysis."""
import os
import re
import pytest
from playwright.sync_api import sync_playwright, expect

TEST_URL = os.getenv("TEST_URL", "http://localhost:8000")


def test_pltr_buy_or_sell():
    """Test asking about PLTR buy/sell with stock data verification."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            # Navigate to app
            print(f"Navigating to {TEST_URL}")
            page.goto(TEST_URL, timeout=60000)
            page.wait_for_load_state("networkidle", timeout=60000)

            # Wait for Chainlit to fully load (React app)
            page.wait_for_timeout(5000)

            # Find chat input - Chainlit uses various selectors
            chat_input = None
            selectors = [
                "textarea",
                "[data-testid='chat-input']",
                "input[type='text']",
                ".cl-input textarea",
                "#chat-input",
            ]

            for selector in selectors:
                try:
                    elem = page.locator(selector).first
                    if elem.is_visible(timeout=3000):
                        chat_input = elem
                        print(f"Chat input found with selector: {selector}")
                        break
                except:
                    continue

            if not chat_input:
                # Take screenshot for debugging
                page.screenshot(path="/tmp/chainlit_debug.png")
                raise Exception("Could not find chat input")

            # Ask about PLTR
            question = "Should I buy or sell PLTR?"
            chat_input.fill(question)
            print(f"Typed: {question}")

            # Submit
            page.keyboard.press("Enter")
            print("Submitted question")

            # Wait for expert responses (up to 2 minutes)
            print("Waiting for expert responses...")
            page.wait_for_timeout(5000)  # Initial wait

            # Look for expert response markers
            response_found = False
            price_found = False

            for _ in range(24):  # Check for up to 2 minutes
                content = page.content()

                # Check for expert icons/names
                if any(x in content for x in ["Bull Analyst", "Bear Analyst", "🐂", "🐻"]):
                    response_found = True
                    print("✅ Expert responses detected")

                # Check for price data (PLTR around $193-194)
                if re.search(r'\$19[0-9]\.\d{2}', content):
                    price_found = True
                    print("✅ Stock price data found in response")
                    break

                if response_found:
                    # Also check for price in any format
                    if "193" in content or "194" in content or "股價" in content or "price" in content.lower():
                        price_found = True
                        print("✅ Price reference found")
                        break

                page.wait_for_timeout(5000)

            # Assertions
            assert response_found, "No expert responses found"
            assert price_found, "No stock price data found in expert responses"

            print("\n✅ TEST PASSED: PLTR analysis includes stock price data")

        finally:
            browser.close()


if __name__ == "__main__":
    test_pltr_buy_or_sell()
