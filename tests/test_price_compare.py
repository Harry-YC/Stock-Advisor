"""Compare prices between local and Railway."""
import os
import re
from playwright.sync_api import sync_playwright

def get_price_from_response(url: str, ticker: str = "PLTR") -> dict:
    """Extract price data from app response."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto(url, timeout=60000)
            page.wait_for_load_state("networkidle", timeout=60000)
            page.wait_for_timeout(5000)

            # Find chat input
            chat_input = None
            for selector in ["textarea", "#chat-input", "input[type='text']"]:
                try:
                    elem = page.locator(selector).first
                    if elem.is_visible(timeout=3000):
                        chat_input = elem
                        break
                except:
                    continue

            if not chat_input:
                return {"error": "No chat input found"}

            # Ask for price
            chat_input.fill(f"What is the current price of {ticker}?")
            page.keyboard.press("Enter")

            # Wait for response
            page.wait_for_timeout(30000)

            # Get page content
            content = page.content()

            # Extract prices
            prices = re.findall(r'\$(\d{2,3}\.\d{2})', content)

            # Look for specific patterns
            quote_match = re.search(r'\*\*' + ticker + r'\*\*[^$]*\$(\d+\.\d{2})', content)

            return {
                "url": url,
                "all_prices": prices[:10],
                "quote_price": quote_match.group(1) if quote_match else None,
            }
        finally:
            browser.close()


if __name__ == "__main__":
    local_url = "http://localhost:8000"
    railway_url = "https://stock-advisor-local7.up.railway.app"

    print("=== Comparing PLTR prices ===\n")

    print(f"Testing LOCAL ({local_url})...")
    local_result = get_price_from_response(local_url)
    print(f"  Quote price: ${local_result.get('quote_price', 'N/A')}")
    print(f"  All prices found: {local_result.get('all_prices', [])}")

    print(f"\nTesting RAILWAY ({railway_url})...")
    railway_result = get_price_from_response(railway_url)
    print(f"  Quote price: ${railway_result.get('quote_price', 'N/A')}")
    print(f"  All prices found: {railway_result.get('all_prices', [])}")

    # Compare
    print("\n=== Comparison ===")
    if local_result.get('quote_price') == railway_result.get('quote_price'):
        print(f"✅ Prices MATCH: ${local_result.get('quote_price')}")
    else:
        print(f"❌ Prices DIFFER:")
        print(f"   Local: ${local_result.get('quote_price')}")
        print(f"   Railway: ${railway_result.get('quote_price')}")
