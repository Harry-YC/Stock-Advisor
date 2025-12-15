"""
Debug script to see what's on the Streamlit page.
"""

import time
from playwright.sync_api import sync_playwright


def debug_page():
    """Debug what's visible on the page."""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=300)
        page = browser.new_page()

        print("\n[1] Navigating to app...")
        page.goto("http://localhost:8501", timeout=30000)

        print("[2] Waiting for page load...")
        page.wait_for_load_state("networkidle")
        time.sleep(5)

        # Take initial screenshot
        page.screenshot(path="/tmp/debug_initial.png")
        print("   Screenshot: /tmp/debug_initial.png")

        # Debug: Print all visible text on page
        print("\n[3] Page title:", page.title())

        # Try to find tabs
        print("\n[4] Looking for tabs...")
        tabs = page.locator('button[role="tab"]')
        print(f"   Found {tabs.count()} tabs with role='tab'")

        # Try data-baseweb tabs
        baseweb_tabs = page.locator('[data-baseweb="tab"]')
        print(f"   Found {baseweb_tabs.count()} tabs with data-baseweb='tab'")

        # Try Streamlit tab list
        st_tabs = page.locator('[data-testid="stTabs"] button')
        print(f"   Found {st_tabs.count()} Streamlit tabs")
        for i in range(st_tabs.count()):
            text = st_tabs.nth(i).inner_text()
            print(f"      Tab {i}: '{text}'")

        # Look for any buttons
        print("\n[5] Looking for buttons...")
        buttons = page.locator('button')
        print(f"   Found {buttons.count()} buttons total")

        # Look for sidebar
        print("\n[6] Looking for sidebar...")
        sidebar = page.locator('[data-testid="stSidebar"]')
        print(f"   Sidebar found: {sidebar.count() > 0}")

        # Try to click on Expert Panel if found
        print("\n[7] Trying to click Expert Panel tab...")
        try:
            # Wait and look for it
            page.wait_for_selector('text=Expert Panel', timeout=10000)
            expert_btn = page.locator('text=Expert Panel').first
            print(f"   Found Expert Panel text: {expert_btn.is_visible()}")
            expert_btn.click()
            print("   Clicked!")
            time.sleep(2)
            page.screenshot(path="/tmp/debug_after_click.png")
        except Exception as e:
            print(f"   Error: {e}")

        # Take final screenshot
        page.screenshot(path="/tmp/debug_final.png", full_page=True)
        print("\n[8] Final screenshot: /tmp/debug_final.png")

        print("\n[9] Waiting 10 seconds for inspection...")
        time.sleep(10)

        browser.close()


if __name__ == "__main__":
    debug_page()
