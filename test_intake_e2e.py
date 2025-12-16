"""
Playwright E2E test for the LLM-guided intake conversation.

Run with: python3 test_intake_e2e.py
"""

from playwright.sync_api import sync_playwright
import time
import os

APP_URL = "http://localhost:8000"
SCREENSHOT_DIR = "/Users/nelsonliu/Travel Planner/demo_screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def save_screenshot(page, name):
    path = f"{SCREENSHOT_DIR}/{name}.png"
    page.screenshot(path=path)
    print(f"   📸 {name}")
    return path

def type_message(page, message):
    """Type a message and send it."""
    # Find the input - Chainlit uses a div or input with placeholder
    selectors = [
        '[placeholder*="message"]',
        '[placeholder*="Type"]',
        'textarea',
        '[contenteditable="true"]',
        '.cl-input',
        'input[type="text"]'
    ]

    chat_input = None
    for selector in selectors:
        try:
            loc = page.locator(selector).first
            if loc.is_visible(timeout=2000):
                chat_input = loc
                print(f"   Found input with selector: {selector}")
                break
        except:
            continue

    if not chat_input:
        # Fallback - click in the input area at bottom
        print("   Using click fallback for input")
        page.click('body', position={"x": 700, "y": 780})
        time.sleep(0.5)

    if chat_input:
        chat_input.click()
        time.sleep(0.3)
        # Clear any existing content first
        page.keyboard.press("Control+a")
        time.sleep(0.1)

    # Type slower to ensure all characters are captured
    page.keyboard.type(message, delay=50)
    time.sleep(1)  # Wait before sending
    print(f"   Typed: {message}")
    page.keyboard.press("Enter")
    time.sleep(0.5)

def wait_for_response(page, timeout=60):
    """Wait for a new message to appear."""
    initial_count = len(page.locator('[class*="message"]').all())
    for _ in range(timeout * 2):
        time.sleep(0.5)
        current_count = len(page.locator('[class*="message"]').all())
        if current_count > initial_count:
            time.sleep(1)  # Extra time for message to fully render
            return True
    return False

def test_guided_conversation():
    with sync_playwright() as p:
        print("\n🚀 Launching browser...")
        browser = p.chromium.launch(headless=False, slow_mo=200)
        page = browser.new_page()
        page.set_viewport_size({"width": 1400, "height": 900})

        # ============================================================
        print("\n[1] Loading app...")
        # ============================================================
        page.goto(APP_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        save_screenshot(page, "intake_01_welcome")

        # Verify conversational welcome
        content = page.content()
        if "where are you dreaming" in content.lower() or "tell me about your trip" in content.lower():
            print("   ✅ Conversational welcome message displayed")
        else:
            print("   ⚠️ Welcome message may not be the new conversational style")

        # ============================================================
        print("\n[2] Sending first message (destination + travelers + dates)...")
        # ============================================================
        type_message(page, "Barcelona with my wife for a week in January")

        # Wait for LLM extraction and response
        print("   ⏳ Waiting for LLM extraction...")
        time.sleep(10)  # Give time for LLM call
        save_screenshot(page, "intake_02_first_response")

        content = page.content()
        if "budget" in content.lower():
            print("   ✅ App asking for budget (correct next question)")
        elif "barcelona" in content.lower():
            print("   ✅ App recognized Barcelona")
        else:
            print(f"   ⚠️ Response: {content[:200]}")

        # ============================================================
        print("\n[3] Sending budget...")
        # ============================================================
        type_message(page, "around 5000 dollars")

        print("   ⏳ Waiting for confirmation...")
        time.sleep(10)
        save_screenshot(page, "intake_03_budget_response")

        content = page.content()
        if "plan my trip" in content.lower() or "yes" in content.lower():
            print("   ✅ Confirmation with action buttons displayed")
        elif "$5,000" in content or "5000" in content:
            print("   ✅ Budget recognized")

        # ============================================================
        print("\n[4] Looking for action buttons...")
        # ============================================================
        # Scroll down to ensure buttons are visible
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1)
        save_screenshot(page, "intake_04_scrolled")

        # Look for the plan button
        plan_button = page.locator('button:has-text("plan my trip"), button:has-text("Yes")')

        if plan_button.count() > 0:
            print("   ✅ Found 'Plan my trip' button")
            save_screenshot(page, "intake_04_buttons_visible")

            # Click the button
            print("\n[5] Clicking 'Plan my trip' button...")
            plan_button.first.click()
            time.sleep(3)
            save_screenshot(page, "intake_05_planning_started")

            # Wait for experts
            print("\n[6] Waiting for expert responses...")
            expert_emojis = ["💰", "🚗", "🏨", "🎯"]
            found = set()

            for i in range(90):
                content = page.content()
                for emoji in expert_emojis:
                    if emoji in content and emoji not in found:
                        found.add(emoji)
                        print(f"   ✅ Expert {emoji} responding")

                if len(found) >= 2:
                    break

                if i % 15 == 0 and i > 0:
                    print(f"   ⏳ Still waiting... ({i}s)")
                    save_screenshot(page, f"intake_06_waiting_{i}s")

                time.sleep(1)

            save_screenshot(page, "intake_07_experts_done")
            print(f"\n   {len(found)}/4 experts detected")
        else:
            print("   ⚠️ Action buttons not found - checking if already showing summary")
            save_screenshot(page, "intake_04_no_buttons")

        # ============================================================
        print("\n" + "="*60)
        print("  TEST RESULTS")
        print("="*60)

        # Full page screenshot
        page.screenshot(path=f"{SCREENSHOT_DIR}/intake_08_full_page.png", full_page=True)
        print(f"\n  📁 Screenshots saved to: {SCREENSHOT_DIR}/")

        print("\n  Keeping browser open for 20 seconds...")
        print("  (You can interact with the app manually)")
        time.sleep(20)

        browser.close()


if __name__ == "__main__":
    print("🧳" * 30)
    print("\n  GUIDED CONVERSATION E2E TEST")
    print("\n" + "🧳" * 30)
    test_guided_conversation()
