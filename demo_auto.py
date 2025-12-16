"""
Automated Demo with Screenshots for Travel Planner

Takes screenshots at each step to show features.
Run with: python3 demo_auto.py
"""

from playwright.sync_api import sync_playwright
import time
import os

APP_URL = "http://localhost:8000"
SCREENSHOT_DIR = "/Users/nelsonliu/Travel Planner/demo_screenshots"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def screenshot(page, name, step_num):
    path = f"{SCREENSHOT_DIR}/{step_num:02d}_{name}.png"
    page.screenshot(path=path)
    print(f"   📸 Screenshot saved: {name}.png")
    return path

def demo():
    with sync_playwright() as p:
        print("\n🚀 Launching browser (headless for speed)...")
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1400, "height": 900})

        # ============================================================
        # STEP 1: Load App
        # ============================================================
        print("\n[1/6] Loading Travel Planner...")
        page.goto(APP_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        screenshot(page, "01_welcome_screen", 1)
        print("   ✅ Welcome message loaded")

        # ============================================================
        # STEP 2: Type Plan Command (without settings - use defaults)
        # ============================================================
        print("\n[2/6] Testing 'Plan my trip' command...")

        # Find and fill the chat input
        # Chainlit typically uses a div with contenteditable or specific class
        page.evaluate("""
            // Find chat input and set value
            const inputs = document.querySelectorAll('textarea, input[type="text"], [contenteditable="true"]');
            for (let input of inputs) {
                if (input.offsetParent !== null) {  // visible
                    input.value = 'Plan my trip';
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    break;
                }
            }
        """)
        time.sleep(1)
        screenshot(page, "02_plan_command_typed", 2)

        # Try to submit
        page.keyboard.press("Enter")
        time.sleep(2)

        # Check if destination is needed
        page_text = page.content()
        if "destination" in page_text.lower() and "setting" in page_text.lower():
            print("   ℹ️ App needs destination configured first")
            screenshot(page, "02b_needs_destination", 2)

        # ============================================================
        # STEP 3: Wait for Expert Responses
        # ============================================================
        print("\n[3/6] Waiting for expert responses...")

        expert_emojis = {
            "💰": "Budget Advisor",
            "🚗": "Logistics Planner",
            "🏨": "Accommodation Specialist",
            "🎯": "Activity Curator",
            "🛡️": "Safety Expert",
            "🌤️": "Weather Analyst",
            "🎎": "Local Culture Guide",
            "🍜": "Food & Dining Expert"
        }

        found_experts = set()
        for i in range(30):  # Wait up to 30 seconds
            content = page.content()
            for emoji, name in expert_emojis.items():
                if emoji in content and emoji not in found_experts:
                    found_experts.add(emoji)
                    print(f"   ✅ {emoji} {name} responded")

            if "complete" in content.lower() or len(found_experts) >= 4:
                break
            time.sleep(1)

        screenshot(page, "03_expert_responses", 3)

        # ============================================================
        # STEP 4: Scroll to show all experts
        # ============================================================
        print("\n[4/6] Capturing full response...")
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1)
        screenshot(page, "04_full_responses", 4)

        # Scroll back up
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1)

        # ============================================================
        # STEP 5: Summary
        # ============================================================
        print("\n[5/6] Creating summary...")

        # Full page screenshot
        page.screenshot(path=f"{SCREENSHOT_DIR}/05_full_page.png", full_page=True)
        print("   📸 Full page screenshot saved")

        # ============================================================
        # RESULTS
        # ============================================================
        print("\n" + "="*60)
        print("  DEMO RESULTS")
        print("="*60)

        if found_experts:
            print(f"\n  Experts that responded: {len(found_experts)}")
            for emoji in found_experts:
                print(f"    {emoji} {expert_emojis.get(emoji, 'Unknown')}")
        else:
            print("\n  ⚠️ No expert responses detected (may need destination config)")

        print(f"\n  Screenshots saved to: {SCREENSHOT_DIR}/")
        print("  Files:")
        for f in sorted(os.listdir(SCREENSHOT_DIR)):
            if f.endswith('.png'):
                print(f"    - {f}")

        browser.close()
        print("\n✅ Demo complete! Check screenshots folder.")


if __name__ == "__main__":
    print("\n" + "🧳"*30)
    print("\n  TRAVEL PLANNER AUTOMATED DEMO")
    print("  (Takes screenshots to show features)")
    print("\n" + "🧳"*30)
    demo()
