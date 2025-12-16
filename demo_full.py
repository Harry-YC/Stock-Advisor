"""
Full Automated Demo - Properly interacts with Chainlit UI

Run with: python3 demo_full.py
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

def demo():
    with sync_playwright() as p:
        print("\n🚀 Launching browser...")
        browser = p.chromium.launch(headless=False, slow_mo=300)  # Visible, slowed down
        page = browser.new_page()
        page.set_viewport_size({"width": 1400, "height": 900})

        # ============================================================
        print("\n[1] Loading app...")
        # ============================================================
        page.goto(APP_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        save_screenshot(page, "01_welcome")
        print("   ✅ App loaded")

        # ============================================================
        print("\n[2] Opening settings (clicking gear icon)...")
        # ============================================================
        # Chainlit settings button - look for the gear SVG icon in the input area
        settings_opened = False

        # Try multiple selectors for settings button
        selectors = [
            'button[id*="settings"]',
            '[class*="settings"]',
            'button:nth-of-type(2)',  # Second button in input area
            'svg[class*="settings"]',
        ]

        # First try clicking in the input area, then find settings
        try:
            # Look for settings icon - it's the gear in the bottom input area
            # The gear icon in Chainlit is typically the second button with an SVG
            all_buttons = page.locator('button').all()
            print(f"   Found {len(all_buttons)} buttons")

            for idx, btn in enumerate(all_buttons):
                try:
                    if btn.is_visible():
                        btn.click(timeout=2000)
                        time.sleep(1.5)
                        # Check if modal opened by looking for inputs
                        modal_inputs = page.locator('input').all()
                        visible_inputs = [inp for inp in modal_inputs if inp.is_visible()]
                        if len(visible_inputs) >= 2:
                            settings_opened = True
                            print(f"   Settings opened via button {idx}")
                            break
                        else:
                            # Not a settings modal, press Escape and try next
                            page.keyboard.press("Escape")
                            time.sleep(0.5)
                except Exception as e:
                    continue
        except Exception as e:
            print(f"   Button search: {e}")

        if settings_opened:
            time.sleep(2)
            save_screenshot(page, "02_settings_open")
            print("   ✅ Settings panel opened")

            # ============================================================
            print("\n[3] Filling trip details...")
            # ============================================================
            try:
                # Look for text inputs in the settings modal
                inputs = page.locator('input').all()
                visible_inputs = [inp for inp in inputs if inp.is_visible()]
                print(f"   Found {len(visible_inputs)} visible input fields")

                # Fill inputs by order - in Chainlit ChatSettings:
                # Input 0: Destination
                # Input 1: Origin
                # Input 2: Departure Date
                # Input 3: Return Date
                filled_dest = False
                filled_origin = False

                for i, inp in enumerate(visible_inputs):
                    placeholder = inp.get_attribute("placeholder") or ""
                    inp_id = inp.get_attribute("id") or ""
                    print(f"   Input {i}: placeholder='{placeholder}', id='{inp_id}'")

                    # Try to match by ID first (Chainlit uses widget IDs)
                    if "destination" in inp_id.lower() or (i == 0 and not filled_dest):
                        inp.fill("Barcelona, Spain")
                        filled_dest = True
                        print("   ✅ Destination: Barcelona, Spain")
                    elif "origin" in inp_id.lower() or (i == 1 and not filled_origin and filled_dest):
                        inp.fill("New York, NY")
                        filled_origin = True
                        print("   ✅ Origin: New York, NY")

                time.sleep(1)
                save_screenshot(page, "03_settings_filled")

                # Submit settings if there's a submit button
                submit_btn = page.locator('button:has-text("Confirm"), button:has-text("Save"), button[type="submit"]').first
                if submit_btn.is_visible():
                    submit_btn.click()
                    print("   ✅ Settings submitted")
                    time.sleep(2)

            except Exception as e:
                print(f"   ⚠️ Could not fill settings: {e}")

            # Close settings
            page.keyboard.press("Escape")
            time.sleep(2)
        else:
            print("   ⚠️ Could not open settings modal - continuing without configuration")

        save_screenshot(page, "04_settings_closed")

        # ============================================================
        print("\n[4] Sending 'Plan my trip' command...")
        # ============================================================
        # Chainlit uses a contenteditable div or textarea
        try:
            # Find the input - could be textarea or div
            chat_input = page.locator('textarea, [contenteditable="true"], [placeholder*="message"]').first
            chat_input.click()
            time.sleep(0.5)

            # Type the message
            page.keyboard.type("Plan my trip", delay=50)
            time.sleep(1)
            save_screenshot(page, "05_command_typed")

            # Submit with Enter
            page.keyboard.press("Enter")
            print("   ✅ Command sent")
        except Exception as e:
            print(f"   ⚠️ Error typing command: {e}")
            # Try using JavaScript to input text
            try:
                page.evaluate('''
                    const input = document.querySelector('textarea') ||
                                  document.querySelector('[contenteditable="true"]');
                    if (input) {
                        if (input.tagName === 'TEXTAREA') {
                            input.value = 'Plan my trip';
                        } else {
                            input.innerText = 'Plan my trip';
                        }
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                ''')
                time.sleep(0.5)
                page.keyboard.press("Enter")
                print("   ✅ Command sent (via JS)")
            except Exception as e2:
                print(f"   ⚠️ JS fallback failed: {e2}")

        # ============================================================
        print("\n[5] Waiting for expert responses...")
        # ============================================================
        expert_emojis = {
            "💰": "Budget Advisor",
            "🚗": "Logistics Planner",
            "🏨": "Accommodation Specialist",
            "🎯": "Activity Curator",
            "🛡️": "Safety Expert",
            "🍜": "Food & Dining Expert"
        }

        found = set()
        for i in range(90):  # Wait up to 90 seconds
            content = page.content()

            for emoji, name in expert_emojis.items():
                if emoji in content and emoji not in found:
                    found.add(emoji)
                    print(f"   ✅ {emoji} {name}")
                    save_screenshot(page, f"06_expert_{name.replace(' ', '_').replace('&', '')}")

            if "complete" in content.lower() and len(found) >= 2:
                break

            if i % 10 == 0 and i > 0:
                print(f"   ⏳ Still waiting... ({i}s)")

            time.sleep(1)

        save_screenshot(page, "07_all_experts")

        # Full page screenshot
        page.screenshot(path=f"{SCREENSHOT_DIR}/08_full_page.png", full_page=True)
        print("   📸 Full page captured")

        # ============================================================
        print("\n" + "="*60)
        print("  RESULTS")
        print("="*60)

        if found:
            print(f"\n  ✅ {len(found)} experts responded:")
            for emoji in found:
                print(f"     {emoji} {expert_emojis.get(emoji, 'Expert')}")
        else:
            print("\n  ⚠️ No expert responses (check if destination was configured)")

        print(f"\n  📁 Screenshots: {SCREENSHOT_DIR}/")

        print("\n  Keeping browser open for 30 seconds...")
        print("  (You can interact with the app manually)")
        time.sleep(30)

        browser.close()


if __name__ == "__main__":
    print("🧳" * 30)
    print("\n  TRAVEL PLANNER FULL DEMO")
    print("  Watch the browser!")
    print("\n" + "🧳" * 30)
    demo()
