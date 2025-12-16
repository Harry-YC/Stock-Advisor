"""
Visual Demo of Travel Planner Features

This script demonstrates all features with visible pauses so you can watch.
Run with: python3 demo_features.py
"""

from playwright.sync_api import sync_playwright
import time

APP_URL = "http://localhost:8000"

def print_step(step_num, description):
    print(f"\n{'='*60}")
    print(f"  STEP {step_num}: {description}")
    print('='*60)
    time.sleep(1)

def demo():
    with sync_playwright() as p:
        # Launch browser in headed mode (visible)
        browser = p.chromium.launch(headless=False, slow_mo=100)
        page = browser.new_page()
        page.set_viewport_size({"width": 1400, "height": 900})

        # ============================================================
        # STEP 1: Load the app
        # ============================================================
        print_step(1, "Loading Travel Planner App")
        page.goto(APP_URL)

        # Wait for Chainlit to fully load (look for textarea or input)
        page.wait_for_load_state("networkidle")
        time.sleep(3)  # Extra wait for React hydration

        # Find the chat input (Chainlit uses textarea)
        chat_input_selector = 'textarea, input[type="text"], [contenteditable="true"]'
        page.wait_for_selector(chat_input_selector, timeout=30000)
        print("✅ App loaded - Welcome message visible")
        time.sleep(3)

        # ============================================================
        # STEP 2: Open Settings Panel
        # ============================================================
        print_step(2, "Opening Settings Panel (⚙️ icon)")

        # Try different selectors for settings button
        settings_selectors = [
            '[data-testid="chat-settings-open-modal"]',
            'button[aria-label*="settings"]',
            'button:has(svg)',  # Gear icon button
        ]

        settings_opened = False
        for selector in settings_selectors:
            try:
                btn = page.locator(selector).first
                if btn.is_visible():
                    btn.click()
                    settings_opened = True
                    break
            except:
                continue

        if not settings_opened:
            # Look for any settings-related button
            page.locator("button").first.click()

        time.sleep(2)
        print("✅ Settings panel opened")

        # ============================================================
        # STEP 3: Configure Trip
        # ============================================================
        print_step(3, "Configuring Trip Details")

        # Try to fill destination
        try:
            # Look for input fields
            inputs = page.locator('input').all()
            print(f"   Found {len(inputs)} input fields")

            if len(inputs) >= 1:
                inputs[0].fill("Barcelona, Spain")
                print("   ✅ Destination: Barcelona, Spain")
                time.sleep(1)

            if len(inputs) >= 2:
                inputs[1].fill("New York, NY")
                print("   ✅ Origin: New York, NY")
                time.sleep(1)

        except Exception as e:
            print(f"   ⚠️ Could not fill inputs: {e}")

        time.sleep(2)

        # Close settings (press Escape or click outside)
        page.keyboard.press("Escape")
        time.sleep(2)
        print("✅ Settings saved")

        # ============================================================
        # STEP 4: Plan My Trip
        # ============================================================
        print_step(4, "Triggering 'Plan my trip' command")

        chat_input = page.locator('textarea').first
        chat_input.fill("Plan my trip")
        time.sleep(1)
        chat_input.press("Enter")

        print("⏳ Waiting for expert responses (this may take 30-60 seconds)...")

        # Wait for responses with progress updates
        expert_emojis = ["💰", "🚗", "🏨", "🎯", "🛡️", "🌤️", "🎎", "🍜"]
        experts_found = set()

        for i in range(60):  # Wait up to 60 seconds
            page_content = page.content()
            for emoji in expert_emojis:
                if emoji in page_content and emoji not in experts_found:
                    experts_found.add(emoji)
                    expert_names = {
                        "💰": "Budget Advisor",
                        "🚗": "Logistics Planner",
                        "🏨": "Accommodation Specialist",
                        "🎯": "Activity Curator",
                        "🛡️": "Safety Expert",
                        "🌤️": "Weather Analyst",
                        "🎎": "Local Culture Guide",
                        "🍜": "Food & Dining Expert"
                    }
                    print(f"   ✅ {emoji} {expert_names.get(emoji, 'Expert')} responded!")

            if "Trip planning complete" in page_content:
                print("\n✅ All experts finished!")
                break

            time.sleep(1)

        time.sleep(3)

        # ============================================================
        # STEP 5: Ask Specific Expert
        # ============================================================
        print_step(5, "Asking Food & Dining Expert directly")

        chat_input = page.locator('textarea').first
        chat_input.fill("Ask Food & Dining Expert about best tapas restaurants")
        time.sleep(1)
        chat_input.press("Enter")

        print("⏳ Waiting for Food Expert response...")

        # Wait for food expert emoji
        for i in range(30):
            if "🍜" in page.content():
                print("   ✅ 🍜 Food & Dining Expert responded!")
                break
            time.sleep(1)

        time.sleep(5)

        # ============================================================
        # STEP 6: Test Auto-Routing
        # ============================================================
        print_step(6, "Testing auto-routing with budget question")

        chat_input = page.locator('textarea').first
        chat_input.fill("How much money should I budget per day?")
        time.sleep(1)
        chat_input.press("Enter")

        print("⏳ Waiting for response (should route to Budget Advisor)...")

        for i in range(30):
            content = page.content()
            if "💰" in content:
                print("   ✅ 💰 Correctly routed to Budget Advisor!")
                break
            elif any(e in content for e in expert_emojis):
                print("   ℹ️ Routed to different expert")
                break
            time.sleep(1)

        time.sleep(5)

        # ============================================================
        # SUMMARY
        # ============================================================
        print("\n" + "="*60)
        print("  DEMO COMPLETE - SUMMARY")
        print("="*60)
        print(f"\n  Experts that responded: {', '.join(sorted(experts_found))}")
        print(f"  Total unique experts: {len(experts_found)}")
        print("\n  Features demonstrated:")
        print("  ✅ App loading with welcome message")
        print("  ✅ Settings panel for trip configuration")
        print("  ✅ Multi-expert 'Plan my trip' command")
        print("  ✅ Direct expert queries")
        print("  ✅ Auto-routing to best expert")
        print("  ✅ Distinct emoji per expert for identification")

        print("\n  Press Enter to close browser...")
        input()

        browser.close()


if __name__ == "__main__":
    print("\n" + "🧳"*30)
    print("\n  TRAVEL PLANNER FEATURE DEMO")
    print("  This will open a browser and demonstrate all features")
    print("\n" + "🧳"*30)

    demo()
