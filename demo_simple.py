"""
Simple Interactive Demo for Travel Planner

This script opens the browser and lets you interact manually
while showing what to test.

Run with: python3 demo_simple.py
"""

from playwright.sync_api import sync_playwright
import time

APP_URL = "http://localhost:8000"

def demo():
    with sync_playwright() as p:
        print("\n🚀 Launching browser...")
        browser = p.chromium.launch(headless=False, slow_mo=50)
        page = browser.new_page()
        page.set_viewport_size({"width": 1400, "height": 900})

        # Load app
        print("📱 Loading Travel Planner...")
        page.goto(APP_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        print("\n" + "="*60)
        print("  ✅ APP LOADED - Welcome message should be visible")
        print("="*60)

        print("""
📋 MANUAL TEST STEPS:

1️⃣  CONFIGURE TRIP:
   - Click the ⚙️ settings icon (top right)
   - Set Destination: "Barcelona, Spain"
   - Set Origin: "New York, NY"
   - Keep default dates and budget
   - Close settings

2️⃣  PLAN TRIP:
   - Type: "Plan my trip"
   - Press Enter
   - Watch for 4 expert responses with emojis:
     💰 Budget Advisor
     🚗 Logistics Planner
     🏨 Accommodation Specialist
     🎯 Activity Curator

3️⃣  ASK SPECIFIC EXPERT:
   - Type: "Ask Food & Dining Expert about tapas"
   - Watch for 🍜 Food & Dining Expert response

4️⃣  TEST AUTO-ROUTING:
   - Type: "How much money do I need?"
   - Should route to 💰 Budget Advisor

Press Enter when ready to start...""")

        input()

        print("\n⏳ Browser is open - follow the steps above")
        print("   Press Enter when done testing to close browser...")
        input()

        # Show summary
        print("\n" + "="*60)
        print("  📊 EXPECTED RESULTS CHECKLIST")
        print("="*60)
        print("""
✅ Welcome message with instructions appeared
✅ Settings panel opened with trip configuration fields
✅ "Plan my trip" triggered multiple expert responses
✅ Each expert has distinct emoji (💰🚗🏨🎯)
✅ Expert responses streamed in real-time
✅ "Trip planning complete" message appeared
✅ Direct expert query worked (🍜 Food Expert)
✅ Auto-routing worked for budget question (💰)
""")

        browser.close()
        print("✅ Demo complete!")


if __name__ == "__main__":
    print("\n" + "🧳"*30)
    print("\n  TRAVEL PLANNER INTERACTIVE DEMO")
    print("\n" + "🧳"*30)
    demo()
