"""
Playwright test for Two-Pass Expert Panel mode.

Tests that:
1. Pass 1 (immediate LLM response) shows quickly
2. Background literature search runs
3. Pass 2 (validation) appears after literature search completes
"""

import time
from playwright.sync_api import sync_playwright


def test_two_pass_expert_panel():
    """Test the two-pass expert panel flow."""

    with sync_playwright() as p:
        # Launch browser (headless=False for visual debugging)
        browser = p.chromium.launch(headless=False, slow_mo=300)
        page = browser.new_page()

        print("\n" + "="*60)
        print("TWO-PASS EXPERT PANEL TEST")
        print("="*60)

        # 1. Navigate to app
        print("\n[1] Navigating to app...")
        page.goto("http://localhost:8501", timeout=30000)
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        # 2. Create a project first (with unique name using timestamp)
        print("[2] Creating a project...")
        import random
        project_name = f"TwoPass_{random.randint(1000,9999)}"

        sidebar = page.locator('[data-testid="stSidebar"]')

        # Find project name input
        project_input = sidebar.locator('input[type="text"]').first
        if project_input.is_visible():
            project_input.fill(project_name)
            time.sleep(0.5)

            # Click Create button
            create_btn = sidebar.get_by_role("button", name="Create")
            if create_btn.is_visible():
                create_btn.click()
                print(f"   Created project '{project_name}'")
                page.wait_for_load_state("networkidle")
                time.sleep(3)

        # 3. Switch to Advanced mode (v3.0 uses Conversational mode by default)
        print("[3] Switching to Advanced mode...")
        switch_btn = sidebar.locator('button:has-text("Switch to Advanced")')
        if switch_btn.count() > 0 and switch_btn.first.is_visible():
            switch_btn.first.click()
            print("   Clicked 'Switch to Advanced' button")
            page.wait_for_load_state("networkidle")
            time.sleep(2)
        else:
            print("   'Switch to Advanced' button not found - may already be in Advanced mode")

        # 4. Navigate to Expert Panel tab
        print("[4] Clicking Expert Panel tab...")
        page.wait_for_selector('text=Expert Panel', timeout=10000)
        expert_tab = page.locator('text=Expert Panel').first
        expert_tab.click()
        time.sleep(2)

        # Take screenshot after clicking Expert Panel
        page.screenshot(path="/tmp/two_pass_step4.png")
        print("   Screenshot: /tmp/two_pass_step4.png")

        # 5. Scroll down to find the question input area
        print("[5] Looking for Research Question input...")
        page.evaluate("window.scrollTo(0, 500)")
        time.sleep(1)

        # Find the Research Question textarea - look for one with "evidence" placeholder
        textareas = page.locator('textarea')
        print(f"   Found {textareas.count()} text areas")

        question_textarea = None
        for i in range(textareas.count()):
            ta = textareas.nth(i)
            if ta.is_visible():
                placeholder = ta.get_attribute("placeholder") or ""
                # The Research Question field has "evidence" in placeholder
                if "evidence" in placeholder.lower():
                    question_textarea = ta
                    print(f"   Found Research Question textarea: '{placeholder[:50]}...'")
                    break

        # 6. Enter the research question
        print("[6] Entering research question...")
        if question_textarea:
            question_textarea.fill("What is the mechanism of action of KRAS G12C inhibitors?")
            print("   Entered question: 'What is the mechanism of action of KRAS G12C inhibitors?'")
            # Press Cmd+Enter to apply (Streamlit text areas require this)
            question_textarea.press("Meta+Enter")
            time.sleep(2)
            page.wait_for_load_state("networkidle")
            time.sleep(2)
        else:
            print("   ERROR: Research Question textarea not found!")
            page.screenshot(path="/tmp/two_pass_error.png")
            browser.close()
            return

        # Take screenshot after entering question
        page.screenshot(path="/tmp/two_pass_step6.png")
        print("   Screenshot: /tmp/two_pass_step6.png")

        # 7. Look for and click Run Round button
        print("[7] Looking for 'Run Round' button...")
        page.evaluate("window.scrollTo(0, 1000)")
        time.sleep(1)

        run_button = page.locator('button').filter(has_text="Run Round")
        if run_button.count() > 0 and run_button.first.is_visible():
            print("   Found Run Round button, clicking...")
            run_button.first.click()
        else:
            print("   Run Round button not found, looking for alternatives...")
            # Try other button patterns
            buttons = page.locator('button')
            for i in range(buttons.count()):
                btn = buttons.nth(i)
                text = btn.inner_text()
                if "run" in text.lower() and "round" in text.lower():
                    print(f"   Found button: '{text}'")
                    btn.click()
                    break

        # 8. Wait for Pass 1 response
        print("[8] Waiting for Pass 1 (immediate response)...")
        try:
            # Look for Pass 1 indicators
            pass1_found = False
            for _ in range(30):  # Wait up to 30 seconds
                time.sleep(1)

                # Check for "Initial Response" text
                if page.locator('text=Initial Response').count() > 0:
                    print("   ✅ Found 'Initial Response' indicator!")
                    pass1_found = True
                    break

                # Check for "Pass 1" text
                if page.locator('text=Pass 1').count() > 0:
                    print("   ✅ Found 'Pass 1' indicator!")
                    pass1_found = True
                    break

                # Check for status messages
                if page.locator('text=Getting immediate expert').count() > 0:
                    print("   ⏳ Pass 1 in progress...")

            if not pass1_found:
                print("   ⚠️ Pass 1 indicator not found")

        except Exception as e:
            print(f"   Error waiting for Pass 1: {e}")

        # Take screenshot
        page.screenshot(path="/tmp/two_pass_step8.png", full_page=True)
        print("   Screenshot: /tmp/two_pass_step8.png")

        # 9. Wait for Pass 2 response
        print("[9] Waiting for Pass 2 (literature validation)...")
        try:
            pass2_found = False
            for _ in range(60):  # Wait up to 60 seconds
                time.sleep(1)

                # Check for Pass 2 indicators
                if page.locator('text=Literature Validated').count() > 0:
                    print("   ✅ Found 'Literature Validated' - Pass 2 complete!")
                    pass2_found = True
                    break

                if page.locator('text=Evidence Update').count() > 0:
                    print("   ✅ Found 'Evidence Update' - contradictions detected!")
                    pass2_found = True
                    break

                if page.locator('text=No papers found').count() > 0:
                    print("   ⚠️ No papers found - Pass 1 only mode")
                    pass2_found = True
                    break

                # Check for progress
                if page.locator('text=Validating against literature').count() > 0:
                    print("   ⏳ Pass 2 validation in progress...")

                if page.locator('text=Waiting for PubMed').count() > 0:
                    print("   ⏳ Waiting for PubMed search...")

            if not pass2_found:
                print("   ⚠️ Pass 2 indicator not found within timeout")

        except Exception as e:
            print(f"   Error waiting for Pass 2: {e}")

        # 10. Final screenshot
        time.sleep(3)
        page.screenshot(path="/tmp/two_pass_final.png", full_page=True)
        print("\n[10] Final screenshot: /tmp/two_pass_final.png")

        # 11. Check for expert responses
        print("[11] Checking for expert response content...")
        expanders = page.locator('[data-testid="stExpander"]')
        print(f"   Found {expanders.count()} expander sections")

        # Check for validation metrics
        if page.locator('text=Supported').count() > 0:
            print("   ✅ Found validation metrics")
        if page.locator('text=Contradicted').count() > 0:
            print("   ✅ Found contradiction metrics")

        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        print("\nScreenshots saved to /tmp/two_pass_*.png")
        print("Browser will stay open for 15 seconds...")
        time.sleep(15)

        browser.close()


if __name__ == "__main__":
    test_two_pass_expert_panel()
