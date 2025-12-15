"""
Custom test for the clinical question.
Tests the full research flow with a complex clinical question.
"""

import os
import time
from playwright.sync_api import sync_playwright

TEST_URL = os.getenv("TEST_URL", "http://localhost:8501")
CLINICAL_QUESTION = "CDP in NSCLC and OVarian Cancer for a FOLR1-targeting, IFNa-carrying, CLEC5a macrophage engager"


def test_clinical_question_flow():
    """Test full research flow with the clinical question."""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=100)  # Headed mode with slight delay
        page = browser.new_page()
        page.set_viewport_size({"width": 1400, "height": 900})  # Larger viewport

        print("\n" + "=" * 60)
        print("TEST: Clinical Question Research Flow")
        print("=" * 60)
        print(f"\nQuestion: {CLINICAL_QUESTION}")

        # Navigate to app
        print(f"\n[1] Navigating to {TEST_URL}...")
        page.goto(TEST_URL, timeout=30000)
        page.wait_for_load_state("networkidle")
        time.sleep(3)  # Wait for Streamlit to fully render

        # Take screenshot of initial state
        page.screenshot(path="/tmp/clinical_test_1_initial.png", full_page=True)
        print("   Screenshot: /tmp/clinical_test_1_initial.png")

        # Find and fill the input - specifically the "Your Question" textarea in the main form
        print("[2] Finding question input...")

        # Use the aria-label to find the correct textarea (not the context textarea)
        input_element = page.locator('textarea[aria-label="Your Question"]')

        # Wait for it to be visible (Streamlit forms may take a moment to render)
        try:
            input_element.wait_for(state="visible", timeout=10000)
            print(f"   Found textarea with aria-label='Your Question'")
        except:
            # Fallback: try to find it by looking in the form
            form = page.locator('[data-testid="stForm"]')
            if form.count() > 0:
                input_element = form.locator('textarea').first
                if input_element.is_visible():
                    print(f"   Found textarea inside form")
                else:
                    page.screenshot(path="/tmp/clinical_test_error.png", full_page=True)
                    raise AssertionError("Could not find question input element")

        # Type the question - click first to focus, then type character by character
        print(f"[3] Typing question...")
        input_element.click()
        time.sleep(0.5)

        # Type character by character to trigger Streamlit updates
        input_element.type(CLINICAL_QUESTION, delay=10)
        time.sleep(1)

        page.screenshot(path="/tmp/clinical_test_2_filled.png", full_page=True)
        print("   Screenshot: /tmp/clinical_test_2_filled.png")

        # Find and click Research button (inside the form)
        print("[4] Clicking Research button...")
        research_btn = page.locator('[data-testid="stFormSubmitButton"] button')

        if research_btn.count() > 0 and research_btn.first.is_visible():
            # Use force click and dispatch click event
            research_btn.first.click(force=True)
            time.sleep(0.5)
            # Also try dispatching a click event
            research_btn.first.dispatch_event('click')
            print("   Clicked Research button (with force)")
        else:
            # Fallback: try Ctrl+Enter to submit form
            print("   Research button not visible, trying Ctrl+Enter...")
            input_element.press("Control+Enter")

        time.sleep(2)
        page.screenshot(path="/tmp/clinical_test_3_clicked.png", full_page=True)
        print("   Screenshot: /tmp/clinical_test_3_clicked.png")

        # Wait for processing to start - look for SPECIFIC indicators
        print("[5] Waiting for processing...")

        # Check for progress indicators - these are specific to the research flow
        processing_started = False
        for i in range(30):  # Try for 30 seconds
            page_text = page.content()
            # Look for status messages that only appear during processing
            if any(indicator in page_text for indicator in [
                "Parsing question", "Searching literature", "Consulting experts",
                "Validating claims", "Synthesizing", "stStatusWidget",
                "Running", "st-emotion-cache"
            ]):
                processing_started = True
                print(f"   ✅ Processing started at {i}s")
                page.screenshot(path="/tmp/clinical_test_4_processing.png", full_page=True)
                print("   Screenshot: /tmp/clinical_test_4_processing.png")
                break
            time.sleep(1)

        if not processing_started:
            page.screenshot(path="/tmp/clinical_test_4_no_processing.png", full_page=True)
            print("   ⚠️ No processing indicator found after 30s")
            print("   Screenshot: /tmp/clinical_test_4_no_processing.png")

        # Wait for results (up to 5 minutes for LLM processing)
        print("[6] Waiting for results (up to 5 minutes)...")

        result_found = False
        error_found = False
        error_message = ""

        # More specific indicators that the research output is showing
        OUTPUT_INDICATORS = [
            "## Recommendation",  # Header in output
            "## Key Supporting",  # Key Supporting Evidence section
            "## Key Risks",       # Key Risks section
            "PROCEED",            # Go/No-Go recommendation
            "DO NOT PROCEED",     # Go/No-Go recommendation
            "CONDITIONAL",        # Go/No-Go recommendation
            "Confidence:",        # Confidence level
            "Expert Perspectives",  # Expert section
            "synthesized",        # Result text
        ]

        for i in range(300):  # Check every second for 5 minutes
            page_text = page.content()

            # Check for error messages (be more specific)
            if "Error:" in page_text or "❌" in page_text:
                if "rate limit" in page_text.lower() or "429" in page_text:
                    error_message = "Rate limit error"
                    error_found = True
                    break
                elif "API key" in page_text:
                    error_message = "API key error"
                    error_found = True
                    break
                elif "overloaded" in page_text.lower() or "503" in page_text:
                    error_message = "Model overloaded (503)"
                    error_found = True
                    break

            # Check for ACTUAL output indicators (not just UI labels)
            for indicator in OUTPUT_INDICATORS:
                if indicator in page_text:
                    result_found = True
                    print(f"   ✅ Results appeared at {i}s (found: '{indicator}')")
                    break

            if result_found:
                break

            # Progress update every 30 seconds
            if i > 0 and i % 30 == 0:
                print(f"   Still waiting... ({i}s)")
                page.screenshot(path=f"/tmp/clinical_test_progress_{i}s.png", full_page=True)

            time.sleep(1)

        # Scroll down to capture full output
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1)

        # Take final screenshot - full page
        page.screenshot(path="/tmp/clinical_test_5_final.png", full_page=True)
        print("   Final screenshot: /tmp/clinical_test_5_final.png")

        # Also take a scrolled view screenshot
        page.evaluate("window.scrollTo(0, 500)")
        time.sleep(0.5)
        page.screenshot(path="/tmp/clinical_test_6_scrolled.png", full_page=True)
        print("   Scrolled screenshot: /tmp/clinical_test_6_scrolled.png")

        # Report results
        print("\n" + "=" * 60)
        if result_found:
            print("✅ TEST PASSED: Research flow completed successfully")
            print("   Output screenshots saved to /tmp/clinical_test_5_final.png and /tmp/clinical_test_6_scrolled.png")
        elif error_found:
            print(f"❌ TEST FAILED: {error_message}")
        else:
            print("⚠️ TEST INCONCLUSIVE: Timeout waiting for results (5 minutes)")
            # Dump page content for debugging
            with open("/tmp/clinical_test_page_content.html", "w") as f:
                f.write(page.content())
            print("   Page content saved to /tmp/clinical_test_page_content.html")
        print("=" * 60)

        browser.close()

        assert result_found, f"Research flow did not complete successfully. Error: {error_message if error_found else 'Timeout'}"


if __name__ == "__main__":
    test_clinical_question_flow()
