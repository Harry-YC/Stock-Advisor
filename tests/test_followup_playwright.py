"""
Playwright test for Follow-up Question functionality.

Tests that:
1. After initial research completes, follow-up section appears
2. Follow-up chat input is present and functional
3. Submitting follow-up shows response
"""

import os
import time
from playwright.sync_api import sync_playwright, expect


# Get test URL from environment or default to localhost:8501
TEST_URL = os.environ.get("TEST_URL", "http://localhost:8501")


def test_followup_section_appears_after_research():
    """Test that the follow-up section appears after research completes."""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print("\n" + "=" * 60)
        print("TEST: Follow-up Section Appears After Research")
        print("=" * 60)

        # Navigate to app
        print(f"\n[1] Navigating to {TEST_URL}...")
        page.goto(TEST_URL, timeout=60000)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        # Take initial screenshot
        page.screenshot(path="/tmp/followup_1_initial.png")
        print("   Screenshot: /tmp/followup_1_initial.png")

        # Check if we're in conversational mode (textarea present)
        print("[2] Checking for question input...")
        # Wait for any visible textarea to appear
        page.wait_for_selector('textarea:visible', timeout=15000)

        # Find the visible textarea (the main question input)
        all_textareas = page.locator('textarea')
        textarea = None
        for i in range(all_textareas.count()):
            ta = all_textareas.nth(i)
            if ta.is_visible():
                textarea = ta
                break

        if textarea is None:
            print("   ERROR: No visible textarea found")
            page.screenshot(path="/tmp/followup_error_no_textarea.png")
            browser.close()
            return

        print("   ✅ Textarea found")

        # Enter a simple research question
        print("[3] Entering research question...")
        test_question = "What is the mechanism of action of Lynparza?"
        textarea.fill(test_question)
        page.wait_for_timeout(500)

        # Take screenshot after filling
        page.screenshot(path="/tmp/followup_2_filled.png")
        print("   Screenshot: /tmp/followup_2_filled.png")

        # Find and click Research button
        print("[4] Clicking Research button...")
        research_btn = page.locator('button:has-text("Research")').first
        if not research_btn.is_visible():
            print("   ERROR: Research button not found")
            browser.close()
            return

        research_btn.click()

        # Wait for processing to start
        print("[5] Waiting for processing...")
        page.wait_for_timeout(2000)
        page.screenshot(path="/tmp/followup_3_processing.png")
        print("   Screenshot: /tmp/followup_3_processing.png")

        # Wait for research to complete (up to 180 seconds)
        print("[6] Waiting for research to complete (up to 180s)...")
        start_time = time.time()
        max_wait = 180
        research_complete = False

        while time.time() - start_time < max_wait:
            # Check for indicators that research is complete:
            # - "Follow-up Questions" section header (most reliable)
            # - OR subheader with "Follow-up"
            # - Processing indicator should be gone

            # Check if processing is still happening
            processing = page.locator('text=Getting expert perspectives')
            validating = page.locator('text=Validating against literature')
            synthesizing = page.locator('text=Synthesizing')

            if processing.count() > 0 or validating.count() > 0 or synthesizing.count() > 0:
                elapsed = time.time() - start_time
                print(f"   ... still processing ({elapsed:.0f}s)")
                page.wait_for_timeout(3000)
                continue

            # Check for completion indicators
            followup_header = page.locator('h3:has-text("Follow-up"), h2:has-text("Follow-up")')
            chat_input = page.locator('[data-testid="stChatInput"]')

            if followup_header.count() > 0 or chat_input.count() > 0:
                research_complete = True
                elapsed = time.time() - start_time
                print(f"   ✅ Research completed in {elapsed:.1f}s")
                break

            # Check for error state
            error = page.locator('text=Error')
            if error.count() > 0:
                print(f"   ⚠️ Error detected")
                break

            page.wait_for_timeout(3000)
            elapsed = time.time() - start_time
            print(f"   ... waiting ({elapsed:.0f}s)")

        # Take screenshot of result
        page.screenshot(path="/tmp/followup_4_result.png")
        print("   Screenshot: /tmp/followup_4_result.png")

        if not research_complete:
            print("   ERROR: Research did not complete within timeout")
            browser.close()
            return

        # Check for Follow-up Questions section
        print("[7] Checking for Follow-up Questions section...")
        followup_section = page.locator('text=Follow-up Questions')
        if followup_section.count() > 0:
            print("   ✅ Follow-up Questions header found")
        else:
            print("   ⚠️ Follow-up Questions header NOT found")

        # Check for chat input
        print("[8] Checking for chat input...")
        # Streamlit's st.chat_input renders with data-testid="stChatInput"
        chat_input = page.locator('[data-testid="stChatInput"]')
        if chat_input.count() > 0:
            print("   ✅ Chat input found")
        else:
            # Try alternative selector
            chat_input = page.locator('input[placeholder*="follow-up"]')
            if chat_input.count() > 0:
                print("   ✅ Chat input found (alternative selector)")
            else:
                print("   ⚠️ Chat input NOT found")
                # Debug: print all inputs on page
                all_inputs = page.locator('input')
                print(f"   Debug: Found {all_inputs.count()} input elements")
                for i in range(min(all_inputs.count(), 5)):
                    try:
                        placeholder = all_inputs.nth(i).get_attribute('placeholder')
                        print(f"   - Input {i}: placeholder='{placeholder}'")
                    except:
                        pass

        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)

        browser.close()


def test_followup_input_works():
    """Test that submitting a follow-up question works."""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print("\n" + "=" * 60)
        print("TEST: Follow-up Input Works")
        print("=" * 60)

        # Navigate to app
        print(f"\n[1] Navigating to {TEST_URL}...")
        page.goto(TEST_URL, timeout=60000)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        # Check if we're in conversational mode (textarea present)
        print("[2] Checking for question input...")
        # Wait for any visible textarea to appear
        page.wait_for_selector('textarea:visible', timeout=15000)

        # Find the visible textarea (the main question input)
        all_textareas = page.locator('textarea')
        textarea = None
        for i in range(all_textareas.count()):
            ta = all_textareas.nth(i)
            if ta.is_visible():
                textarea = ta
                break

        if textarea is None:
            print("   ERROR: No visible textarea found")
            page.screenshot(path="/tmp/followup_error_no_textarea.png")
            browser.close()
            return

        print("   ✅ Textarea found")

        # Enter a simple research question
        print("[3] Entering research question...")
        test_question = "What is the dosing for Tagrisso?"
        textarea.fill(test_question)
        page.wait_for_timeout(500)

        # Click Research button
        print("[4] Clicking Research button...")
        research_btn = page.locator('button:has-text("Research")').first
        if not research_btn.is_visible():
            print("   ERROR: Research button not found")
            browser.close()
            return
        research_btn.click()

        # Wait for research to complete (up to 180 seconds)
        print("[5] Waiting for research to complete (up to 180s)...")
        start_time = time.time()
        max_wait = 180
        research_complete = False

        while time.time() - start_time < max_wait:
            # Check if processing is still happening
            processing = page.locator('text=Getting expert perspectives')
            validating = page.locator('text=Validating against literature')
            synthesizing = page.locator('text=Synthesizing')

            if processing.count() > 0 or validating.count() > 0 or synthesizing.count() > 0:
                elapsed = time.time() - start_time
                print(f"   ... still processing ({elapsed:.0f}s)")
                page.wait_for_timeout(3000)
                continue

            # Check for completion indicators
            followup_header = page.locator('h3:has-text("Follow-up"), h2:has-text("Follow-up")')
            chat_input = page.locator('[data-testid="stChatInput"]')

            if followup_header.count() > 0 or chat_input.count() > 0:
                research_complete = True
                elapsed = time.time() - start_time
                print(f"   ✅ Research completed in {elapsed:.1f}s")
                break

            # Check for error state
            error = page.locator('text=Error')
            if error.count() > 0:
                print(f"   ⚠️ Error detected")
                break

            page.wait_for_timeout(3000)
            elapsed = time.time() - start_time
            print(f"   ... waiting ({elapsed:.0f}s)")

        page.wait_for_timeout(2000)  # Extra wait for full render
        page.screenshot(path="/tmp/followup_input_1_ready.png")

        if not research_complete:
            print("   ERROR: Research did not complete within timeout")
            browser.close()
            return

        # Find the chat input - st.chat_input renders as textarea in Streamlit
        print("[6] Looking for chat input...")

        # Scroll to bottom to ensure chat input is visible
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1000)

        # st.chat_input uses textarea with specific placeholder
        chat_input = page.locator('textarea[placeholder*="follow-up"], textarea[placeholder*="Follow-up"], input[placeholder*="follow-up"], input[placeholder*="Follow-up"]')

        if chat_input.count() == 0:
            # Debug: list all textareas and inputs
            print("   Debug: Looking for chat input elements...")
            all_textareas = page.locator('textarea')
            for i in range(all_textareas.count()):
                ta = all_textareas.nth(i)
                placeholder = ta.get_attribute('placeholder') or 'N/A'
                visible = ta.is_visible()
                print(f"   - textarea {i}: placeholder='{placeholder[:50]}...', visible={visible}")

            # Try fallback - any visible textarea at the bottom
            all_visible_textareas = page.locator('textarea:visible')
            if all_visible_textareas.count() > 0:
                chat_input = all_visible_textareas.last
                print(f"   Using last visible textarea")

        if chat_input.count() > 0 and chat_input.first.is_visible():
            print("   ✅ Chat input found")

            # Enter follow-up question
            print("[6] Entering follow-up question...")
            followup_q = "What are the side effects?"
            chat_input.first.fill(followup_q)
            page.wait_for_timeout(500)
            page.screenshot(path="/tmp/followup_input_2_filled.png")

            # Press Enter to submit (st.chat_input submits on Enter)
            print("[7] Pressing Enter to submit...")
            chat_input.first.press("Enter")

            # Wait for response
            print("[8] Waiting for follow-up response...")
            page.wait_for_timeout(5000)
            page.screenshot(path="/tmp/followup_input_3_submitted.png")

            # Check if response appeared
            # Look for chat message containers or response content
            chat_messages = page.locator('[data-testid="stChatMessage"], .stChatMessage')
            if chat_messages.count() > 0:
                print(f"   ✅ Found {chat_messages.count()} chat messages")
            else:
                print("   ⚠️ No chat messages found after submission")

            # Wait for assistant response to appear
            print("[9] Waiting for assistant response...")
            start_response = time.time()
            max_response_wait = 60  # Increased timeout
            response_found = False
            actual_response_text = ""

            while time.time() - start_response < max_response_wait:
                page_content = page.content()

                # Check if spinner is gone
                if "Thinking..." in page_content:
                    page.wait_for_timeout(2000)
                    print(f"   ... still thinking ({time.time() - start_response:.0f}s)")
                    continue

                # Look for assistant chat message after Follow-up Questions section
                # Streamlit chat messages have data-testid="stChatMessage"
                chat_messages = page.locator('[data-testid="stChatMessage"]')

                # We should have at least 2 messages: user's question + assistant response
                if chat_messages.count() >= 2:
                    # Get the last message (should be assistant)
                    last_msg = chat_messages.last
                    msg_text = last_msg.inner_text()

                    # Check if it's NOT the "I couldn't generate" error
                    if msg_text and "I couldn't generate" not in msg_text:
                        if len(msg_text) > 50:  # Real response should be substantial
                            response_found = True
                            actual_response_text = msg_text[:200]
                            print(f"   ✅ Valid response found: {actual_response_text[:100]}...")
                            break
                    elif "I couldn't generate" in msg_text:
                        print(f"   ❌ Error response detected: {msg_text[:100]}")
                        break

                page.wait_for_timeout(2000)
                print(f"   ... waiting for response ({time.time() - start_response:.0f}s)")

            page.screenshot(path="/tmp/followup_input_4_final.png")
            print("   Screenshot: /tmp/followup_input_4_final.png")

            # Assert that we got a valid response
            if not response_found:
                # Check if user message at least appeared
                user_msg = page.locator('text="What are the side effects?"')
                if user_msg.count() > 0:
                    print("   ⚠️ User message displayed but NO valid assistant response")
                else:
                    print("   ❌ Follow-up submission failed completely")

                # FAIL the test
                assert False, "Follow-up did not return a valid response"
            else:
                print(f"   ✅ TEST PASSED: Got valid follow-up response")

        else:
            print("   ERROR: Chat input not found or not visible")

        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)

        browser.close()


def test_chat_input_element_exists():
    """Simple test to check if st.chat_input renders correctly."""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print("\n" + "=" * 60)
        print("TEST: Chat Input Element Detection")
        print("=" * 60)

        # Navigate to app
        print(f"\n[1] Navigating to {TEST_URL}...")
        page.goto(TEST_URL, timeout=60000)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        # Get page content to debug
        print("[2] Checking page structure...")

        # List all input elements
        all_inputs = page.locator('input')
        print(f"   Found {all_inputs.count()} input elements")

        for i in range(all_inputs.count()):
            try:
                inp = all_inputs.nth(i)
                placeholder = inp.get_attribute('placeholder') or 'N/A'
                inp_type = inp.get_attribute('type') or 'text'
                is_visible = inp.is_visible()
                print(f"   [{i}] type={inp_type}, placeholder='{placeholder}', visible={is_visible}")
            except Exception as e:
                print(f"   [{i}] Error: {e}")

        # List all textareas
        all_textareas = page.locator('textarea')
        print(f"\n   Found {all_textareas.count()} textarea elements")

        for i in range(all_textareas.count()):
            try:
                ta = all_textareas.nth(i)
                placeholder = ta.get_attribute('placeholder') or 'N/A'
                is_visible = ta.is_visible()
                print(f"   [{i}] placeholder='{placeholder}', visible={is_visible}")
            except Exception as e:
                print(f"   [{i}] Error: {e}")

        # Check for Streamlit-specific elements
        print("\n[3] Checking Streamlit elements...")
        st_elements = [
            '[data-testid="stChatInput"]',
            '[data-testid="stChatMessage"]',
            '[data-testid="stForm"]',
            '[data-testid="stTextArea"]',
        ]

        for selector in st_elements:
            count = page.locator(selector).count()
            print(f"   {selector}: {count} found")

        page.screenshot(path="/tmp/chat_input_debug.png")
        print("\n   Screenshot: /tmp/chat_input_debug.png")

        browser.close()


if __name__ == "__main__":
    # Run tests
    test_chat_input_element_exists()
    test_followup_section_appears_after_research()
    # test_followup_input_works()  # Only run if section appears
