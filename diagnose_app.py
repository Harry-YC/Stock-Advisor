"""
Playwright diagnostic script for Travel Planner app.
Monitors the app, captures errors, and diagnoses stuck issues.
"""

import asyncio
import json
from datetime import datetime
from playwright.async_api import async_playwright, Page, ConsoleMessage

# Configuration
APP_URL = "https://travel-planner.up.railway.app"
TRIP_MESSAGE = "San Francisco to New Zealand for 10 days, parents (70s) and two sisters (30s), Feb-2026, make suggestions, will rent a car, budget 18000 USD"
TIMEOUT_MS = 180000  # 3 minutes total timeout
SCREENSHOT_DIR = "/Users/nelsonliu/Travel Planner/outputs/diagnostics"

# Collected data
console_logs = []
network_errors = []
api_calls = []
timestamps = {}


def log(msg: str):
    """Print timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}")


async def capture_console(msg: ConsoleMessage):
    """Capture browser console messages."""
    text = msg.text
    msg_type = msg.type
    console_logs.append({
        "time": datetime.now().isoformat(),
        "type": msg_type,
        "text": text[:500]  # Truncate long messages
    })
    if msg_type == "error":
        log(f"🔴 CONSOLE ERROR: {text[:200]}")
    elif "error" in text.lower() or "fail" in text.lower():
        log(f"⚠️  CONSOLE WARNING: {text[:200]}")


async def capture_request(request):
    """Capture API requests."""
    url = request.url
    if "api" in url or "gemini" in url.lower() or "generativelanguage" in url:
        api_calls.append({
            "time": datetime.now().isoformat(),
            "method": request.method,
            "url": url[:200],
            "status": "pending"
        })
        log(f"📤 API REQUEST: {request.method} {url[:80]}...")


async def capture_response(response):
    """Capture API responses."""
    url = response.url
    if "api" in url or "gemini" in url.lower() or "generativelanguage" in url:
        status = response.status
        # Update the pending request
        for call in reversed(api_calls):
            if call["url"] in url and call["status"] == "pending":
                call["status"] = status
                call["response_time"] = datetime.now().isoformat()
                break

        emoji = "✅" if status < 400 else "❌"
        log(f"{emoji} API RESPONSE: {status} {url[:80]}...")

        if status >= 400:
            network_errors.append({
                "time": datetime.now().isoformat(),
                "url": url,
                "status": status
            })


async def wait_for_message(page: Page, text_contains: str, timeout_ms: int = 30000) -> bool:
    """Wait for a message containing specific text to appear."""
    try:
        await page.wait_for_function(
            f"document.body.innerText.includes('{text_contains}')",
            timeout=timeout_ms
        )
        return True
    except Exception as e:
        log(f"⏱️  Timeout waiting for: '{text_contains}' - {e}")
        return False


async def take_screenshot(page: Page, name: str):
    """Take a screenshot with timestamp."""
    import os
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    path = f"{SCREENSHOT_DIR}/{ts}_{name}.png"
    await page.screenshot(path=path, full_page=True)
    log(f"📸 Screenshot saved: {path}")
    return path


async def run_diagnosis():
    """Run the diagnostic test."""
    log("=" * 60)
    log("🔍 TRAVEL PLANNER DIAGNOSTIC TEST")
    log("=" * 60)
    log(f"URL: {APP_URL}")
    log(f"Trip: {TRIP_MESSAGE[:50]}...")
    log("=" * 60)

    async with async_playwright() as p:
        # Launch browser (headless for CI, headed for local debugging)
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) PlaywrightDiagnostic/1.0"
        )
        page = await context.new_page()

        # Set up event listeners
        page.on("console", capture_console)
        page.on("request", capture_request)
        page.on("response", capture_response)

        try:
            # Step 1: Load the app
            log("\n📍 STEP 1: Loading app...")
            timestamps["load_start"] = datetime.now()

            response = await page.goto(APP_URL, timeout=60000, wait_until="networkidle")

            timestamps["load_end"] = datetime.now()
            load_time = (timestamps["load_end"] - timestamps["load_start"]).total_seconds()
            log(f"✅ App loaded in {load_time:.2f}s (status: {response.status})")

            await take_screenshot(page, "01_loaded")

            # Step 2: Wait for chat input
            log("\n📍 STEP 2: Waiting for chat interface...")

            # Wait for the input field to appear
            input_selector = 'textarea, input[type="text"], [data-testid="chat-input"], .cl-textarea'
            try:
                await page.wait_for_selector(input_selector, timeout=30000)
                log("✅ Chat input found")
            except Exception as e:
                log(f"❌ Chat input not found: {e}")
                await take_screenshot(page, "02_no_input")
                # Try to find any input-like element
                inputs = await page.query_selector_all("textarea, input")
                log(f"   Found {len(inputs)} input elements on page")
                raise

            await take_screenshot(page, "02_ready")

            # Step 3: Enter trip details
            log("\n📍 STEP 3: Entering trip details...")
            timestamps["input_start"] = datetime.now()

            # Find and fill the input
            input_el = await page.query_selector(input_selector)
            if input_el:
                await input_el.fill(TRIP_MESSAGE)
                log(f"✅ Entered trip message ({len(TRIP_MESSAGE)} chars)")
            else:
                log("❌ Could not find input element")
                raise Exception("Input element not found")

            await take_screenshot(page, "03_filled")

            # Step 4: Submit the message
            log("\n📍 STEP 4: Submitting message...")

            # Try multiple submit methods
            submitted = False

            # Method 1: Press Enter
            await input_el.press("Enter")
            await asyncio.sleep(1)

            # Check if message was sent (input should be cleared or new message appears)
            input_value = await input_el.input_value() if input_el else ""
            if not input_value or len(input_value) < len(TRIP_MESSAGE):
                submitted = True
                log("✅ Message submitted via Enter key")

            if not submitted:
                # Method 2: Click send button
                send_btn = await page.query_selector('button[type="submit"], button:has-text("Send"), .send-button')
                if send_btn:
                    await send_btn.click()
                    submitted = True
                    log("✅ Message submitted via Send button")

            timestamps["submit_time"] = datetime.now()
            await take_screenshot(page, "04_submitted")

            # Step 5: Wait for extraction (Trip Info Extractor)
            log("\n📍 STEP 5: Waiting for trip extraction...")

            extraction_found = await wait_for_message(page, "New Zealand", timeout_ms=60000)
            if extraction_found:
                log("✅ Trip extraction completed - destination recognized")
            else:
                log("⚠️  Trip extraction may have failed or is slow")

            await take_screenshot(page, "05_extracted")

            # Step 6: Look for confirmation and click "Plan my trip"
            log("\n📍 STEP 6: Looking for confirmation...")

            confirmation_found = await wait_for_message(page, "Does this look right", timeout_ms=30000)
            if confirmation_found:
                log("✅ Confirmation dialog appeared")
                await take_screenshot(page, "06_confirmation")

                # Click the "Yes, plan my trip!" button
                plan_btn = await page.query_selector('button:has-text("Yes, plan my trip")')
                if plan_btn:
                    log("📍 Clicking 'Yes, plan my trip!' button...")
                    timestamps["plan_start"] = datetime.now()
                    await plan_btn.click()
                    log("✅ Clicked plan button")
                else:
                    log("❌ Could not find 'Yes, plan my trip!' button")
                    # List all buttons
                    buttons = await page.query_selector_all("button")
                    for i, btn in enumerate(buttons):
                        text = await btn.inner_text()
                        log(f"   Button {i}: {text[:50]}")
            else:
                log("⚠️  No confirmation dialog - checking if planning started automatically")

            await take_screenshot(page, "07_plan_started")

            # Step 7: Monitor planning progress
            log("\n📍 STEP 7: Monitoring planning progress...")
            log("   Watching for expert responses (timeout: 3 min)...")

            experts_to_find = ["Booking", "Budget", "Logistics", "Activity"]
            experts_found = []

            start_time = datetime.now()
            check_interval = 5  # seconds
            max_wait = 180  # 3 minutes

            while (datetime.now() - start_time).total_seconds() < max_wait:
                await asyncio.sleep(check_interval)

                page_text = await page.inner_text("body")
                elapsed = (datetime.now() - start_time).total_seconds()

                # Check for expert responses
                for expert in experts_to_find:
                    if expert in page_text and expert not in experts_found:
                        experts_found.append(expert)
                        log(f"✅ [{elapsed:.0f}s] Found {expert} expert response!")
                        await take_screenshot(page, f"08_expert_{expert.lower()}")

                # Check for completion indicators
                if "Trip planning complete" in page_text:
                    log(f"🎉 [{elapsed:.0f}s] PLANNING COMPLETE!")
                    timestamps["plan_end"] = datetime.now()
                    break

                if "Export to Excel" in page_text or "Export to Word" in page_text:
                    log(f"🎉 [{elapsed:.0f}s] Export buttons found - planning complete!")
                    timestamps["plan_end"] = datetime.now()
                    break

                # Check for errors
                if "Error:" in page_text or "could not reach" in page_text.lower():
                    log(f"❌ [{elapsed:.0f}s] Error detected in page!")
                    await take_screenshot(page, "error_detected")

                # Check for "Consulting experts" (still in progress)
                if "Consulting" in page_text and "experts" in page_text:
                    log(f"⏳ [{elapsed:.0f}s] Still consulting experts... ({len(experts_found)}/4 found)")

                # Check for fallback message
                if "Switching to faster model" in page_text:
                    log(f"⚡ [{elapsed:.0f}s] Fallback to faster model triggered!")

            # Final screenshot
            await take_screenshot(page, "09_final")

            # Summary
            log("\n" + "=" * 60)
            log("📊 DIAGNOSTIC SUMMARY")
            log("=" * 60)

            if "plan_end" in timestamps and "plan_start" in timestamps:
                plan_time = (timestamps["plan_end"] - timestamps["plan_start"]).total_seconds()
                log(f"✅ Planning completed in {plan_time:.1f} seconds")
            else:
                log(f"⚠️  Planning did not complete within {max_wait}s timeout")

            log(f"   Experts found: {experts_found}")
            log(f"   Console errors: {len([l for l in console_logs if l['type'] == 'error'])}")
            log(f"   Network errors: {len(network_errors)}")
            log(f"   API calls made: {len(api_calls)}")

            if network_errors:
                log("\n🔴 NETWORK ERRORS:")
                for err in network_errors[-5:]:  # Last 5 errors
                    log(f"   {err['status']} - {err['url'][:60]}...")

            # Check for specific issues
            log("\n🔍 ISSUE ANALYSIS:")

            page_text = await page.inner_text("body")

            if "could not reach server" in page_text.lower():
                log("   ❌ ISSUE: API timeout - server unreachable")
                log("   💡 FIX: Check Gemini API key and quotas")

            if "rate limit" in page_text.lower():
                log("   ❌ ISSUE: Rate limiting detected")
                log("   💡 FIX: Reduce concurrent requests or add delays")

            if len(experts_found) < 4 and "plan_end" not in timestamps:
                log("   ❌ ISSUE: Not all experts responded")
                log(f"   💡 Missing: {[e for e in experts_to_find if e not in experts_found]}")

            if not network_errors and "plan_end" in timestamps:
                log("   ✅ No major issues detected - planning successful!")

        except Exception as e:
            log(f"\n❌ FATAL ERROR: {e}")
            await take_screenshot(page, "error_fatal")
            import traceback
            traceback.print_exc()

        finally:
            # Save diagnostic data
            import os
            os.makedirs(SCREENSHOT_DIR, exist_ok=True)

            diagnostic_data = {
                "timestamp": datetime.now().isoformat(),
                "url": APP_URL,
                "trip_message": TRIP_MESSAGE,
                "console_logs": console_logs[-50:],  # Last 50 logs
                "network_errors": network_errors,
                "api_calls": api_calls[-20:],  # Last 20 API calls
                "timestamps": {k: v.isoformat() if hasattr(v, 'isoformat') else v for k, v in timestamps.items()}
            }

            with open(f"{SCREENSHOT_DIR}/diagnostic_report.json", "w") as f:
                json.dump(diagnostic_data, f, indent=2)

            log(f"\n📁 Diagnostic data saved to: {SCREENSHOT_DIR}/")

            await browser.close()


if __name__ == "__main__":
    asyncio.run(run_diagnosis())
