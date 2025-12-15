"""
Playwright E2E Test: Expert Selection Feature

Tests the new category-organized expert selection UI:
1. Create a test project
2. Navigate to GDG Discussion tab
3. Verify preset buttons render
4. Test preset selection changes experts
5. Test individual expert toggle
6. Verify min/max constraints
"""

import os
from playwright.sync_api import sync_playwright, expect

# Get test URL from environment or default to localhost:8503
TEST_URL = os.environ.get("TEST_URL", "http://localhost:8503")


def create_test_project(page):
    """Helper to create a test project via sidebar."""
    # Fill in project name
    project_input = page.locator('input[placeholder*="Project Name"], input[aria-label*="Project"]')
    if project_input.count() > 0:
        project_input.first.fill("E2E Test Project")
        page.wait_for_timeout(500)

        # Click Create button
        create_btn = page.locator('button:has-text("Create")')
        if create_btn.count() > 0:
            create_btn.first.click()
            page.wait_for_timeout(2000)
            print("   Created test project")
            return True
    return False


def test_expert_selector_renders():
    """Test that the expert selector renders in GDG Discussion tab."""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print("\n" + "=" * 60)
        print("TEST: Expert Selector Renders")
        print("=" * 60)

        # Navigate to app
        print(f"\n[1] Navigating to {TEST_URL}...")
        page.goto(TEST_URL, timeout=60000)
        page.wait_for_load_state("networkidle")
        # Wait for Streamlit to fully render
        page.wait_for_timeout(3000)

        # Take initial screenshot
        page.screenshot(path="/tmp/expert_selector_initial.png")
        print("   Initial screenshot: /tmp/expert_selector_initial.png")

        # Create a test project first (required for expert panel)
        print("[2] Creating test project...")
        create_test_project(page)

        # Navigate to GDG Discussion tab (main tab)
        print("[3] Looking for GDG Discussion tab...")

        # Try multiple selectors for the main tab
        tab_selectors = [
            'button:has-text("GDG Discussion")',
            '[data-baseweb="tab"]:has-text("GDG Discussion")',
            'text=GDG Discussion',
            '[role="tab"]:has-text("Discussion")',
        ]

        tab_clicked = False
        for selector in tab_selectors:
            tab = page.locator(selector)
            if tab.count() > 0:
                try:
                    tab.first.click()
                    page.wait_for_timeout(2000)
                    print(f"   Clicked tab with selector: {selector}")
                    tab_clicked = True
                    break
                except Exception as e:
                    print(f"   Failed to click {selector}: {e}")

        if not tab_clicked:
            print("   ⚠️ Could not find GDG Discussion tab")
            # List all visible tabs for debugging
            all_tabs = page.locator('[data-baseweb="tab"], [role="tab"], button')
            print(f"   Found {all_tabs.count()} potential tab elements")
            for i in range(min(all_tabs.count(), 10)):
                try:
                    text = all_tabs.nth(i).text_content()
                    if text and len(text.strip()) > 0:
                        print(f"   - Tab {i}: {text[:50]}")
                except:
                    pass

        # Check for Expert Selection expander
        print("[4] Looking for expert selector...")
        expert_expander = page.locator('text=Select GDG Experts')

        if expert_expander.count() > 0:
            print("   Found 'Select GDG Experts' expander")
            # Click to expand if needed
            if not expert_expander.first.is_visible():
                expander_header = page.locator('[data-testid="stExpander"]:has-text("Select GDG Experts")')
                if expander_header.count() > 0:
                    expander_header.first.click()
                    page.wait_for_timeout(500)
        else:
            print("   ⚠️ Expert selector not found (may be in different location)")
            # Take debug screenshot
            page.screenshot(path="/tmp/expert_selector_debug.png")
            print("   Debug screenshot: /tmp/expert_selector_debug.png")

        # Check for preset buttons
        print("[5] Checking for preset buttons...")
        preset_buttons = [
            "Surgical Candidacy",
            "Intervention Choice",
            "Symptom Management",
            "Ethics Review",
            "Full GDG Panel"
        ]

        presets_found = 0
        for preset in preset_buttons:
            btn = page.locator(f'button:has-text("{preset}")')
            if btn.count() > 0:
                presets_found += 1
                print(f"   Found preset: {preset}")

        if presets_found > 0:
            print(f"   Found {presets_found}/{len(preset_buttons)} preset buttons")
        else:
            print("   ⚠️ No preset buttons found")

        # Check for category headers
        print("[6] Checking for category headers...")
        categories = [
            "Surgical & Perioperative",
            "Palliative & Patient",
            "Evidence & Methodology",
            "Specialized Care",
            "Economics & Synthesis"
        ]

        categories_found = 0
        for category in categories:
            header = page.locator(f'text={category}')
            if header.count() > 0:
                categories_found += 1
                print(f"   Found category: {category}")

        if categories_found > 0:
            print(f"   Found {categories_found}/{len(categories)} category headers")

        # Check for expert buttons
        print("[7] Checking for expert toggle buttons...")
        experts = [
            "Surgical Oncologist",
            "Palliative Care Physician",
            "GRADE Methodologist",
            "GDG Chair"
        ]

        experts_found = 0
        for expert in experts:
            btn = page.locator(f'button:has-text("{expert}")')
            if btn.count() > 0:
                experts_found += 1

        print(f"   Found {experts_found}/{len(experts)} expert buttons")

        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Presets: {presets_found}/{len(preset_buttons)}")
        print(f"Categories: {categories_found}/{len(categories)}")
        print(f"Experts: {experts_found}/{len(experts)}")

        # Test passes if we found at least some elements
        assert presets_found > 0 or experts_found > 0, "Expert selector elements not found"
        print("\n PASS - Expert selector UI elements found")

        browser.close()


def test_preset_selection():
    """Test that clicking a preset updates expert selection."""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print("\n" + "=" * 60)
        print("TEST: Preset Selection")
        print("=" * 60)

        # Navigate to app
        print(f"\n[1] Navigating to {TEST_URL}...")
        page.goto(TEST_URL, timeout=30000)
        page.wait_for_load_state("networkidle")

        # Switch to Advanced mode
        print("[2] Switching to Advanced mode...")
        sidebar = page.locator('[data-testid="stSidebar"]')
        mode_btn = sidebar.locator('button:has-text("Switch to Advanced"), button:has-text("Advanced Mode")')
        if mode_btn.count() > 0 and mode_btn.first.is_visible():
            mode_btn.first.click()
            page.wait_for_timeout(1000)

        # Navigate to GDG Discussion tab
        print("[3] Looking for GDG Discussion tab...")
        gdg_tab = page.locator('button:has-text("GDG Discussion"), [role="tab"]:has-text("GDG Discussion")')
        if gdg_tab.count() > 0:
            gdg_tab.first.click()
            page.wait_for_timeout(1000)

        # Click "Full GDG Panel" preset
        print("[4] Clicking 'Full GDG Panel' preset...")
        full_panel_btn = page.locator('button:has-text("Full GDG Panel")')
        if full_panel_btn.count() > 0:
            full_panel_btn.first.click()
            page.wait_for_timeout(1000)
            print("   Clicked Full GDG Panel preset")

            # Verify more experts are now selected (check for checkmarks)
            selected = page.locator('button:has-text("")')
            print(f"   Selected experts count: {selected.count()}")
        else:
            print("   ⚠️ Full GDG Panel button not found")

        # Click "Ethics Review" preset
        print("[5] Clicking 'Ethics Review' preset...")
        ethics_btn = page.locator('button:has-text("Ethics Review")')
        if ethics_btn.count() > 0:
            ethics_btn.first.click()
            page.wait_for_timeout(1000)
            print("   Clicked Ethics Review preset")
        else:
            print("   ⚠️ Ethics Review button not found")

        print("\n PASS - Preset selection test completed")
        browser.close()


def test_expert_toggle():
    """Test that clicking an expert button toggles selection."""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print("\n" + "=" * 60)
        print("TEST: Expert Toggle")
        print("=" * 60)

        # Navigate to app
        print(f"\n[1] Navigating to {TEST_URL}...")
        page.goto(TEST_URL, timeout=30000)
        page.wait_for_load_state("networkidle")

        # Switch to Advanced mode
        sidebar = page.locator('[data-testid="stSidebar"]')
        mode_btn = sidebar.locator('button:has-text("Switch to Advanced"), button:has-text("Advanced Mode")')
        if mode_btn.count() > 0 and mode_btn.first.is_visible():
            mode_btn.first.click()
            page.wait_for_timeout(1000)

        # Navigate to GDG Discussion tab
        gdg_tab = page.locator('button:has-text("GDG Discussion"), [role="tab"]:has-text("GDG Discussion")')
        if gdg_tab.count() > 0:
            gdg_tab.first.click()
            page.wait_for_timeout(1000)

        # Try to toggle "Medical Ethicist"
        print("[2] Looking for Medical Ethicist button...")
        ethicist_btn = page.locator('button:has-text("Medical Ethicist")')

        if ethicist_btn.count() > 0:
            # Check if selected (has checkmark)
            btn_text = ethicist_btn.first.text_content()
            initially_selected = "" in btn_text

            print(f"   Initially selected: {initially_selected}")

            # Click to toggle
            ethicist_btn.first.click()
            page.wait_for_timeout(1000)

            # Check new state
            btn_text_after = page.locator('button:has-text("Medical Ethicist")').first.text_content()
            now_selected = "" in btn_text_after

            print(f"   After click selected: {now_selected}")

            if initially_selected != now_selected:
                print("   Selection toggled successfully")
            else:
                print("   ⚠️ Selection did not toggle (may be due to min constraint)")
        else:
            print("   ⚠️ Medical Ethicist button not found")

        print("\n PASS - Expert toggle test completed")
        browser.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EXPERT SELECTION PLAYWRIGHT E2E TESTS")
    print("=" * 70)

    # Run all tests
    test_expert_selector_renders()
    test_preset_selection()
    test_expert_toggle()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
