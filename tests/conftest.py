"""
Pytest configuration and fixtures for Stock Advisor Playwright tests.
"""

import os
import pytest
import subprocess
import time
import socket
from playwright.sync_api import Playwright

# Test configuration
TEST_URL = os.getenv("TEST_URL", "http://localhost:8501")
APP_STARTUP_TIMEOUT = 30  # seconds
APP_COMMAND = ["chainlit", "run", "app_sa.py", "--port", "8501", "--headless"]


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


@pytest.fixture(scope="session")
def app_server():
    """
    Start the Chainlit app server for testing.

    This fixture starts the server once per test session and stops it after.
    Set TEST_URL env var to skip server startup (for testing against running server).
    """
    # If TEST_URL is explicitly set, assume server is already running
    if os.getenv("TEST_URL"):
        yield TEST_URL
        return

    # Check if port is already in use
    if is_port_in_use(8501):
        print("Port 8501 already in use, using existing server")
        yield TEST_URL
        return

    # Start the server
    print("Starting Chainlit server...")
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    process = subprocess.Popen(
        APP_COMMAND,
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )

    # Wait for server to start
    start_time = time.time()
    while time.time() - start_time < APP_STARTUP_TIMEOUT:
        if is_port_in_use(8501):
            print("Server started successfully")
            time.sleep(2)  # Extra time for full initialization
            break
        time.sleep(0.5)
    else:
        process.terminate()
        stdout, stderr = process.communicate()
        raise RuntimeError(
            f"Server failed to start within {APP_STARTUP_TIMEOUT}s.\n"
            f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )

    yield TEST_URL

    # Cleanup
    print("Stopping Chainlit server...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.fixture(scope="function")
def page(playwright: Playwright, app_server: str):
    """
    Create a new browser page for each test.

    Uses Chromium in headless mode by default.
    """
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(
        viewport={"width": 1280, "height": 800},
        locale="en-US",
    )
    page = context.new_page()

    # Navigate to app
    page.goto(app_server)

    yield page

    # Cleanup
    context.close()
    browser.close()


@pytest.fixture(scope="function")
def page_with_tracing(playwright: Playwright, app_server: str, request):
    """
    Create a browser page with tracing enabled for debugging.

    Saves trace on test failure.
    """
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(
        viewport={"width": 1280, "height": 800},
        locale="en-US",
    )

    # Start tracing
    context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    page.goto(app_server)

    yield page

    # Save trace on failure
    if request.node.rep_call.failed if hasattr(request.node, 'rep_call') else False:
        trace_path = f"traces/{request.node.name}.zip"
        os.makedirs("traces", exist_ok=True)
        context.tracing.stop(path=trace_path)
        print(f"Trace saved to {trace_path}")
    else:
        context.tracing.stop()

    context.close()
    browser.close()


# Pytest hooks for better test reporting
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Store test result for trace saving."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


# Common test data
TEST_TICKERS = {
    "large_cap": ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN"],
    "mid_cap": ["CRWD", "DDOG", "NET", "ZS"],
    "small_cap": ["ONDS", "QUBT"],  # For Alpha Vantage fallback testing
}

EXPERT_NAMES = [
    "Bull Analyst",
    "Bear Analyst",
    "Technical Analyst",
    "Fundamental Analyst",
    "Sentiment Analyst",
    "Risk Manager",
]

PRESETS = [
    "Quick Analysis",
    "Deep Dive",
    "KOL Review",
    "Trade Planning",
    "Full Panel",
    "Expert Debate",
]
