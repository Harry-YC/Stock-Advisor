#!/usr/bin/env python3
"""
Test runner for Travel Planner.

Usage:
    python tests/run_tests.py                    # Run all unit tests
    python tests/run_tests.py --e2e              # Run E2E tests (requires server)
    python tests/run_tests.py --all              # Run all tests
    python tests/run_tests.py --coverage         # Run with coverage report
    python tests/run_tests.py -k "test_name"     # Run specific test
"""

import subprocess
import sys
import os
from pathlib import Path

# Change to project root
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)


def run_unit_tests():
    """Run unit tests (no Playwright, no server required)."""
    print("\n" + "=" * 60)
    print("Running Unit Tests")
    print("=" * 60 + "\n")

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_file_upload_validation.py",
        "tests/test_excel_traveler_extraction.py",
        "tests/test_security.py",
        "tests/test_places_enrichment.py",
        "-v",
        "--tb=short",
        "-m", "not playwright",  # Exclude Playwright tests
    ]

    return subprocess.run(cmd).returncode


def run_e2e_tests():
    """Run Playwright E2E tests (requires running server)."""
    print("\n" + "=" * 60)
    print("Running E2E Tests (Playwright)")
    print("=" * 60 + "\n")

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "-m", "playwright",
    ]

    return subprocess.run(cmd).returncode


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running All Tests")
    print("=" * 60 + "\n")

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
    ]

    return subprocess.run(cmd).returncode


def run_with_coverage():
    """Run tests with coverage report."""
    print("\n" + "=" * 60)
    print("Running Tests with Coverage")
    print("=" * 60 + "\n")

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "-m", "not playwright",
        "--cov=services",
        "--cov=integrations",
        "--cov-report=term-missing",
        "--cov-report=html:coverage_html",
    ]

    return subprocess.run(cmd).returncode


def run_specific_test(pattern: str):
    """Run tests matching a pattern."""
    print(f"\n" + "=" * 60)
    print(f"Running Tests matching: {pattern}")
    print("=" * 60 + "\n")

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "-k", pattern,
    ]

    return subprocess.run(cmd).returncode


def print_help():
    """Print usage help."""
    print(__doc__)


def main():
    """Main entry point."""
    args = sys.argv[1:]

    if not args:
        # Default: run unit tests
        sys.exit(run_unit_tests())

    if "--help" in args or "-h" in args:
        print_help()
        sys.exit(0)

    if "--e2e" in args:
        sys.exit(run_e2e_tests())

    if "--all" in args:
        sys.exit(run_all_tests())

    if "--coverage" in args:
        sys.exit(run_with_coverage())

    if "-k" in args:
        idx = args.index("-k")
        if idx + 1 < len(args):
            pattern = args[idx + 1]
            sys.exit(run_specific_test(pattern))

    # Unknown args - pass through to pytest
    cmd = [sys.executable, "-m", "pytest", "tests/"] + args
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
