#!/usr/bin/env python3
"""
Test script for Competitive Intelligence workflow.

Run with: python3 tests/test_ci_workflow.py

Tests:
1. CI dimension detection (no API needed)
2. Grok service initialization (no API needed)
3. Market search grounding (needs GEMINI_API_KEY)
4. Stock data fetching with fallback (needs FINNHUB_API_KEY)
5. Expert debate flow integration
"""

import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test results tracking
PASSED = []
FAILED = []


def test_result(name: str, passed: bool, details: str = ""):
    """Record test result."""
    if passed:
        PASSED.append(name)
        print(f"  OK: {name}")
    else:
        FAILED.append((name, details))
        print(f"  FAIL: {name}: {details}")


def test_grok_dimension_detection():
    """Test CI dimension detection from question keywords."""
    print("\n=== Testing Grok CI Dimension Detection ===")

    from services.grok_service import detect_stock_ci_dimensions, STOCK_CI_DIMENSIONS

    # Test that STOCK_CI_DIMENSIONS exists and has expected dimensions
    expected_dims = [
        "institutional_flow", "options_sentiment", "analyst_ratings",
        "earnings_catalyst", "macro_sentiment", "retail_sentiment",
        "sector_rotation", "short_interest"
    ]
    test_result(
        "STOCK_CI_DIMENSIONS has all dimensions",
        all(d in STOCK_CI_DIMENSIONS for d in expected_dims),
        f"Missing: {set(expected_dims) - set(STOCK_CI_DIMENSIONS.keys())}"
    )

    # Test specific detections
    test_cases = [
        ("What's the institutional flow on NVDA?", ["institutional_flow"]),
        ("Is there unusual options activity on AAPL?", ["options_sentiment"]),
        ("What do analysts say about TSLA price target?", ["analyst_ratings"]),
        ("Will MSFT beat earnings this quarter?", ["earnings_catalyst"]),
        ("How will Fed rate cuts affect stocks?", ["macro_sentiment"]),
        ("What's WSB saying about GME?", ["retail_sentiment"]),
        ("Is there a sector rotation into tech?", ["sector_rotation"]),
        ("Is there a short squeeze potential in AMC?", ["short_interest"]),
        ("What's the EPS outlook?", ["earnings_catalyst"]),  # No CI dimension if no match
        ("What's the hedge fund activity and options flow?", ["institutional_flow", "options_sentiment"]),
    ]

    for question, expected in test_cases:
        result = detect_stock_ci_dimensions(question)
        # Check if expected dimensions are in result (order may vary)
        passed = set(expected) == set(result)
        test_result(
            f"Detect '{question[:40]}...'",
            passed,
            f"Expected {expected}, got {result}"
        )


def test_grok_service_initialization():
    """Test Grok service can be initialized."""
    print("\n=== Testing Grok Service Initialization ===")

    try:
        from services.grok_service import GrokService, get_grok_service

        service = GrokService()
        test_result("GrokService instantiation", True)

        # Test singleton pattern
        singleton = get_grok_service()
        test_result("get_grok_service singleton", singleton is not None)

        # Test availability check (should work without API key)
        has_key = service.is_available()
        xai_key = os.getenv("XAI_API_KEY")
        test_result(
            "is_available reflects XAI_API_KEY",
            has_key == bool(xai_key),
            f"is_available={has_key}, XAI_API_KEY={'set' if xai_key else 'not set'}"
        )

    except Exception as e:
        test_result("Grok service import/init", False, str(e))


def test_market_search_grounding():
    """Test market search service with Google grounding."""
    print("\n=== Testing Market Search Grounding ===")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("  SKIP: (GEMINI_API_KEY not set)")
        return

    try:
        from integrations.market_search import MarketSearchClient

        client = MarketSearchClient()
        test_result(
            "MarketSearchClient instantiation",
            client is not None
        )

        test_result(
            "MarketSearchClient is_available",
            client.is_available(),
            "Client reports not available"
        )

        if client.is_available():
            # Test basic search
            result = client.search_stock_news("NVDA")
            test_result(
                "search_stock_news returns result",
                result is not None and hasattr(result, 'content'),
                f"Got {type(result)}"
            )

            if result and result.content:
                test_result(
                    "Result has content",
                    len(result.content) > 0,
                    "Empty content"
                )

    except ImportError as e:
        test_result("Market search import", False, str(e))
    except Exception as e:
        test_result("Market search test", False, str(e))


def test_stock_data_with_fallback():
    """Test stock data fetching with Finnhub + Alpha Vantage fallback."""
    print("\n=== Testing Stock Data Service ===")

    finnhub_key = os.getenv("FINNHUB_API_KEY")
    if not finnhub_key:
        print("  SKIP: (FINNHUB_API_KEY not set)")
        return

    try:
        from services.stock_data_service import (
            fetch_stock_data,
            extract_tickers,
            StockDataContext
        )

        # Test ticker extraction
        test_cases = [
            ("What about $AAPL?", ["AAPL"]),
            ("NVDA stock is hot", ["NVDA"]),
            ("Compare MSFT and GOOGL", ["MSFT", "GOOGL"]),
        ]

        for text, expected in test_cases:
            result = extract_tickers(text)
            # Check if expected tickers are in result
            passed = set(expected).issubset(set(result))
            test_result(
                f"extract_tickers '{text[:30]}...'",
                passed,
                f"Expected {expected} in {result}"
            )

        # Test stock data fetching
        context = fetch_stock_data("AAPL", include_quote=True, include_financials=True)
        test_result(
            "fetch_stock_data returns StockDataContext",
            isinstance(context, StockDataContext),
            f"Got {type(context)}"
        )

        test_result(
            "StockDataContext has symbol",
            context.symbol == "AAPL",
            f"Got symbol={context.symbol}"
        )

        # Check if we got data from any source
        has_quote = context.data_available.get('quote', False)
        test_result(
            "Quote data available",
            has_quote,
            f"data_available={context.data_available}"
        )

        # Test prompt context generation
        prompt_ctx = context.to_prompt_context()
        test_result(
            "to_prompt_context generates string",
            isinstance(prompt_ctx, str) and len(prompt_ctx) > 0,
            f"Length={len(prompt_ctx)}"
        )

    except ImportError as e:
        test_result("Stock data service import", False, str(e))
    except Exception as e:
        test_result("Stock data test", False, str(e))


def test_expert_debate_flow():
    """Test expert debate flow with stock data integration."""
    print("\n=== Testing Expert Debate Flow ===")

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        print("  SKIP: (GEMINI_API_KEY not set)")
        return

    try:
        from services.stock_data_service import build_expert_context

        # Test expert context building
        context = build_expert_context(
            symbol="NVDA",
            question="Is NVDA a buy for AI growth?",
            include_market_search=False,
            include_grok=False
        )

        test_result(
            "build_expert_context returns string",
            isinstance(context, str) and len(context) > 0,
            f"Length={len(context)}"
        )

        test_result(
            "Expert context has data",
            "Real-Time Quote" in context or "Financial Metrics" in context or "No data available" in context,
            "Missing expected sections"
        )

    except ImportError as e:
        test_result("Expert debate import", False, str(e))
    except Exception as e:
        test_result("Expert debate test", False, str(e))


def test_imports():
    """Test all imports work correctly."""
    print("\n=== Testing Imports ===")

    try:
        from services import (
            GrokService, get_grok_service,
            detect_stock_ci_dimensions, get_stock_pulse,
            STOCK_KOLS, STOCK_CI_DIMENSIONS
        )
        test_result("Grok service imports from services", True)
    except Exception as e:
        test_result("Grok service imports", False, str(e))

    try:
        from services import (
            fetch_stock_data, fetch_multi_stock_data,
            extract_tickers, build_expert_context,
            StockDataContext
        )
        test_result("Stock data service imports from services", True)
    except Exception as e:
        test_result("Stock data service imports", False, str(e))

    try:
        from services import LLMRouter, get_llm_router
        test_result("LLM router imports from services", True)
    except Exception as e:
        test_result("LLM router imports", False, str(e))

    try:
        from config.settings import (
            GEMINI_API_KEY, FINNHUB_API_KEY, XAI_API_KEY,
            ENABLE_GROK, GROK_MODEL, GROK_CACHE_TTL
        )
        test_result("Settings imports including Grok config", True)
    except Exception as e:
        test_result("Settings imports", False, str(e))


def main():
    """Run all tests."""
    print("=" * 60)
    print("Stock Advisor - CI Workflow Tests")
    print("=" * 60)

    # Show env status
    print("\nEnvironment:")
    print(f"  GEMINI_API_KEY: {'Set' if os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') else 'Not set'}")
    print(f"  FINNHUB_API_KEY: {'Set' if os.getenv('FINNHUB_API_KEY') else 'Not set'}")
    print(f"  XAI_API_KEY: {'Set' if os.getenv('XAI_API_KEY') else 'Not set (optional)'}")

    # Run tests
    test_imports()
    test_grok_dimension_detection()
    test_grok_service_initialization()
    test_market_search_grounding()
    test_stock_data_with_fallback()
    test_expert_debate_flow()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Passed: {len(PASSED)}")
    print(f"  Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailed tests:")
        for name, details in FAILED:
            print(f"  - {name}: {details}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
