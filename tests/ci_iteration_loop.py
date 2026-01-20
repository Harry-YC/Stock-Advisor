#!/usr/bin/env python3
"""
CI Iteration Loop - Autonomous Quality Improvement

Runs 5 iterations of:
1. Ask 3 different questions to the services
2. Evaluate outputs with Gemini 3 Pro
3. Get improvement feedback
4. Track quality scores

Usage: python3 tests/ci_iteration_loop.py
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')


# Test questions covering different scenarios
TEST_QUESTIONS = [
    {
        "id": "q1_institutional",
        "question": "What's the institutional activity and hedge fund positioning on NVDA?",
        "expected_elements": ["hedge fund", "institutional", "position", "buying", "selling"],
        "category": "institutional_flow",
    },
    {
        "id": "q2_options",
        "question": "Is there unusual options activity on AAPL? What's the put/call sentiment?",
        "expected_elements": ["options", "calls", "puts", "flow", "sentiment"],
        "category": "options_sentiment",
    },
    {
        "id": "q3_synthesis",
        "question": "What are the bull and bear cases for TSLA according to finance KOLs?",
        "expected_elements": ["bull", "bear", "KOL", "@", "argument"],
        "category": "synthesis",
    },
]


def get_llm_client():
    """Get Gemini client for evaluation."""
    try:
        from core.llm_utils import get_llm_client as get_client
        return get_client(model="gemini-3-pro-preview")
    except Exception as e:
        print(f"Failed to get LLM client: {e}")
        return None


def test_grok_service(question: dict, use_mock: bool = True) -> dict:
    """Test Grok service with a question."""
    result = {
        "question_id": question["id"],
        "question": question["question"],
        "category": question["category"],
        "output": "",
        "error": None,
        "latency_ms": 0,
        "has_expected_elements": False,
    }

    try:
        from services.grok_service import GrokService, detect_stock_ci_dimensions

        # Always test CI dimension detection (works without API)
        dims = detect_stock_ci_dimensions(question["question"])
        result["detected_dimensions"] = dims

        # Check if real API available
        real_api_available = bool(os.getenv("XAI_API_KEY"))

        # Use mock mode if no API key and use_mock is True
        if not real_api_available and use_mock:
            service = GrokService(mock_mode=True)
            result["mock_mode"] = True
        else:
            service = GrokService()
            result["mock_mode"] = False

        if not service.is_available():
            result["error"] = "Service unavailable"
            result["output"] = f"SKIPPED: Service unavailable. Detected CI dimensions: {dims}"
            result["grok_available"] = False
            return result

        result["grok_available"] = True
        start = time.time()

        # Use appropriate method based on category
        if question["category"] == "synthesis":
            # Extract ticker from question or default to TSLA
            output = service.synthesize_kol_views("TSLA", ["macro", "tech_growth", "media"])
        elif question["category"] == "institutional_flow":
            output = service.deep_stock_research("NVDA", question["question"])
        elif question["category"] == "options_sentiment":
            output = service.competitive_intelligence_search("options_sentiment", "AAPL")
        else:
            output = service.get_kol_insights(question["question"])

        result["latency_ms"] = int((time.time() - start) * 1000)
        result["output"] = output[:3000] if output else "Empty response"

        # Check for expected elements
        output_lower = output.lower() if output else ""
        found = sum(1 for elem in question["expected_elements"] if elem.lower() in output_lower)
        result["has_expected_elements"] = found >= len(question["expected_elements"]) // 2
        result["elements_found"] = found
        result["elements_total"] = len(question["expected_elements"])

    except Exception as e:
        result["error"] = str(e)[:200]
        result["output"] = f"ERROR: {str(e)[:200]}"

    return result


def test_stock_data_service(symbol: str = "NVDA") -> dict:
    """Test stock data service."""
    result = {
        "service": "stock_data",
        "symbol": symbol,
        "output": "",
        "error": None,
        "data_available": {},
    }

    try:
        from services.stock_data_service import fetch_stock_data, build_expert_context

        # Test fetch_stock_data
        ctx = fetch_stock_data(symbol, include_quote=True, include_financials=True)
        result["data_available"] = ctx.data_available

        # Test build_expert_context
        context = build_expert_context(
            symbol=symbol,
            question=f"Analyze {symbol}",
            include_grok=False,  # Skip Grok to test other components
        )
        result["output"] = context[:1500] if context else "Empty"
        result["context_length"] = len(context) if context else 0

    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def test_market_search(symbol: str = "NVDA") -> dict:
    """Test market search with Google grounding."""
    result = {
        "service": "market_search",
        "symbol": symbol,
        "output": "",
        "error": None,
    }

    try:
        from integrations.market_search import MarketSearchClient

        client = MarketSearchClient()
        if not client.is_available():
            result["error"] = "Market search not available"
            return result

        search_result = client.search_stock_news(symbol)
        if search_result and search_result.content:
            result["output"] = search_result.content[:1500]
            result["has_sources"] = bool(search_result.sources)
            result["source_count"] = len(search_result.sources) if search_result.sources else 0
        else:
            result["output"] = "No results"

    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def test_kol_profiles() -> dict:
    """Test KOL profile data quality."""
    result = {
        "service": "kol_profiles",
        "output": "",
        "error": None,
    }

    try:
        from services.grok_service import KOL_PROFILES, KOL_CATEGORIES, HIGH_SIGNAL_KOLS

        # Check profile completeness
        total_kols = len(KOL_PROFILES)
        complete_profiles = 0
        required_fields = ["handle", "category", "bias", "specialty", "credibility"]

        for name, profile in KOL_PROFILES.items():
            if all(field in profile for field in required_fields):
                complete_profiles += 1

        result["total_kols"] = total_kols
        result["complete_profiles"] = complete_profiles
        result["categories"] = list(KOL_CATEGORIES.keys())
        result["high_signal_topics"] = list(HIGH_SIGNAL_KOLS.keys())

        result["output"] = (
            f"KOL Profiles: {total_kols} total, {complete_profiles} complete\n"
            f"Categories: {len(KOL_CATEGORIES)}\n"
            f"High-signal topics: {len(HIGH_SIGNAL_KOLS)}"
        )

    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def evaluate_with_gemini(test_results: list, iteration: int) -> dict:
    """Evaluate test results with Gemini 3 Pro."""
    client = get_llm_client()
    if not client:
        return {"score": 5, "feedback": "Could not connect to Gemini", "improvements": []}

    # Build evaluation prompt
    results_summary = []
    for r in test_results:
        if "question" in r:
            results_summary.append(f"""
**Question**: {r['question']}
**Category**: {r.get('category', 'N/A')}
**Latency**: {r.get('latency_ms', 'N/A')}ms
**Expected Elements Found**: {r.get('elements_found', 0)}/{r.get('elements_total', 0)}
**Error**: {r.get('error', 'None')}
**Output Sample** (first 800 chars):
{r.get('output', 'N/A')[:800]}
""")
        else:
            results_summary.append(f"""
**Service**: {r.get('service', 'unknown')}
**Data Available**: {r.get('data_available', {})}
**Error**: {r.get('error', 'None')}
""")

    prompt = f"""You are a QA engineer evaluating a Stock Advisor app's Grok/X Twitter integration.

This is iteration {iteration}/5 of automated testing.

## Test Results:
{''.join(results_summary)}

## Evaluation Criteria:
1. **Response Quality** (0-10): Are the KOL insights specific, with @handles, quotes, dates?
2. **Coverage** (0-10): Does it cover bulls AND bears, multiple perspectives?
3. **Actionability** (0-10): Is the information useful for trading decisions?
4. **Technical** (0-10): Reasonable latency, no errors, expected elements found?

## Your Task:
1. Score overall quality (1-10)
2. List specific issues found
3. Suggest 2-3 concrete code improvements

Respond in JSON format:
{{
    "score": <1-10>,
    "quality_breakdown": {{
        "response_quality": <0-10>,
        "coverage": <0-10>,
        "actionability": <0-10>,
        "technical": <0-10>
    }},
    "issues": ["issue1", "issue2", ...],
    "improvements": [
        {{"file": "path/to/file.py", "change": "description of change", "priority": "high/medium/low"}},
        ...
    ],
    "feedback": "Overall assessment paragraph"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gemini-3-pro-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3,
        )

        content = response.choices[0].message.content

        # Try to parse JSON
        # Find JSON block
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0]
        else:
            json_str = content

        return json.loads(json_str.strip())

    except json.JSONDecodeError as e:
        return {
            "score": 5,
            "feedback": content if 'content' in dir() else "JSON parse error",
            "improvements": [],
            "issues": [f"JSON parse error: {str(e)[:50]}"],
        }
    except Exception as e:
        return {
            "score": 5,
            "feedback": f"Evaluation failed: {str(e)[:100]}",
            "improvements": [],
            "issues": [str(e)[:100]],
        }


def run_iteration(iteration: int) -> dict:
    """Run one iteration of the CI loop."""
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration}/5")
    print(f"{'='*60}")

    results = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "test_results": [],
        "evaluation": {},
    }

    # Run test questions
    print("\n--- Running Grok Service Tests ---")
    xai_available = bool(os.getenv("XAI_API_KEY"))
    mode_str = "LIVE API" if xai_available else "MOCK MODE"
    print(f"  Mode: {mode_str}")

    for q in TEST_QUESTIONS:
        print(f"  Testing: {q['id']}...")
        test_result = test_grok_service(q, use_mock=True)
        results["test_results"].append(test_result)

        if test_result.get("grok_available"):
            status = "OK" if not test_result.get("error") else "FAIL"
            elements = f"{test_result.get('elements_found', 0)}/{test_result.get('elements_total', 0)}"
            latency = test_result.get("latency_ms", "N/A")
            mock_tag = " [MOCK]" if test_result.get("mock_mode") else ""
            print(f"    {status}: {elements} elements, {latency}ms{mock_tag}")
        else:
            dims = test_result.get("detected_dimensions", [])
            print(f"    SKIP. Detected dims: {dims}")

    # Run stock data service test
    print("\n--- Running Stock Data Service Test ---")
    stock_result = test_stock_data_service("NVDA")
    results["test_results"].append(stock_result)
    print(f"  Data available: {stock_result.get('data_available', {})}")

    # Run market search test
    print("\n--- Running Market Search Test ---")
    market_result = test_market_search("NVDA")
    results["test_results"].append(market_result)
    if market_result.get("error"):
        print(f"  Error: {market_result['error'][:60]}")
    else:
        print(f"  Sources: {market_result.get('source_count', 0)}")

    # Run KOL profiles test
    print("\n--- Running KOL Profiles Test ---")
    kol_result = test_kol_profiles()
    results["test_results"].append(kol_result)
    print(f"  {kol_result.get('output', 'N/A')}")

    # Evaluate with Gemini
    print("\n--- Evaluating with Gemini 3 Pro ---")
    evaluation = evaluate_with_gemini(results["test_results"], iteration)
    results["evaluation"] = evaluation

    score = evaluation.get("score", 0)
    print(f"  Score: {score}/10")

    if evaluation.get("quality_breakdown"):
        qb = evaluation["quality_breakdown"]
        print(f"  Breakdown: Quality={qb.get('response_quality', 'N/A')}, "
              f"Coverage={qb.get('coverage', 'N/A')}, "
              f"Actionability={qb.get('actionability', 'N/A')}, "
              f"Technical={qb.get('technical', 'N/A')}")

    if evaluation.get("issues"):
        print(f"  Issues: {len(evaluation['issues'])}")
        for issue in evaluation["issues"][:3]:
            print(f"    - {issue[:80]}")

    if evaluation.get("improvements"):
        print(f"  Improvements suggested: {len(evaluation['improvements'])}")
        for imp in evaluation["improvements"][:2]:
            print(f"    - [{imp.get('priority', 'N/A')}] {imp.get('file', 'N/A')}: {imp.get('change', 'N/A')[:60]}")

    return results


def save_results(all_results: list):
    """Save all iteration results."""
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"ci_iterations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    """Run 5 iterations of CI testing."""
    print("="*60)
    print("CI ITERATION LOOP - Stock Advisor Quality Testing")
    print("="*60)
    print(f"Started: {datetime.now().isoformat()}")

    # Check environment
    xai_key = os.getenv("XAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    print(f"\nEnvironment:")
    print(f"  XAI_API_KEY: {'Set' if xai_key else 'NOT SET (Grok tests will skip)'}")
    print(f"  GEMINI_API_KEY: {'Set' if gemini_key else 'NOT SET (evaluation will fail)'}")

    if not gemini_key:
        print("\nERROR: GEMINI_API_KEY required for evaluation. Exiting.")
        sys.exit(1)

    all_results = []
    scores = []

    # Run 5 iterations
    for i in range(1, 6):
        try:
            result = run_iteration(i)
            all_results.append(result)

            score = result.get("evaluation", {}).get("score", 0)
            scores.append(score)

            # Brief pause between iterations
            if i < 5:
                print(f"\n  Waiting 3s before next iteration...")
                time.sleep(3)

        except Exception as e:
            print(f"\nERROR in iteration {i}: {e}")
            all_results.append({
                "iteration": i,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })

    # Final Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    print(f"\nScores by iteration: {scores}")
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"Average score: {avg_score:.1f}/10")
        print(f"Best score: {max(scores)}/10")
        print(f"Trend: {'Improving' if len(scores) > 1 and scores[-1] > scores[0] else 'Stable/Declining'}")

    # Aggregate improvements
    all_improvements = []
    for r in all_results:
        if r.get("evaluation", {}).get("improvements"):
            all_improvements.extend(r["evaluation"]["improvements"])

    if all_improvements:
        print(f"\nTop Improvements Suggested ({len(all_improvements)} total):")
        # Count by file
        by_file = {}
        for imp in all_improvements:
            f = imp.get("file", "unknown")
            if f not in by_file:
                by_file[f] = []
            by_file[f].append(imp.get("change", ""))

        for f, changes in list(by_file.items())[:5]:
            print(f"  {f}: {len(changes)} suggestions")

    # Save results
    output_file = save_results(all_results)

    print(f"\nCompleted: {datetime.now().isoformat()}")

    return all_results


if __name__ == "__main__":
    main()
