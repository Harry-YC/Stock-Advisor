#!/usr/bin/env python3
"""
Autonomous CI Improvement Loop for Stock Advisor

Runs 5 iterations of:
1. Ask different stock questions
2. Generate answers with Grok KOL insights
3. Synthesize and evaluate with Gemini 3 Pro
4. Apply code improvements based on feedback
5. Commit changes to git
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Stock questions for each iteration - diverse topics
ITERATION_QUESTIONS = [
    # Iteration 1: Tech stocks
    [
        {"symbol": "NVDA", "question": "What's the institutional sentiment on NVDA? Are hedge funds buying or selling?"},
        {"symbol": "AAPL", "question": "What do finance KOLs think about Apple's AI strategy?"},
        {"symbol": "MSFT", "question": "Is Microsoft overvalued at current levels?"},
    ],
    # Iteration 2: Semiconductor & AI
    [
        {"symbol": "TSM", "question": "What's the geopolitical risk for TSM and how are KOLs positioning?"},
        {"symbol": "AMD", "question": "AMD vs NVDA - which do retail traders prefer on X?"},
        {"symbol": "AVGO", "question": "What's the options flow sentiment on Broadcom?"},
    ],
    # Iteration 3: EV & Energy
    [
        {"symbol": "TSLA", "question": "What are the bull and bear cases for Tesla according to X KOLs?"},
        {"symbol": "RIVN", "question": "Is Rivian a buy at current prices? What do analysts say?"},
        {"symbol": "ENPH", "question": "What's the short interest situation on Enphase?"},
    ],
    # Iteration 4: Finance & Healthcare
    [
        {"symbol": "JPM", "question": "How are banks positioned for interest rate changes?"},
        {"symbol": "UNH", "question": "What's the sentiment on healthcare stocks after recent news?"},
        {"symbol": "V", "question": "Is Visa a defensive play in current market conditions?"},
    ],
    # Iteration 5: Mixed portfolio
    [
        {"symbol": "GOOGL", "question": "What do KOLs think about Google's AI competition with OpenAI?"},
        {"symbol": "AMZN", "question": "Is Amazon's cloud growth sustainable? What's the X sentiment?"},
        {"symbol": "META", "question": "Meta's metaverse pivot - are KOLs bullish or bearish?"},
    ],
]


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run_grok_query(symbol: str, question: str) -> dict:
    """Run a Grok query for KOL insights."""
    try:
        from services.grok_service import get_grok_service, detect_stock_ci_dimensions

        service = get_grok_service()
        if not service.is_available():
            return {"error": "Grok not available", "content": ""}

        # Detect CI dimensions
        dimensions = detect_stock_ci_dimensions(question)

        start = time.time()

        # Get KOL insights
        if "bull" in question.lower() and "bear" in question.lower():
            result = service.synthesize_kol_views(symbol)
        elif dimensions:
            result = service.competitive_intelligence_search(dimensions[0], symbol)
        else:
            # get_stock_sentiment only takes symbol parameter
            result = service.get_stock_sentiment(symbol)

        latency = int((time.time() - start) * 1000)

        return {
            "symbol": symbol,
            "question": question,
            "dimensions": dimensions,
            "content": result[:2000] if result else "",
            "latency_ms": latency,
            "success": bool(result),
        }
    except Exception as e:
        return {"error": str(e), "content": "", "success": False}


def run_market_search(symbol: str) -> dict:
    """Run market search for real-time news."""
    try:
        from integrations.market_search import MarketSearchClient

        client = MarketSearchClient()
        if not client.is_available():
            return {"error": "Market search not available", "sources": 0}

        result = client.search_stock_news(symbol)
        return {
            "symbol": symbol,
            "content": result.content[:1000] if result.content else "",
            "sources": len(result.sources),
            "sentiment": result.sentiment,
            "success": bool(result.content),
        }
    except Exception as e:
        return {"error": str(e), "sources": 0, "success": False}


def synthesize_insights(grok_results: list, market_results: list) -> str:
    """Synthesize all gathered insights."""
    synthesis = "## Gathered Intelligence\n\n"

    for g in grok_results:
        if g.get("success"):
            synthesis += f"### {g['symbol']} - KOL Insights\n"
            synthesis += f"Dimensions: {g.get('dimensions', [])}\n"
            synthesis += f"Latency: {g.get('latency_ms', 0)}ms\n"
            synthesis += f"{g.get('content', '')[:500]}\n\n"

    for m in market_results:
        if m.get("success"):
            synthesis += f"### {m['symbol']} - Market News\n"
            synthesis += f"Sources: {m.get('sources', 0)}, Sentiment: {m.get('sentiment', 'unknown')}\n"
            synthesis += f"{m.get('content', '')[:300]}\n\n"

    return synthesis


def evaluate_with_gemini(synthesis: str, iteration: int, code_context: str) -> dict:
    """Get Gemini 3 Pro to evaluate and suggest improvements."""
    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        prompt = f"""You are a senior software engineer reviewing a Stock Advisor application.

## Current Iteration: {iteration}/5

## Test Results (Synthesized Intelligence):
{synthesis}

## Current Code Context:
{code_context}

## Your Task:
1. Score the quality of responses (1-10)
2. Identify 1-2 specific, small code improvements
3. Provide EXACT code changes that can be applied with string replacement

CRITICAL: Return ONLY valid JSON, no markdown, no explanation. Example:
{{"score": 6, "quality_breakdown": {{"kol_insights": 7, "market_data": 6, "synthesis": 5, "response_time": 6}}, "issues": [{{"severity": "medium", "description": "Missing error handling", "file": "services/grok_service.py"}}], "improvements": [{{"file": "services/grok_service.py", "description": "Add timeout", "old_code": "timeout=60", "new_code": "timeout=90"}}], "summary": "Good but needs improvements"}}

Rules for improvements:
- old_code must be EXACT text that exists in the file (copy-paste from code context)
- new_code must be valid Python
- Keep changes small (1-3 lines)
- Focus on: error handling, timeouts, cache TTLs, logging

Return only JSON:"""

        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=4000,
            ),
        )

        # Parse JSON from response
        text = response.text
        # Find JSON in response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())

    except json.JSONDecodeError as e:
        log(f"JSON parse error: {e}")
        # Try to extract score from raw text
        try:
            import re
            score_match = re.search(r'"score":\s*(\d+)', text)
            score = int(score_match.group(1)) if score_match else 5
            return {"score": score, "issues": [], "improvements": [], "summary": "JSON parse error - using extracted score"}
        except:
            return {"score": 5, "issues": [], "improvements": [], "summary": "Parse error"}
    except Exception as e:
        log(f"Gemini evaluation error: {e}")
        return {"score": 5, "issues": [], "improvements": [], "summary": str(e)}


def apply_improvement(improvement: dict) -> bool:
    """Apply a single code improvement."""
    try:
        file_path = Path(__file__).parent.parent / improvement["file"]

        if not file_path.exists():
            log(f"  File not found: {improvement['file']}")
            return False

        content = file_path.read_text()
        old_code = improvement.get("old_code", "")
        new_code = improvement.get("new_code", "")

        if not old_code or not new_code:
            log(f"  Missing old_code or new_code")
            return False

        if old_code not in content:
            log(f"  Old code not found in {improvement['file']}")
            return False

        # Apply the change
        new_content = content.replace(old_code, new_code, 1)
        file_path.write_text(new_content)

        # Verify syntax
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(file_path)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            # Revert on syntax error
            file_path.write_text(content)
            log(f"  Syntax error, reverted: {result.stderr}")
            return False

        log(f"  Applied: {improvement['description'][:50]}...")
        return True

    except Exception as e:
        log(f"  Error applying improvement: {e}")
        return False


def git_commit_and_push(iteration: int, summary: str) -> bool:
    """Commit changes and push to git."""
    try:
        # Check for changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        if not result.stdout.strip():
            log("No changes to commit")
            return False

        # Stage all changes
        subprocess.run(
            ["git", "add", "-A"],
            cwd=Path(__file__).parent.parent
        )

        # Commit
        commit_msg = f"""CI Iteration {iteration}/5: Auto-improvements

{summary[:200]}

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"""

        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        if result.returncode != 0:
            log(f"Commit failed: {result.stderr}")
            return False

        log(f"Committed: CI Iteration {iteration}")

        # Try to push (may fail if remote not configured)
        result = subprocess.run(
            ["git", "push"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30
        )

        if result.returncode == 0:
            log("Pushed to remote")
        else:
            log(f"Push skipped (remote issue): {result.stderr[:100]}")

        return True

    except Exception as e:
        log(f"Git error: {e}")
        return False


def get_code_context() -> str:
    """Get relevant code context for review."""
    files = [
        "services/grok_service.py",
        "services/stock_data_service.py",
        "integrations/market_search.py",
    ]

    context = ""
    base = Path(__file__).parent.parent

    for f in files:
        path = base / f
        if path.exists():
            content = path.read_text()
            # Get first 100 lines
            lines = content.split('\n')[:100]
            context += f"\n### {f}\n```python\n" + '\n'.join(lines) + "\n```\n"

    return context[:8000]  # Limit context size


def run_iteration(iteration: int) -> dict:
    """Run a single iteration of the CI loop."""
    log(f"\n{'='*60}")
    log(f"ITERATION {iteration}/5")
    log(f"{'='*60}")

    questions = ITERATION_QUESTIONS[iteration - 1]
    results = {
        "iteration": iteration,
        "started": datetime.now().isoformat(),
        "grok_results": [],
        "market_results": [],
        "evaluation": {},
        "improvements_applied": 0,
        "committed": False,
    }

    # Step 1: Run Grok queries for KOL insights
    log("\n--- Step 1: Gathering KOL Insights (Grok) ---")
    for q in questions:
        log(f"  Querying: {q['symbol']} - {q['question'][:40]}...")
        result = run_grok_query(q["symbol"], q["question"])
        results["grok_results"].append(result)
        if result.get("success"):
            log(f"    OK: {result.get('latency_ms', 0)}ms, dims={result.get('dimensions', [])}")
        else:
            log(f"    Error: {result.get('error', 'unknown')}")
        time.sleep(0.5)  # Rate limiting

    # Step 2: Run market search
    log("\n--- Step 2: Gathering Market News ---")
    for q in questions:
        log(f"  Searching: {q['symbol']}...")
        result = run_market_search(q["symbol"])
        results["market_results"].append(result)
        if result.get("success"):
            log(f"    OK: {result.get('sources', 0)} sources, sentiment={result.get('sentiment', 'unknown')}")
        else:
            log(f"    Error: {result.get('error', 'unknown')}")

    # Step 3: Synthesize insights
    log("\n--- Step 3: Synthesizing Insights ---")
    synthesis = synthesize_insights(results["grok_results"], results["market_results"])
    log(f"  Synthesis length: {len(synthesis)} chars")

    # Step 4: Evaluate with Gemini 3 Pro
    log("\n--- Step 4: Gemini 3 Pro Evaluation ---")
    code_context = get_code_context()
    evaluation = evaluate_with_gemini(synthesis, iteration, code_context)
    results["evaluation"] = evaluation

    score = evaluation.get("score", 0)
    log(f"  Score: {score}/10")
    if evaluation.get("quality_breakdown"):
        qb = evaluation["quality_breakdown"]
        log(f"  Breakdown: KOL={qb.get('kol_insights')}, Market={qb.get('market_data')}, Synth={qb.get('synthesis')}")

    if evaluation.get("issues"):
        log(f"  Issues found: {len(evaluation['issues'])}")
        for issue in evaluation["issues"][:3]:
            log(f"    [{issue.get('severity', '?')}] {issue.get('description', '')[:60]}")

    # Step 5: Apply improvements
    log("\n--- Step 5: Applying Improvements ---")
    improvements = evaluation.get("improvements", [])
    if improvements:
        log(f"  {len(improvements)} improvements suggested")
        for imp in improvements[:5]:  # Limit to 5 improvements per iteration
            success = apply_improvement(imp)
            if success:
                results["improvements_applied"] += 1
    else:
        log("  No improvements suggested")

    # Step 6: Commit and push
    log("\n--- Step 6: Git Commit & Push ---")
    if results["improvements_applied"] > 0:
        summary = evaluation.get("summary", f"Iteration {iteration} improvements")
        results["committed"] = git_commit_and_push(iteration, summary)
    else:
        log("  No changes to commit")

    results["completed"] = datetime.now().isoformat()
    results["final_score"] = score

    log(f"\n  Iteration {iteration} complete: Score={score}/10, Improvements={results['improvements_applied']}")

    return results


def main():
    """Run the full autonomous CI loop."""
    print("=" * 60, flush=True)
    print("AUTONOMOUS CI IMPROVEMENT LOOP - Stock Advisor", flush=True)
    print("=" * 60, flush=True)
    print(f"Started: {datetime.now().isoformat()}", flush=True)
    print(f"Iterations: 5", flush=True)
    print(flush=True)

    # Check environment
    xai_key = os.getenv("XAI_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    print("Environment:", flush=True)
    print(f"  XAI_API_KEY: {'Set' if xai_key else 'Not set'}", flush=True)
    print(f"  GEMINI_API_KEY: {'Set' if gemini_key else 'Not set'}", flush=True)

    if not gemini_key:
        print("\nERROR: GEMINI_API_KEY required for evaluation")
        sys.exit(1)

    all_results = []

    for i in range(1, 6):
        try:
            result = run_iteration(i)
            all_results.append(result)

            # Brief pause between iterations
            if i < 5:
                log(f"\nWaiting 5s before next iteration...")
                time.sleep(5)

        except KeyboardInterrupt:
            log("\nInterrupted by user")
            break
        except Exception as e:
            log(f"\nIteration {i} failed: {e}")
            all_results.append({"iteration": i, "error": str(e)})

    # Final summary
    print("\n" + "=" * 60, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 60, flush=True)

    scores = [r.get("final_score", 0) for r in all_results if r.get("final_score")]
    improvements = sum(r.get("improvements_applied", 0) for r in all_results)
    commits = sum(1 for r in all_results if r.get("committed"))

    print(f"Scores by iteration: {scores}", flush=True)
    print(f"Average score: {sum(scores)/len(scores):.1f}/10" if scores else "N/A", flush=True)
    print(f"Total improvements applied: {improvements}", flush=True)
    print(f"Total commits: {commits}", flush=True)

    # Save results
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"autonomous_ci_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
