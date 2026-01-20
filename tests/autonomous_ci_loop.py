#!/usr/bin/env python3
"""
Stock Advisor Improvement Loop for Stock Advisor

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

# Stock questions for continuous iterations - cycles through these
ITERATION_QUESTIONS = [
    # Set 1: Mega-cap Tech
    [
        {"symbol": "NVDA", "question": "What's the institutional sentiment on NVDA? Are hedge funds buying or selling?"},
        {"symbol": "AAPL", "question": "What do finance KOLs think about Apple's AI strategy and iPhone sales?"},
        {"symbol": "MSFT", "question": "Is Microsoft overvalued? What's the sentiment on Azure growth?"},
    ],
    # Set 2: Semiconductor & AI
    [
        {"symbol": "TSM", "question": "What's the geopolitical risk for TSM and how are KOLs positioning?"},
        {"symbol": "AMD", "question": "AMD vs NVDA - which do retail traders prefer? Options flow?"},
        {"symbol": "AVGO", "question": "What's the options flow and institutional sentiment on Broadcom?"},
    ],
    # Set 3: EV & Clean Energy
    [
        {"symbol": "TSLA", "question": "What are the bull and bear cases for Tesla on X?"},
        {"symbol": "RIVN", "question": "Is Rivian a buy? What do analysts and KOLs say?"},
        {"symbol": "ENPH", "question": "What's the short interest and sentiment on Enphase Energy?"},
    ],
    # Set 4: Finance & Healthcare
    [
        {"symbol": "JPM", "question": "How are banks positioned for Fed rate decisions?"},
        {"symbol": "UNH", "question": "What's the sentiment on healthcare stocks and UnitedHealth?"},
        {"symbol": "V", "question": "Is Visa a defensive play? What do KOLs think?"},
    ],
    # Set 5: Big Tech
    [
        {"symbol": "GOOGL", "question": "What do KOLs think about Google's AI vs ChatGPT competition?"},
        {"symbol": "AMZN", "question": "Is Amazon's AWS growth sustainable? Retail vs cloud debate?"},
        {"symbol": "META", "question": "Meta's AI pivot - are KOLs bullish or bearish now?"},
    ],
    # Set 6: Growth & Momentum
    [
        {"symbol": "PLTR", "question": "Is Palantir overvalued? What's the retail sentiment on X?"},
        {"symbol": "SNOW", "question": "Snowflake growth outlook - what do tech analysts say?"},
        {"symbol": "CRWD", "question": "CrowdStrike after the outage - buy the dip or avoid?"},
    ],
    # Set 7: Value & Dividend
    [
        {"symbol": "BRK.B", "question": "Berkshire Hathaway cash pile - what's Buffett thinking?"},
        {"symbol": "JNJ", "question": "Johnson & Johnson as defensive play - KOL sentiment?"},
        {"symbol": "PG", "question": "Consumer staples in this economy - is P&G a safe bet?"},
    ],
    # Set 8: China & Emerging
    [
        {"symbol": "BABA", "question": "Alibaba risk/reward - what do KOLs say about China exposure?"},
        {"symbol": "PDD", "question": "PDD Holdings (Temu) growth - sustainable or bubble?"},
        {"symbol": "NIO", "question": "NIO vs Chinese EV competition - buy, hold, or sell?"},
    ],
    # Set 9: Retail & Consumer
    [
        {"symbol": "WMT", "question": "Walmart vs Amazon - who wins in retail? KOL views?"},
        {"symbol": "COST", "question": "Costco premium valuation - justified or overpriced?"},
        {"symbol": "TGT", "question": "Target turnaround - what's the institutional sentiment?"},
    ],
    # Set 10: Biotech & Pharma
    [
        {"symbol": "LLY", "question": "Eli Lilly weight-loss drugs - is the rally sustainable?"},
        {"symbol": "MRNA", "question": "Moderna post-COVID - what's next? KOL sentiment?"},
        {"symbol": "ABBV", "question": "AbbVie dividend and growth outlook - analyst views?"},
    ],
]

# Number of sets to cycle through
NUM_QUESTION_SETS = len(ITERATION_QUESTIONS)


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
    import re
    text = ""

    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        # Simplified prompt for reliable JSON
        prompt = f"""Evaluate this Stock Advisor output and suggest improvements.

SYNTHESIS:
{synthesis[:3000]}

CODE SNIPPETS:
{code_context[:4000]}

Return ONLY this JSON structure (no markdown, no explanation):
{{"score": 7, "kol_quality": 7, "market_quality": 7, "issues": ["issue1"], "improvements": [{{"file": "services/grok_service.py", "desc": "description", "old": "exact old code", "new": "new code"}}], "summary": "brief summary"}}

RULES:
1. score: 1-10 based on data quality
2. improvements: max 2, with EXACT code snippets from the context above
3. old must match exactly what's in the code
4. Keep changes small (1-3 lines)

JSON only:"""

        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=2000,
            ),
        )

        # Safely get response text
        text = response.text if response and response.text else ""

        if not text:
            log("Empty response from Gemini")
            return {"score": 6, "issues": [], "improvements": [], "summary": "Empty response"}

        # Clean up response - extract JSON
        text = text.strip()

        # Remove markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]

        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        result = json.loads(text.strip())

        # Normalize the result structure
        normalized = {
            "score": result.get("score", 6),
            "quality_breakdown": {
                "kol_insights": result.get("kol_quality", result.get("quality_breakdown", {}).get("kol_insights", 6)),
                "market_data": result.get("market_quality", result.get("quality_breakdown", {}).get("market_data", 6)),
            },
            "issues": result.get("issues", []),
            "improvements": [],
            "summary": result.get("summary", "Evaluation complete"),
        }

        # Convert improvements to standard format
        for imp in result.get("improvements", []):
            if isinstance(imp, dict) and imp.get("old") and imp.get("new"):
                normalized["improvements"].append({
                    "file": imp.get("file", "services/grok_service.py"),
                    "description": imp.get("desc", imp.get("description", "Improvement")),
                    "old_code": imp.get("old", imp.get("old_code", "")),
                    "new_code": imp.get("new", imp.get("new_code", "")),
                })

        return normalized

    except json.JSONDecodeError as e:
        log(f"JSON parse error: {e}")
        # Extract score with regex fallback
        try:
            score_match = re.search(r'"score"[:\s]+(\d+)', text)
            score = int(score_match.group(1)) if score_match else 6
            return {"score": score, "issues": [], "improvements": [], "summary": f"JSON parse error, extracted score={score}"}
        except:
            return {"score": 6, "issues": [], "improvements": [], "summary": "Parse error fallback"}
    except Exception as e:
        log(f"Gemini evaluation error: {e}")
        return {"score": 6, "issues": [], "improvements": [], "summary": str(e)}


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


def run_iteration(iteration: int, total_iterations: int = 0) -> dict:
    """Run a single iteration of the CI loop."""
    log(f"\n{'='*60}")
    if total_iterations > 0:
        log(f"ITERATION {iteration}/{total_iterations}")
    else:
        log(f"ITERATION {iteration} (continuous mode)")
    log(f"{'='*60}")

    # Cycle through question sets
    question_idx = (iteration - 1) % NUM_QUESTION_SETS
    questions = ITERATION_QUESTIONS[question_idx]
    log(f"Using question set {question_idx + 1}/{NUM_QUESTION_SETS}")
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
    """Run the full Stock Advisor improvement loop."""
    import argparse

    parser = argparse.ArgumentParser(description="Stock Advisor Improvement Loop")
    parser.add_argument("-n", "--iterations", type=int, default=0,
                        help="Number of iterations (0 = unlimited)")
    parser.add_argument("--no-push", action="store_true",
                        help="Skip git push (still commits)")
    parser.add_argument("--target-score", type=int, default=8,
                        help="Stop when average score reaches this (default: 8)")
    args = parser.parse_args()

    max_iterations = args.iterations
    mode = "continuous" if max_iterations == 0 else f"{max_iterations} iterations"

    print("=" * 60, flush=True)
    print("STOCK ADVISOR IMPROVEMENT LOOP - Stock Advisor", flush=True)
    print("=" * 60, flush=True)
    print(f"Started: {datetime.now().isoformat()}", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Target score: {args.target_score}/10", flush=True)
    print(f"Question sets: {NUM_QUESTION_SETS}", flush=True)
    print(flush=True)

    # Check environment
    xai_key = os.getenv("XAI_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    finnhub_key = os.getenv("FINNHUB_API_KEY")

    print("Environment:", flush=True)
    print(f"  XAI_API_KEY: {'Set' if xai_key else 'Not set'}", flush=True)
    print(f"  GEMINI_API_KEY: {'Set' if gemini_key else 'Not set'}", flush=True)
    print(f"  FINNHUB_API_KEY: {'Set' if finnhub_key else 'Not set'}", flush=True)

    if not gemini_key:
        print("\nERROR: GEMINI_API_KEY required for evaluation", flush=True)
        sys.exit(1)

    all_results = []
    iteration = 0
    consecutive_high_scores = 0

    while True:
        iteration += 1

        # Check if we've reached max iterations
        if max_iterations > 0 and iteration > max_iterations:
            log(f"\nReached max iterations ({max_iterations})")
            break

        try:
            result = run_iteration(iteration, max_iterations)
            all_results.append(result)

            # Track high scores
            score = result.get("final_score", 0)
            if score >= args.target_score:
                consecutive_high_scores += 1
                if consecutive_high_scores >= 3:
                    log(f"\nReached target score {args.target_score}/10 for 3 consecutive iterations!")
                    break
            else:
                consecutive_high_scores = 0

            # Save intermediate results every 5 iterations
            if iteration % 5 == 0:
                save_results(all_results, f"checkpoint_iter{iteration}")
                print_summary(all_results)

            # Brief pause between iterations
            log(f"\nWaiting 3s before next iteration...")
            time.sleep(3)

        except KeyboardInterrupt:
            log("\nInterrupted by user")
            break
        except Exception as e:
            log(f"\nIteration {iteration} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"iteration": iteration, "error": str(e)})
            # Continue on error
            time.sleep(5)

    # Final summary
    print_summary(all_results)
    save_results(all_results, "final")


def print_summary(all_results: list):
    """Print summary of results."""
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)

    scores = [r.get("final_score", 0) for r in all_results if r.get("final_score")]
    improvements = sum(r.get("improvements_applied", 0) for r in all_results)
    commits = sum(1 for r in all_results if r.get("committed"))

    print(f"Iterations completed: {len(all_results)}", flush=True)
    print(f"Scores: {scores[-10:]}" + (" ..." if len(scores) > 10 else ""), flush=True)
    print(f"Average score: {sum(scores)/len(scores):.1f}/10" if scores else "N/A", flush=True)
    print(f"Total improvements applied: {improvements}", flush=True)
    print(f"Total commits: {commits}", flush=True)


def save_results(all_results: list, suffix: str = ""):
    """Save results to JSON file."""
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"autonomous_ci_{timestamp}_{suffix}.json"

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"Results saved to: {output_file}", flush=True)


if __name__ == "__main__":
    main()
