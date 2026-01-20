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

# Stock questions for continuous iterations - focused on TSLA, ONDS, GOOGL, NVDA
# 2-3 diverse questions per stock covering different analysis dimensions
ITERATION_QUESTIONS = [
    # Set 1: TSLA - Technical & Sentiment
    [
        {"symbol": "TSLA", "question": "What's the technical analysis on Tesla? Key support/resistance levels and chart patterns?"},
        {"symbol": "TSLA", "question": "What's the retail vs institutional sentiment on Tesla right now on X/Twitter?"},
        {"symbol": "TSLA", "question": "Is Tesla overvalued at current P/E? What do value investors say?"},
    ],
    # Set 2: NVDA - AI & Growth
    [
        {"symbol": "NVDA", "question": "What's the institutional flow on NVIDIA? Are hedge funds accumulating or distributing?"},
        {"symbol": "NVDA", "question": "NVIDIA's AI chip dominance - how sustainable is it? What do tech analysts say?"},
        {"symbol": "NVDA", "question": "What's the options flow on NVDA? Any unusual activity or big bets?"},
    ],
    # Set 3: GOOGL - Competition & Valuation
    [
        {"symbol": "GOOGL", "question": "Google vs OpenAI/Microsoft in AI - who's winning? KOL sentiment?"},
        {"symbol": "GOOGL", "question": "Is Alphabet undervalued compared to other Mag7? What's the consensus?"},
        {"symbol": "GOOGL", "question": "What's the impact of AI on Google Search revenue? Bear case analysis?"},
    ],
    # Set 4: ONDS - Microcap Analysis
    [
        {"symbol": "ONDS", "question": "What's the sentiment on Ondas Holdings? Any insider activity or institutional interest?"},
        {"symbol": "ONDS", "question": "Ondas drone technology - competitive positioning and growth potential?"},
        {"symbol": "ONDS", "question": "ONDS short interest and squeeze potential - what do traders say?"},
    ],
    # Set 5: TSLA - EV Market & Competition
    [
        {"symbol": "TSLA", "question": "Tesla vs BYD and Chinese EV makers - who wins the global EV war?"},
        {"symbol": "TSLA", "question": "What's the sentiment on Tesla's robotaxi and FSD progress?"},
        {"symbol": "TSLA", "question": "Tesla energy storage and solar business - undervalued catalyst?"},
    ],
    # Set 6: NVDA - Fundamentals & Risks
    [
        {"symbol": "NVDA", "question": "NVIDIA earnings expectations - what's priced in? Risk of disappointment?"},
        {"symbol": "NVDA", "question": "China export restrictions impact on NVIDIA - how big is the risk?"},
        {"symbol": "NVDA", "question": "NVIDIA vs AMD vs Intel - competitive moat analysis from KOLs?"},
    ],
    # Set 7: GOOGL - Revenue & Products
    [
        {"symbol": "GOOGL", "question": "YouTube and Cloud growth trajectory - what do analysts project?"},
        {"symbol": "GOOGL", "question": "Antitrust risks for Google - how are investors pricing this in?"},
        {"symbol": "GOOGL", "question": "Waymo autonomous driving - hidden value in Alphabet? KOL views?"},
    ],
    # Set 8: Mixed - Cross Analysis
    [
        {"symbol": "NVDA", "question": "Is NVIDIA the best AI play or is it overextended? Contrarian views?"},
        {"symbol": "TSLA", "question": "Elon Musk political involvement impact on Tesla brand - bull vs bear?"},
        {"symbol": "GOOGL", "question": "Google Gemini vs GPT-5 expectations - who leads in 2025?"},
    ],
    # Set 9: ONDS & Small Cap
    [
        {"symbol": "ONDS", "question": "Ondas government contracts and revenue outlook - growth catalysts?"},
        {"symbol": "ONDS", "question": "Is ONDS a takeover target? M&A speculation in drone sector?"},
        {"symbol": "TSLA", "question": "Tesla Optimus robot timeline and market potential - realistic assessment?"},
    ],
    # Set 10: Portfolio Strategy
    [
        {"symbol": "NVDA", "question": "Should I buy NVDA dips or wait for correction? Timing strategies?"},
        {"symbol": "GOOGL", "question": "GOOGL as a value play in tech - dividend potential and buybacks?"},
        {"symbol": "TSLA", "question": "Tesla as a long-term hold vs trade - what's the smart money doing?"},
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
    """Synthesize all gathered insights into a comprehensive report."""
    # Group by symbol
    symbols = set()
    for g in grok_results:
        symbols.add(g.get("symbol", "UNKNOWN"))
    for m in market_results:
        symbols.add(m.get("symbol", "UNKNOWN"))

    synthesis = "## Stock Intelligence Report\n\n"

    # Summary stats
    grok_success = sum(1 for g in grok_results if g.get("success"))
    market_success = sum(1 for m in market_results if m.get("success"))
    total_sources = sum(m.get("sources", 0) for m in market_results)

    synthesis += f"**Coverage:** {len(symbols)} stocks | {grok_success} KOL queries | {market_success} market searches | {total_sources} news sources\n\n"

    # Sentiment summary
    sentiments = [m.get("sentiment", "unknown") for m in market_results if m.get("success")]
    bullish = sentiments.count("bullish")
    bearish = sentiments.count("bearish")
    mixed = sentiments.count("mixed")
    synthesis += f"**Overall Sentiment:** 🟢 Bullish: {bullish} | 🔴 Bearish: {bearish} | ⚪ Mixed: {mixed}\n\n"

    synthesis += "---\n\n"

    # Detailed by symbol
    for symbol in sorted(symbols):
        synthesis += f"### {symbol}\n\n"

        # KOL Insights for this symbol
        symbol_grok = [g for g in grok_results if g.get("symbol") == symbol and g.get("success")]
        if symbol_grok:
            synthesis += "**KOL Insights:**\n"
            for g in symbol_grok:
                dims = g.get("dimensions", [])
                dim_str = f" [{', '.join(dims)}]" if dims else ""
                synthesis += f"- {dim_str} {g.get('content', '')[:400]}...\n"
            synthesis += "\n"

        # Market News for this symbol
        symbol_market = [m for m in market_results if m.get("symbol") == symbol and m.get("success")]
        if symbol_market:
            synthesis += "**Market News:**\n"
            for m in symbol_market:
                sentiment_emoji = {"bullish": "🟢", "bearish": "🔴", "mixed": "⚪"}.get(m.get("sentiment"), "⚪")
                synthesis += f"- {sentiment_emoji} ({m.get('sources', 0)} sources) {m.get('content', '')[:300]}...\n"
            synthesis += "\n"

        synthesis += "---\n\n"

    return synthesis


def calculate_quality_score(grok_results: list, market_results: list, synthesis: str) -> dict:
    """Calculate quality score based on concrete metrics."""
    scores = {}

    # 1. Grok KOL Insights Quality (0-10)
    grok_successes = sum(1 for g in grok_results if g.get("success"))
    grok_total = len(grok_results) if grok_results else 1
    grok_content_len = sum(len(g.get("content", "")) for g in grok_results)
    grok_dimensions = sum(len(g.get("dimensions", [])) for g in grok_results)

    grok_score = min(10, (
        (grok_successes / grok_total) * 4 +  # Success rate (0-4)
        min(4, grok_content_len / 2000) +     # Content depth (0-4)
        min(2, grok_dimensions)                # Dimension coverage (0-2)
    ))
    scores["kol_insights"] = round(grok_score, 1)

    # 2. Market News Quality (0-10)
    market_successes = sum(1 for m in market_results if m.get("success"))
    market_total = len(market_results) if market_results else 1
    total_sources = sum(m.get("sources", 0) for m in market_results)
    sentiment_diversity = len(set(m.get("sentiment", "unknown") for m in market_results))

    market_score = min(10, (
        (market_successes / market_total) * 4 +  # Success rate (0-4)
        min(4, total_sources / 15) +              # Source count (0-4)
        sentiment_diversity                        # Sentiment diversity (0-2)
    ))
    scores["market_data"] = round(market_score, 1)

    # 3. Synthesis Quality (0-10)
    synth_len = len(synthesis)
    has_kol = "KOL" in synthesis or "Insights" in synthesis
    has_market = "Market" in synthesis or "News" in synthesis
    has_sentiment = "bullish" in synthesis.lower() or "bearish" in synthesis.lower()

    synth_score = min(10, (
        min(4, synth_len / 1000) +    # Length (0-4)
        (2 if has_kol else 0) +       # Has KOL section (0-2)
        (2 if has_market else 0) +    # Has market section (0-2)
        (2 if has_sentiment else 0)   # Has sentiment (0-2)
    ))
    scores["synthesis"] = round(synth_score, 1)

    # Overall score (weighted average)
    overall = (scores["kol_insights"] * 0.4 + scores["market_data"] * 0.3 + scores["synthesis"] * 0.3)
    scores["overall"] = round(overall, 1)

    return scores


def evaluate_with_gemini(synthesis: str, iteration: int, code_context: str,
                         grok_results: list = None, market_results: list = None) -> dict:
    """Get Gemini 3 Pro to evaluate and suggest improvements."""
    import re
    text = ""

    # Calculate metrics-based score first
    if grok_results and market_results:
        quality_scores = calculate_quality_score(grok_results, market_results, synthesis)
        base_score = quality_scores["overall"]
    else:
        quality_scores = {"kol_insights": 6, "market_data": 6, "synthesis": 6, "overall": 6}
        base_score = 6

    # Get learnings from previous iterations
    learnings = get_learnings_summary()

    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        # Simpler, more focused prompt
        prompt = f"""Rate this stock analysis and suggest ONE improvement.

ANALYSIS OUTPUT:
{synthesis[:2500]}

CURRENT CODE VALUES (from services/grok_service.py):
- temperature: 0.35 (lower = more focused)
- max_tokens: 3200 (higher = longer responses)
- max_retries: 7
- timeout: 180s
- cache_ttl: 14400s

CALCULATED METRICS:
- KOL Quality: {quality_scores['kol_insights']}/10
- Market Data: {quality_scores['market_data']}/10
- Synthesis: {quality_scores['synthesis']}/10
- Base Score: {base_score}/10

TASK: Return JSON with score and ONE improvement suggestion.

Example response:
{{"score": 7, "issues": ["responses too short", "missing dimensions"], "improvement": {{"file": "services/grok_service.py", "desc": "Increase depth", "old": "max_tokens: 3200", "new": "max_tokens: 3500"}}, "summary": "Increased tokens"}}

Your JSON response:"""

        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,  # Lower for more consistent JSON
                max_output_tokens=1000,
            ),
        )

        # Safely get response text
        text = response.text if response and response.text else ""

        if not text or len(text.strip()) < 10:
            log(f"Empty/short response from Gemini (len={len(text)})")
            return {
                "score": int(base_score),
                "quality_breakdown": quality_scores,
                "issues": ["Gemini returned empty response"],
                "improvements": [],
                "summary": f"Using calculated score: {base_score}"
            }

        # Clean up response - extract JSON
        text = text.strip()

        # Remove markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]

        # Try to find JSON object - more permissive regex
        text = text.strip()
        if not text.startswith("{"):
            json_start = text.find("{")
            if json_start >= 0:
                text = text[json_start:]

        # Find matching brace
        brace_count = 0
        json_end = 0
        for i, c in enumerate(text):
            if c == "{":
                brace_count += 1
            elif c == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break

        if json_end > 0:
            text = text[:json_end]

        result = json.loads(text)

        # Extract score - use Gemini's score if reasonable, otherwise use calculated
        gemini_score = result.get("score", base_score)
        if isinstance(gemini_score, (int, float)) and 1 <= gemini_score <= 10:
            # Blend Gemini's assessment with calculated metrics
            final_score = int((gemini_score * 0.6 + base_score * 0.4))
        else:
            final_score = int(base_score)

        # Normalize the result structure
        normalized = {
            "score": final_score,
            "quality_breakdown": quality_scores,
            "issues": result.get("issues", []),
            "improvements": [],
            "summary": result.get("summary", "Evaluation complete"),
        }

        # Handle single improvement or list
        improvements = result.get("improvements", [])
        if not improvements and result.get("improvement"):
            improvements = [result.get("improvement")]

        for imp in (improvements if isinstance(improvements, list) else [improvements]):
            if isinstance(imp, dict) and (imp.get("old") or imp.get("old_code")):
                normalized["improvements"].append({
                    "file": imp.get("file", "services/grok_service.py"),
                    "description": imp.get("desc", imp.get("description", "Improvement")),
                    "old_code": imp.get("old", imp.get("old_code", "")),
                    "new_code": imp.get("new", imp.get("new_code", "")),
                })

        log(f"  Gemini evaluation: score={final_score}, issues={len(normalized['issues'])}, improvements={len(normalized['improvements'])}")
        return normalized

    except json.JSONDecodeError as e:
        log(f"JSON parse error: {e}")
        # Return calculated score on parse error
        return {
            "score": int(base_score),
            "quality_breakdown": quality_scores,
            "issues": [f"JSON parse error: {str(e)[:50]}"],
            "improvements": [],
            "summary": f"Parse error, using calculated score: {base_score}"
        }
    except Exception as e:
        log(f"Gemini evaluation error: {e}")
        return {
            "score": int(base_score),
            "quality_breakdown": quality_scores,
            "issues": [str(e)[:100]],
            "improvements": [],
            "summary": f"Error: {str(e)[:50]}"
        }


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


# Global learnings tracker
ITERATION_LEARNINGS = []

# Predefined improvements to apply when Gemini doesn't suggest any
# NOTE: Already applied in previous runs: timeout (60→90), GROK_CACHE_TTL (3600→7200),
#       QUOTE_CACHE_TTL (300→600), FINANCIALS_CACHE_TTL (3600→7200)
FALLBACK_IMPROVEMENTS = [
    # Round 5: Fresh improvements based on current code state (after Round 4)
    # Current: max_retries=7, timeout=180, cache_ttl=14400
    # Current: QUOTE_CACHE_TTL=1500, PROFILE_CACHE_TTL=10800, NEWS_CACHE_TTL=1800
    {
        "file": "services/grok_service.py",
        "description": "Increase main prompt max tokens for deeper KOL analysis",
        "old_code": '"temperature": 0.35,\n            "max_tokens": 3200',
        "new_code": '"temperature": 0.35,\n            "max_tokens": 3500',
    },
    {
        "file": "services/grok_service.py",
        "description": "Increase max retries for better reliability",
        "old_code": "max_retries: int = 7,",
        "new_code": "max_retries: int = 8,",
    },
    {
        "file": "services/grok_service.py",
        "description": "Increase timeout for complex queries",
        "old_code": "timeout: int = 180,",
        "new_code": "timeout: int = 200,",
    },
    {
        "file": "services/grok_service.py",
        "description": "Increase Grok cache TTL for efficiency",
        "old_code": 'self._cache_ttl = int(os.getenv("GROK_CACHE_TTL", "14400"))',
        "new_code": 'self._cache_ttl = int(os.getenv("GROK_CACHE_TTL", "18000"))',
    },
    {
        "file": "integrations/finnhub.py",
        "description": "Increase financials cache TTL",
        "old_code": "FINANCIALS_CACHE_TTL = 7200",
        "new_code": "FINANCIALS_CACHE_TTL = 10800",
    },
    {
        "file": "services/grok_service.py",
        "description": "Increase search_and_summarize tokens",
        "old_code": '"temperature": 0.3,\n            "max_tokens": 2000',
        "new_code": '"temperature": 0.3,\n            "max_tokens": 2200',
    },
    {
        "file": "services/grok_service.py",
        "description": "Increase synthesize_kol_views tokens",
        "old_code": '"temperature": 0.3,\n            "max_tokens": 2800',
        "new_code": '"temperature": 0.3,\n            "max_tokens": 3000',
    },
    {
        "file": "services/grok_service.py",
        "description": "Increase CI search max tokens",
        "old_code": '"temperature": 0.25,\n            "max_tokens": 2500',
        "new_code": '"temperature": 0.25,\n            "max_tokens": 2800',
    },
    {
        "file": "integrations/finnhub.py",
        "description": "Increase news cache TTL for stability",
        "old_code": "NEWS_CACHE_TTL = 1800",
        "new_code": "NEWS_CACHE_TTL = 2100",
    },
    {
        "file": "integrations/finnhub.py",
        "description": "Increase quote cache TTL",
        "old_code": "QUOTE_CACHE_TTL = 1500",
        "new_code": "QUOTE_CACHE_TTL = 1800",
    },
    # Round 6: Additional improvements
    {
        "file": "services/grok_service.py",
        "description": "Further increase main prompt tokens",
        "old_code": '"temperature": 0.35,\n            "max_tokens": 3500',
        "new_code": '"temperature": 0.35,\n            "max_tokens": 3800',
    },
    {
        "file": "services/grok_service.py",
        "description": "Lower temperature for more focused responses",
        "old_code": '"temperature": 0.35,\n            "max_tokens": 3800',
        "new_code": '"temperature": 0.32,\n            "max_tokens": 3800',
    },
    {
        "file": "services/grok_service.py",
        "description": "Increase research max tokens",
        "old_code": '"temperature": 0.25,  # Lower temp for research accuracy\n            "max_tokens": 3000',
        "new_code": '"temperature": 0.25,  # Lower temp for research accuracy\n            "max_tokens": 3300',
    },
    {
        "file": "integrations/finnhub.py",
        "description": "Increase candle cache TTL",
        "old_code": "CANDLE_CACHE_TTL = 7200",
        "new_code": "CANDLE_CACHE_TTL = 10800",
    },
    {
        "file": "services/grok_service.py",
        "description": "Increase CI search tokens further",
        "old_code": '"temperature": 0.25,\n            "max_tokens": 2800',
        "new_code": '"temperature": 0.25,\n            "max_tokens": 3100',
    },
    {
        "file": "services/grok_service.py",
        "description": "Increase synthesize tokens further",
        "old_code": '"temperature": 0.3,\n            "max_tokens": 3000',
        "new_code": '"temperature": 0.3,\n            "max_tokens": 3200',
    },
    {
        "file": "integrations/finnhub.py",
        "description": "Increase profile cache TTL",
        "old_code": "PROFILE_CACHE_TTL = 10800",
        "new_code": "PROFILE_CACHE_TTL = 14400",
    },
    {
        "file": "services/grok_service.py",
        "description": "Increase search_and_summarize further",
        "old_code": '"temperature": 0.3,\n            "max_tokens": 2200',
        "new_code": '"temperature": 0.3,\n            "max_tokens": 2400',
    },
    {
        "file": "services/grok_service.py",
        "description": "Increase max retries to 9",
        "old_code": "max_retries: int = 8,",
        "new_code": "max_retries: int = 9,",
    },
    {
        "file": "services/grok_service.py",
        "description": "Increase timeout to 220s",
        "old_code": "timeout: int = 200,",
        "new_code": "timeout: int = 220,",
    },
]

# Track which fallback improvements have been applied
APPLIED_FALLBACKS = set()


def get_fallback_improvements(iteration: int, score: int) -> list:
    """Get fallback improvements when Gemini doesn't suggest any."""
    available = []
    for i, imp in enumerate(FALLBACK_IMPROVEMENTS):
        if i not in APPLIED_FALLBACKS:
            available.append(imp)
            APPLIED_FALLBACKS.add(i)
            break  # Only return one at a time
    return available


def get_learnings_summary() -> str:
    """Get a summary of learnings from previous iterations."""
    if not ITERATION_LEARNINGS:
        return "No previous iterations yet."

    summary = f"Previous {len(ITERATION_LEARNINGS)} iterations:\n"
    for learning in ITERATION_LEARNINGS[-5:]:  # Last 5 iterations
        summary += f"- Iter {learning['iteration']}: Score {learning['score']}/10, "
        summary += f"Issues: {learning.get('issues', 'none')}\n"
    return summary


def run_iteration(iteration: int, total_iterations: int = 0, previous_results: list = None) -> dict:
    """Run a single iteration of the improvement loop."""
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

    # Show learnings from previous iterations
    if ITERATION_LEARNINGS:
        log(f"  Learning from {len(ITERATION_LEARNINGS)} previous iterations")

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
    evaluation = evaluate_with_gemini(
        synthesis, iteration, code_context,
        grok_results=results["grok_results"],
        market_results=results["market_results"]
    )
    results["evaluation"] = evaluation

    score = evaluation.get("score", 0)
    log(f"  Score: {score}/10")
    if evaluation.get("quality_breakdown"):
        qb = evaluation["quality_breakdown"]
        log(f"  Breakdown: KOL={qb.get('kol_insights')}, Market={qb.get('market_data')}, Synth={qb.get('synthesis')}")

    if evaluation.get("issues"):
        log(f"  Issues found: {len(evaluation['issues'])}")
        for issue in evaluation["issues"][:3]:
            # Handle both string and dict issues
            if isinstance(issue, dict):
                log(f"    [{issue.get('severity', '?')}] {issue.get('description', '')[:60]}")
            else:
                log(f"    - {str(issue)[:70]}")

    # Step 5: Apply improvements
    log("\n--- Step 5: Applying Improvements ---")
    improvements = evaluation.get("improvements", [])

    # If no improvements from Gemini, use fallback improvements based on score
    if not improvements and score < 8:
        log("  No Gemini improvements, using fallback strategy")
        improvements = get_fallback_improvements(iteration, score)

    if improvements:
        log(f"  {len(improvements)} improvements to apply")
        for imp in improvements[:3]:  # Limit to 3 improvements per iteration
            success = apply_improvement(imp)
            if success:
                results["improvements_applied"] += 1
                log(f"    ✓ Applied: {imp.get('description', imp.get('desc', 'improvement'))[:50]}")
    else:
        log("  No improvements to apply")

    # Save learnings for next iteration
    ITERATION_LEARNINGS.append({
        "iteration": iteration,
        "score": score,
        "issues": evaluation.get("issues", [])[:3],
        "applied": results["improvements_applied"],
    })

    # Step 6: Commit and push
    log("\n--- Step 6: Git Commit & Push ---")
    if results["improvements_applied"] > 0:
        summary = evaluation.get("summary", f"Iteration {iteration} improvements")
        results["committed"] = git_commit_and_push(iteration, summary)
    else:
        log("  No changes to commit")

    results["completed"] = datetime.now().isoformat()
    results["final_score"] = score
    results["synthesis"] = synthesis  # Save synthesis for review

    # Save detailed iteration output
    save_iteration_output(iteration, results, questions)

    log(f"\n  Iteration {iteration} complete: Score={score}/10, Improvements={results['improvements_applied']}")

    return results


def save_iteration_output(iteration: int, results: dict, questions: list):
    """Save detailed output for each iteration."""
    output_dir = Path(__file__).parent.parent / "output" / "iterations"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"iter_{iteration:03d}_{timestamp}.json"

    # Create detailed output
    output = {
        "iteration": iteration,
        "timestamp": timestamp,
        "questions": questions,
        "results": {
            "grok_insights": results.get("grok_results", []),
            "market_news": results.get("market_results", []),
            "synthesis": results.get("synthesis", ""),
            "evaluation": results.get("evaluation", {}),
        },
        "improvements": {
            "applied": results.get("improvements_applied", 0),
            "committed": results.get("committed", False),
        },
        "score": results.get("final_score", 0),
    }

    output_path = output_dir / filename
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log(f"  Output saved: {output_path.name}")


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
