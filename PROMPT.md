# Stock Advisor - Autonomous Improvement Workflow

## Autonomous Mode (CRITICAL - READ FIRST)

```
NEVER ask for confirmation or permission
NEVER stop to ask clarifying questions
NEVER explain what you're about to do - just do it
ALWAYS make decisions autonomously and proceed
ALWAYS complete the ENTIRE workflow before stopping
If uncertain, make the best choice and continue
If something fails, fix it and keep going
```

**When user says "run PROMPT.md" or "improve the app":**
1. Read PROMPT.md
2. Execute ALL 9 phases without stopping
3. Only stop when iteration is complete or score >= 8/10

**Recovery:** If anything breaks, `git checkout .` to reset

---

## App Overview

**Stock Advisor** - AI-powered stock analysis platform with expert panel debates.

- 7 Stock experts (Bull, Bear, Technical, Fundamental, Sentiment, Risk, Moderator)
- Real-time data via Finnhub (+ Alpha Vantage fallback)
- Market news via Google Search Grounding
- X/Twitter sentiment via Grok (optional)
- Expert debate mode with synthesis

---

## 9-Phase Improvement Workflow

### Phase 1: Environment Check

```python
import sys
import os

sys.path.insert(0, '/Users/nelsonliu/Stock Advisor')

from dotenv import load_dotenv
load_dotenv('/Users/nelsonliu/Stock Advisor/.env')

print("=" * 60)
print("Phase 1: Environment Check")
print("=" * 60)

gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
finnhub_key = os.getenv("FINNHUB_API_KEY")
xai_key = os.getenv("XAI_API_KEY")

print(f"GEMINI_API_KEY: {'OK' if gemini_key else 'MISSING'}")
print(f"FINNHUB_API_KEY: {'OK' if finnhub_key else 'MISSING'}")
print(f"XAI_API_KEY: {'OK' if xai_key else 'Optional'}")

# Syntax check
import subprocess
files = ["app_sa.py", "services/grok_service.py", "services/stock_data_service.py"]
for f in files:
    result = subprocess.run(["python3", "-m", "py_compile", f], capture_output=True)
    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"{f}: {status}")
```

---

### Phase 2: Run Tests

```python
print("\n" + "=" * 60)
print("Phase 2: Run Tests")
print("=" * 60)

import subprocess
result = subprocess.run(
    ["python3", "tests/test_ci_workflow.py"],
    capture_output=True,
    text=True,
    cwd="/Users/nelsonliu/Stock Advisor"
)

print(result.stdout)
if result.returncode != 0:
    print("FAIL: Tests failed!")
    print(result.stderr)
else:
    print("OK: All tests passed")
```

---

### Phase 3: Test Research Pipeline

```python
print("\n" + "=" * 60)
print("Phase 3: Test Research Pipeline")
print("=" * 60)

# Test questions covering different CI dimensions
test_questions = [
    "What's the institutional flow on NVDA?",  # institutional_flow
    "Is there unusual options activity on AAPL?",  # options_sentiment
    "What do analysts say about TSLA price target?",  # analyst_ratings
    "Will MSFT beat earnings this quarter?",  # earnings_catalyst
    "How will Fed rate cuts affect tech stocks?",  # macro_sentiment
    "What's WSB saying about GME?",  # retail_sentiment
]

from services.grok_service import detect_stock_ci_dimensions

for q in test_questions:
    dims = detect_stock_ci_dimensions(q)
    status = "OK" if dims else "No CI"
    print(f"{status}: '{q[:50]}...' -> {dims}")
```

---

### Phase 4: Test Expert Debate

```python
print("\n" + "=" * 60)
print("Phase 4: Test Expert Debate")
print("=" * 60)

from services.stock_data_service import fetch_stock_data, build_expert_context

# Test stock data fetching
context = fetch_stock_data("AAPL", include_quote=True, include_financials=True)
print(f"Stock data fetched: {list(context.data_available.keys())}")

# Test expert context building
expert_context = build_expert_context(
    symbol="NVDA",
    question="Is NVDA a good buy for AI growth?",
    include_market_search=False
)
print(f"Expert context length: {len(expert_context)} chars")

if len(expert_context) > 100:
    print("OK: Expert context building working")
else:
    print("WARN: Expert context seems short")
```

---

### Phase 5: Code Quality Scan

```python
print("\n" + "=" * 60)
print("Phase 5: Code Quality Scan")
print("=" * 60)

issues = []

# Read key files
files_to_check = [
    "app_sa.py",
    "services/grok_service.py",
    "services/stock_data_service.py",
    "integrations/market_search.py",
]

for filepath in files_to_check:
    try:
        with open(f"/Users/nelsonliu/Stock Advisor/{filepath}", "r") as f:
            content = f.read()
            lines = content.split("\n")
    except FileNotFoundError:
        print(f"SKIP: {filepath} not found")
        continue

    # Check for issues
    file_issues = []

    # 1. Bare except clauses
    for i, line in enumerate(lines):
        if "except:" in line and "except Exception" not in line:
            file_issues.append(f"Line {i+1}: Bare except clause")

    # 2. TODO/FIXME comments
    for i, line in enumerate(lines):
        if "TODO" in line or "FIXME" in line:
            file_issues.append(f"Line {i+1}: {line.strip()[:60]}")

    # 3. Long functions (>100 lines)
    in_func = False
    func_start = 0
    func_name = ""
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            if in_func and (i - func_start) > 100:
                file_issues.append(f"Long function: {func_name} ({i - func_start} lines)")
            in_func = True
            func_start = i
            func_name = line.strip().split("(")[0].replace("def ", "")

    # 4. Missing error handling in API calls
    if "client.chat.completions.create" in content:
        if content.count("try:") < content.count("client.chat.completions.create"):
            file_issues.append("Some LLM calls may lack error handling")

    if file_issues:
        issues.append((filepath, file_issues))
        print(f"\nISSUES in {filepath}:")
        for issue in file_issues[:5]:  # Show first 5
            print(f"   - {issue}")
    else:
        print(f"OK: {filepath}")

total_issues = sum(len(i[1]) for i in issues)
print(f"\nTotal issues found: {total_issues}")
```

---

### Phase 6: Identify Top 3 Improvements

```python
print("\n" + "=" * 60)
print("Phase 6: Identify Top 3 Improvements")
print("=" * 60)

from core.llm_utils import get_llm_client

client = get_llm_client(model_type="pro")

# Read current state
with open("/Users/nelsonliu/Stock Advisor/services/grok_service.py", "r") as f:
    grok_content = f.read()[:4000]

with open("/Users/nelsonliu/Stock Advisor/services/stock_data_service.py", "r") as f:
    service_content = f.read()[:4000]

analysis_prompt = f'''
You are a senior Python developer reviewing a Chainlit stock advisor app.

=== grok_service.py (excerpt) ===
{grok_content}

=== stock_data_service.py (excerpt) ===
{service_content}

=== Issues Found ===
{issues}

Identify the TOP 3 improvements to make, prioritized by:
1. Bug fixes (highest priority)
2. Error handling gaps
3. Performance improvements
4. Code clarity

For each improvement, provide:
- File to modify
- Specific change to make
- Why it matters

Format:
## Improvement 1: [Title]
- **File:** path/to/file.py
- **Change:** [Specific code change]
- **Why:** [Impact]

## Improvement 2: ...
## Improvement 3: ...
'''

response = client.chat.completions.create(
    model="gemini-3-pro-preview",
    messages=[{"role": "user", "content": analysis_prompt}],
    max_tokens=2000
)

improvements = response.choices[0].message.content
print(improvements)

# Save for reference
os.makedirs("/Users/nelsonliu/Stock Advisor/output", exist_ok=True)
with open("/Users/nelsonliu/Stock Advisor/output/improvements.txt", "w") as f:
    f.write(improvements)
```

---

### Phase 7: Implement Improvements

```
This phase is handled by Claude Code:
1. Read output/improvements.txt
2. Make the code changes
3. Verify syntax after each change
4. Run tests to ensure nothing broke
```

After making changes:
```python
import subprocess
for f in ["app_sa.py", "services/grok_service.py", "services/stock_data_service.py"]:
    result = subprocess.run(
        ["python3", "-m", "py_compile", f],
        capture_output=True,
        cwd="/Users/nelsonliu/Stock Advisor"
    )
    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"{f}: {status}")
```

---

### Phase 8: Quality Review

```python
print("\n" + "=" * 60)
print("Phase 8: Quality Review")
print("=" * 60)

import subprocess
import os

# Re-run tests
result = subprocess.run(
    ["python3", "tests/test_ci_workflow.py"],
    capture_output=True,
    text=True,
    cwd="/Users/nelsonliu/Stock Advisor"
)

# Parse test results
passed = result.stdout.count("OK")
failed = result.stdout.count("FAIL")
skipped = result.stdout.count("SKIP")

print(f"Tests: {passed} passed, {failed} failed, {skipped} skipped")

# Score calculation
quality_score = 0

# Tests passing (40%)
test_score = min(10, (passed / max(1, passed + failed)) * 10)
quality_score += test_score * 0.4
print(f"Test Score: {test_score:.1f}/10 (weight: 40%)")

# Syntax check (20%)
syntax_ok = True
files_to_check = ["app_sa.py", "services/grok_service.py", "services/stock_data_service.py"]
for f in files_to_check:
    result = subprocess.run(
        ["python3", "-m", "py_compile", f],
        capture_output=True,
        cwd="/Users/nelsonliu/Stock Advisor"
    )
    if result.returncode != 0:
        syntax_ok = False
        break

syntax_score = 10 if syntax_ok else 0
quality_score += syntax_score * 0.2
print(f"Syntax Score: {syntax_score}/10 (weight: 20%)")

# Code issues (20%)
issues_score = max(0, 10 - total_issues)
quality_score += issues_score * 0.2
print(f"Issues Score: {issues_score}/10 (weight: 20%)")

# Feature completeness (20%)
features = {
    "CI detection": "detect_stock_ci_dimensions" in open("/Users/nelsonliu/Stock Advisor/services/grok_service.py").read(),
    "Market search": os.path.exists("/Users/nelsonliu/Stock Advisor/integrations/market_search.py"),
    "Stock data service": os.path.exists("/Users/nelsonliu/Stock Advisor/services/stock_data_service.py"),
    "Tests": os.path.exists("/Users/nelsonliu/Stock Advisor/tests/test_ci_workflow.py"),
}
feature_score = sum(features.values()) / len(features) * 10
quality_score += feature_score * 0.2
print(f"Feature Score: {feature_score:.1f}/10 (weight: 20%)")

print(f"\n{'=' * 40}")
print(f"TOTAL QUALITY SCORE: {quality_score:.1f}/10")
print(f"{'=' * 40}")

if quality_score >= 8:
    print("OK: Quality target achieved!")
else:
    print(f"NEED: {8 - quality_score:.1f} more points. Iterate Phase 6-8.")
```

---

### Phase 9: Commit & Report

```python
print("\n" + "=" * 60)
print("Phase 9: Commit & Report")
print("=" * 60)

import subprocess
import datetime

# Git status
result = subprocess.run(
    ["git", "status", "--porcelain"],
    capture_output=True,
    text=True,
    cwd="/Users/nelsonliu/Stock Advisor"
)

changed_files = [l.split()[-1] for l in result.stdout.strip().split("\n") if l]
print(f"Changed files: {len(changed_files)}")
for f in changed_files[:10]:
    print(f"  - {f}")

if changed_files and quality_score >= 8:
    # Commit changes
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    commit_msg = f"Auto-improve: Score {quality_score:.1f}/10 - {timestamp}"

    subprocess.run(["git", "add", "-A"], cwd="/Users/nelsonliu/Stock Advisor")
    subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd="/Users/nelsonliu/Stock Advisor"
    )
    print(f"Committed: {commit_msg}")
else:
    print("No commit (score < 8 or no changes)")

# Final report
print("\n" + "=" * 60)
print("IMPROVEMENT CYCLE COMPLETE")
print("=" * 60)
print(f"Quality Score: {quality_score:.1f}/10")
print(f"Tests: {passed} passed, {failed} failed")
print(f"Files Changed: {len(changed_files)}")
print(f"Issues Remaining: {total_issues}")

if quality_score < 8:
    print("\nScore below 8. Run another iteration:")
    print("   1. Review output/improvements.txt")
    print("   2. Make additional fixes")
    print("   3. Re-run Phase 8-9")
```

---

## Quality Standards

| Category | Weight | Score 10 | Score 5 | Score 0 |
|----------|--------|----------|---------|---------|
| **Tests** | 40% | All pass | >50% pass | <50% pass |
| **Syntax** | 20% | No errors | - | Has errors |
| **Issues** | 20% | 0 issues | 5 issues | 10+ issues |
| **Features** | 20% | All present | >50% | <50% |

**Target:** >= 8/10 to commit

---

## Iteration Flow

```
Phase 1-2: Environment & Tests
    |
Phase 3-4: Test Research & Expert Context
    |
Phase 5: Code Quality Scan
    |
Phase 6: Identify Improvements
    |
Phase 7: Implement Changes
    |
Phase 8: Quality Review
    |
Score >= 8? --Yes--> Phase 9: Commit
    |
    No
    |
Back to Phase 6
```

---

## Test Questions for Manual Testing

Use these to verify the app works end-to-end:

1. **Institutional Flow:** "What's the institutional activity on NVDA?"
2. **Options Sentiment:** "Is there unusual options activity on AAPL?"
3. **Analyst Ratings:** "What are analysts saying about TSLA price target?"
4. **Earnings Catalyst:** "Will AMZN beat earnings this quarter?"
5. **Macro Sentiment:** "How will Fed rate decisions affect tech stocks?"
6. **Retail Sentiment:** "What's Reddit saying about GME?"
7. **General Analysis:** "Analyze MSFT - is it a buy right now?"

---

## File Outputs

All outputs saved to `/Users/nelsonliu/Stock Advisor/output/`:

| File | Content |
|------|---------|
| `improvements.txt` | LLM-identified improvements |
| `test_results.txt` | Test run output |
| `quality_report.txt` | Quality score breakdown |

---

## Quick Reference Commands

```bash
# Run the app
chainlit run app_sa.py

# Development mode (auto-reload)
chainlit run app_sa.py -w

# Check syntax
python3 -m py_compile app_sa.py

# Test imports
python3 -c "from services import GrokService, fetch_stock_data; print('OK')"

# Run CI workflow tests
python3 tests/test_ci_workflow.py

# Reset if broken
git checkout .
```
