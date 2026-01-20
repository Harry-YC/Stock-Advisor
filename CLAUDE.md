# CLAUDE.md - AI Assistant Context

## Autonomous Mode (CRITICAL - READ FIRST)

```
NEVER ask for confirmation or permission
NEVER stop to ask clarifying questions
NEVER explain what you're about to do - just do it
ALWAYS make decisions autonomously and proceed
ALWAYS complete the ENTIRE task before stopping
If uncertain, make the best choice and continue
If something fails, fix it and keep going
```

**When user says "run PROMPT.md" or "improve the app":**
1. Read PROMPT.md
2. Execute ALL 9 phases without stopping
3. Only stop when iteration is complete or score >= 8/10

**Recovery:** If anything breaks, `git checkout .` to reset

---

## Project Overview

**Stock Advisor** is an AI-powered stock analysis app using Chainlit. Users can ask about stocks, upload KOL screenshots, and get personalized analysis from 7 AI stock experts with real-time data from Finnhub (with Alpha Vantage fallback), Google Search grounding, and X/Twitter sentiment via Grok.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (copy .env.example to .env)
cp .env.example .env

# Run the app
chainlit run app_sa.py

# Development mode (auto-reload)
chainlit run app_sa.py -w
```

## Core Workflow

1. **Ask about stocks** - Type a ticker or question like "Analyze NVDA"
2. **Upload KOL screenshots** - Share Twitter/Reddit posts for analysis
3. **Expert recommendations** - Get advice from selected expert panel
4. **Expert Debate Mode** - Multi-round debate with synthesis (new!)
5. **Follow-up questions** - Ask specific experts for more details
6. **Track portfolio** - Use MCP tools for alerts and positions

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Chainlit 1.3+ |
| AI Model | Google Gemini 3 Pro/Flash |
| Stock Data | Finnhub API (primary) |
| Stock Data Fallback | Alpha Vantage API (for micro-cap stocks) |
| News Search | Google Search Grounding |
| X/Twitter Sentiment | Grok (xAI API) |
| Vision OCR | Gemini Vision (3 Flash) |
| Persistence | SQLite (MCP server) |

## Project Structure

```
Stock Advisor/
├── app_sa.py                     # Main Chainlit app
├── chainlit.md                   # Welcome message
├── requirements.txt              # Python dependencies
│
├── config/
│   └── settings.py               # Centralized configuration
│
├── stocks/
│   └── stock_personas.py         # 7 Expert definitions + presets + debate prompts
│
├── integrations/
│   ├── finnhub.py                # Finnhub API client (quotes, financials, news)
│   ├── alpha_vantage.py          # Alpha Vantage API client (fallback)
│   ├── gemini_vision.py          # KOL screenshot OCR
│   └── market_search.py          # Google Search grounding for stocks
│
├── services/
│   ├── llm_router.py             # Gemini API routing with caching & circuit breaker
│   ├── stock_data_service.py     # Multi-source data aggregation (Finnhub + Alpha Vantage + Grok)
│   ├── grok_service.py           # X/Twitter KOL insights via Grok API
│   └── kol_analyzer.py           # KOL text analysis service
│
├── core/
│   └── llm_utils.py              # LLM client factory
│
├── mcp_server/
│   ├── database.py               # SQLite for alerts, portfolio, watchlist, chat history
│   └── financial_mcp.py          # MCP tools (13 tools)
│
└── outputs/
    └── stock_advisor.db          # SQLite database (auto-created)
```

## Environment Variables

```bash
# Required
GEMINI_API_KEY=xxx                # From aistudio.google.com

# Recommended
FINNHUB_API_KEY=xxx               # From finnhub.io (free: 60 calls/min)

# Optional - Fallback for micro-cap stocks
ALPHA_VANTAGE_API_KEY=xxx         # From alphavantage.co (free: 25 calls/day)

# Optional - X/Twitter sentiment
XAI_API_KEY=xxx                   # From x.ai/api - enables Grok KOL insights

# Optional - Grok configuration
GROK_MODEL=grok-3-latest          # Default Grok model
GROK_CACHE_TTL=3600               # Cache TTL in seconds (default: 1 hour)

# Optional
APP_ENV=dev                       # Environment (dev/prod)
API_TIMEOUT=240                   # API timeout in seconds
EXPERT_MODEL=gemini-3-pro-preview # Primary expert model
EXPERT_FALLBACK_MODEL=gemini-3-flash-preview  # Fallback on timeout
```

## 7 Stock Experts

| Expert | Icon | Focus |
|--------|------|-------|
| Bull Analyst | 🐂 | Growth catalysts, upside targets, bullish scenarios |
| Bear Analyst | 🐻 | Risk factors, downside targets, valuation concerns |
| Technical Analyst | 📈 | Chart patterns, support/resistance, indicators |
| Fundamental Analyst | 📊 | Financials, valuation metrics, DCF analysis |
| Sentiment Analyst | 📰 | News sentiment, KOL analysis, social trends |
| Risk Manager | 🛡️ | Position sizing, hedging, stop-loss strategy |
| **Debate Moderator** | ⚖️ | Neutral synthesis, consensus building (debate mode only) |

## Expert Presets

| Preset | Experts Included |
|--------|------------------|
| Quick Analysis | Bull, Bear, Technical |
| Deep Dive | Bull, Bear, Technical, Fundamental, Risk |
| KOL Review | Sentiment, Bull, Bear |
| Trade Planning | Technical, Risk |
| Full Panel | All 6 experts |
| **Expert Debate** | All 6 experts + Moderator (3 rounds + synthesis) |

## Key Features

### Expert Debate Mode (New!)
Multi-round expert debate with synthesis:
- **Round 1**: Initial analysis from all experts (parallel)
- **Round 2**: Experts respond to each other, challenge views
- **Round 3**: Final rebuttals and verdicts
- **Synthesis**: Debate Moderator synthesizes consensus/disagreements
- Select "Expert Debate" preset to enable

### Stock Data with Fallback
- **Primary**: Finnhub API (60 calls/min free tier)
- **Fallback**: Alpha Vantage API (25 calls/day free tier)
- Automatic fallback for stocks not covered by Finnhub (e.g., micro-cap stocks)
- Data source tracked in context for transparency

### KOL Screenshot Analysis
- Upload screenshots from Twitter/X, Reddit, StockTwits
- Automatic ticker extraction via Gemini Vision OCR
- Sentiment classification (bullish/bearish/neutral/mixed)
- Key claims extraction for expert validation

### Real-Time Stock Data
- Finnhub integration for quotes, financials, news
- Alpha Vantage fallback for micro-cap/OTC stocks
- 5-min cache for quotes, 1-hour for fundamentals
- Rate limiting (60 req/min Finnhub, 25/day Alpha Vantage)

### Google Search Grounding
- Real-time market news and analyst ratings
- "Why did X stock move?" explanations
- Earnings and guidance updates

### Grok X/Twitter Sentiment (Optional)
When `XAI_API_KEY` is set, enables real-time X/Twitter sentiment from finance KOLs:
- **Stock KOLs tracked**: Burry, Cramer, Ackman, Cathie Wood, Keith Gill, etc.
- **CI Dimension Detection**: Auto-detects relevant dimensions from questions
- **8 CI Dimensions**: institutional_flow, options_sentiment, analyst_ratings, earnings_catalyst, macro_sentiment, retail_sentiment, sector_rotation, short_interest

| CI Dimension | Triggers | What it Searches |
|--------------|----------|------------------|
| `institutional_flow` | institutional, 13f, hedge fund, whale | What hedge funds are buying/selling |
| `options_sentiment` | options, calls, puts, gamma, squeeze | Unusual options activity |
| `analyst_ratings` | analyst, upgrade, downgrade, price target | Rating changes, targets |
| `earnings_catalyst` | earnings, eps, guidance, beat, miss | Earnings expectations |
| `macro_sentiment` | fed, rates, inflation, recession, powell | Macro environment views |
| `retail_sentiment` | reddit, wsb, meme, stocktwits | Retail trader sentiment |
| `sector_rotation` | sector, rotation, cyclical, defensive | Sector flow trends |
| `short_interest` | short, squeeze, borrow, hindenburg | Short squeeze potential |

### MCP Tools (13 tools)
| Tool | Description |
|------|-------------|
| `stock_quote` | Real-time price and metrics |
| `stock_financials` | Fundamental financial data |
| `stock_news` | Recent news articles |
| `add_alert` | Set price alert (above/below) |
| `list_alerts` | View active alerts |
| `delete_alert` | Remove an alert |
| `portfolio_add` | Add position with cost basis |
| `portfolio_view` | View holdings with P&L |
| `portfolio_sell` | Sell/reduce position |
| `kol_sentiment` | Aggregated KOL sentiment |
| `watchlist_add` | Add to watchlist |
| `watchlist_view` | View watchlist |
| `watchlist_remove` | Remove from watchlist |

## User Commands

| Command | Action |
|---------|--------|
| "Analyze NVDA" | Run expert panel analysis |
| "Why did TSLA fall?" | Search for movement explanation |
| "Ask Bull about AAPL" | Consult specific expert |
| [Upload screenshot] | OCR + sentiment analysis |

## API Costs (Approximate)

| API | Free Tier | Cost After |
|-----|-----------|------------|
| Gemini | Generous | Pay-per-token |
| Finnhub | 60 calls/min | Premium plans available |
| Alpha Vantage | 25 calls/day | $49.99/month for higher limits |
| Google Search | Included with Gemini | - |

## Architecture Features

### LLM Router (`services/llm_router.py`)
- Response caching (1-hour TTL) for repeated queries
- Circuit breaker to avoid hammering failing APIs
- Retry logic with exponential backoff
- Automatic fallback from Pro to Flash on timeout

### Thread-Safe Caching
- Finnhub client uses `threading.RLock()`
- Alpha Vantage client with rate limiting (25/day)
- All cache operations protected

### Prompt Injection Protection
- `_sanitize_for_prompt()` in stock_data_service.py
- Blocks "ignore previous instructions" patterns
- Limits input length

### File Upload Limits
- Images: 5MB maximum
- Supported: PNG, JPG, JPEG, GIF, WebP

## Troubleshooting

### 403 PERMISSION_DENIED
- Create fresh key from AI Studio (not Cloud Console)
- Enable "Generative Language API" in Cloud Console

### No stock data for micro-cap stocks
- Configure ALPHA_VANTAGE_API_KEY for fallback coverage
- Get free key at: https://www.alphavantage.co/support/#api-key

### No stock data
- Verify FINNHUB_API_KEY is set correctly
- Check API rate limits (60/min free tier)

### Vision OCR fails
- Ensure image is under 5MB
- Use supported format (PNG, JPG, etc.)

## Development

### Run MCP Server Standalone
```bash
python -m mcp_server.financial_mcp
```

### Test Stock Data Service
```python
from services.stock_data_service import fetch_stock_data
ctx = fetch_stock_data('AAPL')
print(ctx.data_available)  # Shows source: 'finnhub' or 'alpha_vantage'
```

### Database Location
SQLite database is created at:
`outputs/stock_advisor.db`

Contains tables:
- `alerts` - Price alerts
- `portfolio` - Stock positions
- `watchlist` - Watched tickers
- `sentiment_cache` - KOL sentiment history
- `alert_history` - Triggered alert log
- `sessions` - Chat sessions
- `chat_history` - Chat messages
- `trades` - Trade history for performance tracking
- `daily_snapshots` - Portfolio snapshots
- `kol_claims` - KOL claim tracking

## Language Support

Supports **English**, **繁體中文**, and **简体中文**.
Experts respond in the same language as the user's query.

## Quick Commands

```bash
# Run the app
chainlit run app_sa.py

# Development mode (auto-reload)
chainlit run app_sa.py -w

# Run CI workflow tests
python3 tests/test_ci_workflow.py

# Check syntax
python3 -m py_compile app_sa.py services/grok_service.py

# Test imports
python3 -c "from services import GrokService, fetch_stock_data; print('OK')"

# Run improvement workflow
# Read PROMPT.md and execute phases 1-9

# Reset if broken
git checkout .
```

## Auto-Improve Workflow

See `PROMPT.md` for the 9-phase autonomous improvement workflow:

1. **Environment Check** - Verify API keys and syntax
2. **Run Tests** - Execute test_ci_workflow.py
3. **Test Research Pipeline** - Verify CI dimension detection
4. **Test Expert Context** - Test stock data + context building
5. **Code Quality Scan** - Find issues
6. **Identify Improvements** - LLM-generated recommendations
7. **Implement Changes** - Make code modifications
8. **Quality Review** - Score calculation (target: 8/10)
9. **Commit & Report** - Git commit if score >= 8

---
*Last updated: 2026-01-19*
