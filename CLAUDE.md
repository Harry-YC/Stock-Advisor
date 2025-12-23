# CLAUDE.md - AI Assistant Context

## Project Overview

**Stock Advisor** is an AI-powered stock analysis app using Chainlit. Users can ask about stocks, upload KOL screenshots, and get personalized analysis from 7 AI stock experts with real-time data from Finnhub (with Alpha Vantage fallback) and Google Search.

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
│   ├── stock_data_service.py     # Multi-source data aggregation (Finnhub + Alpha Vantage)
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

---
*Last updated: 2025-12-23*
