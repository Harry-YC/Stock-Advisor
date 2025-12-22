# CLAUDE.md - AI Assistant Context

## Project Overview

**Stock Advisor** is an AI-powered stock analysis app using Chainlit. Users can ask about stocks, upload KOL screenshots, and get personalized analysis from 6 AI stock experts with real-time data from Finnhub and Google Search.

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
4. **Follow-up questions** - Ask specific experts for more details
5. **Track portfolio** - Use MCP tools for alerts and positions

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Chainlit 1.3+ |
| AI Model | Google Gemini 3 Pro/Flash |
| Stock Data | Finnhub API |
| News Search | Google Search Grounding |
| Vision OCR | Gemini Vision (2.0 Flash) |
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
│   └── stock_personas.py         # 6 Expert definitions + presets
│
├── integrations/
│   ├── finnhub.py                # Finnhub API client (quotes, financials, news)
│   ├── gemini_vision.py          # KOL screenshot OCR
│   └── market_search.py          # Google Search grounding for stocks
│
├── services/
│   ├── llm_router.py             # Gemini API routing
│   └── stock_data_service.py     # Multi-source data aggregation
│
├── mcp_server/
│   ├── database.py               # SQLite for alerts, portfolio, watchlist
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

# Optional
APP_ENV=dev                       # Environment (dev/prod)
API_TIMEOUT=240                   # API timeout in seconds
```

## 6 Stock Experts

| Expert | Icon | Focus |
|--------|------|-------|
| Bull Analyst | :green_circle: | Growth catalysts, upside targets, bullish scenarios |
| Bear Analyst | :red_circle: | Risk factors, downside targets, valuation concerns |
| Technical Analyst | :chart_with_upwards_trend: | Chart patterns, support/resistance, indicators |
| Fundamental Analyst | :bar_chart: | Financials, valuation metrics, DCF analysis |
| Sentiment Analyst | :speech_balloon: | News sentiment, KOL analysis, social trends |
| Risk Manager | :shield: | Position sizing, hedging, stop-loss strategy |

## Expert Presets

| Preset | Experts Included |
|--------|------------------|
| Quick Analysis | Bull, Bear, Technical |
| Deep Dive | Bull, Bear, Technical, Fundamental, Risk |
| KOL Review | Sentiment, Bull, Bear |
| Trade Planning | Technical, Risk |
| Full Panel | All 6 experts |

## Key Features

### KOL Screenshot Analysis
- Upload screenshots from Twitter/X, Reddit, StockTwits
- Automatic ticker extraction via Gemini Vision OCR
- Sentiment classification (bullish/bearish/neutral/mixed)
- Key claims extraction for expert validation

### Real-Time Stock Data
- Finnhub integration for quotes, financials, news
- 5-min cache for quotes, 1-hour for fundamentals
- Rate limiting (60 req/min on free tier)

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
| Google Search | Included with Gemini | - |

## Security Features

### Prompt Injection Protection
- `_sanitize_for_prompt()` in stock_data_service.py
- Blocks "ignore previous instructions" patterns
- Limits input length

### Thread-Safe Caching
- Finnhub client uses `threading.RLock()`
- All cache operations protected

### File Upload Limits
- Images: 5MB maximum
- Supported: PNG, JPG, JPEG, GIF, WebP

## Troubleshooting

### 403 PERMISSION_DENIED
- Create fresh key from AI Studio (not Cloud Console)
- Enable "Generative Language API" in Cloud Console

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

### Database Location
SQLite database is created at:
`outputs/stock_advisor.db`

Contains tables:
- `alerts` - Price alerts
- `portfolio` - Stock positions
- `watchlist` - Watched tickers
- `sentiment_cache` - KOL sentiment history
- `alert_history` - Triggered alert log
