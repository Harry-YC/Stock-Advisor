# Stock Advisor

AI-powered stock analysis assistant with 6 expert perspectives and real-time market data.

## Features

- **6 AI Stock Experts** - Bull, Bear, Technical, Fundamental, Sentiment, Risk analysts
- **KOL Screenshot Analysis** - Upload Twitter/X, Reddit, StockTwits screenshots for OCR + sentiment
- **Real-Time Stock Data** - Finnhub integration for quotes, financials, news
- **Market News Search** - Google Search grounding for latest market developments
- **MCP Tools** - Portfolio tracking, price alerts, watchlist management

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the app
chainlit run app_sa.py

# Development mode (auto-reload)
chainlit run app_sa.py -w
```

## Environment Variables

```bash
# Required
GEMINI_API_KEY=xxx              # From aistudio.google.com

# Recommended
FINNHUB_API_KEY=xxx             # From finnhub.io (free: 60 calls/min)
```

## Expert Panel

| Expert | Focus |
|--------|-------|
| Bull Analyst | Growth catalysts, upside targets, bullish scenarios |
| Bear Analyst | Risk factors, downside targets, valuation concerns |
| Technical Analyst | Chart patterns, support/resistance, indicators |
| Fundamental Analyst | Financials, valuation metrics, DCF analysis |
| Sentiment Analyst | News sentiment, KOL analysis, social trends |
| Risk Manager | Position sizing, hedging, stop-loss strategy |

## Expert Presets

| Preset | Experts |
|--------|---------|
| Quick Analysis | Bull, Bear, Technical |
| Deep Dive | Bull, Bear, Technical, Fundamental, Risk |
| KOL Review | Sentiment, Bull, Bear |
| Trade Planning | Technical, Risk |
| Full Panel | All 6 experts |

## Usage Examples

```
"Analyze NVDA"              # Full expert panel analysis
"Why did TSLA fall today?"  # Search for movement explanation
"Ask Bull about AAPL"       # Consult specific expert
[Upload screenshot]         # KOL screenshot analysis
```

## Project Structure

```
Stock Advisor/
├── app_sa.py                 # Main Chainlit app
├── stocks/
│   └── stock_personas.py     # 6 Expert definitions
├── integrations/
│   ├── finnhub.py            # Finnhub API client
│   ├── gemini_vision.py      # KOL screenshot OCR
│   └── market_search.py      # Google Search grounding
├── services/
│   ├── stock_data_service.py # Data aggregation
│   └── llm_router.py         # LLM routing
├── mcp_server/
│   ├── database.py           # SQLite persistence
│   └── financial_mcp.py      # MCP tools
└── config/
    └── settings.py           # Configuration
```

## AI Model

Uses **Google Gemini** for all AI features:
- **Gemini 3 Pro** (`gemini-3-pro-preview`) - Expert panel discussions
- **Gemini 2.0 Flash** - Vision OCR and Search grounding

## API Costs

| API | Free Tier |
|-----|-----------|
| Gemini | Generous free tier |
| Finnhub | 60 calls/min |

---

**Version**: 1.0
**Last Updated**: December 2025
