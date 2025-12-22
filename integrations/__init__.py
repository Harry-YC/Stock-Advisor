"""
External API Integrations for Stock Advisor

Modules:
- finnhub: Finnhub API for real-time stock data
- gemini_vision: Gemini Vision for KOL screenshot OCR
- market_search: Google Search grounding for market news
"""

from integrations.finnhub import FinnhubClient
from integrations.gemini_vision import GeminiVisionClient, analyze_kol_screenshot
from integrations.market_search import MarketSearchClient, search_stock_news

__all__ = [
    'FinnhubClient',
    'GeminiVisionClient',
    'analyze_kol_screenshot',
    'MarketSearchClient',
    'search_stock_news',
]
