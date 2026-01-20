"""
Services Layer for Stock Advisor

Business logic services for the stock analysis application.

Services:
- stock_data_service: Aggregate stock data from multiple sources
- llm_router: Routes LLM calls to appropriate models
- grok_service: X/Twitter KOL insights via Grok

Usage:
    from services import (
        fetch_stock_data,
        extract_tickers,
        LLMRouter,
        GrokService,
        get_grok_service,
    )
"""

# Stock data fetching
from services.stock_data_service import (
    fetch_stock_data,
    fetch_multi_stock_data,
    extract_tickers,
    build_expert_context,
    build_kol_research_context,
    get_category_insights,
    analyze_kol_screenshot,
    StockDataContext,
)

# LLM routing
from services.llm_router import LLMRouter, get_llm_router

# Grok/X Twitter insights
from services.grok_service import (
    GrokService,
    get_grok_service,
    detect_stock_ci_dimensions,
    get_stock_pulse,
    get_known_kols,
    get_kol_profiles,
    get_kol_categories,
    get_high_signal_kols,
    deep_research,
    synthesize_views,
    search_category_kols,
    STOCK_KOLS,
    STOCK_CI_DIMENSIONS,
    KOL_PROFILES,
    KOL_CATEGORIES,
    HIGH_SIGNAL_KOLS,
)

__all__ = [
    # Stock data
    'fetch_stock_data',
    'fetch_multi_stock_data',
    'extract_tickers',
    'build_expert_context',
    'build_kol_research_context',
    'get_category_insights',
    'analyze_kol_screenshot',
    'StockDataContext',

    # LLM
    'LLMRouter',
    'get_llm_router',

    # Grok/X - Core
    'GrokService',
    'get_grok_service',
    'detect_stock_ci_dimensions',
    'get_stock_pulse',

    # Grok/X - Enhanced KOL functions
    'get_known_kols',
    'get_kol_profiles',
    'get_kol_categories',
    'get_high_signal_kols',
    'deep_research',
    'synthesize_views',
    'search_category_kols',

    # Grok/X - Data
    'STOCK_KOLS',
    'STOCK_CI_DIMENSIONS',
    'KOL_PROFILES',
    'KOL_CATEGORIES',
    'HIGH_SIGNAL_KOLS',
]
