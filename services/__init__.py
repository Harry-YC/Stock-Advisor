"""
Services Layer for Stock Advisor

Business logic services for the stock analysis application.

Services:
- stock_data_service: Aggregate stock data from multiple sources
- llm_router: Routes LLM calls to appropriate models

Usage:
    from services import (
        fetch_stock_data,
        extract_tickers,
        LLMRouter,
    )
"""

# Stock data fetching
from services.stock_data_service import (
    fetch_stock_data,
    fetch_multi_stock_data,
    extract_tickers,
    build_expert_context,
    analyze_kol_screenshot,
    StockDataContext,
)

# LLM routing
from services.llm_router import LLMRouter, get_llm_router

__all__ = [
    # Stock data
    'fetch_stock_data',
    'fetch_multi_stock_data',
    'extract_tickers',
    'build_expert_context',
    'analyze_kol_screenshot',
    'StockDataContext',

    # LLM
    'LLMRouter',
    'get_llm_router',
]
