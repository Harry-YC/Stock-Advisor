"""
Financial MCP Server Package

Provides MCP tools for:
- Stock quotes and financials
- Price alerts
- Portfolio tracking
- Watchlist management
- KOL sentiment aggregation
"""

from mcp_server.database import (
    get_database,
    FinancialDatabase,
    PriceAlert,
    PortfolioPosition,
    WatchlistItem,
    SentimentEntry,
)

from mcp_server.financial_mcp import (
    TOOL_DEFINITIONS,
    dispatch_tool,
    handle_stock_quote,
    handle_add_alert,
    handle_list_alerts,
    handle_portfolio_view,
)

__all__ = [
    "get_database",
    "FinancialDatabase",
    "PriceAlert",
    "PortfolioPosition",
    "WatchlistItem",
    "SentimentEntry",
    "TOOL_DEFINITIONS",
    "dispatch_tool",
    "handle_stock_quote",
    "handle_add_alert",
    "handle_list_alerts",
    "handle_portfolio_view",
]
