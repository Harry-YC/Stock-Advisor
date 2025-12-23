"""
Financial MCP Server

Model Context Protocol server providing stock-related tools:
- stock_quote: Get real-time price and metrics
- stock_screen: Screen by PE, market cap, sector
- add_alert: Set price alert (above/below threshold)
- list_alerts: List active alerts
- delete_alert: Remove an alert
- portfolio_add: Add position with cost basis
- portfolio_view: View holdings with P&L
- portfolio_remove: Sell/reduce position
- kol_sentiment: Get aggregated KOL sentiment

Uses MCP SDK for tool definitions and handlers.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.database import get_database, PriceAlert, PortfolioPosition

logger = logging.getLogger(__name__)

# Lazy-loaded clients
_finnhub_client = None


def _get_finnhub():
    """Lazy-load Finnhub client."""
    global _finnhub_client
    if _finnhub_client is None:
        try:
            from integrations.finnhub import FinnhubClient
            _finnhub_client = FinnhubClient()
        except ImportError:
            logger.warning("Finnhub client not available")
            return None
    return _finnhub_client


# =============================================================================
# MCP TOOL DEFINITIONS
# =============================================================================

TOOL_DEFINITIONS = [
    {
        "name": "stock_quote",
        "description": "Get real-time stock quote with price, change, and key metrics",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., NVDA, AAPL)"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "stock_financials",
        "description": "Get fundamental financial metrics for a stock",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "stock_news",
        "description": "Get recent news articles for a stock",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of articles (default 5)",
                    "default": 5
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "add_alert",
        "description": "Set a price alert for a stock",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "condition": {
                    "type": "string",
                    "enum": ["above", "below"],
                    "description": "Trigger when price goes above or below threshold"
                },
                "target_price": {
                    "type": "number",
                    "description": "Price threshold for the alert"
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes about the alert"
                }
            },
            "required": ["symbol", "condition", "target_price"]
        }
    },
    {
        "name": "list_alerts",
        "description": "List all active price alerts",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Optional: filter by symbol"
                }
            }
        }
    },
    {
        "name": "delete_alert",
        "description": "Delete a price alert",
        "inputSchema": {
            "type": "object",
            "properties": {
                "alert_id": {
                    "type": "integer",
                    "description": "ID of the alert to delete"
                }
            },
            "required": ["alert_id"]
        }
    },
    {
        "name": "portfolio_add",
        "description": "Add shares to your portfolio (creates position or averages in)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "shares": {
                    "type": "number",
                    "description": "Number of shares to add"
                },
                "price": {
                    "type": "number",
                    "description": "Purchase price per share"
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes"
                }
            },
            "required": ["symbol", "shares", "price"]
        }
    },
    {
        "name": "portfolio_view",
        "description": "View portfolio holdings with current P&L",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Optional: view specific position"
                }
            }
        }
    },
    {
        "name": "portfolio_sell",
        "description": "Sell/reduce shares from a position",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "shares": {
                    "type": "number",
                    "description": "Number of shares to sell"
                }
            },
            "required": ["symbol", "shares"]
        }
    },
    {
        "name": "kol_sentiment",
        "description": "Get aggregated KOL sentiment for a stock",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "days": {
                    "type": "integer",
                    "description": "Look back period in days (default 7)",
                    "default": 7
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "watchlist_add",
        "description": "Add a stock to your watchlist",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes about why you're watching"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags (e.g., 'AI', 'dividend')"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "watchlist_view",
        "description": "View your watchlist",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "watchlist_remove",
        "description": "Remove a stock from your watchlist",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol to remove"
                }
            },
            "required": ["symbol"]
        }
    }
]


# =============================================================================
# TOOL HANDLERS
# =============================================================================

def handle_stock_quote(symbol: str) -> Dict[str, Any]:
    """Get real-time stock quote."""
    finnhub = _get_finnhub()
    if not finnhub or not finnhub.is_available():
        return {"error": "Finnhub API not configured. Set FINNHUB_API_KEY in environment."}

    try:
        quote = finnhub.get_quote(symbol)
        if not quote:
            return {"error": f"No quote data for {symbol}"}

        return {
            "symbol": symbol.upper(),
            "current_price": quote.current_price,
            "change": quote.change,
            "change_percent": quote.change_percent,
            "high": quote.high,
            "low": quote.low,
            "open": quote.open,
            "previous_close": quote.previous_close,
            "timestamp": quote.timestamp.isoformat() if quote.timestamp else None,
            "formatted": quote.format_summary(),
        }
    except Exception as e:
        logger.error(f"Quote fetch failed: {e}")
        return {"error": str(e)}


def handle_stock_financials(symbol: str) -> Dict[str, Any]:
    """Get fundamental financial metrics."""
    finnhub = _get_finnhub()
    if not finnhub or not finnhub.is_available():
        return {"error": "Finnhub API not configured"}

    try:
        financials = finnhub.get_basic_financials(symbol)
        if not financials:
            return {"error": f"No financial data for {symbol}"}

        return {
            "symbol": symbol.upper(),
            "pe_ratio": financials.pe_ratio,
            "pb_ratio": financials.pb_ratio,
            "ps_ratio": financials.ps_ratio,
            "eps": financials.eps,
            "dividend_yield": financials.dividend_yield,
            "beta": financials.beta,
            "52_week_high": financials.high_52w,
            "52_week_low": financials.low_52w,
            "gross_margin": financials.gross_margin,
            "operating_margin": financials.operating_margin,
            "roe": financials.roe,
            "formatted": financials.format_summary(),
        }
    except Exception as e:
        logger.error(f"Financials fetch failed: {e}")
        return {"error": str(e)}


def handle_stock_news(symbol: str, limit: int = 5) -> Dict[str, Any]:
    """Get recent news articles."""
    finnhub = _get_finnhub()
    if not finnhub or not finnhub.is_available():
        return {"error": "Finnhub API not configured"}

    try:
        news = finnhub.get_company_news(symbol, limit=limit)
        if not news:
            return {"symbol": symbol.upper(), "articles": [], "message": "No recent news"}

        articles = []
        for item in news:
            articles.append({
                "headline": item.headline,
                "summary": item.summary[:200] + "..." if len(item.summary) > 200 else item.summary,
                "source": item.source,
                "url": item.url,
                "datetime": item.datetime.isoformat() if item.datetime else None,
            })

        return {
            "symbol": symbol.upper(),
            "articles": articles,
            "count": len(articles),
        }
    except Exception as e:
        logger.error(f"News fetch failed: {e}")
        return {"error": str(e)}


def handle_add_alert(
    symbol: str,
    condition: str,
    target_price: float,
    notes: str = ""
) -> Dict[str, Any]:
    """Add a price alert."""
    try:
        db = get_database()
        alert = db.add_alert(symbol, condition, target_price, notes)
        return {
            "success": True,
            "alert_id": alert.id,
            "message": f"Alert created: {alert.format_description()}",
            "alert": alert.to_dict(),
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Add alert failed: {e}")
        return {"success": False, "error": str(e)}


def handle_list_alerts(symbol: Optional[str] = None) -> Dict[str, Any]:
    """List active alerts."""
    try:
        db = get_database()
        alerts = db.get_active_alerts(symbol)

        if not alerts:
            return {
                "alerts": [],
                "count": 0,
                "message": "No active alerts" + (f" for {symbol}" if symbol else ""),
            }

        return {
            "alerts": [a.to_dict() for a in alerts],
            "count": len(alerts),
            "formatted": "\n".join([f"[{a.id}] {a.format_description()}" for a in alerts]),
        }
    except Exception as e:
        logger.error(f"List alerts failed: {e}")
        return {"error": str(e)}


def handle_delete_alert(alert_id: int) -> Dict[str, Any]:
    """Delete an alert."""
    try:
        db = get_database()
        deleted = db.delete_alert(alert_id)
        if deleted:
            return {"success": True, "message": f"Alert {alert_id} deleted"}
        return {"success": False, "error": f"Alert {alert_id} not found"}
    except Exception as e:
        logger.error(f"Delete alert failed: {e}")
        return {"success": False, "error": str(e)}


def handle_portfolio_add(
    symbol: str,
    shares: float,
    price: float,
    notes: str = ""
) -> Dict[str, Any]:
    """Add shares to portfolio."""
    try:
        db = get_database()
        position = db.add_position(symbol, shares, price, notes)

        # Get current price for P&L
        finnhub = _get_finnhub()
        current_price = None
        if finnhub and finnhub.is_available():
            quote = finnhub.get_quote(symbol)
            if quote:
                current_price = quote.current_price

        return {
            "success": True,
            "message": f"Added {shares} shares of {symbol} @ ${price:.2f}",
            "position": position.to_dict(),
            "summary": position.format_summary(current_price),
        }
    except Exception as e:
        logger.error(f"Portfolio add failed: {e}")
        return {"success": False, "error": str(e)}


def handle_portfolio_view(symbol: Optional[str] = None) -> Dict[str, Any]:
    """View portfolio with P&L."""
    try:
        db = get_database()
        finnhub = _get_finnhub()

        if symbol:
            position = db.get_position(symbol)
            if not position:
                return {"error": f"No position for {symbol}"}

            current_price = None
            if finnhub and finnhub.is_available():
                quote = finnhub.get_quote(symbol)
                if quote:
                    current_price = quote.current_price

            pnl_dollars, pnl_percent = position.calculate_pnl(current_price) if current_price else (0, 0)

            return {
                "position": position.to_dict(),
                "current_price": current_price,
                "pnl_dollars": pnl_dollars,
                "pnl_percent": pnl_percent,
                "summary": position.format_summary(current_price),
            }

        # Get all positions
        positions = db.get_portfolio()
        if not positions:
            return {"positions": [], "count": 0, "message": "Portfolio is empty"}

        total_cost = 0
        total_value = 0
        position_summaries = []

        for pos in positions:
            current_price = None
            if finnhub and finnhub.is_available():
                quote = finnhub.get_quote(pos.symbol)
                if quote:
                    current_price = quote.current_price

            total_cost += pos.cost_basis
            if current_price:
                total_value += pos.shares * current_price

            position_summaries.append({
                "position": pos.to_dict(),
                "current_price": current_price,
                "summary": pos.format_summary(current_price),
            })

        total_pnl = total_value - total_cost if total_value > 0 else 0
        total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0

        return {
            "positions": position_summaries,
            "count": len(positions),
            "total_cost": total_cost,
            "total_value": total_value,
            "total_pnl": total_pnl,
            "total_pnl_percent": total_pnl_percent,
        }
    except Exception as e:
        logger.error(f"Portfolio view failed: {e}")
        return {"error": str(e)}


def handle_portfolio_sell(symbol: str, shares: float) -> Dict[str, Any]:
    """Sell shares from portfolio."""
    try:
        db = get_database()
        position = db.reduce_position(symbol, shares)

        if position:
            return {
                "success": True,
                "message": f"Sold {shares} shares of {symbol}",
                "remaining": position.to_dict(),
            }
        else:
            return {
                "success": True,
                "message": f"Position {symbol} closed (all shares sold)",
                "remaining": None,
            }
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Portfolio sell failed: {e}")
        return {"success": False, "error": str(e)}


def handle_kol_sentiment(symbol: str, days: int = 7) -> Dict[str, Any]:
    """Get aggregated KOL sentiment."""
    try:
        db = get_database()
        summary = db.get_sentiment_summary(symbol, days)

        if summary["total_entries"] == 0:
            return {
                "symbol": symbol.upper(),
                "message": f"No KOL sentiment data for {symbol} in the last {days} days",
                "suggestion": "Upload KOL screenshots to build sentiment history",
            }

        return {
            "symbol": symbol.upper(),
            "period_days": days,
            "total_entries": summary["total_entries"],
            "sentiment_breakdown": {
                "bullish": summary["bullish"],
                "bearish": summary["bearish"],
                "neutral": summary["neutral"],
                "mixed": summary["mixed"],
            },
            "dominant_sentiment": summary["dominant_sentiment"],
            "key_claims": summary["key_claims"],
        }
    except Exception as e:
        logger.error(f"KOL sentiment failed: {e}")
        return {"error": str(e)}


def handle_watchlist_add(
    symbol: str,
    notes: str = "",
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Add stock to watchlist."""
    try:
        db = get_database()
        item = db.add_to_watchlist(symbol, notes, tags)
        return {
            "success": True,
            "message": f"Added {symbol} to watchlist",
            "item": item.to_dict(),
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Watchlist add failed: {e}")
        return {"success": False, "error": str(e)}


def handle_watchlist_view() -> Dict[str, Any]:
    """View watchlist."""
    try:
        db = get_database()
        items = db.get_watchlist()

        if not items:
            return {"items": [], "count": 0, "message": "Watchlist is empty"}

        return {
            "items": [item.to_dict() for item in items],
            "count": len(items),
            "symbols": [item.symbol for item in items],
        }
    except Exception as e:
        logger.error(f"Watchlist view failed: {e}")
        return {"error": str(e)}


def handle_watchlist_remove(symbol: str) -> Dict[str, Any]:
    """Remove stock from watchlist."""
    try:
        db = get_database()
        removed = db.remove_from_watchlist(symbol)
        if removed:
            return {"success": True, "message": f"Removed {symbol} from watchlist"}
        return {"success": False, "error": f"{symbol} not found in watchlist"}
    except Exception as e:
        logger.error(f"Watchlist remove failed: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# TOOL DISPATCHER
# =============================================================================

TOOL_HANDLERS = {
    "stock_quote": lambda args: handle_stock_quote(args["symbol"]),
    "stock_financials": lambda args: handle_stock_financials(args["symbol"]),
    "stock_news": lambda args: handle_stock_news(args["symbol"], args.get("limit", 5)),
    "add_alert": lambda args: handle_add_alert(
        args["symbol"], args["condition"], args["target_price"], args.get("notes", "")
    ),
    "list_alerts": lambda args: handle_list_alerts(args.get("symbol")),
    "delete_alert": lambda args: handle_delete_alert(args["alert_id"]),
    "portfolio_add": lambda args: handle_portfolio_add(
        args["symbol"], args["shares"], args["price"], args.get("notes", "")
    ),
    "portfolio_view": lambda args: handle_portfolio_view(args.get("symbol")),
    "portfolio_sell": lambda args: handle_portfolio_sell(args["symbol"], args["shares"]),
    "kol_sentiment": lambda args: handle_kol_sentiment(args["symbol"], args.get("days", 7)),
    "watchlist_add": lambda args: handle_watchlist_add(
        args["symbol"], args.get("notes", ""), args.get("tags")
    ),
    "watchlist_view": lambda args: handle_watchlist_view(),
    "watchlist_remove": lambda args: handle_watchlist_remove(args["symbol"]),
}


def dispatch_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a tool call to the appropriate handler."""
    if name not in TOOL_HANDLERS:
        return {"error": f"Unknown tool: {name}"}

    try:
        return TOOL_HANDLERS[name](arguments)
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return {"error": str(e)}


# =============================================================================
# MCP SERVER (if running standalone)
# =============================================================================

def create_mcp_server():
    """Create MCP server instance (for standalone mode)."""
    try:
        from mcp import Server, types
    except ImportError:
        logger.error("MCP package not installed. Run: pip install mcp")
        return None

    server = Server("financial-mcp")

    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        """List available tools."""
        return [
            types.Tool(
                name=tool["name"],
                description=tool["description"],
                inputSchema=tool["inputSchema"],
            )
            for tool in TOOL_DEFINITIONS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle tool calls."""
        import json
        result = dispatch_tool(name, arguments)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    return server


async def run_server():
    """Run the MCP server."""
    server = create_mcp_server()
    if not server:
        return

    from mcp import stdio

    async with stdio.stdio_server() as (read, write):
        await server.run(read, write)


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_server())
