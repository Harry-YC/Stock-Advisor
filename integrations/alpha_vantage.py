"""
Alpha Vantage API Integration

Fallback data source for stocks not covered by Finnhub.
https://www.alphavantage.co/documentation/

Free tier: 25 API calls/day (use sparingly as fallback)
"""

import os
import logging
import requests
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class AVQuote:
    """Alpha Vantage stock quote."""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    latest_trading_day: str
    previous_close: float
    open: float
    high: float
    low: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "current_price": self.price,
            "change": self.change,
            "percent_change": self.change_percent,
            "volume": self.volume,
            "latest_trading_day": self.latest_trading_day,
            "previous_close": self.previous_close,
            "open": self.open,
            "high": self.high,
            "low": self.low,
        }

    @property
    def change_emoji(self) -> str:
        if self.change_percent > 2:
            return "🚀"
        elif self.change_percent > 0:
            return "📈"
        elif self.change_percent < -2:
            return "📉"
        elif self.change_percent < 0:
            return "🔻"
        return "➖"

    def format_summary(self) -> str:
        sign = "+" if self.change >= 0 else ""
        return (
            f"{self.change_emoji} **{self.symbol}**: ${self.price:.2f} "
            f"({sign}{self.change:.2f}, {sign}{self.change_percent:.2f}%)\n"
            f"   High: ${self.high:.2f} | Low: ${self.low:.2f} | Vol: {self.volume:,}"
        )


@dataclass
class AVOverview:
    """Alpha Vantage company overview (fundamentals)."""
    symbol: str
    name: str
    description: str
    exchange: str
    currency: str
    country: str
    sector: str
    industry: str
    market_cap: float
    pe_ratio: Optional[float]
    peg_ratio: Optional[float]
    book_value: Optional[float]
    dividend_yield: Optional[float]
    eps: Optional[float]
    revenue_ttm: Optional[float]
    profit_margin: Optional[float]
    beta: Optional[float]
    high_52w: Optional[float]
    low_52w: Optional[float]
    moving_avg_50: Optional[float]
    moving_avg_200: Optional[float]
    shares_outstanding: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def format_summary(self) -> str:
        lines = [f"**{self.name}** ({self.symbol})"]
        lines.append(f"Sector: {self.sector} | Industry: {self.industry}")
        lines.append(f"Exchange: {self.exchange} | Country: {self.country}")

        if self.market_cap:
            cap_str = self._format_large_number(self.market_cap)
            lines.append(f"Market Cap: {cap_str}")

        lines.append("\n**Valuation:**")
        if self.pe_ratio:
            lines.append(f"  P/E: {self.pe_ratio:.2f}")
        if self.peg_ratio:
            lines.append(f"  PEG: {self.peg_ratio:.2f}")
        if self.eps:
            lines.append(f"  EPS: ${self.eps:.2f}")

        if self.high_52w and self.low_52w:
            lines.append(f"\n**52-Week Range:** ${self.low_52w:.2f} - ${self.high_52w:.2f}")

        if self.moving_avg_50 or self.moving_avg_200:
            lines.append("\n**Moving Averages:**")
            if self.moving_avg_50:
                lines.append(f"  50-day: ${self.moving_avg_50:.2f}")
            if self.moving_avg_200:
                lines.append(f"  200-day: ${self.moving_avg_200:.2f}")

        return "\n".join(lines)

    @staticmethod
    def _format_large_number(num: float) -> str:
        if num >= 1e12:
            return f"${num/1e12:.2f}T"
        elif num >= 1e9:
            return f"${num/1e9:.2f}B"
        elif num >= 1e6:
            return f"${num/1e6:.2f}M"
        return f"${num:,.0f}"


class AlphaVantageClient:
    """
    Client for Alpha Vantage API.

    Used as fallback when Finnhub doesn't have data.
    Free tier is limited (25 calls/day), so use sparingly.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    # Cache TTLs (longer than Finnhub since we have limited calls)
    QUOTE_CACHE_TTL = 600  # 10 minutes
    OVERVIEW_CACHE_TTL = 86400  # 24 hours

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(settings, 'ALPHA_VANTAGE_API_KEY', None) or os.getenv("ALPHA_VANTAGE_API_KEY")
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._call_count = 0
        self._last_reset = datetime.now()

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits (25/day free tier)."""
        now = datetime.now()
        # Reset counter daily
        if (now - self._last_reset).days >= 1:
            self._call_count = 0
            self._last_reset = now

        if self._call_count >= 25:
            logger.warning("Alpha Vantage daily rate limit reached")
            return False
        return True

    def _get_cache(self, key: str, ttl: int) -> Optional[Any]:
        """Get item from cache if not expired."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if datetime.now() - entry["timestamp"] < timedelta(seconds=ttl):
                    logger.debug(f"AV Cache hit: {key}")
                    return entry["data"]
        return None

    def _set_cache(self, key: str, data: Any) -> None:
        """Set item in cache."""
        with self._lock:
            self._cache[key] = {
                "data": data,
                "timestamp": datetime.now()
            }

    def _make_request(self, function: str, params: Optional[Dict] = None) -> Dict:
        """Make API request."""
        if not self.is_available():
            raise RuntimeError("Alpha Vantage API key not configured")

        if not self._check_rate_limit():
            raise RuntimeError("Alpha Vantage daily rate limit exceeded")

        params = params or {}
        params["function"] = function
        params["apikey"] = self.api_key

        try:
            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=15,
            )
            response.raise_for_status()
            self._call_count += 1

            data = response.json()

            # Check for API errors
            if "Error Message" in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return {}
            if "Note" in data:
                logger.warning(f"Alpha Vantage note: {data['Note']}")
                return {}

            return data

        except requests.RequestException as e:
            logger.error(f"Alpha Vantage request failed: {e}")
            raise

    def get_quote(self, symbol: str) -> Optional[AVQuote]:
        """
        Get real-time quote for a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            AVQuote object or None
        """
        symbol = symbol.upper()
        cache_key = f"av_quote:{symbol}"

        # Check cache
        cached = self._get_cache(cache_key, self.QUOTE_CACHE_TTL)
        if cached:
            return cached

        try:
            data = self._make_request("GLOBAL_QUOTE", {"symbol": symbol})

            if not data or "Global Quote" not in data:
                logger.warning(f"No AV quote data for {symbol}")
                return None

            q = data["Global Quote"]
            if not q or "05. price" not in q:
                logger.warning(f"Empty AV quote for {symbol}")
                return None

            quote = AVQuote(
                symbol=symbol,
                price=float(q.get("05. price", 0)),
                change=float(q.get("09. change", 0)),
                change_percent=float(q.get("10. change percent", "0%").rstrip("%")),
                volume=int(q.get("06. volume", 0)),
                latest_trading_day=q.get("07. latest trading day", ""),
                previous_close=float(q.get("08. previous close", 0)),
                open=float(q.get("02. open", 0)),
                high=float(q.get("03. high", 0)),
                low=float(q.get("04. low", 0)),
            )

            self._set_cache(cache_key, quote)
            logger.info(f"AV: Retrieved quote for {symbol}: ${quote.price}")
            return quote

        except Exception as e:
            logger.error(f"AV quote fetch failed for {symbol}: {e}")
            return None

    def get_company_overview(self, symbol: str) -> Optional[AVOverview]:
        """
        Get company overview/fundamentals.

        Args:
            symbol: Stock ticker symbol

        Returns:
            AVOverview object or None
        """
        symbol = symbol.upper()
        cache_key = f"av_overview:{symbol}"

        cached = self._get_cache(cache_key, self.OVERVIEW_CACHE_TTL)
        if cached:
            return cached

        try:
            data = self._make_request("OVERVIEW", {"symbol": symbol})

            if not data or "Symbol" not in data:
                logger.warning(f"No AV overview for {symbol}")
                return None

            def safe_float(val, default=None):
                if val in (None, "", "None", "-"):
                    return default
                try:
                    return float(val)
                except:
                    return default

            overview = AVOverview(
                symbol=symbol,
                name=data.get("Name", symbol),
                description=data.get("Description", ""),
                exchange=data.get("Exchange", ""),
                currency=data.get("Currency", "USD"),
                country=data.get("Country", ""),
                sector=data.get("Sector", ""),
                industry=data.get("Industry", ""),
                market_cap=safe_float(data.get("MarketCapitalization"), 0),
                pe_ratio=safe_float(data.get("PERatio")),
                peg_ratio=safe_float(data.get("PEGRatio")),
                book_value=safe_float(data.get("BookValue")),
                dividend_yield=safe_float(data.get("DividendYield")),
                eps=safe_float(data.get("EPS")),
                revenue_ttm=safe_float(data.get("RevenueTTM")),
                profit_margin=safe_float(data.get("ProfitMargin")),
                beta=safe_float(data.get("Beta")),
                high_52w=safe_float(data.get("52WeekHigh")),
                low_52w=safe_float(data.get("52WeekLow")),
                moving_avg_50=safe_float(data.get("50DayMovingAverage")),
                moving_avg_200=safe_float(data.get("200DayMovingAverage")),
                shares_outstanding=safe_float(data.get("SharesOutstanding")),
            )

            self._set_cache(cache_key, overview)
            logger.info(f"AV: Retrieved overview for {symbol}: {overview.name}")
            return overview

        except Exception as e:
            logger.error(f"AV overview fetch failed for {symbol}: {e}")
            return None

    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive stock data (quote + overview).

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with all available data
        """
        result = {
            "symbol": symbol.upper(),
            "source": "alpha_vantage",
            "quote": None,
            "overview": None,
            "summary": "",
        }

        quote = self.get_quote(symbol)
        if quote:
            result["quote"] = quote.to_dict()

        overview = self.get_company_overview(symbol)
        if overview:
            result["overview"] = overview.to_dict()

        # Build summary
        parts = []
        if quote:
            parts.append(quote.format_summary())
        if overview:
            parts.append("\n" + overview.format_summary())

        result["summary"] = "\n".join(parts) if parts else f"No data available for {symbol}"
        return result


# Convenience functions
def get_alpha_vantage_quote(symbol: str) -> Optional[AVQuote]:
    """Quick quote lookup via Alpha Vantage."""
    client = AlphaVantageClient()
    if not client.is_available():
        logger.warning("Alpha Vantage API not configured")
        return None
    return client.get_quote(symbol)


def get_alpha_vantage_data(symbol: str) -> Dict[str, Any]:
    """Quick comprehensive data lookup via Alpha Vantage."""
    client = AlphaVantageClient()
    if not client.is_available():
        logger.warning("Alpha Vantage API not configured")
        return {"symbol": symbol, "error": "API not configured"}
    return client.get_stock_data(symbol)
