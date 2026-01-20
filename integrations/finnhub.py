"""
Finnhub API Integration

Real-time stock data for stock analysis.
https://finnhub.io/docs/api

Free tier: 60 API calls/minute
"""

import os
import logging
import requests
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class StockQuote:
    """Real-time stock quote."""
    symbol: str
    current_price: float
    change: float
    percent_change: float
    high: float
    low: float
    open: float
    previous_close: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "change": self.change,
            "percent_change": self.percent_change,
            "high": self.high,
            "low": self.low,
            "open": self.open,
            "previous_close": self.previous_close,
            "timestamp": self.timestamp.isoformat(),
        }

    @property
    def change_emoji(self) -> str:
        """Return emoji based on price change."""
        if self.percent_change > 2:
            return "🚀"
        elif self.percent_change > 0:
            return "📈"
        elif self.percent_change < -2:
            return "📉"
        elif self.percent_change < 0:
            return "🔻"
        return "➖"

    def format_summary(self) -> str:
        """Format quote as a readable summary."""
        sign = "+" if self.change >= 0 else ""
        return (
            f"{self.change_emoji} **{self.symbol}**: ${self.current_price:.2f} "
            f"({sign}{self.change:.2f}, {sign}{self.percent_change:.2f}%)\n"
            f"   High: ${self.high:.2f} | Low: ${self.low:.2f} | Open: ${self.open:.2f}"
        )


@dataclass
class CompanyProfile:
    """Company profile information."""
    symbol: str
    name: str
    country: str
    currency: str
    exchange: str
    industry: str
    logo: str
    market_cap: float
    shares_outstanding: float
    website: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "country": self.country,
            "currency": self.currency,
            "exchange": self.exchange,
            "industry": self.industry,
            "logo": self.logo,
            "market_cap": self.market_cap,
            "shares_outstanding": self.shares_outstanding,
            "website": self.website,
        }

    def format_summary(self) -> str:
        """Format profile as readable summary."""
        market_cap_str = self._format_large_number(self.market_cap)
        return (
            f"**{self.name}** ({self.symbol})\n"
            f"Industry: {self.industry}\n"
            f"Exchange: {self.exchange} | Country: {self.country}\n"
            f"Market Cap: {market_cap_str}"
        )

    @staticmethod
    def _format_large_number(num: float) -> str:
        """Format large numbers with B/M suffix."""
        if num >= 1e12:
            return f"${num/1e12:.2f}T"
        elif num >= 1e9:
            return f"${num/1e9:.2f}B"
        elif num >= 1e6:
            return f"${num/1e6:.2f}M"
        return f"${num:,.0f}"


@dataclass
class BasicFinancials:
    """Basic financial metrics."""
    symbol: str
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    eps: Optional[float] = None
    eps_growth: Optional[float] = None
    revenue_growth: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    beta: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def format_summary(self) -> str:
        """Format financials as readable summary."""
        lines = [f"**Financial Metrics for {self.symbol}**\n"]

        # Valuation
        lines.append("**Valuation:**")
        if self.pe_ratio:
            lines.append(f"  P/E: {self.pe_ratio:.2f}")
        if self.pb_ratio:
            lines.append(f"  P/B: {self.pb_ratio:.2f}")
        if self.ps_ratio:
            lines.append(f"  P/S: {self.ps_ratio:.2f}")

        # Growth
        if self.eps_growth or self.revenue_growth:
            lines.append("\n**Growth:**")
            if self.eps_growth:
                lines.append(f"  EPS Growth: {self.eps_growth:.1f}%")
            if self.revenue_growth:
                lines.append(f"  Revenue Growth: {self.revenue_growth:.1f}%")

        # Margins
        if any([self.gross_margin, self.operating_margin, self.net_margin]):
            lines.append("\n**Margins:**")
            if self.gross_margin:
                lines.append(f"  Gross: {self.gross_margin:.1f}%")
            if self.operating_margin:
                lines.append(f"  Operating: {self.operating_margin:.1f}%")
            if self.net_margin:
                lines.append(f"  Net: {self.net_margin:.1f}%")

        # Returns
        if self.roe or self.roa:
            lines.append("\n**Returns:**")
            if self.roe:
                lines.append(f"  ROE: {self.roe:.1f}%")
            if self.roa:
                lines.append(f"  ROA: {self.roa:.1f}%")

        # 52-week range
        if self.high_52w and self.low_52w:
            lines.append(f"\n**52-Week Range:** ${self.low_52w:.2f} - ${self.high_52w:.2f}")

        return "\n".join(lines)


@dataclass
class CompanyNews:
    """News article about a company."""
    headline: str
    summary: str
    source: str
    url: str
    datetime: datetime
    category: str
    related: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "headline": self.headline,
            "summary": self.summary,
            "source": self.source,
            "url": self.url,
            "datetime": self.datetime.isoformat(),
            "category": self.category,
            "related": self.related,
        }


@dataclass
class CandleData:
    """OHLCV candle data for charting."""
    symbol: str
    resolution: str  # 1, 5, 15, 30, 60, D, W, M
    timestamps: List[int]
    opens: List[float]
    highs: List[float]
    lows: List[float]
    closes: List[float]
    volumes: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "resolution": self.resolution,
            "timestamps": self.timestamps,
            "opens": self.opens,
            "highs": self.highs,
            "lows": self.lows,
            "closes": self.closes,
            "volumes": self.volumes,
        }

    @property
    def dates(self) -> List[datetime]:
        """Convert timestamps to datetime objects."""
        return [datetime.fromtimestamp(ts) for ts in self.timestamps]

    def __len__(self) -> int:
        return len(self.timestamps)


@dataclass
class EarningsEvent:
    """Earnings calendar event."""
    symbol: str
    date: str
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    hour: Optional[str] = None  # 'bmo' (before market), 'amc' (after market)
    quarter: Optional[int] = None
    year: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @property
    def surprise_pct(self) -> Optional[float]:
        """Calculate EPS surprise percentage."""
        if self.eps_actual is not None and self.eps_estimate is not None and self.eps_estimate != 0:
            return ((self.eps_actual - self.eps_estimate) / abs(self.eps_estimate)) * 100
        return None

    def format_summary(self) -> str:
        """Format earnings event as readable string."""
        parts = [f"{self.symbol} - {self.date}"]
        if self.hour:
            parts.append(f"({self.hour.upper()})")
        if self.eps_estimate:
            parts.append(f"EPS Est: ${self.eps_estimate:.2f}")
        if self.eps_actual:
            surprise = self.surprise_pct
            if surprise is not None:
                emoji = "✅" if surprise >= 0 else "❌"
                parts.append(f"Actual: ${self.eps_actual:.2f} {emoji} ({surprise:+.1f}%)")
        return " | ".join(parts)


class FinnhubClient:
    """
    Client for Finnhub API.

    Provides real-time quotes, company profiles, financials, and news.
    Thread-safe with tiered caching.
    """

    BASE_URL = "https://finnhub.io/api/v1"

    # Cache TTLs (seconds)
    QUOTE_CACHE_TTL = 1500  # 5 minutes
    PROFILE_CACHE_TTL = 10800  # 1 hour
    FINANCIALS_CACHE_TTL = 7200  # 1 hour
    NEWS_CACHE_TTL = 1500  # 15 minutes
    CANDLE_CACHE_TTL = 7200  # 1 hour for daily candles
    EARNINGS_CACHE_TTL = 172800  # 1 day for earnings calendar

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(settings, 'FINNHUB_API_KEY', None) or os.getenv("FINNHUB_API_KEY")
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    def _get_cache(self, key: str, ttl: int) -> Optional[Any]:
        """Get item from cache if not expired."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if datetime.now() - entry["timestamp"] < timedelta(seconds=ttl):
                    logger.debug(f"Cache hit: {key}")
                    return entry["data"]
        return None

    def _set_cache(self, key: str, data: Any) -> None:
        """Set item in cache."""
        with self._lock:
            self._cache[key] = {
                "data": data,
                "timestamp": datetime.now()
            }

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request."""
        if not self.is_available():
            raise RuntimeError("Finnhub API key not configured")

        params = params or {}
        params["token"] = self.api_key

        try:
            response = requests.get(
                f"{self.BASE_URL}/{endpoint}",
                params=params,
                timeout=25,
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.error(f"Finnhub API request failed: {e}")
            raise

    def get_quote(self, symbol: str) -> Optional[StockQuote]:
        """
        Get real-time quote for a stock.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'NVDA')

        Returns:
            StockQuote object or None
        """
        symbol = symbol.upper()
        cache_key = f"quote:{symbol}"

        # Check cache
        cached = self._get_cache(cache_key, self.QUOTE_CACHE_TTL)
        if cached:
            return cached

        try:
            data = self._make_request("quote", {"symbol": symbol})

            if not data or data.get("c") is None or data.get("c") == 0:
                logger.warning(f"No quote data for {symbol}")
                return None

            quote = StockQuote(
                symbol=symbol,
                current_price=data["c"],
                change=data["d"] or 0,
                percent_change=data["dp"] or 0,
                high=data["h"],
                low=data["l"],
                open=data["o"],
                previous_close=data["pc"],
                timestamp=datetime.fromtimestamp(data["t"]) if data.get("t") else datetime.now(),
            )

            self._set_cache(cache_key, quote)
            logger.info(f"Retrieved quote for {symbol}: ${quote.current_price}")
            return quote

        except Exception as e:
            logger.error(f"Quote fetch failed for {symbol}: {e}")
            return None

    def get_company_profile(self, symbol: str) -> Optional[CompanyProfile]:
        """
        Get company profile.

        Args:
            symbol: Stock ticker symbol

        Returns:
            CompanyProfile object or None
        """
        symbol = symbol.upper()
        cache_key = f"profile:{symbol}"

        cached = self._get_cache(cache_key, self.PROFILE_CACHE_TTL)
        if cached:
            return cached

        try:
            data = self._make_request("stock/profile2", {"symbol": symbol})

            if not data or not data.get("name"):
                logger.warning(f"No profile data for {symbol}")
                return None

            profile = CompanyProfile(
                symbol=symbol,
                name=data.get("name", ""),
                country=data.get("country", ""),
                currency=data.get("currency", "USD"),
                exchange=data.get("exchange", ""),
                industry=data.get("finnhubIndustry", ""),
                logo=data.get("logo", ""),
                market_cap=data.get("marketCapitalization", 0) * 1e6,  # Convert to actual value
                shares_outstanding=data.get("shareOutstanding", 0) * 1e6,
                website=data.get("weburl", ""),
            )

            self._set_cache(cache_key, profile)
            logger.info(f"Retrieved profile for {symbol}: {profile.name}")
            return profile

        except Exception as e:
            logger.error(f"Profile fetch failed for {symbol}: {e}")
            return None

    def get_basic_financials(self, symbol: str) -> Optional[BasicFinancials]:
        """
        Get basic financial metrics.

        Args:
            symbol: Stock ticker symbol

        Returns:
            BasicFinancials object or None
        """
        symbol = symbol.upper()
        cache_key = f"financials:{symbol}"

        cached = self._get_cache(cache_key, self.FINANCIALS_CACHE_TTL)
        if cached:
            return cached

        try:
            data = self._make_request("stock/metric", {"symbol": symbol, "metric": "all"})

            if not data or not data.get("metric"):
                logger.warning(f"No financials data for {symbol}")
                return None

            m = data["metric"]

            financials = BasicFinancials(
                symbol=symbol,
                pe_ratio=m.get("peNormalizedAnnual"),
                pb_ratio=m.get("pbAnnual"),
                ps_ratio=m.get("psAnnual"),
                dividend_yield=m.get("dividendYieldIndicatedAnnual"),
                eps=m.get("epsNormalizedAnnual"),
                eps_growth=m.get("epsGrowth5Y"),
                revenue_growth=m.get("revenueGrowth5Y"),
                gross_margin=m.get("grossMarginAnnual"),
                operating_margin=m.get("operatingMarginAnnual"),
                net_margin=m.get("netProfitMarginAnnual"),
                roe=m.get("roeAnnual"),
                roa=m.get("roaAnnual"),
                debt_to_equity=m.get("totalDebt/totalEquityAnnual"),
                current_ratio=m.get("currentRatioAnnual"),
                quick_ratio=m.get("quickRatioAnnual"),
                beta=m.get("beta"),
                high_52w=m.get("52WeekHigh"),
                low_52w=m.get("52WeekLow"),
            )

            self._set_cache(cache_key, financials)
            logger.info(f"Retrieved financials for {symbol}")
            return financials

        except Exception as e:
            logger.error(f"Financials fetch failed for {symbol}: {e}")
            return None

    def get_company_news(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 10
    ) -> List[CompanyNews]:
        """
        Get recent news for a company.

        Args:
            symbol: Stock ticker symbol
            from_date: Start date (YYYY-MM-DD), defaults to 7 days ago
            to_date: End date (YYYY-MM-DD), defaults to today
            limit: Maximum number of articles

        Returns:
            List of CompanyNews objects
        """
        symbol = symbol.upper()
        cache_key = f"news:{symbol}"

        cached = self._get_cache(cache_key, self.NEWS_CACHE_TTL)
        if cached:
            return cached[:limit]

        # Default date range: last 7 days
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")
        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        try:
            data = self._make_request("company-news", {
                "symbol": symbol,
                "from": from_date,
                "to": to_date,
            })

            if not data:
                logger.warning(f"No news for {symbol}")
                return []

            news_list = []
            for item in data[:limit]:
                news_list.append(CompanyNews(
                    headline=item.get("headline", ""),
                    summary=item.get("summary", ""),
                    source=item.get("source", ""),
                    url=item.get("url", ""),
                    datetime=datetime.fromtimestamp(item.get("datetime", 0)),
                    category=item.get("category", ""),
                    related=item.get("related", symbol),
                ))

            self._set_cache(cache_key, news_list)
            logger.info(f"Retrieved {len(news_list)} news articles for {symbol}")
            return news_list

        except Exception as e:
            logger.error(f"News fetch failed for {symbol}: {e}")
            return []

    def format_news(self, news_list: List[CompanyNews], max_items: int = 5) -> str:
        """Format news list for display."""
        if not news_list:
            return "No recent news available."

        lines = ["## Recent News\n"]
        for i, news in enumerate(news_list[:max_items]):
            date_str = news.datetime.strftime("%m/%d %H:%M")
            lines.append(f"**{i+1}. {news.headline}**")
            lines.append(f"   {news.source} | {date_str}")
            if news.summary:
                summary = news.summary[:150] + "..." if len(news.summary) > 150 else news.summary
                lines.append(f"   {summary}")
            lines.append("")

        return "\n".join(lines)

    def get_candles(
        self,
        symbol: str,
        resolution: str = "D",
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Optional[CandleData]:
        """
        Get OHLCV candle data for charting.

        Args:
            symbol: Stock ticker symbol
            resolution: Candle resolution (1, 5, 15, 30, 60, D, W, M)
            from_date: Start date (defaults to 6 months ago)
            to_date: End date (defaults to today)

        Returns:
            CandleData object or None
        """
        symbol = symbol.upper()

        # Default date range: last 6 months
        if not to_date:
            to_date = datetime.now()
        if not from_date:
            from_date = to_date - timedelta(days=180)

        # Convert to Unix timestamps
        from_ts = int(from_date.timestamp())
        to_ts = int(to_date.timestamp())

        cache_key = f"candles:{symbol}:{resolution}:{from_ts}:{to_ts}"

        cached = self._get_cache(cache_key, self.CANDLE_CACHE_TTL)
        if cached:
            return cached

        try:
            data = self._make_request("stock/candle", {
                "symbol": symbol,
                "resolution": resolution,
                "from": from_ts,
                "to": to_ts,
            })

            if not data or data.get("s") == "no_data" or not data.get("c"):
                logger.warning(f"No candle data for {symbol}")
                return None

            candles = CandleData(
                symbol=symbol,
                resolution=resolution,
                timestamps=data.get("t", []),
                opens=data.get("o", []),
                highs=data.get("h", []),
                lows=data.get("l", []),
                closes=data.get("c", []),
                volumes=data.get("v", []),
            )

            self._set_cache(cache_key, candles)
            logger.info(f"Retrieved {len(candles)} candles for {symbol}")
            return candles

        except Exception as e:
            logger.error(f"Candle fetch failed for {symbol}: {e}")
            return None

    def get_earnings_calendar(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> List[EarningsEvent]:
        """
        Get earnings calendar.

        Args:
            from_date: Start date (YYYY-MM-DD), defaults to today
            to_date: End date (YYYY-MM-DD), defaults to 30 days from now
            symbol: Optional filter by symbol

        Returns:
            List of EarningsEvent objects
        """
        # Default date range: next 30 days
        if not from_date:
            from_date = datetime.now().strftime("%Y-%m-%d")
        if not to_date:
            to_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

        cache_key = f"earnings:{from_date}:{to_date}:{symbol or 'all'}"

        cached = self._get_cache(cache_key, self.EARNINGS_CACHE_TTL)
        if cached:
            return cached

        try:
            params = {"from": from_date, "to": to_date}
            if symbol:
                params["symbol"] = symbol.upper()

            data = self._make_request("calendar/earnings", params)

            if not data or not data.get("earningsCalendar"):
                logger.warning("No earnings calendar data")
                return []

            events = []
            for item in data["earningsCalendar"]:
                events.append(EarningsEvent(
                    symbol=item.get("symbol", ""),
                    date=item.get("date", ""),
                    eps_estimate=item.get("epsEstimate"),
                    eps_actual=item.get("epsActual"),
                    revenue_estimate=item.get("revenueEstimate"),
                    revenue_actual=item.get("revenueActual"),
                    hour=item.get("hour"),
                    quarter=item.get("quarter"),
                    year=item.get("year"),
                ))

            # Filter by symbol if specified
            if symbol:
                events = [e for e in events if e.symbol.upper() == symbol.upper()]

            self._set_cache(cache_key, events)
            logger.info(f"Retrieved {len(events)} earnings events")
            return events

        except Exception as e:
            logger.error(f"Earnings calendar fetch failed: {e}")
            return []

    def get_earnings_history(self, symbol: str, limit: int = 4) -> List[EarningsEvent]:
        """
        Get historical earnings for a stock.

        Args:
            symbol: Stock ticker symbol
            limit: Number of past quarters

        Returns:
            List of EarningsEvent objects (most recent first)
        """
        symbol = symbol.upper()
        cache_key = f"earnings_history:{symbol}"

        cached = self._get_cache(cache_key, self.EARNINGS_CACHE_TTL)
        if cached:
            return cached[:limit]

        try:
            data = self._make_request("stock/earnings", {"symbol": symbol})

            if not data:
                logger.warning(f"No earnings history for {symbol}")
                return []

            events = []
            for item in data[:limit]:
                events.append(EarningsEvent(
                    symbol=symbol,
                    date=item.get("period", ""),
                    eps_estimate=item.get("estimate"),
                    eps_actual=item.get("actual"),
                    quarter=item.get("quarter"),
                    year=item.get("year"),
                ))

            self._set_cache(cache_key, events)
            logger.info(f"Retrieved {len(events)} historical earnings for {symbol}")
            return events

        except Exception as e:
            logger.error(f"Earnings history fetch failed for {symbol}: {e}")
            return []

    def format_earnings(self, events: List[EarningsEvent], max_items: int = 5) -> str:
        """Format earnings events for display."""
        if not events:
            return "No earnings data available."

        lines = ["## Earnings\n"]
        for event in events[:max_items]:
            lines.append(event.format_summary())

        return "\n".join(lines)

    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive stock data (quote + profile + financials + news).

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with all available data
        """
        result = {
            "symbol": symbol.upper(),
            "quote": None,
            "profile": None,
            "financials": None,
            "news": [],
            "summary": "",
        }

        quote = self.get_quote(symbol)
        if quote:
            result["quote"] = quote.to_dict()

        profile = self.get_company_profile(symbol)
        if profile:
            result["profile"] = profile.to_dict()

        financials = self.get_basic_financials(symbol)
        if financials:
            result["financials"] = financials.to_dict()

        news = self.get_company_news(symbol, limit=5)
        if news:
            result["news"] = [n.to_dict() for n in news]

        # Build summary
        result["summary"] = self._build_summary(quote, profile, financials, news)

        return result

    def _build_summary(
        self,
        quote: Optional[StockQuote],
        profile: Optional[CompanyProfile],
        financials: Optional[BasicFinancials],
        news: List[CompanyNews]
    ) -> str:
        """Build markdown summary of all data."""
        parts = []

        if quote:
            parts.append(quote.format_summary())

        if profile:
            parts.append("\n" + profile.format_summary())

        if financials:
            parts.append("\n" + financials.format_summary())

        if news:
            parts.append("\n" + self.format_news(news, max_items=3))

        return "\n".join(parts) if parts else "No data available."


# Convenience functions
def get_stock_quote(symbol: str) -> Optional[StockQuote]:
    """Quick quote lookup."""
    client = FinnhubClient()
    if not client.is_available():
        logger.warning("Finnhub API not configured")
        return None
    return client.get_quote(symbol)


def get_stock_data(symbol: str) -> Dict[str, Any]:
    """Quick comprehensive data lookup."""
    client = FinnhubClient()
    if not client.is_available():
        logger.warning("Finnhub API not configured")
        return {"symbol": symbol, "error": "API not configured"}
    return client.get_stock_data(symbol)


def format_stock_context(symbol: str) -> str:
    """Get formatted stock context for expert prompts."""
    client = FinnhubClient()
    if not client.is_available():
        return f"[Finnhub API not configured - no real-time data for {symbol}]"

    data = client.get_stock_data(symbol)
    return data.get("summary", f"No data available for {symbol}")
