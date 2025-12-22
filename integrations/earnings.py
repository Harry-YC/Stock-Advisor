"""
Earnings Calendar Integration

Provides earnings calendar data and analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class EarningsAnalyzer:
    """
    Analyzer for earnings calendar data.

    Features:
    - Upcoming earnings for watchlist
    - Historical earnings surprises
    - Beat/miss rate calculation
    """

    def __init__(self, finnhub_client=None):
        self._finnhub = finnhub_client

    def _get_finnhub(self):
        """Lazy load Finnhub client."""
        if self._finnhub is None:
            try:
                from integrations.finnhub import FinnhubClient
                self._finnhub = FinnhubClient()
            except ImportError:
                logger.error("Finnhub client not available")
        return self._finnhub

    def get_upcoming_earnings(
        self,
        symbols: Optional[List[str]] = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get upcoming earnings dates.

        Args:
            symbols: Optional list of symbols to filter
            days: Number of days to look ahead

        Returns:
            List of earnings events
        """
        finnhub = self._get_finnhub()
        if not finnhub or not finnhub.is_available():
            return []

        from_date = datetime.now()
        to_date = from_date + timedelta(days=days)

        events = finnhub.get_earnings_calendar(
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
        )

        # Filter by symbols if provided
        if symbols:
            symbols_upper = [s.upper() for s in symbols]
            events = [e for e in events if e.symbol.upper() in symbols_upper]

        # Convert to dicts and sort by date
        result = []
        for event in events:
            result.append({
                "symbol": event.symbol,
                "date": event.date,
                "eps_estimate": event.eps_estimate,
                "revenue_estimate": event.revenue_estimate,
                "hour": event.hour or "N/A",
                "quarter": event.quarter,
                "year": event.year,
            })

        result.sort(key=lambda x: x["date"])
        return result

    def get_earnings_history(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get historical earnings for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            List of historical earnings with surprises
        """
        finnhub = self._get_finnhub()
        if not finnhub or not finnhub.is_available():
            return []

        events = finnhub.get_earnings_history(symbol)

        result = []
        for event in events:
            if event.eps_actual is not None and event.eps_estimate is not None:
                surprise = event.eps_actual - event.eps_estimate
                surprise_pct = (surprise / abs(event.eps_estimate) * 100) if event.eps_estimate != 0 else 0
                beat = event.eps_actual > event.eps_estimate
            else:
                surprise = None
                surprise_pct = None
                beat = None

            result.append({
                "symbol": event.symbol,
                "date": event.date,
                "quarter": event.quarter,
                "year": event.year,
                "eps_estimate": event.eps_estimate,
                "eps_actual": event.eps_actual,
                "surprise": surprise,
                "surprise_pct": surprise_pct,
                "beat": beat,
                "revenue_estimate": event.revenue_estimate,
                "revenue_actual": event.revenue_actual,
            })

        return result

    def calculate_surprise_stats(self, symbol: str) -> Dict[str, Any]:
        """
        Calculate earnings surprise statistics for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            Dict with beat/miss rates and average surprise
        """
        history = self.get_earnings_history(symbol)

        if not history:
            return {
                "symbol": symbol.upper(),
                "total_quarters": 0,
                "beats": 0,
                "misses": 0,
                "in_line": 0,
                "beat_rate": 0,
                "avg_surprise_pct": 0,
            }

        # Count beats/misses
        beats = sum(1 for e in history if e.get("beat") is True)
        misses = sum(1 for e in history if e.get("beat") is False)
        in_line = sum(1 for e in history if e.get("surprise") == 0)

        # Calculate average surprise
        surprises = [e["surprise_pct"] for e in history if e.get("surprise_pct") is not None]
        avg_surprise = sum(surprises) / len(surprises) if surprises else 0

        total = len(history)

        return {
            "symbol": symbol.upper(),
            "total_quarters": total,
            "beats": beats,
            "misses": misses,
            "in_line": in_line,
            "beat_rate": (beats / total * 100) if total > 0 else 0,
            "avg_surprise_pct": avg_surprise,
        }

    def get_earnings_for_watchlist(
        self,
        watchlist: List[str],
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get earnings calendar for watchlist stocks.

        Args:
            watchlist: List of ticker symbols
            days: Days to look ahead

        Returns:
            Dict with earnings by date
        """
        upcoming = self.get_upcoming_earnings(symbols=watchlist, days=days)

        # Group by date
        by_date: Dict[str, List[Dict]] = {}
        for event in upcoming:
            date = event["date"]
            if date not in by_date:
                by_date[date] = []
            by_date[date].append(event)

        return {
            "total_events": len(upcoming),
            "by_date": by_date,
            "watchlist_size": len(watchlist),
            "days_ahead": days,
        }

    def format_earnings_summary(self, symbol: str) -> str:
        """
        Generate markdown summary of earnings data.

        Args:
            symbol: Stock ticker

        Returns:
            Markdown string
        """
        stats = self.calculate_surprise_stats(symbol)
        upcoming = self.get_upcoming_earnings([symbol], days=60)

        lines = [
            f"## Earnings Data ({symbol})",
            "",
        ]

        # Next earnings
        if upcoming:
            next_event = upcoming[0]
            lines.extend([
                f"**Next Earnings:** {next_event['date']} ({next_event['hour']})",
                f"- EPS Estimate: ${next_event['eps_estimate']:.2f}" if next_event.get('eps_estimate') else "",
                "",
            ])
        else:
            lines.append("**Next Earnings:** Not scheduled within 60 days")
            lines.append("")

        # Historical stats
        if stats["total_quarters"] > 0:
            lines.extend([
                "### Historical Performance",
                f"- Quarters Tracked: {stats['total_quarters']}",
                f"- Beat Rate: {stats['beat_rate']:.1f}%",
                f"- Avg Surprise: {stats['avg_surprise_pct']:+.1f}%",
                f"- Beats: {stats['beats']} | Misses: {stats['misses']}",
            ])
        else:
            lines.append("*No historical earnings data available*")

        return "\n".join(lines)

    def format_upcoming_earnings(
        self,
        symbols: Optional[List[str]] = None,
        days: int = 14
    ) -> str:
        """
        Format upcoming earnings as markdown.

        Args:
            symbols: Optional filter by symbols
            days: Days ahead

        Returns:
            Markdown string
        """
        upcoming = self.get_upcoming_earnings(symbols, days)

        if not upcoming:
            return "No upcoming earnings in the next {days} days."

        lines = [
            f"## Upcoming Earnings ({days} days)",
            "",
        ]

        current_date = None
        for event in upcoming:
            if event["date"] != current_date:
                current_date = event["date"]
                lines.append(f"### {current_date}")

            hour_str = f" ({event['hour']})" if event.get("hour") and event["hour"] != "N/A" else ""
            eps_str = f" | Est: ${event['eps_estimate']:.2f}" if event.get("eps_estimate") else ""
            lines.append(f"- **{event['symbol']}**{hour_str}{eps_str}")

        return "\n".join(lines)


# Convenience functions
def get_earnings_analyzer() -> EarningsAnalyzer:
    """Get earnings analyzer instance."""
    return EarningsAnalyzer()


def get_earnings_summary(symbol: str) -> str:
    """Quick earnings summary lookup."""
    analyzer = EarningsAnalyzer()
    return analyzer.format_earnings_summary(symbol)


def get_upcoming_earnings_list(days: int = 14) -> str:
    """Get formatted list of upcoming earnings."""
    analyzer = EarningsAnalyzer()
    return analyzer.format_upcoming_earnings(days=days)
