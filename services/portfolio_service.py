"""
Portfolio Performance Service

Tracks portfolio performance with trade history and daily snapshots.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path

from mcp_server.database import (
    FinancialDatabase,
    get_database,
    Trade,
    DailySnapshot,
    PortfolioPosition,
)

logger = logging.getLogger(__name__)


class PortfolioPerformanceService:
    """
    Service for tracking portfolio performance.

    Features:
    - Trade recording and history
    - Daily snapshots for performance charts
    - Performance metrics calculation
    - Benchmark comparison
    """

    def __init__(self, db: Optional[FinancialDatabase] = None):
        self.db = db or get_database()

    def record_trade(
        self,
        symbol: str,
        action: str,
        shares: float,
        price: float,
        fees: float = 0,
        trade_date: Optional[datetime] = None,
        notes: str = ""
    ) -> Trade:
        """
        Record a trade for performance tracking.

        Args:
            symbol: Stock ticker
            action: 'buy' or 'sell'
            shares: Number of shares
            price: Price per share
            fees: Trading fees
            trade_date: Date of trade (default: now)
            notes: Optional notes

        Returns:
            Trade object
        """
        trade = self.db.record_trade(
            symbol=symbol,
            action=action,
            shares=shares,
            price=price,
            fees=fees,
            trade_date=trade_date,
            notes=notes,
        )

        # Also update the portfolio position
        if action == "buy":
            self.db.add_position(symbol, shares, price, notes)
        elif action == "sell":
            try:
                self.db.reduce_position(symbol, shares)
            except ValueError:
                pass  # Position doesn't exist, just record the trade

        return trade

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        days: Optional[int] = None
    ) -> List[Trade]:
        """
        Get trade history.

        Args:
            symbol: Filter by ticker
            days: Filter by recent days

        Returns:
            List of Trade objects
        """
        return self.db.get_trades(symbol, days)

    def take_daily_snapshot(self, finnhub_client=None) -> Optional[DailySnapshot]:
        """
        Take a daily portfolio snapshot.

        Should be called at market close or end of day.

        Args:
            finnhub_client: Optional Finnhub client for prices

        Returns:
            DailySnapshot or None if no positions
        """
        positions = self.db.get_portfolio()

        if not positions:
            logger.info("No positions to snapshot")
            return None

        total_value = 0.0
        total_cost = 0.0
        positions_dict = {}

        for pos in positions:
            # Get current price
            current_price = pos.avg_price  # Default to avg price

            if finnhub_client:
                try:
                    quote = finnhub_client.get_quote(pos.symbol)
                    if quote:
                        current_price = quote.current_price
                except Exception as e:
                    logger.warning(f"Failed to get price for {pos.symbol}: {e}")

            position_value = pos.shares * current_price
            total_value += position_value
            total_cost += pos.cost_basis

            positions_dict[pos.symbol] = {
                "shares": pos.shares,
                "avg_price": pos.avg_price,
                "current_price": current_price,
                "value": position_value,
                "cost_basis": pos.cost_basis,
                "pnl": position_value - pos.cost_basis,
            }

        return self.db.save_daily_snapshot(total_value, total_cost, positions_dict)

    def get_performance_data(self, days: int = 30) -> Dict[str, Any]:
        """
        Get portfolio performance data for charts.

        Args:
            days: Number of days of history

        Returns:
            Dict with dates, values, and metrics
        """
        snapshots = self.db.get_snapshots(days)

        if not snapshots:
            return {
                "dates": [],
                "values": [],
                "pnl": [],
                "pnl_pct": [],
                "has_data": False,
            }

        return {
            "dates": [s.date for s in snapshots],
            "values": [s.total_value for s in snapshots],
            "pnl": [s.daily_pnl for s in snapshots],
            "pnl_pct": [s.daily_pnl_pct for s in snapshots],
            "has_data": True,
        }

    def calculate_metrics(self, days: int = 30) -> Dict[str, Any]:
        """
        Calculate portfolio performance metrics.

        Args:
            days: Period for calculations

        Returns:
            Dict with performance metrics
        """
        snapshots = self.db.get_snapshots(days)
        trades = self.get_trade_history(days=days)

        if not snapshots:
            return {
                "total_return": 0,
                "total_return_pct": 0,
                "daily_avg_pnl": 0,
                "best_day": 0,
                "worst_day": 0,
                "win_rate": 0,
                "total_trades": len(trades),
            }

        # Calculate metrics
        first_value = snapshots[0].total_value if snapshots else 0
        last_value = snapshots[-1].total_value if snapshots else 0
        total_cost = snapshots[-1].total_cost if snapshots else 0

        total_return = last_value - total_cost
        total_return_pct = (total_return / total_cost * 100) if total_cost > 0 else 0

        pnl_values = [s.daily_pnl for s in snapshots]
        daily_avg = sum(pnl_values) / len(pnl_values) if pnl_values else 0

        winning_days = sum(1 for p in pnl_values if p > 0)
        win_rate = (winning_days / len(pnl_values) * 100) if pnl_values else 0

        return {
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "daily_avg_pnl": daily_avg,
            "best_day": max(pnl_values) if pnl_values else 0,
            "worst_day": min(pnl_values) if pnl_values else 0,
            "win_rate": win_rate,
            "total_trades": len(trades),
            "current_value": last_value,
            "total_cost": total_cost,
        }

    def get_trades_by_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Get trade analysis for a specific symbol.

        Args:
            symbol: Stock ticker

        Returns:
            Trade analysis with P&L
        """
        trades = self.get_trade_history(symbol=symbol)

        if not trades:
            return {
                "symbol": symbol.upper(),
                "total_trades": 0,
                "buys": 0,
                "sells": 0,
                "total_invested": 0,
                "total_sold": 0,
                "realized_pnl": 0,
            }

        buys = [t for t in trades if t.action == "buy"]
        sells = [t for t in trades if t.action == "sell"]

        total_invested = sum(t.shares * t.price + t.fees for t in buys)
        total_sold = sum(t.shares * t.price - t.fees for t in sells)

        # Calculate realized P&L (FIFO)
        realized_pnl = total_sold - (
            sum(t.shares for t in sells) /
            sum(t.shares for t in buys) * total_invested
            if buys and sells else 0
        )

        return {
            "symbol": symbol.upper(),
            "total_trades": len(trades),
            "buys": len(buys),
            "sells": len(sells),
            "total_invested": total_invested,
            "total_sold": total_sold,
            "realized_pnl": realized_pnl,
            "total_fees": sum(t.fees for t in trades),
        }

    def compare_to_benchmark(
        self,
        benchmark: str = "SPY",
        days: int = 30,
        finnhub_client=None
    ) -> Dict[str, Any]:
        """
        Compare portfolio performance to benchmark.

        Args:
            benchmark: Benchmark ticker (default: SPY)
            days: Period for comparison
            finnhub_client: Finnhub client for benchmark data

        Returns:
            Comparison metrics
        """
        snapshots = self.db.get_snapshots(days)

        if not snapshots or len(snapshots) < 2:
            return {
                "portfolio_return_pct": 0,
                "benchmark_return_pct": 0,
                "alpha": 0,
                "benchmark": benchmark,
                "has_data": False,
            }

        # Portfolio return
        first = snapshots[0].total_value
        last = snapshots[-1].total_value
        portfolio_return = ((last - first) / first * 100) if first > 0 else 0

        # Benchmark return (if client available)
        benchmark_return = 0
        if finnhub_client:
            try:
                from datetime import datetime, timedelta
                to_date = datetime.now()
                from_date = to_date - timedelta(days=days)

                candles = finnhub_client.get_candles(benchmark, "D", from_date, to_date)
                if candles and candles.closes:
                    first_price = candles.closes[0]
                    last_price = candles.closes[-1]
                    benchmark_return = ((last_price - first_price) / first_price * 100)
            except Exception as e:
                logger.warning(f"Failed to get benchmark data: {e}")

        return {
            "portfolio_return_pct": portfolio_return,
            "benchmark_return_pct": benchmark_return,
            "alpha": portfolio_return - benchmark_return,
            "benchmark": benchmark,
            "has_data": True,
        }

    def generate_performance_chart(self, days: int = 30) -> str:
        """
        Generate a portfolio performance chart.

        Args:
            days: Days of history

        Returns:
            Path to chart file
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("Plotly not installed")
            return ""

        data = self.get_performance_data(days)

        if not data["has_data"]:
            return ""

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=("Portfolio Value", "Daily P&L")
        )

        # Portfolio value line
        fig.add_trace(
            go.Scatter(
                x=data["dates"],
                y=data["values"],
                mode='lines+markers',
                name='Portfolio Value',
                line=dict(color='#2196f3', width=2),
            ),
            row=1, col=1
        )

        # Daily P&L bars
        colors = ['#4caf50' if p >= 0 else '#f44336' for p in data["pnl"]]
        fig.add_trace(
            go.Bar(
                x=data["dates"],
                y=data["pnl"],
                name='Daily P&L',
                marker_color=colors,
            ),
            row=2, col=1
        )

        fig.update_layout(
            title="Portfolio Performance",
            template="plotly_dark",
            height=600,
            showlegend=True,
        )

        # Save chart
        output_dir = Path("outputs/charts")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"portfolio_{timestamp}.png"

        try:
            fig.write_image(str(filepath))
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save chart: {e}")
            html_path = filepath.with_suffix('.html')
            fig.write_html(str(html_path))
            return str(html_path)

    def format_performance_summary(self, days: int = 30) -> str:
        """
        Format performance summary as markdown.

        Args:
            days: Period for summary

        Returns:
            Markdown string
        """
        metrics = self.calculate_metrics(days)

        if not metrics.get("current_value"):
            return "No portfolio data available for performance analysis."

        sign = "+" if metrics["total_return"] >= 0 else ""

        lines = [
            f"## Portfolio Performance ({days} days)",
            "",
            f"**Current Value:** ${metrics['current_value']:,.2f}",
            f"**Total Cost:** ${metrics['total_cost']:,.2f}",
            f"**Total Return:** {sign}${metrics['total_return']:,.2f} ({sign}{metrics['total_return_pct']:.1f}%)",
            "",
            "### Daily Stats",
            f"- Average Daily P&L: ${metrics['daily_avg_pnl']:,.2f}",
            f"- Best Day: ${metrics['best_day']:,.2f}",
            f"- Worst Day: ${metrics['worst_day']:,.2f}",
            f"- Win Rate: {metrics['win_rate']:.1f}%",
            "",
            f"**Total Trades:** {metrics['total_trades']}",
        ]

        return "\n".join(lines)


# Convenience function
def get_portfolio_service() -> PortfolioPerformanceService:
    """Get portfolio performance service instance."""
    return PortfolioPerformanceService()
