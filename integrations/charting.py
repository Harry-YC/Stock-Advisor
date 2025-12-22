"""
Chart Generation Module

Generates candlestick charts with technical indicators using Plotly.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import CandleData type hint
try:
    from integrations.finnhub import CandleData
except ImportError:
    CandleData = None


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    width: int = 1200
    height: int = 800
    theme: str = "plotly_dark"
    show_volume: bool = True
    title_font_size: int = 16


class ChartGenerator:
    """
    Generates stock charts with technical indicators.

    Supports:
    - Candlestick charts
    - SMA (Simple Moving Average)
    - EMA (Exponential Moving Average)
    - Bollinger Bands
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Volume bars
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or "outputs/charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_candlestick_chart(
        self,
        candles: "CandleData",
        indicators: Optional[List[str]] = None,
        config: Optional[ChartConfig] = None
    ) -> str:
        """
        Generate a candlestick chart with optional indicators.

        Args:
            candles: OHLCV candle data
            indicators: List of indicators to add:
                - 'sma_20', 'sma_50', 'sma_200' - Simple Moving Averages
                - 'ema_12', 'ema_26' - Exponential Moving Averages
                - 'bollinger' - Bollinger Bands (20-period)
                - 'rsi' - RSI (14-period)
                - 'macd' - MACD indicator
            config: Chart configuration

        Returns:
            Path to saved PNG file
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("Plotly not installed. Run: pip install plotly kaleido")
            return ""

        indicators = indicators or ['sma_20', 'sma_50']
        config = config or ChartConfig()

        # Determine subplot layout
        has_rsi = 'rsi' in indicators
        has_macd = 'macd' in indicators
        has_volume = config.show_volume

        row_count = 1
        row_heights = [0.6]

        if has_volume:
            row_count += 1
            row_heights.append(0.15)
        if has_rsi:
            row_count += 1
            row_heights.append(0.15)
        if has_macd:
            row_count += 1
            row_heights.append(0.15)

        # Normalize row heights
        total = sum(row_heights)
        row_heights = [h / total for h in row_heights]

        fig = make_subplots(
            rows=row_count,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=self._get_subplot_titles(has_volume, has_rsi, has_macd)
        )

        dates = candles.dates

        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=candles.opens,
                high=candles.highs,
                low=candles.lows,
                close=candles.closes,
                name="Price",
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
            ),
            row=1, col=1
        )

        # Add moving averages
        for indicator in indicators:
            if indicator.startswith('sma_'):
                period = int(indicator.split('_')[1])
                sma = self._calc_sma(candles.closes, period)
                color = self._get_ma_color(period)
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=sma,
                        mode='lines',
                        name=f'SMA {period}',
                        line=dict(color=color, width=1),
                    ),
                    row=1, col=1
                )

            elif indicator.startswith('ema_'):
                period = int(indicator.split('_')[1])
                ema = self._calc_ema(candles.closes, period)
                color = self._get_ma_color(period)
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=ema,
                        mode='lines',
                        name=f'EMA {period}',
                        line=dict(color=color, width=1, dash='dot'),
                    ),
                    row=1, col=1
                )

        # Bollinger Bands
        if 'bollinger' in indicators:
            upper, middle, lower = self._calc_bollinger(candles.closes)
            fig.add_trace(
                go.Scatter(
                    x=dates, y=upper, mode='lines',
                    name='BB Upper', line=dict(color='rgba(173, 216, 230, 0.5)', width=1),
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=dates, y=lower, mode='lines', fill='tonexty',
                    name='BB Lower', line=dict(color='rgba(173, 216, 230, 0.5)', width=1),
                    fillcolor='rgba(173, 216, 230, 0.1)',
                ),
                row=1, col=1
            )

        current_row = 2

        # Volume subplot
        if has_volume:
            colors = ['#26a69a' if c >= o else '#ef5350'
                      for o, c in zip(candles.opens, candles.closes)]
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=candles.volumes,
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7,
                ),
                row=current_row, col=1
            )
            current_row += 1

        # RSI subplot
        if has_rsi:
            rsi = self._calc_rsi(candles.closes)
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=rsi,
                    mode='lines',
                    name='RSI',
                    line=dict(color='#9c27b0', width=1),
                ),
                row=current_row, col=1
            )
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                          row=current_row, col=1, opacity=0.5)
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                          row=current_row, col=1, opacity=0.5)
            current_row += 1

        # MACD subplot
        if has_macd:
            macd_line, signal_line, histogram = self._calc_macd(candles.closes)

            # MACD histogram
            colors = ['#26a69a' if h >= 0 else '#ef5350' for h in histogram]
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=histogram,
                    name='MACD Hist',
                    marker_color=colors,
                ),
                row=current_row, col=1
            )

            # MACD and Signal lines
            fig.add_trace(
                go.Scatter(
                    x=dates, y=macd_line, mode='lines',
                    name='MACD', line=dict(color='#2196f3', width=1),
                ),
                row=current_row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=dates, y=signal_line, mode='lines',
                    name='Signal', line=dict(color='#ff9800', width=1),
                ),
                row=current_row, col=1
            )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{candles.symbol} - {candles.resolution} Chart",
                font=dict(size=config.title_font_size),
            ),
            template=config.theme,
            xaxis_rangeslider_visible=False,
            height=config.height,
            width=config.width,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )

        # Hide x-axis labels on all but bottom subplot
        for i in range(1, row_count):
            fig.update_xaxes(showticklabels=False, row=i, col=1)

        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{candles.symbol}_{timestamp}.png"
        filepath = self.output_dir / filename

        try:
            fig.write_image(str(filepath))
            logger.info(f"Chart saved to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save chart: {e}")
            # Try HTML fallback
            html_path = filepath.with_suffix('.html')
            fig.write_html(str(html_path))
            logger.info(f"Chart saved as HTML to {html_path}")
            return str(html_path)

    def _get_subplot_titles(self, has_volume: bool, has_rsi: bool, has_macd: bool) -> Tuple[str, ...]:
        """Get subplot titles based on indicators."""
        titles = [""]  # Empty for main chart (title in layout)
        if has_volume:
            titles.append("Volume")
        if has_rsi:
            titles.append("RSI (14)")
        if has_macd:
            titles.append("MACD")
        return tuple(titles)

    def _get_ma_color(self, period: int) -> str:
        """Get color for moving average based on period."""
        colors = {
            12: '#ff9800',  # Orange
            20: '#2196f3',  # Blue
            26: '#f44336',  # Red
            50: '#4caf50',  # Green
            100: '#9c27b0', # Purple
            200: '#795548', # Brown
        }
        return colors.get(period, '#607d8b')  # Default gray

    def _calc_sma(self, data: List[float], period: int) -> List[Optional[float]]:
        """Calculate Simple Moving Average."""
        sma = []
        for i in range(len(data)):
            if i < period - 1:
                sma.append(None)
            else:
                avg = sum(data[i - period + 1:i + 1]) / period
                sma.append(avg)
        return sma

    def _calc_ema(self, data: List[float], period: int) -> List[Optional[float]]:
        """Calculate Exponential Moving Average."""
        ema = []
        multiplier = 2 / (period + 1)

        for i in range(len(data)):
            if i < period - 1:
                ema.append(None)
            elif i == period - 1:
                # First EMA is SMA
                avg = sum(data[:period]) / period
                ema.append(avg)
            else:
                prev_ema = ema[-1]
                if prev_ema is not None:
                    new_ema = (data[i] - prev_ema) * multiplier + prev_ema
                    ema.append(new_ema)
                else:
                    ema.append(None)

        return ema

    def _calc_bollinger(
        self,
        data: List[float],
        period: int = 20,
        std_dev: int = 2
    ) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
        """Calculate Bollinger Bands."""
        middle = self._calc_sma(data, period)
        upper = []
        lower = []

        for i in range(len(data)):
            if i < period - 1:
                upper.append(None)
                lower.append(None)
            else:
                window = data[i - period + 1:i + 1]
                std = (sum((x - middle[i]) ** 2 for x in window) / period) ** 0.5
                upper.append(middle[i] + std_dev * std)
                lower.append(middle[i] - std_dev * std)

        return upper, middle, lower

    def _calc_rsi(self, data: List[float], period: int = 14) -> List[Optional[float]]:
        """Calculate Relative Strength Index."""
        rsi = [None] * (period)

        gains = []
        losses = []

        for i in range(1, len(data)):
            change = data[i] - data[i - 1]
            gains.append(max(0, change))
            losses.append(max(0, -change))

        for i in range(period, len(data)):
            if i == period:
                avg_gain = sum(gains[:period]) / period
                avg_loss = sum(losses[:period]) / period
            else:
                avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

            if avg_loss == 0:
                rsi.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100 - (100 / (1 + rs)))

        return rsi

    def _calc_macd(
        self,
        data: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = self._calc_ema(data, fast)
        ema_slow = self._calc_ema(data, slow)

        macd_line = []
        for f, s in zip(ema_fast, ema_slow):
            if f is not None and s is not None:
                macd_line.append(f - s)
            else:
                macd_line.append(None)

        # Calculate signal line (EMA of MACD)
        signal_line = []
        valid_macd = [m for m in macd_line if m is not None]

        if len(valid_macd) >= signal:
            macd_start = len(macd_line) - len(valid_macd)
            signal_ema = self._calc_ema(valid_macd, signal)

            signal_line = [None] * macd_start
            signal_line.extend(signal_ema)
        else:
            signal_line = [None] * len(macd_line)

        # Calculate histogram
        histogram = []
        for m, s in zip(macd_line, signal_line):
            if m is not None and s is not None:
                histogram.append(m - s)
            else:
                histogram.append(None)

        return macd_line, signal_line, histogram


def generate_stock_chart(
    symbol: str,
    indicators: Optional[List[str]] = None,
    days: int = 180
) -> str:
    """
    Convenience function to generate a stock chart.

    Args:
        symbol: Stock ticker symbol
        indicators: List of indicators (default: ['sma_20', 'sma_50', 'rsi'])
        days: Number of days of data to fetch

    Returns:
        Path to saved chart file
    """
    from integrations.finnhub import FinnhubClient
    from datetime import datetime, timedelta

    client = FinnhubClient()
    if not client.is_available():
        logger.error("Finnhub API not configured")
        return ""

    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    candles = client.get_candles(symbol, "D", from_date, to_date)
    if not candles:
        logger.error(f"No candle data for {symbol}")
        return ""

    generator = ChartGenerator()
    indicators = indicators or ['sma_20', 'sma_50', 'rsi']

    return generator.create_candlestick_chart(candles, indicators)
