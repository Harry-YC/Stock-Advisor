"""
Options Data Integration

Provides options chain data using Yahoo Finance (free alternative to Finnhub premium).
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OptionContract:
    """Individual option contract."""
    strike: float
    expiration: str
    contract_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    in_the_money: bool = False
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @property
    def mid_price(self) -> float:
        """Calculate mid-point between bid and ask."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def volume_oi_ratio(self) -> float:
        """Calculate volume to open interest ratio."""
        if self.open_interest > 0:
            return self.volume / self.open_interest
        return 0


@dataclass
class OptionChain:
    """Complete option chain for a symbol."""
    symbol: str
    underlying_price: float
    expiration_dates: List[str]
    calls: List[OptionContract] = field(default_factory=list)
    puts: List[OptionContract] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "underlying_price": self.underlying_price,
            "expiration_dates": self.expiration_dates,
            "calls": [c.to_dict() for c in self.calls],
            "puts": [p.to_dict() for p in self.puts],
        }


class OptionsClient:
    """
    Client for options data using Yahoo Finance.

    Provides:
    - Option chains (calls and puts)
    - Put/call ratio calculation
    - Max pain analysis
    - Unusual activity detection
    """

    def __init__(self):
        self._yf = None

    def _get_yfinance(self):
        """Lazy load yfinance."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                logger.error("yfinance not installed. Run: pip install yfinance")
                raise RuntimeError("yfinance not installed")
        return self._yf

    def is_available(self) -> bool:
        """Check if yfinance is available."""
        try:
            self._get_yfinance()
            return True
        except RuntimeError:
            return False

    def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None
    ) -> Optional[OptionChain]:
        """
        Get option chain for a symbol.

        Args:
            symbol: Stock ticker symbol
            expiration: Specific expiration date (YYYY-MM-DD) or None for nearest

        Returns:
            OptionChain object or None
        """
        try:
            yf = self._get_yfinance()
            ticker = yf.Ticker(symbol.upper())

            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                logger.warning(f"No options available for {symbol}")
                return None

            # Select expiration
            if expiration and expiration in expirations:
                selected_exp = expiration
            else:
                selected_exp = expirations[0]  # Nearest expiration

            # Get the option chain
            chain = ticker.option_chain(selected_exp)

            # Get underlying price
            info = ticker.info
            underlying_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

            calls = []
            puts = []

            # Process calls
            if hasattr(chain, 'calls') and not chain.calls.empty:
                for _, row in chain.calls.iterrows():
                    calls.append(OptionContract(
                        strike=row.get('strike', 0),
                        expiration=selected_exp,
                        contract_type='call',
                        bid=row.get('bid', 0) or 0,
                        ask=row.get('ask', 0) or 0,
                        last=row.get('lastPrice', 0) or 0,
                        volume=int(row.get('volume', 0) or 0),
                        open_interest=int(row.get('openInterest', 0) or 0),
                        implied_volatility=row.get('impliedVolatility', 0) or 0,
                        in_the_money=row.get('inTheMoney', False),
                    ))

            # Process puts
            if hasattr(chain, 'puts') and not chain.puts.empty:
                for _, row in chain.puts.iterrows():
                    puts.append(OptionContract(
                        strike=row.get('strike', 0),
                        expiration=selected_exp,
                        contract_type='put',
                        bid=row.get('bid', 0) or 0,
                        ask=row.get('ask', 0) or 0,
                        last=row.get('lastPrice', 0) or 0,
                        volume=int(row.get('volume', 0) or 0),
                        open_interest=int(row.get('openInterest', 0) or 0),
                        implied_volatility=row.get('impliedVolatility', 0) or 0,
                        in_the_money=row.get('inTheMoney', False),
                    ))

            logger.info(f"Retrieved option chain for {symbol}: {len(calls)} calls, {len(puts)} puts")

            return OptionChain(
                symbol=symbol.upper(),
                underlying_price=underlying_price,
                expiration_dates=list(expirations),
                calls=calls,
                puts=puts,
            )

        except Exception as e:
            logger.error(f"Failed to get option chain for {symbol}: {e}")
            return None

    def calculate_put_call_ratio(self, chain: OptionChain) -> Dict[str, float]:
        """
        Calculate put/call ratio for an option chain.

        Returns:
            Dict with volume and open interest ratios
        """
        total_call_volume = sum(c.volume for c in chain.calls)
        total_put_volume = sum(p.volume for p in chain.puts)
        total_call_oi = sum(c.open_interest for c in chain.calls)
        total_put_oi = sum(p.open_interest for p in chain.puts)

        return {
            "volume_ratio": total_put_volume / total_call_volume if total_call_volume > 0 else 0,
            "oi_ratio": total_put_oi / total_call_oi if total_call_oi > 0 else 0,
            "total_call_volume": total_call_volume,
            "total_put_volume": total_put_volume,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
        }

    def calculate_max_pain(self, chain: OptionChain) -> Dict[str, Any]:
        """
        Calculate max pain strike price.

        Max pain is the strike where option writers would pay the least.

        Returns:
            Dict with max pain strike and analysis
        """
        strikes = set()
        for c in chain.calls:
            strikes.add(c.strike)
        for p in chain.puts:
            strikes.add(p.strike)

        if not strikes:
            return {"max_pain": 0, "analysis": "No option data available"}

        strikes = sorted(strikes)
        min_pain = float('inf')
        max_pain_strike = strikes[0]

        for test_strike in strikes:
            total_pain = 0

            # Calculate pain for call holders (OI * max(0, stock_price - strike))
            for call in chain.calls:
                if test_strike > call.strike:
                    total_pain += call.open_interest * (test_strike - call.strike) * 100

            # Calculate pain for put holders (OI * max(0, strike - stock_price))
            for put in chain.puts:
                if test_strike < put.strike:
                    total_pain += put.open_interest * (put.strike - test_strike) * 100

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike

        # Determine bias
        if chain.underlying_price > max_pain_strike * 1.02:
            bias = "bearish (price above max pain)"
        elif chain.underlying_price < max_pain_strike * 0.98:
            bias = "bullish (price below max pain)"
        else:
            bias = "neutral (price near max pain)"

        return {
            "max_pain": max_pain_strike,
            "current_price": chain.underlying_price,
            "distance_pct": ((chain.underlying_price - max_pain_strike) / max_pain_strike * 100)
                if max_pain_strike > 0 else 0,
            "bias": bias,
        }

    def get_unusual_activity(
        self,
        chain: OptionChain,
        volume_oi_threshold: float = 1.5,
        min_volume: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Find options with unusual activity.

        Args:
            chain: Option chain to analyze
            volume_oi_threshold: Minimum volume/OI ratio to flag
            min_volume: Minimum volume to consider

        Returns:
            List of unusual contracts
        """
        unusual = []

        all_contracts = chain.calls + chain.puts

        for contract in all_contracts:
            if contract.volume >= min_volume:
                ratio = contract.volume_oi_ratio
                if ratio >= volume_oi_threshold or contract.open_interest == 0:
                    unusual.append({
                        "type": contract.contract_type,
                        "strike": contract.strike,
                        "expiration": contract.expiration,
                        "volume": contract.volume,
                        "open_interest": contract.open_interest,
                        "volume_oi_ratio": ratio,
                        "implied_volatility": contract.implied_volatility,
                        "last_price": contract.last,
                    })

        # Sort by volume (highest first)
        unusual.sort(key=lambda x: x['volume'], reverse=True)

        return unusual[:10]  # Top 10

    def format_options_summary(self, symbol: str) -> str:
        """
        Generate a markdown summary of options data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Formatted markdown string
        """
        chain = self.get_option_chain(symbol)
        if not chain:
            return f"No options data available for {symbol}"

        pc_ratio = self.calculate_put_call_ratio(chain)
        max_pain = self.calculate_max_pain(chain)
        unusual = self.get_unusual_activity(chain)

        lines = [
            f"## Options Data ({symbol})",
            f"**Expiration:** {chain.expiration_dates[0] if chain.expiration_dates else 'N/A'}",
            f"**Underlying:** ${chain.underlying_price:.2f}",
            "",
            "### Put/Call Ratio",
            f"- Volume: {pc_ratio['volume_ratio']:.2f}",
            f"- Open Interest: {pc_ratio['oi_ratio']:.2f}",
            "",
            "### Max Pain Analysis",
            f"- Max Pain Strike: ${max_pain['max_pain']:.2f}",
            f"- Distance: {max_pain['distance_pct']:+.1f}%",
            f"- Bias: {max_pain['bias']}",
        ]

        if unusual:
            lines.extend([
                "",
                "### Unusual Activity (Top 5)",
            ])
            for i, u in enumerate(unusual[:5]):
                lines.append(
                    f"- {u['type'].upper()} ${u['strike']:.0f}: "
                    f"Vol {u['volume']:,} / OI {u['open_interest']:,} "
                    f"(IV: {u['implied_volatility']*100:.1f}%)"
                )

        return "\n".join(lines)


# Convenience functions
def get_options_summary(symbol: str) -> str:
    """Quick options summary lookup."""
    client = OptionsClient()
    if not client.is_available():
        return "Options data not available (yfinance not installed)"
    return client.format_options_summary(symbol)


def get_put_call_ratio(symbol: str) -> Optional[Dict[str, float]]:
    """Get put/call ratio for a symbol."""
    client = OptionsClient()
    if not client.is_available():
        return None

    chain = client.get_option_chain(symbol)
    if not chain:
        return None

    return client.calculate_put_call_ratio(chain)
