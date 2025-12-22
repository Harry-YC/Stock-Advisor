"""
KOL Analyzer Service

Extracts structured claims from KOL text (pasted from Twitter, Substack, etc.)
and provides analysis context for expert panel.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class ExtractedClaim:
    """A claim extracted from KOL text."""
    author: str = "Unknown"
    platform: str = ""
    ticker: str = ""
    direction: str = "neutral"  # bullish, bearish, neutral
    target_price: Optional[float] = None
    target_date: Optional[str] = None
    thesis: str = ""
    key_points: List[str] = field(default_factory=list)
    confidence: str = "medium"  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        return {
            "author": self.author,
            "platform": self.platform,
            "ticker": self.ticker,
            "direction": self.direction,
            "target_price": self.target_price,
            "target_date": self.target_date,
            "thesis": self.thesis,
            "key_points": self.key_points,
            "confidence": self.confidence,
        }

    def format_summary(self) -> str:
        """Format as readable summary."""
        lines = [f"**@{self.author}** ({self.platform or 'Unknown'})"]

        if self.ticker:
            direction_emoji = {"bullish": "🐂", "bearish": "🐻", "neutral": "➖"}.get(
                self.direction.lower(), "❓"
            )
            lines.append(f"**{self.ticker}**: {direction_emoji} {self.direction.title()}")

        if self.target_price:
            lines.append(f"**Target:** ${self.target_price:.2f}")

        if self.target_date:
            lines.append(f"**Timeframe:** {self.target_date}")

        if self.thesis:
            lines.append(f"\n**Thesis:** {self.thesis}")

        if self.key_points:
            lines.append("\n**Key Points:**")
            for point in self.key_points[:5]:
                lines.append(f"• {point}")

        return "\n".join(lines)


class KOLAnalyzer:
    """
    Analyzes KOL text to extract structured claims.

    Uses Gemini to parse natural language posts and extract:
    - Author and platform
    - Ticker symbols and direction
    - Price targets and timeframes
    - Key thesis points
    """

    def __init__(self):
        self._router = None

    def _get_router(self):
        """Lazy load LLM router."""
        if self._router is None:
            from services.llm_router import get_llm_router
            self._router = get_llm_router()
        return self._router

    def extract_claims(self, text: str) -> ExtractedClaim:
        """
        Extract structured claims from KOL text.

        Args:
            text: Raw KOL post text (copy/pasted)

        Returns:
            ExtractedClaim with parsed data
        """
        router = self._get_router()
        if not router:
            return self._fallback_extract(text)

        prompt = f"""Extract stock trading claims from this KOL post. Return ONLY valid JSON.

POST:
\"\"\"
{text[:2000]}
\"\"\"

Return JSON:
{{
  "author": "username or name of the poster",
  "platform": "Twitter/X, Substack, YouTube, etc.",
  "ticker": "primary stock ticker mentioned (e.g., NVDA, AAPL)",
  "direction": "bullish, bearish, or neutral",
  "target_price": null or number (e.g., 150.00),
  "target_date": null or timeframe string (e.g., "Q2 2025", "by March"),
  "thesis": "one sentence summary of their main argument",
  "key_points": ["list", "of", "key", "claims"],
  "confidence": "low, medium, or high based on specificity of claims"
}}

If no clear stock opinion is found, return direction as "neutral".
Return ONLY the JSON object, no other text."""

        try:
            response_text = ""
            for chunk in router.call_expert_stream(
                prompt=prompt,
                system="You are a financial text parser. Extract stock trading claims from social media posts. Return only valid JSON."
            ):
                if chunk.get("type") == "chunk":
                    response_text += chunk.get("content", "")

            # Parse JSON from response
            try:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start != -1 and end > start:
                    data = json.loads(response_text[start:end])
                    return ExtractedClaim(
                        author=data.get("author", "Unknown"),
                        platform=data.get("platform", ""),
                        ticker=data.get("ticker", "").upper() if data.get("ticker") else "",
                        direction=data.get("direction", "neutral"),
                        target_price=data.get("target_price"),
                        target_date=data.get("target_date"),
                        thesis=data.get("thesis", ""),
                        key_points=data.get("key_points", []),
                        confidence=data.get("confidence", "medium"),
                    )
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM JSON response")

        except Exception as e:
            logger.error(f"KOL extraction failed: {e}")

        return self._fallback_extract(text)

    def _fallback_extract(self, text: str) -> ExtractedClaim:
        """Fallback extraction using regex patterns."""
        # Extract tickers
        ticker_pattern = r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b(?=\s+(?:stock|shares|calls?|puts?))'
        matches = re.findall(ticker_pattern, text.upper())
        tickers = [m[0] or m[1] for m in matches if m[0] or m[1]]

        # Filter common words
        common_words = {'THE', 'FOR', 'AND', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAS'}
        tickers = [t for t in tickers if t not in common_words]

        # Detect direction
        text_lower = text.lower()
        bullish_words = ['buy', 'long', 'bullish', 'moon', 'calls', 'upgrade', 'undervalued']
        bearish_words = ['sell', 'short', 'bearish', 'puts', 'downgrade', 'overvalued']

        bullish_count = sum(1 for w in bullish_words if w in text_lower)
        bearish_count = sum(1 for w in bearish_words if w in text_lower)

        if bullish_count > bearish_count:
            direction = "bullish"
        elif bearish_count > bullish_count:
            direction = "bearish"
        else:
            direction = "neutral"

        # Extract price targets
        price_pattern = r'\$(\d+(?:\.\d{2})?)\b'
        prices = re.findall(price_pattern, text)
        target_price = float(prices[0]) if prices else None

        # Extract author (look for @ mentions)
        author_pattern = r'@(\w+)'
        authors = re.findall(author_pattern, text)
        author = authors[0] if authors else "Unknown"

        return ExtractedClaim(
            author=author,
            ticker=tickers[0] if tickers else "",
            direction=direction,
            target_price=target_price,
            thesis=text[:200] + "..." if len(text) > 200 else text,
        )

    def format_for_expert_context(self, claim: ExtractedClaim) -> str:
        """Format extracted claim as context for expert panel."""
        lines = [
            "## KOL Opinion Being Analyzed",
            "",
            f"**Author:** @{claim.author}" + (f" ({claim.platform})" if claim.platform else ""),
        ]

        if claim.ticker:
            lines.append(f"**Ticker:** {claim.ticker}")
            lines.append(f"**Direction:** {claim.direction.title()}")

        if claim.target_price:
            lines.append(f"**Price Target:** ${claim.target_price:.2f}")

        if claim.target_date:
            lines.append(f"**Timeframe:** {claim.target_date}")

        if claim.thesis:
            lines.append(f"\n**Their Thesis:**\n> {claim.thesis}")

        if claim.key_points:
            lines.append("\n**Claims to Validate:**")
            for i, point in enumerate(claim.key_points[:5], 1):
                lines.append(f"{i}. {point}")

        lines.append(f"\n*Claim specificity: {claim.confidence}*")

        return "\n".join(lines)


# Convenience functions
def analyze_kol_text(text: str) -> ExtractedClaim:
    """Quick analysis of KOL text."""
    analyzer = KOLAnalyzer()
    return analyzer.extract_claims(text)


def get_kol_context(text: str) -> str:
    """Get formatted context from KOL text for expert panel."""
    analyzer = KOLAnalyzer()
    claim = analyzer.extract_claims(text)
    return analyzer.format_for_expert_context(claim)
