"""
Gemini Vision API Integration

OCR and analysis for KOL (Key Opinion Leader) screenshot uploads.
Extracts author, platform, tickers, sentiment, and key claims.

Uses Gemini 2.0 Flash for vision capabilities.
"""

import os
import re
import json
import base64
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result of KOL screenshot analysis."""
    success: bool
    author: str = ""
    platform: str = ""
    tickers: List[str] = field(default_factory=list)
    sentiment: str = ""  # bullish, bearish, neutral, mixed
    key_claims: List[str] = field(default_factory=list)
    full_text: str = ""
    confidence: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "author": self.author,
            "platform": self.platform,
            "tickers": self.tickers,
            "sentiment": self.sentiment,
            "key_claims": self.key_claims,
            "full_text": self.full_text,
            "confidence": self.confidence,
            "error": self.error,
        }

    def format_summary(self) -> str:
        """Format result as markdown summary."""
        if not self.success:
            return f"**OCR Error:** {self.error}"

        lines = ["**KOL Screenshot Analysis**\n"]

        if self.author:
            lines.append(f"**Author:** {self.author}")
        if self.platform:
            lines.append(f"**Platform:** {self.platform}")

        if self.tickers:
            lines.append(f"**Tickers Mentioned:** {', '.join(self.tickers)}")

        if self.sentiment:
            sentiment_emoji = {
                "bullish": "🐂",
                "bearish": "🐻",
                "neutral": "➖",
                "mixed": "🔄"
            }.get(self.sentiment.lower(), "❓")
            lines.append(f"**Sentiment:** {sentiment_emoji} {self.sentiment.title()}")

        if self.key_claims:
            lines.append("\n**Key Claims:**")
            for i, claim in enumerate(self.key_claims[:5], 1):
                lines.append(f"  {i}. {claim}")

        if self.confidence:
            lines.append(f"\n*Confidence: {self.confidence:.0%}*")

        return "\n".join(lines)


# Common ticker patterns
TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b(?=\s+(?:stock|shares|price|calls?|puts?))')

# Known platforms
PLATFORMS = {
    "twitter": ["twitter", "x.com", "@"],
    "reddit": ["reddit", "r/", "wsb", "wallstreetbets"],
    "stocktwits": ["stocktwits"],
    "youtube": ["youtube", "youtu.be"],
    "tradingview": ["tradingview"],
    "discord": ["discord"],
    "telegram": ["telegram", "t.me"],
}

# Sentiment keywords
BULLISH_KEYWORDS = [
    "buy", "bullish", "long", "calls", "moon", "rocket", "squeeze",
    "undervalued", "breakout", "accumulate", "load up", "going up",
    "target", "upside", "strong", "growth", "opportunity"
]
BEARISH_KEYWORDS = [
    "sell", "bearish", "short", "puts", "crash", "dump", "overvalued",
    "breakdown", "avoid", "sell off", "going down", "downside",
    "weak", "risk", "decline", "warning"
]


class GeminiVisionClient:
    """
    Client for Gemini Vision API.

    Analyzes KOL screenshots to extract structured information.
    """

    MAX_IMAGE_SIZE_MB = 5
    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(settings, 'GEMINI_API_KEY', None) or os.getenv("GEMINI_API_KEY")
        self._client = None

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    def _get_client(self):
        """Lazy-load Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel("gemini-3-flash-preview")
            except ImportError:
                logger.error("google-generativeai package not installed")
                raise RuntimeError("google-generativeai package required for vision")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                raise
        return self._client

    def _validate_image(self, image_path: str) -> tuple[bool, str]:
        """Validate image file."""
        path = Path(image_path)

        if not path.exists():
            return False, f"File not found: {image_path}"

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            return False, f"Unsupported format: {suffix}. Supported: {', '.join(self.SUPPORTED_FORMATS)}"

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.MAX_IMAGE_SIZE_MB:
            return False, f"File too large: {size_mb:.1f}MB. Max: {self.MAX_IMAGE_SIZE_MB}MB"

        return True, ""

    def _load_image(self, image_path: str) -> tuple[bytes, str]:
        """Load image and return bytes and mime type."""
        path = Path(image_path)
        suffix = path.suffix.lower()

        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }

        with open(path, "rb") as f:
            image_bytes = f.read()

        return image_bytes, mime_types.get(suffix, "image/png")

    def extract_kol_post(self, image_path: str) -> OCRResult:
        """
        Extract KOL post information from a screenshot.

        Args:
            image_path: Path to the image file

        Returns:
            OCRResult with extracted information
        """
        if not self.is_available():
            return OCRResult(success=False, error="Gemini API key not configured")

        # Validate image
        valid, error = self._validate_image(image_path)
        if not valid:
            return OCRResult(success=False, error=error)

        try:
            # Load image
            image_bytes, mime_type = self._load_image(image_path)

            # Create prompt for structured extraction
            prompt = """Analyze this screenshot of a social media post or financial content.

Extract the following information in JSON format:
{
    "author": "The author's name or handle (e.g., @username)",
    "platform": "The platform (twitter, reddit, stocktwits, youtube, etc.)",
    "tickers": ["List of stock ticker symbols mentioned (e.g., NVDA, AAPL)"],
    "sentiment": "Overall sentiment: bullish, bearish, neutral, or mixed",
    "key_claims": ["List of key claims or predictions made about stocks"],
    "full_text": "The full text content of the post",
    "confidence": 0.0 to 1.0 confidence in the extraction
}

Focus on:
1. Identifying stock tickers (with $ prefix or mentioned with "stock", "shares", etc.)
2. Determining if the sentiment is bullish (buy/long) or bearish (sell/short)
3. Extracting specific price targets, dates, or predictions
4. Noting any disclaimers or risk warnings

Return ONLY valid JSON, no markdown or explanations."""

            # Call Gemini Vision
            client = self._get_client()

            # Create image part for the API
            import google.generativeai as genai
            image_part = {
                "mime_type": mime_type,
                "data": base64.b64encode(image_bytes).decode()
            }

            response = client.generate_content([prompt, image_part])

            if not response or not response.text:
                return OCRResult(success=False, error="No response from Gemini Vision")

            # Parse JSON response
            try:
                # Clean response text (remove markdown code blocks if present)
                text = response.text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                text = text.strip()

                data = json.loads(text)

                return OCRResult(
                    success=True,
                    author=data.get("author", ""),
                    platform=data.get("platform", ""),
                    tickers=data.get("tickers", []),
                    sentiment=data.get("sentiment", ""),
                    key_claims=data.get("key_claims", []),
                    full_text=data.get("full_text", ""),
                    confidence=float(data.get("confidence", 0.7)),
                )

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                # Fallback: try to extract information from raw text
                return self._fallback_extraction(response.text)

        except Exception as e:
            logger.error(f"Vision extraction failed: {e}")
            return OCRResult(success=False, error=str(e))

    def _fallback_extraction(self, text: str) -> OCRResult:
        """Fallback extraction when JSON parsing fails."""
        result = OCRResult(success=True, full_text=text, confidence=0.5)

        # Extract tickers
        ticker_matches = TICKER_PATTERN.findall(text.upper())
        tickers = set()
        for match in ticker_matches:
            ticker = match[0] or match[1]
            if ticker and len(ticker) >= 2:
                tickers.add(ticker)
        result.tickers = list(tickers)

        # Detect platform
        text_lower = text.lower()
        for platform, keywords in PLATFORMS.items():
            if any(kw in text_lower for kw in keywords):
                result.platform = platform
                break

        # Detect sentiment
        bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)

        if bullish_count > bearish_count + 2:
            result.sentiment = "bullish"
        elif bearish_count > bullish_count + 2:
            result.sentiment = "bearish"
        elif bullish_count > 0 and bearish_count > 0:
            result.sentiment = "mixed"
        else:
            result.sentiment = "neutral"

        # Extract potential author (@ mentions)
        author_match = re.search(r'@(\w+)', text)
        if author_match:
            result.author = f"@{author_match.group(1)}"

        return result

    def extract_from_bytes(self, image_bytes: bytes, mime_type: str = "image/png") -> OCRResult:
        """
        Extract KOL post information from image bytes.

        Args:
            image_bytes: Image data as bytes
            mime_type: MIME type of the image

        Returns:
            OCRResult with extracted information
        """
        if not self.is_available():
            return OCRResult(success=False, error="Gemini API key not configured")

        # Check size
        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > self.MAX_IMAGE_SIZE_MB:
            return OCRResult(success=False, error=f"Image too large: {size_mb:.1f}MB. Max: {self.MAX_IMAGE_SIZE_MB}MB")

        try:
            prompt = """Analyze this screenshot of a social media post or financial content.

Extract the following information in JSON format:
{
    "author": "The author's name or handle (e.g., @username)",
    "platform": "The platform (twitter, reddit, stocktwits, youtube, etc.)",
    "tickers": ["List of stock ticker symbols mentioned (e.g., NVDA, AAPL)"],
    "sentiment": "Overall sentiment: bullish, bearish, neutral, or mixed",
    "key_claims": ["List of key claims or predictions made about stocks"],
    "full_text": "The full text content of the post",
    "confidence": 0.0 to 1.0 confidence in the extraction
}

Return ONLY valid JSON, no markdown or explanations."""

            client = self._get_client()

            image_part = {
                "mime_type": mime_type,
                "data": base64.b64encode(image_bytes).decode()
            }

            response = client.generate_content([prompt, image_part])

            if not response or not response.text:
                return OCRResult(success=False, error="No response from Gemini Vision")

            # Parse response
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            try:
                data = json.loads(text)
                return OCRResult(
                    success=True,
                    author=data.get("author", ""),
                    platform=data.get("platform", ""),
                    tickers=data.get("tickers", []),
                    sentiment=data.get("sentiment", ""),
                    key_claims=data.get("key_claims", []),
                    full_text=data.get("full_text", ""),
                    confidence=float(data.get("confidence", 0.7)),
                )
            except json.JSONDecodeError:
                return self._fallback_extraction(response.text)

        except Exception as e:
            logger.error(f"Vision extraction from bytes failed: {e}")
            return OCRResult(success=False, error=str(e))


# Convenience functions
def analyze_kol_screenshot(image_path: str) -> OCRResult:
    """
    Analyze a KOL screenshot.

    Args:
        image_path: Path to image file

    Returns:
        OCRResult with extracted information
    """
    client = GeminiVisionClient()
    if not client.is_available():
        logger.warning("Gemini Vision API not configured")
        return OCRResult(success=False, error="API not configured")
    return client.extract_kol_post(image_path)


def extract_tickers_from_text(text: str) -> List[str]:
    """
    Extract stock tickers from text.

    Args:
        text: Text to search

    Returns:
        List of ticker symbols
    """
    matches = TICKER_PATTERN.findall(text.upper())
    tickers = set()
    for match in matches:
        ticker = match[0] or match[1]
        if ticker and len(ticker) >= 2:
            tickers.add(ticker)
    return list(tickers)
