"""
Market Search Integration using Google Search Grounding

Real-time stock news, analyst ratings, and market sentiment via Gemini's
native Google Search grounding.

Adapted from google_search.py for stock-specific queries.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)

# Default model for market search
DEFAULT_SEARCH_MODEL = "gemini-3-flash-preview"


def _normalize_model_name(model_name: str) -> str:
    """Normalize model name - use as-is for Gemini 3 models."""
    return model_name


@dataclass
class MarketSource:
    """A source from market news search."""
    title: str
    url: str
    snippet: str = ""
    source_type: str = "news"  # news, analysis, research

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source_type": self.source_type,
        }


@dataclass
class MarketSearchResult:
    """Result from market search with grounding."""
    content: str
    sources: List[MarketSource] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    sentiment: str = ""  # bullish, bearish, neutral, mixed
    key_topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "sources": [s.to_dict() for s in self.sources],
            "search_queries": self.search_queries,
            "sentiment": self.sentiment,
            "key_topics": self.key_topics,
        }

    def format_sources(self, max_sources: int = 5) -> str:
        """Format sources for display."""
        if not self.sources:
            return ""

        lines = ["\n**Sources:**"]
        for i, source in enumerate(self.sources[:max_sources], 1):
            lines.append(f"{i}. [{source.title[:50]}]({source.url})")

        return "\n".join(lines)


class MarketSearchClient:
    """
    Client for stock market search using Gemini with Google Search grounding.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(settings, 'GEMINI_API_KEY', None) or os.getenv("GOOGLE_API_KEY")
        self._model = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        """Get or initialize the genai client."""
        if getattr(self, '_client', None) is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                logger.info("Initialized google.genai client")
            except ImportError:
                logger.error("google-genai package not installed")
                return None
            except Exception as e:
                logger.error(f"Failed to initialize client: {e}")
                return None
        return self._client

    def search_stock_news(
        self,
        symbol: str,
        query: Optional[str] = None,
        model_name: str = "gemini-3-flash-preview"
    ) -> MarketSearchResult:
        """
        Search for recent news about a stock.

        Args:
            symbol: Stock ticker symbol
            query: Optional specific query (defaults to general news)
            model_name: Model to use

        Returns:
            MarketSearchResult with news content and sources
        """
        if not query:
            query = f"What is the latest news about {symbol} stock? Include recent headlines, analyst opinions, and any significant developments."

        system_context = (
            f"You are a financial news analyst. Search for and summarize the most recent news "
            f"about {symbol} stock. Focus on:\n"
            "- Major news headlines and developments\n"
            "- Analyst ratings and price target changes\n"
            "- Earnings or guidance updates\n"
            "- Sector or market trends affecting the stock\n\n"
            "Provide a factual summary with dates when available. Note the overall sentiment."
        )

        return self._search_with_grounding(query, system_context, model_name)

    def search_market_sentiment(
        self,
        topic: str,
        model_name: str = "gemini-3-flash-preview"
    ) -> MarketSearchResult:
        """
        Search for market sentiment on a topic.

        Args:
            topic: Market topic or theme (e.g., "AI stocks", "semiconductor sector")
            model_name: Model to use

        Returns:
            MarketSearchResult with sentiment analysis
        """
        query = f"What is the current market sentiment on {topic}? Include recent analyst views, institutional positioning, and retail sentiment."

        system_context = (
            f"You are a market sentiment analyst. Search for and analyze current sentiment about {topic}.\n"
            "Include:\n"
            "- Professional analyst consensus\n"
            "- Institutional investor positioning\n"
            "- Retail investor sentiment (from social media, forums)\n"
            "- Recent price action and trading volume context\n\n"
            "Classify overall sentiment as: bullish, bearish, neutral, or mixed."
        )

        return self._search_with_grounding(query, system_context, model_name)

    def search_analyst_ratings(
        self,
        symbol: str,
        model_name: str = "gemini-3-flash-preview"
    ) -> MarketSearchResult:
        """
        Search for analyst ratings and price targets.

        Args:
            symbol: Stock ticker symbol
            model_name: Model to use

        Returns:
            MarketSearchResult with analyst data
        """
        query = f"What are the current analyst ratings and price targets for {symbol}? Include recent upgrades, downgrades, and consensus estimates."

        system_context = (
            f"You are a research analyst. Search for the latest analyst coverage of {symbol}.\n"
            "Include:\n"
            "- Current consensus rating (Buy/Hold/Sell distribution)\n"
            "- Average and range of price targets\n"
            "- Recent rating changes (upgrades/downgrades)\n"
            "- Notable analyst commentary or thesis\n\n"
            "Focus on data from major investment banks and research firms."
        )

        return self._search_with_grounding(query, system_context, model_name)

    def search_earnings_info(
        self,
        symbol: str,
        model_name: str = "gemini-3-flash-preview"
    ) -> MarketSearchResult:
        """
        Search for earnings information and estimates.

        Args:
            symbol: Stock ticker symbol
            model_name: Model to use

        Returns:
            MarketSearchResult with earnings data
        """
        query = f"What are the earnings estimates and recent results for {symbol}? Include next earnings date, EPS estimates, and recent earnings surprises."

        system_context = (
            f"You are an earnings analyst. Search for earnings information about {symbol}.\n"
            "Include:\n"
            "- Next earnings date and time\n"
            "- Current quarter EPS and revenue estimates\n"
            "- Recent earnings history (beat/miss)\n"
            "- Key metrics to watch\n"
            "- Any pre-announcements or guidance updates\n"
        )

        return self._search_with_grounding(query, system_context, model_name)

    def search_why_stock_moved(
        self,
        symbol: str,
        direction: str = "moved",  # "up", "down", "moved"
        model_name: str = "gemini-3-flash-preview"
    ) -> MarketSearchResult:
        """
        Search for reasons behind stock price movement.

        Args:
            symbol: Stock ticker symbol
            direction: Price direction (up, down, or moved)
            model_name: Model to use

        Returns:
            MarketSearchResult explaining the move
        """
        direction_text = {
            "up": "went up",
            "down": "went down",
            "moved": "moved significantly"
        }.get(direction, "moved")

        query = f"Why did {symbol} stock {direction_text} recently? What news or events caused the price movement?"

        system_context = (
            f"You are a market analyst explaining stock movements. Search for why {symbol} {direction_text}.\n"
            "Include:\n"
            "- Primary catalyst or reason for the move\n"
            "- Supporting news or events\n"
            "- Market context (sector moves, macro factors)\n"
            "- Timeline of events if relevant\n\n"
            "Be specific about dates and sources."
        )

        return self._search_with_grounding(query, system_context, model_name)

    def _search_with_grounding(
        self,
        query: str,
        system_context: str,
        model_name: str = "gemini-3-flash-preview"
    ) -> MarketSearchResult:
        """
        Internal method to search with Google grounding.

        Args:
            query: Search query
            system_context: System instructions
            model_name: Model to use

        Returns:
            MarketSearchResult
        """
        if not self.is_available():
            return MarketSearchResult(
                content="Market search not available - API key not configured",
                sources=[],
            )

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return MarketSearchResult(
                content="google-genai package not installed",
                sources=[],
            )

        client = self._get_client()
        if not client:
            return MarketSearchResult(
                content="Failed to initialize client",
                sources=[],
            )

        # Configure grounding tool using new google.genai API
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        config = types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0.5,
        )

        full_prompt = f"System: {system_context}\n\nUser: {query}"

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=config,
            )

            # Extract content and sources
            content_text = ""
            sources = []
            search_queries = []

            if hasattr(response, 'text'):
                content_text = response.text

            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]

                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    metadata = candidate.grounding_metadata

                    # Get search queries
                    if hasattr(metadata, 'web_search_queries') and metadata.web_search_queries:
                        search_queries = list(metadata.web_search_queries)

                    # Get grounding chunks (sources)
                    if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                        for chunk in metadata.grounding_chunks:
                            if hasattr(chunk, 'web') and chunk.web:
                                sources.append(MarketSource(
                                    title=getattr(chunk.web, 'title', None) or "Web Source",
                                    url=getattr(chunk.web, 'uri', None) or "",
                                    snippet="",
                                    source_type="news"
                                ))

            # Detect sentiment from content
            sentiment = self._detect_sentiment(content_text)

            return MarketSearchResult(
                content=content_text,
                sources=sources,
                search_queries=search_queries,
                sentiment=sentiment,
            )

        except Exception as e:
            logger.error(f"Market search failed: {e}")
            return MarketSearchResult(
                content=f"Search failed: {str(e)}",
                sources=[],
            )

    def _detect_sentiment(self, text: str) -> str:
        """Detect sentiment from text content."""
        text_lower = text.lower()

        bullish_keywords = [
            "upgrade", "buy", "bullish", "outperform", "strong buy",
            "positive", "growth", "beat", "exceeded", "raised",
            "surge", "rally", "breakout"
        ]
        bearish_keywords = [
            "downgrade", "sell", "bearish", "underperform", "hold",
            "negative", "decline", "miss", "lowered", "cut",
            "drop", "fall", "breakdown"
        ]

        bullish_count = sum(1 for kw in bullish_keywords if kw in text_lower)
        bearish_count = sum(1 for kw in bearish_keywords if kw in text_lower)

        if bullish_count > bearish_count + 2:
            return "bullish"
        elif bearish_count > bullish_count + 2:
            return "bearish"
        elif bullish_count > 0 and bearish_count > 0:
            return "mixed"
        return "neutral"


# Convenience functions
def search_stock_news(symbol: str, query: Optional[str] = None) -> MarketSearchResult:
    """Quick stock news search."""
    client = MarketSearchClient()
    if not client.is_available():
        logger.warning("Market search API not configured")
        return MarketSearchResult(content="API not configured", sources=[])
    return client.search_stock_news(symbol, query)


def search_why_moved(symbol: str, direction: str = "moved") -> MarketSearchResult:
    """Search for why a stock moved."""
    client = MarketSearchClient()
    if not client.is_available():
        return MarketSearchResult(content="API not configured", sources=[])
    return client.search_why_stock_moved(symbol, direction)


def search_general_query(query: str) -> MarketSearchResult:
    """
    Search for general stock/market questions using Google Search grounding.

    Handles questions like:
    - "When does SpaceX go public?"
    - "What is the best AI stock to buy?"
    - "How do interest rates affect stocks?"

    Args:
        query: User's general stock/market question

    Returns:
        MarketSearchResult with answer and sources
    """
    client = MarketSearchClient()
    if not client.is_available():
        return MarketSearchResult(content="API not configured", sources=[])

    system_context = (
        "You are a knowledgeable financial analyst and market researcher. "
        "Answer the user's question about stocks, markets, investing, or finance. "
        "Search for the most current and accurate information.\n\n"
        "Guidelines:\n"
        "- Provide factual, well-researched answers\n"
        "- Include specific dates, numbers, and sources when available\n"
        "- For IPO/public offering questions, note the current status and any announced timelines\n"
        "- For investment questions, present balanced perspectives\n"
        "- Always note that this is for informational purposes, not financial advice\n"
    )

    return client._search_with_grounding(query, system_context)


def get_market_context(symbol: str) -> str:
    """Get formatted market context for expert prompts."""
    client = MarketSearchClient()
    if not client.is_available():
        return f"[Market search not available for {symbol}]"

    result = client.search_stock_news(symbol)
    if result.content:
        context = f"## Real-Time Market News for {symbol}\n\n{result.content}"
        if result.sources:
            context += result.format_sources(max_sources=3)
        return context
    return f"[No recent news found for {symbol}]"
