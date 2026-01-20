"""
Stock Data Service

Aggregates data from multiple sources (Finnhub, Market Search, KOL OCR)
and provides formatted context for expert prompts.
"""

import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Lazy-loaded clients
_finnhub_client = None
_alpha_vantage_client = None
_market_search_client = None
_vision_client = None
_grok_client = None


def _get_finnhub_client():
    """Lazy-load Finnhub client."""
    global _finnhub_client
    if _finnhub_client is None:
        try:
            from integrations.finnhub import FinnhubClient
            import os
            logger.info(f"Creating FinnhubClient, FINNHUB_API_KEY set: {bool(os.getenv('FINNHUB_API_KEY'))}")
            _finnhub_client = FinnhubClient()
            logger.info(f"FinnhubClient created, is_available: {_finnhub_client.is_available()}")
        except ImportError as e:
            logger.warning(f"Finnhub client not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create Finnhub client: {e}")
            return None
    return _finnhub_client


def _get_alpha_vantage_client():
    """Lazy-load Alpha Vantage client (fallback)."""
    global _alpha_vantage_client
    if _alpha_vantage_client is None:
        try:
            from integrations.alpha_vantage import AlphaVantageClient
            _alpha_vantage_client = AlphaVantageClient()
        except ImportError:
            logger.warning("Alpha Vantage client not available")
            return None
    return _alpha_vantage_client


def _get_market_search_client():
    """Lazy-load Market Search client."""
    global _market_search_client
    if _market_search_client is None:
        try:
            from integrations.market_search import MarketSearchClient
            _market_search_client = MarketSearchClient()
        except ImportError:
            logger.warning("Market search client not available")
            return None
    return _market_search_client


def _get_vision_client():
    """Lazy-load Vision client."""
    global _vision_client
    if _vision_client is None:
        try:
            from integrations.gemini_vision import GeminiVisionClient
            _vision_client = GeminiVisionClient()
        except ImportError:
            logger.warning("Vision client not available")
            return None
    return _vision_client


def _get_grok_client():
    """Lazy-load Grok client for X/Twitter insights."""
    global _grok_client
    if _grok_client is None:
        try:
            from services.grok_service import GrokService
            _grok_client = GrokService()
        except ImportError:
            logger.warning("Grok client not available")
            return None
    return _grok_client


# Common ticker pattern for extraction
TICKER_PATTERN = re.compile(
    r'\$([A-Z]{1,5})\b'  # $AAPL style
    r'|\b([A-Z]{2,5})\b(?=\s+(?:stock|shares|price|calls?|puts?|options?))'  # AAPL stock
)


@dataclass
class StockDataContext:
    """Aggregated stock data context for expert prompts."""
    symbol: str
    quote_summary: str = ""
    financials_summary: str = ""
    news_summary: str = ""
    market_context: str = ""
    kol_context: str = ""
    grok_context: str = ""  # X/Twitter sentiment from Grok
    data_available: Dict[str, bool] = field(default_factory=dict)

    def to_prompt_context(self) -> str:
        """Format all data as prompt context."""
        sections = []

        if self.quote_summary:
            sections.append(f"## Real-Time Quote\n{self.quote_summary}")

        if self.financials_summary:
            sections.append(f"## Financial Metrics\n{self.financials_summary}")

        if self.news_summary:
            sections.append(f"## Recent News\n{self.news_summary}")

        if self.market_context:
            sections.append(f"## Market Context\n{self.market_context}")

        if self.kol_context:
            sections.append(f"## KOL Analysis\n{self.kol_context}")

        if self.grok_context:
            sections.append(f"## X/Twitter Sentiment (Grok)\n{self.grok_context}")

        if not sections:
            return f"[No data available for {self.symbol}]"

        return "\n\n".join(sections)


def extract_tickers(text: str) -> List[str]:
    """
    Extract stock ticker symbols from text.

    Args:
        text: Text to search for tickers

    Returns:
        List of unique ticker symbols
    """
    matches = TICKER_PATTERN.findall(text.upper())
    tickers = set()
    for match in matches:
        ticker = match[0] or match[1]
        if ticker and len(ticker) >= 2:
            # Filter out common false positives
            if ticker not in {'THE', 'FOR', 'AND', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL',
                             'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'HAS', 'HIS',
                             'HOW', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'WAY', 'WHO',
                             'ITS', 'LET', 'SAY', 'SHE', 'TOO', 'USE'}:
                tickers.add(ticker)
    return list(tickers)


def _sanitize_for_prompt(text: str, max_length: int = 500) -> str:
    """
    Sanitize text for safe prompt injection.

    Blocks prompt injection patterns and limits length.
    """
    if not text:
        return ""

    # Block obvious prompt injection patterns
    injection_patterns = [
        r'ignore\s+(previous|all|above)',
        r'forget\s+(previous|all|your)',
        r'disregard\s+(previous|all|above)',
        r'new\s+instructions?:',
        r'system\s*:',
        r'assistant\s*:',
        r'user\s*:',
    ]

    text_lower = text.lower()
    for pattern in injection_patterns:
        if re.search(pattern, text_lower):
            logger.warning(f"Blocked potential prompt injection: {pattern}")
            return "[Content filtered for safety]"

    # Limit length
    if len(text) > max_length:
        text = text[:max_length] + "..."

    return text


def fetch_stock_data(
    symbol: str,
    include_quote: bool = True,
    include_financials: bool = True,
    include_news: bool = True,
    include_market_search: bool = False,  # Off by default - uses API calls
    include_grok: bool = False,  # Off by default - uses API calls
) -> StockDataContext:
    """
    Fetch comprehensive stock data from all available sources.

    Uses Finnhub as primary source, falls back to Alpha Vantage for
    stocks not covered by Finnhub (e.g., micro-cap stocks).

    Args:
        symbol: Stock ticker symbol
        include_quote: Fetch real-time quote from Finnhub
        include_financials: Fetch financial metrics from Finnhub
        include_news: Fetch recent news from Finnhub
        include_market_search: Fetch market context via Google Search
        include_grok: Fetch X/Twitter sentiment via Grok

    Returns:
        StockDataContext with all available data
    """
    symbol = symbol.upper()
    context = StockDataContext(symbol=symbol)
    used_fallback = False

    logger.info(f"fetch_stock_data called for {symbol}")

    # Finnhub data (primary source)
    finnhub = _get_finnhub_client()
    logger.info(f"Finnhub client: {finnhub}, available: {finnhub.is_available() if finnhub else 'N/A'}")
    if finnhub and finnhub.is_available():
        if include_quote:
            try:
                quote = finnhub.get_quote(symbol)
                if quote:
                    context.quote_summary = quote.format_summary()
                    context.data_available['quote'] = True
                    context.data_available['quote_source'] = 'finnhub'
                    logger.info(f"Fetched quote for {symbol} from Finnhub")
            except Exception as e:
                logger.error(f"Quote fetch failed for {symbol}: {e}")

        if include_financials:
            try:
                financials = finnhub.get_basic_financials(symbol)
                if financials:
                    context.financials_summary = financials.format_summary()
                    context.data_available['financials'] = True
                    context.data_available['financials_source'] = 'finnhub'
                    logger.info(f"Fetched financials for {symbol} from Finnhub")
            except Exception as e:
                logger.error(f"Financials fetch failed for {symbol}: {e}")

        if include_news:
            try:
                news = finnhub.get_company_news(symbol, limit=5)
                if news:
                    context.news_summary = finnhub.format_news(news, max_items=5)
                    context.data_available['news'] = True
                    logger.info(f"Fetched {len(news)} news items for {symbol}")
            except Exception as e:
                logger.error(f"News fetch failed for {symbol}: {e}")
    else:
        logger.warning("Finnhub client not available")

    # Alpha Vantage fallback for missing data
    alpha_vantage = _get_alpha_vantage_client()
    if alpha_vantage and alpha_vantage.is_available():
        # Fallback for quote if Finnhub didn't have it
        if include_quote and not context.data_available.get('quote'):
            try:
                av_quote = alpha_vantage.get_quote(symbol)
                if av_quote:
                    context.quote_summary = av_quote.format_summary()
                    context.data_available['quote'] = True
                    context.data_available['quote_source'] = 'alpha_vantage'
                    used_fallback = True
                    logger.info(f"Fetched quote for {symbol} from Alpha Vantage (fallback)")
            except Exception as e:
                logger.error(f"Alpha Vantage quote fetch failed for {symbol}: {e}")

        # Fallback for financials if Finnhub didn't have it
        if include_financials and not context.data_available.get('financials'):
            try:
                av_overview = alpha_vantage.get_company_overview(symbol)
                if av_overview:
                    context.financials_summary = av_overview.format_summary()
                    context.data_available['financials'] = True
                    context.data_available['financials_source'] = 'alpha_vantage'
                    used_fallback = True
                    logger.info(f"Fetched financials for {symbol} from Alpha Vantage (fallback)")
            except Exception as e:
                logger.error(f"Alpha Vantage overview fetch failed for {symbol}: {e}")

        if used_fallback:
            logger.info(f"Used Alpha Vantage as fallback for {symbol} (Finnhub had no data)")
    elif not context.data_available.get('quote'):
        logger.warning(f"No quote data available for {symbol} - Finnhub empty and Alpha Vantage not configured")

    # Market search (Google grounding)
    if include_market_search:
        search_client = _get_market_search_client()
        if search_client and search_client.is_available():
            try:
                result = search_client.search_stock_news(symbol)
                if result.content:
                    context.market_context = _sanitize_for_prompt(result.content, max_length=800)
                    # Inject sources for citation
                    if hasattr(result, 'sources') and result.sources:
                        sources_text = "\n\nSources:\n" + "\n".join(f"- {s}" for s in result.sources[:3])
                        context.market_context += sources_text
                    context.data_available['market_search'] = True
                    logger.info(f"Fetched market context for {symbol}")
            except Exception as e:
                logger.error(f"Market search failed for {symbol}: {e}")

    # Grok X/Twitter sentiment
    if include_grok:
        grok_client = _get_grok_client()
        if grok_client and grok_client.is_available():
            try:
                grok_result = grok_client.get_stock_sentiment(symbol)
                if grok_result and not grok_result.startswith("Error"):
                    context.grok_context = _sanitize_for_prompt(grok_result, max_length=1500)
                    context.data_available['grok'] = True
                    logger.info(f"Fetched Grok sentiment for {symbol}")
            except Exception as e:
                logger.error(f"Grok sentiment failed for {symbol}: {e}")

    return context


def fetch_multi_stock_data(
    symbols: List[str],
    include_quote: bool = True,
    include_financials: bool = False,  # Off by default for multi
    include_news: bool = False,  # Off by default for multi
) -> Dict[str, StockDataContext]:
    """
    Fetch data for multiple stocks.

    Args:
        symbols: List of ticker symbols
        include_quote: Fetch quotes
        include_financials: Fetch financials
        include_news: Fetch news

    Returns:
        Dict mapping symbol to StockDataContext
    """
    results = {}
    for symbol in symbols[:10]:  # Limit to 10 symbols
        results[symbol] = fetch_stock_data(
            symbol,
            include_quote=include_quote,
            include_financials=include_financials,
            include_news=include_news,
            include_market_search=False,  # Never for multi
        )
    return results


def analyze_kol_screenshot(image_path: str) -> Dict[str, Any]:
    """
    Analyze a KOL screenshot and return extracted data with stock context.

    Args:
        image_path: Path to the screenshot

    Returns:
        Dict with OCR result and any related stock data
    """
    vision = _get_vision_client()
    if not vision or not vision.is_available():
        return {
            "success": False,
            "error": "Vision API not available",
            "ocr_result": None,
            "stock_data": {},
        }

    try:
        from integrations.gemini_vision import OCRResult
        ocr_result = vision.extract_kol_post(image_path)

        result = {
            "success": ocr_result.success,
            "error": ocr_result.error if not ocr_result.success else "",
            "ocr_result": ocr_result.to_dict(),
            "stock_data": {},
        }

        # Fetch data for detected tickers
        if ocr_result.success and ocr_result.tickers:
            for ticker in ocr_result.tickers[:3]:  # Max 3 tickers
                try:
                    stock_data = fetch_stock_data(
                        ticker,
                        include_quote=True,
                        include_financials=True,
                        include_news=True,
                    )
                    result["stock_data"][ticker] = stock_data.to_prompt_context()
                except Exception as e:
                    logger.error(f"Failed to fetch data for {ticker}: {e}")

        return result

    except Exception as e:
        logger.error(f"KOL screenshot analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "ocr_result": None,
            "stock_data": {},
        }


def build_expert_context(
    symbol: str,
    question: str,
    kol_content: Optional[str] = None,
    include_market_search: bool = False,
    include_grok: bool = False,
) -> str:
    """
    Build comprehensive context for expert prompts.

    Args:
        symbol: Stock ticker symbol
        question: User's question
        kol_content: Optional KOL content being analyzed
        include_market_search: Include Google Search results
        include_grok: Include X/Twitter sentiment via Grok

    Returns:
        Formatted context string for expert prompts
    """
    # Fetch stock data
    context = fetch_stock_data(
        symbol,
        include_quote=True,
        include_financials=True,
        include_news=True,
        include_market_search=include_market_search,
        include_grok=include_grok,
    )

    # Add KOL content if provided
    if kol_content:
        context.kol_context = _sanitize_for_prompt(kol_content, max_length=800)

    # If include_grok is True, also check for CI dimensions and add specific CI searches
    if include_grok:
        grok_client = _get_grok_client()
        if grok_client and grok_client.is_available():
            try:
                from services.grok_service import detect_stock_ci_dimensions
                ci_dims = detect_stock_ci_dimensions(question)
                if ci_dims:
                    ci_results = []
                    for dim in ci_dims[:2]:  # Limit to 2 dimensions to avoid API overuse
                        ci_result = grok_client.competitive_intelligence_search(
                            dim, context=f"{symbol}: {question}"
                        )
                        if ci_result and not ci_result.startswith("Error"):
                            ci_results.append(f"### {dim.replace('_', ' ').title()}\n{ci_result}")
                    if ci_results:
                        context.grok_context += "\n\n" + "\n\n".join(ci_results)
                        context.data_available['grok_ci'] = True
            except Exception as e:
                logger.error(f"Grok CI search failed: {e}")

    # Build final context
    sections = [context.to_prompt_context()]

    # Add data availability summary
    available = [k for k, v in context.data_available.items() if v]
    if available:
        sections.append(f"\n*Data sources: {', '.join(available)}*")

    return "\n".join(sections)


def search_why_stock_moved(symbol: str, direction: str = "moved") -> str:
    """
    Search for reasons behind a stock's price movement.

    Args:
        symbol: Stock ticker
        direction: "up", "down", or "moved"

    Returns:
        Formatted explanation
    """
    search_client = _get_market_search_client()
    if not search_client or not search_client.is_available():
        return f"Market search not available for {symbol}"

    try:
        result = search_client.search_why_stock_moved(symbol, direction)
        if result.content:
            content = _sanitize_for_prompt(result.content, max_length=1500)
            if result.sources:
                content += result.format_sources(max_sources=3)
            return content
        return f"No explanation found for {symbol}'s movement"
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Search failed for {symbol}"


def get_analyst_ratings(symbol: str) -> str:
    """
    Get analyst ratings and price targets for a stock.

    Args:
        symbol: Stock ticker

    Returns:
        Formatted analyst data
    """
    search_client = _get_market_search_client()
    if not search_client or not search_client.is_available():
        return f"Analyst ratings not available for {symbol}"

    try:
        result = search_client.search_analyst_ratings(symbol)
        if result.content:
            return _sanitize_for_prompt(result.content, max_length=1000)
        return f"No analyst ratings found for {symbol}"
    except Exception as e:
        logger.error(f"Analyst ratings search failed: {e}")
        return f"Failed to fetch analyst ratings for {symbol}"
