"""
Grok Service for Stock Advisor

Interact with xAI API to fetch real-time KOL insights from X.
Optimized for stock/finance topics and trading sentiment.
"""

import os
import json
import time
import logging
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

GROK_MODELS = {
    "default": "grok-3-latest",
    "fast": "grok-3-fast",
    "beta": "grok-beta",
}

# =============================================================================
# ENHANCED KOL DATABASE WITH PROFILES
# =============================================================================

# KOL profiles with category, handle, bias, specialty, and activity level
KOL_PROFILES = {
    # MACRO STRATEGISTS - Big picture market calls
    "Michael Burry": {
        "handle": "@michaeljburry",
        "category": "macro",
        "bias": "contrarian_bear",
        "specialty": ["macro", "shorts", "value", "crash_calls"],
        "credibility": "high",  # The Big Short fame
        "post_frequency": "rare",  # Deletes posts often
        "signal_quality": "high_when_active",
    },
    "Stan Druckenmiller": {
        "handle": "@Stanadruckenmil",
        "category": "macro",
        "bias": "flexible",
        "specialty": ["macro", "currencies", "fed_policy", "tech"],
        "credibility": "legendary",
        "post_frequency": "low",
        "signal_quality": "very_high",
    },
    "Ray Dalio": {
        "handle": "@RayDalio",
        "category": "macro",
        "bias": "balanced",
        "specialty": ["macro", "debt_cycles", "china", "diversification"],
        "credibility": "high",
        "post_frequency": "medium",
        "signal_quality": "educational",
    },
    "Mohamed El-Erian": {
        "handle": "@elerianm",
        "category": "macro",
        "bias": "balanced",
        "specialty": ["fed", "bonds", "inflation", "central_banks"],
        "credibility": "high",
        "post_frequency": "high",
        "signal_quality": "high",
    },
    "Howard Marks": {
        "handle": "@HowardMarksBook",
        "category": "macro",
        "bias": "value_cautious",
        "specialty": ["cycles", "risk", "credit", "value"],
        "credibility": "legendary",
        "post_frequency": "low",
        "signal_quality": "very_high",
    },

    # ACTIVIST / HEDGE FUND MANAGERS - Position transparency
    "Bill Ackman": {
        "handle": "@BillAckman",
        "category": "activist",
        "bias": "concentrated_bull",
        "specialty": ["activism", "large_caps", "macro_hedges"],
        "credibility": "high",
        "post_frequency": "high",
        "signal_quality": "high",  # Often talks book
    },
    "David Einhorn": {
        "handle": "@davideinhorn",
        "category": "activist",
        "bias": "value_short",
        "specialty": ["value", "shorts", "financials"],
        "credibility": "high",
        "post_frequency": "low",
        "signal_quality": "high",
    },
    "Carl Icahn": {
        "handle": "@Carl_C_Icahn",
        "category": "activist",
        "bias": "activist_value",
        "specialty": ["activism", "energy", "conglomerates"],
        "credibility": "legendary",
        "post_frequency": "low",
        "signal_quality": "actionable",
    },

    # TECH / GROWTH ANALYSTS - Sector specialists
    "Cathie Wood": {
        "handle": "@CathieDWood",
        "category": "tech_growth",
        "bias": "perma_bull_innovation",
        "specialty": ["tech", "AI", "genomics", "fintech", "autonomous"],
        "credibility": "polarizing",  # Big wins and losses
        "post_frequency": "high",
        "signal_quality": "directional",  # Good for themes, timing varies
    },
    "Dan Ives": {
        "handle": "@DivesTech",
        "category": "tech_analyst",
        "bias": "tech_bull",
        "specialty": ["tech", "software", "cyber", "AI"],
        "credibility": "medium_high",
        "post_frequency": "high",
        "signal_quality": "good_coverage",
    },
    "Gene Munster": {
        "handle": "@munaboroster",
        "category": "tech_analyst",
        "bias": "tech_bull",
        "specialty": ["AAPL", "TSLA", "tech_hardware", "AR_VR"],
        "credibility": "medium",
        "post_frequency": "medium",
        "signal_quality": "medium",
    },
    "Beth Kindig": {
        "handle": "@Beth_Kindig",
        "category": "tech_analyst",
        "bias": "tech_selective",
        "specialty": ["semiconductors", "AI", "cloud", "deep_tech"],
        "credibility": "medium_high",
        "post_frequency": "medium",
        "signal_quality": "research_driven",
    },

    # OPTIONS / FLOW TRADERS - Real-time flow intel
    "Unusual Whales": {
        "handle": "@unusual_whales",
        "category": "options_flow",
        "bias": "data_neutral",
        "specialty": ["options", "flow", "congress_trades", "dark_pool"],
        "credibility": "data_source",
        "post_frequency": "very_high",
        "signal_quality": "raw_data",
    },
    "Cheddar Flow": {
        "handle": "@CheddarFlow",
        "category": "options_flow",
        "bias": "data_neutral",
        "specialty": ["options", "sweeps", "large_trades"],
        "credibility": "data_source",
        "post_frequency": "very_high",
        "signal_quality": "raw_data",
    },
    "SpotGamma": {
        "handle": "@spotgamma",
        "category": "options_flow",
        "bias": "technical",
        "specialty": ["gamma", "options_greeks", "dealer_positioning"],
        "credibility": "high",
        "post_frequency": "high",
        "signal_quality": "actionable",
    },

    # RETAIL-INFLUENTIAL - Move retail sentiment
    "Keith Gill": {
        "handle": "@TheRoaringKitty",
        "category": "retail_leader",
        "bias": "meme_bull",
        "specialty": ["GME", "value_memes", "retail_rallying"],
        "credibility": "cult_following",
        "post_frequency": "sporadic",  # Periods of silence then activity
        "signal_quality": "market_moving",
    },
    "Chamath Palihapitiya": {
        "handle": "@chamath",
        "category": "retail_leader",
        "bias": "vc_tech",
        "specialty": ["SPACs", "tech", "macro_takes"],
        "credibility": "declining",  # SPAC track record
        "post_frequency": "high",
        "signal_quality": "entertainment",
    },

    # FINANCE MEDIA / JOURNALISTS - News and analysis
    "Jim Cramer": {
        "handle": "@jimcramer",
        "category": "media",
        "bias": "momentum",
        "specialty": ["broad_market", "stock_picks", "hot_takes"],
        "credibility": "inverse_indicator",  # Famous for being contrarian signal
        "post_frequency": "very_high",
        "signal_quality": "contrarian_signal",
    },
    "Josh Brown": {
        "handle": "@ReformedBroker",
        "category": "media",
        "bias": "balanced_bull",
        "specialty": ["behavioral", "advisor_perspective", "market_commentary"],
        "credibility": "high",
        "post_frequency": "very_high",
        "signal_quality": "educational",
    },
    "Barry Ritholtz": {
        "handle": "@ritholtz",
        "category": "media",
        "bias": "balanced",
        "specialty": ["data_driven", "behavioral", "long_term"],
        "credibility": "high",
        "post_frequency": "high",
        "signal_quality": "educational",
    },
    "Joe Weisenthal": {
        "handle": "@TheStalwart",
        "category": "media",
        "bias": "curious",
        "specialty": ["macro", "markets", "economics", "oddities"],
        "credibility": "high",
        "post_frequency": "very_high",
        "signal_quality": "informational",
    },
    "Matt Levine": {
        "handle": "@matt_levine",
        "category": "media",
        "bias": "analytical",
        "specialty": ["finance_structure", "deals", "regulation", "crypto"],
        "credibility": "very_high",
        "post_frequency": "low",  # Mostly newsletter
        "signal_quality": "educational",
    },
    "Kyla Scanlon": {
        "handle": "@kaborost",
        "category": "media",
        "bias": "educational",
        "specialty": ["vibes", "fed", "economy", "explainers"],
        "credibility": "rising",
        "post_frequency": "high",
        "signal_quality": "educational",
    },

    # QUANT / DATA ANALYSTS
    "Tom Lee": {
        "handle": "@fundstrat",
        "category": "strategist",
        "bias": "perma_bull",
        "specialty": ["targets", "crypto", "macro_bull"],
        "credibility": "medium",
        "post_frequency": "medium",
        "signal_quality": "directional",
    },
    "Jesse Felder": {
        "handle": "@jessefelder",
        "category": "strategist",
        "bias": "value_bear",
        "specialty": ["valuation", "bubbles", "macro_bear"],
        "credibility": "good_track_record",
        "post_frequency": "medium",
        "signal_quality": "contrarian",
    },

    # SHORT SELLERS / BEARS
    "Hindenburg Research": {
        "handle": "@HindenburgRes",
        "category": "short_seller",
        "bias": "bear_activist",
        "specialty": ["fraud", "accounting", "short_reports"],
        "credibility": "high_impact",
        "post_frequency": "low",  # Report-driven
        "signal_quality": "market_moving",
    },
    "Citron Research": {
        "handle": "@CitronResearch",
        "category": "short_seller",
        "bias": "bear_activist",
        "specialty": ["shorts", "fraud", "overvaluation"],
        "credibility": "mixed",
        "post_frequency": "low",
        "signal_quality": "volatile",
    },

    # SEMICONDUCTOR / AI SPECIALISTS
    "Dylan Patel": {
        "handle": "@dylan522p",
        "category": "sector_specialist",
        "bias": "tech_neutral",
        "specialty": ["semiconductors", "AI_chips", "supply_chain"],
        "credibility": "very_high",
        "post_frequency": "high",
        "signal_quality": "deep_technical",
    },
}

# Backwards compatibility - list of all KOL names
STOCK_KOLS = list(KOL_PROFILES.keys())

# Extract handles for backwards compatibility
STOCK_HANDLES = {name: profile["handle"] for name, profile in KOL_PROFILES.items()}

# KOL categories for targeted searches
KOL_CATEGORIES = {
    "macro": ["Michael Burry", "Stan Druckenmiller", "Ray Dalio", "Mohamed El-Erian", "Howard Marks"],
    "activist": ["Bill Ackman", "David Einhorn", "Carl Icahn"],
    "tech_growth": ["Cathie Wood", "Dan Ives", "Gene Munster", "Beth Kindig"],
    "options_flow": ["Unusual Whales", "Cheddar Flow", "SpotGamma"],
    "retail_leader": ["Keith Gill", "Chamath Palihapitiya"],
    "media": ["Jim Cramer", "Josh Brown", "Barry Ritholtz", "Joe Weisenthal", "Matt Levine", "Kyla Scanlon"],
    "strategist": ["Tom Lee", "Jesse Felder"],
    "short_seller": ["Hindenburg Research", "Citron Research"],
    "sector_specialist": ["Dylan Patel"],
}

# High-signal KOLs by topic
HIGH_SIGNAL_KOLS = {
    "market_crash": ["Michael Burry", "Jesse Felder", "Howard Marks"],
    "fed_policy": ["Mohamed El-Erian", "Joe Weisenthal", "Kyla Scanlon"],
    "tech_stocks": ["Cathie Wood", "Dan Ives", "Beth Kindig", "Dylan Patel"],
    "options_flow": ["Unusual Whales", "SpotGamma", "Cheddar Flow"],
    "meme_stocks": ["Keith Gill", "Unusual Whales"],
    "short_reports": ["Hindenburg Research", "Citron Research"],
    "semiconductors": ["Dylan Patel", "Beth Kindig"],
    "institutional": ["Bill Ackman", "Stan Druckenmiller", "David Einhorn"],
}

# Topics relevant to stock/trading ecosystem
STOCK_TOPICS = [
    # Market Sentiment
    "bull market", "bear market", "correction", "rally", "selloff",
    "market crash", "all-time high", "buy the dip",
    # Options & Derivatives
    "calls", "puts", "gamma squeeze", "short squeeze", "VIX",
    "options flow", "unusual options activity",
    # Fundamentals
    "earnings", "EPS", "revenue", "guidance", "beat", "miss",
    "PE ratio", "valuation", "overvalued", "undervalued",
    # Macro
    "Fed", "interest rates", "inflation", "recession",
    "Powell", "FOMC", "rate cut", "rate hike",
    # Sectors
    "tech stocks", "AI stocks", "semiconductor", "biotech",
    "energy", "financials", "consumer",
    # Retail Trading
    "WSB", "WallStreetBets", "diamond hands", "YOLO",
    "meme stock", "retail traders", "hedge fund",
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GrokInsightResult:
    """Result from Grok KOL insights query."""
    topic: str
    insights: str
    kols_mentioned: List[str]
    sentiment: str  # bullish, bearish, mixed, neutral
    timestamp: float
    cached: bool = False

    def format_summary(self) -> str:
        """Format for display."""
        sentiment_emoji = {
            "bullish": "🟢",
            "bearish": "🔴",
            "mixed": "🟡",
            "neutral": "⚪",
        }.get(self.sentiment, "⚪")

        header = f"{sentiment_emoji} **X/Twitter Sentiment: {self.sentiment.upper()}**\n\n"
        return header + self.insights

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "topic": self.topic,
            "insights": self.insights,
            "kols_mentioned": self.kols_mentioned,
            "sentiment": self.sentiment,
            "timestamp": self.timestamp,
            "cached": self.cached,
        }


# =============================================================================
# GROK SERVICE
# =============================================================================

class GrokService:
    """
    Service for fetching real-time KOL insights from X via xAI's Grok API.
    Supports mock mode for testing without API key.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "default",
        max_retries: int = 6,
        timeout: int = 120,
        mock_mode: bool = False
    ):
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.base_url = "https://api.x.ai/v1"
        self.model = GROK_MODELS.get(model, GROK_MODELS["default"])
        self.max_retries = max_retries
        self.timeout = timeout
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = int(os.getenv("GROK_CACHE_TTL", "7200"))  # 1 hour cache
        self._cache_file = os.path.join(os.getcwd(), ".grok_cache_stock.json")

        # Mock mode for testing without API
        self.mock_mode = mock_mode or os.getenv("GROK_MOCK_MODE", "").lower() == "true"

        # Load cache from disk
        self._load_cache()

        if not self.api_key and not self.mock_mode:
            logger.warning("No XAI_API_KEY provided or found in environment.")

    def is_available(self) -> bool:
        """Check if service is available (API key set or mock mode)."""
        return bool(self.api_key) or self.mock_mode

    def _get_mock_response(self, topic: str, response_type: str = "general") -> str:
        """Generate mock response for testing without API."""
        mock_responses = {
            "institutional_flow": f"""## Key Findings
**Institutional Activity on {topic}:**

### Notable Moves
- **Michael Burry** (@michaeljburry): Increased position by 15% based on latest 13F filing
- **Bill Ackman** (@BillAckman): "We remain bullish on {topic} - strong fundamentals"
- **Cathie Wood** (@CathieDWood): ARK added shares across multiple funds

### Sentiment Snapshot
- Institutional: **Bullish** (65% buying vs 35% selling)
- Smart Money Flow: Net positive $2.3B in Q4
- Hedge Fund Positioning: Above average

## Trading Implications
- Watch for continuation if institutional buying persists
- Key support at 50-day MA

*[MOCK DATA - Set XAI_API_KEY for real-time X/Twitter data]*""",

            "options_sentiment": f"""## Options Flow Analysis for {topic}

### Unusual Activity Detected
- **SpotGamma** (@spotgamma): "GEX flipping positive - dealer hedging pressure easing"
- **Unusual Whales** (@unusual_whales): Large call sweeps detected at +5% OTM strikes
- **Cheddar Flow** (@CheddarFlow): Premium flow skewed 70/30 calls vs puts

### Key Levels
- Put Wall: $150
- Call Wall: $180
- Max Pain: $165

### Sentiment
- Options Market: **Bullish**
- Put/Call Ratio: 0.65 (below average = bullish)
- IV Rank: 45%

## Trading Implications
- Gamma squeeze potential above $175
- Support from dealer hedging at current levels

*[MOCK DATA - Set XAI_API_KEY for real-time X/Twitter data]*""",

            "synthesis": f"""# {topic} - Multi-Perspective KOL Analysis

## 🐂 Bull Case
**Who's Bullish:**
- **Cathie Wood** (@CathieDWood): "Innovation leader with 5-year CAGR potential of 25%"
- **Dan Ives** (@DivesTech): "$250 price target - AI tailwinds underappreciated"
- **Tom Lee** (@fundstrat): "Risk/reward attractive at current levels"

**Main Bull Arguments:**
1. Strong revenue growth trajectory
2. Market leadership in key segments
3. Multiple expansion potential

## 🐻 Bear Case
**Who's Bearish:**
- **Michael Burry** (@michaeljburry): Silent but reduced position per 13F
- **Jesse Felder** (@jessefelder): "Valuation stretched vs historical norms"

**Main Bear Arguments:**
1. Premium valuation leaves little margin for error
2. Competition intensifying
3. Macro headwinds (rates, growth)

## 📊 Sentiment Gauge
- Institutional: **Bullish**
- Retail: **Very Bullish**
- Options Market: **Bullish**
- Overall: **Bullish with caution**

## 🎯 Synthesis
**Consensus:** Cautiously bullish with debate on valuation
**Key Debate:** Growth sustainability vs multiple compression
**Contrarian View:** Bears argue mean reversion inevitable

*[MOCK DATA - Set XAI_API_KEY for real-time X/Twitter data]*""",

            "general": f"""## Market Pulse: {topic}

### Key Narratives
- **Josh Brown** (@ReformedBroker): "Market structure remains supportive"
- **Joe Weisenthal** (@TheStalwart): "Economic data continues to surprise"
- **Kyla Scanlon** (@kaborost): "Vibes improving but watch credit spreads"

### Sentiment Snapshot
- Overall: **Constructive**
- Key debates ongoing
- Watch for catalyst events

## Notable Voices
Multiple perspectives captured from finance Twitter.

*[MOCK DATA - Set XAI_API_KEY for real-time X/Twitter data]*"""
        }
        return mock_responses.get(response_type, mock_responses["general"])

    def _load_cache(self):
        """Load cache from disk."""
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file, 'r') as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached Grok queries.")
        except Exception as e:
            logger.warning(f"Failed to load Grok cache: {e}")
            self._cache = {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self._cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save Grok cache: {e}")

    def _get_cache_key(self, topic: str) -> str:
        """Generate cache key for a topic."""
        return hashlib.md5(topic.lower().encode()).hexdigest()

    def _get_cached(self, topic: str) -> Optional[str]:
        """Get cached result if still valid."""
        key = self._get_cache_key(topic)
        if key in self._cache:
            cached = self._cache[key]
            if time.time() - cached['timestamp'] < self._cache_ttl:
                logger.info(f"Cache hit for topic: {topic[:30]}...")
                return cached['result']
            else:
                logger.info(f"Cache expired for topic: {topic[:30]}...")
                del self._cache[key]
                self._save_cache()
        return None

    def _set_cache(self, topic: str, result: str):
        """Cache a result."""
        key = self._get_cache_key(topic)
        self._cache[key] = {
            'result': result,
            'timestamp': time.time()
        }
        self._save_cache()

    def _make_request(self, payload: Dict) -> Dict[str, Any]:
        """Make API request with retry logic."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    logger.warning(f"Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")

            except requests.exceptions.ConnectionError:
                last_error = "Connection failed"
                logger.warning(f"Connection error on attempt {attempt + 1}/{self.max_retries}")

            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:100]}"
                logger.error(f"HTTP error: {last_error}")
                # Don't retry on 4xx errors (except 429)
                if 400 <= e.response.status_code < 500:
                    break

            except Exception as e:
                last_error = str(e)
                logger.error(f"Unexpected error: {e}")

            # Exponential backoff
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)

        raise Exception(f"API request failed after {self.max_retries} attempts: {last_error}")

    def _validate_response(self, data: Dict) -> str:
        """Validate and extract content from API response."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict response, got {type(data)}")

        choices = data.get('choices')
        if not choices or not isinstance(choices, list):
            raise ValueError("Response missing 'choices' array")

        if len(choices) == 0:
            raise ValueError("Response 'choices' array is empty")

        message = choices[0].get('message')
        if not message or not isinstance(message, dict):
            raise ValueError("Response missing 'message' object")

        content = message.get('content')
        if not content or not isinstance(content, str):
            raise ValueError("Response missing 'content' string")

        return content.strip()

    def get_kol_insights(
        self,
        topic: str,
        focus_area: str = "stock_trading",
        include_kol_list: bool = True
    ) -> str:
        """
        Query Grok to get KOL insights from X about a specific stock/trading topic.

        Args:
            topic: The topic to search for (e.g., stock ticker, market event)
            focus_area: Focus area for context (stock_trading, macro, options, etc.)
            include_kol_list: Whether to specifically search for known KOLs

        Returns:
            Formatted KOL insights string
        """
        # Mock mode for testing
        if self.mock_mode:
            logger.info(f"Mock mode: Returning mock KOL insights for {topic}")
            return self._get_mock_response(topic, "general")

        if not self.api_key:
            return "No xAI API Key provided. Add XAI_API_KEY to enable X/Twitter sentiment."

        # Check cache first
        cache_key = f"{topic}:{focus_area}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Build KOL-specific search context with handles
        kol_context = ""
        if include_kol_list:
            kol_list = []
            for name in STOCK_KOLS[:15]:
                handle = STOCK_HANDLES.get(name, "")
                if handle:
                    kol_list.append(f"{name} ({handle})")
                else:
                    kol_list.append(name)
            kol_names = ", ".join(kol_list)
            kol_context = f"""
**PRIORITY KOLs to search** (check their recent posts):
{kol_names}

Also search for posts from: hedge fund managers, financial analysts, options traders, macro strategists, finance journalists, retail trading communities.
"""

        # Build topic-specific context
        topic_keywords = ", ".join(STOCK_TOPICS)

        # Enhanced prompt for stock/trading ecosystem
        prompt = f"""Search X (Twitter) comprehensively for recent high-engagement posts about: "{topic}"

{kol_context}

**SEARCH ACROSS THE STOCK/TRADING ECOSYSTEM:**
- Institutional activity and 13F filings
- Options flow and unusual activity
- Analyst ratings and price targets
- Earnings expectations and guidance
- Macro factors and Fed commentary
- Retail sentiment and social trends
- Short interest and squeeze potential

Related keywords and hashtags: {topic_keywords}, #stocks, #trading, #investing, #options, #earnings

**IMPORTANT**: Find at least 5-8 different voices with diverse perspectives (bulls vs bears, institutions vs retail, long-term vs traders).

Your task: Create a comprehensive "MARKET PULSE" briefing for trading decisions.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

## Key Narratives (What Traders/Analysts Are Saying)
- **[Narrative 1]**: [Name] (@handle) - "[Specific quote]" (date if known)
- **[Narrative 2]**: [Name] (@handle) - "[Specific quote]"
- **[Narrative 3]**: [Name] (@handle) - "[Specific quote]"
(Include at least 4-5 different voices)

## Hot Takes & Debates
- **[Debate 1]**: [Who's bullish vs bearish and why]
- **[Debate 2]**: [Key disagreements]
(Include bulls AND bears - who's buying? who's selling?)

## Sentiment Snapshot
- Overall tone: [Bullish/Bearish/Mixed/Cautious]
- Institutional stance: [Buying/Selling/Holding]
- Retail sentiment: [FOMO/Fear/Neutral]
- Key concerns: [List 2-3 main risks]
- Key catalysts: [What could move the stock?]

## Trading Implications
- [Short-term outlook]
- [Key levels to watch]
- [What contrarians are saying]

## Quotable Moments (for reference)
- "[Quote 1]" - @handle (Name)
- "[Quote 2]" - @handle (Name)
- "[Quote 3]" - @handle (Name)

Be thorough and specific. Use real names and handles. Include dates when possible. Show diverse perspectives.
"""

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": """You are Grok, xAI's AI with real-time access to X (Twitter) data.

Your specialty: Capturing the pulse of Finance Twitter - the traders, analysts, debates, and sentiment.

Rules:
1. Be SPECIFIC - names, handles, approximate dates
2. Be HONEST - if you can't find recent posts, say so
3. Be RELEVANT - focus on posts from the last 7 days when possible
4. Be BALANCED - show multiple perspectives, not just hype or doom"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": self.model,
            "stream": False,
            "temperature": 0.35,
            "max_tokens": 2500
        }

        try:
            data = self._make_request(payload)
            result = self._validate_response(data)

            # Cache the result
            self._set_cache(cache_key, result)

            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in get_kol_insights: {error_msg}")

            # Return helpful error message
            if "401" in error_msg:
                return "Grok API Error: Invalid API key. Please check your xAI API key."
            elif "429" in error_msg:
                return "Grok API Error: Rate limited. Please try again in a few minutes."
            elif "timeout" in error_msg.lower():
                return "Grok API Error: Request timed out. X might be slow - try again."
            else:
                return f"Grok API Error: {error_msg[:100]}"

    def get_stock_sentiment(self, symbol: str) -> str:
        """
        Get trading sentiment for a specific stock ticker.
        """
        if self.mock_mode:
            return self._get_mock_response(symbol, "general")

        if not self.api_key:
            return "No API key"

        prompt = f"""Search X for recent trader and investor commentary on ${symbol}.

Summarize:
1. Overall sentiment (bullish/bearish/mixed)
2. Key bull arguments
3. Key bear arguments
4. Notable price targets mentioned
5. Any unusual activity (options flow, short interest)

Focus on posts from known traders, analysts, and finance accounts."""

        payload = {
            "messages": [
                {"role": "system", "content": "You are Grok with real-time X access. Focus on stock trading voices."},
                {"role": "user", "content": prompt}
            ],
            "model": self.model,
            "temperature": 0.3,
            "max_tokens": 1200
        }

        try:
            data = self._make_request(payload)
            return self._validate_response(data)
        except Exception as e:
            return f"Error: {str(e)[:50]}"

    def search_and_summarize(self, query: str) -> str:
        """
        General search and summarize function for arbitrary queries.
        """
        if not self.api_key:
            return "No API key"

        prompt = f"""Search X and summarize findings for: {query}

Provide:
1. Key findings with sources
2. Expert opinions (with @handles)
3. Any controversies or debates
4. Confidence assessment

Be specific and cite sources."""

        payload = {
            "messages": [
                {"role": "system", "content": "You are Grok. Search X and summarize findings with sources."},
                {"role": "user", "content": prompt}
            ],
            "model": self.model,
            "temperature": 0.3,
            "max_tokens": 1800
        }

        try:
            data = self._make_request(payload)
            return self._validate_response(data)
        except Exception as e:
            return f"Error: {str(e)[:50]}"

    def competitive_intelligence_search(self, dimension: str, context: str = "") -> str:
        """
        Search X for stock market competitive intelligence on a specific dimension.

        Dimensions:
        - institutional_flow: What institutions are doing (13F, hedge fund moves)
        - options_sentiment: Options flow, gamma exposure, unusual activity
        - analyst_ratings: Upgrades, downgrades, price targets
        - earnings_catalyst: Earnings expectations, guidance, beat/miss history
        - macro_sentiment: Fed, rates, inflation, recession risk
        - retail_sentiment: WSB, retail traders, social trends
        - sector_rotation: Sector movements, cyclical vs defensive
        - short_interest: Short squeezes, borrow rates, bears

        Args:
            dimension: The CI dimension to search
            context: Optional additional context (e.g., specific tickers)

        Returns:
            Formatted competitive intelligence findings
        """
        # Mock mode for testing
        if self.mock_mode:
            logger.info(f"Mock mode: Returning mock CI for {dimension}")
            mock_type = dimension if dimension in ["institutional_flow", "options_sentiment"] else "general"
            return self._get_mock_response(context or dimension, mock_type)

        if not self.api_key:
            return ""

        # Check cache
        cache_key = f"ci:{dimension}:{context[:50]}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        dimension_prompts = {
            "institutional_flow": f"""Search X for INSTITUTIONAL ACTIVITY in stocks:

{f'Specific focus: {context}' if context else ''}

Look for:
1. **13F Filings**: What did Burry, Ackman, Druckenmiller buy/sell?
2. **Hedge Fund Moves**: Large position changes, new stakes, exits
3. **Smart Money**: What are institutions accumulating vs distributing?
4. **Whale Activity**: Large block trades, dark pool prints
5. **Insider Trading**: Notable insider buys or sells

Find SPECIFIC intel:
- Fund names and position sizes
- Entry/exit prices when mentioned
- @handles and dates

Search terms: 13F filing, hedge fund position, smart money, institutional buying, whale alert""",

            "options_sentiment": f"""Search X for OPTIONS FLOW and sentiment:

{f'Specific focus: {context}' if context else ''}

Look for:
1. **Unusual Options Activity**: Large premium, unusual strikes
2. **Gamma Exposure**: GEX levels, gamma squeeze potential
3. **Put/Call Ratios**: Extreme readings, sentiment shift
4. **Sweep Orders**: Aggressive buying patterns
5. **VIX Commentary**: Volatility expectations

Find SPECIFIC examples:
- Strike prices and expiration dates
- Premium amounts
- Bullish vs bearish flow
- @handles from options traders

Search terms: options flow, unusual activity, gamma squeeze, calls, puts, VIX""",

            "analyst_ratings": f"""Search X for ANALYST RATINGS and price targets:

{f'Specific focus: {context}' if context else ''}

Look for:
1. **Upgrades/Downgrades**: Recent rating changes
2. **Price Targets**: New targets, raised vs lowered
3. **Initiation Coverage**: New analyst coverage
4. **Research Notes**: Key findings from reports
5. **Analyst Debates**: Disagreements on valuations

Find SPECIFIC examples:
- Analyst names and firms
- Price targets with upside/downside
- Rating changes with rationale
- @handles of research accounts

Search terms: analyst upgrade, price target, rating change, Wall Street analyst""",

            "earnings_catalyst": f"""Search X for EARNINGS expectations and catalysts:

{f'Specific focus: {context}' if context else ''}

Look for:
1. **Earnings Whisper**: What traders expect vs consensus
2. **Guidance Focus**: What management will say
3. **Beat/Miss History**: Track record patterns
4. **Key Metrics**: What numbers matter most
5. **Post-Earnings Moves**: Historical reaction patterns

Find SPECIFIC intel:
- EPS/revenue expectations
- Key metrics to watch
- Historical patterns
- Trader positioning

Search terms: earnings, EPS, revenue, guidance, beat, miss, quarter""",

            "macro_sentiment": f"""Search X for MACRO sentiment and Fed commentary:

{f'Specific focus: {context}' if context else ''}

Look for:
1. **Fed Commentary**: Powell, FOMC, rate expectations
2. **Inflation Views**: Hot/cold CPI, PCE reactions
3. **Recession Odds**: Hard landing vs soft landing
4. **Rate Expectations**: Cuts vs hikes, timing
5. **Economic Data**: Jobs, GDP, housing

Find SPECIFIC insights:
- Rate probabilities mentioned
- Economic forecasts
- @handles from macro voices (El-Erian, Dalio, etc.)

Search terms: Fed, Powell, rate cut, inflation, recession, FOMC""",

            "retail_sentiment": f"""Search X for RETAIL SENTIMENT and social trends:

{f'Specific focus: {context}' if context else ''}

Look for:
1. **WSB Activity**: WallStreetBets top plays
2. **Meme Stocks**: Trending tickers in retail
3. **FOMO Indicators**: Retail chasing patterns
4. **Fear Gauges**: Panic selling signs
5. **Social Volume**: Trending tickers, mentions

Find SPECIFIC examples:
- Trending tickers
- Sentiment shift indicators
- Reddit/Twitter hype levels
- Contrarian signals

Search terms: WSB, WallStreetBets, retail traders, meme stock, diamond hands, YOLO""",

            "sector_rotation": f"""Search X for SECTOR ROTATION and trends:

{f'Specific focus: {context}' if context else ''}

Look for:
1. **Hot Sectors**: What's leading the market
2. **Cold Sectors**: What's lagging, underperforming
3. **Rotation Signals**: Money moving between sectors
4. **Cyclical vs Defensive**: Risk-on vs risk-off
5. **Theme Plays**: AI, energy, infrastructure

Find SPECIFIC examples:
- Sector ETF flows
- Relative performance data
- Rotation triggers
- @handles of sector analysts

Search terms: sector rotation, tech sector, energy stocks, defensive, cyclical, growth vs value""",

            "short_interest": f"""Search X for SHORT INTEREST and squeeze potential:

{f'Specific focus: {context}' if context else ''}

Look for:
1. **High Short Interest**: Most shorted stocks
2. **Squeeze Candidates**: Days to cover, utilization
3. **Short Reports**: Hindenburg, Muddy Waters, etc.
4. **Borrow Rates**: Hard to borrow signals
5. **Short Covering**: Squeeze in progress

Find SPECIFIC examples:
- Short interest percentages
- Days to cover
- Borrow rates
- Short report targets

Search terms: short interest, short squeeze, borrow rate, short report, Hindenburg"""
        }

        prompt = dimension_prompts.get(dimension, dimension_prompts["institutional_flow"])

        prompt += """

FORMAT YOUR RESPONSE:

## Key Findings
[Most important discoveries - be specific with names, numbers, dates]

## Notable Voices
[Who's saying what - include @handles and quotes]

## Trading Implications
[What this means for trading decisions]

Be SPECIFIC. Include @handles, dates, numbers, price levels. Distinguish fact from speculation."""

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": """You are Grok with real-time X access, conducting market intelligence research for traders.
Be specific and actionable. Include @handles, dates, numbers.
Distinguish between confirmed facts and speculation.
Focus on the last 7 days when possible."""
                },
                {"role": "user", "content": prompt}
            ],
            "model": self.model,
            "temperature": 0.25,
            "max_tokens": 2200
        }

        try:
            data = self._make_request(payload)
            result = self._validate_response(data)
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"CI search failed for {dimension}: {e}")
            return ""

    def get_kol_by_category(
        self,
        category: str,
        topic: str,
        max_kols: int = 5
    ) -> str:
        """
        Get insights from KOLs in a specific category about a topic.

        Categories: macro, activist, tech_growth, options_flow, retail_leader,
                   media, strategist, short_seller, sector_specialist

        Args:
            category: KOL category to search
            topic: Topic to search for (e.g., stock ticker, market event)
            max_kols: Maximum number of KOLs to search

        Returns:
            Formatted insights from category KOLs
        """
        if not self.api_key:
            return ""

        # Get KOLs in category
        kols_in_category = KOL_CATEGORIES.get(category, [])[:max_kols]
        if not kols_in_category:
            logger.warning(f"Unknown KOL category: {category}")
            return ""

        # Build KOL context with handles and specialties
        kol_details = []
        for name in kols_in_category:
            profile = KOL_PROFILES.get(name, {})
            handle = profile.get("handle", "")
            specialty = ", ".join(profile.get("specialty", [])[:3])
            bias = profile.get("bias", "unknown")
            kol_details.append(f"- **{name}** ({handle}) - Specialty: {specialty}, Bias: {bias}")

        kol_context = "\n".join(kol_details)

        # Check cache
        cache_key = f"cat:{category}:{topic[:30]}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        prompt = f"""Search X for what these specific {category.upper()} voices are saying about: "{topic}"

**PRIORITY KOLs TO SEARCH** (find their actual recent posts):
{kol_context}

**YOUR TASK:**
1. Find ACTUAL recent posts from these specific accounts about {topic}
2. Quote them directly with @handles
3. Note their stance (bullish/bearish/neutral)
4. Highlight any disagreements between them

**FORMAT:**

## {category.replace('_', ' ').title()} Perspectives on {topic}

### Individual Takes
For each KOL who has posted about this:
- **[Name]** (@handle): "[Direct quote or paraphrase]"
  - Stance: [Bullish/Bearish/Neutral]
  - Key Point: [Main argument]

### Consensus vs Disagreement
- **Agreement**: [What they agree on]
- **Disagreement**: [Where they differ]

### Signal Quality Assessment
- Most credible take: [Who and why]
- Contrarian view: [Who's going against the grain]

Be SPECIFIC. Use real quotes and dates. If a KOL hasn't posted about this topic recently, say so."""

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": f"""You are Grok with real-time X access, researching {category} voices in finance.
Your job is to find ACTUAL posts from the specific accounts listed.
Be honest if someone hasn't posted about the topic recently.
Include @handles and approximate dates for all quotes."""
                },
                {"role": "user", "content": prompt}
            ],
            "model": self.model,
            "temperature": 0.3,
            "max_tokens": 1800
        }

        try:
            data = self._make_request(payload)
            result = self._validate_response(data)
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Category search failed for {category}: {e}")
            return ""

    def synthesize_kol_views(
        self,
        symbol: str,
        include_categories: Optional[List[str]] = None
    ) -> str:
        """
        Synthesize views from multiple KOL categories into a comprehensive analysis.

        Args:
            symbol: Stock ticker symbol
            include_categories: Categories to include (default: all major ones)

        Returns:
            Synthesized multi-perspective analysis
        """
        # Mock mode for testing
        if self.mock_mode:
            logger.info(f"Mock mode: Returning mock synthesis for {symbol}")
            return self._get_mock_response(symbol, "synthesis")

        if not self.api_key:
            return ""

        # Default categories for comprehensive coverage
        if include_categories is None:
            include_categories = ["macro", "tech_growth", "options_flow", "media", "short_seller"]

        # Build comprehensive KOL list
        all_kols = []
        for cat in include_categories:
            kols = KOL_CATEGORIES.get(cat, [])[:3]  # Top 3 from each category
            for name in kols:
                profile = KOL_PROFILES.get(name, {})
                all_kols.append({
                    "name": name,
                    "handle": profile.get("handle", ""),
                    "category": cat,
                    "bias": profile.get("bias", "unknown"),
                    "credibility": profile.get("credibility", "unknown")
                })

        # Build KOL reference
        kol_reference = "\n".join([
            f"- {k['name']} ({k['handle']}) - {k['category']}, bias: {k['bias']}, credibility: {k['credibility']}"
            for k in all_kols
        ])

        # Check cache
        cache_key = f"synth:{symbol}:{','.join(include_categories)}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        prompt = f"""Conduct a COMPREHENSIVE analysis of ${symbol} by searching for posts from multiple finance perspectives on X.

**KOLs TO SEARCH** (organized by category):
{kol_reference}

**SYNTHESIS TASK:**
1. Search for recent posts about ${symbol} from each KOL
2. Identify the BULL case (who's bullish and why)
3. Identify the BEAR case (who's bearish and why)
4. Note unusual signals (options flow, insider activity)
5. Synthesize into actionable intelligence

**FORMAT YOUR RESPONSE:**

# ${symbol} - Multi-Perspective KOL Analysis

## 🐂 Bull Case
**Who's Bullish:**
- [Name] (@handle): "[Quote/stance]" - [Key argument]
- [Name] (@handle): "[Quote/stance]" - [Key argument]

**Main Bull Arguments:**
1. [Argument 1]
2. [Argument 2]

## 🐻 Bear Case
**Who's Bearish:**
- [Name] (@handle): "[Quote/stance]" - [Key argument]
- [Name] (@handle): "[Quote/stance]" - [Key argument]

**Main Bear Arguments:**
1. [Argument 1]
2. [Argument 2]

## 📊 Options & Flow Signals
[What options flow accounts are showing]

## 📰 Media Narrative
[What finance media is saying]

## ⚠️ Short Seller Activity
[Any short reports or bearish activism]

## 🎯 Synthesis & Signal
**Consensus View:** [Bullish/Bearish/Mixed]
**Conviction Level:** [High/Medium/Low]
**Key Debate:** [Main point of disagreement]
**Contrarian Opportunity:** [What contrarians might argue]

## 📌 Quotable Highlights
1. "[Best bull quote]" - @handle
2. "[Best bear quote]" - @handle
3. "[Most insightful take]" - @handle

**Data Quality Note:** [How recent/reliable are these takes]

Be thorough but honest. If a KOL hasn't mentioned ${symbol} recently, note that."""

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": """You are Grok conducting comprehensive market intelligence research.
Your job is to synthesize MULTIPLE perspectives into actionable intelligence.
Be specific with @handles, quotes, and dates.
Distinguish between high-credibility and low-credibility sources.
Be honest about what you can and cannot find."""
                },
                {"role": "user", "content": prompt}
            ],
            "model": self.model,
            "temperature": 0.3,
            "max_tokens": 2500
        }

        try:
            data = self._make_request(payload)
            result = self._validate_response(data)
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Synthesis failed for {symbol}: {e}")
            return ""

    def deep_stock_research(
        self,
        symbol: str,
        question: Optional[str] = None
    ) -> str:
        """
        Conduct deep research on a stock combining multiple intelligence sources.
        This is the most comprehensive research function.

        Args:
            symbol: Stock ticker symbol
            question: Optional specific question to focus on

        Returns:
            Comprehensive research report with KOL insights
        """
        # Mock mode for testing
        if self.mock_mode:
            logger.info(f"Mock mode: Returning mock deep research for {symbol}")
            return self._get_mock_response(symbol, "synthesis")

        if not self.api_key:
            return "Deep research requires XAI_API_KEY to access X/Twitter data."

        # Identify relevant high-signal KOLs based on question
        relevant_kols = []

        # Always include top macro voices
        for name in HIGH_SIGNAL_KOLS.get("institutional", []):
            relevant_kols.append(name)

        # Add topic-specific KOLs
        if question:
            q_lower = question.lower()
            if any(w in q_lower for w in ["tech", "ai", "software", "chip", "semiconductor"]):
                relevant_kols.extend(HIGH_SIGNAL_KOLS.get("tech_stocks", []))
            if any(w in q_lower for w in ["options", "calls", "puts", "flow"]):
                relevant_kols.extend(HIGH_SIGNAL_KOLS.get("options_flow", []))
            if any(w in q_lower for w in ["short", "bear", "fraud"]):
                relevant_kols.extend(HIGH_SIGNAL_KOLS.get("short_reports", []))
            if any(w in q_lower for w in ["fed", "rates", "inflation", "macro"]):
                relevant_kols.extend(HIGH_SIGNAL_KOLS.get("fed_policy", []))

        # Remove duplicates while preserving order
        seen = set()
        unique_kols = []
        for k in relevant_kols:
            if k not in seen:
                seen.add(k)
                unique_kols.append(k)

        # Build KOL profiles for the prompt
        kol_profiles_text = []
        for name in unique_kols[:12]:  # Limit to 12 KOLs
            profile = KOL_PROFILES.get(name, {})
            kol_profiles_text.append(
                f"- **{name}** ({profile.get('handle', '')}) - "
                f"Category: {profile.get('category', 'unknown')}, "
                f"Bias: {profile.get('bias', 'unknown')}, "
                f"Signal: {profile.get('signal_quality', 'unknown')}"
            )

        kol_context = "\n".join(kol_profiles_text)

        # Check cache
        cache_key = f"deep:{symbol}:{(question or '')[:30]}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        question_context = f'\n**SPECIFIC QUESTION TO ADDRESS:** "{question}"' if question else ""

        prompt = f"""Conduct DEEP RESEARCH on ${symbol} by searching X for insights from the most credible finance voices.
{question_context}

**HIGH-SIGNAL KOLs TO PRIORITIZE:**
{kol_context}

**RESEARCH OBJECTIVES:**
1. Find what the smartest money is saying about ${symbol}
2. Identify bulls vs bears with their specific arguments
3. Detect any unusual signals (positioning, flow, insider activity)
4. Synthesize into a clear investment thesis framework

**COMPREHENSIVE RESEARCH REPORT:**

# ${symbol} Deep Research Report

## Executive Summary
[2-3 sentence summary of the overall picture]

## Smart Money Positioning
**What institutions/hedge funds are doing:**
- [Specific moves with @handles and details]

## Bull Thesis (Who's Buying & Why)
**Strongest Bull Voices:**
1. **[Name]** (@handle): [Their bull case with quote]
2. **[Name]** (@handle): [Their bull case with quote]

**Key Catalysts Bulls See:**
- [Catalyst 1]
- [Catalyst 2]

## Bear Thesis (Who's Selling & Why)
**Strongest Bear Voices:**
1. **[Name]** (@handle): [Their bear case with quote]
2. **[Name]** (@handle): [Their bear case with quote]

**Key Risks Bears Highlight:**
- [Risk 1]
- [Risk 2]

## Technical & Flow Signals
**Options Flow:** [What flow accounts show]
**Gamma Levels:** [If mentioned]
**Unusual Activity:** [Any notable patterns]

## Sentiment Gauge
- Institutional: [Bullish/Bearish/Neutral]
- Retail: [Bullish/Bearish/Neutral]
- Options Market: [Bullish/Bearish/Neutral]
- Overall: [Bullish/Bearish/Mixed]

## Key Debates & Controversies
[What are people arguing about?]

## Actionable Takeaways
1. **If Bullish:** [What bulls should watch]
2. **If Bearish:** [What bears should watch]
3. **Key Levels:** [Important price levels mentioned]

## Source Quality Assessment
- **High Confidence:** [Well-sourced findings]
- **Medium Confidence:** [Reasonable but less sourced]
- **Speculation:** [Things to verify]

---
*Research based on X posts from the last 7 days*"""

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": """You are Grok, xAI's AI with real-time X access, conducting institutional-grade research.

Your standards:
1. SPECIFICITY - Names, @handles, dates, numbers
2. HONESTY - Say what you can't find
3. BALANCE - Show both sides fairly
4. CREDIBILITY - Weight sources by their track record
5. ACTIONABILITY - Make it useful for trading decisions

Prioritize high-signal accounts over noise. Distinguish fact from opinion."""
                },
                {"role": "user", "content": prompt}
            ],
            "model": self.model,
            "temperature": 0.25,  # Lower temp for research accuracy
            "max_tokens": 3000
        }

        try:
            data = self._make_request(payload)
            result = self._validate_response(data)
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Deep research failed for {symbol}: {e}")
            return f"Deep research failed: {str(e)[:50]}"


# =============================================================================
# STOCK CI DIMENSIONS - Keywords that indicate CI-relevant questions
# =============================================================================

STOCK_CI_DIMENSIONS = {
    "institutional_flow": ["institutional", "13f", "hedge fund", "smart money", "whale", "insider"],
    "options_sentiment": ["options", "calls", "puts", "gamma", "squeeze", "vix", "flow"],
    "analyst_ratings": ["analyst", "upgrade", "downgrade", "price target", "rating", "coverage"],
    "earnings_catalyst": ["earnings", "eps", "guidance", "beat", "miss", "quarter", "revenue"],
    "macro_sentiment": ["fed", "rates", "inflation", "recession", "powell", "fomc", "macro"],
    "retail_sentiment": ["reddit", "wsb", "retail", "meme", "stocktwits", "wallstreetbets", "fomo"],
    "sector_rotation": ["sector", "rotation", "cyclical", "defensive", "growth", "value"],
    "short_interest": ["short", "squeeze", "borrow", "utilization", "hindenburg", "citron"],
}


def detect_stock_ci_dimensions(question: str) -> List[str]:
    """
    Detect which CI dimensions are relevant based on question keywords.

    Args:
        question: The user's question

    Returns:
        List of relevant dimension names
    """
    q_lower = question.lower()
    dimensions = []

    for dimension, triggers in STOCK_CI_DIMENSIONS.items():
        if any(trigger in q_lower for trigger in triggers):
            dimensions.append(dimension)

    return dimensions


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_grok_service_instance = None


def get_grok_service(api_key: Optional[str] = None) -> GrokService:
    """
    Get singleton instance of GrokService.
    """
    global _grok_service_instance
    if _grok_service_instance is None:
        _grok_service_instance = GrokService(api_key=api_key)
    return _grok_service_instance


def get_stock_pulse(topic: str, api_key: Optional[str] = None) -> str:
    """
    Quick function to get trader/investor pulse for stock topics.
    """
    service = get_grok_service(api_key)
    return service.get_kol_insights(topic, focus_area="stock_trading")


def get_known_kols() -> List[str]:
    """Return list of known stock/finance KOLs."""
    return STOCK_KOLS.copy()


def get_kol_profiles() -> Dict[str, Dict]:
    """Return KOL profiles with metadata."""
    return KOL_PROFILES.copy()


def get_kol_categories() -> Dict[str, List[str]]:
    """Return KOL categories mapping."""
    return KOL_CATEGORIES.copy()


def get_high_signal_kols(topic: str) -> List[str]:
    """Get high-signal KOLs for a specific topic."""
    return HIGH_SIGNAL_KOLS.get(topic, []).copy()


def deep_research(symbol: str, question: str = None, api_key: str = None) -> str:
    """
    Convenience function for deep stock research with KOL synthesis.

    Args:
        symbol: Stock ticker (e.g., "NVDA")
        question: Optional specific question
        api_key: Optional API key

    Returns:
        Comprehensive research report
    """
    service = get_grok_service(api_key)
    return service.deep_stock_research(symbol, question)


def synthesize_views(symbol: str, categories: List[str] = None, api_key: str = None) -> str:
    """
    Convenience function for multi-category KOL synthesis.

    Args:
        symbol: Stock ticker
        categories: KOL categories to include
        api_key: Optional API key

    Returns:
        Synthesized multi-perspective analysis
    """
    service = get_grok_service(api_key)
    return service.synthesize_kol_views(symbol, categories)


def search_category_kols(category: str, topic: str, api_key: str = None) -> str:
    """
    Search KOLs in a specific category about a topic.

    Args:
        category: KOL category (macro, tech_growth, options_flow, etc.)
        topic: Topic to search
        api_key: Optional API key

    Returns:
        Insights from category KOLs
    """
    service = get_grok_service(api_key)
    return service.get_kol_by_category(category, topic)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Grok Service for Stock Advisor...")

    service = GrokService()

    if service.api_key:
        print(f"Model: {service.model}")
        print(f"Max retries: {service.max_retries}")
        print("\nFetching KOL insights for 'NVDA AI semiconductor'...")
        print("-" * 50)
        result = service.get_kol_insights("NVDA AI semiconductor")
        print(result)
    else:
        print("Skipping test: No XAI_API_KEY found")
        print("\nTo test, set: export XAI_API_KEY=your-key")
