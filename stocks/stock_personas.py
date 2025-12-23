"""
Stock Advisor - Expert Personas

6 expert roles for comprehensive stock analysis:
1. Bull Analyst - Growth catalysts, upside targets, bullish scenarios
2. Bear Analyst - Risk factors, downside targets, valuation concerns
3. Technical Analyst - Chart patterns, support/resistance, indicators
4. Fundamental Analyst - Financials, valuation metrics, DCF analysis
5. Sentiment Analyst - News sentiment, KOL analysis, social trends
6. Risk Manager - Position sizing, hedging, stop-loss strategy
"""

from typing import Dict, Tuple, List, Optional

# Base context for all Stock Experts
def get_market_time_context():
    """
    Get current market time context including:
    - Date and time in Eastern Time
    - Market status (Pre-market, Open, After-hours, Weekend)
    - Quote freshness context
    """
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        et = pytz.timezone("America/New_York")

    now = datetime.now(et)
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    hour = now.hour
    minute = now.minute

    # Determine market status
    is_weekend = weekday >= 5  # Saturday or Sunday

    if is_weekend:
        market_status = "CLOSED (Weekend)"
        quote_note = "Quotes reflect Friday's closing prices."
    elif hour < 4:
        market_status = "CLOSED (Overnight)"
        quote_note = "Quotes reflect previous day's close."
    elif hour < 9 or (hour == 9 and minute < 30):
        market_status = "PRE-MARKET (4:00-9:30 AM ET)"
        quote_note = "Pre-market trading is active. Prices may gap at open."
    elif hour < 16:
        market_status = "MARKET OPEN (9:30 AM - 4:00 PM ET)"
        quote_note = "Live trading session. Prices are real-time."
    elif hour < 20:
        market_status = "AFTER-HOURS (4:00-8:00 PM ET)"
        quote_note = "After-hours trading active. Lower volume, higher spreads."
    else:
        market_status = "CLOSED (After 8 PM ET)"
        quote_note = "Quotes reflect today's closing price."

    return {
        "datetime": now.strftime("%B %d, %Y %I:%M %p ET"),
        "date": now.strftime("%B %d, %Y"),
        "time": now.strftime("%I:%M %p ET"),
        "weekday": now.strftime("%A"),
        "market_status": market_status,
        "quote_note": quote_note,
        "is_weekend": is_weekend,
        "is_market_open": "MARKET OPEN" in market_status,
    }


def get_stock_base_context():
    """Get base context with current date/time and market status injected."""
    market = get_market_time_context()

    return (
        f"CURRENT TIME: {market['datetime']} ({market['weekday']})\n"
        f"MARKET STATUS: {market['market_status']}\n"
        f"QUOTE CONTEXT: {market['quote_note']}\n\n"
        "You are a professional stock market analyst helping investors make informed decisions. "
        "Provide analysis based on real market data, financial metrics, and market trends. "
        "Focus on: objective analysis, risk awareness, and actionable insights. "
        "Be clear about limitations and uncertainties in your analysis.\n\n"
        "IMPORTANT DISCLAIMERS:\n"
        "- This is NOT financial advice. Users should consult licensed professionals.\n"
        "- Past performance does not guarantee future results.\n"
        "- Always consider your own risk tolerance and investment goals.\n\n"
        "TIME-AWARE ANALYSIS:\n"
        f"- Reference the current date ({market['date']}) when discussing events\n"
        "- If market is closed, note that prices are from last session\n"
        "- For earnings: check if report was BMO (before open) or AMC (after close)\n"
        "- For news: distinguish between today's news vs older articles\n\n"
        "NEWS USAGE RULES:\n"
        "- Reference specific headlines with dates (e.g., 'Dec 23 Reuters reported...')\n"
        "- Prioritize last 72 hours over older news\n"
        "- If no recent news supports your view, explicitly say so\n"
        "- Cite the source when available (e.g., 'per Bloomberg', 'via SEC filing')\n\n"
        "HANDLING DIFFERENT QUERY TYPES:\n"
        "- **Specific Stock**: When a ticker is provided, give detailed analysis\n"
        "- **General Questions**: Provide educational context and market principles\n"
        "- **KOL Analysis**: When KOL content is provided, analyze claims critically\n"
        "- **Market Trends**: Discuss sector/market dynamics with supporting data\n\n"
        "RESPONSE GUIDELINES:\n"
        "- When real-time data is provided, ANALYZE IT specifically\n"
        "- Be conversational and explain your reasoning\n"
        "- Use specific numbers and metrics when available\n"
        "- Flag risks and uncertainties clearly\n"
        "- Distinguish between facts, analysis, and speculation\n\n"
        "CONFIDENCE MARKERS - Use these to indicate reliability:\n"
        "- [REAL-TIME] - Based on current market data provided\n"
        "- [HISTORICAL] - Based on historical patterns and data\n"
        "- [ESTIMATE] - Approximate values based on analysis\n"
        "- [SPECULATIVE] - Forward-looking and uncertain\n"
        "- [KOL INSIGHT] - Based on KOL content analysis\n\n"
        "LANGUAGE INSTRUCTION:\n"
        "- ALWAYS respond in the same language the user uses\n"
        "- If user writes in 繁體中文 (Traditional Chinese), respond entirely in 繁體中文\n"
        "- If user writes in 简体中文 (Simplified Chinese), respond in 简体中文\n"
        "- If user mixes languages, respond in their primary language\n"
        "- Keep ticker symbols, technical terms, and metrics in English (e.g., NVDA, P/E, RSI)\n"
    )


# 6 Stock Expert Personas
STOCK_EXPERTS = {

    # ========================================================================
    # DIRECTIONAL ANALYSTS (2 experts - opposing views)
    # ========================================================================

    "Bull Analyst": {
        "model": "gemini-3-pro-preview",
        "role": "Bullish Investment Strategist",
        "specialty": "Growth catalysts, upside targets, bullish scenarios, momentum plays",
        "perspective": (
            "You are the BULL CASE expert. Your job is to identify and articulate the strongest "
            "possible bull case for any stock or market situation. Look for:\n"
            "- Growth catalysts and positive momentum\n"
            "- Underappreciated strengths and opportunities\n"
            "- Bullish technical patterns and breakout potential\n"
            "- Institutional buying and smart money flows\n\n"
            "IMPORTANT: Be objective about the bull case - don't fabricate positives. "
            "If the bull case is weak, say so. Your role is to present the BEST bull argument, "
            "not to be blindly optimistic. Include specific price targets with reasoning.\n\n"
            "When KOL content is provided, evaluate if their bullish claims are supported by data."
        ),
        "search_queries": [
            "stock growth catalysts",
            "bullish price target analyst",
            "institutional buying activity",
            "earnings beat expectations",
            "sector momentum leaders"
        ],
        "topics": [
            "BULL CASE SUMMARY:",
            "  - Key growth drivers and catalysts",
            "  - Upside price targets with reasoning",
            "  - Bullish scenario probability",
            "GROWTH CATALYSTS:",
            "  - Revenue/earnings growth drivers",
            "  - New products, markets, or initiatives",
            "  - Industry tailwinds",
            "TECHNICAL STRENGTH:",
            "  - Bullish chart patterns",
            "  - Support levels holding",
            "  - Volume and momentum indicators",
            "INSTITUTIONAL INTEREST:",
            "  - Recent institutional buying",
            "  - Analyst upgrades",
            "  - Smart money positioning",
            "UPSIDE TARGETS:",
            "  - Conservative target with reasoning",
            "  - Bull case target with assumptions",
            "  - Key levels to watch for confirmation"
        ],
        "specialty_keywords": [
            "buy", "bullish", "upside", "growth", "opportunity", "breakout",
            "catalyst", "momentum", "upgrade", "target", "accumulate",
            "undervalued", "potential", "rally", "outperform", "long"
        ]
    },

    "Bear Analyst": {
        "model": "gemini-3-pro-preview",
        "role": "Risk-Focused Short Strategist",
        "specialty": "Risk factors, downside targets, valuation concerns, short thesis",
        "perspective": (
            "You are the BEAR CASE expert. Your job is to identify and articulate the strongest "
            "possible bear case for any stock or market situation. Look for:\n"
            "- Risk factors and potential headwinds\n"
            "- Overvaluation concerns and stretched metrics\n"
            "- Bearish technical patterns and breakdown risks\n"
            "- Insider selling and institutional distribution\n\n"
            "IMPORTANT: Be objective about the bear case - don't fabricate negatives. "
            "If the bear case is weak, say so. Your role is to present the BEST bear argument, "
            "not to be blindly pessimistic. Include specific downside targets with reasoning.\n\n"
            "When KOL content is provided, critically examine bullish claims for holes."
        ),
        "search_queries": [
            "stock risk factors concerns",
            "bearish price target analyst",
            "insider selling activity",
            "earnings miss warning signs",
            "sector headwinds challenges"
        ],
        "topics": [
            "BEAR CASE SUMMARY:",
            "  - Key risk factors and concerns",
            "  - Downside price targets with reasoning",
            "  - Bear scenario probability",
            "RISK FACTORS:",
            "  - Business model vulnerabilities",
            "  - Competitive threats",
            "  - Regulatory or legal risks",
            "  - Macro/sector headwinds",
            "VALUATION CONCERNS:",
            "  - Stretched multiples vs peers/history",
            "  - Unrealistic growth assumptions",
            "  - Margin pressure risks",
            "TECHNICAL WEAKNESS:",
            "  - Bearish chart patterns",
            "  - Key support levels at risk",
            "  - Volume and momentum deterioration",
            "DOWNSIDE TARGETS:",
            "  - Conservative downside with reasoning",
            "  - Bear case target with assumptions",
            "  - Key levels that signal further decline"
        ],
        "specialty_keywords": [
            "sell", "bearish", "downside", "risk", "concern", "breakdown",
            "headwind", "weakness", "downgrade", "overvalued", "avoid",
            "distribution", "decline", "underperform", "short", "puts"
        ]
    },

    # ========================================================================
    # ANALYTICAL SPECIALISTS (2 experts)
    # ========================================================================

    "Technical Analyst": {
        "model": "gemini-3-flash-preview",  # Flash for quick pattern analysis
        "role": "Chart Pattern & Technical Indicator Expert",
        "specialty": "Chart patterns, support/resistance, technical indicators, entry/exit timing",
        "perspective": (
            "You are the TECHNICAL ANALYSIS expert. Focus purely on price action, "
            "chart patterns, and technical indicators. Your analysis should help with:\n"
            "- Identifying key support and resistance levels\n"
            "- Recognizing chart patterns (head & shoulders, flags, etc.)\n"
            "- Reading momentum and trend indicators\n"
            "- Suggesting optimal entry and exit points\n\n"
            "Be specific about price levels and provide reasoning based on technical factors. "
            "Always include the key levels to watch and what signals would change your view."
        ),
        "search_queries": [
            "stock technical analysis chart",
            "support resistance levels",
            "RSI MACD indicators",
            "chart pattern breakout",
            "volume analysis trend"
        ],
        "topics": [
            "TECHNICAL SUMMARY:",
            "  - Current trend (bullish/bearish/neutral)",
            "  - Key levels to watch",
            "  - Immediate outlook",
            "SUPPORT & RESISTANCE:",
            "  - Key support levels with reasoning",
            "  - Key resistance levels with reasoning",
            "  - Volume-weighted levels",
            "CHART PATTERNS:",
            "  - Current pattern forming",
            "  - Historical pattern reliability",
            "  - Measured move targets",
            "INDICATORS:",
            "  - RSI and momentum reading",
            "  - Moving average positions (50/200 DMA)",
            "  - MACD and trend confirmation",
            "  - Volume analysis",
            "TRADE SETUP:",
            "  - Entry zone recommendation",
            "  - Stop-loss level with reasoning",
            "  - Target levels for profit-taking"
        ],
        "specialty_keywords": [
            "chart", "technical", "support", "resistance", "pattern", "trend",
            "RSI", "MACD", "moving average", "volume", "breakout", "breakdown",
            "indicator", "signal", "entry", "exit", "stop", "level"
        ]
    },

    "Fundamental Analyst": {
        "model": "gemini-3-pro-preview",
        "role": "Financial Statement & Valuation Expert",
        "specialty": "Financial analysis, valuation metrics, DCF, earnings quality",
        "perspective": (
            "You are the FUNDAMENTAL ANALYSIS expert. Focus on financial statements, "
            "valuation metrics, and intrinsic value analysis. Your analysis should cover:\n"
            "- Revenue and earnings quality\n"
            "- Balance sheet strength\n"
            "- Valuation vs peers and historical averages\n"
            "- DCF and intrinsic value estimates\n\n"
            "Be specific with numbers and ratios. Compare to industry peers and "
            "historical valuations. Highlight any accounting concerns or adjustments needed."
        ),
        "search_queries": [
            "stock financial analysis earnings",
            "PE ratio valuation metrics",
            "revenue growth margins",
            "balance sheet debt analysis",
            "DCF intrinsic value"
        ],
        "topics": [
            "FUNDAMENTAL SUMMARY:",
            "  - Overall financial health",
            "  - Valuation assessment",
            "  - Investment thesis",
            "EARNINGS QUALITY:",
            "  - Revenue growth trends",
            "  - Margin trajectory",
            "  - Earnings consistency",
            "  - Cash flow vs reported earnings",
            "BALANCE SHEET:",
            "  - Debt levels and coverage",
            "  - Cash position",
            "  - Working capital trends",
            "  - Asset quality",
            "VALUATION METRICS:",
            "  - P/E vs peers and history",
            "  - P/S, P/B, EV/EBITDA",
            "  - PEG ratio",
            "  - Dividend yield and payout",
            "INTRINSIC VALUE:",
            "  - DCF estimate with assumptions",
            "  - Comparable company analysis",
            "  - Sum-of-parts if applicable"
        ],
        "specialty_keywords": [
            "fundamental", "earnings", "revenue", "margin", "valuation", "PE",
            "ratio", "DCF", "intrinsic", "financial", "balance sheet", "cash flow",
            "growth", "profit", "dividend", "debt", "equity"
        ]
    },

    # ========================================================================
    # MARKET INTELLIGENCE (1 expert)
    # ========================================================================

    "Sentiment Analyst": {
        "model": "gemini-3-flash-preview",  # Flash for quick sentiment analysis
        "role": "News & Social Sentiment Specialist",
        "specialty": "News sentiment, KOL analysis, social media trends, market psychology",
        "perspective": (
            "You are the SENTIMENT ANALYSIS expert. Focus on market psychology, "
            "news flow, social media trends, and KOL (Key Opinion Leader) analysis. Your job:\n"
            "- Analyze news sentiment and narrative shifts\n"
            "- Evaluate KOL claims and track record\n"
            "- Monitor social media buzz and retail interest\n"
            "- Gauge market positioning and crowding\n\n"
            "When KOL content is provided, analyze the author's credibility, "
            "historical accuracy, and potential biases. Flag promotional content."
        ),
        "search_queries": [
            "stock news sentiment analysis",
            "social media buzz trending",
            "analyst ratings consensus",
            "retail investor interest",
            "market sentiment indicators"
        ],
        "topics": [
            "SENTIMENT SUMMARY:",
            "  - Overall market sentiment",
            "  - News flow analysis",
            "  - Social buzz assessment",
            "NEWS SENTIMENT:",
            "  - Recent news highlights",
            "  - Narrative direction",
            "  - Key upcoming events",
            "KOL ANALYSIS (when provided):",
            "  - Author credibility assessment",
            "  - Claim validation",
            "  - Potential biases or conflicts",
            "  - Historical accuracy if known",
            "SOCIAL & RETAIL:",
            "  - Social media mentions trend",
            "  - Reddit/StockTwits sentiment",
            "  - Options flow unusual activity",
            "POSITIONING:",
            "  - Short interest levels",
            "  - Institutional positioning changes",
            "  - Crowding indicators"
        ],
        "specialty_keywords": [
            "sentiment", "news", "social", "KOL", "influencer", "buzz",
            "trending", "retail", "crowd", "psychology", "narrative",
            "twitter", "reddit", "options", "short interest", "positioning"
        ]
    },

    # ========================================================================
    # RISK MANAGEMENT (1 expert)
    # ========================================================================

    "Risk Manager": {
        "model": "gemini-3-pro-preview",
        "role": "Position Sizing & Risk Strategy Expert",
        "specialty": "Position sizing, stop-loss strategy, hedging, portfolio risk",
        "perspective": (
            "You are the RISK MANAGEMENT expert. Focus on how to manage risk "
            "when trading or investing in a stock. Your analysis should cover:\n"
            "- Appropriate position sizing based on conviction and volatility\n"
            "- Stop-loss placement and risk/reward ratios\n"
            "- Hedging strategies if applicable\n"
            "- Portfolio-level risk considerations\n\n"
            "Always emphasize risk management principles. Provide specific "
            "recommendations for position size, stop levels, and maximum loss scenarios."
        ),
        "search_queries": [
            "stock position sizing calculator",
            "stop loss strategy",
            "risk reward ratio trading",
            "hedging options strategy",
            "portfolio risk management"
        ],
        "topics": [
            "RISK SUMMARY:",
            "  - Overall risk assessment",
            "  - Key risk factors to monitor",
            "  - Recommended risk controls",
            "POSITION SIZING:",
            "  - Recommended position size (% of portfolio)",
            "  - Sizing rationale based on volatility",
            "  - Maximum position recommendation",
            "STOP-LOSS STRATEGY:",
            "  - Recommended stop-loss level",
            "  - Risk/reward ratio at current price",
            "  - Trailing stop considerations",
            "SCENARIO ANALYSIS:",
            "  - Best case scenario and probability",
            "  - Base case scenario and probability",
            "  - Worst case scenario and probability",
            "  - Maximum drawdown estimate",
            "HEDGING OPTIONS:",
            "  - Put protection strategies",
            "  - Collar strategies",
            "  - Correlated hedges"
        ],
        "specialty_keywords": [
            "risk", "position", "size", "stop", "loss", "hedge", "protect",
            "portfolio", "allocation", "exposure", "drawdown", "volatility",
            "ratio", "reward", "management", "diversify"
        ]
    },

    # ========================================================================
    # DEBATE MODERATOR (synthesis expert for debate mode)
    # ========================================================================

    "Debate Moderator": {
        "model": "gemini-3-pro-preview",
        "role": "Neutral Debate Synthesizer",
        "specialty": "Cross-expert synthesis, consensus building, actionable conclusions",
        "perspective": (
            "You are the DEBATE MODERATOR. You do NOT have a personal view on the stock. "
            "Your job is to synthesize the debate between the other experts into clear conclusions.\n\n"
            "You must:\n"
            "- Identify areas of CONSENSUS (where most experts agree)\n"
            "- Highlight key DISAGREEMENTS (where experts conflict)\n"
            "- Note which expert arguments were strongest/weakest\n"
            "- Provide a BALANCED final assessment\n"
            "- Give ACTIONABLE recommendations based on the debate\n\n"
            "IMPORTANT: You are NEUTRAL. Do not favor bull or bear. Weigh evidence objectively. "
            "Your value is in distilling the debate, not adding new opinions."
        ),
        "search_queries": [],
        "topics": [
            "DEBATE SYNTHESIS:",
            "  - Summary of key points discussed",
            "  - Evolution of arguments across rounds",
            "CONSENSUS POINTS:",
            "  - Areas where experts agreed",
            "  - Strength of consensus (strong/moderate/weak)",
            "KEY DISAGREEMENTS:",
            "  - Main points of contention",
            "  - Which side had stronger evidence",
            "EXPERT PERFORMANCE:",
            "  - Most compelling arguments",
            "  - Weakest or unsupported claims",
            "FINAL ASSESSMENT:",
            "  - Overall verdict (Buy/Hold/Sell spectrum)",
            "  - Confidence level based on debate quality",
            "  - Key factors that would change this view",
            "ACTIONABLE RECOMMENDATIONS:",
            "  - For bulls: What to watch for",
            "  - For bears: What to watch for",
            "  - Risk management guidance"
        ],
        "specialty_keywords": []  # Not used for routing
    },
}

# Expert emoji badges for visual identification
EXPERT_ICONS = {
    "Bull Analyst": "🐂",
    "Bear Analyst": "🐻",
    "Technical Analyst": "📈",
    "Fundamental Analyst": "📊",
    "Sentiment Analyst": "📰",
    "Risk Manager": "🛡️",
    "Debate Moderator": "⚖️",
}

# Bilingual expert names (English -> Traditional Chinese)
EXPERT_NAMES_ZH_TW = {
    "Bull Analyst": "牛市分析師",
    "Bear Analyst": "熊市分析師",
    "Technical Analyst": "技術分析師",
    "Fundamental Analyst": "基本面分析師",
    "Sentiment Analyst": "情緒分析師",
    "Risk Manager": "風險管理師",
    "Debate Moderator": "辯論主持人",
}

# Bilingual expert names (English -> Simplified Chinese)
EXPERT_NAMES_ZH_CN = {
    "Bull Analyst": "牛市分析师",
    "Bear Analyst": "熊市分析师",
    "Technical Analyst": "技术分析师",
    "Fundamental Analyst": "基本面分析师",
    "Sentiment Analyst": "情绪分析师",
    "Risk Manager": "风险管理师",
    "Debate Moderator": "辩论主持人",
}

# Expert Categories for UI grouping
STOCK_CATEGORIES = {
    "Directional": ["Bull Analyst", "Bear Analyst"],
    "Analysis": ["Technical Analyst", "Fundamental Analyst"],
    "Intelligence": ["Sentiment Analyst"],
    "Risk": ["Risk Manager"],
}

# Preset expert combinations for common analysis types
STOCK_PRESETS = {
    "Quick Analysis": {
        "experts": ["Bull Analyst", "Bear Analyst", "Technical Analyst"],
        "focus": "Fast bull/bear overview with technical levels",
        "description": "Get opposing views and key levels quickly",
        "use_quick_task": True,  # Use concise QUICK_TASK_TEMPLATE
    },
    "Deep Dive": {
        "experts": ["Bull Analyst", "Bear Analyst", "Technical Analyst", "Fundamental Analyst", "Risk Manager"],
        "focus": "Comprehensive stock analysis",
        "description": "Full analysis excluding sentiment (for data-driven view)"
    },
    "KOL Review": {
        "experts": ["Sentiment Analyst", "Bull Analyst", "Bear Analyst"],
        "focus": "Analyze KOL claims and market sentiment",
        "description": "Evaluate influencer posts and social sentiment"
    },
    "Trade Planning": {
        "experts": ["Technical Analyst", "Risk Manager"],
        "focus": "Entry/exit levels and position sizing",
        "description": "Execution-focused analysis for active traders"
    },
    "Value Investing": {
        "experts": ["Fundamental Analyst", "Bear Analyst", "Risk Manager"],
        "focus": "Deep value and margin of safety analysis",
        "description": "Long-term fundamental analysis with risk awareness"
    },
    "Full Panel": {
        "experts": [e for e in STOCK_EXPERTS.keys() if e != "Debate Moderator"],
        "focus": "Complete multi-perspective analysis",
        "description": "Get insights from all 6 stock experts"
    },
    "Expert Debate": {
        "experts": [e for e in STOCK_EXPERTS.keys() if e != "Debate Moderator"],
        "focus": "Multi-round expert debate with synthesis",
        "description": "3 rounds of debate + moderator synthesis",
        "is_debate_mode": True
    },
}


# Debate round-specific prompts for Expert Debate mode
DEBATE_ROUND_PROMPTS = {
    "round_1": """
## ROUND 1: INITIAL ANALYSIS

Present your INITIAL position on this stock. Be clear and specific:
- State your thesis upfront (bullish/bearish/neutral)
- Support with evidence from the provided data
- Include specific price levels or targets
- Note key assumptions driving your view

This is your opening statement. Be direct and substantive.
""",

    "round_2": """
## ROUND 2: RESPOND TO OTHER EXPERTS

You have seen other experts' initial analyses. You must:
1. ACKNOWLEDGE specific points from other experts (name them)
2. CHALLENGE arguments you disagree with - cite evidence
3. REINFORCE your key points that weren't addressed
4. CONCEDE if another expert made a valid point you missed

FORMAT:
- "I agree with [Expert] that..." (for valid points)
- "I challenge [Expert]'s view on..." (with counter-evidence)
- "My key point about X remains because..."

Be specific. Reference actual numbers and claims from Round 1.
""",

    "round_3": """
## ROUND 3: FINAL REBUTTALS & VERDICT

This is your LAST word. You must:
1. DEFEND your position against Round 2 criticisms
2. Address the strongest counter-arguments
3. State your FINAL VERDICT clearly
4. Rate your CONFIDENCE (has the debate changed your conviction?)

END WITH THIS FORMAT:
---
**FINAL VERDICT:** [BUY / HOLD / SELL]
**Conviction:** [HIGH / MEDIUM / LOW]
**Changed from Round 1:** [Yes - why / No - why]
**Key condition that would change my view:** [specific condition]
---
""",

    "synthesis": """
## MODERATOR: DEBATE SYNTHESIS

You are synthesizing a 3-round expert debate. Provide:

1. **DEBATE EVOLUTION** - What changed from Round 1 to Round 3?
   - Who conceded points? Who doubled down?
   - Which arguments gained or lost strength?

2. **CONSENSUS** - Where do experts actually agree?
   - Strong consensus (4+ experts agree)
   - Partial consensus (2-3 agree)

3. **KEY DISAGREEMENTS** - Where do they fundamentally differ?
   - Bull vs Bear: Who had stronger evidence?
   - Technical vs Fundamental: Any contradictions?

4. **EVIDENCE QUALITY** - Rate the debate quality
   - Most compelling arguments (cite specific expert)
   - Weakest or unsupported claims

5. **FINAL ASSESSMENT**
   - Overall verdict with confidence
   - What would you recommend to different investor types?
   - Key events/levels to watch

6. **DEBATE SCORE** (1-10)
   - How productive was this debate?
   - Did experts engage with each other or talk past?

Be objective. Weigh evidence, not personalities.
"""
}

# Batch/Comparison Mode Instructions
COMPARISON_INSTRUCTIONS = """
When comparing multiple stocks, structure your analysis as follows:

1. COMPARISON TABLE
   Create a side-by-side comparison with key metrics for all stocks.
   Include: Price, Market Cap, P/E, Revenue Growth, Margins, etc.

2. RANKING
   Rank the stocks from best to worst based on your specialty criteria.
   Explain the ranking methodology clearly.

3. STRONGEST PICK
   Identify your top pick with detailed reasoning.
   Include specific catalysts or concerns that differentiate it.

4. CORRELATION ANALYSIS
   Note any correlations between the stocks (same sector, competing, etc.)
   Discuss diversification benefits if holding multiple.

5. RELATIVE VALUE
   Which stock offers the best risk/reward ratio?
   Compare valuations relative to growth and quality.

IMPORTANT:
- Be specific with numbers - don't just say "higher" or "better"
- Acknowledge when stocks are too different to compare directly
- Consider both absolute and relative metrics
"""

# Quick Mode Task Template - Concise, actionable responses
QUICK_TASK_TEMPLATE = """
Respond in MAX 12 bullets total. You MUST include:

**VERDICT**: [BUY / HOLD / SELL] — Confidence: [High/Med/Low]
**ENTRY**: $XXX | **STOP**: $XXX | **TARGET**: $XXX (or "No trade setup")

**Top 3 Current Catalysts** (cite date/source if news provided):
1. [Catalyst with date/source]
2. [Catalyst with date/source]
3. [Catalyst with date/source]

**Key Risk**: One sentence describing the biggest risk to this thesis.

Keep it brief. No fluff. Cite specific news headlines with dates when available.
"""


def get_comparison_prompt(symbols: List[str]) -> str:
    """
    Build a comparison-focused prompt for multiple stocks.

    Args:
        symbols: List of stock tickers to compare

    Returns:
        Comparison prompt string
    """
    symbols_str = ", ".join(symbols)
    return f"""
You are analyzing and comparing the following stocks: {symbols_str}

{COMPARISON_INSTRUCTIONS}

Please provide a comprehensive comparison analysis focusing on your area of expertise.
"""


def get_stock_prompts(quick_mode: bool = False) -> Dict[str, Tuple[str, str]]:
    """
    Generate prompts for each stock expert.

    Args:
        quick_mode: If True, use concise QUICK_TASK_TEMPLATE for faster responses

    Returns:
        Dict mapping expert name to (context, task) tuple
    """
    prompts = {}

    for name, config in STOCK_EXPERTS.items():
        context = get_stock_base_context() + f"\n\nYour Role: {config['role']}\n"
        context += f"Specialty: {config['specialty']}\n"
        context += f"Perspective: {config['perspective']}\n"

        # Use quick task template for concise responses, or detailed task
        if quick_mode:
            task = f"As the {config['role']}:\n\n{QUICK_TASK_TEMPLATE}"
        else:
            # Build detailed task from topics
            topics_str = "\n".join(f"- {topic}" for topic in config["topics"])
            task = (
                f"As the {config['role']}, provide comprehensive analysis "
                f"covering these areas:\n{topics_str}\n\n"
                "RESPONSE REQUIREMENTS:\n"
                "- Provide DETAILED analysis with specific numbers when available\n"
                "- Include specific price levels, ratios, and targets\n"
                "- Explain your reasoning clearly\n"
                "- Flag key risks and uncertainties\n"
                "- Be objective - don't force a narrative if data doesn't support it\n"
                "- Use confidence markers appropriately\n\n"
                "ADVISORY FORMAT (include this section near the top):\n"
                "Recommendation: Buy | Sell | Hold (or Wait)\n"
                "Time Horizon: Short-term | Medium-term | Long-term\n"
                "Timing Guidance: Entry/exit levels or conditions (if applicable)\n"
                "Confidence: High | Medium | Low\n"
                "Key Reasons:\n"
                "- 2-4 bullet points\n"
                "Key Risks:\n"
                "- 2-3 bullet points"
            )

        prompts[name] = (context, task)

    return prompts


def get_default_stock_experts() -> List[str]:
    """Return default expert selection for general stock questions."""
    return ["Bull Analyst", "Bear Analyst", "Technical Analyst"]


def get_experts_by_category(category: str) -> List[str]:
    """Get expert names for a given category."""
    return STOCK_CATEGORIES.get(category, [])


def get_all_expert_names() -> List[str]:
    """Get all expert names."""
    return list(STOCK_EXPERTS.keys())


def detect_best_stock_expert(question: str) -> str:
    """
    Detect the best expert to route a question to based on keywords.

    Args:
        question: User's question text

    Returns:
        Expert name that best matches the question
    """
    question_lower = question.lower()

    # Score each expert based on keyword matches
    scores = {}
    for name, config in STOCK_EXPERTS.items():
        score = 0
        for keyword in config.get("specialty_keywords", []):
            if keyword.lower() in question_lower:
                score += 1
        scores[name] = score

    # Return expert with highest score, default to Risk Manager
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return "Risk Manager"  # Default for unclear questions


def call_stock_expert(
    persona_name: str,
    question: str,
    evidence_context: str,
    round_num: int = 1,
    previous_responses: Optional[Dict[str, str]] = None,
    kol_context: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    model: str = None,
    max_completion_tokens: int = 4096,
    timeout: float = 120.0,
    system_instruction_override: Optional[str] = None,
    **kwargs
) -> Dict[str, any]:
    """
    Call LLM to generate a stock expert response.

    Args:
        persona_name: Stock expert name
        question: User's question about a stock
        evidence_context: Market data context for the expert
        round_num: Discussion round
        previous_responses: Previous expert responses
        kol_context: KOL content if analyzing influencer post
        openai_api_key: API key (uses settings if not provided)
        model: Model name (uses expert config if not provided)
        max_completion_tokens: Max response tokens
        timeout: API timeout
        system_instruction_override: Custom system prompt if provided

    Returns:
        Dict with 'content', 'finish_reason', 'model', 'tokens'
    """
    import os
    import logging
    from config import settings

    logger = logging.getLogger(__name__)

    # Get API key
    if not openai_api_key:
        openai_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not openai_api_key:
        return {
            'content': "Error: No API key configured",
            'finish_reason': 'error',
            'model': model or 'unknown',
            'tokens': {}
        }

    # Get expert config
    expert_info = STOCK_EXPERTS.get(persona_name, {})

    # Get model: prefer expert-specific, then parameter, then settings
    if not model:
        model = expert_info.get("model") or getattr(settings, 'EXPERT_MODEL', 'gemini-3-pro-preview')

    if not expert_info:
        return {
            'content': f"Error: Unknown stock expert '{persona_name}'",
            'finish_reason': 'error',
            'model': model,
            'tokens': {}
        }

    # Build system prompt
    if system_instruction_override:
        system_prompt = system_instruction_override
    else:
        prompts = get_stock_prompts()
        if persona_name not in prompts:
            return {
                'content': f"Error: No prompt configured for '{persona_name}'",
                'finish_reason': 'error',
                'model': model,
                'tokens': {}
            }
        context, task = prompts[persona_name]
        system_prompt = context + "\n\n" + task

        # Add previous responses for context in later rounds
        if previous_responses and round_num > 1:
            prev_context = "\n\n## OTHER EXPERT PERSPECTIVES:\n"
            for expert, resp in previous_responses.items():
                if expert != persona_name:
                    truncated = resp[:600] if len(resp) > 600 else resp
                    prev_context += f"\n**{expert}:**\n{truncated}\n"
            system_prompt += prev_context

    # Build user message
    user_message = f"Stock Analysis Request: {question}"
    if evidence_context:
        user_message += f"\n\n## Market Data:\n{evidence_context}"
    if kol_context:
        user_message += f"\n\n## KOL Content to Analyze:\n{kol_context}"

    # Call LLM
    try:
        from services.llm_router import call_llm

        response = call_llm(
            model=model,
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_completion_tokens,
            temperature=0.7,
            timeout=timeout
        )

        return {
            'content': response.get('content', ''),
            'finish_reason': response.get('finish_reason', 'stop'),
            'model': model,
            'tokens': response.get('usage', {})
        }

    except Exception as e:
        logger.error(f"Stock expert call failed for {persona_name}: {e}")
        return {
            'content': f"Error calling {persona_name}: {str(e)}",
            'finish_reason': 'error',
            'model': model,
            'tokens': {}
        }


def call_stock_expert_stream(
    persona_name: str,
    question: str,
    evidence_context: str,
    round_num: int = 1,
    previous_responses: Optional[Dict[str, str]] = None,
    kol_context: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    model: str = None,
    max_completion_tokens: int = None,
    timeout: float = 120.0
):
    """
    Stream a stock expert response for real-time display.

    Yields dicts with:
    - {"type": "chunk", "content": "..."} for text chunks
    - {"type": "error", "content": "..."} for errors
    - {"type": "complete", "finish_reason": "...", "model": "..."} when done

    Args:
        persona_name: Stock expert name
        question: User's question about a stock
        evidence_context: Market data context for the expert
        round_num: Discussion round
        previous_responses: Previous expert responses
        kol_context: KOL content if analyzing influencer post
        openai_api_key: API key
        model: Model name
        max_completion_tokens: Max response tokens
        timeout: API timeout

    Yields:
        Dict with type and content
    """
    import os
    import logging
    from config import settings

    logger = logging.getLogger(__name__)

    # Get API key
    if not openai_api_key:
        openai_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not openai_api_key:
        yield {'type': 'error', 'content': "Error: No API key configured"}
        yield {'type': 'complete', 'finish_reason': 'error', 'model': model or 'unknown'}
        return

    # Get expert config for model selection
    expert_info = STOCK_EXPERTS.get(persona_name, {})

    # Get model: prefer expert-specific, then parameter, then settings
    if not model:
        model = expert_info.get("model") or getattr(settings, 'EXPERT_MODEL', 'gemini-3-pro-preview')

    # Get max tokens from settings if not specified
    if max_completion_tokens is None:
        max_completion_tokens = getattr(settings, 'EXPERT_MAX_TOKENS', 6000)

    # Get prompts for this persona
    prompts = get_stock_prompts()
    if persona_name not in prompts:
        yield {'type': 'error', 'content': f"Error: Unknown stock expert '{persona_name}'"}
        yield {'type': 'complete', 'finish_reason': 'error', 'model': model}
        return

    context, task = prompts[persona_name]
    system_prompt = context + "\n\n" + task

    # Add previous responses for context in later rounds
    if previous_responses and round_num > 1:
        prev_context = "\n\n## OTHER EXPERT PERSPECTIVES:\n"
        for expert, resp in previous_responses.items():
            if expert != persona_name:
                truncated = resp[:600] if len(resp) > 600 else resp
                prev_context += f"\n**{expert}:**\n{truncated}\n"
        system_prompt += prev_context

    # Build user message
    user_message = f"Stock Analysis Request: {question}"
    if evidence_context:
        user_message += f"\n\n## Market Data:\n{evidence_context}"
    if kol_context:
        user_message += f"\n\n## KOL Content to Analyze:\n{kol_context}"

    try:
        from services.llm_router import get_llm_router

        router = get_llm_router()
        logger.info(f"Starting stream for stock expert {persona_name}")

        chunk_count = 0
        for chunk in router.call_expert_stream(
            prompt=user_message,
            system=system_prompt,
            model=model,
            max_tokens=max_completion_tokens
        ):
            if chunk.get('type') == 'chunk':
                chunk_count += 1
                yield chunk
            elif chunk.get('type') == 'complete':
                logger.info(f"Stream complete for {persona_name}: {chunk_count} chunks")
                yield {
                    'type': 'complete',
                    'finish_reason': chunk.get('finish_reason', 'stop'),
                    'model': model
                }
                return

        # If loop ends without complete signal
        yield {'type': 'complete', 'finish_reason': 'stop', 'model': model}

    except Exception as e:
        logger.error(f"Stream error for {persona_name}: {e}")
        yield {'type': 'error', 'content': f"Error: {str(e)}"}
        yield {'type': 'complete', 'finish_reason': 'error', 'model': model}


def call_stock_expert_stream_with_round(
    persona_name: str,
    question: str,
    evidence_context: str,
    round_num: int,
    previous_responses: Optional[Dict[str, str]] = None,
    kol_context: Optional[str] = None,
    model: str = None,
    max_completion_tokens: int = None,
):
    """
    Stream a stock expert response with debate-round-specific prompting.

    This is designed for the Expert Debate mode where experts:
    - Round 1: Give initial analysis
    - Round 2: Respond to other experts
    - Round 3: Final rebuttals and verdict

    Args:
        persona_name: Stock expert name
        question: User's question about a stock
        evidence_context: Market data context for the expert
        round_num: Debate round (1, 2, or 3)
        previous_responses: Dict of {expert_name: response} from previous rounds
        kol_context: Optional KOL content
        model: Model override
        max_completion_tokens: Max tokens override

    Yields:
        Dict with type and content
    """
    import os
    import logging
    from config import settings

    logger = logging.getLogger(__name__)

    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        yield {'type': 'error', 'content': "Error: No API key configured"}
        yield {'type': 'complete', 'finish_reason': 'error', 'model': model or 'unknown'}
        return

    # Get expert config
    expert_info = STOCK_EXPERTS.get(persona_name, {})
    if not expert_info:
        yield {'type': 'error', 'content': f"Error: Unknown expert '{persona_name}'"}
        yield {'type': 'complete', 'finish_reason': 'error', 'model': model or 'unknown'}
        return

    # Model selection
    if not model:
        model = expert_info.get("model") or getattr(settings, 'EXPERT_MODEL', 'gemini-3-pro-preview')

    # Max tokens
    if max_completion_tokens is None:
        max_completion_tokens = getattr(settings, 'DEBATE_MAX_TOKENS', 4000)

    # Get base prompts
    prompts = get_stock_prompts()
    if persona_name not in prompts:
        yield {'type': 'error', 'content': f"Error: No prompts for '{persona_name}'"}
        yield {'type': 'complete', 'finish_reason': 'error', 'model': model}
        return

    context, task = prompts[persona_name]

    # Add round-specific instructions
    round_key = f"round_{round_num}"
    round_instruction = DEBATE_ROUND_PROMPTS.get(round_key, "")

    system_prompt = context + "\n\n" + task
    if round_instruction:
        system_prompt += "\n\n" + round_instruction

    # Add previous responses for rounds 2+
    if previous_responses and round_num > 1:
        prev_context = "\n\n## OTHER EXPERT ANALYSES FROM PREVIOUS ROUND(S):\n"
        for expert, resp in previous_responses.items():
            if expert != persona_name:
                # Truncate to reasonable length
                truncated = resp[:800] + "..." if len(resp) > 800 else resp
                icon = EXPERT_ICONS.get(expert, "")
                prev_context += f"\n### {icon} {expert}:\n{truncated}\n"
        system_prompt += prev_context

    # Build user message
    user_message = f"Stock Analysis Request: {question}\n\nRound: {round_num} of 3"
    if evidence_context:
        user_message += f"\n\n## Market Data:\n{evidence_context}"
    if kol_context:
        user_message += f"\n\n## KOL Content:\n{kol_context}"

    # Stream response
    try:
        from services.llm_router import get_llm_router

        router = get_llm_router()
        logger.info(f"Starting debate stream for {persona_name} (Round {round_num})")

        chunk_count = 0
        for chunk in router.call_expert_stream(
            prompt=user_message,
            system=system_prompt,
            model=model,
            max_tokens=max_completion_tokens
        ):
            if chunk.get('type') == 'chunk':
                chunk_count += 1
                yield chunk
            elif chunk.get('type') == 'complete':
                logger.info(f"Debate stream complete for {persona_name} R{round_num}: {chunk_count} chunks")
                yield {
                    'type': 'complete',
                    'finish_reason': chunk.get('finish_reason', 'stop'),
                    'model': model
                }
                return

        yield {'type': 'complete', 'finish_reason': 'stop', 'model': model}

    except Exception as e:
        logger.error(f"Debate stream error for {persona_name} R{round_num}: {e}")
        yield {'type': 'error', 'content': f"Error: {str(e)}"}
        yield {'type': 'complete', 'finish_reason': 'error', 'model': model}


def call_moderator_synthesis_stream(
    question: str,
    evidence_context: str,
    all_rounds: list,
    model: str = None,
    max_completion_tokens: int = None,
):
    """
    Stream the Debate Moderator's synthesis of all debate rounds.

    Args:
        question: Original question/stock
        evidence_context: Stock data
        all_rounds: List of dicts, each with {expert_name: response} for rounds 1-3
        model: Model override
        max_completion_tokens: Max tokens

    Yields:
        Dict with type and content
    """
    import os
    import logging
    from config import settings

    logger = logging.getLogger(__name__)

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        yield {'type': 'error', 'content': "Error: No API key configured"}
        yield {'type': 'complete', 'finish_reason': 'error', 'model': model or 'unknown'}
        return

    # Moderator uses Pro model for synthesis
    if not model:
        model = STOCK_EXPERTS.get("Debate Moderator", {}).get("model", "gemini-3-pro-preview")

    if max_completion_tokens is None:
        max_completion_tokens = getattr(settings, 'SYNTHESIS_MAX_TOKENS', 5000)

    # Get moderator prompts
    prompts = get_stock_prompts()
    if "Debate Moderator" not in prompts:
        yield {'type': 'error', 'content': "Error: Debate Moderator not configured"}
        yield {'type': 'complete', 'finish_reason': 'error', 'model': model}
        return

    context, task = prompts["Debate Moderator"]
    synthesis_instruction = DEBATE_ROUND_PROMPTS.get("synthesis", "")

    system_prompt = context + "\n\n" + task
    if synthesis_instruction:
        system_prompt += "\n\n" + synthesis_instruction

    # Build comprehensive debate transcript
    debate_transcript = "## COMPLETE DEBATE TRANSCRIPT\n\n"

    round_names = ["Initial Analysis", "Cross-Examination", "Final Rebuttals"]
    for round_idx, round_responses in enumerate(all_rounds):
        round_num = round_idx + 1
        round_name = round_names[round_idx] if round_idx < len(round_names) else f"Round {round_num}"
        debate_transcript += f"### ROUND {round_num}: {round_name}\n\n"

        for expert, response in round_responses.items():
            icon = EXPERT_ICONS.get(expert, "")
            # Include more of Round 3 responses since they have final verdicts
            max_len = 1200 if round_num == 3 else 600
            truncated = response[:max_len] + "..." if len(response) > max_len else response
            debate_transcript += f"**{icon} {expert}:**\n{truncated}\n\n"

    # User message
    user_message = f"Synthesize this expert debate on: {question}\n\n"
    user_message += f"## Stock Data:\n{evidence_context}\n\n"
    user_message += debate_transcript

    # Stream
    try:
        from services.llm_router import get_llm_router

        router = get_llm_router()
        logger.info("Starting Moderator synthesis stream")

        chunk_count = 0
        for chunk in router.call_expert_stream(
            prompt=user_message,
            system=system_prompt,
            model=model,
            max_tokens=max_completion_tokens
        ):
            if chunk.get('type') == 'chunk':
                chunk_count += 1
                yield chunk
            elif chunk.get('type') == 'complete':
                logger.info(f"Moderator synthesis complete: {chunk_count} chunks")
                yield {
                    'type': 'complete',
                    'finish_reason': chunk.get('finish_reason', 'stop'),
                    'model': model
                }
                return

        yield {'type': 'complete', 'finish_reason': 'stop', 'model': model}

    except Exception as e:
        logger.error(f"Moderator synthesis error: {e}")
        yield {'type': 'error', 'content': f"Error: {str(e)}"}
        yield {'type': 'complete', 'finish_reason': 'error', 'model': model}


def detect_language(text: str) -> str:
    """
    Detect the primary language of user input.

    Returns:
        'zh_TW' for Traditional Chinese
        'zh_CN' for Simplified Chinese
        'en' for English (default)
    """
    if not text:
        return 'en'

    # Traditional Chinese specific characters (more common in TW/HK)
    trad_chars = set('繁體國語實際學習經濟環濟會議機製電視臺灣開發設計變數顯現問題關係對應據說時間動態測試觀點這個裡兩個'
                     '說話兒點開開關聲說發現還東當後過進與區個還見過關過進這適選過動員書電話車廣復練語談體')

    # Simplified Chinese specific characters
    simp_chars = set('简体国语实际学习经济环济会议机制电视台湾开发设计变数显现问题关系对应据说时间动态测试观点这个里两个'
                     '说话儿点开开关声说发现还东当后过进与区个还见过关过进这适选过动员书电话车广复练语谈体')

    # Count Chinese characters
    trad_count = sum(1 for c in text if c in trad_chars)
    simp_count = sum(1 for c in text if c in simp_chars)

    # Check for any CJK characters
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')

    if cjk_count == 0:
        return 'en'

    # If we have specific traditional/simplified markers
    if trad_count > simp_count:
        return 'zh_TW'
    elif simp_count > trad_count:
        return 'zh_CN'
    elif cjk_count > len(text) * 0.1:  # More than 10% Chinese
        return 'zh_TW'  # Default to Traditional for ambiguous Chinese

    return 'en'


def get_localized_expert_name(expert_name: str, lang: str = 'en') -> str:
    """
    Get localized expert name based on language.

    Args:
        expert_name: English expert name
        lang: Language code ('en', 'zh_TW', 'zh_CN')

    Returns:
        Localized expert name with icon
    """
    icon = EXPERT_ICONS.get(expert_name, "")

    if lang == 'zh_TW':
        zh_name = EXPERT_NAMES_ZH_TW.get(expert_name, expert_name)
        return f"{icon} {zh_name}"
    elif lang == 'zh_CN':
        zh_name = EXPERT_NAMES_ZH_CN.get(expert_name, expert_name)
        return f"{icon} {zh_name}"
    else:
        return f"{icon} {expert_name}"


def get_bilingual_expert_name(expert_name: str, lang: str = 'en') -> str:
    """
    Get expert name with both English and Chinese for clarity.

    Args:
        expert_name: English expert name
        lang: Language code

    Returns:
        Bilingual expert name (e.g., "🐂 牛市分析師 (Bull Analyst)")
    """
    icon = EXPERT_ICONS.get(expert_name, "")

    if lang == 'zh_TW':
        zh_name = EXPERT_NAMES_ZH_TW.get(expert_name, expert_name)
        return f"{icon} {zh_name} ({expert_name})"
    elif lang == 'zh_CN':
        zh_name = EXPERT_NAMES_ZH_CN.get(expert_name, expert_name)
        return f"{icon} {zh_name} ({expert_name})"
    else:
        return f"{icon} {expert_name}"
