"""
Stock Advisor - Simple Conversational Stock Assistant

A friendly chatbot that helps beginners decide whether to buy stocks.
Ask simple questions like "Should I buy NVDA?" and get clear answers.

Run with: chainlit run app_sa.py
"""

import asyncio
import chainlit as cl
from typing import Optional
import logging
import os
import re

from config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# Simple Stock Advisor Configuration
# =============================================================================

SYSTEM_PROMPT = """You are a friendly stock advisor helping beginners make investment decisions.

YOUR PERSONALITY:
- Speak in plain, simple language - no jargon
- Be honest and direct
- Always mention risks clearly
- Never pressure anyone to buy
- Admit when you're uncertain

WHEN ASKED ABOUT A STOCK:
1. Give a clear recommendation: BUY, HOLD, or WAIT
2. Explain WHY in 2-3 simple sentences
3. Mention the biggest risk in plain terms
4. Suggest how much to invest (always conservative for beginners)

RESPONSE FORMAT (keep it short!):
📊 **[STOCK NAME]** - Current Price: $XXX

**My Take:** [BUY / HOLD / WAIT]

**Why:** [2-3 simple sentences explaining your reasoning]

**Biggest Risk:** [One clear risk in plain English]

**For Beginners:** [Suggestion like "Start small - maybe $100-500 to learn"]

⚠️ *Remember: I'm an AI assistant, not a financial advisor. Never invest money you can't afford to lose.*

IMPORTANT RULES:
- If someone seems to be risking too much money, WARN them
- If a stock is very risky (penny stocks, meme stocks), say so clearly
- Always encourage diversification ("don't put all eggs in one basket")
- If you don't have current data, be honest about it
- Keep responses SHORT - beginners get overwhelmed by walls of text
"""

# =============================================================================
# Helper Functions
# =============================================================================

def extract_ticker(message: str) -> Optional[str]:
    """Extract stock ticker from user message."""
    message_upper = message.upper()

    # Common English words to exclude (extensive list)
    COMMON_WORDS = {
        # Articles, pronouns, prepositions
        'THE', 'AND', 'FOR', 'NOT', 'YES', 'CAN', 'YOU', 'HOW', 'WHY', 'WHAT',
        'THIS', 'THAT', 'WITH', 'FROM', 'INTO', 'SOME', 'JUST', 'ANY', 'ALL',
        'ARE', 'WAS', 'WERE', 'BEEN', 'BEING', 'HAS', 'HAD', 'DOES', 'DID',
        'WILL', 'BUT', 'OUT', 'WHO', 'HIS', 'HER', 'ITS', 'OUR', 'YOUR',
        # Verbs
        'BUY', 'SELL', 'GET', 'GOT', 'PUT', 'LET', 'SAY', 'SAID', 'MAKE',
        'TAKE', 'COME', 'SEE', 'KNOW', 'THINK', 'WANT', 'GIVE', 'USE', 'FIND',
        'TELL', 'ASK', 'WORK', 'SEEM', 'FEEL', 'TRY', 'LEAVE', 'CALL', 'KEEP',
        'HAVE', 'NEED', 'MEAN', 'MOVE', 'LIVE', 'HEAR', 'LOOK', 'LIKE',
        # Adjectives/Adverbs
        'GOOD', 'BAD', 'NEW', 'OLD', 'BIG', 'LONG', 'BEST', 'MORE', 'MOST',
        'VERY', 'MUCH', 'WELL', 'EVEN', 'ALSO', 'BACK', 'ONLY', 'JUST',
        'NOW', 'THEN', 'HERE', 'WHEN', 'WHERE', 'STILL', 'SURE', 'REAL',
        # Time words
        'TODAY', 'TIME', 'YEAR', 'DAY', 'WEEK', 'MONTH',
        # Finance words (not tickers)
        'STOCK', 'STOCKS', 'INVEST', 'MONEY', 'SHARE', 'SHARES', 'PRICE',
        'MARKET', 'TRADE', 'RISK', 'CASH', 'FUND', 'BOND', 'HOLD',
        # Question words
        'SHOULD', 'WOULD', 'COULD', 'ABOUT', 'REALLY', 'MAYBE',
        # Other common words
        'GOING', 'DOING', 'THING', 'HELP', 'PLEASE', 'THANKS', 'OKAY',
        # Two-letter words
        'IS', 'IT', 'IN', 'TO', 'DO', 'GO', 'SO', 'NO', 'IF', 'OR', 'AS',
        'AT', 'BY', 'ON', 'UP', 'AN', 'BE', 'HE', 'WE', 'ME', 'MY', 'OF',
    }

    # Known company name -> ticker mappings
    COMPANY_NAMES = {
        'APPLE': 'AAPL', 'GOOGLE': 'GOOGL', 'AMAZON': 'AMZN', 'MICROSOFT': 'MSFT',
        'TESLA': 'TSLA', 'NVIDIA': 'NVDA', 'META': 'META', 'FACEBOOK': 'META',
        'NETFLIX': 'NFLX', 'DISNEY': 'DIS', 'INTEL': 'INTC', 'COSTCO': 'COST',
        'WALMART': 'WMT', 'TARGET': 'TGT', 'STARBUCKS': 'SBUX', 'NIKE': 'NKE',
        'PEPSI': 'PEP', 'COCA': 'KO', 'COKE': 'KO', 'BOEING': 'BA',
        'PALANTIR': 'PLTR', 'GAMESTOP': 'GME', 'COINBASE': 'COIN',
        'PAYPAL': 'PYPL', 'SPOTIFY': 'SPOT', 'UBER': 'UBER', 'LYFT': 'LYFT',
        'AIRBNB': 'ABNB', 'ZOOM': 'ZM', 'SLACK': 'WORK', 'SHOPIFY': 'SHOP',
        'SNOWFLAKE': 'SNOW', 'CROWDSTRIKE': 'CRWD', 'DATADOG': 'DDOG',
        'SALESFORCE': 'CRM', 'ORACLE': 'ORCL', 'IBM': 'IBM', 'CISCO': 'CSCO',
        'QUALCOMM': 'QCOM', 'BROADCOM': 'AVGO', 'BERKSHIRE': 'BRK',
    }

    # Check for company names first
    for company, ticker in COMPANY_NAMES.items():
        if company in message_upper:
            return ticker

    # Priority patterns (most specific first)
    patterns = [
        r'\$([A-Z]{1,5})\b',                    # $NVDA (highest priority)
        r'\bbuy\s+([A-Z]{2,5})\b',              # buy NVDA
        r'\bsell\s+([A-Z]{2,5})\b',             # sell NVDA
        r'\babout\s+([A-Z]{2,5})\b',            # about NVDA
        r'\b([A-Z]{2,5})\s+stock\b',            # NVDA stock
        r'\b([A-Z]{2,5})\s+shares?\b',          # NVDA shares
        r'\binto\s+([A-Z]{2,5})\b',             # into NVDA
        r'\binvest(?:ing)?\s+in\s+([A-Z]{2,5})\b',  # invest in NVDA
    ]

    for pattern in patterns:
        match = re.search(pattern, message_upper)
        if match:
            ticker = match.group(1)
            if ticker not in COMMON_WORDS and len(ticker) >= 2:
                return ticker

    # Fallback: find likely tickers (all caps, 2-5 letters, not common words)
    # Look for words that are ALL CAPS in the original message (user emphasized them)
    original_words = re.findall(r'\b([A-Z]{2,5})\b', message)
    for word in original_words:
        if word.upper() not in COMMON_WORDS:
            return word.upper()

    return None


async def get_stock_info(ticker: str) -> str:
    """Fetch stock data and format it simply."""
    try:
        from services.stock_data_service import fetch_stock_data

        stock_data = await asyncio.to_thread(
            fetch_stock_data,
            ticker,
            include_quote=True,
            include_financials=True,
            include_news=True,
            include_market_search=False,  # Keep it simple
        )

        return stock_data.to_prompt_context()
    except Exception as e:
        logger.error(f"Failed to fetch stock data for {ticker}: {e}")
        return f"Note: I couldn't fetch live data for {ticker}. I'll give you my general thoughts based on what I know."


async def chat_with_advisor(user_message: str, stock_context: str = "") -> str:
    """Get response from the simple stock advisor."""
    try:
        from services.llm_router import get_llm_router

        router = get_llm_router()

        # Build the prompt
        full_prompt = user_message
        if stock_context:
            full_prompt = f"{user_message}\n\n--- CURRENT STOCK DATA ---\n{stock_context}"

        # Stream the response
        response_text = ""
        for chunk in router.call_expert_stream(
            prompt=full_prompt,
            system=SYSTEM_PROMPT,
            model=settings.EXPERT_MODEL,
            max_tokens=1500  # Keep responses short
        ):
            if chunk.get("type") == "chunk":
                response_text += chunk.get("content", "")

        return response_text

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return f"Sorry, I'm having trouble right now. Please try again in a moment. (Error: {str(e)[:100]})"


# =============================================================================
# Chainlit Handlers
# =============================================================================

@cl.on_chat_start
async def start():
    """Welcome the user."""
    welcome = """# 👋 Hey! I'm your Stock Advisor

I help beginners figure out whether to buy stocks. Just ask me things like:

- **"Should I buy NVDA?"**
- **"Is Apple a good investment?"**
- **"What do you think about Tesla?"**
- **"I have $500 to invest, any suggestions?"**

I'll give you a straight answer with simple explanations - no confusing jargon!

⚠️ *Quick disclaimer: I'm an AI, not a licensed financial advisor. Always do your own research and never invest more than you can afford to lose.*

---

**What stock are you curious about?** 🤔"""

    await cl.Message(content=welcome).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages."""
    user_input = message.content.strip()

    if not user_input:
        await cl.Message(content="Just type a stock name or ask me a question!").send()
        return

    # Check if they mentioned a specific stock
    ticker = extract_ticker(user_input)

    # Show thinking indicator
    async with cl.Step(name="Thinking...", type="run") as step:
        stock_context = ""

        if ticker:
            step.output = f"Looking up {ticker}..."
            stock_context = await get_stock_info(ticker)
        else:
            step.output = "Thinking about your question..."

        # Get advisor response
        response = await chat_with_advisor(user_input, stock_context)
        step.output = "Done!"

    # Send the response
    await cl.Message(content=response).send()

    # Add a helpful follow-up prompt for beginners
    if ticker:
        await cl.Message(
            content=f"---\n💡 *Want to know more? Ask me things like:*\n- \"What are the risks of {ticker}?\"\n- \"How much should I invest in {ticker}?\"\n- \"Is now a good time to buy {ticker}?\"",
        ).send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
