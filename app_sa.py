"""
Stock Advisor - Chainlit App

A conversational stock analysis assistant with AI stock experts.
Upload KOL screenshots, ask about stocks, get multi-expert analysis.

Run with: chainlit run app_sa.py
"""

import asyncio
import chainlit as cl
from chainlit.input_widget import TextInput, Select
from datetime import date
from typing import Dict, List, Optional
from pathlib import Path
import logging
import os
import json
import re

from config import settings
from stocks.stock_personas import (
    STOCK_EXPERTS,
    STOCK_PRESETS,
    EXPERT_ICONS,
    get_default_stock_experts,
    call_stock_expert,
    call_stock_expert_stream,
    detect_best_stock_expert,
)
from services.stock_data_service import (
    fetch_stock_data,
    extract_tickers,
    build_expert_context,
    analyze_kol_screenshot,
    search_why_stock_moved,
)
from services.kol_analyzer import KOLAnalyzer, analyze_kol_text

logger = logging.getLogger(__name__)


# =============================================================================
# Input Validation & Sanitization
# =============================================================================

MAX_MESSAGE_LENGTH = 2000
MAX_TICKER_LENGTH = 10

INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?previous\s+instructions',
    r'disregard\s+(all\s+)?above',
    r'forget\s+(everything|all)',
    r'you\s+are\s+now\s+a',
    r'new\s+instructions:',
    r'system\s*:\s*',
]


def sanitize_input(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    """Sanitize user input."""
    if not text:
        return ""
    text = str(text).strip()
    if len(text) > max_length:
        text = text[:max_length]
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            logger.warning(f"Potential injection pattern detected: {pattern}")
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text


def validate_ticker(ticker: str) -> tuple[bool, str]:
    """Validate stock ticker symbol."""
    if not ticker:
        return False, "Ticker is required"
    ticker = ticker.upper().strip()
    if len(ticker) > MAX_TICKER_LENGTH:
        return False, "Ticker too long"
    if not re.match(r'^[A-Z]{1,5}$', ticker):
        return False, "Invalid ticker format"
    return True, ticker


def extract_advisory_summary(response: str) -> str:
    """Extract advisory summary fields from expert response."""
    labels = [
        "Recommendation",
        "Time Horizon",
        "Timing Guidance",
        "Confidence",
        "Key Reasons",
        "Key Risks",
    ]
    summary_lines = []
    for label in labels:
        pattern = rf"^{label}\s*:\s*(.+)$"
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            summary_lines.append(f"**{label}:** {match.group(1).strip()}")
    return "\n".join(summary_lines)


def validate_tickers(tickers: List[str]) -> tuple[List[str], List[str]]:
    """Validate a list of tickers, returning valid and invalid lists."""
    valid = []
    invalid = []
    for ticker in tickers:
        is_valid, result = validate_ticker(ticker)
        if is_valid:
            valid.append(result)
        else:
            invalid.append(ticker)
    return valid, invalid


# =============================================================================
# Stock Analysis Functions
# =============================================================================

async def extract_analysis_request(user_message: str) -> dict:
    """Extract stock analysis request from natural language."""
    from services.llm_router import get_llm_router
    router = get_llm_router()

    user_message = sanitize_input(user_message)

    prompt = f"""Extract stock analysis details from this message. Return ONLY valid JSON.

User message: "{user_message}"

Return JSON:
{{
  "tickers": ["list of ticker symbols mentioned, e.g., NVDA, AAPL"],
  "question_type": "news|why_moved|analysis|comparison|general",
  "direction": "up|down|null (if asking why stock moved)",
  "specific_question": "the user's specific question or null"
}}

Examples:
- "Why did NVDA fall yesterday?" -> {{"tickers": ["NVDA"], "question_type": "why_moved", "direction": "down"}}
- "Analyze AAPL stock" -> {{"tickers": ["AAPL"], "question_type": "analysis"}}
- "Compare NVDA and AMD" -> {{"tickers": ["NVDA", "AMD"], "question_type": "comparison"}}
- "What's happening with tech stocks?" -> {{"tickers": [], "question_type": "general"}}

Return ONLY JSON."""

    try:
        response_text = ""
        for chunk in router.call_expert_stream(
            prompt=prompt,
            system="You are a JSON extraction assistant. Return only valid JSON."
        ):
            if chunk.get("type") == "chunk":
                response_text += chunk.get("content", "")

        # Parse JSON
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response_text[start:end])
        except json.JSONDecodeError:
            pass

        # Fallback: extract tickers from message directly
        tickers = extract_tickers(user_message)
        return {
            "tickers": tickers,
            "question_type": "analysis" if tickers else "general",
            "specific_question": user_message
        }

    except Exception as e:
        logger.error(f"Analysis request extraction failed: {e}")
        return {"tickers": [], "question_type": "general", "specific_question": user_message}


async def run_expert_panel(
    question: str,
    tickers: List[str],
    preset: str = "Quick Analysis",
    kol_context: str = ""
) -> Dict[str, str]:
    """Run expert panel analysis on stocks."""
    logger.info(f"Running expert panel: preset={preset}, tickers={tickers}")
    preset_config = STOCK_PRESETS.get(preset, STOCK_PRESETS["Quick Analysis"])
    expert_names = preset_config["experts"]

    responses = {}
    primary_ticker = tickers[0] if tickers else None

    # Fetch stock data for context (run in thread to avoid blocking)
    # Enable market search for real-time news via Google Search grounding
    evidence_context = ""
    if primary_ticker:
        stock_context = await asyncio.to_thread(
            fetch_stock_data,
            primary_ticker,
            include_quote=True,
            include_financials=True,
            include_news=True,
            include_market_search=True,  # Real-time Google Search grounding
        )
        evidence_context = stock_context.to_prompt_context()

    # Show progress message
    progress_msg = await cl.Message(
        content=f"**Consulting {len(expert_names)} experts...**\n" +
                " | ".join([f"⏳ {EXPERT_ICONS.get(e, '📊')} {e}" for e in expert_names])
    ).send()

    # Run experts in parallel
    failed_experts = []

    async def call_expert_async(expert_name: str) -> tuple[str, str, bool]:
        try:
            full_response = ""
            async for chunk in stream_expert_response(
                expert_name, question, evidence_context, kol_context
            ):
                full_response += chunk
            return expert_name, full_response, True
        except Exception as e:
            logger.error(f"Expert {expert_name} failed: {e}")
            return expert_name, f"*Analysis unavailable*", False

    # Execute all experts concurrently
    tasks = [call_expert_async(name) for name in expert_names]
    logger.info(f"Waiting for {len(tasks)} expert tasks to complete...")

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"All expert tasks completed, processing {len(results)} results")
    except Exception as e:
        logger.error(f"asyncio.gather failed: {e}")
        results = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Expert task {i} raised exception: {result}")
            expert_name = expert_names[i] if i < len(expert_names) else f"Unknown-{i}"
            responses[expert_name] = "*Analysis unavailable due to error*"
            failed_experts.append(expert_name)
        else:
            expert_name, response, success = result
            responses[expert_name] = response
            if not success:
                failed_experts.append(expert_name)
            else:
                logger.info(f"Expert {expert_name} response length: {len(response)} chars")

    # Update progress to complete
    logger.info("Removing progress message...")
    try:
        await progress_msg.remove()
        logger.info("Progress message removed successfully")
    except Exception as e:
        logger.error(f"Failed to remove progress message: {e}")

    # Notify user of any failures
    if failed_experts:
        await cl.Message(
            content=f"⚠️ Some experts encountered issues: {', '.join(failed_experts)}. "
                    f"Analysis may be incomplete."
        ).send()
        logger.warning(f"Expert panel partial failure: {failed_experts}")

    return responses


async def stream_expert_response(
    expert_name: str,
    question: str,
    evidence_context: str,
    kol_context: str = ""
):
    """Stream expert response chunks using thread pool to avoid blocking."""
    import queue
    import threading

    result_queue = queue.Queue()
    error_holder = [None]

    def run_sync_generator():
        try:
            for chunk in call_stock_expert_stream(
                persona_name=expert_name,
                question=question,
                evidence_context=evidence_context,
                kol_context=kol_context
            ):
                result_queue.put(chunk)
            result_queue.put(None)  # Signal completion
        except Exception as e:
            error_holder[0] = e
            result_queue.put(None)

    # Start the sync generator in a thread
    thread = threading.Thread(target=run_sync_generator, daemon=True)
    thread.start()

    # Yield chunks as they become available
    while True:
        # Non-blocking check with small timeout to allow asyncio to breathe
        try:
            chunk = await asyncio.to_thread(result_queue.get, timeout=0.1)
        except:
            await asyncio.sleep(0.05)
            continue

        if chunk is None:
            if error_holder[0]:
                logger.error(f"Stream error: {error_holder[0]}")
            break

        if chunk.get("type") == "chunk":
            yield chunk.get("content", "")


async def display_expert_responses(responses: Dict[str, str], question: str):
    """Display expert responses with formatting."""
    logger.info(f"display_expert_responses called with {len(responses)} responses")

    for expert_name, response in responses.items():
        logger.info(f"Displaying response from {expert_name} ({len(response)} chars)")
        icon = EXPERT_ICONS.get(expert_name, "📊")
        advisory_summary = extract_advisory_summary(response)
        advisory_block = f"> {advisory_summary}\n\n" if advisory_summary else ""
        try:
            await cl.Message(
                content=f"## {icon} {expert_name}\n\n{advisory_block}{response}"
            ).send()
            logger.info(f"Successfully displayed {expert_name} response")
        except Exception as e:
            logger.error(f"Failed to display {expert_name} response: {e}")

    # Store responses for follow-up
    cl.user_session.set("expert_responses", responses)
    cl.user_session.set("last_question", question)

    # Offer follow-up actions
    actions = [
        cl.Action(name="ask_followup", label="💬 Ask Follow-up", value="followup", payload={}),
        cl.Action(name="deep_dive", label="🔍 Deep Dive", value="deep", payload={}),
        cl.Action(name="advisory_buy", label="🟢 Should I buy?", value="buy", payload={}),
        cl.Action(name="advisory_sell", label="🔴 Should I sell?", value="sell", payload={}),
    ]
    await cl.Message(
        content="---\n*Ask a follow-up question or request a deep dive analysis.*",
        actions=actions
    ).send()


# =============================================================================
# Image Upload Handler
# =============================================================================

async def handle_image_upload(message: cl.Message):
    """Handle uploaded KOL screenshot."""
    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB

    supported_types = {'image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp'}
    processed_any = False

    for element in message.elements:
        if element.mime not in supported_types:
            await cl.Message(
                content=f"⚠️ Unsupported file type: `{element.mime}`. "
                        f"Supported: PNG, JPEG, GIF, WebP."
            ).send()
            logger.info(f"Skipped unsupported file type: {element.mime}")
            continue

        processed_any = True
        logger.info(f"Processing image upload: type={element.mime}")

        # Validate size
        if element.path and os.path.exists(element.path):
            file_size = os.path.getsize(element.path)
            if file_size > MAX_IMAGE_SIZE:
                await cl.Message(
                    content=f"Image too large ({file_size/1024/1024:.1f}MB). Max 5MB allowed."
                ).send()
                logger.warning(f"Image rejected: size={file_size/1024/1024:.1f}MB exceeds limit")
                continue

        # Analyze the screenshot
        async with cl.Step(name="KOL Screenshot Analysis", type="tool") as step:
            step.output = "Analyzing screenshot..."

            result = await asyncio.to_thread(analyze_kol_screenshot, element.path)

            if not result["success"]:
                await cl.Message(
                    content=f"**Analysis failed:** {result['error']}"
                ).send()
                logger.error(f"KOL screenshot analysis failed: {result.get('error')}")
                return

            ocr = result["ocr_result"]
            logger.info(f"KOL analysis complete: tickers={ocr.get('tickers')}, sentiment={ocr.get('sentiment')}")

        # Display extracted info
        summary_lines = ["## 📸 KOL Screenshot Analyzed\n"]

        if ocr.get("author"):
            summary_lines.append(f"**Author:** {ocr['author']}")
        if ocr.get("platform"):
            summary_lines.append(f"**Platform:** {ocr['platform']}")
        if ocr.get("tickers"):
            summary_lines.append(f"**Tickers:** {', '.join(ocr['tickers'])}")
        if ocr.get("sentiment"):
            sentiment_emoji = {
                "bullish": "🐂", "bearish": "🐻",
                "neutral": "➖", "mixed": "🔄"
            }.get(ocr["sentiment"].lower(), "❓")
            summary_lines.append(f"**Sentiment:** {sentiment_emoji} {ocr['sentiment'].title()}")

        if ocr.get("key_claims"):
            summary_lines.append("\n**Key Claims:**")
            for i, claim in enumerate(ocr["key_claims"][:5], 1):
                summary_lines.append(f"  {i}. {claim}")

        await cl.Message(content="\n".join(summary_lines)).send()

        # Store for analysis
        cl.user_session.set("kol_analysis", result)
        cl.user_session.set("detected_tickers", ocr.get("tickers", []))

        # Offer analysis options
        if ocr.get("tickers"):
            actions = [
                cl.Action(
                    name="analyze_kol_ticker",
                    label=f"📊 Analyze {ocr['tickers'][0]}",
                    value=ocr["tickers"][0],
                    payload={"ticker": ocr["tickers"][0]}
                ),
                cl.Action(
                    name="validate_claims",
                    label="✅ Validate Claims",
                    value="validate",
                    payload={}
                ),
            ]
            await cl.Message(
                content="What would you like to do?",
                actions=actions
            ).send()


@cl.action_callback("analyze_kol_ticker")
async def on_analyze_kol_ticker(action: cl.Action):
    """Analyze ticker from KOL screenshot."""
    ticker = action.payload.get("ticker", action.value)
    kol_analysis = cl.user_session.get("kol_analysis", {})
    kol_context = ""

    if kol_analysis and kol_analysis.get("ocr_result"):
        ocr = kol_analysis["ocr_result"]
        kol_context = f"""
KOL Post Analysis:
- Author: {ocr.get('author', 'Unknown')}
- Platform: {ocr.get('platform', 'Unknown')}
- Sentiment: {ocr.get('sentiment', 'Unknown')}
- Key Claims: {'; '.join(ocr.get('key_claims', []))}
"""

    question = f"Analyze {ticker} stock considering the KOL's claims"
    preset = "KOL Review"

    responses = await run_expert_panel(
        question=question,
        tickers=[ticker],
        preset=preset,
        kol_context=kol_context
    )
    await display_expert_responses(responses, question)


@cl.action_callback("validate_claims")
async def on_validate_claims(action: cl.Action):
    """Validate KOL claims with data."""
    kol_analysis = cl.user_session.get("kol_analysis", {})
    if not kol_analysis or not kol_analysis.get("ocr_result"):
        await cl.Message(content="No KOL analysis available to validate.").send()
        return

    ocr = kol_analysis["ocr_result"]
    claims = ocr.get("key_claims", [])
    tickers = ocr.get("tickers", [])

    if not claims:
        await cl.Message(content="No specific claims found to validate.").send()
        return

    question = f"Validate these claims about {', '.join(tickers)}: {'; '.join(claims)}"

    responses = await run_expert_panel(
        question=question,
        tickers=tickers,
        preset="Deep Dive"
    )
    await display_expert_responses(responses, question)


# =============================================================================
# KOL Text Analysis Handler
# =============================================================================

def detect_kol_text(text: str) -> bool:
    """
    Detect if the input looks like a KOL opinion to analyze.

    Patterns that suggest KOL text:
    - Starts with "analyze this" or similar
    - Contains @ mentions
    - Long text with stock opinions
    - Quoted text or multiple lines
    """
    text_lower = text.lower().strip()

    # Explicit instructions to analyze
    analyze_triggers = [
        "analyze this",
        "what do you think of this",
        "evaluate this",
        "review this opinion",
        "is this true",
        "validate this",
        "here's what",
        "this person says",
        "according to",
    ]

    for trigger in analyze_triggers:
        if text_lower.startswith(trigger):
            return True

    # Contains @ mentions and is relatively long
    has_mentions = "@" in text and len(text) > 100

    # Multiple lines or quoted text
    has_quotes = text.count('"') >= 2 or text.count("'") >= 2
    is_multiline = "\n" in text and len(text) > 150

    # Contains opinion indicators
    opinion_words = ["bullish", "bearish", "buy", "sell", "long", "short", "target", "prediction"]
    has_opinion = any(word in text_lower for word in opinion_words)

    return (has_mentions and has_opinion) or (is_multiline and has_opinion) or (has_quotes and len(text) > 100)


async def handle_kol_text(text: str):
    """
    Handle pasted KOL text for analysis.

    Extracts claims, displays summary, and runs expert panel.
    """
    async with cl.Step(name="Analyzing KOL Opinion", type="tool") as step:
        step.output = "Extracting claims..."
        claim = await asyncio.to_thread(analyze_kol_text, text)
        step.output = f"Found: @{claim.author} on {claim.ticker or 'general market'}"

    # Display extracted info
    await cl.Message(content=f"## 📝 KOL Opinion Extracted\n\n{claim.format_summary()}").send()

    # Store for follow-up
    cl.user_session.set("kol_claim", claim)
    if claim.ticker:
        cl.user_session.set("detected_tickers", [claim.ticker])

    # Build KOL context for experts
    analyzer = KOLAnalyzer()
    kol_context = analyzer.format_for_expert_context(claim)

    # Offer analysis options
    if claim.ticker:
        actions = [
            cl.Action(
                name="analyze_kol_claim",
                label=f"📊 Analyze {claim.ticker}",
                value=claim.ticker,
                payload={"ticker": claim.ticker}
            ),
            cl.Action(
                name="validate_kol_claim",
                label="✅ Validate Claims",
                value="validate",
                payload={}
            ),
        ]
        await cl.Message(
            content="What would you like me to do with this opinion?",
            actions=actions
        ).send()
    else:
        await cl.Message(
            content="No specific ticker found. You can ask me to analyze a particular stock mentioned."
        ).send()


@cl.action_callback("analyze_kol_claim")
async def on_analyze_kol_claim(action: cl.Action):
    """Analyze ticker from KOL text claim."""
    ticker = action.payload.get("ticker", action.value)
    claim = cl.user_session.get("kol_claim")

    kol_context = ""
    if claim:
        analyzer = KOLAnalyzer()
        kol_context = analyzer.format_for_expert_context(claim)

    question = f"Analyze {ticker} stock and evaluate the KOL's thesis"

    responses = await run_expert_panel(
        question=question,
        tickers=[ticker],
        preset="KOL Review",
        kol_context=kol_context
    )
    await display_expert_responses(responses, question)


@cl.action_callback("validate_kol_claim")
async def on_validate_kol_claim(action: cl.Action):
    """Validate claims from KOL text."""
    claim = cl.user_session.get("kol_claim")
    if not claim:
        await cl.Message(content="No KOL claim to validate.").send()
        return

    tickers = [claim.ticker] if claim.ticker else []
    if not tickers:
        await cl.Message(content="No ticker found in the claim to validate.").send()
        return

    claims_text = "; ".join(claim.key_points) if claim.key_points else claim.thesis
    question = f"Validate these claims about {claim.ticker}: {claims_text}"

    responses = await run_expert_panel(
        question=question,
        tickers=tickers,
        preset="Deep Dive"
    )
    await display_expert_responses(responses, question)


# =============================================================================
# Action Callbacks
# =============================================================================

@cl.action_callback("ask_followup")
async def on_ask_followup(action: cl.Action):
    """Prompt user for follow-up question."""
    await cl.Message(
        content="What would you like to know more about? You can ask about:\n"
                "- Specific price targets or levels\n"
                "- Risk factors in more detail\n"
                "- Technical or fundamental specifics\n"
                "- Comparison with other stocks"
    ).send()


@cl.action_callback("deep_dive")
async def on_deep_dive(action: cl.Action):
    """Run full panel deep dive."""
    last_question = cl.user_session.get("last_question", "")
    tickers = cl.user_session.get("detected_tickers", [])

    if not tickers:
        tickers = extract_tickers(last_question)

    if tickers:
        responses = await run_expert_panel(
            question=f"Deep dive analysis of {', '.join(tickers)}",
            tickers=tickers,
            preset="Deep Dive"
        )
        await display_expert_responses(responses, last_question)
    else:
        await cl.Message(content="No tickers found for deep dive. Please specify a stock symbol.").send()


@cl.action_callback("quick_analysis")
async def on_quick_analysis(action: cl.Action):
    """Run quick analysis on a ticker."""
    ticker = action.value
    responses = await run_expert_panel(
        question=f"Quick analysis of {ticker}",
        tickers=[ticker],
        preset="Quick Analysis"
    )
    await display_expert_responses(responses, f"Quick analysis of {ticker}")


@cl.action_callback("advisory_buy")
async def on_advisory_buy(action: cl.Action):
    """Provide buy-focused advisory guidance."""
    tickers = cl.user_session.get("detected_tickers", [])
    if not tickers:
        await cl.Message(content="Please specify a stock symbol for buy guidance.").send()
        return
    ticker = tickers[0]
    question = (
        f"Should I buy {ticker}? Provide a Buy/Sell/Hold recommendation with timing guidance, "
        "confidence level, key reasons, and key risks."
    )
    responses = await run_expert_panel(
        question=question,
        tickers=[ticker],
        preset="Trade Planning"
    )
    await display_expert_responses(responses, question)


@cl.action_callback("advisory_sell")
async def on_advisory_sell(action: cl.Action):
    """Provide sell-focused advisory guidance."""
    tickers = cl.user_session.get("detected_tickers", [])
    if not tickers:
        await cl.Message(content="Please specify a stock symbol for sell guidance.").send()
        return
    ticker = tickers[0]
    question = (
        f"Should I sell {ticker}? Provide a Buy/Sell/Hold recommendation with timing guidance, "
        "confidence level, key reasons, and key risks."
    )
    responses = await run_expert_panel(
        question=question,
        tickers=[ticker],
        preset="Trade Planning"
    )
    await display_expert_responses(responses, question)


# =============================================================================
# Main Handlers
# =============================================================================

@cl.on_chat_start
async def start():
    """Initialize chat session."""

    # Set up ChatSettings
    settings_widgets = [
        TextInput(
            id="ticker",
            label="Stock Symbol",
            placeholder="NVDA",
            initial=""
        ),
        Select(
            id="preset",
            label="Expert Panel",
            values=list(STOCK_PRESETS.keys()),
            initial_value="Quick Analysis"
        ),
    ]

    await cl.ChatSettings(settings_widgets).send()

    # Initialize session state
    cl.user_session.set("expert_responses", {})
    cl.user_session.set("detected_tickers", [])
    cl.user_session.set("kol_analysis", None)
    cl.user_session.set("last_question", "")

    # Welcome message
    await cl.Message(
        content="""# 📊 Stock Advisor

Hi! I'm your AI stock analysis assistant with 6 expert perspectives:
- 🐂 **Bull Analyst** - Growth catalysts and upside
- 🐻 **Bear Analyst** - Risk factors and downside
- 📈 **Technical Analyst** - Chart patterns and levels
- 📊 **Fundamental Analyst** - Valuations and financials
- 📰 **Sentiment Analyst** - News and social sentiment
- 🛡️ **Risk Manager** - Position sizing and risk

**Ask me anything about stocks**, upload a screenshot, or paste a KOL opinion.

_Examples:_
- "Why did NVDA fall yesterday?"
- "Analyze AAPL stock"
- "Compare TSLA and RIVN"
- Upload a screenshot of a stock tweet
- Paste a KOL's opinion: "Analyze this: @investor says NVDA will hit $200..."

⚠️ *Not financial advice. Always do your own research.*"""
    ).send()


@cl.on_settings_update
async def on_settings_update(settings_dict: Dict):
    """Handle settings update."""
    ticker = settings_dict.get("ticker", "").upper().strip()
    preset = settings_dict.get("preset", "Quick Analysis")

    # Always persist the preset selection
    cl.user_session.set("current_preset", preset)
    logger.info(f"Preset updated: {preset}")

    if ticker:
        is_valid, validated = validate_ticker(ticker)
        if is_valid:
            cl.user_session.set("detected_tickers", [validated])
            await cl.Message(
                content=f"Settings updated: **{validated}** with {preset} preset"
            ).send()
        else:
            await cl.Message(
                content=f"Invalid ticker '{ticker}'. Please use 1-5 letters (e.g., NVDA, AAPL)."
            ).send()
    else:
        await cl.Message(
            content=f"Expert panel updated: **{preset}**"
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages."""

    # Check for image uploads first
    if message.elements:
        image_elements = [e for e in message.elements if e.mime and e.mime.startswith('image/')]
        if image_elements:
            await handle_image_upload(message)
            return

    # Handle text messages
    user_input = sanitize_input(message.content)

    if not user_input:
        await cl.Message(content="Please enter a question or upload an image.").send()
        return

    # Check if this looks like pasted KOL text to analyze
    if detect_kol_text(user_input):
        logger.info("Detected KOL text input")
        await handle_kol_text(user_input)
        return

    # Extract analysis request
    async with cl.Step(name="Understanding Request", type="run") as step:
        request = await extract_analysis_request(user_input)
        step.output = f"Detected: {request.get('question_type', 'analysis')}"

    raw_tickers = request.get("tickers", [])
    question_type = request.get("question_type", "analysis")

    # Validate extracted tickers
    tickers, invalid_tickers = validate_tickers(raw_tickers)
    logger.info(f"Extracted tickers: {tickers}, question_type: {question_type}")

    if invalid_tickers:
        await cl.Message(
            content=f"⚠️ Skipping invalid ticker(s): {', '.join(invalid_tickers)}"
        ).send()

    # Store detected tickers
    if tickers:
        cl.user_session.set("detected_tickers", tickers)

    # Handle different question types
    if question_type == "why_moved" and tickers:
        # Special handling for "why did X move" questions
        direction = request.get("direction", "moved")
        async with cl.Step(name="Market Search", type="tool") as step:
            explanation = await asyncio.to_thread(
                search_why_stock_moved, tickers[0], direction
            )
            step.output = f"Found explanation for {tickers[0]}"

        await cl.Message(
            content=f"## Why {tickers[0]} {direction}\n\n{explanation}"
        ).send()

        # Also run expert panel for deeper analysis
        responses = await run_expert_panel(
            question=user_input,
            tickers=tickers,
            preset="Quick Analysis"
        )
        await display_expert_responses(responses, user_input)

    elif tickers:
        # Stock analysis with expert panel
        preset = cl.user_session.get("current_preset", "Quick Analysis")
        logger.info(f"Starting expert panel for tickers={tickers}, preset={preset}")
        try:
            responses = await run_expert_panel(
                question=user_input,
                tickers=tickers,
                preset=preset
            )
            logger.info(f"run_expert_panel returned {len(responses)} responses")
            await display_expert_responses(responses, user_input)
            logger.info("display_expert_responses completed")
        except Exception as e:
            logger.error(f"Expert panel failed: {e}", exc_info=True)
            await cl.Message(content=f"**Error:** Analysis failed - {e}").send()

    else:
        # General question without specific tickers
        from services.llm_router import get_llm_router
        router = get_llm_router()

        response_content = ""
        async with cl.Step(name="Answering", type="run") as step:
            for chunk in router.call_expert_stream(
                prompt=user_input,
                system="You are a knowledgeable stock market analyst. Answer questions about stocks, "
                       "markets, and investing. Be factual and balanced. Always note that this is not "
                       "financial advice."
            ):
                if chunk.get("type") == "chunk":
                    response_content += chunk.get("content", "")
            step.output = "Done"

        await cl.Message(content=response_content).send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
