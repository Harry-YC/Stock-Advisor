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
    preset_config = STOCK_PRESETS.get(preset, STOCK_PRESETS["Quick Analysis"])
    expert_names = preset_config["experts"]

    responses = {}
    primary_ticker = tickers[0] if tickers else None

    # Fetch stock data for context
    evidence_context = ""
    if primary_ticker:
        stock_context = fetch_stock_data(
            primary_ticker,
            include_quote=True,
            include_financials=True,
            include_news=True
        )
        evidence_context = stock_context.to_prompt_context()

    # Show progress message
    progress_msg = await cl.Message(
        content=f"**Consulting {len(expert_names)} experts...**\n" +
                " | ".join([f"⏳ {EXPERT_ICONS.get(e, '📊')} {e}" for e in expert_names])
    ).send()

    # Run experts in parallel
    async def call_expert_async(expert_name: str) -> tuple[str, str]:
        try:
            full_response = ""
            async for chunk in stream_expert_response(
                expert_name, question, evidence_context, kol_context
            ):
                full_response += chunk
            return expert_name, full_response
        except Exception as e:
            logger.error(f"Expert {expert_name} failed: {e}")
            return expert_name, f"*Error: {str(e)}*"

    # Execute all experts concurrently
    tasks = [call_expert_async(name) for name in expert_names]
    results = await asyncio.gather(*tasks)

    for expert_name, response in results:
        responses[expert_name] = response

    # Update progress to complete
    await progress_msg.remove()

    return responses


async def stream_expert_response(
    expert_name: str,
    question: str,
    evidence_context: str,
    kol_context: str = ""
):
    """Stream expert response chunks."""
    for chunk in call_stock_expert_stream(
        persona_name=expert_name,
        question=question,
        evidence_context=evidence_context,
        kol_context=kol_context
    ):
        if chunk.get("type") == "chunk":
            yield chunk.get("content", "")


async def display_expert_responses(responses: Dict[str, str], question: str):
    """Display expert responses with formatting."""
    for expert_name, response in responses.items():
        icon = EXPERT_ICONS.get(expert_name, "📊")
        await cl.Message(
            content=f"## {icon} {expert_name}\n\n{response}"
        ).send()

    # Store responses for follow-up
    cl.user_session.set("expert_responses", responses)
    cl.user_session.set("last_question", question)

    # Offer follow-up actions
    actions = [
        cl.Action(name="ask_followup", label="💬 Ask Follow-up", value="followup", payload={}),
        cl.Action(name="deep_dive", label="🔍 Deep Dive", value="deep", payload={}),
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

    for element in message.elements:
        if element.mime not in supported_types:
            continue

        # Validate size
        if element.path and os.path.exists(element.path):
            file_size = os.path.getsize(element.path)
            if file_size > MAX_IMAGE_SIZE:
                await cl.Message(
                    content=f"Image too large ({file_size/1024/1024:.1f}MB). Max 5MB allowed."
                ).send()
                continue

        # Analyze the screenshot
        async with cl.Step(name="KOL Screenshot Analysis", type="tool") as step:
            step.output = "Analyzing screenshot..."

            result = analyze_kol_screenshot(element.path)

            if not result["success"]:
                await cl.Message(
                    content=f"**Analysis failed:** {result['error']}"
                ).send()
                return

            ocr = result["ocr_result"]

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

**Ask me anything about stocks**, or upload a KOL screenshot for analysis.

_Examples:_
- "Why did NVDA fall yesterday?"
- "Analyze AAPL stock"
- "Compare TSLA and RIVN"
- [Upload screenshot of a stock tweet]

⚠️ *Not financial advice. Always do your own research.*"""
    ).send()


@cl.on_settings_update
async def on_settings_update(settings_dict: Dict):
    """Handle settings update."""
    ticker = settings_dict.get("ticker", "").upper().strip()
    preset = settings_dict.get("preset", "Quick Analysis")

    if ticker:
        cl.user_session.set("detected_tickers", [ticker])
        await cl.Message(
            content=f"Settings updated: **{ticker}** with {preset} preset"
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

    # Extract analysis request
    async with cl.Step(name="Understanding Request", type="run") as step:
        request = await extract_analysis_request(user_input)
        step.output = f"Detected: {request.get('question_type', 'analysis')}"

    tickers = request.get("tickers", [])
    question_type = request.get("question_type", "analysis")

    # Store detected tickers
    if tickers:
        cl.user_session.set("detected_tickers", tickers)

    # Handle different question types
    if question_type == "why_moved" and tickers:
        # Special handling for "why did X move" questions
        direction = request.get("direction", "moved")
        async with cl.Step(name="Market Search", type="tool") as step:
            explanation = search_why_stock_moved(tickers[0], direction)
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
        responses = await run_expert_panel(
            question=user_input,
            tickers=tickers,
            preset=preset
        )
        await display_expert_responses(responses, user_input)

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
