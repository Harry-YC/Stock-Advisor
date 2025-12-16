"""
Travel Planner - Chainlit App

A conversational travel planning assistant with AI travel experts.
Uses ChatSettings for trip configuration and streaming responses.
SQLite persistence for conversation history.

Run with: chainlit run app.py
"""

import chainlit as cl
import chainlit.data as cl_data
from chainlit.input_widget import TextInput, Select, Slider
from datetime import date, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import logging
import os

from config import settings
from services.travel_data_service import TravelDataService
from travel.travel_personas import (
    TRAVEL_EXPERTS,
    TRAVEL_PRESETS,
    EXPERT_ICONS,
    get_default_travel_experts,
    call_travel_expert_stream
)

logger = logging.getLogger(__name__)

# =============================================================================
# SQLite Persistence Setup
# =============================================================================

# Database path - in project directory
DB_PATH = Path(__file__).parent / "data" / "travel_planner.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Connection string for SQLite with aiosqlite
DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# Initialize SQLAlchemy data layer for persistence
try:
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

    cl_data._data_layer = SQLAlchemyDataLayer(
        conninfo=DB_URL,
        ssl_require=False,
    )
    logger.info(f"SQLite persistence enabled: {DB_PATH}")
except ImportError as e:
    logger.warning(f"SQLAlchemy data layer not available: {e}")
    logger.warning("Install with: pip install aiosqlite sqlalchemy")
except Exception as e:
    logger.warning(f"Failed to initialize SQLite persistence: {e}")
    logger.warning("Conversations will not persist across sessions")

# =============================================================================
# Initialize Services
# =============================================================================

travel_service = TravelDataService()


@cl.on_chat_start
async def start():
    """Initialize chat session with trip configuration settings."""

    # Set up ChatSettings for trip form
    settings_widgets = [
        TextInput(
            id="destination",
            label="Destination",
            placeholder="Paris, France",
            initial=""
        ),
        TextInput(
            id="origin",
            label="Origin (for flights)",
            placeholder="San Francisco, CA",
            initial=""
        ),
        TextInput(
            id="departure",
            label="Departure Date",
            placeholder="YYYY-MM-DD",
            initial=(date.today() + timedelta(days=30)).isoformat()
        ),
        TextInput(
            id="return_date",
            label="Return Date",
            placeholder="YYYY-MM-DD",
            initial=(date.today() + timedelta(days=37)).isoformat()
        ),
        Select(
            id="travelers",
            label="Travelers",
            values=["1 adult", "2 adults", "2 adults + 1 child", "2 adults + 2 children", "Group (4+)"],
            initial_value="2 adults"
        ),
        Slider(
            id="budget",
            label="Budget (USD)",
            min=500,
            max=20000,
            step=500,
            initial=5000
        ),
        Select(
            id="preset",
            label="Expert Panel",
            values=list(TRAVEL_PRESETS.keys()),
            initial_value="Quick Trip Planning"
        ),
    ]

    await cl.ChatSettings(settings_widgets).send()

    # Initialize session state
    cl.user_session.set("trip_config", {})
    cl.user_session.set("trip_data", None)
    cl.user_session.set("expert_responses", {})

    # Welcome message
    await cl.Message(
        content="""# Travel Planner

Welcome! I'll help you plan your perfect trip with AI travel experts.

**To get started:**
1. Click the **settings icon** (gear) to configure your trip details
2. Set your destination, dates, travelers, and budget
3. Say **"Plan my trip"** to get expert recommendations

**Available commands:**
- "Plan my trip" - Get comprehensive travel recommendations
- "Ask [expert name]" - Consult a specific expert (e.g., "Ask Budget Advisor about cheap flights")
- Ask any follow-up questions about your trip!

**Expert Panel Presets:**
- Quick Trip Planning (4 core experts)
- Adventure Travel (outdoor focus)
- Budget Backpacking (value focus)
- Cultural Immersion (local experiences)
- Family Vacation (safety & logistics)
- Full Panel (all 8 experts)
"""
    ).send()


@cl.on_settings_update
async def on_settings_update(settings_dict: Dict):
    """Store updated trip configuration."""
    cl.user_session.set("trip_config", settings_dict)

    # Confirm update
    destination = settings_dict.get("destination", "")
    if destination:
        await cl.Message(
            content=f"Trip settings updated: **{destination}** | "
                    f"Budget: ${settings_dict.get('budget', 5000):,} | "
                    f"Dates: {settings_dict.get('departure', 'TBD')} to {settings_dict.get('return_date', 'TBD')}"
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages."""

    trip_config = cl.user_session.get("trip_config", {})
    trip_data = cl.user_session.get("trip_data")

    user_input = message.content.lower().strip()

    # Check for "plan my trip" command
    if "plan" in user_input and ("trip" in user_input or "my" in user_input):
        await handle_plan_trip(trip_config)
        return

    # Check for specific expert request
    if user_input.startswith("ask "):
        await handle_expert_question(message.content, trip_config, trip_data)
        return

    # General follow-up question
    await handle_followup(message.content, trip_config, trip_data)


async def handle_plan_trip(trip_config: Dict):
    """Execute full trip planning with expert panel."""

    destination = trip_config.get("destination", "").strip()
    if not destination:
        await cl.Message(
            content="Please set your **destination** in the settings panel first (click the gear icon)."
        ).send()
        return

    # Parse dates
    try:
        departure = date.fromisoformat(trip_config.get("departure", ""))
        return_date = date.fromisoformat(trip_config.get("return_date", ""))
    except (ValueError, TypeError):
        await cl.Message(
            content="Please set valid **departure and return dates** in the settings (format: YYYY-MM-DD)."
        ).send()
        return

    # Get expert preset
    preset_name = trip_config.get("preset", "Quick Trip Planning")
    preset = TRAVEL_PRESETS.get(preset_name, TRAVEL_PRESETS["Quick Trip Planning"])
    selected_experts = preset["experts"]

    # Show planning started
    await cl.Message(
        content=f"## Planning your trip to {destination}\n\n"
                f"**Dates:** {departure.strftime('%b %d')} - {return_date.strftime('%b %d, %Y')}\n"
                f"**Budget:** ${trip_config.get('budget', 5000):,}\n"
                f"**Experts:** {', '.join(selected_experts)}\n\n"
                f"Fetching real-time data..."
    ).send()

    # Fetch travel data
    async with cl.Step(name="Fetching Travel Data", type="tool") as step:
        try:
            trip_data = travel_service.fetch_travel_data(
                destination=destination,
                origin=trip_config.get("origin") or None,
                departure_date=departure,
                return_date=return_date,
                travelers=trip_config.get("travelers", "2 adults"),
                budget=int(trip_config.get("budget", 5000))
            )
            cl.user_session.set("trip_data", trip_data)

            step.output = f"Fetched: Weather, Flights, Hotels, Car Rentals"
        except Exception as e:
            logger.error(f"Travel data fetch failed: {e}")
            trip_data = {"summary": f"Trip to {destination}"}
            step.output = f"Some data unavailable: {e}"

    # Show data summary if available
    if trip_data.get("summary"):
        await cl.Message(
            content=f"## Real-Time Data\n\n{trip_data['summary'][:2000]}"
        ).send()

    # Run experts in parallel and stream responses
    expert_responses = {}

    for expert_name in selected_experts:
        icon = EXPERT_ICONS.get(expert_name, "🧭")
        msg = cl.Message(content="", author=f"{icon} {expert_name}")

        expert_info = TRAVEL_EXPERTS.get(expert_name, {})
        role = expert_info.get("role", expert_name)

        # Add header with icon
        await msg.stream_token(f"## {icon} {expert_name}\n*{role}*\n\n")

        # Stream expert response
        full_response = ""
        try:
            for chunk in call_travel_expert_stream(
                persona_name=expert_name,
                clinical_question=f"Plan a trip to {destination}",
                evidence_context=trip_data.get("summary", ""),
                model=settings.EXPERT_MODEL,
                openai_api_key=settings.GEMINI_API_KEY
            ):
                if chunk.get("type") == "chunk":
                    content = chunk.get("content", "")
                    full_response += content
                    await msg.stream_token(content)
                elif chunk.get("type") == "error":
                    await msg.stream_token(f"\n\n*Error: {chunk.get('content')}*")
        except Exception as e:
            logger.error(f"Expert {expert_name} failed: {e}")
            await msg.stream_token(f"\n\n*Error getting response: {e}*")

        await msg.send()
        expert_responses[expert_name] = full_response

    # Store responses
    cl.user_session.set("expert_responses", expert_responses)

    # Summary message
    await cl.Message(
        content="---\n\n**Trip planning complete!** Ask me any follow-up questions about your trip, "
                "or ask a specific expert for more details (e.g., \"Ask Food & Dining Expert about restaurants\")."
    ).send()


async def handle_expert_question(user_input: str, trip_config: Dict, trip_data: Optional[Dict]):
    """Handle direct question to a specific expert."""

    # Extract expert name from input
    input_lower = user_input.lower()
    expert_name = None
    question = user_input

    for name in TRAVEL_EXPERTS.keys():
        if name.lower() in input_lower:
            expert_name = name
            # Extract the question part
            idx = input_lower.find(name.lower())
            question = user_input[idx + len(name):].strip()
            if question.startswith("about"):
                question = question[5:].strip()
            break

    if not expert_name:
        await cl.Message(
            content=f"I couldn't identify which expert you want to consult. Available experts:\n\n"
                    + "\n".join(f"- **{name}**: {info['specialty']}" for name, info in TRAVEL_EXPERTS.items())
        ).send()
        return

    destination = trip_config.get("destination", "your destination")
    context = trip_data.get("summary", "") if trip_data else ""

    # Stream expert response
    icon = EXPERT_ICONS.get(expert_name, "🧭")
    msg = cl.Message(content="", author=f"{icon} {expert_name}")
    expert_info = TRAVEL_EXPERTS.get(expert_name, {})
    await msg.stream_token(f"## {icon} {expert_name}\n*Answering: {question or 'your question'}*\n\n")

    try:
        for chunk in call_travel_expert_stream(
            persona_name=expert_name,
            clinical_question=question or f"Provide detailed advice for a trip to {destination}",
            evidence_context=context,
            model=settings.EXPERT_MODEL,
            openai_api_key=settings.GEMINI_API_KEY
        ):
            if chunk.get("type") == "chunk":
                await msg.stream_token(chunk.get("content", ""))
            elif chunk.get("type") == "error":
                await msg.stream_token(f"\n\n*Error: {chunk.get('content')}*")
    except Exception as e:
        await msg.stream_token(f"\n\n*Error: {e}*")

    await msg.send()


def detect_best_expert(question: str) -> Optional[str]:
    """Detect the best expert to answer a question based on keywords."""
    question_lower = question.lower()

    # Score each expert based on keyword matches
    scores = {}
    for expert_name, expert_info in TRAVEL_EXPERTS.items():
        keywords = expert_info.get("specialty_keywords", [])
        score = sum(1 for kw in keywords if kw.lower() in question_lower)
        if score > 0:
            scores[expert_name] = score

    if scores:
        # Return expert with highest score
        return max(scores, key=scores.get)
    return None


async def handle_followup(question: str, trip_config: Dict, trip_data: Optional[Dict]):
    """Handle general follow-up questions by routing to the best expert."""

    destination = trip_config.get("destination", "")
    context = ""

    if trip_data:
        context = trip_data.get("summary", "")[:3000]

    # Get previous expert responses for context
    expert_responses = cl.user_session.get("expert_responses", {})
    if expert_responses:
        context += "\n\n## Previous Expert Recommendations:\n"
        for expert, response in expert_responses.items():
            context += f"\n**{expert}:** {response[:500]}...\n"

    # Detect best expert for this question
    best_expert = detect_best_expert(question)

    if best_expert:
        # Route to specific expert with persona
        icon = EXPERT_ICONS.get(best_expert, "🧭")
        expert_info = TRAVEL_EXPERTS.get(best_expert, {})
        role = expert_info.get("role", best_expert)

        msg = cl.Message(content="", author=f"{icon} {best_expert}")
        await msg.stream_token(f"## {icon} {best_expert}\n*{role}*\n\n")

        try:
            for chunk in call_travel_expert_stream(
                persona_name=best_expert,
                clinical_question=question,
                evidence_context=context,
                model=settings.EXPERT_MODEL,
                openai_api_key=settings.GEMINI_API_KEY
            ):
                if chunk.get("type") == "chunk":
                    await msg.stream_token(chunk.get("content", ""))
                elif chunk.get("type") == "error":
                    await msg.stream_token(f"\n\n*Error: {chunk.get('content')}*")
        except Exception as e:
            await msg.stream_token(f"\n\n*Error: {e}*")

        await msg.send()
    else:
        # No clear expert match - use general assistant but suggest experts
        msg = cl.Message(content="", author="🧭 Travel Assistant")

        try:
            from services.llm_router import get_llm_router
            router = get_llm_router()

            system = """You are a helpful travel planning assistant. Answer the user's question
            based on the trip context and previous expert recommendations. Be specific and practical.
            At the end of your response, suggest which specific expert they could ask for more detailed advice.
            Available experts: Budget Advisor, Logistics Planner, Safety Expert, Weather Analyst,
            Local Culture Guide, Food & Dining Expert, Activity Curator, Accommodation Specialist."""

            prompt = f"Trip: {destination or 'Not specified'}\n\nContext:\n{context}\n\nQuestion: {question}"

            for chunk in router.call_expert_stream(prompt=prompt, system=system):
                if chunk.get("type") == "chunk":
                    await msg.stream_token(chunk.get("content", ""))
                elif chunk.get("type") == "error":
                    await msg.stream_token(f"\n\n*Error: {chunk.get('content')}*")
        except Exception as e:
            await msg.stream_token(f"I apologize, I encountered an error: {e}\n\n"
                                   "Try asking a specific expert, e.g., \"Ask Budget Advisor about costs\"")

        await msg.send()


# Chainlit lifecycle hooks for persistence (optional - requires LiteralAI)
@cl.on_chat_resume
async def on_chat_resume(thread):
    """Resume a previous chat session."""
    # Restore session state from thread metadata if available
    if thread.metadata:
        cl.user_session.set("trip_config", thread.metadata.get("trip_config", {}))
        cl.user_session.set("trip_data", thread.metadata.get("trip_data"))
        cl.user_session.set("expert_responses", thread.metadata.get("expert_responses", {}))

    await cl.Message(
        content="Welcome back! Your previous trip planning session has been restored. "
                "Continue asking questions or say \"Plan my trip\" to start fresh."
    ).send()


if __name__ == "__main__":
    # For local development, run with: chainlit run app.py
    pass
