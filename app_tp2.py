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

import json
import re

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
# LLM-Guided Intake Conversation
# =============================================================================

async def extract_trip_info(user_message: str, existing_info: dict) -> dict:
    """Use LLM to extract trip details from natural language."""
    from services.llm_router import get_llm_router
    router = get_llm_router()

    prompt = f"""Extract trip details from this user message. Return ONLY valid JSON, no markdown.

Current known info: {json.dumps(existing_info)}

User message: "{user_message}"

Return JSON with these fields (use null if not mentioned):
{{
  "destination": "city/country or null",
  "dates": "date range string or null",
  "duration_days": number or null,
  "travelers": "description or null",
  "budget": number or null
}}

Examples:
- "Barcelona next month with my wife" -> {{"destination": "Barcelona, Spain", "dates": "next month", "travelers": "2 adults (couple)", "budget": null, "duration_days": null}}
- "around 3000 dollars" -> {{"destination": null, "dates": null, "travelers": null, "budget": 3000, "duration_days": null}}
- "a week in mid-January" -> {{"destination": null, "dates": "mid-January", "duration_days": 7, "travelers": null, "budget": null}}

Return ONLY the JSON object, nothing else."""

    try:
        response_text = ""
        for chunk in router.call_expert_stream(
            prompt=prompt,
            system="You are a JSON extraction assistant. Return only valid JSON, no explanation."
        ):
            if chunk.get("type") == "chunk":
                response_text += chunk.get("content", "")

        # Parse JSON from response
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            extracted = json.loads(json_match.group())
        else:
            extracted = json.loads(response_text.strip())

        # Merge with existing info (don't overwrite with nulls)
        result = existing_info.copy()
        for key, value in extracted.items():
            if value is not None:
                result[key] = value

        return result

    except Exception as e:
        logger.error(f"Trip info extraction failed: {e}")
        return existing_info


def get_next_question(trip_info: dict) -> Optional[str]:
    """Determine what essential info is still missing."""
    if not trip_info.get("destination"):
        return "Where would you like to go? 🌍"
    if not trip_info.get("dates") and not trip_info.get("duration_days"):
        return "When are you thinking of traveling? 📅"
    if not trip_info.get("travelers"):
        return "Who's joining you on this trip? 👥"
    if not trip_info.get("budget"):
        return "What's your approximate budget for this trip? 💰"
    return None  # All essentials collected!


def format_trip_summary(trip_info: dict) -> str:
    """Format collected trip info as a nice summary."""
    dest = trip_info.get("destination", "TBD")
    dates = trip_info.get("dates", "")
    duration = trip_info.get("duration_days")
    travelers = trip_info.get("travelers", "TBD")
    budget = trip_info.get("budget")

    date_str = dates if dates else f"{duration} days" if duration else "TBD"
    budget_str = f"${budget:,}" if budget else "TBD"

    return f"""**Here's what I've got:**

📍 **Destination:** {dest}
📅 **When:** {date_str}
👥 **Travelers:** {travelers}
💰 **Budget:** {budget_str}

Does this look right?"""


async def handle_intake_message(message: str):
    """Handle messages during the intake conversation phase."""
    trip_info = cl.user_session.get("trip_info", {})

    # Show spinner while extracting info
    async with cl.Step(name="Trip Details", type="run") as step:
        trip_info = await extract_trip_info(message, trip_info)
        cl.user_session.set("trip_info", trip_info)

        # Show what we extracted
        extracted = []
        if trip_info.get("destination"):
            extracted.append(f"📍 {trip_info['destination']}")
        if trip_info.get("dates") or trip_info.get("duration_days"):
            date_info = trip_info.get("dates") or f"{trip_info.get('duration_days')} days"
            extracted.append(f"📅 {date_info}")
        if trip_info.get("travelers"):
            extracted.append(f"👥 {trip_info['travelers']}")
        if trip_info.get("budget"):
            extracted.append(f"💰 ${trip_info['budget']:,}")

        step.output = " | ".join(extracted) if extracted else "Processing..."

    # Check what's still needed
    next_question = get_next_question(trip_info)

    if next_question:
        # Still need more info
        await cl.Message(content=next_question).send()
    else:
        # All essentials collected - show confirmation
        await show_trip_confirmation(trip_info)


async def show_trip_confirmation(trip_info: dict):
    """Show trip summary and ask for confirmation."""
    summary = format_trip_summary(trip_info)

    actions = [
        cl.Action(name="plan_trip", label="✅ Yes, plan my trip!", value="plan", payload={"action": "plan"}),
        cl.Action(name="adjust_trip", label="✏️ Let me adjust", value="adjust", payload={"action": "adjust"}),
    ]

    await cl.Message(content=summary, actions=actions).send()


def convert_trip_info_to_config(trip_info: dict) -> dict:
    """Convert intake trip_info to the trip_config format used by handle_plan_trip."""
    from datetime import datetime, timedelta

    # Parse dates from natural language
    dates_str = trip_info.get("dates", "")
    duration = trip_info.get("duration_days")

    # Default to 30 days from now, 7 days duration
    departure = date.today() + timedelta(days=30)
    return_date = departure + timedelta(days=duration or 7)

    # Try to parse specific dates if provided
    if dates_str:
        import re
        dates_lower = dates_str.lower()
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
            "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }

        # Try to extract day, month, year from various formats
        # Patterns: "26Dec2025", "Dec 26 2025", "December 26, 2025", "26 December 2025"
        day = None
        month_num = None
        year = None

        # Find month
        for month_name, m_num in months.items():
            if month_name in dates_lower:
                month_num = m_num
                break

        if month_num:
            # Extract numbers from the string
            numbers = re.findall(r'\d+', dates_str)
            for num_str in numbers:
                num = int(num_str)
                if num >= 2024 and num <= 2030:  # Year
                    year = num
                elif num >= 1 and num <= 31 and day is None:  # Day
                    day = num

            # Set defaults if not found
            if year is None:
                year = date.today().year
                if month_num < date.today().month:
                    year += 1
            if day is None:
                day = 15  # Mid-month default

            try:
                departure = date(year, month_num, day)
                return_date = departure + timedelta(days=duration or 7)
            except ValueError:
                # Invalid date, use defaults
                pass

    # Map travelers string to select value
    travelers_str = trip_info.get("travelers", "2 adults")
    travelers_lower = travelers_str.lower() if travelers_str else ""
    if "1" in travelers_lower or "solo" in travelers_lower or "alone" in travelers_lower:
        travelers = "1 adult"
    elif "child" in travelers_lower or "kid" in travelers_lower or "family" in travelers_lower:
        if "2" in travelers_lower:
            travelers = "2 adults + 2 children"
        else:
            travelers = "2 adults + 1 child"
    elif "group" in travelers_lower or "4" in travelers_lower or "5" in travelers_lower:
        travelers = "Group (4+)"
    else:
        travelers = "2 adults"

    return {
        "destination": trip_info.get("destination", ""),
        "origin": "",  # Not collected in intake yet
        "departure": departure.isoformat(),
        "return_date": return_date.isoformat(),
        "travelers": travelers,
        "budget": trip_info.get("budget", 5000),
        "preset": "Quick Trip Planning"
    }


@cl.action_callback("plan_trip")
async def on_plan_trip_action(action: cl.Action):
    """Handle 'Yes, plan my trip!' button click."""
    trip_info = cl.user_session.get("trip_info", {})
    trip_config = convert_trip_info_to_config(trip_info)

    cl.user_session.set("trip_config", trip_config)
    cl.user_session.set("intake_mode", False)

    await cl.Message(content="Great! Let me get my expert team working on your trip... 🧳").send()
    await handle_plan_trip(trip_config)


@cl.action_callback("adjust_trip")
async def on_adjust_trip_action(action: cl.Action):
    """Handle 'Let me adjust' button click."""
    await cl.Message(
        content="No problem! What would you like to change?\n\n"
                "You can tell me naturally (e.g., 'Actually, make it Tokyo' or 'Budget should be $4000'), "
                "or use the ⚙️ settings panel for precise control."
    ).send()


@cl.action_callback("ask_expert")
async def on_ask_expert(action: cl.Action):
    """Handle quick expert consultation button click."""
    expert_name = action.payload.get("expert") or action.value
    trip_config = cl.user_session.get("trip_config", {})
    trip_data = cl.user_session.get("trip_data")

    destination = trip_config.get("destination", "your destination")
    context = trip_data.get("summary", "") if trip_data else ""

    # Get previous expert responses for context
    expert_responses = cl.user_session.get("expert_responses", {})
    if expert_responses:
        context += "\n\n## Previous Expert Recommendations:\n"
        for expert, response in expert_responses.items():
            context += f"\n**{expert}:** {response[:500]}...\n"

    # Stream expert response
    icon = EXPERT_ICONS.get(expert_name, "🧭")
    msg = cl.Message(content="", author=f"{icon} {expert_name}")
    expert_info = TRAVEL_EXPERTS.get(expert_name, {})
    role = expert_info.get("role", expert_name)
    await msg.stream_token(f"## {icon} {expert_name}\n*{role}*\n\n")

    full_response = ""
    try:
        for chunk in call_travel_expert_stream(
            persona_name=expert_name,
            clinical_question=f"Provide your expert advice for a trip to {destination}",
            evidence_context=context,
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

    # Store the new expert response
    expert_responses[expert_name] = full_response
    cl.user_session.set("expert_responses", expert_responses)


@cl.action_callback("export_excel")
async def on_export_excel(action: cl.Action):
    """Export trip plan to Excel."""
    from services.excel_export_service import export_travel_plan_to_excel
    import tempfile

    trip_config = cl.user_session.get("trip_config", {})
    trip_data = cl.user_session.get("trip_data", {})
    expert_responses = cl.user_session.get("expert_responses", {})

    destination = trip_config.get("destination", "trip")
    safe_dest = "".join(c for c in destination if c.isalnum() or c in " -_").strip()[:30]

    await cl.Message(content="Generating your Excel trip plan...").send()

    try:
        # Build question and recommendation for parser
        question = f"Trip to {destination}, {trip_config.get('travelers', '2 adults')}, budget ${trip_config.get('budget', 5000)}"
        recommendation = trip_data.get("summary", "") if trip_data else ""

        # Convert expert_responses to expected format
        expert_data = {}
        for expert, response in expert_responses.items():
            expert_data[expert] = {"content": response} if isinstance(response, str) else response

        excel_buffer = export_travel_plan_to_excel(
            question=question,
            recommendation=recommendation,
            expert_responses=expert_data,
            trip_data=trip_data
        )

        # Save to temp file
        file_path = f"/tmp/Trip_Plan_{safe_dest}.xlsx"
        with open(file_path, "wb") as f:
            f.write(excel_buffer.getvalue())

        file = cl.File(path=file_path, name=f"Trip_Plan_{safe_dest}.xlsx")
        await cl.Message(
            content=f"Here's your trip plan for **{destination}**:",
            elements=[file]
        ).send()

    except Exception as e:
        logger.error(f"Excel export failed: {e}")
        await cl.Message(content=f"Sorry, I couldn't generate the Excel file: {e}").send()


@cl.action_callback("export_word")
async def on_export_word(action: cl.Action):
    """Export trip plan to Word document."""
    from services.word_export_service import export_travel_plan_to_word

    trip_config = cl.user_session.get("trip_config", {})
    trip_data = cl.user_session.get("trip_data", {})
    expert_responses = cl.user_session.get("expert_responses", {})

    destination = trip_config.get("destination", "trip")
    safe_dest = "".join(c for c in destination if c.isalnum() or c in " -_").strip()[:30]

    await cl.Message(content="Generating your Word trip plan...").send()

    try:
        word_buffer = export_travel_plan_to_word(
            trip_config=trip_config,
            trip_data=trip_data,
            expert_responses=expert_responses
        )

        # Save to temp file
        file_path = f"/tmp/Trip_Plan_{safe_dest}.docx"
        with open(file_path, "wb") as f:
            f.write(word_buffer.getvalue())

        file = cl.File(path=file_path, name=f"Trip_Plan_{safe_dest}.docx")
        await cl.Message(
            content=f"Here's your trip plan for **{destination}**:",
            elements=[file]
        ).send()

    except Exception as e:
        logger.error(f"Word export failed: {e}")
        await cl.Message(content=f"Sorry, I couldn't generate the Word file: {e}").send()


# =============================================================================
# SQLite Persistence Setup
# =============================================================================

# Database path - in project directory
DB_PATH = Path(__file__).parent / "data" / "travel_planner.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Connection string for SQLite with aiosqlite
DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# Initialize SQLAlchemy data layer for persistence
# NOTE: Disabled temporarily due to missing table errors in Chainlit 2.3.0
# try:
#     from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
#
#     cl_data._data_layer = SQLAlchemyDataLayer(
#         conninfo=DB_URL,
#         ssl_require=False,
#     )
#     logger.info(f"SQLite persistence enabled: {DB_PATH}")
# except ImportError as e:
#     logger.warning(f"SQLAlchemy data layer not available: {e}")
#     logger.warning("Install with: pip install aiosqlite sqlalchemy")
# except Exception as e:
#     logger.warning(f"Failed to initialize SQLite persistence: {e}")
#     logger.warning("Conversations will not persist across sessions")
logger.info("Running without persistence (disabled for debugging)")

# =============================================================================
# Initialize Services
# =============================================================================

travel_service = TravelDataService()


@cl.on_chat_start
async def start():
    """Initialize chat session with conversational intake and settings fallback."""

    # Set up ChatSettings for power users (fallback option)
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

    # NEW: Initialize intake mode
    cl.user_session.set("intake_mode", True)
    cl.user_session.set("trip_info", {})

    # Conversational welcome
    await cl.Message(
        content="""# ✈️ Travel Planner

Hi! I'm your AI travel planning assistant with a team of expert advisors.

**Tell me about your trip** - where are you dreaming of going? 🌍

*(Or click the ⚙️ settings icon to fill in details manually)*"""
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
    intake_mode = cl.user_session.get("intake_mode", False)

    user_input = message.content.lower().strip()

    # Check for "plan my trip" command - this bypasses intake mode (power user)
    if "plan" in user_input and ("trip" in user_input or "my" in user_input):
        # If in intake mode with collected info, use that; otherwise use form settings
        if intake_mode:
            trip_info = cl.user_session.get("trip_info", {})
            if trip_info.get("destination"):
                # Convert trip_info to trip_config format
                trip_config = convert_trip_info_to_config(trip_info)
                cl.user_session.set("trip_config", trip_config)
        cl.user_session.set("intake_mode", False)
        await handle_plan_trip(trip_config)
        return

    # If in intake mode, handle conversationally (before other keyword checks)
    if intake_mode:
        await handle_intake_message(message.content)
        return

    # Check for on-demand car rental request (only after trip is planned)
    if "car" in user_input and ("rental" in user_input or "rent" in user_input):
        await handle_car_rental_request(trip_config, trip_data)
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
    async with cl.Step(name="Travel Data", type="tool") as step:
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

            step.output = f"Fetched: Flights, Hotels, Dining, Weather"
        except Exception as e:
            logger.error(f"Travel data fetch failed: {e}")
            trip_data = {"summary": f"Trip to {destination}"}
            step.output = f"Some data unavailable: {e}"

    # Show data summary if available
    if trip_data.get("summary"):
        await cl.Message(
            content=f"## Real-Time Data\n\n{trip_data['summary'][:2000]}"
        ).send()

    # Fetch Google Search context for experts that need real-time data
    search_contexts = {}

    # Safety Expert always gets search context (advisories, visa, health)
    if "Safety Expert" in selected_experts:
        async with cl.Step(name="Travel Advisories", type="tool") as step:
            try:
                advisories = travel_service.fetch_safety_advisories(destination)
                if advisories:
                    search_contexts["Safety Expert"] = f"\n\n## Current Travel Advisories (Google Search)\n{advisories}"
                    step.output = "Found current travel advisories"
                else:
                    step.output = "No advisories available"
            except Exception as e:
                logger.warning(f"Safety advisory fetch failed: {e}")
                step.output = f"Advisory fetch failed: {e}"

    # For near-term trips (< 30 days), fetch current events for all experts
    days_until_departure = (departure - date.today()).days
    if days_until_departure <= 30:
        async with cl.Step(name="Current Events", type="tool") as step:
            try:
                events = travel_service.fetch_current_events(destination, departure)
                if events:
                    # Add events context to all experts for near-term trips
                    for expert_name in selected_experts:
                        if expert_name not in search_contexts:
                            search_contexts[expert_name] = ""
                        search_contexts[expert_name] += f"\n\n## Current Events & News (Google Search)\n{events}"
                    step.output = f"Found events/news for {destination}"
                else:
                    step.output = "No major events found"
            except Exception as e:
                logger.warning(f"Current events fetch failed: {e}")
                step.output = f"Events fetch failed: {e}"

    # Run experts and stream responses
    expert_responses = {}

    for expert_name in selected_experts:
        icon = EXPERT_ICONS.get(expert_name, "🧭")
        msg = cl.Message(content="", author=f"{icon} {expert_name}")

        expert_info = TRAVEL_EXPERTS.get(expert_name, {})
        role = expert_info.get("role", expert_name)

        # Add header with icon
        await msg.stream_token(f"## {icon} {expert_name}\n*{role}*\n\n")

        # Build context with any Google Search data for this expert
        expert_context = trip_data.get("summary", "")
        if expert_name in search_contexts:
            expert_context += search_contexts[expert_name]

        # Stream expert response
        full_response = ""
        try:
            for chunk in call_travel_expert_stream(
                persona_name=expert_name,
                clinical_question=f"Plan a trip to {destination}",
                evidence_context=expert_context,
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

    # Summary message with export buttons
    export_actions = [
        cl.Action(name="export_excel", label="Export to Excel", value="excel", payload={"format": "excel"}),
        cl.Action(name="export_word", label="Export to Word", value="word", payload={"format": "word"}),
    ]
    await cl.Message(
        content="---\n\n**Trip planning complete!** Ask me any follow-up questions about your trip, "
                "or ask a specific expert for more details (e.g., \"Ask Food & Dining Expert about restaurants\").\n\n"
                "*Need a rental car? Just say \"show car rentals\"!*",
        actions=export_actions
    ).send()

    # Suggest other experts not in the current panel
    all_experts = list(TRAVEL_EXPERTS.keys())
    other_experts = [e for e in all_experts if e not in selected_experts]

    if other_experts:
        # Build suggestion message with expert specialties
        expert_suggestions = []
        for expert in other_experts:
            icon = EXPERT_ICONS.get(expert, "🧭")
            specialty = TRAVEL_EXPERTS[expert].get("specialty", "")
            # Take first part of specialty for brevity
            short_specialty = specialty.split(",")[0] if specialty else ""
            expert_suggestions.append(f"- **{icon} {expert}**: {short_specialty}")

        # Create action buttons for quick access to other experts
        expert_actions = [
            cl.Action(
                name="ask_expert",
                label=f"{EXPERT_ICONS.get(expert, '🧭')} {expert.split()[0]}",
                value=expert,
                payload={"expert": expert}
            )
            for expert in other_experts[:4]  # Limit to 4 buttons
        ]

        await cl.Message(
            content=f"**Would you like to hear from other experts?**\n\n"
                    + "\n".join(expert_suggestions) + "\n\n"
                    "*Click a button below or say \"Ask [Expert Name] about...\"*",
            actions=expert_actions
        ).send()


async def handle_car_rental_request(trip_config: Dict, trip_data: Optional[Dict]):
    """Fetch and display car rental options on demand."""
    destination = trip_config.get("destination", "")

    if not destination:
        await cl.Message(
            content="Please tell me your destination first so I can find car rentals for you."
        ).send()
        return

    # Parse dates
    try:
        departure = date.fromisoformat(trip_config.get("departure", ""))
        return_date = date.fromisoformat(trip_config.get("return_date", ""))
    except (ValueError, TypeError):
        departure = date.today() + timedelta(days=30)
        return_date = departure + timedelta(days=7)

    await cl.Message(content=f"Looking for car rentals in {destination}...").send()

    async with cl.Step(name="Car Rentals", type="tool") as step:
        try:
            car_data = travel_service.fetch_car_rentals_on_demand(
                destination=destination,
                pickup=departure,
                dropoff=return_date
            )

            if car_data and car_data != "Car rental data unavailable. Please try again later.":
                step.output = "Found car rental options"
                await cl.Message(
                    content=f"## Car Rental Options\n\n{car_data}"
                ).send()
            else:
                step.output = "No car rentals available"
                await cl.Message(
                    content="Sorry, I couldn't find car rental options for this destination. "
                           "You may want to check directly with rental agencies."
                ).send()
        except Exception as e:
            logger.error(f"Car rental fetch failed: {e}")
            step.output = f"Error: {e}"
            await cl.Message(
                content="Sorry, I had trouble fetching car rental data. Please try again later."
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
