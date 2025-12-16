"""
Travel Planner - Expert Personas

8 expert roles for comprehensive travel planning:
1. Budget Advisor - Cost optimization, deals, budget allocation
2. Safety Expert - Travel advisories, medical insurance (top 3 providers + prices)
3. Local Culture Guide - Customs, etiquette, authentic experiences
4. Logistics Planner - Routes, connections, car rental (top 3 providers + prices)
5. Food & Dining Expert - Local cuisine, restaurants, dietary needs
6. Activity Curator - Tours, attractions, entertainment
7. Accommodation Specialist - Hotels, rentals, location strategy
8. Weather Analyst - Climate, best times to visit, packing
"""

from typing import Dict, Tuple, List, Optional

# Base context for all Travel Experts
TRAVEL_BASE_CONTEXT = (
    "You are a travel planning expert helping users plan their trips. "
    "Provide practical, actionable advice based on real travel data and experiences. "
    "Focus on: value for money, safety, authentic experiences, and logistics. "
    "Return clear, organized recommendations. "
    "Each point should be specific and actionable. "
    "\n\n"
    "RESPONSE GUIDELINES:\n"
    "- Be specific with prices, times, and locations when available\n"
    "- Include practical tips that travelers often overlook\n"
    "- Mention seasonal considerations when relevant\n"
    "- Flag any safety concerns or travel advisories\n"
    "- Suggest alternatives for different budgets when appropriate\n"
    "\n"
    "CONFIDENCE MARKERS - Use these to indicate reliability:\n"
    "- [VERIFIED] - Based on official sources, recent data, or confirmed information\n"
    "- [TYPICAL] - Based on general travel patterns and common experiences\n"
    "- [ESTIMATE] - Approximate values that may vary\n"
    "- [SEASONAL] - Varies significantly by time of year\n"
    "- [CHECK CURRENT] - Recommend verifying before travel (prices, hours, policies)\n"
)

# 8 Travel Expert Personas
TRAVEL_EXPERTS = {

    # ========================================================================
    # PLANNING & BUDGET (2 experts)
    # ========================================================================

    "Budget Advisor": {
        "role": "Travel Budget & Finance Expert",
        "specialty": "Cost optimization, deals, budget allocation, money-saving strategies",
        "perspective": (
            "Help travelers maximize value for money. Identify hidden costs, "
            "suggest money-saving alternatives, and create realistic budget breakdowns. "
            "Balance quality experiences with cost efficiency."
        ),
        "search_queries": [
            "budget travel tips destination",
            "cheap flights hotels deals",
            "travel cost breakdown expenses",
            "money saving travel hacks",
            "affordable alternatives activities"
        ],
        "topics": [
            "daily budget estimate (accommodation, food, transport, activities)",
            "money-saving tips specific to this destination",
            "best value accommodations in different price ranges",
            "hidden costs to watch out for (tourist taxes, tips, scams)",
            "best time to book for lowest prices",
            "free or low-cost activities and attractions",
            "payment tips (cash vs card, currency exchange)"
        ],
        "specialty_keywords": [
            "budget", "cost", "price", "cheap", "affordable", "expensive",
            "money", "save", "deal", "discount", "free", "value", "tip",
            "currency", "exchange", "payment", "booking"
        ]
    },

    "Logistics Planner": {
        "role": "Transportation & Logistics Coordinator",
        "specialty": "Routes, connections, timing, transit options, car rental comparison",
        "perspective": (
            "Optimize travel routes and timing. Plan efficient connections between "
            "destinations, recommend best transportation options including CAR RENTAL "
            "COMPARISON with top 3 providers, prices, and availability. Create "
            "realistic day-by-day itineraries with travel times."
        ),
        "search_queries": [
            "transportation options destination",
            "best route itinerary travel",
            "car rental comparison prices",
            "train bus connections schedule",
            "travel time between cities"
        ],
        "topics": [
            "best transportation options (flights, trains, buses, car rental)",
            "CAR RENTAL COMPARISON - TOP 3 PROVIDERS:",
            "  - Provider name, vehicle type, price per day and total",
            "  - What's included (insurance, mileage, features)",
            "  - Best for: budget vs comfort vs families",
            "  - Local vs international providers (e.g., Toyota Rent-a-Car in Japan)",
            "airport/station transfers and connections",
            "realistic travel times between locations",
            "day-by-day itinerary optimization",
            "public transportation tips and passes",
            "driving tips for this country (license requirements, road rules)",
            "backup plans for delays or cancellations"
        ],
        "specialty_keywords": [
            "flight", "train", "bus", "car", "rental", "transport", "transit",
            "route", "connection", "transfer", "airport", "station", "schedule",
            "itinerary", "timing", "distance", "travel time", "hertz", "avis",
            "europcar", "sixt", "budget", "enterprise", "driving", "license"
        ]
    },

    # ========================================================================
    # SAFETY & PRACTICAL (2 experts)
    # ========================================================================

    "Safety Expert": {
        "role": "Travel Safety & Insurance Specialist",
        "specialty": "Travel advisories, medical insurance recommendations, emergency prep",
        "perspective": (
            "Ensure travelers stay safe and properly insured. Provide current safety "
            "information, recommend the BEST 3 travel medical insurance providers "
            "with estimated prices for the destination, and emergency contacts."
        ),
        "search_queries": [
            "travel advisory safety destination",
            "best travel medical insurance destination",
            "travel insurance price comparison",
            "emergency contacts embassy",
            "scam warnings tourist safety"
        ],
        "topics": [
            "current travel advisories and safety level",
            "TOP 3 TRAVEL MEDICAL INSURANCE PROVIDERS with prices:",
            "  - Provider name, coverage type, estimated cost per day/trip",
            "  - What's covered (medical, evacuation, trip cancellation)",
            "  - Best for: budget travelers vs comprehensive coverage",
            "  - Special requirements for this country (e.g., Schengen min €30K)",
            "common scams and how to avoid them",
            "safe vs unsafe areas in the destination",
            "emergency contacts (embassy, police, hospitals)",
            "important local laws to follow"
        ],
        "specialty_keywords": [
            "safe", "safety", "danger", "risk", "warning", "advisory",
            "health", "insurance", "emergency", "police", "coverage",
            "embassy", "scam", "crime", "hospital", "medical", "provider"
        ]
    },

    "Weather Analyst": {
        "role": "Climate & Seasonal Planning Expert",
        "specialty": "Weather patterns, best times to visit, packing advice, seasonal events",
        "perspective": (
            "Help travelers choose the best time to visit and pack appropriately. "
            "Provide weather forecasts, seasonal considerations, and climate-based "
            "recommendations for activities and clothing."
        ),
        "search_queries": [
            "weather forecast destination month",
            "best time to visit climate",
            "seasonal events festivals",
            "packing list weather",
            "rainy season dry season"
        ],
        "topics": [
            "expected weather during travel dates",
            "best and worst times to visit",
            "seasonal events and festivals",
            "packing recommendations for the weather",
            "how weather affects planned activities",
            "indoor alternatives for bad weather days",
            "climate considerations (humidity, altitude, UV)"
        ],
        "specialty_keywords": [
            "weather", "climate", "temperature", "rain", "sun", "season",
            "forecast", "hot", "cold", "humid", "dry", "pack", "clothing",
            "festival", "event", "holiday"
        ]
    },

    # ========================================================================
    # EXPERIENCE (3 experts)
    # ========================================================================

    "Local Culture Guide": {
        "role": "Cultural Expert & Local Customs Advisor",
        "specialty": "Local traditions, etiquette, authentic experiences, cultural immersion",
        "perspective": (
            "Help travelers connect authentically with local culture. Share "
            "customs, etiquette tips, and off-the-beaten-path experiences that "
            "provide genuine cultural immersion."
        ),
        "search_queries": [
            "local customs etiquette destination",
            "cultural experiences authentic",
            "off beaten path hidden gems",
            "local traditions festivals",
            "respectful tourism tips"
        ],
        "topics": [
            "essential local customs and etiquette",
            "basic phrases in the local language",
            "authentic local experiences vs tourist traps",
            "cultural dos and don'ts",
            "local neighborhoods to explore",
            "interacting respectfully with locals",
            "unique cultural experiences not in guidebooks"
        ],
        "specialty_keywords": [
            "culture", "local", "tradition", "custom", "etiquette", "language",
            "authentic", "experience", "hidden", "gem", "neighborhood",
            "community", "heritage", "history", "respect"
        ]
    },

    "Food & Dining Expert": {
        "role": "Culinary & Restaurant Specialist",
        "specialty": "Local cuisine, restaurants, reservations, dietary needs, food experiences",
        "perspective": (
            "Guide travelers through the local food scene. Recommend must-try "
            "dishes, best restaurants for different budgets, food markets, and "
            "how to handle dietary restrictions. ALWAYS mention if popular restaurants "
            "require advance reservations and note payment preferences (cash only, etc.)."
        ),
        "search_queries": [
            "best restaurants destination cuisine",
            "local food must try dishes",
            "food markets street food",
            "vegetarian vegan options",
            "food tours cooking classes"
        ],
        "topics": [
            "must-try local dishes and where to find them",
            "restaurant recommendations by budget (cheap eats to fine dining)",
            "RESERVATION ALERTS - Flag popular restaurants that need advance booking:",
            "  - 🔴 BOOK AHEAD: Mention how far in advance (days/weeks)",
            "  - Only mention if reservations are needed or strongly recommended",
            "CASH ONLY ALERTS - Flag restaurants that don't accept cards:",
            "  - 💵 CASH ONLY: Traditional spots, markets, street food that need cash",
            "  - Mention local payment apps if useful (PayPay in Japan, etc.)",
            "food markets and street food spots",
            "handling dietary restrictions (vegetarian, vegan, allergies, halal, kosher)",
            "best times to visit (avoid crowds, freshest food, happy hours)",
            "food tours and cooking classes",
            "tipping customs and dining etiquette",
            "food safety tips"
        ],
        "specialty_keywords": [
            "food", "restaurant", "cuisine", "dish", "eat", "dining",
            "market", "street food", "vegetarian", "vegan", "halal",
            "breakfast", "lunch", "dinner", "cafe", "bar", "reservation",
            "book", "cash", "payment", "credit card"
        ]
    },

    "Activity Curator": {
        "role": "Experience & Activity Specialist",
        "specialty": "Tours, attractions, entertainment, outdoor activities, day trips",
        "perspective": (
            "Curate the best activities and experiences for travelers. Match "
            "attractions to interests, suggest day trips, and help prioritize "
            "what to see when time is limited."
        ),
        "search_queries": [
            "top attractions things to do",
            "tours activities experiences",
            "day trips excursions",
            "museums galleries entertainment",
            "outdoor activities adventure"
        ],
        "topics": [
            "must-see attractions and best times to visit",
            "recommended tours (walking, bike, boat, etc.)",
            "day trip options from the destination",
            "activities matched to traveler interests",
            "booking tips (advance tickets, skip-the-line)",
            "free activities and hidden gems",
            "nightlife and entertainment options"
        ],
        "specialty_keywords": [
            "attraction", "tour", "activity", "museum", "park", "beach",
            "mountain", "hike", "adventure", "entertainment", "show",
            "ticket", "booking", "excursion", "day trip", "nightlife"
        ]
    },

    # ========================================================================
    # ACCOMMODATION (1 expert)
    # ========================================================================

    "Accommodation Specialist": {
        "role": "Lodging & Accommodation Expert",
        "specialty": "Hotels, rentals, hostels, location strategy, amenities",
        "perspective": (
            "Help travelers find the perfect place to stay. Balance location, "
            "price, and amenities. Advise on neighborhoods, booking strategies, "
            "and accommodation types for different travel styles."
        ),
        "search_queries": [
            "best hotels destination area",
            "airbnb vacation rental options",
            "best neighborhood to stay",
            "hostel budget accommodation",
            "hotel booking tips deals"
        ],
        "topics": [
            "best neighborhoods/areas to stay and why",
            "accommodation options by budget (hostel, hotel, rental)",
            "booking platforms and timing for best prices",
            "important amenities to look for",
            "location trade-offs (central vs cheaper outskirts)",
            "family-friendly vs solo traveler options",
            "unique accommodation experiences (boutique, historic, etc.)"
        ],
        "specialty_keywords": [
            "hotel", "hostel", "airbnb", "rental", "accommodation", "stay",
            "room", "apartment", "resort", "booking", "location", "area",
            "neighborhood", "amenity", "wifi", "breakfast"
        ]
    },
}

# Expert emoji badges for visual identification
EXPERT_ICONS = {
    "Budget Advisor": "💰",
    "Logistics Planner": "🚗",
    "Safety Expert": "🛡️",
    "Weather Analyst": "🌤️",
    "Local Culture Guide": "🎎",
    "Food & Dining Expert": "🍜",
    "Activity Curator": "🎯",
    "Accommodation Specialist": "🏨",
}

# Expert Categories for UI grouping
TRAVEL_CATEGORIES = {
    "Planning & Budget": ["Budget Advisor", "Logistics Planner"],
    "Safety & Practical": ["Safety Expert", "Weather Analyst"],
    "Experience": ["Local Culture Guide", "Food & Dining Expert", "Activity Curator"],
    "Accommodation": ["Accommodation Specialist"],
}

# Preset expert combinations for common trip types
TRAVEL_PRESETS = {
    "Quick Trip Planning": {
        "experts": ["Budget Advisor", "Logistics Planner", "Accommodation Specialist", "Activity Curator"],
        "focus": "Fast itinerary for a specific destination",
        "description": "Essential experts for planning any trip quickly"
    },
    "Adventure Travel": {
        "experts": ["Safety Expert", "Activity Curator", "Weather Analyst", "Local Culture Guide"],
        "focus": "Outdoor and adventure trip planning",
        "description": "For hiking, outdoor activities, and adventure seekers"
    },
    "Budget Backpacking": {
        "experts": ["Budget Advisor", "Accommodation Specialist", "Food & Dining Expert", "Safety Expert"],
        "focus": "Maximum value on minimal budget",
        "description": "Stretch your money further with smart planning"
    },
    "Cultural Immersion": {
        "experts": ["Local Culture Guide", "Food & Dining Expert", "Activity Curator"],
        "focus": "Deep local experience",
        "description": "Connect authentically with local culture"
    },
    "Family Vacation": {
        "experts": ["Safety Expert", "Accommodation Specialist", "Activity Curator", "Logistics Planner"],
        "focus": "Family-friendly trip planning",
        "description": "Safe, convenient, and fun for all ages"
    },
    "Full Panel": {
        "experts": list(TRAVEL_EXPERTS.keys()),
        "focus": "Comprehensive travel planning",
        "description": "Get advice from all 8 travel experts"
    },
}


def get_travel_prompts(bullets_per_role: int = 10) -> Dict[str, Tuple[str, str]]:
    """
    Generate prompts for each travel expert.

    Returns:
        Dict mapping expert name to (context, task) tuple
    """
    prompts = {}

    for name, config in TRAVEL_EXPERTS.items():
        context = TRAVEL_BASE_CONTEXT + f"\n\nYour Role: {config['role']}\n"
        context += f"Specialty: {config['specialty']}\n"
        context += f"Perspective: {config['perspective']}\n"

        # Build task from topics
        topics_str = "\n".join(f"• {topic}" for topic in config["topics"])
        task = (
            f"As the {config['role']}, provide comprehensive, detailed recommendations "
            f"covering these areas:\n{topics_str}\n\n"
            "RESPONSE REQUIREMENTS:\n"
            "- Provide DETAILED paragraphs, not just bullet points\n"
            "- Include specific names of places, shops, restaurants, services\n"
            "- Add practical tips that locals would know\n"
            "- Include price ranges in local currency and USD\n"
            "- Mention timing (best hours, days to visit, seasonal considerations)\n"
            "- Give alternatives for different budgets or preferences\n"
            "- Be thorough - aim for a comprehensive guide, not a summary"
        )

        prompts[name] = (context, task)

    return prompts


def get_default_travel_experts() -> List[str]:
    """Return default expert selection for general travel questions."""
    return ["Budget Advisor", "Logistics Planner", "Accommodation Specialist", "Activity Curator"]


def get_experts_by_category(category: str) -> List[str]:
    """Get expert names for a given category."""
    return TRAVEL_CATEGORIES.get(category, [])


def get_all_expert_names() -> List[str]:
    """Get all expert names."""
    return list(TRAVEL_EXPERTS.keys())


def call_travel_expert(
    persona_name: str,
    clinical_question: str,  # Named for compatibility, actually travel question
    evidence_context: str,
    round_num: int = 1,
    previous_responses: Optional[Dict[str, str]] = None,
    priors_text: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    model: str = None,
    max_completion_tokens: int = 4096,
    timeout: float = 120.0,
    system_instruction_override: Optional[str] = None,
    **kwargs  # Accept additional kwargs for compatibility
) -> Dict[str, any]:
    """
    Call LLM to generate a travel expert response.

    Args:
        persona_name: Travel expert name
        clinical_question: Travel question (named for compatibility)
        evidence_context: Web/API context for the expert
        round_num: Discussion round
        previous_responses: Previous expert responses
        priors_text: Not used for travel (compatibility)
        openai_api_key: API key (uses settings if not provided)
        model: Model name (uses settings.EXPERT_MODEL if not provided)
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
        openai_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        return {
            'content': "Error: No API key configured",
            'finish_reason': 'error',
            'model': model or 'unknown',
            'tokens': {}
        }

    # Get model from settings if not specified
    if not model:
        model = getattr(settings, 'EXPERT_MODEL', 'gemini-2.0-flash')

    # Get expert config
    expert_info = TRAVEL_EXPERTS.get(persona_name, {})
    if not expert_info:
        return {
            'content': f"Error: Unknown travel expert '{persona_name}'",
            'finish_reason': 'error',
            'model': model,
            'tokens': {}
        }

    # Build system prompt
    if system_instruction_override:
        system_prompt = system_instruction_override
    else:
        prompts = get_travel_prompts()
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
    user_message = f"Travel Question: {clinical_question}"
    if evidence_context:
        user_message += f"\n\n## Available Information:\n{evidence_context}"

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
        logger.error(f"Travel expert call failed for {persona_name}: {e}")
        return {
            'content': f"Error calling {persona_name}: {str(e)}",
            'finish_reason': 'error',
            'model': model,
            'tokens': {}
        }


def call_travel_expert_stream(
    persona_name: str,
    clinical_question: str,  # Named for compatibility, actually travel question
    evidence_context: str,
    round_num: int = 1,
    previous_responses: Optional[Dict[str, str]] = None,
    priors_text: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    model: str = None,
    max_completion_tokens: int = None,  # Will use settings.EXPERT_MAX_TOKENS if None
    timeout: float = 120.0
):
    """
    Stream a travel expert response for real-time display.

    Yields dicts with:
    - {"type": "chunk", "content": "..."} for text chunks
    - {"type": "error", "content": "..."} for errors
    - {"type": "complete", "finish_reason": "...", "model": "..."} when done

    Args:
        persona_name: Travel expert name
        clinical_question: Travel question (named for compatibility)
        evidence_context: Web/API context for the expert
        round_num: Discussion round
        previous_responses: Previous expert responses
        priors_text: Not used (compatibility)
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
        openai_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        yield {'type': 'error', 'content': "Error: No API key configured"}
        yield {'type': 'complete', 'finish_reason': 'error', 'model': model or 'unknown'}
        return

    # Get model from settings if not specified
    if not model:
        model = getattr(settings, 'EXPERT_MODEL', 'gemini-2.0-flash')

    # Get max tokens from settings if not specified
    if max_completion_tokens is None:
        max_completion_tokens = getattr(settings, 'EXPERT_MAX_TOKENS', 6000)

    # Get prompts for this persona
    prompts = get_travel_prompts()
    if persona_name not in prompts:
        yield {'type': 'error', 'content': f"Error: Unknown travel expert '{persona_name}'"}
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
    user_message = f"Travel Question: {clinical_question}"
    if evidence_context:
        user_message += f"\n\n## Available Information:\n{evidence_context}"

    try:
        from core.llm_utils import get_llm_client

        client = get_llm_client(api_key=openai_api_key, model=model)
        logger.info(f"Starting stream for travel expert {persona_name}")

        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_completion_tokens=max_completion_tokens,
            stream=True
        )

        chunk_count = 0
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunk_count += 1
                yield {'type': 'chunk', 'content': chunk.choices[0].delta.content}

            if chunk.choices and chunk.choices[0].finish_reason:
                logger.info(f"Stream complete for {persona_name}: {chunk_count} chunks")
                yield {
                    'type': 'complete',
                    'finish_reason': chunk.choices[0].finish_reason,
                    'model': model
                }
                return

        # If loop ends without finish_reason
        yield {'type': 'complete', 'finish_reason': 'unknown', 'model': model}

    except Exception as e:
        logger.error(f"Stream error for {persona_name}: {e}")
        yield {'type': 'error', 'content': f"Error: {str(e)}"}
        yield {'type': 'complete', 'finish_reason': 'error', 'model': model}
