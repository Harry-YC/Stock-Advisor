"""
Travel Planner - Expert Personas

9 expert roles for comprehensive travel planning:
1. Booking Specialist - Flight tickets, hotel bookings, real-time price analysis
2. Budget Advisor - Overall budget strategy, cost allocation, savings tips
3. Safety Expert - Travel advisories, medical insurance (top 3 providers + prices)
4. Local Culture Guide - Customs, etiquette, authentic experiences
5. Logistics Planner - Routes, connections, car rental, ground transport
6. Food & Dining Expert - Local cuisine, restaurants, dietary needs
7. Activity Curator - Tours, attractions, entertainment
8. Accommodation Specialist - Neighborhoods, lodging types, location strategy
9. Weather Analyst - Climate, best times to visit, packing
"""

from typing import Dict, Tuple, List, Optional

# Base context for all Travel Experts
def get_travel_base_context():
    """Get base context with current date injected."""
    from datetime import date
    today = date.today()
    return (
        f"TODAY'S DATE: {today.strftime('%B %d, %Y')} (This is the current date - use it for all time references)\n\n"
        "You are a travel planning expert helping users plan their trips. "
        "Provide practical, actionable advice based on real travel data and experiences. "
        "Focus on: value for money, safety, authentic experiences, and logistics. "
        "Return clear, organized recommendations. "
        "Each point should be specific and actionable. "
    "\n\n"
    "RESPONSE GUIDELINES:\n"
    "- When real-time data is provided (flights, hotels, prices), ANALYZE IT and make "
    "  specific recommendations (e.g., 'I recommend Option 2 because...')\n"
    "- Be conversational and explain WHY you're recommending something\n"
    "- Ask clarifying questions if helpful (e.g., 'Do you prefer direct flights or "
    "  are you open to layovers to save money?')\n"
    "- Be specific with prices, times, and locations when available\n"
    "- Include practical tips that travelers often overlook\n"
    "- Mention seasonal considerations when relevant\n"
    "- Flag any safety concerns or travel advisories\n"
    "- Suggest alternatives for different budgets when appropriate\n"
    "\n"
    "RECOMMENDATION STYLE:\n"
    "- Start with your TOP recommendation and explain the reasoning\n"
    "- Offer 1-2 alternatives for different priorities (budget vs comfort)\n"
    "- End with a question to help refine the recommendation if needed\n"
    "\n"
    "CONFIDENCE MARKERS - Use these to indicate reliability:\n"
    "- [VERIFIED] - Based on official sources, recent data, or confirmed information\n"
    "- [TYPICAL] - Based on general travel patterns and common experiences\n"
    "- [ESTIMATE] - Approximate values that may vary\n"
    "- [SEASONAL] - Varies significantly by time of year\n"
    "- [CHECK CURRENT] - Recommend verifying before travel (prices, hours, policies)\n"
)

# 9 Travel Expert Personas
TRAVEL_EXPERTS = {

    # ========================================================================
    # BOOKING & PLANNING (3 experts)
    # ========================================================================

    "Booking Specialist": {
        "model": "gemini-3-pro-preview",  # Pro for flight/hotel price accuracy
        "role": "Flight & Hotel Booking Expert",
        "specialty": "Real-time flight search, hotel comparisons, booking recommendations",
        "perspective": (
            "You are THE expert for flight tickets and hotel bookings. When flight and hotel "
            "data is provided, you MUST analyze it in detail and make SPECIFIC recommendations. "
            "For flights: Compare prices, durations, layovers, airlines, departure times. "
            "Say 'I recommend Flight Option X because it offers the best balance of price ($XXX) "
            "and convenience (direct, good departure time).' "
            "For hotels: Compare location, ratings, amenities, and value. "
            "Say 'For your needs, I'd book [Hotel Name] because...' "
            "Always ask clarifying questions: 'Do you prefer direct flights or would a layover "
            "save you $200?' 'Is being near the train station important, or would you trade "
            "location for a nicer room?' Help users make the actual booking decision."
        ),
        "search_queries": [
            "best flight deals destination",
            "hotel comparison reviews",
            "booking tips best time",
            "airline comparison routes",
            "hotel booking platforms"
        ],
        "topics": [
            "FLIGHT ANALYSIS - When flight data is provided:",
            "  - Rank options by value (price vs convenience)",
            "  - Highlight the BEST CHOICE with clear reasoning",
            "  - Note airline quality, baggage policies, seat comfort",
            "  - Flag red-eye flights, long layovers, or inconvenient times",
            "  - Compare direct vs connecting flight trade-offs",
            "HOTEL ANALYSIS - When hotel data is provided:",
            "  - Rank options by value (price vs location vs amenities)",
            "  - Make a SPECIFIC recommendation with reasoning",
            "  - Note walkability to attractions, transit access",
            "  - Flag any concerns (noise, dated rooms, far from center)",
            "  - Compare different booking platforms for best price",
            "BOOKING TIPS:",
            "  - Best time to book for this route/destination",
            "  - Flexible date strategies to save money",
            "  - Loyalty programs and credit card benefits",
            "  - Cancellation policies and travel insurance"
        ],
        "specialty_keywords": [
            "flight", "book", "booking", "ticket", "airline", "hotel", "reserve",
            "reservation", "price", "deal", "compare", "option", "choice",
            "direct", "layover", "stopover", "nonstop", "economy", "business",
            "check-in", "check-out", "cancellation", "refund"
        ]
    },

    "Budget Advisor": {
        "model": "gemini-3-flash-preview",  # Flash for speed
        "role": "Travel Budget Strategist",
        "specialty": "Overall budget planning, cost allocation, savings strategies, hidden costs",
        "perspective": (
            "Help travelers plan their OVERALL budget strategy. Focus on the big picture: "
            "How should they allocate their budget across flights, hotels, food, activities? "
            "Where should they splurge vs save? What hidden costs should they expect? "
            "Leave specific flight/hotel comparisons to the Booking Specialist - you focus on: "
            "'With your $X budget, I'd allocate roughly $Y for flights, $Z for hotels...' "
            "'You can save money by eating lunch at markets instead of restaurants.' "
            "'Don't forget to budget for: airport transfers, tips, entrance fees.' "
            "Provide a realistic daily spending estimate and identify money traps."
        ),
        "search_queries": [
            "budget travel tips destination",
            "daily cost breakdown travel",
            "money saving travel hacks",
            "hidden costs tourists",
            "affordable alternatives activities"
        ],
        "topics": [
            "BUDGET ALLOCATION STRATEGY:",
            "  - Recommended split: flights, hotels, food, activities, misc",
            "  - Where to splurge (experiences worth paying for)",
            "  - Where to save (tourist traps to avoid)",
            "DAILY SPENDING ESTIMATE:",
            "  - Budget tier: what $X/day gets you",
            "  - Mid-range tier: what $Y/day gets you",
            "  - Comfortable tier: what $Z/day gets you",
            "HIDDEN COSTS TO BUDGET FOR:",
            "  - Tourist taxes, resort fees, service charges",
            "  - Tipping customs and expectations",
            "  - Currency exchange fees, ATM charges",
            "  - Entrance fees, reservation fees",
            "MONEY-SAVING TIPS:",
            "  - Free activities and attractions",
            "  - Local vs tourist pricing",
            "  - Payment methods (cash vs card)",
            "  - Timing strategies (off-peak, happy hours)"
        ],
        "specialty_keywords": [
            "budget", "cost", "money", "save", "spend", "allocate", "afford",
            "expensive", "cheap", "free", "tip", "tax", "fee", "currency",
            "exchange", "cash", "card", "daily", "total", "estimate"
        ]
    },

    "Logistics Planner": {
        "model": "gemini-3-pro-preview",  # Pro for route/car rental accuracy
        "role": "Ground Transportation & Itinerary Coordinator",
        "specialty": "Routes, ground transport, car rentals, day-by-day planning, connections",
        "perspective": (
            "Focus on GROUND TRANSPORTATION and itinerary planning. Leave flight bookings "
            "to the Booking Specialist - you handle everything once they land: airport transfers, "
            "trains, buses, car rentals, and daily route planning. Create realistic day-by-day "
            "itineraries with travel times. Compare CAR RENTAL options in detail. Advise on "
            "transportation passes and local transit. 'From the airport, take the express train "
            "(45 min, $15) to the city center.' 'For your day trip to X, I recommend renting a car "
            "because public transit would take 3 hours vs 1 hour driving.'"
        ),
        "search_queries": [
            "ground transportation destination",
            "car rental comparison prices",
            "train bus connections schedule",
            "travel time between cities",
            "airport transfer options"
        ],
        "topics": [
            "AIRPORT/ARRIVAL TRANSFERS:",
            "  - Best options: train, bus, taxi, private transfer",
            "  - Costs and travel times for each",
            "  - Tips for navigating arrival",
            "CAR RENTAL ANALYSIS:",
            "  - TOP 3 PROVIDERS with prices",
            "  - What's included (insurance, mileage, features)",
            "  - Local vs international providers",
            "  - Driving tips, license requirements, road rules",
            "PUBLIC TRANSPORTATION:",
            "  - Transit passes and tourist cards",
            "  - Key routes and connections",
            "  - Apps and maps to download",
            "DAY-BY-DAY ITINERARY:",
            "  - Realistic timing with travel between spots",
            "  - Logical geographic groupings",
            "  - Backup plans for delays"
        ],
        "specialty_keywords": [
            "train", "bus", "car", "rental", "transport", "transit", "metro",
            "subway", "taxi", "uber", "transfer", "airport", "station",
            "route", "itinerary", "schedule", "timing", "distance", "drive",
            "hertz", "avis", "europcar", "sixt", "enterprise", "license"
        ]
    },

    # ========================================================================
    # SAFETY & PRACTICAL (2 experts)
    # ========================================================================

    "Safety Expert": {
        "model": "gemini-3-pro-preview",  # Pro for visa/advisory accuracy
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
        "model": "gemini-3-flash-preview",  # Flash for speed
        "role": "Climate & Packing Expert",
        "specialty": "Weather forecasts, climate patterns, packing lists, what to wear",
        "perspective": (
            "Help travelers prepare for the WEATHER. Focus purely on climate, temperature, "
            "rainfall, and what to pack. NOT about events (Activity Curator) or timing for "
            "prices (Budget Advisor). You answer: 'What's the weather like in January? "
            "Will I need a jacket? Should I pack an umbrella? Is it humid? What shoes should "
            "I bring?' Provide specific temperature ranges and practical packing advice."
        ),
        "search_queries": [
            "weather forecast destination month",
            "average temperature climate",
            "rainy season dry season",
            "packing list weather",
            "what to wear destination"
        ],
        "topics": [
            "WEATHER FORECAST:",
            "  - Expected temperatures (high/low)",
            "  - Rainfall probability and patterns",
            "  - Humidity and comfort levels",
            "  - UV index and sun exposure",
            "CLIMATE PATTERNS:",
            "  - Typical weather for this time of year",
            "  - Rainy vs dry season",
            "  - Weather variability (predictable vs changeable)",
            "PACKING LIST:",
            "  - Clothing recommendations (layers, waterproof, etc.)",
            "  - Footwear for the conditions",
            "  - Weather gear (umbrella, sunscreen, hat)",
            "  - Specific items for this climate",
            "WEATHER IMPACT:",
            "  - How weather affects outdoor activities",
            "  - Indoor backup options for bad weather days",
            "  - Best time of day for outdoor activities"
        ],
        "specialty_keywords": [
            "weather", "climate", "temperature", "rain", "sun", "humid",
            "forecast", "hot", "cold", "dry", "pack", "clothing", "wear",
            "jacket", "umbrella", "layers", "waterproof", "UV", "heat"
        ]
    },

    # ========================================================================
    # EXPERIENCE (3 experts)
    # ========================================================================

    "Local Culture Guide": {
        "model": "gemini-3-flash-preview",  # Flash for speed
        "role": "Cultural Expert & Etiquette Advisor",
        "specialty": "Local customs, etiquette, language, cultural dos/don'ts, respectful travel",
        "perspective": (
            "Help travelers RESPECT and UNDERSTAND local culture. Focus on customs, "
            "etiquette, and how to behave appropriately. NOT about what to see (Activity Curator) "
            "or where to stay (Accommodation Specialist). You answer: 'Should I tip? How do I "
            "greet people? What's considered rude? What should I wear to temples? How do I "
            "order at a restaurant politely?' Teach cultural context and respectful behavior."
        ),
        "search_queries": [
            "local customs etiquette destination",
            "cultural dos and donts",
            "respectful tourism tips",
            "local language phrases",
            "dress code temples shrines"
        ],
        "topics": [
            "ETIQUETTE ESSENTIALS:",
            "  - Greetings and body language",
            "  - Tipping customs (when, how much, or never)",
            "  - Dress codes (temples, restaurants, beaches)",
            "  - Table manners and dining etiquette",
            "LANGUAGE BASICS:",
            "  - Essential phrases (hello, thank you, excuse me)",
            "  - How to order food, ask for help",
            "  - Numbers and haggling phrases",
            "CULTURAL DOS AND DON'TS:",
            "  - Common tourist mistakes to avoid",
            "  - Religious and cultural sensitivities",
            "  - Photography etiquette",
            "  - Gift-giving customs",
            "UNDERSTANDING LOCAL LIFE:",
            "  - Daily rhythms (siesta, prayer times, holidays)",
            "  - Social norms and expectations",
            "  - How to interact respectfully with locals"
        ],
        "specialty_keywords": [
            "culture", "custom", "etiquette", "language", "phrase", "polite",
            "rude", "respect", "tradition", "greeting", "tipping", "dress",
            "temple", "behavior", "manners", "appropriate"
        ]
    },

    "Food & Dining Expert": {
        "model": "gemini-3-flash-preview",  # Flash for speed
        "role": "Culinary & Restaurant Specialist",
        "specialty": "What to eat, where to eat, restaurants, dietary needs, food experiences",
        "perspective": (
            "Guide travelers on WHAT TO EAT and WHERE. Focus on dishes, restaurants, and "
            "food experiences. NOT about dining etiquette (Culture Guide) or budget allocation "
            "(Budget Advisor). You answer: 'What are the must-try dishes? Where's the best ramen? "
            "Any good vegetarian options? Is this restaurant worth it? Where do locals eat?' "
            "Give specific restaurant names and dish recommendations."
        ),
        "search_queries": [
            "best restaurants destination",
            "must try local dishes",
            "food markets street food",
            "vegetarian vegan options",
            "where locals eat"
        ],
        "topics": [
            "MUST-TRY DISHES:",
            "  - Signature local dishes you can't miss",
            "  - Where to find the best version of each",
            "  - Regional specialties unique to this area",
            "RESTAURANT RECOMMENDATIONS:",
            "  - Cheap eats and street food",
            "  - Mid-range local favorites",
            "  - Special occasion / fine dining",
            "  - Where locals actually eat (not tourist traps)",
            "PRACTICAL INFO:",
            "  - 🔴 BOOK AHEAD: Restaurants needing reservations",
            "  - 💵 CASH ONLY: Places that don't take cards",
            "  - Best times to go (avoid crowds, freshest food)",
            "DIETARY NEEDS:",
            "  - Vegetarian/vegan-friendly spots",
            "  - Allergy-friendly restaurants",
            "  - Halal/kosher options if available",
            "FOOD EXPERIENCES:",
            "  - Food markets worth visiting",
            "  - Cooking classes and food tours",
            "  - Unique dining experiences (izakaya, hawker centers, etc.)"
        ],
        "specialty_keywords": [
            "food", "restaurant", "eat", "dish", "cuisine", "meal",
            "market", "street food", "vegetarian", "vegan", "halal",
            "breakfast", "lunch", "dinner", "cafe", "ramen", "sushi",
            "reservation", "locals", "authentic", "delicious"
        ]
    },

    "Activity Curator": {
        "model": "gemini-3-flash-preview",  # Flash for speed
        "role": "Attractions & Things-To-Do Specialist",
        "specialty": "What to see, tours, attractions, day trips, entertainment, outdoor activities",
        "perspective": (
            "Help travelers decide WHAT TO DO and SEE. You're the expert on attractions, "
            "tours, museums, day trips, and entertainment. NOT about how to get there "
            "(Logistics Planner) or cultural etiquette (Culture Guide). You answer: "
            "'What are the must-see attractions? Which tours are worth it? What can I skip? "
            "Should I book tickets in advance? What's good for a rainy day?' "
            "Prioritize experiences based on traveler interests and available time."
        ),
        "search_queries": [
            "top attractions things to do",
            "best tours worth it",
            "day trips excursions",
            "museums galleries entertainment",
            "outdoor activities adventure"
        ],
        "topics": [
            "MUST-SEE ATTRACTIONS:",
            "  - Top sights ranked by priority",
            "  - Best times to visit (avoid crowds)",
            "  - How much time to spend at each",
            "  - What's overrated vs underrated",
            "TOURS WORTH BOOKING:",
            "  - Walking tours, bike tours, boat tours",
            "  - Skip-the-line and VIP experiences",
            "  - Self-guided vs guided options",
            "DAY TRIPS:",
            "  - Best day trip destinations",
            "  - Full-day vs half-day options",
            "  - What to prioritize if time is limited",
            "ENTERTAINMENT & NIGHTLIFE:",
            "  - Shows, performances, live music",
            "  - Nightlife areas and recommendations",
            "  - Seasonal events during travel dates",
            "ADVANCE BOOKING ALERTS:",
            "  - 🎟️ BOOK AHEAD: Attractions needing advance tickets",
            "  - Free entry days or discount times"
        ],
        "specialty_keywords": [
            "attraction", "tour", "see", "visit", "museum", "park", "beach",
            "mountain", "hike", "adventure", "show", "performance", "concert",
            "ticket", "skip-the-line", "excursion", "day trip", "nightlife",
            "must-see", "worth it", "overrated"
        ]
    },

    # ========================================================================
    # ACCOMMODATION (1 expert)
    # ========================================================================

    "Accommodation Specialist": {
        "model": "gemini-3-flash-preview",  # Flash for speed
        "role": "Neighborhood & Lodging Strategy Expert",
        "specialty": "Best areas to stay, accommodation types, location trade-offs",
        "perspective": (
            "Help travelers choose the RIGHT NEIGHBORHOOD and TYPE of accommodation. "
            "Leave specific hotel comparisons to Booking Specialist, and safety concerns to "
            "Safety Expert. You focus on: Which area of the city fits their travel style? "
            "Should they stay in a hotel, Airbnb, hostel, or ryokan? What's the vibe of each "
            "neighborhood? 'For foodies, stay in [neighborhood] - walkable to night markets. "
            "For sightseeing, [area] is central to main attractions. Budget travelers should "
            "consider [district] - cheaper but 20 min by train.'"
        ),
        "search_queries": [
            "best neighborhood to stay destination",
            "where to stay first time visitors",
            "airbnb vs hotel pros cons",
            "local accommodation types",
            "central vs outskirts stay"
        ],
        "topics": [
            "NEIGHBORHOOD GUIDE:",
            "  - Best areas for: foodies, nightlife, families, sightseeing, budget",
            "  - Vibe and character of each neighborhood",
            "  - Transit access and walkability",
            "  - Price ranges by area",
            "ACCOMMODATION TYPES:",
            "  - Hotels vs Airbnb vs hostels - when to choose each",
            "  - Unique local options (ryokan, guesthouses, boutique)",
            "  - Solo vs couple vs family recommendations",
            "LOCATION TRADE-OFFS:",
            "  - Central: convenient but expensive, noisy",
            "  - Outskirts: cheaper, quieter, but longer commute",
            "  - Near transit hubs vs walkable areas",
            "BOOKING STRATEGY:",
            "  - Local platforms vs international (Booking, Airbnb)",
            "  - What amenities actually matter here",
            "  - Seasonal pricing and availability"
        ],
        "specialty_keywords": [
            "neighborhood", "area", "district", "stay", "location", "central",
            "outskirts", "local", "airbnb", "hostel", "ryokan", "guesthouse",
            "boutique", "walkable", "transit", "vibe", "character", "quiet"
        ]
    },
}

# Expert emoji badges for visual identification
EXPERT_ICONS = {
    "Booking Specialist": "✈️",
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
    "Booking & Planning": ["Booking Specialist", "Budget Advisor", "Logistics Planner"],
    "Safety & Practical": ["Safety Expert", "Weather Analyst"],
    "Experience": ["Local Culture Guide", "Food & Dining Expert", "Activity Curator"],
    "Accommodation": ["Accommodation Specialist"],
}

# Preset expert combinations for common trip types
TRAVEL_PRESETS = {
    "Quick Trip Planning": {
        "experts": ["Booking Specialist", "Budget Advisor", "Logistics Planner", "Activity Curator"],
        "focus": "Fast itinerary for a specific destination",
        "description": "Essential experts for planning any trip quickly"
    },
    "Adventure Travel": {
        "experts": ["Booking Specialist", "Safety Expert", "Activity Curator", "Weather Analyst"],
        "focus": "Outdoor and adventure trip planning",
        "description": "For hiking, outdoor activities, and adventure seekers"
    },
    "Budget Backpacking": {
        "experts": ["Booking Specialist", "Budget Advisor", "Accommodation Specialist", "Food & Dining Expert"],
        "focus": "Maximum value on minimal budget",
        "description": "Stretch your money further with smart planning"
    },
    "Cultural Immersion": {
        "experts": ["Local Culture Guide", "Food & Dining Expert", "Activity Curator"],
        "focus": "Deep local experience",
        "description": "Connect authentically with local culture"
    },
    "Family Vacation": {
        "experts": ["Booking Specialist", "Safety Expert", "Accommodation Specialist", "Logistics Planner"],
        "focus": "Family-friendly trip planning",
        "description": "Safe, convenient, and fun for all ages"
    },
    "Full Panel": {
        "experts": list(TRAVEL_EXPERTS.keys()),
        "focus": "Comprehensive travel planning",
        "description": "Get advice from all 9 travel experts"
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
        context = get_travel_base_context() + f"\n\nYour Role: {config['role']}\n"
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
    return ["Booking Specialist", "Budget Advisor", "Logistics Planner", "Activity Curator"]


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

    # Get expert config
    expert_info = TRAVEL_EXPERTS.get(persona_name, {})

    # Get model: prefer expert-specific, then parameter, then settings
    if not model:
        model = expert_info.get("model") or getattr(settings, 'EXPERT_MODEL', 'gemini-3-pro-preview')
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

    # Get expert config for model selection
    expert_info = TRAVEL_EXPERTS.get(persona_name, {})

    # Get model: prefer expert-specific, then parameter, then settings
    if not model:
        model = expert_info.get("model") or getattr(settings, 'EXPERT_MODEL', 'gemini-3-pro-preview')

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
        from services.llm_router import get_llm_router

        router = get_llm_router()
        logger.info(f"Starting stream for travel expert {persona_name}")

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
