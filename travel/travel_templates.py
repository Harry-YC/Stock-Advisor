"""
Travel Planner - Question Templates

Question types for travel planning with expert mappings and output templates.
"""

from typing import Dict, List, Optional
import re

# Travel Question Types with expert mappings
TRAVEL_QUESTION_TYPES = {
    "destination_planning": {
        "name": "Destination Planning",
        "description": "Plan a complete trip to a destination",
        "icon": "✈️",
        "experts": ["Budget Advisor", "Logistics Planner", "Accommodation Specialist", "Activity Curator"],
        "synthesis_focus": "itinerary_recommendation",
        "keywords": ["plan", "trip", "travel to", "visit", "going to", "vacation", "holiday", "itinerary"],
        "output_template": """
## Trip Overview
[Destination, duration, and travel style summary]

## Recommended Itinerary
[Day-by-day breakdown with activities, timing, and locations]

## Budget Estimate
| Category | Estimated Cost |
|----------|----------------|
| Flights | $ |
| Accommodation | $/night |
| Transportation | $ |
| Activities | $ |
| Food | $/day |
| **Total** | **$** |

## Key Recommendations
[Top 3-5 must-do items]

## Important Tips
[Practical advice and things to watch out for]
"""
    },

    "budget_optimization": {
        "name": "Budget Travel",
        "description": "Plan a trip with budget constraints",
        "icon": "💰",
        "experts": ["Budget Advisor", "Accommodation Specialist", "Food & Dining Expert", "Logistics Planner"],
        "synthesis_focus": "cost_optimization",
        "keywords": ["budget", "cheap", "affordable", "save money", "cost", "inexpensive", "backpack"],
        "output_template": """
## Budget Breakdown
[Detailed cost analysis]

## Money-Saving Tips
[Specific ways to reduce costs]

## Best Value Options
- Accommodation: [cheapest good options]
- Food: [cheap eats and markets]
- Transport: [budget transport options]
- Activities: [free/cheap things to do]

## What's Worth Spending On
[Where NOT to skimp]
"""
    },

    "activity_search": {
        "name": "Things To Do",
        "description": "Find activities and attractions",
        "icon": "🎯",
        "experts": ["Activity Curator", "Local Culture Guide", "Food & Dining Expert"],
        "synthesis_focus": "activity_recommendations",
        "keywords": ["things to do", "activities", "attractions", "see", "visit", "experience", "tour"],
        "output_template": """
## Must-See Attractions
[Top attractions with timing and tips]

## Unique Experiences
[Off-the-beaten-path recommendations]

## Day Trips
[Nearby excursions worth considering]

## Booking Tips
[When and how to book for best experience]
"""
    },

    "food_guide": {
        "name": "Food & Dining",
        "description": "Restaurant and food recommendations",
        "icon": "🍽️",
        "experts": ["Food & Dining Expert", "Local Culture Guide", "Budget Advisor"],
        "synthesis_focus": "dining_recommendations",
        "keywords": ["food", "eat", "restaurant", "cuisine", "dining", "dish", "meal", "vegetarian", "vegan"],
        "output_template": """
## Must-Try Local Dishes
[Signature dishes and where to find them]

## Restaurant Recommendations
### Budget-Friendly
[Cheap eats]

### Mid-Range
[Good value restaurants]

### Special Occasion
[Worth the splurge]

## Food Markets & Street Food
[Best spots for authentic local food]

## Dietary Notes
[Tips for vegetarians, allergies, etc.]
"""
    },

    "accommodation_search": {
        "name": "Where To Stay",
        "description": "Find the best place to stay",
        "icon": "🏨",
        "experts": ["Accommodation Specialist", "Budget Advisor", "Safety Expert"],
        "synthesis_focus": "accommodation_recommendations",
        "keywords": ["hotel", "stay", "accommodation", "hostel", "airbnb", "where to sleep", "neighborhood"],
        "output_template": """
## Best Neighborhoods
[Area recommendations with pros/cons]

## Accommodation Options
### Budget (Under $X/night)
[Options]

### Mid-Range ($X-$X/night)
[Options]

### Luxury ($X+/night)
[Options]

## Booking Tips
[Best platforms, timing, and strategies]
"""
    },

    "safety_check": {
        "name": "Safety Assessment",
        "description": "Safety and health information",
        "icon": "🛡️",
        "experts": ["Safety Expert", "Local Culture Guide", "Weather Analyst"],
        "synthesis_focus": "safety_assessment",
        "keywords": ["safe", "safety", "danger", "health", "vaccine", "insurance", "scam", "warning"],
        "output_template": """
## Safety Overview
[Current safety level and advisories]

## Health Requirements
- Vaccinations: [Required/recommended]
- Medications: [What to bring]
- Insurance: [Recommendations]

## Common Scams & How to Avoid Them
[Specific warnings]

## Emergency Information
- Embassy: [Contact]
- Police: [Number]
- Hospital: [Nearest to tourist areas]

## Areas to Avoid
[Specific unsafe areas if any]
"""
    },

    "weather_planning": {
        "name": "Weather & Timing",
        "description": "Best time to visit and weather info",
        "icon": "🌤️",
        "experts": ["Weather Analyst", "Activity Curator", "Budget Advisor"],
        "synthesis_focus": "timing_recommendation",
        "keywords": ["weather", "best time", "when to visit", "season", "climate", "rain", "temperature"],
        "output_template": """
## Weather Forecast
[Expected conditions for travel dates]

## Best Time to Visit
[Optimal months and why]

## Seasonal Considerations
- Peak Season: [When, pros/cons]
- Shoulder Season: [When, pros/cons]
- Off Season: [When, pros/cons]

## Packing Recommendations
[What to bring based on weather]

## Weather-Dependent Activities
[What might be affected]
"""
    },

    "flight_search": {
        "name": "Flight Search",
        "description": "Find flights and booking tips",
        "icon": "🛫",
        "experts": ["Logistics Planner", "Budget Advisor"],
        "synthesis_focus": "flight_recommendations",
        "keywords": ["flight", "fly", "airline", "airport", "booking", "ticket"],
        "output_template": """
## Flight Options
[Available routes and airlines]

## Price Comparison
| Route | Airline | Price | Duration |
|-------|---------|-------|----------|
| | | | |

## Booking Tips
[Best time to book, platforms to use]

## Airport Information
[Terminals, transfers, lounges]
"""
    },

    "general": {
        "name": "General Travel Question",
        "description": "Any travel-related question",
        "icon": "❓",
        "experts": ["Budget Advisor", "Logistics Planner", "Local Culture Guide", "Activity Curator"],
        "synthesis_focus": "general_advice",
        "keywords": [],  # Fallback for unmatched queries
        "output_template": """
## Summary
[Direct answer to the question]

## Details
[Supporting information]

## Practical Tips
[Actionable recommendations]

## Additional Considerations
[Related things to think about]
"""
    },
}


def detect_travel_question_type(question: str) -> str:
    """
    Detect the type of travel question based on keywords.

    Args:
        question: User's travel question

    Returns:
        Question type key (e.g., 'destination_planning', 'budget_optimization')
    """
    question_lower = question.lower()

    # Check each question type's keywords
    for qtype, config in TRAVEL_QUESTION_TYPES.items():
        if qtype == "general":
            continue  # Skip general, it's the fallback

        keywords = config.get("keywords", [])
        for keyword in keywords:
            if keyword in question_lower:
                return qtype

    # Check for destination patterns (e.g., "trip to Paris", "visiting Tokyo")
    destination_patterns = [
        r"trip to \w+",
        r"travel to \w+",
        r"visit(?:ing)? \w+",
        r"going to \w+",
        r"vacation in \w+",
        r"holiday in \w+",
    ]
    for pattern in destination_patterns:
        if re.search(pattern, question_lower):
            return "destination_planning"

    # Default to general
    return "general"


def get_experts_for_question_type(question_type: str) -> List[str]:
    """
    Get recommended experts for a question type.

    Args:
        question_type: Type key from TRAVEL_QUESTION_TYPES

    Returns:
        List of expert names
    """
    config = TRAVEL_QUESTION_TYPES.get(question_type, TRAVEL_QUESTION_TYPES["general"])
    return config.get("experts", [])


def get_output_template(question_type: str) -> str:
    """
    Get the output template for a question type.

    Args:
        question_type: Type key from TRAVEL_QUESTION_TYPES

    Returns:
        Output template string
    """
    config = TRAVEL_QUESTION_TYPES.get(question_type, TRAVEL_QUESTION_TYPES["general"])
    return config.get("output_template", "")


def get_question_type_info(question_type: str) -> Dict:
    """
    Get full configuration for a question type.

    Args:
        question_type: Type key from TRAVEL_QUESTION_TYPES

    Returns:
        Question type configuration dict
    """
    return TRAVEL_QUESTION_TYPES.get(question_type, TRAVEL_QUESTION_TYPES["general"])


def get_all_question_types() -> List[Dict]:
    """
    Get all question types with their info for UI display.

    Returns:
        List of question type configs
    """
    return [
        {"key": key, **config}
        for key, config in TRAVEL_QUESTION_TYPES.items()
    ]
