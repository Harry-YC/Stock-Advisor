"""
Budget Estimation Service for Travel Planner

Uses Gemini-3-pro-preview to estimate travel budgets for top 15% travelers
based on destination, duration, travelers, and trip characteristics.
"""

import json
import logging
import re
from datetime import date
from typing import Any, Dict, List, Optional

from services.llm_router import get_llm_router

logger = logging.getLogger(__name__)


class BudgetEstimationService:
    """Estimates travel budgets using Gemini-3-pro-preview for reasoning."""

    # Required model for accurate budget reasoning
    MODEL = "gemini-3-pro-preview"

    # Default fallback if LLM fails
    DEFAULT_BUDGET = 5000
    DEFAULT_CURRENCY = "USD"

    def __init__(self):
        self.router = get_llm_router()

    def estimate_top_15_percent_budget(
        self,
        destination: str,
        origin: Optional[str],
        departure_date: date,
        return_date: date,
        num_travelers: int,
        travel_interests: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Use Gemini-3-pro-preview to estimate what top 15% travelers spend.

        Args:
            destination: Trip destination (city, country, or region)
            origin: Departure city/country (for flight estimation)
            departure_date: Trip start date
            return_date: Trip end date
            num_travelers: Number of people traveling
            travel_interests: Optional list of interests (food, adventure, culture, etc.)

        Returns:
            Dict with:
            - total_budget: int (in USD)
            - currency: str ("USD")
            - breakdown: Dict with category totals
            - per_person_per_day: int
            - rationale: str (brief explanation)
        """
        duration_days = (return_date - departure_date).days
        if duration_days <= 0:
            duration_days = 1

        interests_str = ", ".join(travel_interests) if travel_interests else "general sightseeing"
        origin_str = origin if origin else "unknown origin"

        prompt = f"""You are a luxury travel budget analyst. Estimate what TOP 15% of travelers (affluent, quality-focused) would spend on this trip:

- Destination: {destination}
- Origin: {origin_str}
- Dates: {departure_date.strftime('%B %d, %Y')} to {return_date.strftime('%B %d, %Y')} ({duration_days} days)
- Travelers: {num_travelers} {'person' if num_travelers == 1 else 'people'}
- Interests: {interests_str}

Top 15% travelers typically:
- Fly business/premium economy for long-haul flights, economy+ for short-haul
- Stay at 4-5 star hotels or luxury boutique properties
- Dine at quality restaurants (not necessarily Michelin, but good local spots)
- Book guided tours and premium experiences
- Don't penny-pinch but are value-conscious

Consider:
- Flight costs based on origin-destination distance and class
- Accommodation costs for the destination's luxury market
- Local cost of living for food and activities
- Typical tourist activities and experiences

Return ONLY valid JSON (no markdown, no explanation):
{{"total_budget": <integer in USD>, "currency": "USD", "breakdown": {{"flights": <int>, "accommodation": <int>, "food": <int>, "activities": <int>, "transport": <int>, "misc": <int>}}, "per_person_per_day": <int>, "rationale": "<one sentence explaining the estimate>"}}"""

        system_prompt = """You are a travel budget analyst specializing in luxury travel.
Provide accurate budget estimates based on current travel costs.
Always respond with valid JSON only, no additional text."""

        try:
            response = self.router.call_expert(
                prompt=prompt,
                system=system_prompt,
                model=self.MODEL,
                temperature=0.3,  # Lower temperature for more consistent estimates
                max_tokens=500,
                use_fallback=False  # Must use gemini-3-pro-preview for reasoning
            )

            # Parse JSON response
            result = self._parse_budget_response(response.content)

            if result:
                logger.info(
                    f"Budget estimated for {destination}: ${result['total_budget']:,} "
                    f"({result['rationale']})"
                )
                return result
            else:
                logger.warning("Failed to parse budget response, using default")
                return self._get_default_estimate(destination, duration_days, num_travelers)

        except Exception as e:
            logger.error(f"Budget estimation failed: {e}", exc_info=True)
            return self._get_default_estimate(destination, duration_days, num_travelers)

    def _parse_budget_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into budget dict."""
        try:
            # Try to extract JSON from response
            content = content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = re.sub(r"```(?:json)?\s*", "", content)
                content = content.rstrip("`").strip()

            result = json.loads(content)

            # Validate required fields
            if not isinstance(result.get("total_budget"), (int, float)):
                return None
            if not isinstance(result.get("breakdown"), dict):
                return None

            # Ensure total_budget is int
            result["total_budget"] = int(result["total_budget"])

            # Ensure breakdown values are ints
            for key in result["breakdown"]:
                result["breakdown"][key] = int(result["breakdown"][key])

            # Ensure per_person_per_day is int
            if "per_person_per_day" in result:
                result["per_person_per_day"] = int(result["per_person_per_day"])

            # Ensure currency is set
            if "currency" not in result:
                result["currency"] = "USD"

            # Ensure rationale is set
            if "rationale" not in result:
                result["rationale"] = "Estimated based on top 15% traveler spending patterns"

            return result

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse budget JSON: {e}")
            return None

    def _get_default_estimate(
        self,
        destination: str,
        duration_days: int,
        num_travelers: int
    ) -> Dict[str, Any]:
        """Return a reasonable default estimate when LLM fails."""
        # Base daily rate for top 15% traveler (per person)
        base_daily_rate = 400  # USD per person per day

        # Adjust for destination (rough multipliers)
        destination_lower = destination.lower()
        if any(x in destination_lower for x in ["switzerland", "iceland", "norway", "maldives"]):
            base_daily_rate = 600
        elif any(x in destination_lower for x in ["japan", "australia", "uk", "france", "italy"]):
            base_daily_rate = 450
        elif any(x in destination_lower for x in ["thailand", "vietnam", "mexico", "indonesia"]):
            base_daily_rate = 250

        # Calculate totals
        accommodation = int(base_daily_rate * 0.45 * duration_days * (num_travelers / 2 + 0.5))  # Shared rooms
        food = int(base_daily_rate * 0.25 * duration_days * num_travelers)
        activities = int(base_daily_rate * 0.20 * duration_days * num_travelers)
        transport = int(base_daily_rate * 0.10 * duration_days * num_travelers)
        flights = int(2000 * num_travelers)  # Rough estimate for premium flights
        misc = int((accommodation + food + activities + transport) * 0.1)

        total = accommodation + food + activities + transport + flights + misc

        return {
            "total_budget": total,
            "currency": "USD",
            "breakdown": {
                "flights": flights,
                "accommodation": accommodation,
                "food": food,
                "activities": activities,
                "transport": transport,
                "misc": misc
            },
            "per_person_per_day": int(total / duration_days / num_travelers),
            "rationale": f"Default estimate for {duration_days}-day trip to {destination}"
        }
