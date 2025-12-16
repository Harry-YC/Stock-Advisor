"""
Travel Data Service

Fetches real data from Weather, Flight, Car Rental, and Hotel APIs based on form inputs.
Returns formatted context for expert consumption.

Data Sources:
- Weather: OpenWeatherMap API
- Flights: Amadeus API
- Car Rentals: Amadeus API
- Hotels/Places: Google Maps Grounding (Gemini 2.5)
"""

from typing import Dict, Optional
from datetime import date
import logging

from config import settings

logger = logging.getLogger(__name__)


class TravelDataService:
    """Fetches and formats real travel data for expert context."""

    def __init__(self):
        self._weather_client = None
        self._flight_client = None
        self._car_client = None

    @property
    def weather_client(self):
        if self._weather_client is None:
            from integrations.weather import OpenWeatherClient
            self._weather_client = OpenWeatherClient()
        return self._weather_client

    @property
    def flight_client(self):
        if self._flight_client is None:
            from integrations.amadeus_flights import AmadeusClient
            self._flight_client = AmadeusClient()
        return self._flight_client

    @property
    def car_client(self):
        if self._car_client is None:
            from integrations.amadeus_cars import AmadeusCarClient
            self._car_client = AmadeusCarClient()
        return self._car_client

    def fetch_travel_data(
        self,
        destination: str,
        origin: Optional[str],
        departure_date: date,
        return_date: date,
        travelers: str,
        budget: int
    ) -> Dict[str, str]:
        """
        Fetch all available travel data.

        Returns:
            Dict with keys: 'weather', 'flights', 'car_rentals', 'hotels', 'summary'
        """
        results = {
            "weather": "",
            "flights": "",
            "car_rentals": "",
            "hotels": "",
            "summary": ""
        }

        # Parse travelers
        adults = self._parse_travelers(travelers)

        # Fetch weather
        if self.weather_client.is_available():
            try:
                weather_data = self._fetch_weather(destination, departure_date, return_date)
                results["weather"] = weather_data
                logger.info(f"Weather data fetched for {destination}")
            except Exception as e:
                logger.warning(f"Weather fetch failed: {e}")

        # Fetch flights
        if origin and self.flight_client.is_available():
            try:
                flight_data = self._fetch_flights(
                    origin, destination,
                    departure_date, return_date,
                    adults
                )
                results["flights"] = flight_data
                logger.info(f"Flight data fetched for {origin} -> {destination}")
            except Exception as e:
                logger.warning(f"Flight fetch failed: {e}")

        # Fetch car rentals
        try:
            car_data = self._fetch_car_rentals(
                destination, departure_date, return_date
            )
            results["car_rentals"] = car_data
            if car_data:
                logger.info(f"Car rental data fetched for {destination}")
        except Exception as e:
            logger.warning(f"Car rental fetch failed: {e}")

        # Fetch hotels using Maps Grounding (Gemini 2.5 Flash)
        if settings.ENABLE_MAPS_GROUNDING:
            try:
                hotel_data = self._fetch_hotels_with_grounding(
                    destination, departure_date, return_date, adults, budget
                )
                results["hotels"] = hotel_data
                if hotel_data:
                    logger.info(f"Hotel data fetched for {destination} via Maps Grounding")
            except Exception as e:
                logger.warning(f"Hotel fetch failed: {e}")

        # Build summary
        results["summary"] = self._build_summary(
            destination, departure_date, return_date,
            travelers, budget, results
        )

        return results

    def _parse_travelers(self, travelers: str) -> int:
        """Extract adult count from travelers string."""
        if "1 adult" in travelers:
            return 1
        elif "2 adults" in travelers:
            return 2
        return 2  # default

    def _fetch_weather(self, destination: str, start: date, end: date) -> str:
        """Fetch weather forecast for destination."""
        days = (end - start).days + 1
        days = min(days, 5)  # OpenWeatherMap free tier limit

        forecasts = self.weather_client.get_forecast(city=destination, days=days)
        if not forecasts:
            return ""

        return self.weather_client.format_forecast(forecasts)

    def _fetch_flights(
        self, origin: str, destination: str,
        departure: date, return_date: date,
        adults: int
    ) -> str:
        """Fetch flight options."""
        # Get IATA codes by searching airports
        origin_airports = self.flight_client.search_airports(origin, max_results=1)
        dest_airports = self.flight_client.search_airports(destination, max_results=1)

        if not origin_airports or not dest_airports:
            logger.warning(f"Could not find airports for {origin} or {destination}")
            return ""

        origin_code = origin_airports[0].get('iataCode')
        dest_code = dest_airports[0].get('iataCode')

        if not origin_code or not dest_code:
            return ""

        flights = self.flight_client.search_flights(
            origin=origin_code,
            destination=dest_code,
            departure_date=departure.isoformat(),
            return_date=return_date.isoformat(),
            adults=adults,
            max_results=5
        )

        if not flights:
            return ""

        return self.flight_client.format_flight_results(flights)

    def _fetch_car_rentals(
        self, destination: str, pickup: date, dropoff: date
    ) -> str:
        """Fetch car rental options for destination."""
        recommendations = self.car_client.get_top_recommendations(
            pickup_location=destination,
            pickup_date=pickup.isoformat(),
            dropoff_date=dropoff.isoformat(),
            top_n=3
        )

        if not recommendations:
            return ""

        return self.car_client.format_car_results(recommendations)

    def _fetch_hotels_with_grounding(
        self, destination: str, checkin: date, checkout: date,
        adults: int, budget: int
    ) -> str:
        """
        Fetch hotel recommendations using Google Maps Grounding.

        Uses Gemini 2.5 Flash with Maps grounding to get real hotel data
        including ratings, reviews, and prices from Google Maps.
        """
        try:
            from integrations.google_search import TravelGroundingClient
        except ImportError:
            logger.warning("TravelGroundingClient not available")
            return ""

        client = TravelGroundingClient()
        if not client.is_available():
            return ""

        nights = (checkout - checkin).days
        per_night_budget = budget // (nights * 2) if nights > 0 else budget // 2  # ~50% of budget for hotels

        prompt = f"""Find the TOP 5 hotels in {destination} for {adults} adults.
Check-in: {checkin.strftime('%B %d, %Y')}
Check-out: {checkout.strftime('%B %d, %Y')} ({nights} nights)
Budget: around ${per_night_budget}/night

For each hotel provide:
1. Hotel name and star rating
2. Google rating and number of reviews
3. Price per night (if available)
4. Key amenities and location highlights
5. Why it's recommended

Focus on hotels with high ratings (4.0+) and good value for the budget."""

        try:
            result = client.generate_with_travel_grounding(
                prompt=prompt,
                destination=destination,
                include_maps=True,
                include_search=False,  # Maps only, no web search
                model=settings.MAPS_GROUNDING_MODEL
            )
            return result.content if result.content else ""
        except Exception as e:
            logger.warning(f"Maps grounding hotel fetch failed: {e}")
            return ""

    def _build_summary(
        self, destination: str, start: date, end: date,
        travelers: str, budget: int, data: Dict
    ) -> str:
        """Build a summary context block for experts."""
        days = (end - start).days

        summary = f"""## TRIP PARAMETERS
- **Destination**: {destination}
- **Dates**: {start.strftime('%b %d')} - {end.strftime('%b %d, %Y')} ({days} nights)
- **Travelers**: {travelers}
- **Budget**: ${budget:,} USD

"""
        if data.get("weather"):
            summary += f"""## WEATHER FORECAST (Real Data)
{data['weather']}

"""

        if data.get("flights"):
            summary += f"""## FLIGHT OPTIONS (Real Prices)
{data['flights']}

"""

        if data.get("car_rentals"):
            summary += f"""## CAR RENTAL OPTIONS (Top 3 Recommendations)
{data['car_rentals']}

"""

        if data.get("hotels"):
            summary += f"""## HOTEL OPTIONS (Google Maps Data)
{data['hotels']}

"""

        return summary


def get_travel_data_context(trip_form: Dict) -> str:
    """
    Convenience function to get formatted context from form data.

    Args:
        trip_form: Dict with keys: destination, origin, departure_date,
                   return_date, travelers, budget

    Returns:
        Formatted string for expert context
    """
    if not trip_form.get("destination"):
        return ""

    service = TravelDataService()
    data = service.fetch_travel_data(
        destination=trip_form["destination"],
        origin=trip_form.get("origin"),
        departure_date=trip_form["departure_date"],
        return_date=trip_form["return_date"],
        travelers=trip_form["travelers"],
        budget=trip_form["budget"]
    )

    return data.get("summary", "")
