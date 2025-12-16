"""
External API Integrations for Travel Planner

Modules:
- weather: OpenWeatherMap API for weather forecasts
- amadeus_flights: Amadeus API for flight search
- amadeus_cars: Amadeus API for car rentals
- google_places: Google Places API for ratings and reviews
- google_search: Google Search/Maps grounding for hotel data
"""

from integrations.weather import OpenWeatherClient
from integrations.amadeus_flights import AmadeusClient
from integrations.amadeus_cars import AmadeusCarClient
from integrations.google_places import GooglePlacesClient
from integrations.google_search import GoogleSearchClient, TravelGroundingClient

__all__ = [
    'OpenWeatherClient',
    'AmadeusClient',
    'AmadeusCarClient',
    'GooglePlacesClient',
    'GoogleSearchClient',
    'TravelGroundingClient',
]
