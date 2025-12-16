"""
Amadeus Self-Service API Integration

Flight and hotel search using Amadeus for Developers API.
https://developers.amadeus.com/

Features:
- Flight search (one-way, round-trip)
- Hotel search by city
- Airport/city lookup
"""

import os
import logging
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class FlightOffer:
    """Flight offer from Amadeus."""
    id: str
    price: float
    currency: str
    airline: str
    airline_name: str
    departure_airport: str
    arrival_airport: str
    departure_time: str
    arrival_time: str
    duration: str
    stops: int
    segments: List[Dict]
    booking_class: str = "ECONOMY"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "price": self.price,
            "currency": self.currency,
            "airline": self.airline,
            "airline_name": self.airline_name,
            "departure_airport": self.departure_airport,
            "arrival_airport": self.arrival_airport,
            "departure_time": self.departure_time,
            "arrival_time": self.arrival_time,
            "duration": self.duration,
            "stops": self.stops,
            "booking_class": self.booking_class,
        }


@dataclass
class HotelOffer:
    """Hotel offer from Amadeus."""
    hotel_id: str
    name: str
    rating: Optional[int]
    price: float
    currency: str
    check_in: str
    check_out: str
    room_type: str
    address: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hotel_id": self.hotel_id,
            "name": self.name,
            "rating": self.rating,
            "price": self.price,
            "currency": self.currency,
            "check_in": self.check_in,
            "check_out": self.check_out,
            "room_type": self.room_type,
            "address": self.address,
        }


class AmadeusClient:
    """
    Client for Amadeus Self-Service API.

    Handles authentication and provides flight/hotel search.
    """

    AUTH_URL = "https://api.amadeus.com/v1/security/oauth2/token"
    FLIGHT_SEARCH_URL = "https://api.amadeus.com/v2/shopping/flight-offers"
    HOTEL_SEARCH_URL = "https://api.amadeus.com/v1/reference-data/locations/hotels/by-city"
    CITY_SEARCH_URL = "https://api.amadeus.com/v1/reference-data/locations"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        self.api_key = api_key or settings.AMADEUS_API_KEY
        self.api_secret = api_secret or settings.AMADEUS_API_SECRET
        self._access_token = None
        self._token_expires = None

    def is_available(self) -> bool:
        """Check if Amadeus credentials are configured."""
        return bool(self.api_key and self.api_secret)

    def _get_access_token(self) -> str:
        """Get or refresh OAuth2 access token."""
        # Return cached token if still valid
        if self._access_token and self._token_expires:
            if datetime.now() < self._token_expires:
                return self._access_token

        # Request new token
        try:
            response = requests.post(
                self.AUTH_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.api_key,
                    "client_secret": self.api_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            self._access_token = data["access_token"]
            # Token typically expires in 1799 seconds, refresh 5 min early
            expires_in = data.get("expires_in", 1799) - 300
            self._token_expires = datetime.now() + timedelta(seconds=expires_in)

            logger.info("Amadeus access token refreshed")
            return self._access_token

        except requests.RequestException as e:
            logger.error(f"Amadeus authentication failed: {e}")
            raise RuntimeError(f"Amadeus authentication failed: {e}")

    def _make_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict:
        """Make authenticated request to Amadeus API."""
        if not self.is_available():
            raise RuntimeError("Amadeus API credentials not configured")

        token = self._get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=30)
            else:
                response = requests.post(url, json=json_data, headers=headers, timeout=30)

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.error(f"Amadeus API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: Optional[str] = None,
        adults: int = 1,
        travel_class: str = "ECONOMY",
        max_results: int = 10,
        non_stop: bool = False,
    ) -> List[FlightOffer]:
        """
        Search for flight offers.

        Args:
            origin: Origin airport IATA code (e.g., 'LAX')
            destination: Destination airport IATA code (e.g., 'JFK')
            departure_date: Departure date (YYYY-MM-DD)
            return_date: Return date for round-trip (YYYY-MM-DD), optional
            adults: Number of adult passengers
            travel_class: ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST
            max_results: Maximum number of offers to return
            non_stop: Only show non-stop flights

        Returns:
            List of FlightOffer objects
        """
        params = {
            "originLocationCode": origin.upper(),
            "destinationLocationCode": destination.upper(),
            "departureDate": departure_date,
            "adults": adults,
            "travelClass": travel_class,
            "max": max_results,
            "currencyCode": "USD",
        }

        if return_date:
            params["returnDate"] = return_date
        if non_stop:
            params["nonStop"] = "true"

        try:
            data = self._make_request("GET", self.FLIGHT_SEARCH_URL, params=params)
            offers = []

            for offer in data.get("data", []):
                # Parse the first itinerary (outbound)
                itinerary = offer.get("itineraries", [{}])[0]
                segments = itinerary.get("segments", [])

                if not segments:
                    continue

                first_segment = segments[0]
                last_segment = segments[-1]

                # Get airline info from dictionaries
                carrier_code = first_segment.get("carrierCode", "")
                dictionaries = data.get("dictionaries", {})
                airline_name = dictionaries.get("carriers", {}).get(carrier_code, carrier_code)

                flight_offer = FlightOffer(
                    id=offer.get("id", ""),
                    price=float(offer.get("price", {}).get("total", 0)),
                    currency=offer.get("price", {}).get("currency", "USD"),
                    airline=carrier_code,
                    airline_name=airline_name,
                    departure_airport=first_segment.get("departure", {}).get("iataCode", ""),
                    arrival_airport=last_segment.get("arrival", {}).get("iataCode", ""),
                    departure_time=first_segment.get("departure", {}).get("at", ""),
                    arrival_time=last_segment.get("arrival", {}).get("at", ""),
                    duration=itinerary.get("duration", ""),
                    stops=len(segments) - 1,
                    segments=segments,
                    booking_class=travel_class,
                )
                offers.append(flight_offer)

            logger.info(f"Found {len(offers)} flight offers from {origin} to {destination}")
            return offers

        except Exception as e:
            logger.error(f"Flight search failed: {e}")
            return []

    def search_airports(
        self,
        keyword: str,
        max_results: int = 5
    ) -> List[Dict]:
        """
        Search for airports/cities by keyword.

        Args:
            keyword: Search term (city name, airport code, etc.)
            max_results: Maximum results to return

        Returns:
            List of location dictionaries with iataCode, name, cityName
        """
        params = {
            "keyword": keyword,
            "subType": "AIRPORT,CITY",
            "page[limit]": max_results,
        }

        try:
            data = self._make_request("GET", self.CITY_SEARCH_URL, params=params)
            locations = []

            for loc in data.get("data", []):
                locations.append({
                    "iataCode": loc.get("iataCode", ""),
                    "name": loc.get("name", ""),
                    "cityName": loc.get("address", {}).get("cityName", ""),
                    "countryCode": loc.get("address", {}).get("countryCode", ""),
                    "type": loc.get("subType", ""),
                })

            return locations

        except Exception as e:
            logger.error(f"Airport search failed: {e}")
            return []

    def format_flight_results(self, offers: List[FlightOffer]) -> str:
        """Format flight offers for display."""
        if not offers:
            return "No flights found."

        lines = ["## Flight Options\n"]
        lines.append("| Airline | Route | Departure | Duration | Stops | Price |")
        lines.append("|---------|-------|-----------|----------|-------|-------|")

        for offer in offers[:10]:
            dep_time = offer.departure_time.split("T")[1][:5] if "T" in offer.departure_time else offer.departure_time
            route = f"{offer.departure_airport}→{offer.arrival_airport}"
            stops = "Non-stop" if offer.stops == 0 else f"{offer.stops} stop(s)"
            duration = offer.duration.replace("PT", "").replace("H", "h ").replace("M", "m")

            lines.append(
                f"| {offer.airline_name[:15]} | {route} | {dep_time} | {duration} | {stops} | ${offer.price:.0f} |"
            )

        return "\n".join(lines)


# Convenience function
def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
) -> List[FlightOffer]:
    """
    Quick flight search.

    Args:
        origin: Origin airport code
        destination: Destination airport code
        departure_date: YYYY-MM-DD
        return_date: Optional return date
        adults: Number of passengers

    Returns:
        List of FlightOffer objects
    """
    client = AmadeusClient()
    if not client.is_available():
        logger.warning("Amadeus API not configured")
        return []

    return client.search_flights(
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        return_date=return_date,
        adults=adults,
    )
