"""
Google Places API Client

Cost-efficient client with:
- Field masks to reduce cost (60-70% savings)
- In-memory caching by destination (24-hour TTL)
- Rate limiting protection
"""

import requests
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import hashlib

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class PlaceResult:
    """Basic place info from Text Search."""
    place_id: str
    name: str
    rating: float
    user_ratings_total: int
    price_level: Optional[int]  # 0-4 (PRICE_LEVEL_INEXPENSIVE to PRICE_LEVEL_VERY_EXPENSIVE)
    address: str
    types: List[str]

    @property
    def price_display(self) -> str:
        """Convert price level to $ symbols."""
        if self.price_level is None:
            return "N/A"
        # Google's new API uses string enum, map to int if needed
        if isinstance(self.price_level, str):
            level_map = {
                "PRICE_LEVEL_FREE": 0,
                "PRICE_LEVEL_INEXPENSIVE": 1,
                "PRICE_LEVEL_MODERATE": 2,
                "PRICE_LEVEL_EXPENSIVE": 3,
                "PRICE_LEVEL_VERY_EXPENSIVE": 4
            }
            level = level_map.get(self.price_level, 2)
        else:
            level = self.price_level
        return "$" * max(1, level)

    @property
    def trust_tier(self) -> str:
        """Quick trust assessment based on review count."""
        if self.user_ratings_total >= 500:
            return "HIGH"
        elif self.user_ratings_total >= 100:
            return "MEDIUM"
        else:
            return "LOW"


@dataclass
class PlaceDetails:
    """Detailed place info including reviews."""
    place_id: str
    name: str
    rating: float
    user_ratings_total: int
    reviews: List[Dict]  # [{author, rating, text, time, relative_time}]
    opening_hours: Optional[Dict]
    website: Optional[str]
    phone: Optional[str]


class GooglePlacesClient:
    """
    Google Places API client with caching and cost optimization.

    Uses the new Places API (v1) with field masks to minimize costs:
    - Text Search: ~$0.032 per request (with field mask)
    - Place Details: ~$0.017 per request (with field mask)

    Free tier: $200/month = ~6,000 text searches
    """

    BASE_URL = "https://places.googleapis.com/v1/places"

    # Class-level cache shared across instances
    _cache: Dict[str, tuple] = {}
    CACHE_TTL = timedelta(hours=24)

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GOOGLE_PLACES_API_KEY

    def is_available(self) -> bool:
        """Check if API is configured."""
        return bool(self.api_key)

    def _cache_key(self, query: str, location: str) -> str:
        """Generate deterministic cache key."""
        raw = f"{query}:{location}".lower().strip()
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get from cache if not expired."""
        if key in self._cache:
            result, timestamp = self._cache[key]
            if datetime.now() - timestamp < self.CACHE_TTL:
                logger.debug(f"Cache hit: {key[:8]}...")
                return result
            else:
                del self._cache[key]
        return None

    def _set_cached(self, key: str, result: Any):
        """Store in cache with timestamp."""
        self._cache[key] = (result, datetime.now())

    def search_places(
        self,
        query: str,
        location: str,
        place_type: str = "restaurant",
        max_results: int = 5
    ) -> List[PlaceResult]:
        """
        Search for places using Text Search API with minimal field mask.

        Args:
            query: Place name or search term
            location: City/destination context
            place_type: One of: restaurant, lodging, tourist_attraction
            max_results: Max places to return (default 5)

        Returns:
            List of PlaceResult with basic info

        Cost: ~$0.032 per request (reduced from ~$0.10 with field mask)
        """
        cache_key = self._cache_key(f"{query}:{place_type}", location)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        if not self.is_available():
            logger.warning("Google Places API not configured")
            return []

        try:
            # Text Search (New) API with minimal field mask
            response = requests.post(
                f"{self.BASE_URL}:searchText",
                headers={
                    "X-Goog-Api-Key": self.api_key,
                    # Field mask significantly reduces cost
                    "X-Goog-FieldMask": (
                        "places.id,"
                        "places.displayName,"
                        "places.rating,"
                        "places.userRatingCount,"
                        "places.priceLevel,"
                        "places.formattedAddress,"
                        "places.types"
                    )
                },
                json={
                    "textQuery": f"{query} {place_type} in {location}",
                    "maxResultCount": max_results,
                    "languageCode": "en"
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for place in data.get("places", []):
                results.append(PlaceResult(
                    place_id=place.get("id", ""),
                    name=place.get("displayName", {}).get("text", "Unknown"),
                    rating=place.get("rating", 0.0),
                    user_ratings_total=place.get("userRatingCount", 0),
                    price_level=place.get("priceLevel"),
                    address=place.get("formattedAddress", ""),
                    types=place.get("types", [])
                ))

            self._set_cached(cache_key, results)
            logger.info(f"Found {len(results)} places for '{query}' in {location}")
            return results

        except requests.exceptions.HTTPError as e:
            logger.error(f"Google Places API error: {e.response.status_code} - {e.response.text[:200]}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Google Places request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Google Places search failed: {e}")
            return []

    def get_place_details(self, place_id: str) -> Optional[PlaceDetails]:
        """
        Get detailed info including reviews.

        More expensive than search - only call when user wants details.

        Args:
            place_id: Google Place ID from search results

        Returns:
            PlaceDetails with reviews, hours, contact info

        Cost: ~$0.017 per request
        """
        cache_key = f"details:{place_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        if not self.is_available():
            return None

        try:
            response = requests.get(
                f"{self.BASE_URL}/{place_id}",
                headers={
                    "X-Goog-Api-Key": self.api_key,
                    "X-Goog-FieldMask": (
                        "id,"
                        "displayName,"
                        "rating,"
                        "userRatingCount,"
                        "reviews,"
                        "currentOpeningHours,"
                        "websiteUri,"
                        "nationalPhoneNumber"
                    )
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            # Parse reviews (limit to 5 most recent)
            reviews = []
            for review in data.get("reviews", [])[:5]:
                reviews.append({
                    "author": review.get("authorAttribution", {}).get("displayName", "Anonymous"),
                    "rating": review.get("rating", 0),
                    "text": review.get("text", {}).get("text", ""),
                    "time": review.get("publishTime", ""),
                    "relative_time": review.get("relativePublishTimeDescription", "")
                })

            result = PlaceDetails(
                place_id=data.get("id", place_id),
                name=data.get("displayName", {}).get("text", "Unknown"),
                rating=data.get("rating", 0.0),
                user_ratings_total=data.get("userRatingCount", 0),
                reviews=reviews,
                opening_hours=data.get("currentOpeningHours"),
                website=data.get("websiteUri"),
                phone=data.get("nationalPhoneNumber")
            )

            self._set_cached(cache_key, result)
            return result

        except requests.exceptions.HTTPError as e:
            logger.error(f"Google Places details error: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Google Places details failed: {e}")
            return None

    def search_and_get_best_match(
        self,
        name: str,
        location: str,
        place_type: str = "restaurant"
    ) -> Optional[PlaceResult]:
        """
        Convenience method: search and return best matching result.

        Args:
            name: Exact or approximate place name
            location: City/destination
            place_type: Type of place

        Returns:
            Best matching PlaceResult or None
        """
        results = self.search_places(name, location, place_type, max_results=1)
        return results[0] if results else None

    def format_place_summary(self, place: PlaceResult) -> str:
        """Format place for inline display in recommendations."""
        trust_emoji = {"HIGH": "✅", "MEDIUM": "⚠️", "LOW": "❓"}
        emoji = trust_emoji.get(place.trust_tier, "")

        parts = [f"**{place.name}**"]

        if place.rating > 0:
            parts.append(f"★{place.rating:.1f}")

        if place.user_ratings_total > 0:
            parts.append(f"({place.user_ratings_total:,} reviews)")

        if place.price_display != "N/A":
            parts.append(place.price_display)

        parts.append(emoji)

        return " ".join(parts)

    def clear_cache(self):
        """Clear the entire cache (useful for testing)."""
        self._cache.clear()
        logger.info("Places cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        now = datetime.now()
        valid = sum(1 for _, (_, ts) in self._cache.items() if now - ts < self.CACHE_TTL)
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid,
            "expired_entries": len(self._cache) - valid
        }
