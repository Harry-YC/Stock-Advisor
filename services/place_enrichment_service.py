"""
Place Enrichment Service

Orchestrates Google Places lookups and trust filtering for travel recommendations.

Features:
- Batch place lookups with caching
- Trust scoring based on review patterns
- Sample review extraction for high-trust places
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from integrations.google_places import GooglePlacesClient, PlaceResult, PlaceDetails
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class EnrichedPlace:
    """Place with ratings and trust assessment."""
    name: str
    rating: float
    review_count: int
    price_level: str
    trust_score: str  # HIGH, MEDIUM, LOW, SUSPICIOUS, NOT_FOUND, UNKNOWN
    trust_reason: str
    sample_review: Optional[str]
    address: str
    place_id: Optional[str] = None

    @property
    def trust_emoji(self) -> str:
        """Get emoji for trust level."""
        emoji_map = {
            "HIGH": "✅",
            "MEDIUM": "⚠️",
            "LOW": "❓",
            "SUSPICIOUS": "🚩",
            "NOT_FOUND": "❌",
            "UNKNOWN": "❔"
        }
        return emoji_map.get(self.trust_score, "")


class PlaceEnrichmentService:
    """
    Enriches place recommendations with Google Places ratings and trust scores.

    Trust Scoring Algorithm:
    - HIGH (✅): 200+ reviews, 4.0+ rating, consistent patterns
    - MEDIUM (⚠️): 50-200 reviews, decent rating
    - LOW (❓): <50 reviews, uncertain quality
    - SUSPICIOUS (🚩): Perfect 5.0 rating with low volume (possible fakes)
    - NOT_FOUND (❌): Place not on Google Maps
    """

    # Trust thresholds
    MIN_REVIEWS_HIGH_TRUST = 200
    MIN_REVIEWS_MEDIUM_TRUST = 50
    MIN_RATING = 3.5

    def __init__(self):
        self._places_client = None

    @property
    def places_client(self) -> GooglePlacesClient:
        """Lazy-load places client."""
        if self._places_client is None:
            self._places_client = GooglePlacesClient()
        return self._places_client

    def is_available(self) -> bool:
        """Check if enrichment service is available."""
        return self.places_client.is_available()

    def enrich_recommendations(
        self,
        place_names: List[str],
        destination: str,
        place_type: str = "restaurant"
    ) -> List[EnrichedPlace]:
        """
        Look up places and add trust scores.

        Args:
            place_names: List of place names from expert recommendations
            destination: City/location context (e.g., "Barcelona, Spain")
            place_type: One of: restaurant, lodging, tourist_attraction

        Returns:
            List of EnrichedPlace with ratings and trust info
        """
        if not self.is_available():
            logger.warning("Google Places not available, returning unenriched")
            return [
                EnrichedPlace(
                    name=name,
                    rating=0,
                    review_count=0,
                    price_level="N/A",
                    trust_score="UNKNOWN",
                    trust_reason="Google Places API not configured",
                    sample_review=None,
                    address=""
                )
                for name in place_names
            ]

        enriched = []
        for name in place_names:
            place = self._enrich_single_place(name, destination, place_type)
            enriched.append(place)

        return enriched

    def _enrich_single_place(
        self,
        name: str,
        destination: str,
        place_type: str
    ) -> EnrichedPlace:
        """Enrich a single place with Google data."""
        # Search for place
        results = self.places_client.search_places(
            query=name,
            location=destination,
            place_type=place_type,
            max_results=1
        )

        if not results:
            return EnrichedPlace(
                name=name,
                rating=0,
                review_count=0,
                price_level="N/A",
                trust_score="NOT_FOUND",
                trust_reason="Place not found on Google Maps",
                sample_review=None,
                address=""
            )

        place = results[0]
        trust_score, trust_reason = self._calculate_trust(place)

        # Get sample review for trusted places
        sample_review = None
        if trust_score in ["HIGH", "MEDIUM"] and place.user_ratings_total >= 50:
            sample_review = self._get_best_review(place.place_id)

        return EnrichedPlace(
            name=place.name,
            rating=place.rating,
            review_count=place.user_ratings_total,
            price_level=place.price_display,
            trust_score=trust_score,
            trust_reason=trust_reason,
            sample_review=sample_review,
            address=place.address,
            place_id=place.place_id
        )

    def _calculate_trust(self, place: PlaceResult) -> Tuple[str, str]:
        """
        Calculate trust score based on review patterns.

        Returns:
            Tuple of (score, reason)
        """
        review_count = place.user_ratings_total
        rating = place.rating

        # Check minimum review threshold
        if review_count < 20:
            return ("LOW", f"Only {review_count} reviews")

        # Check rating sanity
        if rating < self.MIN_RATING:
            return ("LOW", f"Low rating ({rating:.1f})")

        # Suspicious: perfect 5.0 with relatively few reviews
        if rating == 5.0 and review_count < 100:
            return ("SUSPICIOUS", "Perfect rating with few reviews - may be fake")

        # High trust: lots of reviews with good rating
        if review_count >= self.MIN_REVIEWS_HIGH_TRUST:
            if rating >= 4.0:
                return ("HIGH", f"{review_count:,} reviews, consistently rated {rating:.1f}")
            else:
                return ("MEDIUM", f"High volume but mixed reviews ({rating:.1f})")

        # Medium trust: decent review count
        if review_count >= self.MIN_REVIEWS_MEDIUM_TRUST:
            if rating >= 4.0:
                return ("MEDIUM", f"{review_count} reviews, {rating:.1f} rating")
            else:
                return ("LOW", f"Only {rating:.1f} rating")

        return ("LOW", "Insufficient review volume")

    def _get_best_review(self, place_id: str) -> Optional[str]:
        """
        Get the most helpful review for a place.

        Selects the longest review with 4+ stars (more specific = more helpful).
        """
        try:
            details = self.places_client.get_place_details(place_id)
            if not details or not details.reviews:
                return None

            # Filter to good reviews (4+ stars) with actual text
            good_reviews = [
                r for r in details.reviews
                if r.get("rating", 0) >= 4 and len(r.get("text", "")) > 20
            ]

            if not good_reviews:
                return None

            # Pick the longest one (usually more specific/helpful)
            best = max(good_reviews, key=lambda r: len(r.get("text", "")))

            # Format the review
            text = best["text"][:200]
            if len(best["text"]) > 200:
                text += "..."

            relative_time = best.get("relative_time", "")
            if relative_time:
                return f'"{text}" - {relative_time}'
            else:
                return f'"{text}"'

        except Exception as e:
            logger.error(f"Failed to get review for {place_id}: {e}")
            return None

    def format_enriched_places(self, places: List[EnrichedPlace]) -> str:
        """
        Format enriched places as markdown for display.

        Returns:
            Markdown-formatted list of places with ratings and trust info
        """
        lines = []

        for place in places:
            # Main line: name + emoji + rating + reviews + price
            line = f"- **{place.name}** {place.trust_emoji}"

            if place.rating > 0:
                line += f" ★{place.rating:.1f} ({place.review_count:,} reviews)"

            if place.price_level != "N/A":
                line += f" {place.price_level}"

            lines.append(line)

            # Trust info line
            if place.trust_reason:
                lines.append(f"  - Trust: {place.trust_score} - {place.trust_reason}")

            # Sample review line
            if place.sample_review:
                lines.append(f"  - {place.sample_review}")

            # Add blank line between places
            lines.append("")

        return "\n".join(lines)

    def extract_place_names(self, text: str, place_type: str = "restaurant") -> List[str]:
        """
        Extract place names from recommendation text.

        Uses heuristics to find likely place names:
        - Text in bold (**name**)
        - Text after "recommend", "try", "visit"
        - Capitalized proper nouns

        Args:
            text: Recommendation text from expert
            place_type: Type hint for better extraction

        Returns:
            List of potential place names
        """
        names = []

        # Pattern 1: Bold text (markdown)
        bold_pattern = r'\*\*([^*]+)\*\*'
        bold_matches = re.findall(bold_pattern, text)
        names.extend(bold_matches)

        # Pattern 2: After "recommend" or "try" or "visit"
        rec_pattern = r'(?:recommend|try|visit|check out|head to)\s+([A-Z][A-Za-z\s&\']+?)(?:[,\.]|\s+(?:for|which|where|–|-))'
        rec_matches = re.findall(rec_pattern, text, re.IGNORECASE)
        names.extend([m.strip() for m in rec_matches])

        # Deduplicate while preserving order
        seen = set()
        unique_names = []
        for name in names:
            name_clean = name.strip()
            if name_clean.lower() not in seen and len(name_clean) > 2:
                seen.add(name_clean.lower())
                unique_names.append(name_clean)

        return unique_names[:10]  # Limit to 10 places

    def enrich_expert_response(
        self,
        response_text: str,
        destination: str,
        expert_type: str = "general"
    ) -> Dict[str, List[EnrichedPlace]]:
        """
        Analyze expert response and enrich mentioned places.

        Args:
            response_text: Full expert recommendation text
            destination: Trip destination
            expert_type: Type of expert (affects place type inference)

        Returns:
            Dict with keys: hotels, restaurants, activities
        """
        result = {
            "hotels": [],
            "restaurants": [],
            "activities": []
        }

        if not self.is_available():
            return result

        # Map expert type to primary place type
        expert_place_map = {
            "Accommodation Specialist": ("hotels", "lodging"),
            "Food & Dining Expert": ("restaurants", "restaurant"),
            "Activity Curator": ("activities", "tourist_attraction"),
            "Local Culture Guide": ("activities", "tourist_attraction"),
            "Booking Specialist": ("hotels", "lodging"),
        }

        # Extract and enrich places
        place_names = self.extract_place_names(response_text)

        if not place_names:
            return result

        # Determine place type based on expert
        category, place_type = expert_place_map.get(
            expert_type,
            ("activities", "tourist_attraction")
        )

        enriched = self.enrich_recommendations(
            place_names=place_names,
            destination=destination,
            place_type=place_type
        )

        result[category] = enriched
        return result

    def format_enrichment_section(self, enriched_results: Dict[str, List[EnrichedPlace]]) -> str:
        """
        Format all enriched places into a markdown section for appending to expert response.

        Args:
            enriched_results: Dict from enrich_expert_response with hotels, restaurants, activities

        Returns:
            Formatted markdown string or empty string if no places found
        """
        all_places = []
        for category in ["restaurants", "hotels", "activities"]:
            all_places.extend(enriched_results.get(category, []))

        if not all_places:
            return ""

        # Filter to only places that were found (not NOT_FOUND)
        found_places = [p for p in all_places if p.trust_score != "NOT_FOUND"]

        if not found_places:
            return ""

        header = "\n\n---\n\n### 📍 Google Places Ratings\n\n"
        content = self.format_enriched_places(found_places)

        return header + content


# Convenience function for quick lookups
def lookup_place(name: str, destination: str, place_type: str = "restaurant") -> Optional[EnrichedPlace]:
    """
    Quick lookup of a single place.

    Args:
        name: Place name
        destination: City/location
        place_type: restaurant, lodging, or tourist_attraction

    Returns:
        EnrichedPlace or None
    """
    service = PlaceEnrichmentService()
    if not service.is_available():
        return None

    results = service.enrich_recommendations([name], destination, place_type)
    return results[0] if results else None
