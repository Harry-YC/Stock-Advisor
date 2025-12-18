"""
Trip Storage Service for Travel Planner

Simple JSON-based storage for saving and loading trip plans.
Stores trips in outputs/trips/ directory.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import hashlib

from config import settings

logger = logging.getLogger(__name__)

# Storage directory
TRIPS_DIR = settings.OUTPUTS_DIR / "trips"
TRIPS_DIR.mkdir(parents=True, exist_ok=True)


def _generate_trip_id(destination: str, departure: str) -> str:
    """Generate a unique trip ID from destination and date."""
    content = f"{destination}:{departure}:{datetime.now().isoformat()}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def save_trip(
    trip_config: Dict,
    expert_responses: Dict[str, str],
    trip_name: Optional[str] = None
) -> str:
    """
    Save a trip plan to storage.

    Args:
        trip_config: Trip configuration dict (destination, dates, budget, etc.)
        expert_responses: Dict mapping expert name to response text
        trip_name: Optional custom name for the trip

    Returns:
        Trip ID
    """
    destination = trip_config.get('destination', 'Unknown')
    departure = trip_config.get('departure_date', '')

    trip_id = _generate_trip_id(destination, departure)

    # Build trip record
    trip_record = {
        "id": trip_id,
        "name": trip_name or f"{destination} Trip",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "config": trip_config,
        "expert_responses": expert_responses,
        "version": "1.0"
    }

    # Save to file
    trip_file = TRIPS_DIR / f"{trip_id}.json"
    with open(trip_file, 'w', encoding='utf-8') as f:
        json.dump(trip_record, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved trip {trip_id}: {destination}")
    return trip_id


def load_trip(trip_id: str) -> Optional[Dict]:
    """
    Load a trip plan from storage.

    Args:
        trip_id: Trip ID

    Returns:
        Trip record dict or None if not found
    """
    trip_file = TRIPS_DIR / f"{trip_id}.json"

    if not trip_file.exists():
        logger.warning(f"Trip not found: {trip_id}")
        return None

    with open(trip_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_trips(limit: int = 20) -> List[Dict]:
    """
    List all saved trips, sorted by most recent first.

    Args:
        limit: Maximum trips to return

    Returns:
        List of trip summaries (id, name, destination, dates, created_at)
    """
    trips = []

    for trip_file in sorted(TRIPS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(trip_file, 'r', encoding='utf-8') as f:
                record = json.load(f)

            trips.append({
                "id": record.get("id"),
                "name": record.get("name"),
                "destination": record.get("config", {}).get("destination"),
                "departure_date": record.get("config", {}).get("departure_date"),
                "return_date": record.get("config", {}).get("return_date"),
                "created_at": record.get("created_at"),
                "experts_count": len(record.get("expert_responses", {}))
            })

            if len(trips) >= limit:
                break

        except Exception as e:
            logger.error(f"Error reading trip file {trip_file}: {e}")
            continue

    return trips


def delete_trip(trip_id: str) -> bool:
    """
    Delete a trip plan.

    Args:
        trip_id: Trip ID

    Returns:
        True if deleted, False if not found
    """
    trip_file = TRIPS_DIR / f"{trip_id}.json"

    if trip_file.exists():
        trip_file.unlink()
        logger.info(f"Deleted trip: {trip_id}")
        return True

    return False


def update_trip(
    trip_id: str,
    trip_config: Optional[Dict] = None,
    expert_responses: Optional[Dict[str, str]] = None,
    trip_name: Optional[str] = None
) -> bool:
    """
    Update an existing trip plan.

    Args:
        trip_id: Trip ID
        trip_config: Updated trip configuration
        expert_responses: Updated expert responses
        trip_name: Updated trip name

    Returns:
        True if updated, False if not found
    """
    record = load_trip(trip_id)
    if not record:
        return False

    if trip_config:
        record["config"] = trip_config
    if expert_responses:
        record["expert_responses"] = expert_responses
    if trip_name:
        record["name"] = trip_name

    record["updated_at"] = datetime.now().isoformat()

    trip_file = TRIPS_DIR / f"{trip_id}.json"
    with open(trip_file, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    logger.info(f"Updated trip: {trip_id}")
    return True


def search_trips(query: str) -> List[Dict]:
    """
    Search trips by destination or name.

    Args:
        query: Search query

    Returns:
        List of matching trip summaries
    """
    query_lower = query.lower()
    all_trips = list_trips(limit=100)

    return [
        t for t in all_trips
        if query_lower in (t.get("destination") or "").lower()
        or query_lower in (t.get("name") or "").lower()
    ]


def get_trips_count() -> int:
    """Get total number of saved trips."""
    return len(list(TRIPS_DIR.glob("*.json")))


# Test
if __name__ == "__main__":
    print("Trip Storage Test")
    print("=" * 50)

    # Test save
    test_config = {
        "destination": "Tokyo, Japan",
        "departure_date": "2025-03-15",
        "return_date": "2025-03-22",
        "budget": 3000,
        "travelers": "2 adults"
    }
    test_responses = {
        "Booking Specialist": "Test flight recommendations...",
        "Budget Advisor": "Test budget advice..."
    }

    trip_id = save_trip(test_config, test_responses, "Spring Tokyo Trip")
    print(f"Saved trip: {trip_id}")

    # Test load
    loaded = load_trip(trip_id)
    print(f"Loaded trip: {loaded.get('name')}")

    # Test list
    trips = list_trips()
    print(f"Total trips: {len(trips)}")

    print("\nTrip Storage working correctly!")
