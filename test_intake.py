"""
Test the LLM-guided intake conversation feature.
"""

import sys
sys.path.insert(0, '/Users/nelsonliu/Travel Planner')

import asyncio
from app_tp2 import extract_trip_info, get_next_question, convert_trip_info_to_config, format_trip_summary

async def test_extraction():
    """Test the LLM extraction function."""
    print("\n" + "="*60)
    print("Testing LLM Trip Info Extraction")
    print("="*60)

    test_cases = [
        ("Barcelona with my wife next month", {}),
        ("around 3000 dollars", {"destination": "Barcelona", "travelers": "2 adults (couple)"}),
        ("a week in mid-January", {"destination": "Barcelona"}),
        ("Tokyo for 2 weeks with kids, budget 10000", {}),
    ]

    for message, existing_info in test_cases:
        print(f"\n--- Testing: '{message}' ---")
        print(f"Existing info: {existing_info}")

        result = await extract_trip_info(message, existing_info)
        print(f"Extracted: {result}")

        next_q = get_next_question(result)
        print(f"Next question: {next_q or 'All info collected!'}")

def test_next_question():
    """Test the follow-up question logic."""
    print("\n" + "="*60)
    print("Testing Follow-up Question Logic")
    print("="*60)

    scenarios = [
        ({}, "Empty info"),
        ({"destination": "Paris"}, "Only destination"),
        ({"destination": "Paris", "dates": "January"}, "Has destination and dates"),
        ({"destination": "Paris", "dates": "January", "travelers": "2 adults"}, "Missing only budget"),
        ({"destination": "Paris", "dates": "January", "travelers": "2 adults", "budget": 5000}, "All filled"),
    ]

    for trip_info, scenario in scenarios:
        next_q = get_next_question(trip_info)
        status = "Ready!" if next_q is None else next_q
        print(f"\n{scenario}: {status}")

def test_conversion():
    """Test converting trip_info to trip_config."""
    print("\n" + "="*60)
    print("Testing Trip Info to Config Conversion")
    print("="*60)

    trip_info = {
        "destination": "Barcelona, Spain",
        "dates": "January 2025",
        "duration_days": 7,
        "travelers": "2 adults (couple)",
        "budget": 5000
    }

    config = convert_trip_info_to_config(trip_info)
    print(f"\nInput trip_info: {trip_info}")
    print(f"\nConverted config:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print(f"\nFormatted summary:\n{format_trip_summary(trip_info)}")

if __name__ == "__main__":
    print("🧳" * 30)
    print("\n  TRAVEL PLANNER INTAKE TEST")
    print("\n" + "🧳" * 30)

    # Test synchronous functions first
    test_next_question()
    test_conversion()

    # Test async LLM extraction
    print("\n\nRunning LLM extraction test (requires API)...")
    asyncio.run(test_extraction())

    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)
