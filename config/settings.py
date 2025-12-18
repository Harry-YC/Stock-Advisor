"""
Configuration settings for Travel Planner App

Centralizes all app configuration including:
- API keys and credentials
- App behavior settings
- UI configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

VERSION = "1.0"
ENV = os.getenv("APP_ENV", "dev")

# =============================================================================
# PATHS
# =============================================================================

APP_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = APP_ROOT / "outputs"
EXPORTS_DIR = OUTPUTS_DIR / "exports"

# Ensure directories exist
for dir_path in [OUTPUTS_DIR, EXPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# API CONFIGURATION
# =============================================================================

# Google Gemini (Primary AI) - from AI Studio
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

# Google Places API - from Cloud Console (can be same or different key)
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# API timeout (increased for Gemini 3 preview models which can be slower)
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "180"))
OPENAI_TIMEOUT = API_TIMEOUT  # Alias for llm_utils compatibility

# =============================================================================
# UI CONFIGURATION
# =============================================================================

APP_TITLE = "Travel Planner"
APP_ICON = "✈️"
SIDEBAR_STATE = "expanded"

# Theme colors - Travel theme
PRIMARY_COLOR = "#2196F3"  # Travel Blue
SECONDARY_COLOR = "#4CAF50"  # Green
SUCCESS_COLOR = "#8BC34A"
WARNING_COLOR = "#FFC107"
ERROR_COLOR = "#F44336"

# Page configuration
PAGE_LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# =============================================================================
# EXPERT PANEL CONFIGURATION
# =============================================================================

EXPERT_MODEL = os.getenv("EXPERT_MODEL", "gemini-3-pro-preview")
EXPERT_MAX_TOKENS = int(os.getenv("EXPERT_MAX_TOKENS", "6000"))

# Aliases for llm_utils compatibility
REASONING_MODEL = EXPERT_MODEL
OPENAI_API_KEY = GEMINI_API_KEY  # Fallback for non-Gemini code paths

# Google Search Grounding (for real-time info)
ENABLE_GOOGLE_SEARCH_GROUNDING = True
GOOGLE_SEARCH_GROUNDING_THRESHOLD = 0.3

# Google Maps Grounding (for hotel/restaurant/attraction data)
# Uses Gemini 2.5 Flash for Maps grounding (required - Maps not available in Gemini 3)
MAPS_GROUNDING_MODEL = os.getenv("MAPS_GROUNDING_MODEL", "gemini-2.5-flash")
ENABLE_MAPS_GROUNDING = bool(GEMINI_API_KEY)

# =============================================================================
# TRAVEL API CONFIGURATION
# =============================================================================

# Amadeus Self-Service API (Flight, Hotel, Car Rental Search)
# Register free at: https://developers.amadeus.com/
AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")
AMADEUS_BASE_URL = "https://api.amadeus.com"
ENABLE_FLIGHT_SEARCH = bool(AMADEUS_API_KEY and AMADEUS_API_SECRET)
ENABLE_CAR_RENTAL = bool(AMADEUS_API_KEY and AMADEUS_API_SECRET)

# OpenWeatherMap API (Weather Forecasts)
# Register free at: https://openweathermap.org/api (1000 calls/day free)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/3.0"
ENABLE_WEATHER_API = bool(OPENWEATHER_API_KEY)

# Google Places API (Place ratings and reviews)
# Uses GOOGLE_PLACES_API_KEY from Cloud Console - enable Places API (New)
# Free tier: $200/month credit (~6,000 text searches)
ENABLE_PLACES_API = bool(GOOGLE_PLACES_API_KEY)
PLACES_CACHE_HOURS = 24
PLACES_MIN_REVIEWS_TRUSTED = 100

# =============================================================================
# FEATURE FLAGS
# =============================================================================

ENABLE_AI_FEATURES = bool(GEMINI_API_KEY)
ENABLE_EXPERT_PANEL = bool(GEMINI_API_KEY)

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate configuration and warn about missing keys"""
    warnings = []

    if not GEMINI_API_KEY:
        warnings.append("GEMINI_API_KEY not set - AI features will be disabled")

    if not AMADEUS_API_KEY or not AMADEUS_API_SECRET:
        warnings.append("Amadeus credentials not set - flight search disabled")

    if not OPENWEATHER_API_KEY:
        warnings.append("OPENWEATHER_API_KEY not set - weather forecasts disabled")

    if not GOOGLE_PLACES_API_KEY:
        warnings.append("GOOGLE_PLACES_API_KEY not set - place ratings disabled")

    return warnings


if __name__ == "__main__":
    print("Travel Planner App - Configuration")
    print("=" * 50)
    print(f"\nApp Root: {APP_ROOT}")
    print(f"Outputs: {OUTPUTS_DIR}")
    print(f"\nAPI Status:")
    print(f"  Gemini AI: {'Enabled' if GEMINI_API_KEY else 'Disabled'}")
    print(f"  Places API: {'Enabled' if ENABLE_PLACES_API else 'Disabled'}")
    print(f"  Weather API: {'Enabled' if ENABLE_WEATHER_API else 'Disabled'}")
    print(f"  Flight API: {'Enabled' if ENABLE_FLIGHT_SEARCH else 'Disabled'}")

    warnings = validate_config()
    if warnings:
        print(f"\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n✓ All APIs configured")
