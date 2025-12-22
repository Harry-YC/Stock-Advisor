"""
Configuration settings for Stock Advisor App

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
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "240"))
OPENAI_TIMEOUT = API_TIMEOUT  # Alias for llm_utils compatibility

# =============================================================================
# UI CONFIGURATION
# =============================================================================

APP_TITLE = "Stock Advisor"
APP_ICON = "📊"
SIDEBAR_STATE = "expanded"

# Theme colors - Stock/Finance theme
PRIMARY_COLOR = "#1976D2"  # Finance Blue
SECONDARY_COLOR = "#388E3C"  # Money Green
SUCCESS_COLOR = "#4CAF50"  # Bull Green
WARNING_COLOR = "#FFC107"
ERROR_COLOR = "#D32F2F"  # Bear Red

# Page configuration
PAGE_LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# =============================================================================
# EXPERT PANEL CONFIGURATION
# =============================================================================

EXPERT_MODEL = os.getenv("EXPERT_MODEL", "gemini-3-pro-preview")
EXPERT_FALLBACK_MODEL = os.getenv("EXPERT_FALLBACK_MODEL", "gemini-3-flash-preview")
EXPERT_MAX_TOKENS = int(os.getenv("EXPERT_MAX_TOKENS", "6000"))
EXPERT_TIMEOUT = int(os.getenv("EXPERT_TIMEOUT", "60"))  # Shorter timeout, then fallback

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
# STOCK API CONFIGURATION
# =============================================================================

# Finnhub API (Real-time stock data)
# Register free at: https://finnhub.io/ (60 calls/min free)
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
ENABLE_FINNHUB = bool(FINNHUB_API_KEY)

# Cache TTLs (seconds)
QUOTE_CACHE_TTL = 300  # 5 minutes for real-time quotes
FINANCIALS_CACHE_TTL = 3600  # 1 hour for fundamental data
NEWS_CACHE_TTL = 900  # 15 minutes for news

# Vision OCR for KOL screenshots
ENABLE_VISION_OCR = bool(GEMINI_API_KEY)
MAX_IMAGE_SIZE_MB = 5

# MCP Server Configuration
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8080"))
ALERT_CHECK_INTERVAL = int(os.getenv("ALERT_CHECK_INTERVAL", "60"))

# =============================================================================
# LEGACY TRAVEL API CONFIGURATION (Keep for compatibility)
# =============================================================================

# Amadeus Self-Service API (Flight, Hotel, Car Rental Search)
AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")
AMADEUS_BASE_URL = "https://api.amadeus.com"
ENABLE_FLIGHT_SEARCH = bool(AMADEUS_API_KEY and AMADEUS_API_SECRET)
ENABLE_CAR_RENTAL = bool(AMADEUS_API_KEY and AMADEUS_API_SECRET)

# OpenWeatherMap API
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/3.0"
ENABLE_WEATHER_API = bool(OPENWEATHER_API_KEY)

# Google Places API
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

    if not FINNHUB_API_KEY:
        warnings.append("FINNHUB_API_KEY not set - real-time stock data disabled")

    return warnings


if __name__ == "__main__":
    print("Stock Advisor App - Configuration")
    print("=" * 50)
    print(f"\nApp Root: {APP_ROOT}")
    print(f"Outputs: {OUTPUTS_DIR}")
    print(f"\nAPI Status:")
    print(f"  Gemini AI: {'Enabled' if GEMINI_API_KEY else 'Disabled'}")
    print(f"  Finnhub: {'Enabled' if ENABLE_FINNHUB else 'Disabled'}")
    print(f"  Vision OCR: {'Enabled' if ENABLE_VISION_OCR else 'Disabled'}")
    print(f"  Search Grounding: {'Enabled' if ENABLE_GOOGLE_SEARCH_GROUNDING else 'Disabled'}")

    warnings = validate_config()
    if warnings:
        print(f"\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n✓ All APIs configured")
