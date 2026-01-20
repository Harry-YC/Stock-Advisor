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

# Google Search Grounding (for real-time market info)
ENABLE_GOOGLE_SEARCH_GROUNDING = True
GOOGLE_SEARCH_GROUNDING_THRESHOLD = 0.3

# =============================================================================
# EXPERT DEBATE MODE CONFIGURATION
# =============================================================================

ENABLE_DEBATE_MODE = True
DEBATE_ROUNDS = 3  # Number of debate rounds before synthesis
DEBATE_MAX_TOKENS = int(os.getenv("DEBATE_MAX_TOKENS", "4000"))  # Per expert per round
SYNTHESIS_MAX_TOKENS = int(os.getenv("SYNTHESIS_MAX_TOKENS", "5000"))  # Moderator
DEBATE_EXPERT_TIMEOUT = int(os.getenv("DEBATE_EXPERT_TIMEOUT", "60"))  # Per expert
MODERATOR_MODEL = os.getenv("MODERATOR_MODEL", "gemini-3-pro-preview")

# =============================================================================
# GROK/xAI API (X/Twitter KOL Insights)
# =============================================================================

# xAI API for Grok (real-time X/Twitter sentiment)
# Register at: https://x.ai/api
XAI_API_KEY = os.getenv("XAI_API_KEY")
ENABLE_GROK = bool(XAI_API_KEY)
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3-latest")
GROK_CACHE_TTL = int(os.getenv("GROK_CACHE_TTL", "3600"))

# =============================================================================
# STOCK API CONFIGURATION
# =============================================================================

# Finnhub API (Real-time stock data)
# Register free at: https://finnhub.io/ (60 calls/min free)
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
ENABLE_FINNHUB = bool(FINNHUB_API_KEY)

# Alpha Vantage API (Fallback for stocks not in Finnhub)
# Register free at: https://www.alphavantage.co/ (25 calls/day free)
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ENABLE_ALPHA_VANTAGE = bool(ALPHA_VANTAGE_API_KEY)

# Cache TTLs (seconds)
QUOTE_CACHE_TTL = 300  # 5 minutes for real-time quotes
FINANCIALS_CACHE_TTL = 3600  # 1 hour for fundamental data
NEWS_CACHE_TTL = 900  # 15 minutes for news

# Vision OCR for KOL screenshots
ENABLE_VISION_OCR = bool(GEMINI_API_KEY)
MAX_IMAGE_SIZE_MB = 5

# MCP Server Configuration
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8080"))
ALERT_CHECK_INTERVAL = int(os.getenv("ALERT_CHECK_INTERVAL", "300"))  # 5 minutes

# =============================================================================
# EMAIL NOTIFICATIONS
# =============================================================================

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ALERT_EMAIL = os.getenv("ALERT_EMAIL")  # Recipient for alerts
ENABLE_EMAIL_ALERTS = bool(SMTP_USER and SMTP_PASSWORD and ALERT_EMAIL)

# =============================================================================
# VOICE INPUT (WHISPER)
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # For Whisper API
ENABLE_VOICE_INPUT = bool(OPENAI_API_KEY)

# =============================================================================
# CHART GENERATION
# =============================================================================

CHARTS_DIR = OUTPUTS_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

CHART_THEME = os.getenv("CHART_THEME", "plotly_dark")
CHART_WIDTH = int(os.getenv("CHART_WIDTH", "1200"))
CHART_HEIGHT = int(os.getenv("CHART_HEIGHT", "800"))

# =============================================================================
# FEATURE FLAGS
# =============================================================================

ENABLE_AI_FEATURES = bool(GEMINI_API_KEY)
ENABLE_EXPERT_PANEL = bool(GEMINI_API_KEY)
ENABLE_CHARTS = True  # Charts use Plotly (no API key needed)
ENABLE_OPTIONS = True  # Options use yfinance (free)
ENABLE_SEC_FILINGS = True  # SEC EDGAR is free

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

    if not ENABLE_GROK:
        warnings.append("XAI_API_KEY not set - X/Twitter KOL insights disabled (optional)")

    if not ENABLE_EMAIL_ALERTS:
        warnings.append("Email alerts not configured (SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL)")

    if not ENABLE_VOICE_INPUT:
        warnings.append("Voice input not configured (OPENAI_API_KEY for Whisper)")

    return warnings


if __name__ == "__main__":
    print("Stock Advisor App - Configuration")
    print("=" * 50)
    print(f"\nApp Root: {APP_ROOT}")
    print(f"Outputs: {OUTPUTS_DIR}")
    print(f"Charts: {CHARTS_DIR}")
    print(f"Exports: {EXPORTS_DIR}")
    print(f"\nAPI Status:")
    print(f"  Gemini AI: {'Enabled' if GEMINI_API_KEY else 'Disabled'}")
    print(f"  Finnhub: {'Enabled' if ENABLE_FINNHUB else 'Disabled'}")
    print(f"  Grok/X: {'Enabled' if ENABLE_GROK else 'Disabled'}")
    print(f"  Vision OCR: {'Enabled' if ENABLE_VISION_OCR else 'Disabled'}")
    print(f"  Search Grounding: {'Enabled' if ENABLE_GOOGLE_SEARCH_GROUNDING else 'Disabled'}")
    print(f"\nNew Features:")
    print(f"  Charts: {'Enabled' if ENABLE_CHARTS else 'Disabled'}")
    print(f"  Options Data: {'Enabled' if ENABLE_OPTIONS else 'Disabled'}")
    print(f"  SEC Filings: {'Enabled' if ENABLE_SEC_FILINGS else 'Disabled'}")
    print(f"  Voice Input: {'Enabled' if ENABLE_VOICE_INPUT else 'Disabled'}")
    print(f"  Email Alerts: {'Enabled' if ENABLE_EMAIL_ALERTS else 'Disabled'}")

    warnings = validate_config()
    if warnings:
        print(f"\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n✓ All features configured")
