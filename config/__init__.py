"""
Configuration Package for Stock Advisor

Provides centralized configuration for the application.

Usage:
    from config import settings

    # Access settings
    api_key = settings.GEMINI_API_KEY
    enable_finnhub = settings.ENABLE_FINNHUB
"""

from config import settings

__all__ = [
    'settings',
]
