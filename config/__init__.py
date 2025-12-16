"""
Configuration Package for Travel Planner

Provides centralized configuration for the application.

Usage:
    from config import settings

    # Access settings
    api_key = settings.GEMINI_API_KEY
    enable_weather = settings.ENABLE_WEATHER_API
"""

from config import settings

__all__ = [
    'settings',
]
