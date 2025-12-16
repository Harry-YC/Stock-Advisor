"""
OpenWeatherMap API Integration

Weather forecasts for travel planning.
https://openweathermap.org/api

Free tier: 1,000 API calls/day
"""

import os
import logging
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class WeatherForecast:
    """Weather forecast for a specific day."""
    date: str
    temp_min: float
    temp_max: float
    temp_avg: float
    humidity: int
    description: str
    icon: str
    wind_speed: float
    precipitation_chance: float
    uv_index: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "temp_min": self.temp_min,
            "temp_max": self.temp_max,
            "temp_avg": self.temp_avg,
            "humidity": self.humidity,
            "description": self.description,
            "icon": self.icon,
            "wind_speed": self.wind_speed,
            "precipitation_chance": self.precipitation_chance,
            "uv_index": self.uv_index,
        }

    @property
    def icon_emoji(self) -> str:
        """Convert OpenWeatherMap icon code to emoji."""
        icon_map = {
            "01": "☀️",  # Clear sky
            "02": "🌤️",  # Few clouds
            "03": "☁️",  # Scattered clouds
            "04": "☁️",  # Broken clouds
            "09": "🌧️",  # Shower rain
            "10": "🌦️",  # Rain
            "11": "⛈️",  # Thunderstorm
            "13": "❄️",  # Snow
            "50": "🌫️",  # Mist
        }
        return icon_map.get(self.icon[:2], "🌡️")


@dataclass
class CurrentWeather:
    """Current weather conditions."""
    temp: float
    feels_like: float
    humidity: int
    description: str
    icon: str
    wind_speed: float
    visibility: int
    clouds: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temp": self.temp,
            "feels_like": self.feels_like,
            "humidity": self.humidity,
            "description": self.description,
            "wind_speed": self.wind_speed,
            "visibility": self.visibility,
            "clouds": self.clouds,
        }


class OpenWeatherClient:
    """
    Client for OpenWeatherMap API.

    Provides current weather and forecasts.
    """

    BASE_URL = "https://api.openweathermap.org/data/2.5"
    GEO_URL = "https://api.openweathermap.org/geo/1.0"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.OPENWEATHER_API_KEY

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make API request."""
        if not self.is_available():
            raise RuntimeError("OpenWeatherMap API key not configured")

        params["appid"] = self.api_key
        params["units"] = "metric"  # Use Celsius

        try:
            response = requests.get(
                f"{self.BASE_URL}/{endpoint}",
                params=params,
                timeout=15,
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.error(f"OpenWeatherMap API request failed: {e}")
            raise

    def geocode(self, city: str, country_code: Optional[str] = None) -> Optional[Dict]:
        """
        Get coordinates for a city.

        Args:
            city: City name
            country_code: Optional ISO 3166 country code

        Returns:
            Dict with lat, lon, name, country or None
        """
        query = f"{city},{country_code}" if country_code else city

        try:
            response = requests.get(
                f"{self.GEO_URL}/direct",
                params={"q": query, "limit": 1, "appid": self.api_key},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            if data:
                loc = data[0]
                return {
                    "lat": loc["lat"],
                    "lon": loc["lon"],
                    "name": loc["name"],
                    "country": loc.get("country", ""),
                }
            return None

        except requests.RequestException as e:
            logger.error(f"Geocoding failed: {e}")
            return None

    def get_current_weather(
        self,
        city: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> Optional[CurrentWeather]:
        """
        Get current weather for a location.

        Args:
            city: City name (will be geocoded)
            lat: Latitude (use with lon)
            lon: Longitude (use with lat)

        Returns:
            CurrentWeather object or None
        """
        params = {}

        if lat is not None and lon is not None:
            params["lat"] = lat
            params["lon"] = lon
        elif city:
            params["q"] = city
        else:
            raise ValueError("Provide either city name or lat/lon coordinates")

        try:
            data = self._make_request("weather", params)

            return CurrentWeather(
                temp=data["main"]["temp"],
                feels_like=data["main"]["feels_like"],
                humidity=data["main"]["humidity"],
                description=data["weather"][0]["description"].title(),
                icon=data["weather"][0]["icon"],
                wind_speed=data["wind"]["speed"],
                visibility=data.get("visibility", 10000),
                clouds=data["clouds"]["all"],
            )

        except Exception as e:
            logger.error(f"Current weather fetch failed: {e}")
            return None

    def get_forecast(
        self,
        city: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        days: int = 5,
    ) -> List[WeatherForecast]:
        """
        Get weather forecast (up to 5 days, 3-hour intervals).

        Args:
            city: City name
            lat: Latitude
            lon: Longitude
            days: Number of days (max 5 for free tier)

        Returns:
            List of WeatherForecast objects (one per day)
        """
        params = {"cnt": min(days * 8, 40)}  # 8 intervals per day, max 40

        if lat is not None and lon is not None:
            params["lat"] = lat
            params["lon"] = lon
        elif city:
            params["q"] = city
        else:
            raise ValueError("Provide either city name or lat/lon coordinates")

        try:
            data = self._make_request("forecast", params)

            # Group by day and aggregate
            daily_data = {}
            for item in data.get("list", []):
                date = item["dt_txt"].split(" ")[0]
                if date not in daily_data:
                    daily_data[date] = {
                        "temps": [],
                        "humidity": [],
                        "descriptions": [],
                        "icons": [],
                        "wind": [],
                        "pop": [],
                    }

                daily_data[date]["temps"].append(item["main"]["temp"])
                daily_data[date]["humidity"].append(item["main"]["humidity"])
                daily_data[date]["descriptions"].append(item["weather"][0]["description"])
                daily_data[date]["icons"].append(item["weather"][0]["icon"])
                daily_data[date]["wind"].append(item["wind"]["speed"])
                daily_data[date]["pop"].append(item.get("pop", 0))

            # Create forecast objects
            forecasts = []
            for date, values in list(daily_data.items())[:days]:
                temps = values["temps"]
                # Most common description
                desc_counts = {}
                for d in values["descriptions"]:
                    desc_counts[d] = desc_counts.get(d, 0) + 1
                main_desc = max(desc_counts, key=desc_counts.get)

                # Most common icon
                icon_counts = {}
                for i in values["icons"]:
                    icon_counts[i] = icon_counts.get(i, 0) + 1
                main_icon = max(icon_counts, key=icon_counts.get)

                forecasts.append(WeatherForecast(
                    date=date,
                    temp_min=min(temps),
                    temp_max=max(temps),
                    temp_avg=sum(temps) / len(temps),
                    humidity=int(sum(values["humidity"]) / len(values["humidity"])),
                    description=main_desc.title(),
                    icon=main_icon,
                    wind_speed=sum(values["wind"]) / len(values["wind"]),
                    precipitation_chance=max(values["pop"]) * 100,
                ))

            logger.info(f"Retrieved {len(forecasts)}-day forecast")
            return forecasts

        except Exception as e:
            logger.error(f"Forecast fetch failed: {e}")
            return []

    def format_forecast(self, forecasts: List[WeatherForecast]) -> str:
        """Format forecast for display."""
        if not forecasts:
            return "Weather forecast unavailable."

        lines = ["## Weather Forecast\n"]
        lines.append("| Date | Weather | Temp (°C) | Humidity | Rain % | Wind |")
        lines.append("|------|---------|-----------|----------|--------|------|")

        for f in forecasts:
            date_str = datetime.strptime(f.date, "%Y-%m-%d").strftime("%a %m/%d")
            temp_range = f"{f.temp_min:.0f}°-{f.temp_max:.0f}°"
            lines.append(
                f"| {date_str} | {f.icon_emoji} {f.description[:15]} | {temp_range} | {f.humidity}% | {f.precipitation_chance:.0f}% | {f.wind_speed:.1f}m/s |"
            )

        return "\n".join(lines)


# Convenience functions
def get_weather_forecast(city: str, days: int = 5) -> List[WeatherForecast]:
    """
    Quick weather forecast lookup.

    Args:
        city: City name
        days: Number of days (max 5)

    Returns:
        List of WeatherForecast objects
    """
    client = OpenWeatherClient()
    if not client.is_available():
        logger.warning("OpenWeatherMap API not configured")
        return []

    return client.get_forecast(city=city, days=days)


def get_current_weather(city: str) -> Optional[CurrentWeather]:
    """
    Quick current weather lookup.

    Args:
        city: City name

    Returns:
        CurrentWeather object or None
    """
    client = OpenWeatherClient()
    if not client.is_available():
        logger.warning("OpenWeatherMap API not configured")
        return None

    return client.get_current_weather(city=city)
