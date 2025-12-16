# Travel Planner

An AI-powered travel planning application built with Chainlit. Configure your trip details and chat with AI travel experts for personalized recommendations, enhanced with real-time weather, flight, car rental, and hotel data.

## Features

### AI Travel Experts
- **8 Expert Personas** - Budget Advisor, Logistics Planner, Safety Expert, Weather Analyst, Local Culture Guide, Food & Dining Expert, Activity Curator, Accommodation Specialist
- **Expert Presets** - Quick Trip, Adventure, Budget, Cultural, Family, Full Panel
- **Streaming Responses** - Real-time expert recommendations
- **Real Data Integration** - Recommendations grounded in actual availability and pricing

### Real-Time Data
- **Weather Forecasts** - OpenWeatherMap integration for destination weather
- **Flight Search** - Amadeus API for real flight prices and availability
- **Car Rentals** - Amadeus API for rental car options
- **Hotel Search** - Google Maps Grounding via Gemini for hotel recommendations with ratings

### Conversational Interface
- **ChatSettings Panel** - Configure trip details (destination, dates, budget, travelers)
- **Natural Chat** - Ask follow-up questions naturally
- **Expert Consultation** - "Ask Budget Advisor about cheap flights"
- **Session Persistence** - Resume conversations (with LiteralAI)

## Quick Start

### Installation

```bash
cd "Travel Planner"
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```bash
# Required - Google APIs
GEMINI_API_KEY=your_aistudio_key        # From aistudio.google.com
GOOGLE_PLACES_API_KEY=your_cloud_key    # From console.cloud.google.com

# Optional - Enhance with real data
OPENWEATHER_API_KEY=your_weather_key    # From openweathermap.org
AMADEUS_API_KEY=your_amadeus_key        # From developers.amadeus.com
AMADEUS_API_SECRET=your_amadeus_secret

# Optional - Conversation persistence
LITERAL_API_KEY=your_literal_key        # From literalai.com
```

### Run the App

```bash
chainlit run app_tp2.py
```

Or with auto-reload for development:
```bash
chainlit run app_tp2.py -w
```

## Usage

1. **Configure Trip** - Click the gear icon to open ChatSettings
2. **Set Details** - Enter destination, dates, budget, travelers
3. **Plan Trip** - Say "Plan my trip" to get expert recommendations
4. **Ask Questions** - Chat naturally or ask specific experts

## Project Structure

```
Travel Planner/
├── app_tp2.py                   # Main Chainlit app
├── chainlit.md                  # Welcome message
├── requirements.txt             # Python dependencies
├── CLAUDE.md                    # AI assistant context
│
├── config/
│   └── settings.py              # API keys and configuration
│
├── core/
│   ├── llm_utils.py             # LLM helper functions
│   ├── logger.py                # Logging configuration
│   └── utils.py                 # General utilities
│
├── integrations/
│   ├── weather.py               # OpenWeatherMap API
│   ├── amadeus_flights.py       # Amadeus Flight API
│   ├── amadeus_cars.py          # Amadeus Car Rental API
│   ├── google_places.py         # Google Places API
│   └── google_search.py         # Google Maps Grounding
│
├── services/
│   ├── expert_service.py        # AI expert panel engine
│   ├── travel_data_service.py   # Weather, flights, cars, hotels
│   ├── llm_router.py            # LLM routing to Gemini
│   └── place_enrichment_service.py  # Google Places trust scoring
│
└── travel/
    └── travel_personas.py       # Expert persona definitions
```

## AI Model

Uses **Google Gemini** for all AI features:
- **Gemini 3 Pro** (`gemini-3-pro-preview`) - Expert panel discussions
- **Gemini 2.5 Flash** - Hotel search via Maps Grounding

## Travel Expert Personas

| Expert | Focus |
|--------|-------|
| Budget Advisor | Cost optimization, deals, budget allocation |
| Logistics Planner | Transportation, car rentals, routes |
| Safety Expert | Travel advisories, insurance providers |
| Weather Analyst | Climate, packing advice, best times |
| Local Culture Guide | Customs, etiquette, authentic experiences |
| Food & Dining Expert | Restaurants, local cuisine, dietary needs |
| Activity Curator | Tours, attractions, day trips |
| Accommodation Specialist | Hotels, neighborhoods, location strategy |

## Expert Presets

| Preset | Experts |
|--------|---------|
| Quick Trip Planning | Budget, Logistics, Accommodation, Activity |
| Adventure Travel | Safety, Activity, Weather, Culture |
| Budget Backpacking | Budget, Accommodation, Food, Safety |
| Cultural Immersion | Culture, Food, Activity |
| Family Vacation | Safety, Accommodation, Activity, Logistics |
| Full Panel | All 8 experts |

## API Costs

| Service | Free Tier |
|---------|-----------|
| Gemini (AI Studio) | Generous free tier |
| Google Places | $200/month credit |
| OpenWeatherMap | 1,000 calls/day |
| Amadeus | Self-service free tier |
| LiteralAI | Free tier available |

---

**Version**: 2.0 (Chainlit)
**Last Updated**: December 2025
