# CLAUDE.md - Travel Partner (app_tp2.py)

## Overview

**Travel Partner** is a Chainlit-based conversational travel planning application that leverages AI travel experts to provide personalized trip recommendations. Users configure trip details through ChatSettings and receive expert advice enhanced with real-time data from weather, flight, and hotel APIs.

## Architecture

```
app_tp2.py (Main Chainlit App)
    │
    ├── config/settings.py          # Configuration & API keys
    │
    ├── services/
    │   ├── travel_data_service.py  # Fetches real-time travel data
    │   └── llm_router.py           # LLM call routing (Gemini)
    │
    ├── travel/
    │   └── travel_personas.py      # 8 Expert persona definitions
    │
    ├── core/
    │   └── llm_utils.py            # LLM client factory
    │
    └── integrations/
        ├── weather.py              # OpenWeatherMap API
        ├── amadeus_flights.py      # Flight search
        ├── amadeus_cars.py         # Car rental search
        ├── google_search.py        # Google Maps Grounding
        └── google_places.py        # Place ratings
```

## Key Components

### 1. Main App (`app_tp2.py`)
- **Lines 1-58**: SQLite persistence setup with SQLAlchemy
- **Lines 66-149**: Chat initialization with ChatSettings widgets
- **Lines 152-164**: Settings update handler
- **Lines 167-188**: Message routing (plan trip, ask expert, followup)
- **Lines 190-293**: Trip planning with expert panel streaming
- **Lines 295-344**: Direct expert questioning
- **Lines 347-434**: Follow-up handling with expert detection

### 2. Expert Personas (`travel/travel_personas.py`)
8 specialized travel experts:
| Expert | Focus | Icon |
|--------|-------|------|
| Budget Advisor | Cost optimization, deals | 💰 |
| Logistics Planner | Routes, car rentals | 🚗 |
| Safety Expert | Advisories, insurance | 🛡️ |
| Weather Analyst | Climate, packing | 🌤️ |
| Local Culture Guide | Customs, authenticity | 🎎 |
| Food & Dining Expert | Cuisine, restaurants | 🍜 |
| Activity Curator | Tours, attractions | 🎯 |
| Accommodation Specialist | Hotels, locations | 🏨 |

### 3. Travel Data Service (`services/travel_data_service.py`)
Aggregates real-time data:
- Weather forecasts (OpenWeatherMap)
- Flight options (Amadeus)
- Car rentals (Amadeus)
- Hotels (Google Maps Grounding)

### 4. LLM Integration
- **Primary Model**: `gemini-3-pro-preview`
- **Maps Grounding**: `gemini-2.5-flash` (required for Google Maps data)
- **API**: Google's OpenAI-compatible endpoint

## Running the App

```bash
# Standard run
chainlit run app_tp2.py

# Development with auto-reload
chainlit run app_tp2.py -w
```

## Environment Variables

```bash
# Required
GEMINI_API_KEY=your_aistudio_key

# Optional - Enhance with real data
GOOGLE_PLACES_API_KEY=your_cloud_key
OPENWEATHER_API_KEY=your_weather_key
AMADEUS_API_KEY=your_amadeus_key
AMADEUS_API_SECRET=your_amadeus_secret
```

## User Commands

| Command | Action |
|---------|--------|
| "Plan my trip" | Execute full expert panel |
| "Ask [Expert] about..." | Consult specific expert |
| General question | Auto-routes to best expert |

## Expert Presets

| Preset | Experts Included |
|--------|------------------|
| Quick Trip Planning | Budget, Logistics, Accommodation, Activity |
| Adventure Travel | Safety, Activity, Weather, Culture |
| Budget Backpacking | Budget, Accommodation, Food, Safety |
| Cultural Immersion | Culture, Food, Activity |
| Family Vacation | Safety, Accommodation, Activity, Logistics |
| Full Panel | All 8 experts |

## Data Flow

1. User configures trip in ChatSettings (destination, dates, budget)
2. User says "Plan my trip"
3. App fetches real-time data (weather, flights, hotels)
4. Selected experts process data and stream responses
5. User can ask follow-up questions

## Persistence

- **Database**: SQLite at `data/travel_planner.db`
- **Layer**: SQLAlchemy with aiosqlite
- **Stores**: Threads, messages, session state

## Key Files to Modify

| Task | File |
|------|------|
| Add new expert | `travel/travel_personas.py` |
| Change default model | `config/settings.py` (EXPERT_MODEL) |
| Add new data source | `services/travel_data_service.py` |
| Modify UI widgets | `app_tp2.py` (start function) |
| Add new command | `app_tp2.py` (on_message function) |

## Common Issues

### API Key Errors
- Use AI Studio key (not Cloud Console) for Gemini
- Maps Grounding requires separate model (gemini-2.5-flash)

### SQLite Errors
- Install: `pip install aiosqlite sqlalchemy`
- Check write permissions for `data/` directory

### Expert Not Found
- Check exact spelling in `TRAVEL_EXPERTS` dict
- Case-sensitive matching in expert detection
