# CLAUDE.md - AI Assistant Context

## Project Overview

**Travel Planner** is an AI-powered travel planning app using Chainlit. Users can describe their trip naturally or use ChatSettings, and get personalized recommendations from 8 AI travel experts with real-time data from weather, flight, hotel, and restaurant APIs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (copy .env.example to .env)
cp .env.example .env

# Run the app
chainlit run app_tp2.py

# Development mode (auto-reload)
chainlit run app_tp2.py -w
```

## Core Workflow

1. **Conversational intake** - Tell the app about your trip naturally
2. **Confirm details** - Review extracted trip info
3. **Expert recommendations** - Get advice from selected expert panel
4. **Follow-up questions** - Ask specific experts for more details
5. **Export** - Download as Excel or Word document

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Chainlit 1.3+ |
| AI Model | Google Gemini 3 Pro |
| Maps/Hotels | Google Maps Grounding (Gemini 2.5 Flash) |
| Search | Google Search Grounding |
| Weather | OpenWeatherMap API |
| Flights | Amadeus Self-Service API |
| Cars | Amadeus Self-Service API |
| Places | Google Places API (New) |
| Export | openpyxl (Excel), python-docx (Word) |

## Project Structure

```
Travel Planner/
├── app_tp2.py                    # Main Chainlit app (~950 lines)
├── chainlit.md                   # Welcome message
├── Procfile                      # Railway/Heroku deployment
├── railway.toml                  # Railway configuration
├── requirements.txt              # Python dependencies
│
├── config/
│   └── settings.py               # Centralized configuration
│
├── core/
│   ├── logger.py                 # Logging setup
│   ├── utils.py                  # General utilities
│   └── database.py               # SQLite DAOs (optional)
│
├── integrations/
│   ├── weather.py                # OpenWeatherMap client
│   ├── amadeus_flights.py        # Flight search
│   ├── amadeus_cars.py           # Car rental search
│   ├── google_search.py          # Google Search + Maps grounding
│   └── google_places.py          # Places ratings (filters closed)
│
├── services/
│   ├── llm_router.py             # Gemini API routing
│   ├── travel_data_service.py    # Data aggregation + search triggers
│   ├── excel_export_service.py   # Professional Excel export
│   └── word_export_service.py    # Word document export
│
└── travel/
    └── travel_personas.py        # 8 Expert definitions + presets
```

## Environment Variables

```bash
# Required
GEMINI_API_KEY=xxx                # From aistudio.google.com

# Optional - Enhanced features
OPENWEATHER_API_KEY=xxx           # Weather forecasts
AMADEUS_API_KEY=xxx               # Flight search
AMADEUS_API_SECRET=xxx            # Flight search
GOOGLE_PLACES_API_KEY=xxx         # Place ratings
```

## 8 Travel Experts

| Expert | Icon | Focus |
|--------|------|-------|
| Budget Advisor | 💰 | Cost optimization, deals, budget allocation |
| Logistics Planner | 🚗 | Transportation, car rentals, routes |
| Safety Expert | 🛡️ | Travel advisories, insurance, visa requirements |
| Weather Analyst | 🌤️ | Climate, packing, seasonal events |
| Local Culture Guide | 🎎 | Customs, etiquette, authentic experiences |
| Food & Dining Expert | 🍜 | Restaurants, reservations, payment tips |
| Activity Curator | 🎯 | Tours, attractions, day trips |
| Accommodation Specialist | 🏨 | Hotels, neighborhoods, booking tips |

## Expert Presets

| Preset | Experts Included |
|--------|------------------|
| Quick Trip Planning | Budget, Logistics, Accommodation, Activity |
| Adventure Travel | Safety, Activity, Weather, Culture |
| Budget Backpacking | Budget, Accommodation, Food, Safety |
| Cultural Immersion | Culture, Food, Activity |
| Family Vacation | Safety, Accommodation, Activity, Logistics |
| Full Panel | All 8 experts |

## Key Features

### Smart Google Search Triggers
- **Safety Expert**: Always gets real-time travel advisories, visa requirements
- **Near-term trips** (<30 days): All experts get current events context
- **Historical weather**: Shows climate patterns for trips >5 days out

### Restaurant Intelligence
- Filters out closed restaurants by default
- Flags cash-only restaurants
- Indicates when reservations are needed

### Professional Export
- **Excel**: 8 sheets with budget tracking, itinerary, checklists
- **Word**: Formatted document with all recommendations

## User Commands

| Command | Action |
|---------|--------|
| "Plan my trip" | Run expert panel with current settings |
| "Ask [Expert] about..." | Consult specific expert |
| "Show car rentals" | Fetch car rental options on demand |
| General question | Auto-routes to best expert |

## Deployment

### Railway
```bash
railway up
```

### Environment Setup
1. Create project on Railway
2. Add environment variables in Railway dashboard
3. Deploy from GitHub or CLI

## API Costs (Approximate)

| API | Free Tier | Cost After |
|-----|-----------|------------|
| Gemini | Generous | Pay-per-token |
| OpenWeatherMap | 1,000 calls/day | $0.001/call |
| Amadeus | 2,000 calls/month | Contact for pricing |
| Google Places | $200/month credit | $0.032/search |

## Recent Updates (Dec 2025)

1. **Google Search grounding** for Safety Expert
2. **Closed restaurant filtering** via businessStatus
3. **Reservation/payment alerts** in Food expert
4. **Historical weather** for future trips
5. **Conversational intake** flow
6. **Expert follow-up suggestions**
7. **Professional Excel export** with 8 sheets

## Troubleshooting

### 403 PERMISSION_DENIED
- Create fresh key from AI Studio (not Cloud Console)
- Enable "Generative Language API" in Cloud Console

### No flight results
- Check Amadeus credentials are correct
- Verify airport codes are valid (uses IATA)

### Weather unavailable
- OpenWeatherMap free tier limited to 5-day forecast
- Historical data shown for trips >5 days out

## Contributing

See `CODE_REVIEW.md` for architecture details and improvement suggestions.
