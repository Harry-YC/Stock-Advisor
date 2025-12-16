"""
Services Layer for Travel Planner

Business logic services for the travel planning application.

Services:
- expert_service: AI travel expert panel discussions
- travel_data_service: Weather, flights, car rentals, hotels
- place_enrichment_service: Google Places ratings and trust scores
- excel_export_service: Export trip plans to Excel
- llm_router: Routes LLM calls to appropriate models

Usage:
    from services import (
        ExpertDiscussionService,
        TravelDataService,
        PlaceEnrichmentService,
        LLMRouter,
    )
"""

# Expert panel service
from services.expert_service import ExpertDiscussionService, DiscussionRoundResult

# Travel data fetching
from services.travel_data_service import TravelDataService, get_travel_data_context

# Place enrichment with Google Places
from services.place_enrichment_service import PlaceEnrichmentService, EnrichedPlace

# Excel export
from services.excel_export_service import ExcelExportService

# LLM routing
from services.llm_router import LLMRouter, get_llm_router

__all__ = [
    # Expert services
    'ExpertDiscussionService',
    'DiscussionRoundResult',

    # Travel data
    'TravelDataService',
    'get_travel_data_context',

    # Place enrichment
    'PlaceEnrichmentService',
    'EnrichedPlace',

    # Excel export
    'ExcelExportService',

    # LLM
    'LLMRouter',
    'get_llm_router',
]
