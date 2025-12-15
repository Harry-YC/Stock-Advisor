"""
Services Layer

Business logic services separated from UI components.

This module provides the main service classes for:
- Expert discussions and GDG panel management
- Analysis (gap analysis, conflict detection, synthesis)
- Chat and research partner functionality
- Search and literature retrieval
- Recommendations and EtD framework
- LLM routing

Usage:
    from services import (
        ExpertDiscussionService,
        AnalysisService,
        SearchService,
        RecommendationService,
        EtDService,
        LLMRouter,
    )
"""

# Core expert services
from services.expert_service import ExpertDiscussionService, DiscussionRoundResult

# Analysis services
from services.analysis_service import AnalysisService

# Chat services
from services.chat_service import ChatService, ResearchAgent

# Search services
from services.search_service import SearchService

# Recommendation services
from services.recommendation_service import (
    RecommendationService,
    Recommendation,
    get_recommendation_service,
)

# Evidence-to-Decision services
from services.etd_service import (
    EtDService,
    EvidenceToDecision,
    DomainJudgment,
    get_etd_service,
)

# LLM routing
from services.llm_router import LLMRouter, get_llm_router

# Follow-up services
from services.follow_up_service import FollowUpService

__all__ = [
    # Expert services
    'ExpertDiscussionService',
    'DiscussionRoundResult',

    # Analysis
    'AnalysisService',

    # Chat
    'ChatService',
    'ResearchAgent',

    # Search
    'SearchService',

    # Recommendations
    'RecommendationService',
    'Recommendation',
    'get_recommendation_service',

    # EtD Framework
    'EtDService',
    'EvidenceToDecision',
    'DomainJudgment',
    'get_etd_service',

    # LLM
    'LLMRouter',
    'get_llm_router',

    # Follow-up
    'FollowUpService',
]
