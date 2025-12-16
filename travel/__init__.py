"""
Travel Planner - Expert Personas and Templates

8 travel expert roles for comprehensive trip planning and recommendations.
"""

from .travel_personas import (
    TRAVEL_EXPERTS,
    TRAVEL_BASE_CONTEXT,
    TRAVEL_CATEGORIES,
    TRAVEL_PRESETS,
    get_travel_prompts,
    get_default_travel_experts,
)

from .travel_templates import (
    TRAVEL_QUESTION_TYPES,
    detect_travel_question_type,
    get_experts_for_question_type,
)

__all__ = [
    "TRAVEL_EXPERTS",
    "TRAVEL_BASE_CONTEXT",
    "TRAVEL_CATEGORIES",
    "TRAVEL_PRESETS",
    "get_travel_prompts",
    "get_default_travel_experts",
    "TRAVEL_QUESTION_TYPES",
    "detect_travel_question_type",
    "get_experts_for_question_type",
]
