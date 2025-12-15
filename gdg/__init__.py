"""
Palliative Surgery Guideline Development Group (GDG) Module

Provides expert personas, utilities, and validation for AI-powered
GDG discussions focused on palliative surgery guideline development.
"""

from .gdg_personas import (
    GDG_PERSONAS,
    get_gdg_prompts,
    GDG_BASE_CONTEXT,
    ROUND_INSTRUCTIONS,
    COGNITIVE_CONSTRAINTS,
    CLINICAL_SCENARIOS,
    GDG_CLINICAL_SCENARIOS,  # Alias for backward compatibility
    REQUIRED_DELIVERABLES,
    EVIDENCE_WEIGHTING,
    get_all_expert_names,
    get_default_experts,
    get_enhanced_expert_prompts,
    get_gdg_search_queries,
    get_persona_roles,
)
from .gdg_utils import (
    score_paper_relevance,
    format_evidence_context,
    call_expert,
    call_expert_stream,
    auto_select_papers_for_experts,
    export_discussion_to_markdown,
    generate_perspective_questions,
    synthesize_expert_responses,
    extract_hypotheses_from_discussion,
    generate_followup_questions,
    process_discussion_for_knowledge,
)
from .response_validator import ResponseValidator

__all__ = [
    # Personas and configuration
    'GDG_PERSONAS',
    'get_gdg_prompts',
    'GDG_BASE_CONTEXT',
    'ROUND_INSTRUCTIONS',
    'COGNITIVE_CONSTRAINTS',
    'CLINICAL_SCENARIOS',
    'GDG_CLINICAL_SCENARIOS',
    'REQUIRED_DELIVERABLES',
    'EVIDENCE_WEIGHTING',
    # Helper functions
    'get_all_expert_names',
    'get_default_experts',
    'get_enhanced_expert_prompts',
    'get_gdg_search_queries',
    'get_persona_roles',
    # Evidence utilities
    'score_paper_relevance',
    'format_evidence_context',
    'call_expert',
    'call_expert_stream',
    'auto_select_papers_for_experts',
    'export_discussion_to_markdown',
    'generate_perspective_questions',
    'synthesize_expert_responses',
    'extract_hypotheses_from_discussion',
    'generate_followup_questions',
    'process_discussion_for_knowledge',
    # Validation
    'ResponseValidator',
]
