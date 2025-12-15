"""
Feature Flags Module

Centralized feature flag management with a clean interface for checking features.

Usage:
    from config.features import Features, is_enabled

    if is_enabled('advanced_tools'):
        # show advanced features

    # Or use the Features class directly
    if Features.ADVANCED_TOOLS:
        # show advanced features
"""

from config import settings


class Features:
    """
    Centralized feature flags for the application.

    All feature checks should go through this class to maintain
    a single source of truth for feature availability.
    """

    # Core AI Features
    AI_FEATURES = settings.ENABLE_AI_FEATURES
    EXPERT_PANEL = settings.ENABLE_EXPERT_PANEL
    AI_SCREENING = settings.ENABLE_AI_SCREENING

    # Advanced Tools (drug development mode)
    ADVANCED_TOOLS = settings.ENABLE_ADVANCED_TOOLS

    # Search & Retrieval
    LOCAL_RAG = settings.ENABLE_LOCAL_RAG
    SEMANTIC_SEARCH = settings.ENABLE_SEMANTIC_SEARCH
    HYBRID_SEARCH = settings.USE_HYBRID_SEARCH
    RERANKING = settings.USE_RERANKING
    HYDE = settings.ENABLE_HYDE
    QUERY_EXPANSION = settings.ENABLE_QUERY_EXPANSION

    # External Integrations
    ZOTERO = settings.ENABLE_ZOTERO
    WEB_FALLBACK = settings.ENABLE_WEB_FALLBACK
    GOOGLE_SEARCH_GROUNDING = settings.ENABLE_GOOGLE_SEARCH_GROUNDING

    # Evidence Quality
    RETRACTION_CHECK = settings.ENABLE_RETRACTION_CHECK
    CITATION_VERIFICATION = settings.ENABLE_CITATION_VERIFICATION
    ACTIVE_LEARNING = settings.ENABLE_ACTIVE_LEARNING

    # Configuration
    PRIORS = settings.ENABLE_PRIORS


# Feature name mapping for string-based lookups
_FEATURE_MAP = {
    'ai_features': 'AI_FEATURES',
    'expert_panel': 'EXPERT_PANEL',
    'ai_screening': 'AI_SCREENING',
    'advanced_tools': 'ADVANCED_TOOLS',
    'local_rag': 'LOCAL_RAG',
    'semantic_search': 'SEMANTIC_SEARCH',
    'hybrid_search': 'HYBRID_SEARCH',
    'reranking': 'RERANKING',
    'hyde': 'HYDE',
    'query_expansion': 'QUERY_EXPANSION',
    'zotero': 'ZOTERO',
    'web_fallback': 'WEB_FALLBACK',
    'google_search_grounding': 'GOOGLE_SEARCH_GROUNDING',
    'retraction_check': 'RETRACTION_CHECK',
    'citation_verification': 'CITATION_VERIFICATION',
    'active_learning': 'ACTIVE_LEARNING',
    'priors': 'PRIORS',
}


def is_enabled(feature_name: str) -> bool:
    """
    Check if a feature is enabled by name.

    Args:
        feature_name: Feature name (case-insensitive, underscores allowed)

    Returns:
        True if feature is enabled, False otherwise

    Raises:
        ValueError: If feature name is unknown

    Example:
        >>> is_enabled('advanced_tools')
        False
        >>> is_enabled('ai_features')
        True
    """
    normalized = feature_name.lower().replace('-', '_')

    if normalized not in _FEATURE_MAP:
        valid_features = ', '.join(sorted(_FEATURE_MAP.keys()))
        raise ValueError(
            f"Unknown feature: '{feature_name}'. "
            f"Valid features: {valid_features}"
        )

    attr_name = _FEATURE_MAP[normalized]
    return getattr(Features, attr_name, False)


def get_enabled_features() -> list:
    """
    Get a list of all currently enabled features.

    Returns:
        List of enabled feature names

    Example:
        >>> get_enabled_features()
        ['ai_features', 'expert_panel', 'local_rag', ...]
    """
    return [
        name for name, attr in _FEATURE_MAP.items()
        if getattr(Features, attr, False)
    ]


def get_disabled_features() -> list:
    """
    Get a list of all currently disabled features.

    Returns:
        List of disabled feature names
    """
    return [
        name for name, attr in _FEATURE_MAP.items()
        if not getattr(Features, attr, False)
    ]


def get_feature_status() -> dict:
    """
    Get status of all features.

    Returns:
        Dictionary mapping feature names to their enabled status

    Example:
        >>> get_feature_status()
        {'ai_features': True, 'advanced_tools': False, ...}
    """
    return {
        name: getattr(Features, attr, False)
        for name, attr in _FEATURE_MAP.items()
    }


def require_feature(feature_name: str) -> bool:
    """
    Decorator/check that raises if feature is not enabled.

    Args:
        feature_name: Feature that must be enabled

    Returns:
        True if feature is enabled

    Raises:
        RuntimeError: If feature is not enabled

    Example:
        >>> require_feature('ai_features')
        True
        >>> require_feature('advanced_tools')  # if disabled
        RuntimeError: Feature 'advanced_tools' is required but not enabled
    """
    if not is_enabled(feature_name):
        raise RuntimeError(
            f"Feature '{feature_name}' is required but not enabled. "
            f"Set the appropriate environment variable to enable it."
        )
    return True
