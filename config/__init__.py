"""
Configuration Package

Provides centralized configuration for the application.

Usage:
    from config import settings
    from config.features import Features, is_enabled

    # Access settings
    api_key = settings.OPENAI_API_KEY

    # Check features
    if Features.ADVANCED_TOOLS:
        # show advanced features

    # Or use the function
    if is_enabled('ai_features'):
        # do AI stuff

    # Access domain vocabulary
    from config import palliative_surgery_vocabulary as vocab
    keywords = vocab.DOMAIN_KEYWORDS
"""

from config import settings
from config.features import (
    Features,
    is_enabled,
    get_enabled_features,
    get_disabled_features,
    get_feature_status,
    require_feature,
)
from config import palliative_surgery_vocabulary
from config.domain_config import (
    DomainConfig,
    get_domain_config,
    get_default_domain,
    list_available_domains,
    PALLIATIVE_SURGERY_CONFIG,
)

__all__ = [
    'settings',
    'Features',
    'is_enabled',
    'get_enabled_features',
    'get_disabled_features',
    'get_feature_status',
    'require_feature',
    'palliative_surgery_vocabulary',
    'DomainConfig',
    'get_domain_config',
    'get_default_domain',
    'list_available_domains',
    'PALLIATIVE_SURGERY_CONFIG',
]
