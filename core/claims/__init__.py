"""
Claims extraction and matching module.

Provides functionality to:
- Extract claims with search hints from expert responses
- Match claims to supporting literature
"""

from .claim_extractor import (
    ExtractedClaim,
    ClaimExtractionResult,
    extract_claims_from_responses,
    get_searchable_claims,
    group_claims_by_expert
)

__all__ = [
    'ExtractedClaim',
    'ClaimExtractionResult',
    'extract_claims_from_responses',
    'get_searchable_claims',
    'group_claims_by_expert'
]
