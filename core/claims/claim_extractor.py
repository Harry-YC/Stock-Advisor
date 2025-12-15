"""
Claim Extractor for Expert-First Evidence Pipeline

Parses expert responses to extract:
- Claims with embedded search hints [SEARCH: query terms]
- Evidence quality markers [STANDARD PRACTICE], [CASE SERIES], etc.
- Quantitative claims for literature matching
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExtractedClaim:
    """A claim extracted from an expert response with search metadata."""
    text: str                           # The claim text
    expert: str                         # Which expert made this claim
    search_hint: Optional[str] = None   # Embedded search query if provided
    evidence_quality: str = "CLINICAL_EXPERIENCE"  # Quality marker
    has_quantitative_data: bool = False  # Contains percentages, rates, etc.
    context: str = ""                   # Surrounding text for context

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "expert": self.expert,
            "search_hint": self.search_hint,
            "evidence_quality": self.evidence_quality,
            "has_quantitative_data": self.has_quantitative_data,
            "context": self.context
        }


@dataclass
class ClaimExtractionResult:
    """Result of claim extraction from all expert responses."""
    claims: List[ExtractedClaim] = field(default_factory=list)
    claims_with_hints: int = 0
    claims_with_quality_markers: int = 0
    quantitative_claims: int = 0

    def to_dict(self) -> Dict:
        return {
            "claims": [c.to_dict() for c in self.claims],
            "claims_with_hints": self.claims_with_hints,
            "claims_with_quality_markers": self.claims_with_quality_markers,
            "quantitative_claims": self.quantitative_claims,
            "total_claims": len(self.claims)
        }


# Patterns for extraction
SEARCH_HINT_PATTERN = re.compile(
    r'([^.\n]+?)\s*\[SEARCH:\s*([^\]]+)\]',
    re.IGNORECASE
)

EVIDENCE_QUALITY_PATTERN = re.compile(
    r'\[(STANDARD PRACTICE|CASE SERIES|CLINICAL EXPERIENCE|NO COMPARATIVE DATA)\]',
    re.IGNORECASE
)

# Quantitative patterns (percentages, rates, numbers with units)
QUANTITATIVE_PATTERN = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*(%|percent|rate|mortality|survival|patients|cases)\b',
    re.IGNORECASE
)

# Alternative patterns for claims without explicit search hints
QUANTITATIVE_CLAIM_PATTERN = re.compile(
    r'([^.]*?\b(?:\d+(?:\.\d+)?)\s*(?:%|percent)[^.]*\.)',
    re.IGNORECASE
)


def extract_claims_from_responses(
    responses: Dict[str, Dict],
    include_quantitative_without_hints: bool = True
) -> ClaimExtractionResult:
    """
    Extract searchable claims from expert responses.

    Looks for:
    1. Claims with explicit [SEARCH: query] hints
    2. Evidence quality markers [STANDARD PRACTICE], etc.
    3. Quantitative claims (percentages, rates) even without hints

    Args:
        responses: Dict mapping expert name to response dict with 'content' key
        include_quantitative_without_hints: Also extract quantitative claims without SEARCH hints

    Returns:
        ClaimExtractionResult with all extracted claims
    """
    result = ClaimExtractionResult()

    for expert, response in responses.items():
        # Handle both dict and string responses
        if isinstance(response, dict):
            content = response.get('content', '')
        else:
            content = str(response)

        if not content:
            continue

        # Extract claims with explicit search hints
        claims_with_hints = _extract_claims_with_search_hints(content, expert)
        for claim in claims_with_hints:
            # Check for evidence quality marker nearby
            quality = _find_evidence_quality_marker(content, claim.text)
            if quality:
                claim.evidence_quality = quality
                result.claims_with_quality_markers += 1

            # Check if quantitative
            if QUANTITATIVE_PATTERN.search(claim.text):
                claim.has_quantitative_data = True
                result.quantitative_claims += 1

            result.claims.append(claim)
            result.claims_with_hints += 1

        # Also extract quantitative claims without explicit hints
        if include_quantitative_without_hints:
            existing_texts = {c.text.lower() for c in result.claims}
            quant_claims = _extract_quantitative_claims(content, expert, existing_texts)

            for claim in quant_claims:
                quality = _find_evidence_quality_marker(content, claim.text)
                if quality:
                    claim.evidence_quality = quality
                    result.claims_with_quality_markers += 1

                claim.has_quantitative_data = True
                result.quantitative_claims += 1
                result.claims.append(claim)

    logger.info(
        f"Extracted {len(result.claims)} claims: "
        f"{result.claims_with_hints} with hints, "
        f"{result.quantitative_claims} quantitative"
    )

    return result


def _extract_claims_with_search_hints(
    content: str,
    expert: str
) -> List[ExtractedClaim]:
    """Extract claims that have explicit [SEARCH: query] hints."""
    claims = []

    for match in SEARCH_HINT_PATTERN.finditer(content):
        claim_text = match.group(1).strip()
        search_hint = match.group(2).strip()

        # Clean up claim text (remove leading bullets, etc.)
        claim_text = re.sub(r'^[\s•\-\*]+', '', claim_text)

        # Get surrounding context (50 chars before and after)
        start = max(0, match.start() - 50)
        end = min(len(content), match.end() + 50)
        context = content[start:end].strip()

        claims.append(ExtractedClaim(
            text=claim_text,
            expert=expert,
            search_hint=search_hint,
            context=context
        ))

    return claims


def _extract_quantitative_claims(
    content: str,
    expert: str,
    existing_texts: set
) -> List[ExtractedClaim]:
    """
    Extract quantitative claims (with percentages/rates) that don't have explicit hints.

    These can still be searched using generated queries.
    """
    claims = []

    # Find sentences with percentages or rates
    for match in QUANTITATIVE_CLAIM_PATTERN.finditer(content):
        claim_text = match.group(1).strip()

        # Skip if too short or already captured
        if len(claim_text) < 20:
            continue
        if claim_text.lower() in existing_texts:
            continue

        # Skip meta-sentences
        if _is_meta_sentence(claim_text):
            continue

        # Generate a search hint from the claim
        search_hint = _generate_search_hint_from_claim(claim_text)

        claims.append(ExtractedClaim(
            text=claim_text,
            expert=expert,
            search_hint=search_hint,  # Auto-generated
            evidence_quality="CLINICAL_EXPERIENCE",  # Default
            has_quantitative_data=True
        ))

    return claims


def _find_evidence_quality_marker(content: str, claim_text: str) -> Optional[str]:
    """
    Find evidence quality marker near a claim.

    Looks for markers like [STANDARD PRACTICE], [CASE SERIES], etc.
    within 100 characters of the claim.
    """
    # Find position of claim in content
    claim_pos = content.lower().find(claim_text.lower()[:50])
    if claim_pos == -1:
        return None

    # Search in window around claim
    window_start = max(0, claim_pos - 100)
    window_end = min(len(content), claim_pos + len(claim_text) + 100)
    window = content[window_start:window_end]

    match = EVIDENCE_QUALITY_PATTERN.search(window)
    if match:
        return match.group(1).upper().replace(" ", "_")

    return None


def _is_meta_sentence(text: str) -> bool:
    """Check if sentence is meta/filler rather than clinical content."""
    meta_starts = [
        'as a', 'i would', 'based on my', 'in my opinion',
        'from my perspective', 'let me', 'this is', 'there are'
    ]
    text_lower = text.lower().strip()
    return any(text_lower.startswith(start) for start in meta_starts)


def _generate_search_hint_from_claim(claim_text: str) -> str:
    """
    Generate a search hint from a quantitative claim.

    Extracts key medical terms for PubMed search.
    """
    # Remove common words and punctuation
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'could', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'with', 'for', 'from', 'to', 'of',
        'in', 'on', 'at', 'by', 'about', 'as', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'between',
        'under', 'again', 'further', 'then', 'once', 'here', 'there',
        'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'or',
        'and', 'but', 'if', 'because', 'as', 'until', 'while',
        'approximately', 'typically', 'usually', 'often', 'sometimes'
    }

    # Extract words, keeping medical terms
    words = re.findall(r'\b[a-zA-Z]{3,}\b', claim_text.lower())
    keywords = [w for w in words if w not in stopwords]

    # Prioritize medical-sounding terms
    medical_indicators = [
        'ectomy', 'plasty', 'otomy', 'ostomy', 'therapy', 'osis', 'itis',
        'emia', 'oma', 'algia', 'metast', 'cancer', 'tumor', 'surgical',
        'palliative', 'mortality', 'survival', 'outcome', 'complication'
    ]

    scored_keywords = []
    for kw in keywords:
        score = 1
        if any(ind in kw for ind in medical_indicators):
            score = 3
        if len(kw) > 6:  # Longer words often more specific
            score += 1
        scored_keywords.append((kw, score))

    # Sort by score and take top terms
    scored_keywords.sort(key=lambda x: x[1], reverse=True)
    top_keywords = [kw for kw, _ in scored_keywords[:5]]

    # Add "outcome" or "outcomes" if not present
    if not any(kw in ['outcome', 'outcomes', 'results'] for kw in top_keywords):
        top_keywords.append('outcomes')

    return ' '.join(top_keywords)


def get_searchable_claims(
    extraction_result: ClaimExtractionResult,
    min_hint_length: int = 10
) -> List[ExtractedClaim]:
    """
    Get claims that have valid search hints.

    Args:
        extraction_result: Result from extract_claims_from_responses
        min_hint_length: Minimum length for search hint to be considered valid

    Returns:
        List of claims with searchable hints
    """
    return [
        claim for claim in extraction_result.claims
        if claim.search_hint and len(claim.search_hint) >= min_hint_length
    ]


def group_claims_by_expert(
    claims: List[ExtractedClaim]
) -> Dict[str, List[ExtractedClaim]]:
    """Group claims by expert for display."""
    grouped = {}
    for claim in claims:
        if claim.expert not in grouped:
            grouped[claim.expert] = []
        grouped[claim.expert].append(claim)
    return grouped
