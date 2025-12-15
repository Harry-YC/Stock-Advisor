"""
Response Validator for GDG (Guideline Development Group) Discussions

Validates expert responses for:
- Proper PMID citation format: EVIDENCE (PMID: XXXXXXXX)
- Citations are from the loaded corpus (no hallucinations)
- Epistemic tags are used appropriately: EVIDENCE, ASSUMPTION, OPINION, EVIDENCE GAP
- Response completeness and evidence quality
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of response validation."""
    is_valid: bool
    compliance_score: float  # 0.0 to 1.0
    errors: List[str]
    warnings: List[str]
    cited_pmids: Set[str]
    hallucinated_pmids: Set[str]
    epistemic_tags_found: Dict[str, int]
    evidence_without_pmid: List[str] = None  # EVIDENCE tags missing PMIDs

    def __post_init__(self):
        if self.evidence_without_pmid is None:
            self.evidence_without_pmid = []


class ResponseValidator:
    """
    Validates expert responses for citation discipline and epistemic tagging.

    Usage:
        validator = ResponseValidator(corpus_pmids={'12345678', '87654321'})
        result = validator.validate(response_text)

        if not result.is_valid:
            print("Validation errors:", result.errors)
    """

    # Epistemic tags we expect experts to use
    EPISTEMIC_TAGS = ['EVIDENCE', 'ASSUMPTION', 'OPINION', 'EVIDENCE GAP']

    def __init__(self, corpus_pmids: Optional[Set[str]] = None):
        """
        Initialize validator.

        Args:
            corpus_pmids: Set of valid PMIDs from loaded literature.
                          If None, PMID validation is skipped.
        """
        self.corpus_pmids = corpus_pmids or set()

    def validate(self, response_text: str, response_mode: str = "literature_verified") -> ValidationResult:
        """
        Validate an expert response.

        Args:
            response_text: The expert's response text
            response_mode: "expert_consensus" (relaxed) or "literature_verified" (strict PMIDs)

        Returns:
            ValidationResult with details
        """
        errors = []
        warnings = []

        # Extract cited PMIDs
        cited_pmids = self._extract_pmids(response_text)

        # Check for hallucinated PMIDs (only in literature_verified mode)
        hallucinated_pmids = set()
        if self.corpus_pmids and response_mode == "literature_verified":
            hallucinated_pmids = cited_pmids - self.corpus_pmids
            if hallucinated_pmids:
                errors.append(
                    f"Hallucinated PMIDs not in corpus: {', '.join(hallucinated_pmids)}"
                )

        # Check epistemic tags
        tags_found = self._count_epistemic_tags(response_text)
        total_tags = sum(tags_found.values())

        if total_tags == 0:
            warnings.append("No epistemic tags found. Responses should use EVIDENCE, ASSUMPTION:, OPINION:, or EVIDENCE GAP")

        # Check for EVIDENCE tags without PMIDs (mode-aware)
        evidence_without_pmid = self._find_evidence_without_pmid(response_text)
        if evidence_without_pmid:
            if response_mode == "literature_verified":
                # Strict mode: this is an error
                errors.append(
                    f"Found {len(evidence_without_pmid)} EVIDENCE tags without PMIDs. "
                    "Use EVIDENCE (PMID: XXXXXXXX) format, or use ASSUMPTION/OPINION for non-cited claims."
                )
            else:
                # Consensus mode: this is just a warning (acceptable)
                warnings.append(
                    f"Found {len(evidence_without_pmid)} EVIDENCE claims without PMID citations (acceptable in Expert Consensus mode)"
                )

        # Check for claims without tags (heuristic: quantitative claims should be tagged)
        untagged_claims = self._find_untagged_claims(response_text)
        if untagged_claims:
            warnings.append(
                f"Found {len(untagged_claims)} potential untagged quantitative claims"
            )

        # Check response length
        word_count = len(response_text.split())
        if word_count < 50:
            warnings.append("Response is very short (< 50 words)")
        elif word_count > 3000:
            warnings.append("Response is very long (> 3000 words)")

        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(
            cited_pmids=cited_pmids,
            hallucinated_pmids=hallucinated_pmids,
            tags_found=tags_found,
            untagged_claims=untagged_claims,
            word_count=word_count
        )

        # Determine validity
        is_valid = len(errors) == 0 and compliance_score >= 0.5

        return ValidationResult(
            is_valid=is_valid,
            compliance_score=compliance_score,
            errors=errors,
            warnings=warnings,
            cited_pmids=cited_pmids,
            hallucinated_pmids=hallucinated_pmids,
            epistemic_tags_found=tags_found,
            evidence_without_pmid=evidence_without_pmid
        )

    def _extract_pmids(self, text: str) -> Set[str]:
        """Extract PMIDs from text."""
        # Match patterns like (PMID: 12345678) or PMID:12345678 or PMID 12345678
        pattern = r'PMID[:\s]*(\d{7,8})'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return set(matches)

    def _count_epistemic_tags(self, text: str) -> Dict[str, int]:
        """Count occurrences of each epistemic tag."""
        counts = {}
        for tag in self.EPISTEMIC_TAGS:
            # Match both GDG style (EVIDENCE (PMID:), OPINION:, etc.) and bracketed style [TAG]
            if tag == 'EVIDENCE':
                # EVIDENCE (PMID: XXXXX) format
                pattern = rf'EVIDENCE\s*\(PMID|\[{tag}[:\]]'
            elif tag == 'EVIDENCE GAP':
                # EVIDENCE GAP → format
                pattern = rf'EVIDENCE GAP\s*→|\[{tag}[:\]]'
            else:
                # ASSUMPTION: or OPINION: format (with colon)
                pattern = rf'{tag}\s*:|\[{tag}[:\]]'
            matches = re.findall(pattern, text, re.IGNORECASE)
            counts[tag] = len(matches)
        return counts

    def _find_evidence_without_pmid(self, text: str) -> List[str]:
        """
        Find EVIDENCE tags that don't have an associated PMID.

        Valid formats:
        - EVIDENCE (PMID: 12345678) - Correct
        - [EVIDENCE] followed by PMID - Acceptable

        Invalid formats:
        - EVIDENCE: without PMID - Should use ASSUMPTION or OPINION
        - [EVIDENCE] without PMID nearby
        """
        invalid_evidence = []

        # Pattern 1: EVIDENCE: without PMID (common mistake)
        # Match "EVIDENCE:" or "EVIDENCE :" not followed by "(PMID" within 50 chars
        pattern_evidence_colon = r'EVIDENCE\s*:(?![^)]*PMID)'
        matches = re.finditer(pattern_evidence_colon, text, re.IGNORECASE)
        for m in matches:
            # Check if PMID appears within 100 characters after
            context = text[m.start():min(m.end() + 100, len(text))]
            if not re.search(r'PMID[:\s]*\d{7,8}', context, re.IGNORECASE):
                snippet = text[max(0, m.start()-20):min(len(text), m.end()+50)]
                invalid_evidence.append(f"EVIDENCE without PMID: '...{snippet}...'")

        # Pattern 2: [EVIDENCE] bracketed style without PMID nearby
        pattern_bracketed = r'\[EVIDENCE[:\]]'
        matches = re.finditer(pattern_bracketed, text, re.IGNORECASE)
        for m in matches:
            # Check if PMID appears within 100 characters after
            context = text[m.start():min(m.end() + 100, len(text))]
            if not re.search(r'PMID[:\s]*\d{7,8}', context, re.IGNORECASE):
                snippet = text[max(0, m.start()-20):min(len(text), m.end()+50)]
                invalid_evidence.append(f"[EVIDENCE] without PMID: '...{snippet}...'")

        return invalid_evidence

    def suggest_corrections(self, text: str) -> List[Dict[str, str]]:
        """
        Suggest corrections for epistemic tagging issues.

        Returns list of dicts with:
        - issue: Description of the problem
        - original: Original text snippet
        - suggestion: Suggested correction
        - tag_type: Suggested tag to use instead
        """
        suggestions = []

        # Find EVIDENCE without PMID and suggest conversion
        invalid = self._find_evidence_without_pmid(text)
        for item in invalid:
            suggestions.append({
                'issue': 'EVIDENCE tag without PMID citation',
                'original': item,
                'suggestion': 'Convert to ASSUMPTION: or OPINION: if no PMID available',
                'tag_type': 'ASSUMPTION'
            })

        # Find quantitative claims without tags
        untagged = self._find_untagged_claims(text)
        for claim in untagged:
            # Determine best tag based on content
            claim_lower = claim.lower()
            if any(term in claim_lower for term in ['study', 'trial', 'showed', 'reported', 'found']):
                suggested_tag = 'EVIDENCE (PMID: XXXXXXXX)'
            elif any(term in claim_lower for term in ['likely', 'probably', 'expect', 'estimate']):
                suggested_tag = 'ASSUMPTION:'
            else:
                suggested_tag = 'EVIDENCE (PMID: XXXXXXXX) or ASSUMPTION:'

            suggestions.append({
                'issue': 'Quantitative claim without epistemic tag',
                'original': claim[:100] + '...' if len(claim) > 100 else claim,
                'suggestion': f'Add {suggested_tag} before the claim',
                'tag_type': suggested_tag.split()[0] if suggested_tag else 'EVIDENCE'
            })

        return suggestions

    def enforce_pmid_requirement(
        self,
        text: str,
        corpus_pmids: Optional[Set[str]] = None,
        strict: bool = True,
        response_mode: str = "literature_verified"
    ) -> Dict[str, any]:
        """
        Strictly enforce PMID requirement for EVIDENCE tags.

        Args:
            text: Response text to validate
            corpus_pmids: Set of valid PMIDs (for hallucination check)
            strict: If True, any EVIDENCE without PMID is an error
            response_mode: "expert_consensus" (relaxed) or "literature_verified" (strict)

        Returns:
            Dict with:
            - valid: bool - passes strict PMID requirements
            - errors: List[str] - blocking errors
            - warnings: List[str] - non-blocking warnings
            - corrections_needed: List[Dict] - suggested fixes
            - pmids_cited: Set[str] - all PMIDs found
            - pmids_valid: Set[str] - PMIDs in corpus
            - pmids_invalid: Set[str] - PMIDs not in corpus (possible hallucinations)
        """
        errors = []
        warnings = []

        # Determine effective strictness based on response_mode
        is_strict = strict and response_mode == "literature_verified"

        # Find all PMIDs cited
        pmids_cited = self._extract_pmids(text)

        # Check against corpus (only in literature_verified mode)
        pmids_valid = set()
        pmids_invalid = set()
        if corpus_pmids and response_mode == "literature_verified":
            pmids_valid = pmids_cited & corpus_pmids
            pmids_invalid = pmids_cited - corpus_pmids
            if pmids_invalid:
                errors.append(
                    f"PMIDs not in evidence corpus (possible hallucination): {', '.join(pmids_invalid)}"
                )

        # Find EVIDENCE tags without PMIDs
        evidence_without_pmid = self._find_evidence_without_pmid(text)
        if evidence_without_pmid:
            if is_strict:
                errors.append(
                    f"Found {len(evidence_without_pmid)} EVIDENCE tag(s) without PMID. "
                    "Rule: EVIDENCE MUST have PMID. Convert to ASSUMPTION: or add citation."
                )
            else:
                warnings.append(
                    f"Found {len(evidence_without_pmid)} EVIDENCE tag(s) without PMID (acceptable in Expert Consensus mode)"
                )

        # Get correction suggestions
        corrections_needed = self.suggest_corrections(text)

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'corrections_needed': corrections_needed,
            'pmids_cited': pmids_cited,
            'pmids_valid': pmids_valid,
            'pmids_invalid': pmids_invalid,
            'evidence_without_pmid_count': len(evidence_without_pmid)
        }

    def _find_untagged_claims(self, text: str) -> List[str]:
        """
        Find potential quantitative claims that lack epistemic tags.

        Heuristic: Sentences with percentages, numbers, or statistical terms
        that don't have an epistemic tag nearby.
        """
        untagged = []

        # Split into sentences
        sentences = re.split(r'[.!?]', text)

        # Patterns that suggest a claim requiring evidence
        claim_patterns = [
            r'\d+%',  # Percentages
            r'\d+\.\d+',  # Decimal numbers
            r'p\s*[<=]\s*0\.\d+',  # P-values
            r'HR\s*[=:]\s*\d',  # Hazard ratios
            r'OR\s*[=:]\s*\d',  # Odds ratios
            r'CI\s*[=:]\s*\[?\d',  # Confidence intervals
            r'median\s+\w+\s+was',  # Median values
            r'response rate',
            r'survival',
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if sentence contains a potential claim
            has_claim = any(re.search(p, sentence, re.IGNORECASE) for p in claim_patterns)

            if has_claim:
                # Check if it has an epistemic tag or PMID (both GDG and bracketed style)
                has_tag = any(
                    re.search(rf'\[{tag}|{tag}\s*[:(→]', sentence, re.IGNORECASE)
                    for tag in self.EPISTEMIC_TAGS
                )
                has_pmid = bool(re.search(r'PMID', sentence, re.IGNORECASE))

                if not has_tag and not has_pmid:
                    untagged.append(sentence[:100])  # First 100 chars

        return untagged

    def _calculate_compliance_score(
        self,
        cited_pmids: Set[str],
        hallucinated_pmids: Set[str],
        tags_found: Dict[str, int],
        untagged_claims: List[str],
        word_count: int
    ) -> float:
        """
        Calculate overall compliance score (0.0 to 1.0).

        Components:
        - Citation validity: 30%
        - Epistemic tagging: 30%
        - Response substance: 20%
        - No hallucinations: 20%
        """
        score = 0.0

        # Citation validity (30%)
        if cited_pmids:
            valid_citations = len(cited_pmids - hallucinated_pmids)
            citation_score = valid_citations / len(cited_pmids) if cited_pmids else 1.0
            score += 0.30 * citation_score
        else:
            score += 0.15  # Partial credit if no citations expected

        # Epistemic tagging (30%)
        total_tags = sum(tags_found.values())
        if total_tags > 0:
            # Ideal: mix of EVIDENCE and other tags
            evidence_ratio = tags_found.get('EVIDENCE', 0) / total_tags
            # Reward having a mix (not all EVIDENCE, not all OPINION)
            tag_diversity = 1.0 - abs(evidence_ratio - 0.5) * 2
            tag_score = min(1.0, total_tags / 5) * 0.5 + tag_diversity * 0.5
            score += 0.30 * tag_score
        else:
            score += 0.10  # Some credit for substantial response

        # Response substance (20%)
        if word_count >= 200:
            score += 0.20
        elif word_count >= 100:
            score += 0.15
        elif word_count >= 50:
            score += 0.10

        # No hallucinations (20%)
        if not hallucinated_pmids:
            score += 0.20
        elif len(hallucinated_pmids) == 1:
            score += 0.10  # Partial credit for single error

        return round(score, 2)

    def format_validation_summary(self, result: ValidationResult) -> str:
        """
        Format validation result as human-readable summary.

        Args:
            result: ValidationResult to format

        Returns:
            Formatted string
        """
        lines = []

        # Overall status
        status = "VALID" if result.is_valid else "NEEDS REVIEW"
        lines.append(f"**Validation Status:** {status}")
        lines.append(f"**Compliance Score:** {result.compliance_score:.0%}")
        lines.append("")

        # Citation summary
        lines.append(f"**Citations:** {len(result.cited_pmids)} PMIDs cited")
        if result.hallucinated_pmids:
            lines.append(f"  - WARNING: {len(result.hallucinated_pmids)} hallucinated PMIDs")

        # Epistemic tags
        total_tags = sum(result.epistemic_tags_found.values())
        lines.append(f"**Epistemic Tags:** {total_tags} found")
        for tag, count in result.epistemic_tags_found.items():
            if count > 0:
                lines.append(f"  - [{tag}]: {count}")

        # EVIDENCE without PMID
        if result.evidence_without_pmid:
            lines.append(f"  - WARNING: {len(result.evidence_without_pmid)} EVIDENCE tags without PMIDs")

        # Errors
        if result.errors:
            lines.append("")
            lines.append("**Errors:**")
            for error in result.errors:
                lines.append(f"  - {error}")

        # Warnings
        if result.warnings:
            lines.append("")
            lines.append("**Warnings:**")
            for warning in result.warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)


def validate_all_responses(
    responses: Dict[str, str],
    corpus_pmids: Set[str]
) -> Dict[str, ValidationResult]:
    """
    Validate responses from multiple experts.

    Args:
        responses: Dict mapping expert_name -> response_text
        corpus_pmids: Set of valid PMIDs

    Returns:
        Dict mapping expert_name -> ValidationResult
    """
    validator = ResponseValidator(corpus_pmids)
    results = {}

    for expert_name, response_text in responses.items():
        results[expert_name] = validator.validate(response_text)

    return results


def get_compliance_summary(results: Dict[str, ValidationResult]) -> Dict[str, any]:
    """
    Get aggregate compliance statistics across all experts.

    Args:
        results: Dict of validation results

    Returns:
        Summary dict with aggregate stats
    """
    if not results:
        return {
            'average_score': 0.0,
            'experts_valid': 0,
            'experts_total': 0,
            'total_hallucinations': 0,
            'total_citations': 0
        }

    scores = [r.compliance_score for r in results.values()]
    valid_count = sum(1 for r in results.values() if r.is_valid)
    total_hallucinations = sum(len(r.hallucinated_pmids) for r in results.values())
    total_citations = sum(len(r.cited_pmids) for r in results.values())

    return {
        'average_score': sum(scores) / len(scores),
        'experts_valid': valid_count,
        'experts_total': len(results),
        'total_hallucinations': total_hallucinations,
        'total_citations': total_citations
    }


# =============================================================================
# EVIDENCE CORPUS INTEGRATION
# =============================================================================

def validate_with_corpus(response_text: str, corpus: 'EvidenceCorpus') -> ValidationResult:
    """
    Validate a response against the Evidence Corpus.

    This is the primary validation function that enforces the rule:
    "GDG experts can ONLY cite from included_pmids"

    Args:
        response_text: Expert response text
        corpus: EvidenceCorpus instance

    Returns:
        ValidationResult with corpus-aware validation
    """
    # Get included PMIDs from corpus
    corpus_pmids = corpus.included_pmids if corpus else set()

    # Create validator with corpus PMIDs
    validator = ResponseValidator(corpus_pmids=corpus_pmids)
    return validator.validate(response_text)


def validate_all_with_corpus(
    responses: Dict[str, str],
    corpus: 'EvidenceCorpus'
) -> Dict[str, ValidationResult]:
    """
    Validate all expert responses against the Evidence Corpus.

    Args:
        responses: Dict mapping expert_name -> response_text
        corpus: EvidenceCorpus instance

    Returns:
        Dict mapping expert_name -> ValidationResult
    """
    corpus_pmids = corpus.included_pmids if corpus else set()
    validator = ResponseValidator(corpus_pmids=corpus_pmids)
    results = {}

    for expert_name, response_text in responses.items():
        results[expert_name] = validator.validate(response_text)

    return results


def get_citation_warnings(
    responses: Dict[str, str],
    corpus: 'EvidenceCorpus'
) -> List[Dict[str, any]]:
    """
    Get list of citation warnings for display in UI.

    Returns warnings for:
    - Hallucinated PMIDs (not in corpus)
    - Claims without citations
    - Low quality evidence only

    Args:
        responses: Expert responses
        corpus: EvidenceCorpus instance

    Returns:
        List of warning dicts with 'type', 'message', 'severity', 'expert'
    """
    warnings = []
    results = validate_all_with_corpus(responses, corpus)

    for expert_name, result in results.items():
        # Hallucinated PMIDs
        if result.hallucinated_pmids:
            warnings.append({
                'type': 'hallucination',
                'message': f"Cited PMIDs not in evidence corpus: {', '.join(result.hallucinated_pmids)}",
                'severity': 'error',
                'expert': expert_name,
                'pmids': list(result.hallucinated_pmids)
            })

        # Missing citations
        if result.epistemic_tags_found.get('EVIDENCE', 0) == 0 and len(result.cited_pmids) == 0:
            warnings.append({
                'type': 'no_citations',
                'message': "Response contains no evidence citations",
                'severity': 'warning',
                'expert': expert_name
            })

        # Untagged claims warning is already in result.warnings
        for w in result.warnings:
            if 'untagged' in w.lower():
                warnings.append({
                    'type': 'untagged_claims',
                    'message': w,
                    'severity': 'warning',
                    'expert': expert_name
                })

    # Check for low quality evidence across all responses
    if corpus and corpus.quality_ratings:
        all_cited = set()
        for result in results.values():
            all_cited.update(result.cited_pmids)

        low_quality_only = True
        for pmid in all_cited:
            rating = corpus.quality_ratings.get(pmid)
            if rating:
                # Check if any high/moderate quality evidence
                certainty = rating.get('certainty', '') if isinstance(rating, dict) else getattr(rating, 'certainty', '')
                if certainty in ['High', 'Moderate']:
                    low_quality_only = False
                    break

        if all_cited and low_quality_only:
            warnings.append({
                'type': 'low_quality_evidence',
                'message': "All cited evidence is low quality or very low quality",
                'severity': 'warning',
                'expert': 'All'
            })

    return warnings


def check_claims_have_citations(response_text: str) -> Dict[str, any]:
    """
    Check if quantitative claims in a response have citations.

    Returns dict with:
    - claims_found: Number of potential claims detected
    - claims_cited: Number with PMID citations
    - uncited_claims: List of claim snippets without citations
    """
    validator = ResponseValidator()
    untagged = validator._find_untagged_claims(response_text)
    cited_pmids = validator._extract_pmids(response_text)

    # Simple heuristic: count sentences with numbers
    sentences = re.split(r'[.!?]', response_text)
    claims_found = 0
    claims_cited = 0

    for sentence in sentences:
        # Check for quantitative patterns
        has_numbers = bool(re.search(r'\d+%|\d+\.\d+|n\s*=\s*\d+', sentence))
        if has_numbers:
            claims_found += 1
            if re.search(r'PMID', sentence, re.IGNORECASE):
                claims_cited += 1

    return {
        'claims_found': claims_found,
        'claims_cited': claims_cited,
        'uncited_claims': untagged,
        'citation_rate': claims_cited / claims_found if claims_found > 0 else 1.0
    }
