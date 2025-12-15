"""
Citation Verification Module

Verifies that LLM-generated citations match the source evidence.
Helps detect hallucinations and ensures claims are grounded in evidence.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of citation verification."""
    verified_count: int = 0
    unverified_count: int = 0
    invalid_count: int = 0
    citations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_citations(self) -> int:
        return self.verified_count + self.unverified_count + self.invalid_count

    @property
    def verification_rate(self) -> float:
        if self.total_citations == 0:
            return 1.0
        return self.verified_count / self.total_citations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified_count": self.verified_count,
            "unverified_count": self.unverified_count,
            "invalid_count": self.invalid_count,
            "total_citations": self.total_citations,
            "verification_rate": self.verification_rate,
            "citations": self.citations
        }


def extract_citations(response_text: str) -> List[Dict[str, Any]]:
    """
    Extract citations from LLM response.

    Supports formats:
    - [1], [2], etc. (numbered references)
    - (PMID: 12345678)
    - [Source: filename.pdf]

    Args:
        response_text: LLM response text

    Returns:
        List of extracted citations with type and reference
    """
    citations = []

    # Pattern 1: Numbered references [1], [2], [1,2], [1-3]
    numbered_pattern = r'\[(\d+(?:\s*,\s*\d+)*(?:\s*-\s*\d+)?)\]'
    for match in re.finditer(numbered_pattern, response_text):
        ref_text = match.group(1)
        # Handle ranges like [1-3]
        if '-' in ref_text:
            parts = ref_text.split('-')
            try:
                start, end = int(parts[0].strip()), int(parts[1].strip())
                for i in range(start, end + 1):
                    citations.append({
                        "type": "numbered",
                        "reference": str(i),
                        "raw": match.group(0),
                        "position": match.start()
                    })
            except ValueError:
                pass
        else:
            # Handle comma-separated like [1,2]
            for ref in ref_text.split(','):
                citations.append({
                    "type": "numbered",
                    "reference": ref.strip(),
                    "raw": match.group(0),
                    "position": match.start()
                })

    # Pattern 2: PMID citations (PMID: 12345678)
    pmid_pattern = r'\(PMID[:\s]+(\d{7,8})\)'
    for match in re.finditer(pmid_pattern, response_text, re.IGNORECASE):
        citations.append({
            "type": "pmid",
            "reference": match.group(1),
            "raw": match.group(0),
            "position": match.start()
        })

    # Pattern 3: Source citations [Source: filename.pdf]
    source_pattern = r'\[Source:\s*([^\]]+)\]'
    for match in re.finditer(source_pattern, response_text, re.IGNORECASE):
        citations.append({
            "type": "source",
            "reference": match.group(1).strip(),
            "raw": match.group(0),
            "position": match.start()
        })

    return citations


def find_supporting_evidence(
    citation: Dict[str, Any],
    evidence_list: List[Dict[str, Any]],
    context_window: int = 500
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Find evidence that supports a citation.

    Args:
        citation: Citation dict with type and reference
        evidence_list: List of evidence chunks/papers
        context_window: Characters around citation to check for matching content

    Returns:
        Tuple of (found, matching_evidence)
    """
    ref_type = citation.get('type')
    ref_value = citation.get('reference', '')

    for evidence in evidence_list:
        # Check numbered references (match by index)
        if ref_type == 'numbered':
            try:
                idx = int(ref_value) - 1  # 1-indexed to 0-indexed
                if 0 <= idx < len(evidence_list):
                    return True, evidence_list[idx]
            except ValueError:
                pass

        # Check PMID references
        if ref_type == 'pmid':
            evidence_pmid = evidence.get('pmid') or evidence.get('source', '')
            if ref_value in str(evidence_pmid):
                return True, evidence

        # Check source references
        if ref_type == 'source':
            evidence_source = evidence.get('source', '')
            if ref_value.lower() in evidence_source.lower():
                return True, evidence

    return False, None


def verify_citations(
    response_text: str,
    evidence_list: List[Dict[str, Any]],
    strict: bool = False
) -> VerificationResult:
    """
    Verify that citations in LLM response match source evidence.

    Args:
        response_text: LLM-generated response
        evidence_list: List of evidence chunks/papers that were provided
        strict: If True, require exact content match; if False, just check reference exists

    Returns:
        VerificationResult with counts and details
    """
    result = VerificationResult()

    # Extract citations
    citations = extract_citations(response_text)

    for citation in citations:
        found, matching_evidence = find_supporting_evidence(citation, evidence_list)

        citation_record = citation.copy()

        if found:
            citation_record['status'] = 'verified'
            citation_record['matching_source'] = matching_evidence.get('source', 'Unknown')
            result.verified_count += 1
        else:
            # Check if it's a valid format but not in our evidence
            if citation.get('type') in ('numbered', 'pmid', 'source'):
                citation_record['status'] = 'unverified'
                result.unverified_count += 1
            else:
                citation_record['status'] = 'invalid'
                result.invalid_count += 1

        result.citations.append(citation_record)

    logger.info(
        f"Citation verification: {result.verified_count} verified, "
        f"{result.unverified_count} unverified, {result.invalid_count} invalid"
    )

    return result


def check_for_hallucinations(
    response_text: str,
    evidence_list: List[Dict[str, Any]],
    key_claims: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Check for potential hallucinations in LLM response.

    Looks for:
    1. Citations that don't match evidence
    2. Numerical claims not found in evidence
    3. Specific factual claims without citation

    Args:
        response_text: LLM-generated response
        evidence_list: Source evidence
        key_claims: Optional list of specific claims to verify

    Returns:
        Dict with hallucination analysis
    """
    # Verify citations first
    citation_result = verify_citations(response_text, evidence_list)

    # Extract numerical claims
    number_pattern = r'(\d+(?:\.\d+)?)\s*%'
    numerical_claims = re.findall(number_pattern, response_text)

    # Check if numerical claims appear in evidence
    ungrounded_numbers = []
    evidence_text = ' '.join(e.get('content', '') for e in evidence_list)

    for num in numerical_claims:
        if num not in evidence_text:
            ungrounded_numbers.append(num)

    return {
        "citation_verification": citation_result.to_dict(),
        "numerical_claims": len(numerical_claims),
        "ungrounded_numbers": ungrounded_numbers,
        "potential_hallucination_risk": (
            citation_result.unverified_count > 0 or
            len(ungrounded_numbers) > 2
        )
    }


def format_verification_report(result: VerificationResult) -> str:
    """
    Format verification result for display.

    Args:
        result: VerificationResult

    Returns:
        Formatted report string
    """
    lines = [
        f"**Citation Verification Report**",
        f"",
        f"- Verified: {result.verified_count}",
        f"- Unverified: {result.unverified_count}",
        f"- Invalid: {result.invalid_count}",
        f"- Verification Rate: {result.verification_rate:.1%}",
        ""
    ]

    if result.unverified_count > 0:
        lines.append("**Unverified Citations:**")
        for c in result.citations:
            if c.get('status') == 'unverified':
                lines.append(f"- {c.get('raw', 'Unknown')}: Not found in evidence")

    return "\n".join(lines)
