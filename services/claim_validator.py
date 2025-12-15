"""
Claim Validator Service

Validates expert claims against two-channel evidence.
Routes claims to the appropriate channel for validation.

Claim Types:
- CLINICAL: "Drug X showed efficacy" → validate against Clinical channel
- BIOLOGY: "FOLR1 is expressed in NSCLC" → validate against Biology channel
- SAFETY: "IL-2 causes CRS" → validate against Biology (toxicity) channel
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

from core.pubmed_client import Citation
from services.two_channel_search_service import TwoChannelSearchResult

logger = logging.getLogger(__name__)


class ClaimType(Enum):
    CLINICAL = "clinical"      # Trial results, efficacy, competitive landscape
    BIOLOGY = "biology"        # Expression, mechanism, biomarker
    SAFETY = "safety"          # Toxicity, adverse events (subset of biology)
    UNKNOWN = "unknown"


class ValidationStatus(Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NO_EVIDENCE = "no_evidence"
    PARTIAL = "partial"


@dataclass
class ValidatedClaim:
    """A claim with its validation result."""
    claim_text: str
    claim_type: ClaimType
    status: ValidationStatus
    supporting_citations: List[Citation] = field(default_factory=list)
    contradicting_citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0  # 0-1
    validation_note: str = ""

    def to_dict(self) -> Dict:
        return {
            "claim_text": self.claim_text,
            "claim_type": self.claim_type.value,
            "status": self.status.value,
            "supporting_pmids": [c.pmid for c in self.supporting_citations],
            "contradicting_pmids": [c.pmid for c in self.contradicting_citations],
            "confidence": self.confidence,
            "validation_note": self.validation_note
        }


@dataclass
class ValidationResult:
    """Complete validation result for an expert response."""
    claims: List[ValidatedClaim] = field(default_factory=list)
    total_claims: int = 0
    claims_supported: int = 0
    claims_contradicted: int = 0
    claims_no_evidence: int = 0
    validation_text: str = ""  # Formatted summary

    def to_dict(self) -> Dict:
        return {
            "claims": [c.to_dict() for c in self.claims],
            "total_claims": self.total_claims,
            "claims_supported": self.claims_supported,
            "claims_contradicted": self.claims_contradicted,
            "claims_no_evidence": self.claims_no_evidence,
            "validation_text": self.validation_text
        }


class ClaimValidator:
    """
    Validates expert claims against two-channel evidence.

    Key insight: Different claims need different evidence channels.
    - "Standard of care is X" → Clinical channel
    - "FOLR1 is expressed in 40% of NSCLC" → Biology channel
    """

    # Keywords for classifying claims
    CLINICAL_KEYWORDS = [
        "phase 2", "phase 3", "phase ii", "phase iii",
        "clinical trial", "trial", "efficacy", "response rate",
        "overall survival", "progression-free", "standard of care",
        "first-line", "second-line", "frontline", "approved",
        "fda", "ema", "median survival", "orr", "pfs", "os"
    ]

    BIOLOGY_KEYWORDS = [
        "expression", "expressed", "prevalence", "biomarker",
        "immunohistochemistry", "ihc", "mechanism", "pathway",
        "signaling", "receptor", "ligand", "binding",
        "tumor microenvironment", "tme", "macrophage", "myeloid"
    ]

    SAFETY_KEYWORDS = [
        "toxicity", "adverse", "side effect", "safety",
        "tolerability", "cytokine release", "crs", "ild",
        "interstitial lung", "hepatotoxicity", "nephrotoxicity",
        "maximum tolerated", "mtd", "dose-limiting"
    ]

    def __init__(self, api_key: str = None):
        """
        Initialize claim validator.

        Args:
            api_key: API key for LLM-based claim extraction (optional)
        """
        self.api_key = api_key

    def extract_claims(self, expert_response: str) -> List[Tuple[str, ClaimType]]:
        """
        Extract verifiable claims from expert response.

        Returns list of (claim_text, claim_type) tuples.
        """
        claims = []

        # Split into sentences
        sentences = re.split(r'[.!?]+', expert_response)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue

            # Skip non-factual sentences
            if self._is_non_factual(sentence):
                continue

            # Classify the claim
            claim_type = self._classify_claim(sentence)
            if claim_type != ClaimType.UNKNOWN:
                claims.append((sentence, claim_type))

        logger.info(f"Extracted {len(claims)} verifiable claims from response")
        return claims

    def _is_non_factual(self, sentence: str) -> bool:
        """Check if sentence is opinion/question/speculation."""
        lower = sentence.lower()

        # Skip questions
        if "?" in sentence:
            return True

        # Skip hedging/speculation
        speculation_markers = [
            "could potentially", "might be", "may be possible",
            "i recommend", "i suggest", "in my opinion",
            "it is worth", "it would be", "should consider"
        ]
        if any(m in lower for m in speculation_markers):
            return True

        return False

    def _classify_claim(self, sentence: str) -> ClaimType:
        """Classify a claim into Clinical, Biology, or Safety."""
        lower = sentence.lower()

        # Count keyword matches
        clinical_score = sum(1 for k in self.CLINICAL_KEYWORDS if k in lower)
        biology_score = sum(1 for k in self.BIOLOGY_KEYWORDS if k in lower)
        safety_score = sum(1 for k in self.SAFETY_KEYWORDS if k in lower)

        # Safety is a subset of biology but gets priority
        if safety_score > 0:
            return ClaimType.SAFETY
        elif clinical_score > biology_score:
            return ClaimType.CLINICAL
        elif biology_score > 0:
            return ClaimType.BIOLOGY
        elif clinical_score > 0:
            return ClaimType.CLINICAL

        return ClaimType.UNKNOWN

    def validate_claims(
        self,
        expert_response: str,
        search_result: TwoChannelSearchResult
    ) -> ValidationResult:
        """
        Validate all claims in expert response against two-channel evidence.

        Args:
            expert_response: The expert's response text
            search_result: Results from two-channel search

        Returns:
            ValidationResult with claim-by-claim validation
        """
        result = ValidationResult()

        # Extract claims
        claims = self.extract_claims(expert_response)
        result.total_claims = len(claims)

        if not claims:
            result.validation_text = "No verifiable claims identified in response."
            return result

        # Validate each claim
        for claim_text, claim_type in claims:
            validated = self._validate_single_claim(
                claim_text,
                claim_type,
                search_result
            )
            result.claims.append(validated)

            # Update counts
            if validated.status == ValidationStatus.SUPPORTED:
                result.claims_supported += 1
            elif validated.status == ValidationStatus.CONTRADICTED:
                result.claims_contradicted += 1
            else:
                result.claims_no_evidence += 1

        # Generate summary text
        result.validation_text = self._generate_validation_summary(result)

        return result

    def _validate_single_claim(
        self,
        claim_text: str,
        claim_type: ClaimType,
        search_result: TwoChannelSearchResult
    ) -> ValidatedClaim:
        """Validate a single claim against the appropriate channel."""
        validated = ValidatedClaim(
            claim_text=claim_text,
            claim_type=claim_type,
            status=ValidationStatus.NO_EVIDENCE
        )

        # Select appropriate citation pool
        if claim_type in [ClaimType.BIOLOGY, ClaimType.SAFETY]:
            citations = search_result.biology.citations
        else:
            citations = search_result.clinical.citations

        if not citations:
            validated.validation_note = f"No {claim_type.value} evidence available"
            return validated

        # Extract key terms from claim
        key_terms = self._extract_key_terms(claim_text)

        # Find supporting/contradicting citations
        for citation in citations:
            text = f"{citation.title} {citation.abstract}".lower()

            # Check term overlap
            matching_terms = [t for t in key_terms if t.lower() in text]
            overlap = len(matching_terms) / len(key_terms) if key_terms else 0

            if overlap >= 0.3:  # At least 30% term overlap
                # Check for contradiction signals
                if self._has_contradiction_signal(claim_text, text):
                    validated.contradicting_citations.append(citation)
                else:
                    validated.supporting_citations.append(citation)

        # Determine status
        n_supporting = len(validated.supporting_citations)
        n_contradicting = len(validated.contradicting_citations)

        if n_contradicting > 0 and n_supporting == 0:
            validated.status = ValidationStatus.CONTRADICTED
            validated.confidence = min(0.9, 0.3 + n_contradicting * 0.2)
        elif n_supporting > 0 and n_contradicting == 0:
            validated.status = ValidationStatus.SUPPORTED
            validated.confidence = min(0.9, 0.3 + n_supporting * 0.2)
        elif n_supporting > 0 and n_contradicting > 0:
            validated.status = ValidationStatus.PARTIAL
            validated.confidence = 0.5
        else:
            validated.status = ValidationStatus.NO_EVIDENCE
            validated.confidence = 0.0

        # Limit citations kept
        validated.supporting_citations = validated.supporting_citations[:3]
        validated.contradicting_citations = validated.contradicting_citations[:2]

        return validated

    def _extract_key_terms(self, claim_text: str) -> List[str]:
        """Extract key biomedical terms from claim."""
        # Simple extraction based on capitalized words and known patterns
        terms = []

        # Capitalized words (likely entities)
        for word in claim_text.split():
            clean = re.sub(r'[^\w]', '', word)
            if len(clean) >= 2 and clean[0].isupper():
                terms.append(clean)

        # Percentage patterns (e.g., "40%")
        percentages = re.findall(r'\d+%', claim_text)
        terms.extend(percentages)

        # Numbers with units
        numbers = re.findall(r'\d+\s*(?:mg|kg|months?|years?|weeks?)', claim_text.lower())
        terms.extend(numbers)

        return list(set(terms))

    def _has_contradiction_signal(self, claim: str, evidence: str) -> bool:
        """Check if evidence contradicts the claim."""
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()

        # Simple contradiction patterns
        contradiction_pairs = [
            ("high expression", "low expression"),
            ("overexpressed", "underexpressed"),
            ("effective", "ineffective"),
            ("safe", "unsafe"),
            ("tolerable", "intolerable"),
            ("positive", "negative"),
            ("significant", "not significant"),
        ]

        for pos, neg in contradiction_pairs:
            if pos in claim_lower and neg in evidence_lower:
                return True
            if neg in claim_lower and pos in evidence_lower:
                return True

        # Check for negation patterns in evidence when claim is positive
        negation_patterns = [
            "no evidence", "failed to show", "did not demonstrate",
            "no significant", "contrary to", "in contrast"
        ]

        for pattern in negation_patterns:
            if pattern in evidence_lower:
                return True

        return False

    def _generate_validation_summary(self, result: ValidationResult) -> str:
        """Generate human-readable validation summary."""
        lines = []

        lines.append(f"**Validation Summary**: {result.total_claims} claims analyzed")
        lines.append("")

        if result.claims_supported > 0:
            lines.append(f"✅ **{result.claims_supported} claims supported** by literature:")
            for claim in result.claims:
                if claim.status == ValidationStatus.SUPPORTED:
                    pmids = ", ".join([c.pmid for c in claim.supporting_citations[:2]])
                    lines.append(f"  - \"{claim.claim_text[:80]}...\" (PMID: {pmids})")
            lines.append("")

        if result.claims_contradicted > 0:
            lines.append(f"⚠️ **{result.claims_contradicted} claims contradicted** by literature:")
            for claim in result.claims:
                if claim.status == ValidationStatus.CONTRADICTED:
                    pmids = ", ".join([c.pmid for c in claim.contradicting_citations[:2]])
                    lines.append(f"  - \"{claim.claim_text[:80]}...\" (PMID: {pmids})")
            lines.append("")

        if result.claims_no_evidence > 0:
            lines.append(f"❓ **{result.claims_no_evidence} claims** could not be verified")

        return "\n".join(lines)


# =============================================================================
# CLAIM LEDGER (for Answer View)
# =============================================================================

@dataclass
class ClaimLedgerEntry:
    """A single claim in the ledger with its validation status."""
    claim: str
    status: str  # SUPPORTED | UNCLEAR | CONTRADICTED
    citations: List[str]  # PMIDs cited
    reason: str  # Why unclear/contradicted
    source_expert: str
    claim_type: str = "factual"  # factual | opinion | assumption | guideline


@dataclass
class ClaimLedger:
    """Collection of validated claims with summary statistics."""
    claims: List[ClaimLedgerEntry] = field(default_factory=list)

    @property
    def supported_count(self) -> int:
        return sum(1 for c in self.claims if c.status == "SUPPORTED")

    @property
    def unclear_count(self) -> int:
        return sum(1 for c in self.claims if c.status == "UNCLEAR")

    @property
    def contradicted_count(self) -> int:
        return sum(1 for c in self.claims if c.status == "CONTRADICTED")

    @property
    def total_count(self) -> int:
        return len(self.claims)

    @property
    def support_rate(self) -> float:
        """Proportion of claims that are supported."""
        if not self.claims:
            return 0.0
        return self.supported_count / len(self.claims)

    def to_dict(self) -> Dict:
        return {
            'claims': [
                {
                    'claim': c.claim,
                    'status': c.status,
                    'citations': c.citations,
                    'reason': c.reason,
                    'source_expert': c.source_expert,
                    'claim_type': c.claim_type
                }
                for c in self.claims
            ],
            'supported': self.supported_count,
            'unclear': self.unclear_count,
            'contradicted': self.contradicted_count,
            'total': self.total_count,
            'support_rate': self.support_rate
        }


def extract_claims_with_pmids(text: str) -> List[Tuple[str, List[str], str]]:
    """
    Extract claims from expert response text based on epistemic tags.

    Returns list of tuples: (claim_text, cited_pmids, claim_type)
    """
    claims = []

    # Pattern for EVIDENCE tags with PMIDs
    evidence_pattern = r'(?:EVIDENCE\s*\(PMID[:\s]*(\d{7,8})\)|PMID[:\s]*(\d{7,8}))[:\s]*([^.!?\n]+[.!?]?)'
    evidence_matches = re.finditer(evidence_pattern, text, re.IGNORECASE)

    for match in evidence_matches:
        pmid = match.group(1) or match.group(2)
        claim_text = match.group(3).strip()
        if claim_text and len(claim_text) > 10:
            claims.append((claim_text[:150], [pmid], 'factual'))

    # Pattern for claims with numbers but no PMID (potential uncited claims)
    number_pattern = r'(?:^|\. )([^.]*?\d+(?:\.\d+)?%[^.]*\.)'
    number_matches = re.finditer(number_pattern, text)

    for match in number_matches:
        claim_text = match.group(1).strip()
        # Check if this claim already captured with PMID
        if not any(claim_text in c[0] for c in claims):
            # Check if PMID is nearby (within 50 chars after)
            start = match.end()
            nearby_text = text[start:start+50]
            pmid_match = re.search(r'PMID[:\s]*(\d{7,8})', nearby_text, re.IGNORECASE)
            if pmid_match:
                claims.append((claim_text[:150], [pmid_match.group(1)], 'factual'))
            else:
                claims.append((claim_text[:150], [], 'factual'))

    # Pattern for OPINION tags
    opinion_pattern = r'OPINION[:\s]*([^.!?\n]+[.!?]?)'
    opinion_matches = re.finditer(opinion_pattern, text, re.IGNORECASE)
    for match in opinion_matches:
        claim_text = match.group(1).strip()
        if claim_text and len(claim_text) > 10:
            claims.append((claim_text[:150], [], 'opinion'))

    # Pattern for ASSUMPTION tags
    assumption_pattern = r'ASSUMPTION[:\s]*([^.!?\n]+[.!?]?)'
    assumption_matches = re.finditer(assumption_pattern, text, re.IGNORECASE)
    for match in assumption_matches:
        claim_text = match.group(1).strip()
        if claim_text and len(claim_text) > 10:
            claims.append((claim_text[:150], [], 'assumption'))

    # Pattern for GUIDELINE tags
    guideline_pattern = r'GUIDELINE[:\s]*\([^)]*\)[:\s]*([^.!?\n]+[.!?]?)'
    guideline_matches = re.finditer(guideline_pattern, text, re.IGNORECASE)
    for match in guideline_matches:
        claim_text = match.group(1).strip()
        if claim_text and len(claim_text) > 10:
            claims.append((claim_text[:150], [], 'guideline'))

    return claims


def build_claim_ledger(
    expert_responses: Dict[str, Dict],
    search_results: List,
    corpus_pmids: Optional[set] = None
) -> ClaimLedger:
    """
    Extract and validate all claims from expert responses.

    Args:
        expert_responses: Dict mapping expert name to response dict
        search_results: List of Citation objects from literature search
        corpus_pmids: Optional set of valid PMIDs (computed from search_results if not provided)

    Returns:
        ClaimLedger with all validated claims
    """
    claims = []

    # Build corpus PMID set
    if corpus_pmids is None:
        corpus_pmids = set()
        for cit in search_results:
            if hasattr(cit, 'pmid'):
                corpus_pmids.add(str(cit.pmid))
            elif isinstance(cit, dict):
                pmid = cit.get('pmid')
                if pmid:
                    corpus_pmids.add(str(pmid))

    for expert, response in expert_responses.items():
        content = response.get('content', '') if isinstance(response, dict) else str(response)

        # Extract claims from this expert's response
        extracted = extract_claims_with_pmids(content)

        for claim_text, cited_pmids, claim_type in extracted:
            # Skip very short claims
            if len(claim_text) < 15:
                continue

            # Determine status
            if claim_type in ('opinion', 'assumption'):
                # Opinions and assumptions don't need PMID validation
                status = "SUPPORTED"  # They're valid as labeled
                reason = f"Labeled as {claim_type}"
            elif claim_type == 'guideline':
                # Guidelines reference external sources
                status = "SUPPORTED"
                reason = "Guideline reference"
            elif cited_pmids:
                # Check if PMIDs are in corpus
                valid_pmids = [p for p in cited_pmids if p in corpus_pmids]
                invalid_pmids = [p for p in cited_pmids if p not in corpus_pmids]

                if valid_pmids and not invalid_pmids:
                    status = "SUPPORTED"
                    reason = ""
                elif invalid_pmids:
                    status = "UNCLEAR"
                    reason = f"PMID(s) {', '.join(invalid_pmids)} not in search results"
                else:
                    status = "UNCLEAR"
                    reason = "Citation not verified"
            else:
                # Factual claim without citation
                status = "UNCLEAR"
                reason = "No citation provided for factual claim"

            claims.append(ClaimLedgerEntry(
                claim=claim_text,
                status=status,
                citations=cited_pmids,
                reason=reason,
                source_expert=expert,
                claim_type=claim_type
            ))

    return ClaimLedger(claims=claims)


# =============================================================================
# COMPARATOR MATCH VALIDATION
# =============================================================================

@dataclass
class ComparatorMatchResult:
    """Result of validating whether evidence matches the question's comparators."""
    question_comparators: Tuple[str, str]  # (intervention, comparator) from question
    evidence_comparators: List[Tuple[str, str]]  # Actual comparisons found in papers
    match_found: bool  # True if evidence directly compares what's asked
    match_type: str  # "direct" | "indirect" | "no_comparison"
    confidence_adjustment: str  # "" | "HIGH→MODERATE" | "MODERATE→LOW"
    original_confidence: str
    adjusted_confidence: str
    warning_message: str
    matching_pmids: List[str]  # Papers with direct comparisons
    indirect_pmids: List[str]  # Papers with indirect/related comparisons

    def to_dict(self) -> Dict:
        return {
            'question_comparators': self.question_comparators,
            'evidence_comparators': self.evidence_comparators,
            'match_found': self.match_found,
            'match_type': self.match_type,
            'confidence_adjustment': self.confidence_adjustment,
            'original_confidence': self.original_confidence,
            'adjusted_confidence': self.adjusted_confidence,
            'warning_message': self.warning_message,
            'matching_pmids': self.matching_pmids,
            'indirect_pmids': self.indirect_pmids
        }


class ComparatorMatcher:
    """
    Validates whether retrieved evidence actually compares the interventions asked about.

    Example:
        Question: "prophylactic fixation vs observation"
        Papers: Compare "impending vs completed fracture" (NOT the same comparison)
        Result: Flag as indirect evidence, downgrade confidence
    """

    # Common intervention terms for orthopedic oncology
    INTERVENTION_SYNONYMS = {
        'prophylactic fixation': ['prophylactic fixation', 'prophylactic stabilization', 'preventive fixation', 'elective fixation'],
        'observation': ['observation', 'watchful waiting', 'conservative management', 'non-operative', 'nonsurgical'],
        'surgery': ['surgery', 'surgical', 'operative', 'operation', 'resection'],
        'radiation': ['radiation', 'radiotherapy', 'RT', 'XRT', 'irradiation'],
        'fixation': ['fixation', 'internal fixation', 'nailing', 'plating', 'stabilization'],
    }

    # Patterns indicating comparison in text
    COMPARISON_PATTERNS = [
        r'(\w+(?:\s+\w+)?)\s+(?:vs\.?|versus|compared\s+(?:to|with)|relative\s+to)\s+(\w+(?:\s+\w+)?)',
        r'comparing\s+(\w+(?:\s+\w+)?)\s+(?:and|to|with)\s+(\w+(?:\s+\w+)?)',
        r'(\w+(?:\s+\w+)?)\s+(?:group|arm|cohort)\s+(?:vs\.?|versus)\s+(\w+(?:\s+\w+)?)',
    ]

    def extract_question_comparators(self, question: str) -> Tuple[str, str]:
        """
        Extract the intervention and comparator from a clinical question.

        Args:
            question: Clinical question text

        Returns:
            Tuple of (intervention, comparator) or ("", "") if not found
        """
        import re
        question_lower = question.lower()

        # Pattern: "X vs Y" or "X versus Y"
        vs_match = re.search(
            r'(?:benefit\s+of\s+)?(\w+(?:\s+\w+){0,2})\s+(?:vs\.?|versus)\s+(\w+(?:\s+\w+){0,2})',
            question_lower
        )
        if vs_match:
            return (vs_match.group(1).strip(), vs_match.group(2).strip())

        # Pattern: "X compared to Y"
        compared_match = re.search(
            r'(\w+(?:\s+\w+){0,2})\s+compared\s+(?:to|with)\s+(\w+(?:\s+\w+){0,2})',
            question_lower
        )
        if compared_match:
            return (compared_match.group(1).strip(), compared_match.group(2).strip())

        # Pattern: "Should we do X or Y"
        or_match = re.search(
            r'should\s+(?:we\s+)?(?:do\s+)?(\w+(?:\s+\w+){0,2})\s+or\s+(\w+(?:\s+\w+){0,2})',
            question_lower
        )
        if or_match:
            return (or_match.group(1).strip(), or_match.group(2).strip())

        return ("", "")

    def extract_paper_comparators(self, citations: List) -> List[Tuple[str, str, str]]:
        """
        Extract comparisons made in each paper from title/abstract.

        Args:
            citations: List of Citation objects

        Returns:
            List of (intervention, comparator, pmid) tuples
        """
        import re
        comparisons = []

        for cit in citations:
            if hasattr(cit, 'title'):
                title = cit.title or ""
                abstract = cit.abstract or ""
                pmid = cit.pmid or ""
            elif isinstance(cit, dict):
                title = cit.get('title', '')
                abstract = cit.get('abstract', '')
                pmid = cit.get('pmid', '')
            else:
                continue

            text = f"{title} {abstract}".lower()

            # Search for comparison patterns
            for pattern in self.COMPARISON_PATTERNS:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match) >= 2:
                        comparisons.append((match[0].strip(), match[1].strip(), pmid))

        return comparisons

    def _normalize_term(self, term: str) -> str:
        """Normalize intervention term for comparison."""
        term_lower = term.lower().strip()

        # Check synonyms
        for canonical, synonyms in self.INTERVENTION_SYNONYMS.items():
            if any(syn in term_lower for syn in synonyms):
                return canonical

        return term_lower

    def _terms_match(self, term1: str, term2: str) -> bool:
        """Check if two intervention terms refer to the same thing."""
        norm1 = self._normalize_term(term1)
        norm2 = self._normalize_term(term2)

        if norm1 == norm2:
            return True

        # Check if one is substring of other (for partial matches)
        if norm1 in norm2 or norm2 in norm1:
            return True

        # Check synonym overlap
        for canonical, synonyms in self.INTERVENTION_SYNONYMS.items():
            if norm1 == canonical or norm2 == canonical:
                if any(syn in norm1 for syn in synonyms) or any(syn in norm2 for syn in synonyms):
                    return True

        return False

    def validate_comparator_match(
        self,
        question: str,
        citations: List,
        current_confidence: str = "HIGH"
    ) -> ComparatorMatchResult:
        """
        Validate whether retrieved papers compare the interventions asked about.

        Args:
            question: Clinical question being asked
            citations: Retrieved citation list
            current_confidence: Current confidence level before adjustment

        Returns:
            ComparatorMatchResult with match status and confidence adjustment
        """
        # Extract comparators from question
        q_intervention, q_comparator = self.extract_question_comparators(question)

        if not q_intervention or not q_comparator:
            # Can't determine question comparators - no adjustment
            return ComparatorMatchResult(
                question_comparators=("", ""),
                evidence_comparators=[],
                match_found=True,  # Assume OK if we can't parse
                match_type="unknown",
                confidence_adjustment="",
                original_confidence=current_confidence,
                adjusted_confidence=current_confidence,
                warning_message="",
                matching_pmids=[],
                indirect_pmids=[]
            )

        # Extract comparators from papers
        paper_comparisons = self.extract_paper_comparators(citations)
        evidence_comparators = [(c[0], c[1]) for c in paper_comparisons]

        # Check for direct matches
        matching_pmids = []
        indirect_pmids = []

        for p_int, p_comp, pmid in paper_comparisons:
            # Check if paper compares the same things as question
            int_match = self._terms_match(q_intervention, p_int) or self._terms_match(q_intervention, p_comp)
            comp_match = self._terms_match(q_comparator, p_int) or self._terms_match(q_comparator, p_comp)

            if int_match and comp_match:
                matching_pmids.append(pmid)
            elif int_match or comp_match:
                indirect_pmids.append(pmid)

        # Determine match type and confidence adjustment
        if matching_pmids:
            match_type = "direct"
            match_found = True
            confidence_adjustment = ""
            adjusted_confidence = current_confidence
            warning_message = ""
        elif indirect_pmids:
            match_type = "indirect"
            match_found = False
            # Downgrade confidence
            confidence_map = {
                "HIGH": "MODERATE",
                "MODERATE": "LOW",
                "LOW": "VERY LOW",
                "VERY LOW": "VERY LOW"
            }
            adjusted_confidence = confidence_map.get(current_confidence, current_confidence)
            if adjusted_confidence != current_confidence:
                confidence_adjustment = f"{current_confidence}→{adjusted_confidence}"
            else:
                confidence_adjustment = ""
            warning_message = (
                f"Evidence is INDIRECT: Question asks about '{q_intervention}' vs '{q_comparator}', "
                f"but retrieved papers compare related but different interventions. "
                f"Confidence adjusted from {current_confidence} to {adjusted_confidence}."
            )
        else:
            match_type = "no_comparison"
            match_found = False
            adjusted_confidence = "LOW" if current_confidence in ["HIGH", "MODERATE"] else current_confidence
            confidence_adjustment = f"{current_confidence}→{adjusted_confidence}" if adjusted_confidence != current_confidence else ""
            warning_message = (
                f"No comparative evidence found: Question asks about '{q_intervention}' vs '{q_comparator}', "
                f"but no papers directly compare these interventions."
            )

        return ComparatorMatchResult(
            question_comparators=(q_intervention, q_comparator),
            evidence_comparators=list(set(evidence_comparators)),
            match_found=match_found,
            match_type=match_type,
            confidence_adjustment=confidence_adjustment,
            original_confidence=current_confidence,
            adjusted_confidence=adjusted_confidence,
            warning_message=warning_message,
            matching_pmids=list(set(matching_pmids)),
            indirect_pmids=list(set(indirect_pmids))
        )


def validate_comparator_match(
    question: str,
    citations: List,
    current_confidence: str = "HIGH"
) -> ComparatorMatchResult:
    """
    Convenience function to validate comparator match.

    Args:
        question: Clinical question
        citations: List of citations
        current_confidence: Current confidence level

    Returns:
        ComparatorMatchResult
    """
    matcher = ComparatorMatcher()
    return matcher.validate_comparator_match(question, citations, current_confidence)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_expert_response(
    response: str,
    search_result: TwoChannelSearchResult,
    api_key: str = None
) -> ValidationResult:
    """
    Validate an expert response against two-channel evidence.

    Args:
        response: Expert's response text
        search_result: TwoChannelSearchResult from search
        api_key: Optional API key for LLM extraction

    Returns:
        ValidationResult with claim validation details
    """
    validator = ClaimValidator(api_key=api_key)
    return validator.validate_claims(response, search_result)


def get_evidence_for_claim_type(
    claim_type: ClaimType,
    search_result: TwoChannelSearchResult
) -> List[Citation]:
    """Get appropriate evidence channel for a claim type."""
    if claim_type in [ClaimType.BIOLOGY, ClaimType.SAFETY]:
        return search_result.biology.citations
    else:
        return search_result.clinical.citations


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING CLAIM VALIDATOR")
    print("=" * 60)

    # Mock expert response
    test_response = """
    FOLR1 is expressed in approximately 40% of NSCLC tumors based on IHC studies.
    The standard of care for 2L+ NSCLC is docetaxel or immunotherapy combinations.
    Phase 2 trials of FOLR1-ADCs have shown response rates around 20-30%.
    IL-2 based therapies carry significant risk of cytokine release syndrome.
    I recommend starting with dose escalation to establish MTD.
    """

    validator = ClaimValidator()

    # Extract claims
    claims = validator.extract_claims(test_response)
    print(f"\nExtracted {len(claims)} claims:")
    for claim, ctype in claims:
        print(f"  [{ctype.value}] {claim[:60]}...")

    # Note: Full validation requires search_result
    print("\n(Full validation requires TwoChannelSearchResult)")
