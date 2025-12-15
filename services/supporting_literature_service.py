"""
Supporting Literature Service

Searches for case series and cohorts that SUPPORT expert claims.
This is NOT validation - it's finding illustrative examples.

Key insight: In low-evidence domains like palliative surgery,
we don't validate expert claims against RCTs (they don't exist).
Instead, we find case series that illustrate the clinical points.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import settings
from core.pubmed_client import PubMedClient, Citation
from core.claims.claim_extractor import ExtractedClaim

logger = logging.getLogger(__name__)


@dataclass
class SupportingPaper:
    """A paper that supports an expert claim."""
    pmid: str
    title: str
    abstract: str = ""
    study_type: str = "unknown"  # case_series, cohort, retrospective, review, RCT
    sample_size: Optional[int] = None
    relevance: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    matched_claim: str = ""
    year: Optional[int] = None
    journal: str = ""

    def to_dict(self) -> Dict:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract[:200] + "..." if len(self.abstract) > 200 else self.abstract,
            "study_type": self.study_type,
            "sample_size": self.sample_size,
            "relevance": self.relevance,
            "matched_claim": self.matched_claim,
            "year": self.year,
            "journal": self.journal
        }


@dataclass
class LiteratureSearchResult:
    """Result of searching supporting literature for all claims."""
    claim_matches: Dict[str, List[SupportingPaper]] = field(default_factory=dict)
    total_papers: int = 0
    claims_with_support: int = 0
    claims_without_support: int = 0
    study_types: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "claim_matches": {
                claim: [p.to_dict() for p in papers]
                for claim, papers in self.claim_matches.items()
            },
            "total_papers": self.total_papers,
            "claims_with_support": self.claims_with_support,
            "claims_without_support": self.claims_without_support,
            "study_types": self.study_types
        }

    def get_all_papers(self) -> List[SupportingPaper]:
        """Get all unique papers across all claims."""
        seen = set()
        papers = []
        for claim_papers in self.claim_matches.values():
            for paper in claim_papers:
                if paper.pmid not in seen:
                    seen.add(paper.pmid)
                    papers.append(paper)
        return papers


class SupportingLiteratureService:
    """
    Service to find supporting literature for expert claims.

    Uses expert-provided search hints to find relevant case series,
    cohorts, and retrospective studies that illustrate clinical points.
    """

    MAX_PAPERS_PER_CLAIM = 3
    MAX_SEARCH_RESULTS = 15

    def __init__(self, pubmed_email: str = None, pubmed_api_key: str = None):
        """Initialize the service."""
        self.pubmed = PubMedClient(
            email=pubmed_email or getattr(settings, 'PUBMED_EMAIL', 'user@example.com'),
            api_key=pubmed_api_key or getattr(settings, 'PUBMED_API_KEY', None)
        )

    def search_for_claims(
        self,
        claims: List[ExtractedClaim],
        max_per_claim: int = None
    ) -> LiteratureSearchResult:
        """
        Search for supporting literature for a list of claims.

        Args:
            claims: List of ExtractedClaim objects with search hints
            max_per_claim: Maximum papers per claim (default: 3)

        Returns:
            LiteratureSearchResult with matched papers
        """
        max_per_claim = max_per_claim or self.MAX_PAPERS_PER_CLAIM
        result = LiteratureSearchResult()

        # Filter to claims with search hints
        searchable_claims = [c for c in claims if c.search_hint]

        if not searchable_claims:
            logger.warning("No claims with search hints to search")
            return result

        logger.info(f"Searching supporting literature for {len(searchable_claims)} claims")

        # Search claims in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._search_single_claim, claim, max_per_claim): claim
                for claim in searchable_claims
            }

            for future in as_completed(futures):
                claim = futures[future]
                try:
                    papers = future.result()
                    result.claim_matches[claim.text] = papers

                    if papers:
                        result.claims_with_support += 1
                        result.total_papers += len(papers)

                        # Track study types
                        for paper in papers:
                            study_type = paper.study_type
                            result.study_types[study_type] = result.study_types.get(study_type, 0) + 1
                    else:
                        result.claims_without_support += 1

                except Exception as e:
                    logger.error(f"Search failed for claim: {claim.text[:50]}... Error: {e}")
                    result.claim_matches[claim.text] = []
                    result.claims_without_support += 1

        logger.info(
            f"Found {result.total_papers} papers for {result.claims_with_support} claims. "
            f"{result.claims_without_support} claims without support."
        )

        return result

    def _search_single_claim(
        self,
        claim: ExtractedClaim,
        max_results: int
    ) -> List[SupportingPaper]:
        """
        Search for papers supporting a single claim.

        Builds a query that prioritizes case series and cohorts
        (the evidence that actually exists in palliative surgery).
        """
        search_hint = claim.search_hint

        # Build query accepting low-evidence study types
        # This is key: we're NOT looking for RCTs, we're looking for
        # the case series and cohorts that actually exist
        query = self._build_supporting_query(search_hint)

        try:
            search_result = self.pubmed.search(query, max_results=self.MAX_SEARCH_RESULTS)
            pmids = search_result.get("pmids", [])

            if not pmids:
                logger.debug(f"No results for query: {query[:100]}...")
                return []

            citations, _ = self.pubmed.fetch_citations(pmids)

            # Filter for on-topic papers
            on_topic_papers = []
            for cit in citations:
                relevance = self._assess_relevance(cit, claim)
                if relevance != "LOW":
                    paper = SupportingPaper(
                        pmid=cit.pmid,
                        title=cit.title,
                        abstract=cit.abstract or "",
                        study_type=self._classify_study_type(cit),
                        sample_size=self._extract_sample_size(cit.abstract),
                        relevance=relevance,
                        matched_claim=claim.text,
                        year=getattr(cit, 'year', None),
                        journal=getattr(cit, 'journal', '')
                    )
                    on_topic_papers.append(paper)

            # Sort by relevance and return top N
            on_topic_papers.sort(
                key=lambda p: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(p.relevance, 2)
            )

            return on_topic_papers[:max_results]

        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []

    def _build_supporting_query(self, search_hint: str) -> str:
        """
        Build a PubMed query from search hint.

        Prioritizes study types that actually exist in palliative surgery:
        - Case series
        - Retrospective studies
        - Cohort studies
        - Clinical experience reports
        """
        # Clean the search hint
        hint_terms = search_hint.strip()

        # Build query with study type filters
        query = f"""
        ({hint_terms}) AND
        (case series[tiab] OR retrospective[tiab] OR cohort[tiab] OR
         outcomes[tiab] OR experience[tiab] OR patients[tiab])
        NOT (pediatric[tiab] OR animal[tiab] OR in vitro[tiab])
        """

        return query.strip()

    def _assess_relevance(self, citation: Citation, claim: ExtractedClaim) -> str:
        """
        Assess how relevant a paper is to the claim.

        Returns: HIGH, MEDIUM, or LOW
        """
        if not claim.search_hint:
            return "LOW"

        text = f"{citation.title} {citation.abstract or ''}".lower()

        # Extract key terms from search hint
        hint_terms = claim.search_hint.lower().split()
        key_terms = [t for t in hint_terms if len(t) >= 4]  # Skip short words

        if not key_terms:
            return "MEDIUM"

        # Count matching terms
        matches = sum(1 for term in key_terms if term in text)
        match_ratio = matches / len(key_terms)

        if match_ratio >= 0.6:
            return "HIGH"
        elif match_ratio >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"

    def _classify_study_type(self, citation: Citation) -> str:
        """
        Classify the study design from title/abstract.

        Returns: RCT, systematic_review, prospective_cohort, retrospective, case_series, unknown
        """
        abstract = (citation.abstract or "").lower()
        title = citation.title.lower()
        text = f"{title} {abstract}"

        if "randomized" in text or "randomised" in text or " rct " in text:
            return "RCT"
        elif "systematic review" in text or "meta-analysis" in text:
            return "systematic_review"
        elif "prospective" in text and "cohort" in text:
            return "prospective_cohort"
        elif "retrospective" in text or "chart review" in text:
            return "retrospective"
        elif "case series" in text:
            return "case_series"
        elif re.search(r'\b\d+\s*patients\b', text):
            # Has patient count - likely case series
            return "case_series"
        else:
            return "unknown"

    def _extract_sample_size(self, abstract: str) -> Optional[int]:
        """
        Extract sample size from abstract.

        Looks for patterns like "47 patients", "n=31", etc.
        """
        if not abstract:
            return None

        abstract_lower = abstract.lower()

        # Pattern: N patients/cases
        match = re.search(r'\b(\d+)\s*(?:patients|cases|subjects)\b', abstract_lower)
        if match:
            return int(match.group(1))

        # Pattern: n=N
        match = re.search(r'\bn\s*=\s*(\d+)\b', abstract_lower)
        if match:
            return int(match.group(1))

        # Pattern: "enrolled N"
        match = re.search(r'\benrolled\s*(\d+)\b', abstract_lower)
        if match:
            return int(match.group(1))

        return None


def search_supporting_literature(
    claims: List[ExtractedClaim],
    max_per_claim: int = 3
) -> LiteratureSearchResult:
    """
    Convenience function to search supporting literature.

    Args:
        claims: List of ExtractedClaim objects
        max_per_claim: Maximum papers per claim

    Returns:
        LiteratureSearchResult with matched papers
    """
    service = SupportingLiteratureService()
    return service.search_for_claims(claims, max_per_claim)


def determine_certainty(search_result: LiteratureSearchResult) -> str:
    """
    Determine GRADE-style certainty from study types found.

    Args:
        search_result: Result from literature search

    Returns:
        Certainty string: "Moderate", "Low", "Very Low (case series/expert consensus)"
    """
    study_types = search_result.study_types

    if study_types.get("RCT", 0) >= 1 or study_types.get("systematic_review", 0) >= 1:
        return "Moderate"
    elif study_types.get("prospective_cohort", 0) >= 1:
        return "Low"
    elif search_result.total_papers > 0:
        return "Very Low (case series/expert consensus)"
    else:
        return "Very Low (expert consensus only)"
