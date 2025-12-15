"""
Tiered Search Service with Fallback Strategies

Executes searches in order of strictness:
1. Tier 1 (strict): All component queries with filters
2. Tier 2 (relaxed): Same queries without clinical trial filters
3. Tier 3 (broad): Very broad single-concept queries

Stops when sufficient results are found.
"""

import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from config import settings
from core.pubmed_client import PubMedClient, Citation
from core.query_extractor import ClinicalQueryExtractor, ExtractedConcepts, get_synonyms
from core.query_builder import ComponentQueryBuilder, QuerySet

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Results from tiered search."""
    citations: List[Citation] = field(default_factory=list)
    trials: List[Dict] = field(default_factory=list)
    concepts: ExtractedConcepts = None
    queries_executed: Dict[str, List[str]] = field(default_factory=dict)
    tier_used: str = ""
    total_before_filter: int = 0
    filtered_count: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "citations": [c.__dict__ if hasattr(c, '__dict__') else c for c in self.citations],
            "trials": self.trials,
            "concepts": self.concepts.to_dict() if self.concepts else {},
            "queries_executed": self.queries_executed,
            "tier_used": self.tier_used,
            "total_before_filter": self.total_before_filter,
            "filtered_count": self.filtered_count,
        }


class TieredSearchService:
    """
    Execute tiered searches with automatic fallback.

    Strategy:
    1. Try all Tier 1 (strict) queries in parallel
    2. If < MIN_RESULTS, add Tier 2 (relaxed) queries
    3. If still < MIN_RESULTS, add Tier 3 (broad) queries
    4. Score all results by concept overlap
    5. Light filter to remove clearly off-topic papers
    """

    MIN_RESULTS = 5
    MAX_RESULTS_PER_QUERY = 15
    MAX_TOTAL_RESULTS = 50

    def __init__(self, api_key: str = None):
        """
        Initialize tiered search service.

        Args:
            api_key: API key for LLM-based extraction
        """
        self.api_key = api_key
        self.extractor = ClinicalQueryExtractor(api_key=api_key)
        self.query_builder = ComponentQueryBuilder()
        self.pubmed = PubMedClient(
            email=getattr(settings, 'PUBMED_EMAIL', 'user@example.com'),
            api_key=getattr(settings, 'PUBMED_API_KEY', None)
        )

    def search(self, question: str, program_profile: Dict = None) -> SearchResult:
        """
        Execute tiered search for a clinical question.

        Args:
            question: User's clinical question
            program_profile: Optional program context for additional anchoring

        Returns:
            SearchResult with citations, trials, and metadata
        """
        result = SearchResult()

        # 1. Extract concepts
        concepts = self.extractor.extract(question)
        result.concepts = concepts

        if concepts.is_empty():
            logger.warning(f"No concepts extracted from question: {question[:100]}...")
            return result

        logger.info(f"Extracted concepts: {concepts.to_dict()}")

        # 2. Build query tiers
        query_set = self.query_builder.build_search_queries(concepts)

        # 3. Execute tiered search
        all_citations = []
        seen_pmids = set()

        # Tier 1: Strict queries
        tier1_results = self._execute_queries(query_set.tier1_strict, "tier1")
        result.queries_executed["tier1"] = query_set.tier1_strict

        for cit in tier1_results:
            if cit.pmid not in seen_pmids:
                seen_pmids.add(cit.pmid)
                all_citations.append(cit)

        logger.info(f"Tier 1 results: {len(all_citations)} unique citations")

        # Tier 2: Relaxed queries (if needed)
        if len(all_citations) < self.MIN_RESULTS:
            result.tier_used = "tier2"
            tier2_results = self._execute_queries(query_set.tier2_relaxed, "tier2")
            result.queries_executed["tier2"] = query_set.tier2_relaxed

            for cit in tier2_results:
                if cit.pmid not in seen_pmids:
                    seen_pmids.add(cit.pmid)
                    all_citations.append(cit)

            logger.info(f"After Tier 2: {len(all_citations)} unique citations")
        else:
            result.tier_used = "tier1"

        # Tier 3: Broad queries (if still needed)
        if len(all_citations) < self.MIN_RESULTS:
            result.tier_used = "tier3"
            tier3_results = self._execute_queries(query_set.tier3_broad, "tier3")
            result.queries_executed["tier3"] = query_set.tier3_broad

            for cit in tier3_results:
                if cit.pmid not in seen_pmids:
                    seen_pmids.add(cit.pmid)
                    all_citations.append(cit)

            logger.info(f"After Tier 3: {len(all_citations)} unique citations")

        result.total_before_filter = len(all_citations)

        # 4. Score by relevance
        scored = [(self._score_relevance(cit, concepts), cit) for cit in all_citations]
        scored.sort(reverse=True, key=lambda x: x[0])

        # 5. Light filter (not aggressive!)
        filtered = self._light_filter(scored, concepts)

        result.citations = [cit for _, cit in filtered[:self.MAX_TOTAL_RESULTS]]
        result.filtered_count = len(all_citations) - len(result.citations)

        # 6. Also search clinical trials
        if concepts.targets or concepts.indications:
            result.trials = self._search_trials(concepts)

        logger.info(f"Final: {len(result.citations)} citations, {len(result.trials)} trials")
        return result

    def _execute_queries(self, queries: List[str], tier_name: str) -> List[Citation]:
        """Execute multiple queries in parallel."""
        all_citations = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._search_pubmed, q): q
                for q in queries[:6]  # Limit to 6 queries per tier
            }

            for future in as_completed(futures):
                query = futures[future]
                try:
                    citations = future.result()
                    all_citations.extend(citations)
                    logger.debug(f"{tier_name} query returned {len(citations)} results")
                except Exception as e:
                    logger.warning(f"{tier_name} query failed: {e}")

        return all_citations

    def _search_pubmed(self, query: str) -> List[Citation]:
        """Execute single PubMed search."""
        try:
            search_result = self.pubmed.search(query, max_results=self.MAX_RESULTS_PER_QUERY)
            pmids = search_result.get("pmids", [])

            if pmids:
                citations, _ = self.pubmed.fetch_citations(pmids)
                return citations
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")

        return []

    def _search_trials(self, concepts: ExtractedConcepts, max_results: int = 10) -> List[Dict]:
        """Search ClinicalTrials.gov."""
        parts = []
        if concepts.targets:
            parts.append(concepts.targets[0])
        if concepts.indications:
            parts.append(concepts.indications[0])

        if not parts:
            return []

        try:
            response = requests.get(
                "https://clinicaltrials.gov/api/v2/studies",
                params={
                    "query.term": " AND ".join(parts),
                    "filter.advanced": "AREA[Phase](PHASE2 OR PHASE3)",
                    "pageSize": max_results,
                    "format": "json"
                },
                timeout=30
            )
            response.raise_for_status()

            studies = response.json().get("studies", [])
            return [{
                "nct_id": s.get("protocolSection", {}).get("identificationModule", {}).get("nctId", ""),
                "title": s.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle", ""),
                "phase": s.get("protocolSection", {}).get("designModule", {}).get("phases", []),
                "status": s.get("protocolSection", {}).get("statusModule", {}).get("overallStatus", ""),
                "sponsor": s.get("protocolSection", {}).get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""),
                "url": f"https://clinicaltrials.gov/study/{s.get('protocolSection', {}).get('identificationModule', {}).get('nctId', '')}"
            } for s in studies]

        except Exception as e:
            logger.error(f"ClinicalTrials search failed: {e}")
            return []

    def _score_relevance(self, citation: Citation, concepts: ExtractedConcepts) -> float:
        """Score citation by concept overlap."""
        text = f"{citation.title} {citation.abstract}".lower()
        score = 0.0

        # Target mentions (highest weight)
        for target in concepts.targets:
            for variant in [target.lower()] + [s.lower() for s in get_synonyms(target, "target")]:
                if variant in text:
                    score += 3.0
                    break

        # Indication mentions
        for ind in concepts.indications:
            for variant in [ind.lower()] + [s.lower() for s in get_synonyms(ind, "indication")]:
                if variant in text:
                    score += 2.0
                    break

        # Mechanism mentions
        for mech in concepts.mechanisms:
            if mech.lower() in text:
                score += 1.5

        # Modality mentions
        for mod in concepts.modalities:
            for variant in [mod.lower()] + [s.lower() for s in get_synonyms(mod, "modality")]:
                if variant in text:
                    score += 1.0
                    break

        # Clinical relevance bonus
        if any(t in text for t in ["phase 2", "phase 3", "phase ii", "phase iii", "clinical trial"]):
            score += 1.0

        return score

    def _light_filter(
        self,
        scored: List[Tuple[float, Citation]],
        concepts: ExtractedConcepts
    ) -> List[Tuple[float, Citation]]:
        """
        Light filtering - require papers mention at least one target.

        Key insight: A paper about "CLEC5a in schizophrenia" is irrelevant for
        a FOLR1-CLEC5a-IFNa macrophage engager query UNLESS it also mentions
        FOLR1 or IFNa. Papers must mention at least one of our targets.
        """
        if not scored:
            return []

        # Build target anchors (must match at least one)
        target_anchors = set()
        for t in concepts.targets:
            target_anchors.add(t.lower())
            # Add synonyms for targets
            for syn in get_synonyms(t, "target"):
                target_anchors.add(syn.lower())

        # Also include modalities as valid anchors (e.g., IFNα papers are relevant)
        for mod in concepts.modalities:
            target_anchors.add(mod.lower())
            for syn in get_synonyms(mod, "modality"):
                target_anchors.add(syn.lower())

        # Include indication as valid anchor
        for ind in concepts.indications:
            target_anchors.add(ind.lower())
            for syn in get_synonyms(ind, "indication"):
                target_anchors.add(syn.lower())

        # Keep papers that mention at least one target/modality/indication
        filtered = []
        for score, cit in scored:
            text = f"{cit.title} {cit.abstract}".lower()
            if any(anchor in text for anchor in target_anchors):
                filtered.append((score, cit))

        # CRITICAL: If filtering removed too many, return originals
        # This prevents the "zero results" problem
        if len(filtered) < self.MIN_RESULTS and len(scored) >= self.MIN_RESULTS:
            logger.warning(f"Filter too aggressive ({len(filtered)}/{len(scored)}), returning unfiltered")
            return scored

        return filtered


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def tiered_search(question: str, api_key: str = None) -> SearchResult:
    """
    Execute tiered search for a clinical question.

    Args:
        question: User's clinical question
        api_key: Optional API key for LLM extraction

    Returns:
        SearchResult with citations, concepts, and metadata
    """
    service = TieredSearchService(api_key=api_key)
    return service.search(question)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import os

    print("=" * 60)
    print("TESTING TIERED SEARCH SERVICE")
    print("=" * 60)

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
    service = TieredSearchService(api_key=api_key)

    question = "Make me a CDP for FOLR1-targeting IFNa-carrying CLEC5a macrophage engager for 2L+ NSCLC"

    print(f"\nSearching: {question[:60]}...")
    result = service.search(question)

    print(f"\nResults:")
    print(f"  Tier used: {result.tier_used}")
    print(f"  Total before filter: {result.total_before_filter}")
    print(f"  Filtered out: {result.filtered_count}")
    print(f"  Final citations: {len(result.citations)}")
    print(f"  Trials found: {len(result.trials)}")

    if result.concepts:
        print(f"\nExtracted concepts:")
        print(f"  Targets: {result.concepts.targets}")
        print(f"  Indications: {result.concepts.indications}")
        print(f"  Mechanisms: {result.concepts.mechanisms}")
        print(f"  Modalities: {result.concepts.modalities}")

    if result.citations:
        print("\nTop 5 citations:")
        for i, cit in enumerate(result.citations[:5]):
            print(f"  {i+1}. {cit.title[:70]}...")

    if result.trials:
        print("\nTrials:")
        for trial in result.trials[:3]:
            print(f"  - {trial['nct_id']}: {trial['title'][:50]}...")

    print("\nQueries executed:")
    for tier, queries in result.queries_executed.items():
        print(f"  {tier}: {len(queries)} queries")
