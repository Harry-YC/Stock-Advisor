"""
Comprehensive Search Service

Multi-source search using extracted concepts.
Runs parallel queries to PubMed and ClinicalTrials.gov,
merges results, and filters irrelevant hits.

Replaces the simple single-query approach that produced
results like diabetes papers for NSCLC questions.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from config import settings
from core.pubmed_client import PubMedClient, Citation
from core.query_extractor import ClinicalQueryExtractor, ExtractedConcepts

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Combined search results from multiple sources."""
    concepts: ExtractedConcepts
    citations: List[Citation] = field(default_factory=list)
    trials: List[Dict] = field(default_factory=list)
    queries_used: Dict[str, str] = field(default_factory=dict)
    filtered_count: int = 0
    total_searched: int = 0


class ComprehensiveSearchService:
    """
    Search multiple sources based on extracted scientific concepts.
    Filters irrelevant results based on program context.

    Flow:
    1. Extract concepts from question (targets, indications, mechanisms)
    2. Build multiple targeted queries
    3. Execute in parallel (PubMed, ClinicalTrials.gov)
    4. Filter results that don't mention key concepts
    5. Return merged, deduplicated results
    """

    def __init__(self, api_key: str = None):
        """
        Initialize search service.

        Args:
            api_key: API key for LLM-based concept extraction
        """
        self.api_key = api_key or getattr(settings, 'OPENAI_API_KEY', None) or getattr(settings, 'GOOGLE_API_KEY', None)
        self.extractor = ClinicalQueryExtractor(api_key=self.api_key)
        self.pubmed = PubMedClient(
            email=getattr(settings, 'PUBMED_EMAIL', 'user@example.com'),
            api_key=getattr(settings, 'PUBMED_API_KEY', None)
        )

    def search(
        self,
        question: str,
        max_per_query: int = 15,
        program_profile: Dict = None,
        include_trials: bool = True
    ) -> SearchResult:
        """
        Execute comprehensive search with concept extraction and filtering.

        Args:
            question: Clinical question to search
            max_per_query: Max results per query
            program_profile: Optional program profile for additional filtering
            include_trials: Whether to search ClinicalTrials.gov

        Returns:
            SearchResult with citations, trials, and metadata
        """
        # 1. Extract concepts
        concepts = self.extractor.extract(question)
        logger.info(
            f"Extracted concepts: targets={concepts.targets}, "
            f"indications={concepts.indications}, mechanisms={concepts.mechanisms}"
        )

        result = SearchResult(concepts=concepts)

        if concepts.is_empty():
            logger.warning(f"No concepts extracted from: {question[:100]}...")
            return result

        # 2. Build queries
        pubmed_queries = concepts.to_pubmed_queries()
        trials_query = concepts.to_trials_query()

        logger.info(f"Generated {len(pubmed_queries)} PubMed queries")
        if trials_query:
            logger.info(f"Trials query: {trials_query}")

        # 3. Execute in parallel
        all_citations = []
        seen_pmids = set()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}

            # Submit PubMed queries
            for i, query in enumerate(pubmed_queries[:4]):  # Max 4 queries
                result.queries_used[f"pubmed_{i}"] = query
                future = executor.submit(self._search_pubmed, query, max_per_query)
                futures[future] = f"pubmed_{i}"

            # Submit ClinicalTrials.gov query
            if include_trials and trials_query:
                result.queries_used["trials"] = trials_query
                future = executor.submit(self._search_trials, trials_query, max_per_query)
                futures[future] = "trials"

            # Collect results
            for future in as_completed(futures):
                source = futures[future]
                try:
                    items = future.result()
                    if isinstance(items, list) and items:
                        # Check if it's citations or trials
                        if items and hasattr(items[0], 'pmid'):
                            for cit in items:
                                if cit.pmid not in seen_pmids:
                                    seen_pmids.add(cit.pmid)
                                    all_citations.append(cit)
                            logger.info(f"{source}: found {len(items)} papers")
                        else:
                            result.trials.extend(items)
                            logger.info(f"{source}: found {len(items)} trials")
                except Exception as e:
                    logger.error(f"Search {source} failed: {e}")

        result.total_searched = len(all_citations)

        # 4. Filter irrelevant results
        original_count = len(all_citations)
        result.citations = self._filter_relevant(
            all_citations, concepts, program_profile
        )
        result.filtered_count = original_count - len(result.citations)

        logger.info(
            f"Search complete: {len(result.citations)} citations "
            f"(filtered {result.filtered_count}), {len(result.trials)} trials"
        )

        return result

    def _search_pubmed(self, query: str, max_results: int) -> List[Citation]:
        """Execute single PubMed search."""
        try:
            # First try with clinical relevance filter
            clinical_query = (
                f'({query}) AND '
                f'(clinical[tiab] OR trial[tiab] OR patient[tiab] OR '
                f'phase[tiab] OR efficacy[tiab] OR safety[tiab])'
            )

            search_result = self.pubmed.search(clinical_query, max_results=max_results)
            pmids = search_result.get("pmids", [])

            # Fallback to unfiltered if no results
            if not pmids:
                logger.debug(f"No results with clinical filter, trying raw query")
                search_result = self.pubmed.search(query, max_results=max_results)
                pmids = search_result.get("pmids", [])

            if pmids:
                citations, _ = self.pubmed.fetch_citations(pmids)
                return citations

        except Exception as e:
            logger.error(f"PubMed search failed: {e}")

        return []

    def _search_trials(self, query: str, max_results: int) -> List[Dict]:
        """Execute ClinicalTrials.gov search."""
        import requests

        try:
            response = requests.get(
                "https://clinicaltrials.gov/api/v2/studies",
                params={
                    "query.term": query,
                    "filter.advanced": "phase:PHASE2,PHASE3",
                    "pageSize": max_results,
                    "format": "json"
                },
                timeout=30
            )
            response.raise_for_status()

            studies = response.json().get("studies", [])
            return [{
                "nct_id": s.get("protocolSection", {}).get(
                    "identificationModule", {}
                ).get("nctId", ""),
                "title": s.get("protocolSection", {}).get(
                    "identificationModule", {}
                ).get("briefTitle", ""),
                "phase": s.get("protocolSection", {}).get(
                    "designModule", {}
                ).get("phases", []),
                "status": s.get("protocolSection", {}).get(
                    "statusModule", {}
                ).get("overallStatus", ""),
                "conditions": s.get("protocolSection", {}).get(
                    "conditionsModule", {}
                ).get("conditions", []),
                "interventions": [
                    i.get("name", "") for i in
                    s.get("protocolSection", {}).get(
                        "armsInterventionsModule", {}
                    ).get("interventions", [])
                ]
            } for s in studies]

        except Exception as e:
            logger.error(f"ClinicalTrials search failed: {e}")

        return []

    def _filter_relevant(
        self,
        citations: List[Citation],
        concepts: ExtractedConcepts,
        program_profile: Dict = None
    ) -> List[Citation]:
        """
        Filter out obviously irrelevant citations.
        A citation must mention at least one target/indication/mechanism.
        """
        # Build anchor terms for relevance checking
        anchors = set()

        for t in concepts.targets:
            anchors.add(t.lower())
        for i in concepts.indications:
            anchors.add(i.lower())
            # Add common variants
            i_lower = i.lower()
            if i_lower == "nsclc":
                anchors.update([
                    "non-small cell", "lung cancer",
                    "lung adenocarcinoma", "lung squamous"
                ])
            elif i_lower == "aml":
                anchors.update(["acute myeloid leukemia", "leukemia"])
            elif i_lower == "tnbc":
                anchors.update([
                    "triple negative", "triple-negative", "breast cancer"
                ])
        for m in concepts.mechanisms:
            anchors.add(m.lower())
        for mod in concepts.modalities:
            anchors.add(mod.lower())

        # Add from program profile if available
        if program_profile:
            if program_profile.get("target"):
                anchors.add(program_profile["target"].lower())
            if program_profile.get("indication"):
                anchors.add(program_profile["indication"].lower())
            for drug in program_profile.get("drug_names", []):
                if drug:
                    anchors.add(drug.lower())

        if not anchors:
            return citations  # No filtering possible

        # Filter citations
        filtered = []
        for cit in citations:
            text = f"{cit.title} {cit.abstract or ''}".lower()
            if any(anchor in text for anchor in anchors):
                filtered.append(cit)

        # If too aggressive (filtered too much), return more
        if len(filtered) < 5 and len(citations) > 5:
            logger.warning(
                f"Aggressive filtering ({len(filtered)}/{len(citations)}), "
                f"returning top 20 unfiltered"
            )
            return citations[:20]

        return filtered


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def comprehensive_search(
    question: str,
    api_key: str = None,
    max_per_query: int = 15
) -> SearchResult:
    """
    Convenience function for comprehensive search.

    Args:
        question: Clinical question
        api_key: API key for LLM extraction
        max_per_query: Max results per query

    Returns:
        SearchResult with citations and trials
    """
    service = ComprehensiveSearchService(api_key=api_key)
    return service.search(question, max_per_query=max_per_query)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import os

    test_questions = [
        "Make a CDP for 2L+ NSCLC with FOLR1-targeting IFNa-carrying CLEC5a engager",
        "What is the competitive landscape for KRAS G12C inhibitors in lung cancer?",
        "Go/no-go decision for CD47 macrophage checkpoint in AML",
    ]

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
    service = ComprehensiveSearchService(api_key=api_key)

    for q in test_questions:
        print(f"\n{'='*70}")
        print(f"Question: {q}")
        print("-" * 70)

        result = service.search(q, max_per_query=10, include_trials=False)

        print(f"\nConcepts extracted:")
        print(f"  Targets: {result.concepts.targets}")
        print(f"  Indications: {result.concepts.indications}")
        print(f"  Mechanisms: {result.concepts.mechanisms}")

        print(f"\nQueries used:")
        for name, query in result.queries_used.items():
            print(f"  {name}: {query[:80]}...")

        print(f"\nResults: {len(result.citations)} citations, {len(result.trials)} trials")
        print(f"Filtered: {result.filtered_count}")

        if result.citations:
            print(f"\nTop 3 citations:")
            for cit in result.citations[:3]:
                print(f"  - [{cit.pmid}] {cit.title[:70]}...")
