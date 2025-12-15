"""
Two-Channel Search Service for Palliative Surgery

Executes parallel searches across:
1. Clinical Channel: Surgical outcomes, comparative effectiveness, morbidity/mortality
2. Symptom Channel: QoL, symptom control, functional outcomes, patient-centered

Returns results organized by channel for comprehensive guideline development.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from config import settings
from core.pubmed_client import PubMedClient, Citation
from core.query_extractor import ClinicalQueryExtractor, ExtractedConcepts, get_synonyms
from core.search_channels import (
    TwoChannelQueryBuilder,
    ChannelQuerySet,
    ChannelQuery,
    SearchChannel
)

logger = logging.getLogger(__name__)


@dataclass
class ChannelResult:
    """Results from a single channel."""
    channel: SearchChannel
    citations: List[Citation] = field(default_factory=list)
    queries_executed: List[str] = field(default_factory=list)
    query_purposes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "channel": self.channel.value,
            "citations": [c.__dict__ if hasattr(c, '__dict__') else c for c in self.citations],
            "queries_executed": self.queries_executed,
            "query_purposes": self.query_purposes,
            "citation_count": len(self.citations)
        }


@dataclass
class QualityReport:
    """Quality report for two-channel search results."""
    clinical_count: int = 0
    symptom_count: int = 0
    condition_coverage: Dict[str, int] = field(default_factory=dict)
    procedure_coverage: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    filtered_out: int = 0

    def to_dict(self) -> Dict:
        return {
            "clinical_count": self.clinical_count,
            "symptom_count": self.symptom_count,
            "condition_coverage": self.condition_coverage,
            "procedure_coverage": self.procedure_coverage,
            "warnings": self.warnings,
            "filtered_out": self.filtered_out
        }


@dataclass
class TwoChannelSearchResult:
    """Results from two-channel search."""
    clinical: ChannelResult = None
    symptom: ChannelResult = None
    concepts: ExtractedConcepts = None
    trials: List[Dict] = field(default_factory=list)
    quality_report: QualityReport = None

    def __post_init__(self):
        if self.clinical is None:
            self.clinical = ChannelResult(channel=SearchChannel.CLINICAL)
        if self.symptom is None:
            self.symptom = ChannelResult(channel=SearchChannel.SYMPTOM)
        if self.quality_report is None:
            self.quality_report = QualityReport()

    @property
    def all_citations(self) -> List[Citation]:
        """Get all citations from both channels (deduplicated)."""
        seen = set()
        result = []
        for cit in self.clinical.citations + self.symptom.citations:
            if cit.pmid not in seen:
                seen.add(cit.pmid)
                result.append(cit)
        return result

    def to_dict(self) -> Dict:
        return {
            "clinical": self.clinical.to_dict(),
            "symptom": self.symptom.to_dict(),
            "concepts": self.concepts.to_dict() if self.concepts else {},
            "trials": self.trials,
            "total_citations": len(self.all_citations),
            "quality_report": self.quality_report.to_dict() if self.quality_report else {}
        }


class TwoChannelSearchService:
    """
    Execute two-channel search for comprehensive palliative surgery evidence.

    Clinical Channel:
    - Surgical outcomes (mortality, morbidity)
    - Comparative effectiveness (surgery vs stent)
    - Perioperative risk
    - Technical success rates

    Symptom Channel:
    - Quality of life outcomes
    - Symptom palliation
    - Functional outcomes (oral intake, mobility)
    - Patient-centered measures
    """

    MAX_QUERIES_PER_CHANNEL = 6
    MAX_RESULTS_PER_QUERY = 15
    MAX_RESULTS_PER_CHANNEL = 30

    def __init__(self, api_key: str = None):
        """
        Initialize two-channel search service.

        Args:
            api_key: API key for LLM-based extraction
        """
        self.api_key = api_key
        self.extractor = ClinicalQueryExtractor(api_key=api_key)
        self.query_builder = TwoChannelQueryBuilder()
        self.pubmed = PubMedClient(
            email=getattr(settings, 'PUBMED_EMAIL', 'user@example.com'),
            api_key=getattr(settings, 'PUBMED_API_KEY', None)
        )

    def search(
        self,
        question: str,
        concepts: ExtractedConcepts = None
    ) -> TwoChannelSearchResult:
        """
        Execute two-channel search.

        Args:
            question: User's clinical question
            concepts: Pre-extracted concepts (optional, will extract if not provided)

        Returns:
            TwoChannelSearchResult with separated channel results
        """
        result = TwoChannelSearchResult()

        # 1. Extract concepts if not provided
        if concepts is None:
            concepts = self.extractor.extract(question)
        result.concepts = concepts

        if concepts.is_empty():
            logger.warning(f"No concepts extracted from question: {question[:100]}...")
            return result

        logger.info(f"Two-channel search with concepts: {concepts.to_dict()}")

        # 2. Build channel-specific queries
        query_set = self.query_builder.build(concepts)

        # 3. Execute both channels in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            clinical_future = executor.submit(
                self._execute_channel,
                query_set.clinical_queries,
                SearchChannel.CLINICAL
            )
            symptom_future = executor.submit(
                self._execute_channel,
                query_set.symptom_queries,
                SearchChannel.SYMPTOM
            )

            result.clinical = clinical_future.result()
            result.symptom = symptom_future.result()

        # 4. Score and rank within each channel
        result.clinical.citations = self._score_and_rank(
            result.clinical.citations,
            concepts,
            is_clinical=True
        )
        result.symptom.citations = self._score_and_rank(
            result.symptom.citations,
            concepts,
            is_clinical=False
        )

        # 5. Filter symptom papers - must mention relevant outcomes
        symptom_before = len(result.symptom.citations)
        result.symptom.citations = self._filter_symptom_strict(
            result.symptom.citations,
            concepts
        )
        symptom_filtered_out = symptom_before - len(result.symptom.citations)

        # 6. Search clinical trials for palliative interventions
        if concepts.conditions or concepts.procedures:
            result.trials = self._search_trials_palliative(concepts)

        # 7. Generate quality report with warnings
        result.quality_report = self._generate_quality_report(result, concepts, symptom_filtered_out)

        logger.info(
            f"Two-channel search complete: "
            f"{len(result.clinical.citations)} clinical, "
            f"{len(result.symptom.citations)} symptom (filtered {symptom_filtered_out}), "
            f"{len(result.trials)} trials"
        )

        if result.quality_report.warnings:
            for warning in result.quality_report.warnings:
                logger.warning(warning)

        return result

    def _execute_channel(
        self,
        queries: List[ChannelQuery],
        channel: SearchChannel
    ) -> ChannelResult:
        """Execute all queries for a single channel."""
        result = ChannelResult(channel=channel)
        seen_pmids = set()
        result_lock = Lock()

        # Sort by priority and limit
        sorted_queries = sorted(queries, key=lambda q: q.priority)
        queries_to_run = sorted_queries[:self.MAX_QUERIES_PER_CHANNEL]

        # Execute queries in parallel with thread safety
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._search_pubmed, q.query): q
                for q in queries_to_run
            }

            for future in as_completed(futures):
                query_obj = futures[future]
                try:
                    citations = future.result()
                    with result_lock:
                        for cit in citations:
                            if cit.pmid not in seen_pmids:
                                seen_pmids.add(cit.pmid)
                                result.citations.append(cit)

                        result.queries_executed.append(query_obj.query)
                        result.query_purposes.append(query_obj.purpose)

                    logger.debug(
                        f"{channel.value} query [{query_obj.purpose}] "
                        f"returned {len(citations)} results"
                    )
                except Exception as e:
                    logger.warning(f"{channel.value} query failed: {e}")

        # Limit results per channel
        if len(result.citations) > self.MAX_RESULTS_PER_CHANNEL:
            result.citations = result.citations[:self.MAX_RESULTS_PER_CHANNEL]

        return result

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

    def _score_and_rank(
        self,
        citations: List[Citation],
        concepts: ExtractedConcepts,
        is_clinical: bool
    ) -> List[Citation]:
        """Score and rank citations based on relevance."""
        if not citations:
            return []

        scored = []
        for cit in citations:
            score = self._score_citation(cit, concepts, is_clinical)
            scored.append((score, cit))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [cit for _, cit in scored]

    def _score_citation(
        self,
        citation: Citation,
        concepts: ExtractedConcepts,
        is_clinical: bool
    ) -> float:
        """Score a citation for relevance to palliative surgery."""
        text = f"{citation.title} {citation.abstract}".lower()
        score = 0.0

        # Condition mentions (high weight)
        for condition in concepts.conditions:
            for variant in [condition.lower()] + [s.lower() for s in get_synonyms(condition, "condition")]:
                if variant in text:
                    score += 3.0
                    break

        # Procedure mentions (high weight)
        for proc in concepts.procedures:
            for variant in [proc.lower()] + [s.lower() for s in get_synonyms(proc, "procedure")]:
                if variant in text:
                    score += 2.5
                    break

        # Cancer type mentions
        for cancer in concepts.cancers:
            if cancer.lower() in text:
                score += 1.5

        # Channel-specific boosts
        if is_clinical:
            # Clinical channel: boost surgical/mortality/morbidity terms
            clinical_terms = [
                "mortality", "morbidity", "complication", "surgical",
                "perioperative", "postoperative", "30-day", "90-day",
                "operative", "procedure", "intervention",
                "retrospective", "prospective",  # Observational studies
                "comparative", "versus", "compared"
            ]
            for term in clinical_terms:
                if term in text:
                    score += 1.0

            # LOW-EVIDENCE DOMAIN: Boost case series as PRIMARY evidence source
            # In palliative surgery, case series ARE the best available evidence
            case_series_terms = [
                "case series", "consecutive patients", "institutional experience",
                "clinical experience", "single-center experience", "multicenter experience",
                "our experience", "single institution", "review of patients"
            ]
            for term in case_series_terms:
                if term in text:
                    score += 2.0  # Higher boost - these are the key supporting examples
                    break  # Only count once

        else:
            # Symptom channel: boost QoL/symptom/functional terms
            symptom_terms = [
                "quality of life", "qol", "symptom", "palliation",
                "functional", "oral intake", "pain", "comfort",
                "patient-reported", "pro", "satisfaction",
                "hospice", "home", "days at home"
            ]
            for term in symptom_terms:
                if term in text:
                    score += 1.0

        # Outcome mentions
        for outcome in concepts.outcomes:
            if outcome.lower() in text:
                score += 1.5

        # Score mentions
        for sc in concepts.scores:
            if sc.lower() in text:
                score += 1.0

        return score

    def _filter_symptom_strict(
        self,
        citations: List[Citation],
        concepts: ExtractedConcepts
    ) -> List[Citation]:
        """
        Filter symptom papers - must mention QoL or symptom-related outcomes.

        Papers that only mention surgical technique without patient outcomes
        are not useful for symptom palliation assessment.

        Args:
            citations: Symptom channel citations
            concepts: Extracted concepts

        Returns:
            Filtered list with relevant symptom/QoL papers
        """
        # Key symptom/QoL terms that must be present
        required_terms = [
            "quality of life", "qol", "symptom", "palliation", "palliative",
            "functional", "oral intake", "pain", "relief", "comfort",
            "patient-reported", "satisfaction", "hospice", "survival",
            "days at home", "gooss", "ecog", "karnofsky"
        ]

        filtered = []

        for cit in citations:
            text = f"{cit.title} {cit.abstract}".lower()

            # Check if paper mentions any symptom/QoL term
            has_symptom_term = any(term in text for term in required_terms)

            # Also check if it mentions relevant outcomes from concepts
            has_outcome = any(
                out.lower() in text for out in concepts.outcomes
            ) if concepts.outcomes else False

            if has_symptom_term or has_outcome:
                filtered.append(cit)
            else:
                logger.debug(f"Filtered out symptom paper (no QoL/symptom terms): {cit.pmid} - {cit.title[:50]}...")

        return filtered

    def _generate_quality_report(
        self,
        result: TwoChannelSearchResult,
        concepts: ExtractedConcepts,
        filtered_out: int
    ) -> QualityReport:
        """
        Generate quality report with condition/procedure coverage and warnings.

        Args:
            result: Search result with citations
            concepts: Extracted concepts
            filtered_out: Number of symptom papers filtered out

        Returns:
            QualityReport with coverage stats and warnings
        """
        report = QualityReport(
            clinical_count=len(result.clinical.citations),
            symptom_count=len(result.symptom.citations),
            filtered_out=filtered_out
        )

        all_citations = result.all_citations

        # Check coverage per condition
        for condition in concepts.conditions:
            synonyms = [condition.lower()] + [s.lower() for s in get_synonyms(condition, "condition")]
            count = sum(
                1 for cit in all_citations
                if any(syn in f"{cit.title} {cit.abstract}".lower() for syn in synonyms)
            )
            report.condition_coverage[condition] = count

            if count == 0:
                report.warnings.append(
                    f"No papers found for condition: {condition}. "
                    f"Evidence may be limited."
                )

        # Check procedure coverage
        for proc in concepts.procedures:
            synonyms = [proc.lower()] + [s.lower() for s in get_synonyms(proc, "procedure")]
            count = sum(
                1 for cit in all_citations
                if any(syn in f"{cit.title} {cit.abstract}".lower() for syn in synonyms)
            )
            report.procedure_coverage[proc] = count

            if count == 0:
                report.warnings.append(
                    f"No papers found for procedure: {proc}. "
                    f"Consider alternative search terms."
                )

        # Overall symptom evidence warning
        if report.symptom_count == 0:
            report.warnings.insert(0,
                "NO SYMPTOM/QoL EVIDENCE: Patient-centered outcomes are critical "
                "for palliative surgery decisions. Consider expanding search."
            )

        # Low evidence acknowledgment - reframe for low-evidence domain
        total = report.clinical_count + report.symptom_count
        if total < 10:
            # In palliative surgery, low evidence volume is EXPECTED, not a failure
            report.warnings.append(
                f"LOW-EVIDENCE DOMAIN: Found {total} papers. "
                f"In palliative surgery, case series and expert consensus form the primary evidence base. "
                f"This is typical for this field."
            )
        elif total < 20:
            # Moderate evidence - still frame appropriately for the domain
            report.warnings.append(
                f"Supporting literature: {total} papers found. "
                f"Expert consensus remains central to recommendations."
            )

        return report

    def _search_trials_palliative(self, concepts: ExtractedConcepts, max_results: int = 10) -> List[Dict]:
        """
        Search ClinicalTrials.gov for palliative surgery trials.

        Focuses on:
        - Palliative interventions
        - Symptom management
        - Surgical vs non-surgical comparisons
        """
        import requests

        trials = []
        seen_ncts = set()

        # Build search terms - condition + procedure + palliative
        search_terms = []

        if concepts.conditions:
            search_terms.append(concepts.conditions[0])
        if concepts.procedures:
            search_terms.append(concepts.procedures[0])

        # Add palliative context
        search_terms.append("palliative")

        if not search_terms:
            return []

        try:
            response = requests.get(
                "https://clinicaltrials.gov/api/v2/studies",
                params={
                    "query.term": " AND ".join(search_terms),
                    "pageSize": max_results,
                    "format": "json"
                },
                timeout=30
            )
            response.raise_for_status()

            for s in response.json().get("studies", []):
                nct_id = s.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "")
                if nct_id and nct_id not in seen_ncts:
                    seen_ncts.add(nct_id)
                    trials.append({
                        "nct_id": nct_id,
                        "title": s.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle", ""),
                        "phase": s.get("protocolSection", {}).get("designModule", {}).get("phases", []),
                        "status": s.get("protocolSection", {}).get("statusModule", {}).get("overallStatus", ""),
                        "sponsor": s.get("protocolSection", {}).get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""),
                        "url": f"https://clinicaltrials.gov/study/{nct_id}"
                    })

        except Exception as e:
            logger.warning(f"Trials search failed: {e}")

        # If no results with palliative, try without
        if not trials and concepts.conditions:
            try:
                response = requests.get(
                    "https://clinicaltrials.gov/api/v2/studies",
                    params={
                        "query.term": concepts.conditions[0],
                        "pageSize": max_results,
                        "format": "json"
                    },
                    timeout=30
                )
                response.raise_for_status()

                for s in response.json().get("studies", []):
                    nct_id = s.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "")
                    if nct_id and nct_id not in seen_ncts:
                        seen_ncts.add(nct_id)
                        trials.append({
                            "nct_id": nct_id,
                            "title": s.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle", ""),
                            "phase": s.get("protocolSection", {}).get("designModule", {}).get("phases", []),
                            "status": s.get("protocolSection", {}).get("statusModule", {}).get("overallStatus", ""),
                            "sponsor": s.get("protocolSection", {}).get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""),
                            "url": f"https://clinicaltrials.gov/study/{nct_id}"
                        })

            except Exception as e:
                logger.error(f"ClinicalTrials fallback search failed: {e}")

        return trials[:max_results]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def two_channel_search(question: str, api_key: str = None) -> TwoChannelSearchResult:
    """
    Execute two-channel search for a palliative surgery question.

    Args:
        question: User's clinical question
        api_key: Optional API key for LLM extraction

    Returns:
        TwoChannelSearchResult with separated channel results
    """
    service = TwoChannelSearchService(api_key=api_key)
    return service.search(question)


def get_clinical_evidence(result: TwoChannelSearchResult) -> List[Citation]:
    """Get citations relevant to clinical outcomes (surgical, morbidity/mortality)."""
    return result.clinical.citations


def get_symptom_evidence(result: TwoChannelSearchResult) -> List[Citation]:
    """Get citations relevant to symptom palliation and QoL."""
    return result.symptom.citations


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import os

    print("=" * 60)
    print("TESTING TWO-CHANNEL SEARCH SERVICE (PALLIATIVE SURGERY)")
    print("=" * 60)

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
    service = TwoChannelSearchService(api_key=api_key)

    question = "Should a patient with malignant bowel obstruction from ovarian cancer undergo gastrojejunostomy or stent placement?"

    print(f"\nSearching: {question[:60]}...")
    result = service.search(question)

    print(f"\nResults Summary:")
    print(f"  Clinical citations: {len(result.clinical.citations)}")
    print(f"  Symptom citations: {len(result.symptom.citations)}")
    print(f"  Total unique: {len(result.all_citations)}")
    print(f"  Trials found: {len(result.trials)}")

    if result.concepts:
        print(f"\nExtracted concepts:")
        print(f"  Conditions: {result.concepts.conditions}")
        print(f"  Procedures: {result.concepts.procedures}")
        print(f"  Cancers: {result.concepts.cancers}")
        print(f"  Outcomes: {result.concepts.outcomes}")

    print("\n" + "=" * 40)
    print("CLINICAL CHANNEL")
    print("=" * 40)
    for i, purpose in enumerate(result.clinical.query_purposes[:5]):
        print(f"  {i+1}. {purpose}")

    if result.clinical.citations:
        print("\nTop 3 clinical papers:")
        for i, cit in enumerate(result.clinical.citations[:3]):
            print(f"  {i+1}. {cit.title[:70]}...")

    print("\n" + "=" * 40)
    print("SYMPTOM CHANNEL")
    print("=" * 40)
    for i, purpose in enumerate(result.symptom.query_purposes[:5]):
        print(f"  {i+1}. {purpose}")

    if result.symptom.citations:
        print("\nTop 3 symptom/QoL papers:")
        for i, cit in enumerate(result.symptom.citations[:3]):
            print(f"  {i+1}. {cit.title[:70]}...")

    if result.quality_report.warnings:
        print("\n" + "=" * 40)
        print("WARNINGS")
        print("=" * 40)
        for warning in result.quality_report.warnings:
            print(f"  {warning}")
