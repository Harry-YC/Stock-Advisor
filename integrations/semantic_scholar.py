"""
Semantic Scholar API Client

Provides citation data for building citation networks.
Free API with generous rate limits (100 requests/second).

API Documentation: https://api.semanticscholar.org/api-docs/
"""

import requests
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class SemanticPaper:
    """Represents a paper from Semantic Scholar"""
    paper_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    citation_count: int = 0
    reference_count: int = 0
    influential_citation_count: int = 0
    venue: str = ""
    doi: str = ""
    pmid: str = ""
    url: str = ""
    fields_of_study: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)  # Paper IDs that cite this
    references: List[str] = field(default_factory=list)  # Paper IDs this cites


class SemanticScholarClient:
    """Client for Semantic Scholar API"""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Semantic Scholar client

        Args:
            api_key: Optional API key for higher rate limits
        """
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests (10 req/sec safe)

    def _rate_limit(self):
        """Ensure we don't exceed rate limits"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None, _retry_count: int = 0) -> Dict:
        """Make a rate-limited request to the API with bounded retries"""
        MAX_RETRIES = 3

        self._rate_limit()
        response = requests.get(
            f"{self.BASE_URL}/{endpoint}",
            headers=self.headers,
            params=params,
            timeout=30
        )
        if response.status_code == 429:
            # Rate limited - retry with exponential backoff, but only up to MAX_RETRIES
            if _retry_count >= MAX_RETRIES:
                raise requests.exceptions.HTTPError(
                    f"Semantic Scholar rate limited after {MAX_RETRIES} retries"
                )
            wait_time = 5 * (2 ** _retry_count)  # Exponential backoff: 5s, 10s, 20s
            time.sleep(wait_time)
            return self._make_request(endpoint, params, _retry_count + 1)
        response.raise_for_status()
        return response.json()

    def get_paper_by_doi(self, doi: str) -> Optional[SemanticPaper]:
        """
        Get paper details by DOI

        Args:
            doi: The paper DOI

        Returns:
            SemanticPaper object or None if not found
        """
        try:
            fields = "paperId,title,authors,year,abstract,citationCount,referenceCount,influentialCitationCount,venue,externalIds,fieldsOfStudy,url"
            data = self._make_request(f"paper/DOI:{doi}", {"fields": fields})
            return self._parse_paper(data)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def get_paper_by_pmid(self, pmid: str) -> Optional[SemanticPaper]:
        """
        Get paper details by PubMed ID

        Args:
            pmid: The PubMed ID

        Returns:
            SemanticPaper object or None if not found
        """
        try:
            fields = "paperId,title,authors,year,abstract,citationCount,referenceCount,influentialCitationCount,venue,externalIds,fieldsOfStudy,url"
            data = self._make_request(f"paper/PMID:{pmid}", {"fields": fields})
            return self._parse_paper(data)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def get_paper(self, paper_id: str) -> Optional[SemanticPaper]:
        """
        Get paper by Semantic Scholar paper ID

        Args:
            paper_id: The Semantic Scholar paper ID

        Returns:
            SemanticPaper object or None if not found
        """
        try:
            fields = "paperId,title,authors,year,abstract,citationCount,referenceCount,influentialCitationCount,venue,externalIds,fieldsOfStudy,url"
            data = self._make_request(f"paper/{paper_id}", {"fields": fields})
            return self._parse_paper(data)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def get_citations(self, paper_id: str, limit: int = 100) -> List[SemanticPaper]:
        """
        Get papers that cite the given paper

        Args:
            paper_id: Semantic Scholar paper ID, DOI (prefixed with DOI:), or PMID (prefixed with PMID:)
            limit: Maximum number of citations to return

        Returns:
            List of SemanticPaper objects
        """
        citations = []
        offset = 0
        fields = "paperId,title,authors,year,citationCount,venue,externalIds"

        while len(citations) < limit:
            try:
                data = self._make_request(
                    f"paper/{paper_id}/citations",
                    {"fields": fields, "limit": min(100, limit - len(citations)), "offset": offset}
                )
            except requests.exceptions.HTTPError:
                break

            batch = data.get("data", [])
            if not batch:
                break

            for item in batch:
                citing_paper = item.get("citingPaper", {})
                if citing_paper.get("paperId"):
                    citations.append(self._parse_paper(citing_paper))

            offset += len(batch)
            if len(batch) < 100:
                break

        return citations

    def get_references(self, paper_id: str, limit: int = 100) -> List[SemanticPaper]:
        """
        Get papers that the given paper cites

        Args:
            paper_id: Semantic Scholar paper ID, DOI (prefixed with DOI:), or PMID (prefixed with PMID:)
            limit: Maximum number of references to return

        Returns:
            List of SemanticPaper objects
        """
        references = []
        offset = 0
        fields = "paperId,title,authors,year,citationCount,venue,externalIds"

        while len(references) < limit:
            try:
                data = self._make_request(
                    f"paper/{paper_id}/references",
                    {"fields": fields, "limit": min(100, limit - len(references)), "offset": offset}
                )
            except requests.exceptions.HTTPError:
                break

            batch = data.get("data", [])
            if not batch:
                break

            for item in batch:
                cited_paper = item.get("citedPaper", {})
                if cited_paper.get("paperId"):
                    references.append(self._parse_paper(cited_paper))

            offset += len(batch)
            if len(batch) < 100:
                break

        return references

    def search(self, query: str, limit: int = 50, year_range: Optional[tuple] = None) -> List[SemanticPaper]:
        """
        Search for papers

        Args:
            query: Search query
            limit: Maximum results
            year_range: Optional (start_year, end_year) tuple

        Returns:
            List of SemanticPaper objects
        """
        papers = []
        offset = 0
        fields = "paperId,title,authors,year,abstract,citationCount,venue,externalIds,fieldsOfStudy"

        params = {"query": query, "fields": fields}
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"

        while len(papers) < limit:
            params["limit"] = min(100, limit - len(papers))
            params["offset"] = offset

            try:
                data = self._make_request("paper/search", params)
            except requests.exceptions.HTTPError:
                break

            batch = data.get("data", [])
            if not batch:
                break

            for paper_data in batch:
                if paper_data.get("paperId"):
                    papers.append(self._parse_paper(paper_data))

            offset += len(batch)
            if len(batch) < 100:
                break

        return papers

    def batch_get_papers(
        self,
        identifiers: List[str],
        id_type: str = "pmid"
    ) -> Dict[str, SemanticPaper]:
        """
        Batch fetch papers by identifiers

        Args:
            identifiers: List of identifiers (PMIDs, DOIs, or paper IDs)
            id_type: Type of identifier ('pmid', 'doi', or 'paper_id')

        Returns:
            Dict mapping identifier to SemanticPaper (missing papers excluded)
        """
        results = {}
        fields = "paperId,title,authors,year,abstract,citationCount,referenceCount,influentialCitationCount,venue,externalIds,fieldsOfStudy,url"

        # Batch API endpoint (POST)
        batch_size = 100
        for i in range(0, len(identifiers), batch_size):
            batch = identifiers[i:i + batch_size]

            # Format IDs based on type
            if id_type == "pmid":
                formatted_ids = [f"PMID:{pid}" for pid in batch]
            elif id_type == "doi":
                formatted_ids = [f"DOI:{doi}" for doi in batch]
            else:
                formatted_ids = batch

            self._rate_limit()
            try:
                response = requests.post(
                    f"{self.BASE_URL}/paper/batch",
                    headers={**self.headers, "Content-Type": "application/json"},
                    json={"ids": formatted_ids},
                    params={"fields": fields},
                    timeout=60
                )
                response.raise_for_status()
                batch_results = response.json()

                for idx, paper_data in enumerate(batch_results):
                    if paper_data:
                        original_id = batch[idx]
                        results[original_id] = self._parse_paper(paper_data)
            except requests.exceptions.HTTPError:
                # Fall back to individual requests
                for identifier in batch:
                    if id_type == "pmid":
                        paper = self.get_paper_by_pmid(identifier)
                    elif id_type == "doi":
                        paper = self.get_paper_by_doi(identifier)
                    else:
                        paper = self.get_paper(identifier)
                    if paper:
                        results[identifier] = paper

        return results

    def _parse_paper(self, data: Dict) -> SemanticPaper:
        """Parse API response into SemanticPaper object"""
        external_ids = data.get("externalIds", {}) or {}
        authors = []
        for author in data.get("authors", []) or []:
            name = author.get("name", "")
            if name:
                authors.append(name)

        return SemanticPaper(
            paper_id=data.get("paperId", ""),
            title=data.get("title", "") or "",
            authors=authors,
            year=data.get("year"),
            abstract=data.get("abstract", "") or "",
            citation_count=data.get("citationCount", 0) or 0,
            reference_count=data.get("referenceCount", 0) or 0,
            influential_citation_count=data.get("influentialCitationCount", 0) or 0,
            venue=data.get("venue", "") or "",
            doi=external_ids.get("DOI", "") or "",
            pmid=external_ids.get("PubMed", "") or "",
            url=data.get("url", "") or "",
            fields_of_study=data.get("fieldsOfStudy", []) or []
        )


def get_citation_metrics_for_papers(
    pmids: List[str],
    api_key: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to get citation metrics for a list of PMIDs

    Args:
        pmids: List of PubMed IDs
        api_key: Optional Semantic Scholar API key

    Returns:
        Dict mapping PMID to metrics dict with keys:
        - citation_count
        - influential_citation_count
        - reference_count
        - semantic_paper_id (for further lookups)
    """
    client = SemanticScholarClient(api_key)
    papers = client.batch_get_papers(pmids, id_type="pmid")

    metrics = {}
    for pmid, paper in papers.items():
        metrics[pmid] = {
            "citation_count": paper.citation_count,
            "influential_citation_count": paper.influential_citation_count,
            "reference_count": paper.reference_count,
            "semantic_paper_id": paper.paper_id,
            "fields_of_study": paper.fields_of_study
        }

    return metrics
