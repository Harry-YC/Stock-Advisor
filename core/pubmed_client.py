"""
PubMed E-utilities API Client

Standalone literature search via PubMed with rate limiting and caching.
Generalized for any research domain (not PICO-specific).

Features:
- Search execution via E-search
- Citation fetching via E-fetch
- Rate limiting (3 req/sec, 10 with API key)
- Query translation display
- Date/study type filters
- Exponential backoff for rate limit errors

Usage:
    from core.pubmed_client import PubMedClient

    client = PubMedClient(email="your@email.com")
    result = client.search("machine learning")
    citations, failed = client.fetch_citations(result["pmids"])
"""

import time
import logging
import requests
import hashlib
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from html import unescape

# Set up logger for this module
logger = logging.getLogger("literature_review.pubmed_client")


@dataclass
class Citation:
    """
    Simplified citation metadata
    """
    pmid: str
    title: str
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    year: str = ""
    abstract: str = ""
    doi: str = ""
    publication_types: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Metadata
    fetched_at: str = ""
    retracted: bool = False
    retraction_reason: str = ""


class PubMedClient:
    """
    PubMedE-utilities API client with rate limiting

    Handles search, citation fetching, and rate limiting for PubMed API.
    """

    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, email: str, api_key: Optional[str] = None):
        """
        Initialize PubMed client

        Args:
            email: Required by PubMed (for tracking abuse)
            api_key: Optional API key (increases rate limit to 10 req/sec)
        """
        self.email = email
        self.api_key = api_key
        self.rate_limit = 10 if api_key else 3  # Requests per second
        self._last_request = None

    def _rate_limit(self) -> None:
        """Ensure we don't exceed PubMed rate limit"""
        if self._last_request:
            elapsed = time.time() - self._last_request
            min_interval = 1.0 / self.rate_limit

            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

        self._last_request = time.time()

    def search(
        self,
        query: str,
        database: str = "pubmed",
        max_results: int = 100,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Execute search query via E-search

        Args:
            query: Search string (can include MeSH terms with [MeSH])
            database: "pubmed" (default) or "pmc"
            max_results: Max results to retrieve (default: 100)
            filters: Optional filters:
                - date_from: "2020/01/01" (YYYY/MM/DD)
                - date_to: "2025/12/31"
                - article_types: ["Clinical Trial", "Randomized Controlled Trial"]

        Returns:
            {
                "pmids": List[str],
                "count": int (total matching, may exceed max_results),
                "query_translation": str (PubMed's interpretation of query)
            }

        Raises:
            requests.exceptions.RequestException: API error
        """
        # Rate limiting
        self._rate_limit()

        # Build E-search parameters
        params = {
            "db": database,
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "email": self.email,
            "tool": "LiteratureReviewPlatform"
        }

        if self.api_key:
            params["api_key"] = self.api_key

        # Apply filters
        if filters:
            if filters.get("date_from"):
                params["mindate"] = filters["date_from"]
            if filters.get("date_to"):
                params["maxdate"] = filters["date_to"]
            if filters.get("article_types"):
                # Add publication type filters to query
                type_query = " OR ".join([f'"{t}"[Publication Type]' for t in filters["article_types"]])
                params["term"] = f'({query}) AND ({type_query})'

        # Execute search with retry logic for rate limiting
        max_retries = 3
        retry_delay = 2  # Start with 2 seconds

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{self.EUTILS_BASE}/esearch.fcgi",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                esearch_result = data.get("esearchresult", {})

                return {
                    "pmids": esearch_result.get("idlist", []),
                    "count": int(esearch_result.get("count", 0)),
                    "query_translation": esearch_result.get("querytranslation", query)
                }

            except requests.exceptions.HTTPError as e:
                # Handle 429 rate limiting errors with exponential backoff
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff: 2s, 4s, 8s
                    continue
                raise requests.exceptions.RequestException(f"PubMed E-search failed: {str(e)}")
            except requests.exceptions.RequestException as e:
                raise requests.exceptions.RequestException(f"PubMed E-search failed: {str(e)}")

    def fetch_citations(
        self,
        pmids: List[str],
        batch_size: int = 100
    ) -> Tuple[List[Citation], List[Dict]]:
        """
        Fetch full citations for PMIDs via E-fetch

        Args:
            pmids: List of PubMed IDs
            batch_size: Process in batches (default: 100)

        Returns:
            Tuple of (citations, failed_batches):
            - citations: List of Citation objects with full metadata
            - failed_batches: List of dicts with 'start', 'end', 'error' keys
        """
        citations = []
        failed_batches = []

        # Process in batches to respect rate limiting
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            batch_end = min(i + batch_size, len(pmids))

            # Rate limiting
            self._rate_limit()

            # Build E-fetch parameters
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
                "email": self.email,
                "tool": "LiteratureReviewPlatform"
            }

            if self.api_key:
                params["api_key"] = self.api_key

            try:
                response = requests.get(
                    f"{self.EUTILS_BASE}/efetch.fcgi",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()

                # Parse XML response using ElementTree
                xml_text = response.text

                # Extract individual citations
                for pmid in batch:
                    citation = self._parse_citation_from_xml(xml_text, pmid)
                    if citation:
                        citations.append(citation)

            except requests.exceptions.RequestException as e:
                # Log error and track failed batch
                error_msg = str(e)
                logger.warning(f"Failed to fetch batch {i}-{batch_end}: {error_msg}")
                failed_batches.append({
                    'start': i,
                    'end': batch_end,
                    'error': error_msg
                })
                continue
            except ET.ParseError as e:
                # XML parsing error
                error_msg = f"XML parse error: {str(e)}"
                logger.warning(f"Failed to parse batch {i}-{batch_end}: {error_msg}")
                failed_batches.append({
                    'start': i,
                    'end': batch_end,
                    'error': error_msg
                })
                continue

        if failed_batches:
            logger.warning(
                f"PubMed fetch completed with {len(failed_batches)} failed batch(es) "
                f"out of {(len(pmids) + batch_size - 1) // batch_size} total"
            )

        return citations, failed_batches

    def fetch_by_doi(self, dois: List[str]) -> Tuple[List[Citation], List[str]]:
        """
        Fetch citations by DOI (Digital Object Identifier)

        Workflow:
        1. Search PubMed for each DOI using esearch
        2. Extract PMIDs from search results
        3. Fetch full citations using existing fetch_citations()

        Args:
            dois: List of DOI strings (with or without URL prefix)

        Returns:
            Tuple of (citations, not_found_dois)
            - citations: Successfully fetched Citation objects
            - not_found_dois: DOIs that were not found in PubMed

        Examples:
            >>> client = PubMedClient("your@email.com")
            >>> citations, not_found = client.fetch_by_doi(["10.1038/nature12373"])
        """
        pmid_map = {}  # DOI -> PMID mapping
        not_found = []

        for doi in dois:
            # Rate limiting
            self._rate_limit()

            # Build esearch parameters to search by DOI
            params = {
                "db": "pubmed",
                "term": f"{doi}[doi]",  # Search for DOI in DOI field
                "retmode": "json",
                "retmax": 1,  # We expect only one result per DOI
                "email": self.email,
                "tool": "LiteratureReviewPlatform"
            }

            if self.api_key:
                params["api_key"] = self.api_key

            try:
                response = requests.get(
                    f"{self.EUTILS_BASE}/esearch.fcgi",
                    params=params,
                    timeout=10
                )
                response.raise_for_status()

                esearch_result = response.json().get("esearchresult", {})
                id_list = esearch_result.get("idlist", [])

                if id_list:
                    # DOI found - map it to PMID
                    pmid = id_list[0]
                    pmid_map[doi] = pmid
                else:
                    # DOI not found in PubMed
                    not_found.append(doi)

            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to search for DOI {doi}: {str(e)}")
                not_found.append(doi)
                continue

        # Fetch citations for all found PMIDs
        if pmid_map:
            pmids = list(pmid_map.values())
            citations, failed_batches = self.fetch_citations(pmids)
            # Log any failed batches during DOI fetch
            if failed_batches:
                logger.warning(f"Some citations failed to fetch during DOI lookup: {len(failed_batches)} batch(es)")
        else:
            citations = []

        return (citations, not_found)

    def _parse_citation_from_xml(self, xml_text: str, pmid: str) -> Optional[Citation]:
        """
        Parse citation from PubMed XML using ElementTree.

        Args:
            xml_text: Full XML response from PubMed E-fetch
            pmid: PMID to extract from the XML

        Returns:
            Citation object or None if not found/parse error
        """
        try:
            # Parse the full XML
            root = ET.fromstring(xml_text)

            # Find the PubmedArticle containing this PMID
            article = None
            for pub_article in root.findall('.//PubmedArticle'):
                pmid_elem = pub_article.find('.//PMID')
                if pmid_elem is not None and pmid_elem.text == pmid:
                    article = pub_article
                    break

            if article is None:
                logger.debug(f"PMID {pmid} not found in XML response")
                return None

            # Extract title
            title_elem = article.find('.//ArticleTitle')
            title = unescape(title_elem.text) if title_elem is not None and title_elem.text else f"[PMID: {pmid}]"

            # Extract journal
            journal_elem = article.find('.//Journal/Title')
            journal = unescape(journal_elem.text) if journal_elem is not None and journal_elem.text else ""

            # Extract year (try multiple locations)
            year = ""
            year_elem = article.find('.//PubDate/Year')
            if year_elem is not None and year_elem.text:
                year = year_elem.text
            else:
                # Try MedlineDate as fallback
                medline_date = article.find('.//PubDate/MedlineDate')
                if medline_date is not None and medline_date.text:
                    # Extract year from MedlineDate (e.g., "2020 Jan-Feb" -> "2020")
                    year = medline_date.text[:4] if len(medline_date.text) >= 4 else ""

            # Extract abstract (may have multiple AbstractText elements)
            abstract_parts = []
            for abstract_elem in article.findall('.//Abstract/AbstractText'):
                if abstract_elem.text:
                    # Include label if present (e.g., "BACKGROUND:", "METHODS:")
                    label = abstract_elem.get('Label', '')
                    if label:
                        abstract_parts.append(f"{label}: {unescape(abstract_elem.text)}")
                    else:
                        abstract_parts.append(unescape(abstract_elem.text))
            abstract = " ".join(abstract_parts)

            # Extract authors (no arbitrary limit - track all)
            authors = []
            author_count = 0
            for author_elem in article.findall('.//Author'):
                last_name = author_elem.find('LastName')
                fore_name = author_elem.find('ForeName')
                if last_name is not None and last_name.text:
                    if fore_name is not None and fore_name.text:
                        authors.append(f"{fore_name.text} {last_name.text}")
                    else:
                        authors.append(last_name.text)
                    author_count += 1

            # Extract DOI
            doi = ""
            for elocation in article.findall('.//ELocationID'):
                if elocation.get('EIdType') == 'doi' and elocation.text:
                    doi = elocation.text
                    break

            # Extract publication types
            pub_types = []
            for pub_type in article.findall('.//PublicationType'):
                if pub_type.text:
                    pub_types.append(pub_type.text)

            # Extract keywords
            keywords = []
            for keyword in article.findall('.//Keyword'):
                if keyword.text:
                    keywords.append(keyword.text)

            return Citation(
                pmid=pmid,
                title=title,
                authors=authors,
                journal=journal,
                year=year,
                abstract=abstract,
                doi=doi,
                publication_types=pub_types,
                keywords=keywords,
                fetched_at=datetime.now().isoformat()
            )

        except ET.ParseError as e:
            logger.warning(f"XML parse error for PMID {pmid}: {str(e)}")
            return None
        except Exception as e:
            logger.warning(f"Failed to parse PMID {pmid}: {str(e)}")
            return None

    def get_query_hash(self, query: str) -> str:
        """
        Generate SHA-256 hash of query for reproducibility tracking

        Args:
            query: Search query string

        Returns:
            Hex digest of SHA-256 hash
        """
        return hashlib.sha256(query.encode('utf-8')).hexdigest()


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("PubMed Client - Standalone Test")
    print("=" * 70)

    # Initialize client
    client = PubMedClient(email="test@example.com")

    # Test search
    print("\n1. Testing PubMed search...")
    query = "machine learning"

    try:
        result = client.search(query, max_results=5)

        print(f"   Query: {query}")
        print(f"   Total results: {result['count']}")
        print(f"   PMIDs retrieved: {len(result['pmids'])}")
        print(f"   Query translation: {result['query_translation'][:80]}...")

        # Test citation fetching
        if result['pmids']:
            print("\n2. Testing citation fetch...")
            pmid = result['pmids'][0]
            citations, failed_batches = client.fetch_citations([pmid])

            if failed_batches:
                print(f"   Warning: {len(failed_batches)} batch(es) failed")

            if citations:
                c = citations[0]
                print(f"   PMID: {c.pmid}")
                print(f"   Title: {c.title[:60]}...")
                print(f"   Authors: {', '.join(c.authors[:3])}...")
                print(f"   Journal: {c.journal}")
                print(f"   Year: {c.year}")

        print("\n" + "=" * 70)
        print("✓ PubMed Client ready for integration")

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("Note: This test requires internet connection and PubMed API access")
