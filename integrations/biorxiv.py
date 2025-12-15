"""
bioRxiv/medRxiv Preprint API Client

Search and retrieve preprints for early access to research.
Uses the bioRxiv API (also covers medRxiv).

API Documentation: https://api.biorxiv.org/
"""

import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class PreprintServer(Enum):
    """Preprint server options"""
    BIORXIV = "biorxiv"
    MEDRXIV = "medrxiv"


@dataclass
class Preprint:
    """Represents a preprint from bioRxiv/medRxiv"""
    doi: str
    title: str
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    date: str = ""  # YYYY-MM-DD format
    server: str = ""  # biorxiv or medrxiv
    category: str = ""
    version: int = 1
    published_doi: Optional[str] = None  # DOI of published version if available
    license: str = ""
    url: str = ""

    @property
    def is_published(self) -> bool:
        """Check if this preprint has been published in a peer-reviewed journal"""
        return self.published_doi is not None and len(self.published_doi) > 0

    @property
    def author_string(self) -> str:
        """Get authors as a formatted string"""
        if len(self.authors) <= 3:
            return ", ".join(self.authors)
        return f"{self.authors[0]} et al."


class BioRxivClient:
    """Client for bioRxiv/medRxiv API"""

    BASE_URL = "https://api.biorxiv.org"

    def __init__(self):
        """Initialize the client"""
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json"
        })

    def search_by_date_range(
        self,
        server: PreprintServer = PreprintServer.BIORXIV,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> List[Preprint]:
        """
        Get preprints from a date range

        Args:
            server: biorxiv or medrxiv
            start_date: Start date (YYYY-MM-DD), defaults to 30 days ago
            end_date: End date (YYYY-MM-DD), defaults to today
            limit: Maximum results

        Returns:
            List of Preprint objects
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        preprints = []
        cursor = 0

        while len(preprints) < limit:
            url = f"{self.BASE_URL}/details/{server.value}/{start_date}/{end_date}/{cursor}"

            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                print(f"bioRxiv API error: {e}")
                break

            collection = data.get("collection", [])
            if not collection:
                break

            for item in collection:
                preprints.append(self._parse_preprint(item, server.value))
                if len(preprints) >= limit:
                    break

            # Check for more results
            messages = data.get("messages", [])
            if messages:
                total = messages[0].get("total", 0)
                if cursor + len(collection) >= total:
                    break

            cursor += len(collection)

        return preprints

    def search_by_keyword(
        self,
        query: str,
        server: Optional[PreprintServer] = None,
        limit: int = 50,
        days_back: int = 365
    ) -> List[Preprint]:
        """
        Search preprints by keyword in title/abstract

        Note: bioRxiv API doesn't have direct keyword search.
        This fetches recent preprints and filters locally.

        Args:
            query: Search query (searches title and abstract)
            server: Optional server filter (None = both)
            limit: Maximum results
            days_back: How far back to search

        Returns:
            List of matching Preprint objects
        """
        query_lower = query.lower()
        query_terms = query_lower.split()

        # Determine which servers to search
        servers = [server] if server else [PreprintServer.BIORXIV, PreprintServer.MEDRXIV]

        all_preprints = []
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        for srv in servers:
            preprints = self.search_by_date_range(
                server=srv,
                start_date=start_date,
                end_date=end_date,
                limit=1000  # Fetch more to filter
            )

            # Filter by query
            for p in preprints:
                text = f"{p.title} {p.abstract}".lower()
                if all(term in text for term in query_terms):
                    all_preprints.append(p)

        # Sort by date (newest first) and limit
        all_preprints.sort(key=lambda x: x.date, reverse=True)
        return all_preprints[:limit]

    def get_preprint_by_doi(self, doi: str) -> Optional[Preprint]:
        """
        Get a specific preprint by DOI

        Args:
            doi: The preprint DOI (e.g., 10.1101/2023.01.01.123456)

        Returns:
            Preprint object or None
        """
        # Clean DOI
        doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

        # Try both servers
        for server in ["biorxiv", "medrxiv"]:
            url = f"{self.BASE_URL}/details/{server}/{doi}"

            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                data = response.json()

                collection = data.get("collection", [])
                if collection:
                    # Return most recent version
                    return self._parse_preprint(collection[-1], server)

            except requests.exceptions.RequestException:
                continue

        return None

    def get_publication_status(self, doi: str) -> Dict[str, Any]:
        """
        Check if a preprint has been published

        Args:
            doi: The preprint DOI

        Returns:
            Dict with publication status info
        """
        # Clean DOI
        doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

        for server in ["biorxiv", "medrxiv"]:
            url = f"{self.BASE_URL}/pub/{server}/{doi}"

            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                data = response.json()

                collection = data.get("collection", [])
                if collection:
                    pub_info = collection[0]
                    return {
                        "is_published": True,
                        "published_doi": pub_info.get("published_doi", ""),
                        "published_journal": pub_info.get("published_journal", ""),
                        "published_date": pub_info.get("published_date", ""),
                        "preprint_doi": doi
                    }

            except requests.exceptions.RequestException:
                continue

        return {
            "is_published": False,
            "preprint_doi": doi
        }

    def get_recent_by_category(
        self,
        category: str,
        server: PreprintServer = PreprintServer.BIORXIV,
        days_back: int = 30,
        limit: int = 50
    ) -> List[Preprint]:
        """
        Get recent preprints in a specific category

        Args:
            category: Category name (e.g., 'pharmacology-toxicology', 'oncology')
            server: biorxiv or medrxiv
            days_back: How many days back to search
            limit: Maximum results

        Returns:
            List of Preprint objects in the category
        """
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        all_preprints = self.search_by_date_range(
            server=server,
            start_date=start_date,
            end_date=end_date,
            limit=500
        )

        category_lower = category.lower()
        filtered = [
            p for p in all_preprints
            if category_lower in p.category.lower()
        ]

        return filtered[:limit]

    def _parse_preprint(self, data: Dict, server: str) -> Preprint:
        """Parse API response into Preprint object"""
        # Parse authors
        authors_str = data.get("authors", "")
        authors = [a.strip() for a in authors_str.split(";") if a.strip()]

        doi = data.get("doi", "")

        return Preprint(
            doi=doi,
            title=data.get("title", ""),
            authors=authors,
            abstract=data.get("abstract", ""),
            date=data.get("date", ""),
            server=server,
            category=data.get("category", ""),
            version=int(data.get("version", 1)),
            published_doi=data.get("published", None),
            license=data.get("license", ""),
            url=f"https://www.{server}.org/content/{doi}"
        )


# Common bioRxiv categories for drug development
DRUG_DEVELOPMENT_CATEGORIES = {
    "biorxiv": [
        "pharmacology-toxicology",
        "cancer-biology",
        "biochemistry",
        "cell-biology",
        "molecular-biology",
        "immunology",
        "systems-biology",
        "bioinformatics"
    ],
    "medrxiv": [
        "oncology",
        "pharmacology-and-therapeutics",
        "infectious-diseases",
        "immunology",
        "hematology"
    ]
}


def search_preprints(
    query: str,
    include_biorxiv: bool = True,
    include_medrxiv: bool = True,
    days_back: int = 180,
    limit: int = 50
) -> List[Preprint]:
    """
    Convenience function to search preprints across servers

    Args:
        query: Search query
        include_biorxiv: Include bioRxiv results
        include_medrxiv: Include medRxiv results
        days_back: How far back to search
        limit: Maximum results

    Returns:
        List of matching Preprint objects
    """
    client = BioRxivClient()
    results = []

    if include_biorxiv:
        biorxiv_results = client.search_by_keyword(
            query,
            server=PreprintServer.BIORXIV,
            limit=limit,
            days_back=days_back
        )
        results.extend(biorxiv_results)

    if include_medrxiv:
        medrxiv_results = client.search_by_keyword(
            query,
            server=PreprintServer.MEDRXIV,
            limit=limit,
            days_back=days_back
        )
        results.extend(medrxiv_results)

    # Sort by date and deduplicate
    seen_dois = set()
    unique_results = []
    for p in sorted(results, key=lambda x: x.date, reverse=True):
        if p.doi not in seen_dois:
            seen_dois.add(p.doi)
            unique_results.append(p)

    return unique_results[:limit]


def preprint_to_citation_format(preprint: Preprint) -> Dict[str, Any]:
    """
    Convert a Preprint to the internal Citation format for unified handling

    Args:
        preprint: Preprint object

    Returns:
        Dict compatible with Citation dataclass
    """
    return {
        "pmid": "",  # Preprints don't have PMIDs
        "title": preprint.title,
        "authors": preprint.authors,
        "journal": f"{preprint.server.upper()} (preprint)",
        "year": preprint.date[:4] if preprint.date else "",
        "abstract": preprint.abstract,
        "doi": preprint.doi,
        "is_preprint": True,
        "preprint_server": preprint.server,
        "published_doi": preprint.published_doi
    }
