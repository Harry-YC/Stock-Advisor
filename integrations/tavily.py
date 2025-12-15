"""
Tavily Web Search Integration

Provides web search fallback when local RAG context is insufficient.
Used to supplement expert panel discussions with real-time web data.
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    """Single web search result."""
    title: str
    url: str
    content: str
    score: float
    source: str = "web"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "score": self.score,
            "source": self.source
        }


class TavilyClient:
    """
    Tavily AI web search client.

    Tavily provides AI-powered search optimized for RAG applications.
    Falls back gracefully if API key not available.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tavily client.

        Args:
            api_key: Tavily API key (defaults to TAVILY_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._client = None

        if not self.api_key:
            logger.warning("TAVILY_API_KEY not set. Web search disabled.")

    def is_available(self) -> bool:
        """Check if Tavily is available for use."""
        return bool(self.api_key)

    def _get_client(self):
        """Lazy-load Tavily client."""
        if self._client is None:
            try:
                from tavily import TavilyClient as TC
                self._client = TC(api_key=self.api_key)
            except ImportError:
                logger.error("tavily-python not installed. Run: pip install tavily-python")
                raise
        return self._client

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> List[WebSearchResult]:
        """
        Search the web using Tavily.

        Args:
            query: Search query
            max_results: Maximum number of results (default 5)
            search_depth: "basic" or "advanced" (advanced is slower but better)
            include_domains: Only search these domains
            exclude_domains: Exclude these domains

        Returns:
            List of WebSearchResult objects
        """
        if not self.is_available():
            return []

        try:
            client = self._get_client()

            # Perform search
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=include_domains,
                exclude_domains=exclude_domains
            )

            results = []
            for i, result in enumerate(response.get('results', [])):
                results.append(WebSearchResult(
                    title=result.get('title', 'Untitled'),
                    url=result.get('url', ''),
                    content=result.get('content', ''),
                    score=result.get('score', 1.0 - (i * 0.1)),  # Fallback score
                    source="tavily"
                ))

            logger.info(f"Tavily search returned {len(results)} results for: {query[:50]}...")
            return results

        except ImportError:
            logger.warning("tavily-python not installed")
            return []
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []

    def search_for_rag(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search and format results for RAG context injection.

        Returns results in the same format as LocalRetriever.retrieve()
        so they can be seamlessly combined.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of dicts compatible with RAG context format
        """
        results = self.search(query, max_results)

        return [
            {
                "content": r.content,
                "source": f"[Web] {r.title}",
                "url": r.url,
                "score": r.score,
                "retrieval_method": "web_search"
            }
            for r in results
        ]


def format_web_results(results: List[WebSearchResult]) -> str:
    """
    Format web search results for display or injection into prompts.

    Args:
        results: List of WebSearchResult objects

    Returns:
        Formatted string
    """
    if not results:
        return ""

    lines = [
        "",
        "=" * 60,
        "WEB SEARCH RESULTS",
        "=" * 60,
        ""
    ]

    for i, r in enumerate(results, 1):
        lines.extend([
            f"[{i}] {r.title}",
            f"    URL: {r.url}",
            f"    {r.content[:500]}...",
            ""
        ])

    return "\n".join(lines)


# Convenience functions

def search_web(query: str, max_results: int = 5) -> List[WebSearchResult]:
    """
    Quick web search using default client.
    
    DEPRECATED: Use integrations.google_search instead.
    """
    import warnings
    warnings.warn(
        "Tavily search is deprecated. Use Google Search grounding instead.",
        DeprecationWarning
    )
    client = TavilyClient()
    return client.search(query, max_results)


def search_web_for_rag(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Quick web search formatted for RAG context.

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        List of dicts compatible with RAG context format
    """
    client = TavilyClient()
    return client.search_for_rag(query, max_results)
