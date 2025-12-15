"""
Citation Utilities Module

Centralized utilities for handling citations across the codebase.
Works with both Citation dataclass objects and dictionary representations.

Usage:
    from core.citation_utils import get_attr, to_dict, format_authors

    # Get attribute from either Citation object or dict
    pmid = get_attr(citation, 'pmid')

    # Convert to normalized dict
    citation_dict = to_dict(citation)

    # Format authors for display
    authors_str = format_authors(citation.authors, max_authors=3)
"""

from typing import Any, List, Optional, Dict, Set
from dataclasses import dataclass


def get_attr(citation: Any, attr: str, default: Any = None) -> Any:
    """
    Unified citation attribute getter.

    Works with Citation dataclass objects, dictionaries, or any object
    with the requested attribute.

    Args:
        citation: Citation object or dict
        attr: Attribute name to retrieve
        default: Default value if attribute not found

    Returns:
        The attribute value or default

    Examples:
        >>> get_attr(citation_obj, 'pmid')
        '12345678'
        >>> get_attr(citation_dict, 'title', 'No title')
        'Study of palliative surgery outcomes'
    """
    if citation is None:
        return default

    # Try object attribute first
    if hasattr(citation, attr):
        value = getattr(citation, attr, default)
        return value if value is not None else default

    # Try dictionary access
    if isinstance(citation, dict):
        value = citation.get(attr, default)
        return value if value is not None else default

    return default


def to_dict(citation: Any) -> Dict[str, Any]:
    """
    Convert Citation object or dict to normalized dictionary.

    Ensures consistent structure regardless of input type.

    Args:
        citation: Citation object or dict

    Returns:
        Normalized dictionary with standard citation fields

    Examples:
        >>> to_dict(Citation(pmid='123', title='Test'))
        {'pmid': '123', 'title': 'Test', 'abstract': '', ...}
    """
    if citation is None:
        return {}

    if isinstance(citation, dict):
        # Already a dict, normalize it
        return {
            'pmid': citation.get('pmid', ''),
            'title': citation.get('title', 'No title'),
            'abstract': citation.get('abstract', ''),
            'authors': citation.get('authors', []),
            'journal': citation.get('journal', ''),
            'year': citation.get('year', ''),
            'doi': citation.get('doi', ''),
            'publication_types': citation.get('publication_types', []),
            'keywords': citation.get('keywords', []),
        }

    # Convert object to dict
    return {
        'pmid': get_attr(citation, 'pmid', ''),
        'title': get_attr(citation, 'title', 'No title'),
        'abstract': get_attr(citation, 'abstract', ''),
        'authors': get_attr(citation, 'authors', []),
        'journal': get_attr(citation, 'journal', ''),
        'year': get_attr(citation, 'year', ''),
        'doi': get_attr(citation, 'doi', ''),
        'publication_types': get_attr(citation, 'publication_types', []),
        'keywords': get_attr(citation, 'keywords', []),
    }


def to_dict_list(citations: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert a list of citations to normalized dictionaries.

    Args:
        citations: List of Citation objects or dicts

    Returns:
        List of normalized dictionaries
    """
    if not citations:
        return []
    return [to_dict(c) for c in citations]


def format_authors(authors: List[str], max_authors: int = 3) -> str:
    """
    Format author list for display.

    Args:
        authors: List of author names
        max_authors: Maximum authors to show before "et al."

    Returns:
        Formatted author string

    Examples:
        >>> format_authors(['Smith J', 'Jones K'])
        'Smith J, Jones K'
        >>> format_authors(['Smith J', 'Jones K', 'Brown M', 'Davis L'])
        'Smith J, Jones K, Brown M, et al. (4 authors)'
    """
    if not authors:
        return "Unknown"

    # Handle case where authors might be a single string
    if isinstance(authors, str):
        return authors

    if len(authors) <= max_authors:
        return ", ".join(authors)

    return f"{', '.join(authors[:max_authors])}, et al. ({len(authors)} authors)"


def format_citation_short(citation: Any) -> str:
    """
    Format citation for short display (e.g., in lists).

    Args:
        citation: Citation object or dict

    Returns:
        Short formatted string like "Smith et al. (2024)"

    Examples:
        >>> format_citation_short(citation)
        'Smith et al. (2024)'
    """
    authors = get_attr(citation, 'authors', [])
    year = get_attr(citation, 'year', 'N/A')

    if not authors:
        first_author = "Unknown"
    elif isinstance(authors, str):
        first_author = authors.split(',')[0].strip()
    else:
        first_author = authors[0] if authors else "Unknown"

    if len(authors) > 1:
        return f"{first_author} et al. ({year})"
    return f"{first_author} ({year})"


def format_citation_full(citation: Any, include_abstract: bool = False) -> str:
    """
    Format citation for full display.

    Args:
        citation: Citation object or dict
        include_abstract: Whether to include abstract

    Returns:
        Full formatted citation string
    """
    pmid = get_attr(citation, 'pmid', '')
    title = get_attr(citation, 'title', 'No title')
    authors = get_attr(citation, 'authors', [])
    journal = get_attr(citation, 'journal', '')
    year = get_attr(citation, 'year', '')
    abstract = get_attr(citation, 'abstract', '')

    authors_str = format_authors(authors)

    parts = [f"**{title}**"]
    if authors_str:
        parts.append(f"*{authors_str}*")
    if journal or year:
        journal_year = f"{journal}" if journal else ""
        if year:
            journal_year += f" ({year})" if journal_year else str(year)
        parts.append(journal_year)
    if pmid:
        parts.append(f"PMID: {pmid}")

    result = "\n".join(parts)

    if include_abstract and abstract:
        result += f"\n\n{abstract}"

    return result


def extract_pmids(citations: List[Any]) -> Set[str]:
    """
    Extract all PMIDs from a list of citations.

    Args:
        citations: List of Citation objects or dicts

    Returns:
        Set of PMID strings
    """
    pmids = set()
    for c in citations:
        pmid = get_attr(c, 'pmid')
        if pmid:
            pmids.add(str(pmid))
    return pmids


def filter_by_pmids(citations: List[Any], pmids: Set[str]) -> List[Any]:
    """
    Filter citations to only those with PMIDs in the given set.

    Args:
        citations: List of Citation objects or dicts
        pmids: Set of PMIDs to keep

    Returns:
        Filtered list of citations
    """
    return [c for c in citations if get_attr(c, 'pmid') in pmids]


def get_citation_by_pmid(citations: List[Any], pmid: str) -> Optional[Any]:
    """
    Find a citation by PMID.

    Args:
        citations: List of Citation objects or dicts
        pmid: PMID to find

    Returns:
        Citation if found, None otherwise
    """
    for c in citations:
        if get_attr(c, 'pmid') == pmid:
            return c
    return None


def merge_citation_lists(*citation_lists: List[Any]) -> List[Any]:
    """
    Merge multiple citation lists, removing duplicates by PMID.

    Args:
        *citation_lists: Variable number of citation lists

    Returns:
        Merged list with duplicates removed (first occurrence kept)
    """
    seen_pmids = set()
    merged = []

    for citations in citation_lists:
        if not citations:
            continue
        for c in citations:
            pmid = get_attr(c, 'pmid')
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                merged.append(c)

    return merged


# Backward compatibility alias
get_citation_attr = get_attr
citation_to_dict = to_dict
