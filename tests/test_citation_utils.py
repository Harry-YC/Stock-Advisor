"""
Unit tests for core/citation_utils.py

Tests shared citation utilities used across the application.
"""

import pytest
from dataclasses import dataclass
from typing import List, Optional

from core.citation_utils import (
    get_attr,
    to_dict,
    format_authors,
    format_citation_short,
    format_citation_full,
    extract_pmids,
    filter_by_pmids,
    get_citation_by_pmid,
    merge_citation_lists,
)


# Mock Citation class for testing
@dataclass
class MockCitation:
    """Mock citation object for testing."""
    pmid: str
    title: str = ""
    abstract: str = ""
    authors: Optional[List[str]] = None
    journal: str = ""
    year: str = ""
    doi: str = ""


class TestGetAttr:
    """Tests for get_attr function."""

    def test_get_from_object(self):
        """Test getting attribute from object."""
        citation = MockCitation(pmid="12345678", title="Test Study")
        assert get_attr(citation, "pmid") == "12345678"
        assert get_attr(citation, "title") == "Test Study"

    def test_get_from_dict(self):
        """Test getting attribute from dictionary."""
        citation = {"pmid": "12345678", "title": "Test Study"}
        assert get_attr(citation, "pmid") == "12345678"
        assert get_attr(citation, "title") == "Test Study"

    def test_default_value(self):
        """Test default value for missing attribute."""
        citation = MockCitation(pmid="12345678")
        assert get_attr(citation, "missing_field", "default") == "default"

    def test_none_citation(self):
        """Test handling None citation."""
        assert get_attr(None, "pmid", "default") == "default"

    def test_none_value_returns_default(self):
        """Test that None value returns default."""
        citation = {"pmid": None}
        assert get_attr(citation, "pmid", "default") == "default"


class TestToDict:
    """Tests for to_dict function."""

    def test_convert_object_to_dict(self):
        """Test converting object to dictionary."""
        citation = MockCitation(
            pmid="12345678",
            title="Test Study",
            authors=["Smith J", "Jones K"],
            journal="Test Journal",
            year="2024"
        )
        d = to_dict(citation)
        assert d["pmid"] == "12345678"
        assert d["title"] == "Test Study"
        assert d["authors"] == ["Smith J", "Jones K"]

    def test_pass_through_dict(self):
        """Test that dict input gets normalized with default fields."""
        citation = {"pmid": "12345678", "title": "Test"}
        d = to_dict(citation)
        # to_dict normalizes the dict with all standard fields
        assert d["pmid"] == "12345678"
        assert d["title"] == "Test"
        # Should have standard fields filled with defaults
        assert "abstract" in d
        assert "authors" in d

    def test_missing_fields_get_defaults(self):
        """Test that missing fields get default values."""
        citation = MockCitation(pmid="12345678")
        d = to_dict(citation)
        assert d["title"] == ""
        assert d["abstract"] == ""


class TestFormatAuthors:
    """Tests for format_authors function."""

    def test_single_author(self):
        """Test formatting single author."""
        authors = ["Smith J"]
        result = format_authors(authors)
        assert result == "Smith J"

    def test_two_authors(self):
        """Test formatting two authors."""
        authors = ["Smith J", "Jones K"]
        result = format_authors(authors)
        assert "Smith J" in result
        assert "Jones K" in result

    def test_many_authors_truncated(self):
        """Test that many authors are truncated."""
        authors = ["Author A", "Author B", "Author C", "Author D", "Author E"]
        result = format_authors(authors, max_authors=3)
        assert "et al" in result
        assert "Author A" in result

    def test_empty_authors(self):
        """Test handling empty author list."""
        result = format_authors([])
        assert result == "Unknown"

    def test_none_authors(self):
        """Test handling None author list."""
        result = format_authors(None)
        assert result == "Unknown"

    def test_nested_list_authors(self):
        """Test handling nested list raises TypeError (edge case)."""
        authors = [["Smith J"], "Jones K"]
        # This is an edge case - nested lists will cause a TypeError
        # The function expects List[str], not List[List[str] | str]
        # Documenting actual behavior
        with pytest.raises(TypeError):
            format_authors(authors, max_authors=3)


class TestFormatCitationShort:
    """Tests for format_citation_short function."""

    def test_format_basic(self):
        """Test basic citation formatting."""
        citation = {
            "authors": ["Smith J", "Jones K"],
            "year": "2024",
            "journal": "Test Journal"
        }
        result = format_citation_short(citation)
        assert "Smith" in result
        assert "2024" in result
        # format_citation_short returns "FirstAuthor et al. (Year)" format
        # Journal is included in format_citation_full, not short

    def test_format_with_object(self):
        """Test formatting Citation object."""
        citation = MockCitation(
            pmid="12345678",
            authors=["Smith J"],
            year="2024",
            journal="Test Journal"
        )
        result = format_citation_short(citation)
        assert "Smith" in result
        assert "2024" in result


class TestFormatCitationFull:
    """Tests for format_citation_full function."""

    def test_format_full(self):
        """Test full citation formatting."""
        citation = {
            "title": "Test Study Title",
            "authors": ["Smith J", "Jones K"],
            "year": "2024",
            "journal": "Test Journal",
            "pmid": "12345678"
        }
        result = format_citation_full(citation)
        assert "Test Study Title" in result
        assert "Smith" in result
        assert "2024" in result
        assert "12345678" in result


class TestExtractPmids:
    """Tests for extract_pmids function."""

    def test_extract_from_objects(self):
        """Test extracting PMIDs from objects."""
        citations = [
            MockCitation(pmid="11111111"),
            MockCitation(pmid="22222222"),
        ]
        pmids = extract_pmids(citations)
        assert "11111111" in pmids
        assert "22222222" in pmids

    def test_extract_from_dicts(self):
        """Test extracting PMIDs from dictionaries."""
        citations = [
            {"pmid": "11111111"},
            {"pmid": "22222222"},
        ]
        pmids = extract_pmids(citations)
        assert len(pmids) == 2

    def test_skip_missing_pmids(self):
        """Test that missing PMIDs are skipped."""
        citations = [
            {"pmid": "11111111"},
            {"title": "No PMID"},
        ]
        pmids = extract_pmids(citations)
        assert len(pmids) == 1

    def test_empty_list(self):
        """Test extracting from empty list."""
        pmids = extract_pmids([])
        assert len(pmids) == 0


class TestFilterByPmids:
    """Tests for filter_by_pmids function."""

    def test_filter_objects(self):
        """Test filtering objects by PMIDs."""
        citations = [
            MockCitation(pmid="11111111", title="Study 1"),
            MockCitation(pmid="22222222", title="Study 2"),
            MockCitation(pmid="33333333", title="Study 3"),
        ]
        filtered = filter_by_pmids(citations, {"11111111", "33333333"})
        pmids = [get_attr(c, "pmid") for c in filtered]
        assert "11111111" in pmids
        assert "33333333" in pmids
        assert "22222222" not in pmids

    def test_filter_dicts(self):
        """Test filtering dictionaries by PMIDs."""
        citations = [
            {"pmid": "11111111"},
            {"pmid": "22222222"},
        ]
        filtered = filter_by_pmids(citations, {"11111111"})
        assert len(filtered) == 1

    def test_filter_empty_pmid_set(self):
        """Test filtering with empty PMID set."""
        citations = [{"pmid": "11111111"}]
        filtered = filter_by_pmids(citations, set())
        assert len(filtered) == 0


class TestGetCitationByPmid:
    """Tests for get_citation_by_pmid function."""

    def test_find_existing(self):
        """Test finding existing citation."""
        citations = [
            {"pmid": "11111111", "title": "Study 1"},
            {"pmid": "22222222", "title": "Study 2"},
        ]
        result = get_citation_by_pmid(citations, "22222222")
        assert result is not None
        assert result["title"] == "Study 2"

    def test_not_found(self):
        """Test when PMID not found."""
        citations = [{"pmid": "11111111"}]
        result = get_citation_by_pmid(citations, "99999999")
        assert result is None

    def test_empty_list(self):
        """Test with empty citation list."""
        result = get_citation_by_pmid([], "11111111")
        assert result is None


class TestMergeCitationLists:
    """Tests for merge_citation_lists function."""

    def test_merge_unique(self):
        """Test merging lists with unique PMIDs."""
        list1 = [{"pmid": "11111111", "title": "Study 1"}]
        list2 = [{"pmid": "22222222", "title": "Study 2"}]
        merged = merge_citation_lists(list1, list2)
        assert len(merged) == 2

    def test_merge_with_duplicates(self):
        """Test merging lists with duplicate PMIDs."""
        list1 = [{"pmid": "11111111", "title": "Study 1"}]
        list2 = [{"pmid": "11111111", "title": "Study 1 Updated"}]
        merged = merge_citation_lists(list1, list2)
        assert len(merged) == 1
        # First occurrence should be kept
        assert merged[0]["title"] == "Study 1"

    def test_merge_empty_lists(self):
        """Test merging empty lists."""
        merged = merge_citation_lists([], [])
        assert len(merged) == 0

    def test_merge_multiple_lists(self):
        """Test merging multiple lists."""
        list1 = [{"pmid": "11111111"}]
        list2 = [{"pmid": "22222222"}]
        list3 = [{"pmid": "33333333"}]
        merged = merge_citation_lists(list1, list2, list3)
        assert len(merged) == 3
