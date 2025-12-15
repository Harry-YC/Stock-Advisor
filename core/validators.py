"""
Identifier Validation for PubMed Citations

Validates PMIDs and DOIs for direct citation import.
Supports batch input (comma-separated or newline-separated).
"""

import re
from typing import List, Tuple


def is_valid_pmid(identifier: str) -> bool:
    """
    Validate a PubMed ID (PMID)

    Args:
        identifier: String to validate as PMID

    Returns:
        True if valid PMID format (1-8 digits)

    Examples:
        >>> is_valid_pmid("12345678")
        True
        >>> is_valid_pmid("123456789")  # Too long
        False
        >>> is_valid_pmid("12345abc")   # Non-numeric
        False
    """
    if not identifier or not isinstance(identifier, str):
        return False

    identifier = identifier.strip()

    # PMID: 1-8 digits only
    pattern = r'^\d{1,8}$'
    return bool(re.match(pattern, identifier))


def is_valid_doi(identifier: str) -> bool:
    """
    Validate a Digital Object Identifier (DOI)

    Args:
        identifier: String to validate as DOI

    Returns:
        True if valid DOI format

    Examples:
        >>> is_valid_doi("10.1038/nature12373")
        True
        >>> is_valid_doi("https://doi.org/10.1038/nature12373")
        True
        >>> is_valid_doi("http://dx.doi.org/10.1016/j.yfrne.2016.01.008")
        True
        >>> is_valid_doi("invalid")
        False
    """
    if not identifier or not isinstance(identifier, str):
        return False

    identifier = identifier.strip()

    # DOI pattern: 10.xxxx/...
    # Optionally prefixed with https://doi.org/ or http://dx.doi.org/
    pattern = r'^(https?://(dx\.)?doi\.org/)?10\.\S+/\S+$'
    return bool(re.match(pattern, identifier))


def clean_doi(doi: str) -> str:
    """
    Remove URL prefixes from DOI

    Args:
        doi: DOI string (possibly with URL prefix)

    Returns:
        Clean DOI starting with "10."

    Examples:
        >>> clean_doi("https://doi.org/10.1038/nature12373")
        "10.1038/nature12373"
        >>> clean_doi("10.1038/nature12373")
        "10.1038/nature12373"
    """
    doi = doi.strip()

    # Remove common DOI URL prefixes
    prefixes = [
        'https://doi.org/',
        'http://doi.org/',
        'https://dx.doi.org/',
        'http://dx.doi.org/',
    ]

    for prefix in prefixes:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break

    return doi


def parse_identifier_input(input_str: str) -> List[str]:
    """
    Parse user input into list of identifiers

    Handles:
    - Comma-separated: "12345678, 23456789, 34567890"
    - Newline-separated: "12345678\\n23456789\\n34567890"
    - Mixed whitespace

    Args:
        input_str: User input string with identifiers

    Returns:
        List of cleaned identifier strings
    """
    if not input_str or not isinstance(input_str, str):
        return []

    # Split by comma or newline
    identifiers = re.split(r'[,\n]', input_str)

    # Strip whitespace and remove empty strings
    identifiers = [id.strip() for id in identifiers if id.strip()]

    return identifiers


def validate_identifiers(input_str: str, id_type: str) -> Tuple[List[str], List[str]]:
    """
    Validate and parse identifier input

    Args:
        input_str: User input with identifiers (comma or newline separated)
        id_type: Either "pmid" or "doi"

    Returns:
        Tuple of (valid_identifiers, error_messages)

    Examples:
        >>> validate_identifiers("12345678, 23456789, invalid", "pmid")
        (['12345678', '23456789'], ['Invalid PMID format: invalid'])

        >>> validate_identifiers("10.1038/test, invalid", "doi")
        (['10.1038/test'], ['Invalid DOI format: invalid'])
    """
    valid = []
    errors = []

    if not input_str or not input_str.strip():
        return ([], ["No identifiers provided"])

    identifiers = parse_identifier_input(input_str)

    if not identifiers:
        return ([], ["No valid identifiers found"])

    # Select validator function
    if id_type.lower() == "pmid":
        validator = is_valid_pmid
        error_prefix = "Invalid PMID format"
    elif id_type.lower() == "doi":
        validator = is_valid_doi
        error_prefix = "Invalid DOI format"
    else:
        return ([], [f"Unknown identifier type: {id_type}"])

    # Validate each identifier
    for identifier in identifiers:
        if validator(identifier):
            # Clean DOIs before adding to valid list
            if id_type.lower() == "doi":
                valid.append(clean_doi(identifier))
            else:
                valid.append(identifier)
        else:
            errors.append(f"{error_prefix}: '{identifier}'")

    return (valid, errors)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Identifier Validators")
    print("=" * 70)

    # Test PMID validation
    print("\n1. PMID Validation")
    test_pmids = ["12345678", "123456789", "abc123", "", "1234"]
    for pmid in test_pmids:
        result = is_valid_pmid(pmid)
        print(f"   {pmid:15s} -> {result}")

    # Test DOI validation
    print("\n2. DOI Validation")
    test_dois = [
        "10.1038/nature12373",
        "https://doi.org/10.1038/nature12373",
        "http://dx.doi.org/10.1016/j.yfrne.2016.01.008",
        "invalid",
        ""
    ]
    for doi in test_dois:
        result = is_valid_doi(doi)
        print(f"   {doi:50s} -> {result}")

    # Test DOI cleaning
    print("\n3. DOI Cleaning")
    test_dois_clean = [
        "https://doi.org/10.1038/nature12373",
        "10.1038/nature12373",
        "http://dx.doi.org/10.1016/test"
    ]
    for doi in test_dois_clean:
        cleaned = clean_doi(doi)
        print(f"   {doi:50s} -> {cleaned}")

    # Test input parsing
    print("\n4. Identifier Input Parsing")
    test_inputs = [
        "12345678, 23456789, 34567890",
        "12345678\\n23456789\\n34567890",
        "12345678,   23456789  ,  34567890   "
    ]
    for input_str in test_inputs:
        parsed = parse_identifier_input(input_str)
        print(f"   Input: {repr(input_str)}")
        print(f"   Parsed: {parsed}")

    # Test batch validation
    print("\n5. Batch Validation (PMIDs)")
    test_batch_pmid = "12345678, 23456789, invalid, 34567890"
    valid, errors = validate_identifiers(test_batch_pmid, "pmid")
    print(f"   Input: {test_batch_pmid}")
    print(f"   Valid: {valid}")
    print(f"   Errors: {errors}")

    print("\n6. Batch Validation (DOIs)")
    test_batch_doi = "10.1038/test, https://doi.org/10.1016/test2, invalid"
    valid, errors = validate_identifiers(test_batch_doi, "doi")
    print(f"   Input: {test_batch_doi}")
    print(f"   Valid: {valid}")
    print(f"   Errors: {errors}")

    print("\n" + "=" * 70)
    print("✅ All validator tests completed!")
