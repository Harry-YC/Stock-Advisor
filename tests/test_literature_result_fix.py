"""
Test to verify LiteratureSearchResult attribute fix.

The bug: research_partner_service.py line 405 accesses .papers
but LiteratureSearchResult has .total_papers instead.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.supporting_literature_service import LiteratureSearchResult, SupportingPaper

def test_literature_search_result_attributes():
    """Test that LiteratureSearchResult has the expected attributes."""
    print("Testing LiteratureSearchResult attributes...")
    
    # Create a sample result
    result = LiteratureSearchResult()
    
    # Check expected attributes exist
    assert hasattr(result, 'total_papers'), "Missing 'total_papers' attribute"
    assert hasattr(result, 'claims_with_support'), "Missing 'claims_with_support' attribute"
    assert hasattr(result, 'claim_matches'), "Missing 'claim_matches' attribute"
    
    # Check that .papers does NOT exist (this was the bug)
    has_papers = hasattr(result, 'papers')
    print(f"  Has '.papers' attribute: {has_papers}")
    print(f"  Has '.total_papers' attribute: True")
    
    if has_papers:
        print("WARNING: .papers exists - old code might work by accident")
    else:
        print("CONFIRMED: .papers does NOT exist - code using .papers will fail")
    
    # Test the correct way to get paper count
    result.total_papers = 5
    print(f"  result.total_papers = {result.total_papers}")
    
    # Test get_all_papers() method
    paper = SupportingPaper(pmid="12345", title="Test Paper")
    result.claim_matches["test claim"] = [paper]
    all_papers = result.get_all_papers()
    print(f"  get_all_papers() returned {len(all_papers)} paper(s)")
    
    print("\nSUCCESS: All attribute checks passed!")
    print("\nFIX NEEDED: Change '.papers' to '.total_papers' in research_partner_service.py line 405")
    
    return True

if __name__ == "__main__":
    test_literature_search_result_attributes()
