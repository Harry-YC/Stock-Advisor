"""
E2E Tests for Palliative Surgery GDG

This package contains end-to-end Playwright tests for the Streamlit application.

Test modules:
- test_citation_highlighting.py - P1: Citation highlighting and cards
- test_quick_answer.py - P4: Quick Q&A mode
- test_challenger.py - P3: Red team challenger
- test_smart_suggestions.py - P5: Smart follow-up suggestions
- test_mark_pen.py - P6: Mark pen for RAG improvement
- test_full_flow.py - Integration test using Mirels Score 9 question

Usage:
    # Run all E2E tests
    TEST_URL=http://localhost:8501 pytest tests/e2e/ -v

    # Run specific feature test
    TEST_URL=http://localhost:8501 pytest tests/e2e/test_citation_highlighting.py -v

    # Run with visible browser
    HEADLESS=false TEST_URL=http://localhost:8501 pytest tests/e2e/ -v

    # Run only unit-style tests (no browser needed)
    pytest tests/e2e/ -v -k "Service or Unit"
"""
