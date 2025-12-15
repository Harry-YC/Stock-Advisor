"""
Literature Search UI Module

Modular components for the Literature Search interface.

This package contains:
- main: Main render_literature_search entry point
- citation_card: Individual citation display component
- results_display: Search results display with sorting and filtering
- visualizations: Timeline, network, and table visualizations
- corpus_actions: Evidence corpus include/exclude actions

Usage:
    from ui.literature_search import render_literature_search

    # Or for individual components:
    from ui.literature_search import (
        render_citation_card,
        render_results_section,
        render_visualizations,
    )
"""

# Main entry point - this is what the app imports
from ui.literature_search.main import render_literature_search

# Component exports
from ui.literature_search.citation_card import render_citation_card, render_scored_citation_card
from ui.literature_search.results_display import render_results_section
from ui.literature_search.visualizations import render_visualizations
from ui.literature_search.corpus_actions import (
    render_corpus_status_badge,
    render_corpus_action_buttons,
)

__all__ = [
    # Main entry point
    'render_literature_search',
    # Components
    'render_citation_card',
    'render_scored_citation_card',
    'render_results_section',
    'render_visualizations',
    'render_corpus_status_badge',
    'render_corpus_action_buttons',
]
