"""
Expert Panel UI Module

Modular components for the Expert Panel Discussion interface.

This package contains:
- main: Main render_expert_panel entry point
- expert_chat: Interactive Q&A with experts
- feedback_loop: Human feedback collection and processing
- hypothesis_tracker: Hypothesis extraction and tracking
- debate_mode: Collaborative debate between experts
- analysis_display: Gap analysis and conflict detection display

Usage:
    from ui.expert_panel import render_expert_panel

    # Or for individual components:
    from ui.expert_panel import (
        render_expert_chat,
        render_feedback_loop,
        render_hypothesis_tracker,
        render_debate_section,
        render_gap_analysis,
        render_conflict_detection,
    )
"""

# Main entry point - this is what the app imports
from ui.expert_panel.main import render_expert_panel

# Component exports
from ui.expert_panel.expert_chat import render_expert_chat
from ui.expert_panel.feedback_loop import render_feedback_loop
from ui.expert_panel.hypothesis_tracker import render_hypothesis_tracker
from ui.expert_panel.debate_mode import render_debate_section
from ui.expert_panel.analysis_display import (
    render_gap_analysis,
    render_conflict_detection,
)

__all__ = [
    # Main entry point
    'render_expert_panel',
    # Components
    'render_expert_chat',
    'render_feedback_loop',
    'render_hypothesis_tracker',
    'render_debate_section',
    'render_gap_analysis',
    'render_conflict_detection',
]
