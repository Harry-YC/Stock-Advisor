"""
Mark Pen UI Component

Allows users to highlight/mark text in expert responses and search results
to improve RAG quality over time.

Usage:
    from ui.mark_pen import render_markable_text, render_mark_sidebar

    # Render text with mark button
    render_markable_text(
        text="The mortality rate was 15%",
        source_type="expert_response",
        source_id="Surgical Oncologist",
        question_context="surgical bypass outcomes"
    )

    # Show marks in sidebar
    render_mark_sidebar(project_id="project_123")
"""

import streamlit as st
from datetime import datetime
from typing import Optional, List, Dict

from services.feedback_service import (
    FeedbackService,
    save_mark,
    get_marks,
    FeedbackMark,
)


# Mark type configuration (Kindle-inspired colors)
MARK_TYPES = {
    'important_data': {'icon': '📊', 'label': 'Important', 'color': '#10B981', 'bg': '#D1FAE5'},
    'key_finding': {'icon': '⭐', 'label': 'Key Finding', 'color': '#F59E0B', 'bg': '#FEF3C7'},
    'evidence_gap': {'icon': '🔍', 'label': 'Gap', 'color': '#6366F1', 'bg': '#EEF2FF'},
    'citation_useful': {'icon': '📚', 'label': 'Good Cite', 'color': '#8B5CF6', 'bg': '#EDE9FE'},
    'disagree': {'icon': '❌', 'label': 'Disagree', 'color': '#EF4444', 'bg': '#FEE2E2'},
    'agree': {'icon': '✓', 'label': 'Agree', 'color': '#10B981', 'bg': '#D1FAE5'},
}


def render_mark_button(
    text: str,
    source_type: str,
    source_id: str,
    question_context: str,
    key_suffix: str = "",
    project_id: Optional[str] = None,
    compact: bool = True,
    inline: bool = False,
) -> Optional[FeedbackMark]:
    """
    Render a mark button for a piece of text.

    Args:
        text: The text that can be marked
        source_type: Type of source (expert_response, search_result, etc.)
        source_id: Source identifier (expert name, PMID, etc.)
        question_context: The question being researched
        key_suffix: Unique suffix for Streamlit keys
        project_id: Optional project ID
        compact: If True, show compact button; if False, show expanded options
        inline: If True, just render the button (no column layout)

    Returns:
        FeedbackMark if user marked the text, None otherwise
    """
    unique_key = f"mark_{source_id}_{key_suffix}_{hash(text[:50])}"

    if compact:
        # Show only pen icon button - subtle style
        if inline:
            # Just the button, caller handles layout
            if st.button("🖊️", key=f"btn_{unique_key}", help="Mark this passage"):
                st.session_state[f'show_mark_options_{unique_key}'] = True
                st.rerun()
        else:
            # Column layout
            col1, col2 = st.columns([10, 1])
            with col2:
                if st.button("🖊️", key=f"btn_{unique_key}", help="Mark this passage"):
                    st.session_state[f'show_mark_options_{unique_key}'] = True
                    st.rerun()

    # Show mark options if triggered (Kindle-style color picker)
    if st.session_state.get(f'show_mark_options_{unique_key}'):
        # Preview of text being marked
        preview = text[:80] + "..." if len(text) > 80 else text

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
                    padding: 12px 16px; border-radius: 8px; color: white; margin: 8px 0;">
            <div style="font-size: 12px; opacity: 0.9; margin-bottom: 4px;">Mark as:</div>
            <div style="font-size: 13px; font-style: italic; opacity: 0.8;">"{preview}"</div>
        </div>
        """, unsafe_allow_html=True)

        # Optional feedback/note input (Kindle-style annotation)
        user_note = st.text_input(
            "Add a note (optional)",
            key=f"note_{unique_key}",
            placeholder="Why is this important? Any corrections?",
            label_visibility="collapsed"
        )

        # Mark type buttons in 2 rows of 3 for better readability
        row1_types = ['important_data', 'key_finding', 'evidence_gap']
        row2_types = ['citation_useful', 'agree', 'disagree']

        # Row 1
        cols1 = st.columns(3)
        for i, mark_type in enumerate(row1_types):
            config = MARK_TYPES[mark_type]
            with cols1[i]:
                if st.button(
                    f"{config['icon']} {config['label']}",
                    key=f"type_{mark_type}_{unique_key}",
                    use_container_width=True
                ):
                    mark = save_mark(
                        text=text,
                        source_type=source_type,
                        source_id=source_id,
                        mark_type=mark_type,
                        question_context=question_context,
                        project_id=project_id,
                        user_note=user_note if user_note else None,
                    )
                    st.session_state[f'show_mark_options_{unique_key}'] = False
                    st.toast(f"{config['icon']} Marked as {config['label']}", icon=config['icon'])
                    return mark

        # Row 2
        cols2 = st.columns(3)
        for i, mark_type in enumerate(row2_types):
            config = MARK_TYPES[mark_type]
            with cols2[i]:
                if st.button(
                    f"{config['icon']} {config['label']}",
                    key=f"type_{mark_type}_{unique_key}",
                    use_container_width=True
                ):
                    mark = save_mark(
                        text=text,
                        source_type=source_type,
                        source_id=source_id,
                        mark_type=mark_type,
                        question_context=question_context,
                        project_id=project_id,
                        user_note=user_note if user_note else None,
                    )
                    st.session_state[f'show_mark_options_{unique_key}'] = False
                    st.toast(f"{config['icon']} Marked as {config['label']}", icon=config['icon'])
                    return mark

        # Cancel button
        if st.button("Cancel", key=f"cancel_{unique_key}", type="secondary"):
            st.session_state[f'show_mark_options_{unique_key}'] = False
            st.rerun()

    return None


def render_markable_text(
    text: str,
    source_type: str,
    source_id: str,
    question_context: str,
    key_suffix: str = "",
    project_id: Optional[str] = None,
    show_full: bool = True,
) -> None:
    """
    Render text with an inline mark button.

    Args:
        text: The text to display
        source_type: Type of source
        source_id: Source identifier
        question_context: Question context
        key_suffix: Unique key suffix
        project_id: Optional project ID
        show_full: If True, show full text; if False, truncate
    """
    unique_key = f"markable_{source_id}_{key_suffix}_{hash(text[:30])}"

    # Display text
    display_text = text if show_full else (text[:200] + "..." if len(text) > 200 else text)

    # Two-column layout: text | mark button
    col1, col2 = st.columns([15, 1])

    with col1:
        st.markdown(display_text)

    with col2:
        if st.button("🖊️", key=f"mark_{unique_key}", help="Mark this passage"):
            st.session_state[f'marking_{unique_key}'] = True
            st.rerun()

    # Show mark type selector if triggered
    if st.session_state.get(f'marking_{unique_key}'):
        _render_mark_type_selector(
            text=text,
            source_type=source_type,
            source_id=source_id,
            question_context=question_context,
            project_id=project_id,
            unique_key=unique_key,
        )


def _render_mark_type_selector(
    text: str,
    source_type: str,
    source_id: str,
    question_context: str,
    project_id: Optional[str],
    unique_key: str,
) -> None:
    """Render the mark type selector popup."""

    st.markdown("""
    <div style="background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
                padding: 12px 16px; border-radius: 8px; color: white; margin: 8px 0;">
        <strong>Mark this text as:</strong>
    </div>
    """, unsafe_allow_html=True)

    # Show text preview
    preview = text[:100] + "..." if len(text) > 100 else text
    st.caption(f'"{preview}"')

    # Mark type buttons in a row
    cols = st.columns(6)
    for i, (mark_type, config) in enumerate(MARK_TYPES.items()):
        with cols[i]:
            if st.button(
                f"{config['icon']}",
                key=f"select_{mark_type}_{unique_key}",
                help=config['label'],
                use_container_width=True
            ):
                # Save the mark
                save_mark(
                    text=text,
                    source_type=source_type,
                    source_id=source_id,
                    mark_type=mark_type,
                    question_context=question_context,
                    project_id=project_id,
                )

                # Clear state
                st.session_state[f'marking_{unique_key}'] = False
                st.toast(f"Marked as {config['label']}!", icon=config['icon'])
                st.rerun()

    # Cancel
    if st.button("Cancel", key=f"cancel_mark_{unique_key}", type="secondary"):
        st.session_state[f'marking_{unique_key}'] = False
        st.rerun()


def render_mark_sidebar(project_id: Optional[str] = None) -> None:
    """
    Render marks summary in sidebar (Kindle "My Clippings" style).

    Args:
        project_id: Filter marks by project
    """
    service = FeedbackService()
    stats = service.get_mark_stats(project_id=project_id)

    if stats['total'] == 0:
        # Minimal placeholder when no marks
        st.sidebar.markdown("---")
        st.sidebar.caption("🖊️ No marks yet")
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 🖊️ Your Marks ({stats['total']})")

    # By type breakdown with expandable previews
    marks = get_marks(project_id=project_id, limit=50)

    # Group by type
    by_type: Dict[str, List] = {}
    for mark in marks:
        mt = mark.mark_type
        if mt not in by_type:
            by_type[mt] = []
        by_type[mt].append(mark)

    # Display each type as expandable section
    for mark_type, type_marks in by_type.items():
        config = MARK_TYPES.get(mark_type, {'icon': '📝', 'label': mark_type, 'color': '#666', 'bg': '#F3F4F6'})
        count = len(type_marks)

        with st.sidebar.expander(f"{config['icon']} {config.get('label', mark_type)} ({count})", expanded=False):
            for mark in type_marks[:5]:  # Show max 5 per type
                # Truncate text for sidebar
                text_preview = mark.text[:60] + "..." if len(mark.text) > 60 else mark.text
                st.markdown(f"""
                <div style="background: {config.get('bg', '#F3F4F6')}; padding: 6px 10px;
                            border-radius: 4px; margin-bottom: 6px; font-size: 12px;
                            border-left: 3px solid {config.get('color', '#666')};">
                    <div style="color: #374151;">"{text_preview}"</div>
                    <div style="color: #9CA3AF; font-size: 10px; margin-top: 2px;">
                        — {mark.source_id}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if count > 5:
                st.caption(f"+{count - 5} more")

    # View all / Export buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("View All", key="view_all_marks", use_container_width=True):
            st.session_state['show_marks_panel'] = True
    with col2:
        # Export as markdown
        export_text = _format_marks_for_export(marks)
        st.download_button(
            "Export",
            data=export_text,
            file_name="my_marks.md",
            mime="text/markdown",
            key="export_marks",
            use_container_width=True
        )


def _format_marks_for_export(marks: List) -> str:
    """Format marks as markdown for export."""
    lines = ["# My Marked Passages\n"]
    lines.append(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Group by type
    by_type: Dict[str, List] = {}
    for mark in marks:
        mt = mark.mark_type
        if mt not in by_type:
            by_type[mt] = []
        by_type[mt].append(mark)

    for mark_type, type_marks in by_type.items():
        config = MARK_TYPES.get(mark_type, {'icon': '📝', 'label': mark_type})
        lines.append(f"\n## {config['icon']} {config.get('label', mark_type)}\n")

        for mark in type_marks:
            lines.append(f"- **{mark.source_id}**: \"{mark.text[:200]}{'...' if len(mark.text) > 200 else ''}\"")
            if mark.user_note:
                lines.append(f"  - *Note:* {mark.user_note}")
            lines.append(f"  - Context: {mark.question_context[:100]}...")
            lines.append("")

    return "\n".join(lines)


def render_marks_panel(project_id: Optional[str] = None) -> None:
    """
    Render a panel showing all marks with delete option.

    Args:
        project_id: Filter marks by project
    """
    if not st.session_state.get('show_marks_panel'):
        return

    st.markdown("---")
    st.markdown("### 🖊️ Your Marked Passages")

    marks = get_marks(project_id=project_id, limit=50)

    if not marks:
        st.info("No marks yet. Use the 🖊️ button to mark important text in expert responses.")
        if st.button("Close", key="close_marks_panel"):
            st.session_state['show_marks_panel'] = False
            st.rerun()
        return

    # Group by mark type
    by_type: Dict[str, List[FeedbackMark]] = {}
    for mark in marks:
        mt = mark.mark_type
        if mt not in by_type:
            by_type[mt] = []
        by_type[mt].append(mark)

    # Display by type
    for mark_type, type_marks in by_type.items():
        config = MARK_TYPES.get(mark_type, {'icon': '📝', 'label': mark_type, 'color': '#666', 'bg': '#F3F4F6'})

        with st.expander(f"{config['icon']} {config.get('label', mark_type)} ({len(type_marks)})", expanded=False):
            for mark in type_marks:
                col1, col2 = st.columns([10, 1])

                with col1:
                    # Text preview
                    text_preview = mark.text[:150] + "..." if len(mark.text) > 150 else mark.text

                    # Build note HTML if present
                    note_html = ""
                    if mark.user_note:
                        note_html = f"""
                        <div style="font-size: 12px; color: #6366F1; margin-top: 4px;
                                    padding: 4px 8px; background: #EEF2FF; border-radius: 4px;">
                            💬 {mark.user_note}
                        </div>
                        """

                    st.markdown(f"""
                    <div style="background: {config.get('bg', '#F9FAFB')}; padding: 8px 12px; border-radius: 4px;
                                border-left: 3px solid {config['color']}; margin-bottom: 8px;">
                        <div style="font-size: 13px; color: #374151;">"{text_preview}"</div>
                        {note_html}
                        <div style="font-size: 11px; color: #9CA3AF; margin-top: 4px;">
                            — {mark.source_id} | {mark.created_at[:10]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    if st.button("🗑️", key=f"delete_{mark.id}", help="Delete this mark"):
                        service = FeedbackService()
                        service.delete_mark(mark.id)
                        st.rerun()

    # Close button
    if st.button("Close Panel", key="close_marks_panel_btn"):
        st.session_state['show_marks_panel'] = False
        st.rerun()


def render_expert_response_with_mark(
    expert_name: str,
    content: str,
    question_context: str,
    project_id: Optional[str] = None,
    key_suffix: str = "",
) -> None:
    """
    Render an expert response with mark buttons on key passages.

    This splits the content into paragraphs and adds mark buttons.

    Args:
        expert_name: Name of the expert
        content: Response content
        question_context: The question being researched
        project_id: Optional project ID
        key_suffix: Unique key suffix
    """
    from ui.citation_utils import format_expert_response

    # Header with expert name and mark-all button
    col1, col2 = st.columns([8, 2])
    with col1:
        st.markdown(f"**{expert_name}**")
    with col2:
        if st.button(
            "🖊️ Mark Passage",
            key=f"mark_all_{expert_name}_{key_suffix}",
            help="Select text to mark",
            type="secondary",
        ):
            st.session_state[f'marking_expert_{expert_name}_{key_suffix}'] = True

    # Format content with highlighting
    formatted = format_expert_response(content)
    st.markdown(formatted, unsafe_allow_html=True)

    # Show mark selector if triggered
    if st.session_state.get(f'marking_expert_{expert_name}_{key_suffix}'):
        # Text selection (simplified - mark full response)
        st.markdown("---")
        st.info("Select what to mark from this response:")

        # Split into sentences for selection
        import re
        sentences = re.split(r'(?<=[.!?])\s+', content)[:10]  # Limit to 10 sentences

        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 20:
                continue

            col1, col2 = st.columns([12, 1])
            with col1:
                st.markdown(f"• {sentence[:100]}...")
            with col2:
                if st.button("✓", key=f"select_sent_{expert_name}_{i}_{key_suffix}"):
                    st.session_state[f'selected_text_{expert_name}_{key_suffix}'] = sentence
                    st.session_state[f'show_mark_type_{expert_name}_{key_suffix}'] = True
                    st.rerun()

        if st.button("Cancel", key=f"cancel_expert_mark_{expert_name}_{key_suffix}"):
            st.session_state[f'marking_expert_{expert_name}_{key_suffix}'] = False
            st.rerun()

    # Show mark type selector if text selected
    if st.session_state.get(f'show_mark_type_{expert_name}_{key_suffix}'):
        selected_text = st.session_state.get(f'selected_text_{expert_name}_{key_suffix}', '')
        _render_mark_type_selector(
            text=selected_text,
            source_type="expert_response",
            source_id=expert_name,
            question_context=question_context,
            project_id=project_id,
            unique_key=f"expert_{expert_name}_{key_suffix}",
        )

    st.markdown("---")
