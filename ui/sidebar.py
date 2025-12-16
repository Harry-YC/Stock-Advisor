"""
Sidebar UI Module for Travel Planner

Handles the sidebar rendering including:
- Project Manager (select/create trip projects)
- Export functionality
- App settings
"""

import streamlit as st
from datetime import datetime
from config import settings


def render_sidebar(project_dao):
    """
    Render the sidebar for Travel Planner.

    Args:
        project_dao: ProjectDAO instance for project operations
    """
    with st.sidebar:
        # Header
        st.title(f"{settings.APP_ICON} {settings.APP_TITLE}")

        # Project Selector
        _render_project_manager(project_dao)

        # Export
        st.markdown("---")
        _render_export_section()

        # API Status
        st.markdown("---")
        _render_api_status()


def _render_project_manager(project_dao):
    """Render project selection and creation."""
    st.markdown("### My Trips")

    # Get all projects from database
    all_projects = project_dao.get_all_projects()
    project_names = [p.name for p in all_projects]

    current_idx = 0
    if st.session_state.get('current_project_name') in project_names:
        current_idx = project_names.index(st.session_state.current_project_name) + 1

    selected_project = st.selectbox(
        "Select Trip",
        options=["-- New Trip --"] + project_names,
        index=current_idx,
        label_visibility="collapsed"
    )

    if selected_project == "-- New Trip --":
        new_name = st.text_input(
            "Trip Name",
            placeholder="e.g., Paris Summer 2025",
            label_visibility="collapsed",
            key="new_project_name_input"
        )
        if st.button("Create Trip", use_container_width=True, key="create_project_btn"):
            if new_name and new_name.strip():
                existing = project_dao.get_project_by_name(new_name.strip())
                if existing:
                    st.warning(f"Trip '{new_name}' already exists")
                else:
                    try:
                        pid = project_dao.create_project(new_name.strip(), "")
                        st.session_state.current_project_id = pid
                        st.session_state.current_project_name = new_name.strip()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create trip: {e}")
    elif selected_project != st.session_state.get('current_project_name'):
        project = project_dao.get_project_by_name(selected_project)
        if project:
            st.session_state.current_project_id = project.id
            st.session_state.current_project_name = project.name
            st.rerun()


def _render_export_section():
    """Render export options."""
    st.markdown("### Export")

    research_result = st.session_state.get('research_result')

    if not research_result:
        st.caption("Plan a trip first to enable export")
        return

    # Excel Export
    if st.button("📊 Export to Excel", use_container_width=True):
        try:
            from services.excel_export_service import export_travel_plan_to_excel

            excel_buffer = export_travel_plan_to_excel(
                question=research_result.get('question', ''),
                recommendation=research_result.get('recommendation', ''),
                expert_responses=research_result.get('expert_responses', {})
            )

            st.download_button(
                label="Download Excel",
                data=excel_buffer,
                file_name=f"trip_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Export failed: {e}")

    # Markdown Export
    if st.button("📝 Export to Markdown", use_container_width=True):
        md_content = _generate_markdown_export(research_result)
        st.download_button(
            label="Download Markdown",
            data=md_content,
            file_name=f"trip_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )


def _generate_markdown_export(research_result):
    """Generate markdown export of trip plan."""
    lines = [
        "# Travel Plan",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Trip Details",
        f"> {research_result.get('question', 'No question')}",
        "",
        "## Recommendation",
        research_result.get('recommendation', 'No recommendation'),
        "",
    ]

    expert_responses = research_result.get('expert_responses', {})
    if expert_responses:
        lines.append("## Expert Insights")
        for expert, response in expert_responses.items():
            lines.append(f"### {expert}")
            content = response.get('content', '') if isinstance(response, dict) else str(response)
            lines.append(content[:2000])
            lines.append("")

    lines.extend([
        "---",
        "*Generated by Travel Planner v1.0*"
    ])

    return "\n".join(lines)


def _render_api_status():
    """Render API connection status."""
    st.markdown("### Status")

    # Gemini API
    if settings.GEMINI_API_KEY:
        st.success("✓ Gemini API")
    else:
        st.error("✗ Gemini API Key Missing")

    # Places API (for ratings/reviews)
    if settings.ENABLE_PLACES_API:
        st.success("✓ Places API")
    else:
        st.caption("○ Places API (optional)")

    # Weather API
    if settings.ENABLE_WEATHER_API:
        st.success("✓ Weather API")
    else:
        st.caption("○ Weather API (optional)")

    # Flight API
    if settings.ENABLE_FLIGHT_SEARCH:
        st.success("✓ Flight API")
    else:
        st.caption("○ Flight API (optional)")
