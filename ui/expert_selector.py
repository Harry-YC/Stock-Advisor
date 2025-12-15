"""
Expert Selection Grid for GDG Panel

Category-organized selection with preset configurations.
Ported from Virtual_Team/ui/simulation/persona_selector.py

Features:
- Category headers with colored backgrounds
- Preset buttons for quick configuration
- Expert cards with toggle selection
- Min/max constraints (2-12 experts)
- Persistent selection across tab switches
"""

import streamlit as st
from typing import List, Optional, Dict

from gdg.gdg_personas import (
    GDG_PERSONAS,
    GDG_CATEGORIES,
    GDG_CATEGORY_COLORS,
    GDG_PRESETS,
    get_experts_by_category,
    get_category_colors,
    get_all_preset_names,
    get_preset_experts,
    get_preset_info,
    get_default_expert_selection,
)


def _render_selection_status(
    count: int,
    min_experts: int,
    max_experts: int
) -> None:
    """Render selection status message."""
    if count < min_experts:
        st.warning(f"Select at least {min_experts} experts to continue ({count} selected)")
    elif count >= max_experts:
        st.info(f"Maximum {max_experts} experts reached ({count} selected)")
    else:
        st.caption(f"{count} of {max_experts} experts selected")


def _render_expert_card(
    expert_name: str,
    is_selected: bool,
    colors: Dict[str, str],
    key_prefix: str
) -> bool:
    """
    Render a single expert card.

    Returns:
        True if clicked (selection toggled)
    """
    config = GDG_PERSONAS.get(expert_name, {})
    role = config.get("role", expert_name)
    specialty = config.get("specialty", "")

    # Card styling based on selection state
    if is_selected:
        border_color = colors["border"]
        bg_color = colors["bg"]
        opacity = "1.0"
        check_mark = "✓ "
    else:
        border_color = "#e0e0e0"
        bg_color = "#ffffff"
        opacity = "0.7"
        check_mark = ""

    # Render card with custom HTML
    card_html = f"""
    <div style="
        border: 2px solid {border_color};
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        background: {bg_color};
        opacity: {opacity};
        cursor: pointer;
        transition: all 0.2s;
    ">
        <div style="font-weight: 600; font-size: 0.95em; margin-bottom: 4px;">
            {check_mark}{expert_name}
        </div>
        <div style="font-size: 0.8em; color: #666;">
            {specialty[:60]}{'...' if len(specialty) > 60 else ''}
        </div>
    </div>
    """

    # Use button for interactivity
    col_key = f"{key_prefix}_{expert_name.replace(' ', '_')}"
    clicked = st.button(
        f"{'✓ ' if is_selected else ''}{expert_name}",
        key=col_key,
        use_container_width=True,
        type="primary" if is_selected else "secondary"
    )

    return clicked


def render_expert_selector(
    key_prefix: str = "gdg_expert",
    default_selection: Optional[List[str]] = None,
    max_experts: int = 12,
    min_experts: int = 2,
    show_presets: bool = True
) -> List[str]:
    """
    Render a category-organized grid of expert cards for selection with preset configurations.

    Args:
        key_prefix: Prefix for session state keys
        default_selection: Default experts to pre-select
        max_experts: Maximum number of experts allowed
        min_experts: Minimum number of experts required
        show_presets: Whether to show preset buttons

    Returns:
        List of selected expert names
    """
    # Session state key for selection
    selection_key = f"{key_prefix}_selection"

    # Initialize selection
    if selection_key not in st.session_state:
        if default_selection:
            st.session_state[selection_key] = list(default_selection)
        else:
            st.session_state[selection_key] = get_default_expert_selection()

    selected: List[str] = st.session_state[selection_key]

    # Preset buttons
    if show_presets:
        st.markdown("**Quick Presets**")
        preset_names = get_all_preset_names()
        preset_cols = st.columns(len(preset_names))

        for i, preset_name in enumerate(preset_names):
            with preset_cols[i]:
                preset_info = get_preset_info(preset_name)
                preset_experts = get_preset_experts(preset_name)

                # Check if this preset is currently active
                is_active = set(selected) == set(preset_experts)

                if st.button(
                    preset_name,
                    key=f"{key_prefix}_preset_{preset_name.replace(' ', '_')}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                    help=preset_info.get("focus", "") if preset_info else ""
                ):
                    st.session_state[selection_key] = list(preset_experts)
                    st.rerun()

        st.divider()

    # Selection status
    _render_selection_status(len(selected), min_experts, max_experts)

    # Category-organized grid
    categories = get_experts_by_category()
    colors = get_category_colors()

    for category_name, expert_names in categories.items():
        category_colors = colors.get(category_name, {
            "border": "#e0e0e0",
            "bg": "#f5f5f5",
            "header": "#333"
        })

        # Category header
        st.markdown(
            f"""<div style="
                background: {category_colors['header']};
                color: white;
                padding: 8px 12px;
                border-radius: 6px 6px 0 0;
                font-weight: 600;
                margin-top: 12px;
            ">{category_name}</div>""",
            unsafe_allow_html=True
        )

        # Expert cards in 2-column grid
        cols = st.columns(2)
        for i, expert_name in enumerate(expert_names):
            with cols[i % 2]:
                is_selected = expert_name in selected
                can_select = len(selected) < max_experts or is_selected

                # Render button for expert
                if st.button(
                    f"{'✓ ' if is_selected else ''}{expert_name}",
                    key=f"{key_prefix}_card_{expert_name.replace(' ', '_')}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                    disabled=not can_select and not is_selected
                ):
                    # Toggle selection
                    if is_selected:
                        if expert_name in selected:
                            selected.remove(expert_name)
                    else:
                        if can_select:
                            selected.append(expert_name)

                    st.session_state[selection_key] = selected
                    st.rerun()

                # Show role below button
                config = GDG_PERSONAS.get(expert_name, {})
                specialty = config.get("specialty", "")
                if specialty:
                    st.caption(specialty[:50] + ("..." if len(specialty) > 50 else ""))

    return selected


def render_expert_selector_compact(
    key_prefix: str = "gdg_expert_compact",
    default_selection: Optional[List[str]] = None,
    max_experts: int = 12,
    min_experts: int = 2
) -> List[str]:
    """
    Render a compact multiselect-based expert selector.

    Use this for simpler UIs where the full grid isn't needed.

    Args:
        key_prefix: Prefix for session state keys
        default_selection: Default experts to pre-select
        max_experts: Maximum number of experts allowed
        min_experts: Minimum number of experts required

    Returns:
        List of selected expert names
    """
    # Session state key for selection
    selection_key = f"{key_prefix}_selection"

    # Initialize selection
    if selection_key not in st.session_state:
        if default_selection:
            st.session_state[selection_key] = list(default_selection)
        else:
            st.session_state[selection_key] = get_default_expert_selection()

    all_experts = list(GDG_PERSONAS.keys())
    default = st.session_state[selection_key]

    # Preset selector
    preset_names = ["Custom"] + get_all_preset_names()

    # Check which preset matches current selection
    current_preset = "Custom"
    for preset_name in get_all_preset_names():
        if set(default) == set(get_preset_experts(preset_name)):
            current_preset = preset_name
            break

    col1, col2 = st.columns([1, 3])

    with col1:
        selected_preset = st.selectbox(
            "Preset",
            options=preset_names,
            index=preset_names.index(current_preset),
            key=f"{key_prefix}_preset_select"
        )

        if selected_preset != "Custom" and selected_preset != current_preset:
            st.session_state[selection_key] = get_preset_experts(selected_preset)
            st.rerun()

    with col2:
        selected = st.multiselect(
            "Select Experts",
            options=all_experts,
            default=default,
            max_selections=max_experts,
            key=f"{key_prefix}_multiselect",
            help=f"Select {min_experts}-{max_experts} experts"
        )

        st.session_state[selection_key] = selected

    # Validation
    if len(selected) < min_experts:
        st.warning(f"Please select at least {min_experts} experts")

    return selected


def render_expert_chips(
    selected_experts: List[str],
    key_prefix: str = "expert_chips",
    on_change: Optional[callable] = None
) -> None:
    """
    Render expert selection as clickable chips/pills.

    Args:
        selected_experts: Currently selected expert names
        key_prefix: Prefix for keys
        on_change: Callback when selection changes
    """
    categories = get_experts_by_category()
    colors = get_category_colors()

    st.caption(f"{len(selected_experts)} experts selected")

    # Flatten all experts with their categories
    all_experts = []
    for category, experts in categories.items():
        for expert in experts:
            all_experts.append((expert, category))

    # Render as pills using st.pills (Streamlit 1.33+)
    try:
        new_selection = st.pills(
            "Experts",
            options=[e[0] for e in all_experts],
            default=selected_experts,
            selection_mode="multi",
            key=f"{key_prefix}_pills"
        )

        if new_selection != selected_experts and on_change:
            on_change(new_selection)

    except AttributeError:
        # Fallback for older Streamlit versions
        st.multiselect(
            "Experts",
            options=[e[0] for e in all_experts],
            default=selected_experts,
            key=f"{key_prefix}_multiselect_fallback"
        )
