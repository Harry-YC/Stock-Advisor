"""
Travel Planner Home - Plan Your Trip (v1.0)

The conversational interface for the Travel Planner platform.
Question-first workflow: ask a travel question → search info → consult experts → recommend.

State Router: home → processing → answer
"""

import streamlit as st
from datetime import datetime, date, timedelta
from typing import Optional, Dict

from config import settings
from core.state_manager import reset_conversational_state
from core.database import ProjectDAO, ProgramProfileDAO, DatabaseManager
from core.utils import user_friendly_error
from services.research_partner_service import ResearchPartnerService, ResearchResult
from ui.answer_view import render_answer_view
from ui.evidence_drawer import render_evidence_drawer
from travel.travel_personas import TRAVEL_EXPERTS, TRAVEL_CATEGORIES, TRAVEL_PRESETS
from travel.travel_templates import TRAVEL_QUESTION_TYPES, detect_travel_question_type


# Centered container styling
_HOME_STYLES = """
<style>
.main-input-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 1rem;
}
.centered-header {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
}
.centered-header h1 {
    font-size: 2.25rem;
    margin-bottom: 0.5rem;
    font-weight: 600;
}
.centered-header p {
    color: #666;
    font-size: 1.1rem;
}
.recent-question-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: background 0.2s;
}
.recent-question-card:hover {
    background: #f5f5f5;
}
div[data-testid="stFormSubmitButton"] button {
    background-color: #1F77B4 !important;
    border-color: #1F77B4 !important;
    color: white !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #165d8f !important;
    border-color: #165d8f !important;
}

/* Expert pills styling - unselected: white, selected: blue */
/* Unselected pills */
button[data-testid="stBaseButton-pills"] {
    background-color: white !important;
    border: 1px solid #d1d5db !important;
    color: #374151 !important;
    border-radius: 1rem !important;
    padding: 0.25rem 0.75rem !important;
    font-size: 0.85rem !important;
    transition: all 0.15s ease !important;
}
button[data-testid="stBaseButton-pills"]:hover {
    background-color: #f3f4f6 !important;
    border-color: #9ca3af !important;
}
button[data-testid="stBaseButton-pills"] p {
    color: #374151 !important;
}

/* Selected/Active pills - light blue */
button[data-testid="stBaseButton-pillsActive"] {
    background-color: #dbeafe !important;
    border: 1px solid #93c5fd !important;
    color: #1e40af !important;
    border-radius: 1rem !important;
    padding: 0.25rem 0.75rem !important;
    font-size: 0.85rem !important;
    transition: all 0.15s ease !important;
}
button[data-testid="stBaseButton-pillsActive"]:hover {
    background-color: #bfdbfe !important;
    border-color: #60a5fa !important;
}
button[data-testid="stBaseButton-pillsActive"] p {
    color: #1e40af !important;
}

/* Toggle switch - light blue instead of red */
/* Target the track div inside the checkbox label */
label[data-baseweb="checkbox"] > div:first-child {
    background-color: #d1d5db !important;
}
label[data-baseweb="checkbox"]:has(input:checked) > div:first-child {
    background-color: #3b82f6 !important;
}
/* Target the thumb/knob */
label[data-baseweb="checkbox"] > div:first-child > div {
    background-color: white !important;
}

/* Primary button (Research) - white text */
button[kind="primary"] {
    color: white !important;
}
button[kind="primary"] p {
    color: white !important;
}

/* Hide "Press Enter to apply" hint from text inputs */
div[data-testid="InputInstructions"] {
    display: none !important;
}
.stTextArea [data-testid="InputInstructions"],
.stTextInput [data-testid="InputInstructions"] {
    display: none !important;
}
/* Hide the Press Cmd+Enter hint (Streamlit emotion cache classes) */
.stTextArea .e1gk92lc2,
.stTextInput .e1gk92lc2,
div[class*="e1gk92lc"] {
    display: none !important;
}
</style>
"""


def _render_trip_form() -> Dict:
    """
    Render structured trip input form.

    Returns dict with: destination, origin, departure_date, return_date,
    travelers, budget, preferences
    """
    # Use columns for compact layout
    col1, col2 = st.columns(2)

    with col1:
        destination = st.text_input(
            "Where to?",
            placeholder="Paris, France",
            key="trip_destination"
        )

        departure_date = st.date_input(
            "Departure",
            value=date.today() + timedelta(days=30),
            min_value=date.today(),
            key="trip_departure_date"
        )

    with col2:
        origin = st.text_input(
            "From",
            placeholder="Los Angeles, CA (optional)",
            key="trip_origin"
        )

        return_date = st.date_input(
            "Return",
            value=date.today() + timedelta(days=37),
            min_value=date.today(),
            key="trip_return_date"
        )

    col3, col4 = st.columns(2)

    with col3:
        travelers = st.selectbox(
            "Travelers",
            options=["1 adult", "2 adults", "2 adults, 1 child", "2 adults, 2 children", "Family (4+)", "Group (5+)"],
            key="trip_travelers"
        )

    with col4:
        budget = st.slider(
            "Budget (USD)",
            min_value=500,
            max_value=20000,
            value=3000,
            step=500,
            format="$%d",
            key="trip_budget"
        )

    # Preferences - the "conversational" part
    preferences = st.text_area(
        "Preferences (optional)",
        placeholder="e.g., boutique hotels, food tours, avoid tourist traps, interested in art museums...",
        height=80,
        key="trip_preferences"
    )

    return {
        "destination": destination,
        "origin": origin,
        "departure_date": departure_date,
        "return_date": return_date,
        "travelers": travelers,
        "budget": budget,
        "preferences": preferences
    }


def _get_default_expert_selection():
    """Get default expert selection for travel questions."""
    # Default to 4 most common experts for travel planning
    return [
        "Budget Advisor",
        "Logistics Planner",
        "Activity Curator",
        "Local Culture Guide"
    ]


def _render_realtime_expert_chips(question: str):
    """
    Render real-time expert chips based on question text.
    Uses st.pills() with multi-select for small clickable chips.
    """
    # Detect question type from current text
    # Only auto-select when question looks complete (ends with ? or . or is long enough)
    detected_type = "general"
    question_complete = False

    if question:
        question_stripped = question.strip()
        # Question is "complete" if it ends with punctuation or is substantial
        question_complete = (
            len(question_stripped) > 30 and
            (question_stripped.endswith('?') or question_stripped.endswith('.') or len(question_stripped) > 50)
        )

        if question_complete and len(question_stripped) > 10:
            # Use travel question type detection
            detected_type = detect_travel_question_type(question)

    # Get previous detected type to check for changes
    previous_type = st.session_state.get('detected_question_type', None)
    type_changed = previous_type != detected_type and previous_type is not None

    # Store detected type
    st.session_state.detected_question_type = detected_type

    # Get all experts and auto-selection based on type
    all_experts = list(TRAVEL_EXPERTS.keys())

    # Map question types to presets
    type_to_preset = {
        "destination_planning": "Full Trip Planning",
        "budget_optimization": "Budget Focus",
        "activity_search": "Activities & Entertainment",
        "food_guide": "Food & Dining",
        "accommodation_search": "Accommodation",
        "safety_check": "Safety & Health",
        "weather_planning": "Weather Planning",
        "flight_search": "Transportation",
        "general": "Full Panel"
    }

    preset_name = type_to_preset.get(detected_type, "Full Trip Planning")
    preset = TRAVEL_PRESETS.get(preset_name, {})
    auto_experts = preset.get("experts", [])

    # Check if user has manually edited experts
    user_edited = st.session_state.get('user_edited_experts', False)

    # Initialize selected experts list with defaults
    if 'selected_experts_list' not in st.session_state:
        st.session_state.selected_experts_list = _get_default_expert_selection()

    # Only auto-select when question is complete and user hasn't manually edited
    if question_complete and auto_experts and not user_edited:
        # Check if this is a new auto-selection (type changed or first time)
        if type_changed or not st.session_state.selected_experts_list:
            st.session_state.selected_experts_list = list(auto_experts)

    current_selection = st.session_state.selected_experts_list or []

    # Filter to valid experts only (but allow empty selection)
    valid_selection = [e for e in current_selection if e in all_experts]
    if set(valid_selection) != set(current_selection):
        st.session_state.selected_experts_list = valid_selection
        current_selection = valid_selection

    # Get type info for display
    type_info = TRAVEL_QUESTION_TYPES.get(detected_type, {})
    type_name = type_info.get('name', 'Travel Question')
    type_icon = type_info.get('icon', '✈️')

    # Show detected type badge
    if question and len(question.strip()) > 10:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span style="background: #e8f5e9; color: #2e7d32; padding: 0.2rem 0.6rem; border-radius: 1rem; font-size: 0.8rem;">
                {type_icon} {type_name}
            </span>
            <span style="color: #888; font-size: 0.75rem;">
                Auto-selecting relevant experts
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Sort experts: selected first, then unselected
    selected_experts = [e for e in all_experts if e in current_selection]
    unselected_experts = [e for e in all_experts if e not in current_selection]
    sorted_experts = selected_experts + unselected_experts

    # Use st.pills with multi-select for clickable expert chips
    selected = st.pills(
        f"{len(current_selection)} of {len(all_experts)} experts",
        options=sorted_experts,
        selection_mode="multi",
        default=current_selection,
        key="expert_pills",
        label_visibility="collapsed"
    )

    # Update selection if changed
    if selected is not None:
        new_selection = list(selected) if selected else []
        if set(new_selection) != set(current_selection):
            st.session_state.selected_experts_list = new_selection
            st.session_state.user_edited_experts = True
            st.rerun()


def render_home():
    """Render the conversational home interface with state routing."""

    # Inject custom styles
    st.markdown(_HOME_STYLES, unsafe_allow_html=True)

    # State router: home → processing → answer (auto-validation, no manual step)
    view_state = st.session_state.get('view_state', 'home')
    research_result = st.session_state.get('research_result')

    # Ensure state consistency between view_state and research_result
    if view_state == 'answer' and not research_result:
        # Result missing, reset to home
        st.session_state.view_state = 'home'
        st.rerun()
    elif view_state != 'answer' and view_state != 'processing' and research_result:
        # Result exists but state doesn't match, sync to answer view
        st.session_state.view_state = 'answer'
        st.rerun()

    if view_state == 'awaiting_validation':
        # Legacy state - redirect to home (validation now happens automatically)
        st.session_state.view_state = 'home'
        st.rerun()
    elif view_state == 'processing':
        _render_processing_view()
    elif view_state == 'answer':
        _render_result_view()
    elif view_state == 'quick_answer':
        _render_quick_answer_view()
    else:
        _render_question_input()


def _render_project_chips():
    """Render recent project chips for quick project selection using st.pills."""
    try:
        db_path = settings.OUTPUTS_DIR / "literature_review.db"
        db_manager = DatabaseManager(db_path)
        project_dao = ProjectDAO(db_manager)
        projects = project_dao.get_all_projects()[:5]  # Get 5 most recent
    except Exception as e:
        projects = []

    if not projects:
        st.caption("No recent projects")
        return

    current_project_id = st.session_state.get('current_project_id')

    # Build project options with their IDs mapped
    project_labels = []
    project_id_map = {}  # label -> project_id

    for project in projects:
        label = project.name
        # Get program profile for context
        try:
            profile_dao = ProgramProfileDAO(db_manager)
            profile = profile_dao.get(project.id)
            if profile:
                target = profile.get('target', '')
                indication = profile.get('indication', '')
                if target or indication:
                    context_parts = [p for p in [target, indication] if p]
                    if context_parts:
                        label = f"{project.name} ({', '.join(context_parts)[:15]})"
        except Exception:
            pass

        project_labels.append(label)
        project_id_map[label] = project.id

    # Find current selection label
    current_selection = None
    if current_project_id:
        for label, pid in project_id_map.items():
            if pid == current_project_id:
                current_selection = label
                break

    # Use st.pills for chip-like selection (styling handled by global CSS in _HOME_STYLES)
    selected = st.pills(
        "Recent Projects",
        options=project_labels,
        default=current_selection,
        key="project_pills"
    )

    # Handle selection change
    if selected and selected != current_selection:
        new_project_id = project_id_map.get(selected)
        if new_project_id:
            st.session_state.current_project_id = new_project_id
            # Find project name
            for p in projects:
                if p.id == new_project_id:
                    st.session_state.current_project_name = p.name
                    break
            st.rerun()
    elif not selected and current_project_id:
        # Deselected
        st.session_state.current_project_id = None
        st.session_state.current_project_name = None
        st.rerun()


def _render_question_input():
    """Render the question input with trip form."""

    # Centered header
    st.markdown("""
    <div class="centered-header">
        <h1>Plan Your Trip</h1>
        <p>Fill in your trip details below. I'll fetch real weather and flight data, then consult expert advisors.</p>
    </div>
    """, unsafe_allow_html=True)

    # Centered container for main content
    col_pad1, col_main, col_pad2 = st.columns([1, 4, 1])

    with col_main:
        # Recent projects chips (replaces question type pills)
        _render_project_chips()

        st.markdown("")  # Spacing

        # Trip form - structured inputs for destination, dates, etc.
        trip_form = _render_trip_form()

        # Store form data in session state for button handler
        st.session_state['trip_form'] = trip_form

        # Build question from form for expert chips detection
        question = ""
        if trip_form.get("destination"):
            question = f"Plan a trip to {trip_form['destination']}"
            if trip_form.get("preferences"):
                question += f". {trip_form['preferences']}"

        # Expert chips - small clickable multi-select pills (hidden in Quick Answer mode)
        if not st.session_state.get('quick_answer_mode', False):
            _render_realtime_expert_chips(question)

        # Context input (collapsed by default) - BELOW the expert chips
        with st.expander("Add context (optional)", expanded=False):
            context_tab1, context_tab2, context_tab3, context_tab4 = st.tabs(["Paste Text", "Upload File", "URL", "📸 Image"])

            with context_tab1:
                context_text = st.text_area(
                    "Background information",
                    placeholder="Paste relevant data, meeting notes, or specific requirements...",
                    height=120,
                    key="home_context_input"
                )

            with context_tab2:
                uploaded_file = st.file_uploader(
                    "Upload a document",
                    type=['pdf', 'txt', 'md', 'html', 'htm', 'docx', 'pptx', 'xlsx'],
                    key="home_file_upload",
                    help="Supported: PDF, Word, PowerPoint, Excel, HTML, TXT, MD (max 10MB)"
                )

                if uploaded_file:
                    # Validate file size (10MB limit)
                    MAX_FILE_SIZE_MB = 10
                    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                    if file_size_mb > MAX_FILE_SIZE_MB:
                        st.error(f"File is too large ({file_size_mb:.1f}MB). Maximum allowed size is {MAX_FILE_SIZE_MB}MB.")
                    else:
                        # Process uploaded file
                        try:
                            from core.document_ingestion import ingest_file
                            with st.spinner(f"Processing {uploaded_file.name}..."):
                                doc = ingest_file(uploaded_file, uploaded_file.name)
                                # Store extracted text as context
                                st.session_state.uploaded_context = doc.content[:10000]  # Limit to 10k chars
                                st.success(f"Loaded: {uploaded_file.name} ({len(doc.content)} chars)")
                        except ImportError:
                            st.warning("Document ingestion not available. Paste text instead.")
                        except Exception as e:
                            st.error(user_friendly_error(e, "processing document"))

            with context_tab3:
                url_input = st.text_input(
                    "Enter URL",
                    placeholder="https://example.com/article",
                    key="home_url_input",
                    label_visibility="collapsed"
                )

                if st.button("Fetch URL", type="secondary", disabled=not url_input, key="home_fetch_url_btn"):
                    try:
                        import requests
                        from bs4 import BeautifulSoup
                        with st.spinner(f"Fetching {url_input}..."):
                            response = requests.get(url_input, timeout=30, headers={
                                'User-Agent': 'Mozilla/5.0 (compatible; ResearchPartner/1.0)'
                            })
                            response.raise_for_status()
                            soup = BeautifulSoup(response.text, 'html.parser')
                            # Extract text from common content tags
                            for script in soup(["script", "style", "nav", "footer", "header"]):
                                script.decompose()
                            text = soup.get_text(separator='\n', strip=True)
                            # Limit and store
                            st.session_state.url_context = text[:10000]
                            st.success(f"Fetched {len(text)} chars from URL")
                    except ImportError:
                        st.warning("requests/beautifulsoup4 not installed. Paste content instead.")
                    except Exception as e:
                        st.error(user_friendly_error(e, "fetching URL"))

            with context_tab4:
                st.caption("Upload photos for AI analysis (maps, itineraries, destination images, etc.)")
                uploaded_images = st.file_uploader(
                    "Drop images here",
                    type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
                    accept_multiple_files=True,
                    key="home_image_upload"
                )

                if uploaded_images:
                    # Validate image sizes (5MB per image limit)
                    MAX_IMAGE_SIZE_MB = 5
                    valid_images = []
                    for img in uploaded_images:
                        img.seek(0)
                        img_size_mb = len(img.read()) / (1024 * 1024)
                        img.seek(0)  # Reset for later use
                        if img_size_mb > MAX_IMAGE_SIZE_MB:
                            st.warning(f"Image '{img.name}' is too large ({img_size_mb:.1f}MB). Max: {MAX_IMAGE_SIZE_MB}MB - skipped.")
                        else:
                            valid_images.append(img)

                    if valid_images:
                        # Show thumbnails
                        cols = st.columns(min(len(valid_images), 3))
                        for i, img in enumerate(valid_images):
                            with cols[i % 3]:
                                st.image(img, width=150)

                        # Analyze button
                        if st.button("🔬 Analyze Images", key="analyze_images_btn"):
                            try:
                                from core.image_analyzer import analyze_image, format_for_expert_context
                                with st.spinner("Analyzing images..."):
                                    analyses = []
                                    for img in valid_images:
                                        # store pointer to beginning to read bytes
                                        img.seek(0)
                                        result = analyze_image(img.read(), img.name)
                                        analyses.append(result)

                                    st.session_state.image_analyses = analyses
                                    st.session_state.image_context = format_for_expert_context(analyses)
                                    st.success(f"Analyzed {len(analyses)} image(s)")
                            except Exception as e:
                                st.error(user_friendly_error(e, "analyzing images"))

        # Combine context from all sources
        context_text_val = st.session_state.get('home_context_input', '')
        uploaded_context = st.session_state.get('uploaded_context', '')
        url_context = st.session_state.get('url_context', '')
        image_context = st.session_state.get('image_context', '')
        all_contexts = [c for c in [context_text_val, uploaded_context, url_context, image_context] if c]
        context = '\n\n'.join(all_contexts) if all_contexts else ''

        # Quick Answer toggle - General AI response (no expert panel)
        st.markdown("")  # Spacing
        previous_quick_mode = st.session_state.get('quick_answer_mode', False)
        quick_mode = st.toggle(
            "Quick Answer",
            value=previous_quick_mode,
            key="quick_answer_toggle",
            help="General AI response (~5s) without expert panel. Toggle off for full expert panel discussion (~60s)."
        )
        st.session_state['quick_answer_mode'] = quick_mode

        # Handle mode change - deselect/restore experts
        if quick_mode != previous_quick_mode:
            if quick_mode:
                # Switching to Quick Answer - save current selection and clear experts
                st.session_state['_saved_expert_selection'] = st.session_state.get('selected_experts_list', [])
                st.session_state.selected_experts_list = []
            else:
                # Switching back to expert mode - restore saved selection
                saved = st.session_state.get('_saved_expert_selection', [])
                if saved:
                    st.session_state.selected_experts_list = saved
                st.session_state.user_edited_experts = False  # Allow auto-selection again
            st.rerun()

        if quick_mode:
            st.caption("General AI response with web search (no expert panel)")
        else:
            st.caption("Expert-first approach: Travel advisors provide insights with real-time data")

        # Research buttons - BELOW add context
        st.markdown("")  # Spacing
        sub_col1, sub_col2, sub_col3 = st.columns([1, 2, 1])
        with sub_col2:
            button_label = "Get Quick Answer" if quick_mode else "Plan My Trip"

            if st.button(
                f"{button_label}",
                type="primary",
                use_container_width=True,
                key="research_btn"
            ):
                # Get trip form data
                trip_form = st.session_state.get('trip_form', {})
                destination = trip_form.get('destination', '')

                if not destination:
                    st.warning("Please enter a destination to plan your trip.")
                else:
                    # Build question from form
                    question = f"Plan a trip to {destination}"
                    if trip_form.get('origin'):
                        question += f" from {trip_form['origin']}"

                    # Get selected experts from chips
                    selected_experts = st.session_state.get('selected_experts_list', [])

                    if quick_mode:
                        _execute_quick_answer(question, context)
                    elif not selected_experts:
                        st.warning("Please select at least one expert.")
                    else:
                        _execute_research(question, "destination_planning", context, selected_experts, trip_form)

        # Recent questions section (below main input)
        _render_recent_questions()


def _execute_research(question: str, question_type: Optional[str], context: Optional[str] = None, selected_experts: Optional[list] = None, trip_form: Optional[Dict] = None):
    """Execute Pass 1 of the research workflow (experts only)."""

    # Check for API key - use Gemini API key for Gemini models, otherwise OpenAI
    model = getattr(settings, 'EXPERT_MODEL', '')
    if model and model.lower().startswith('gemini'):
        api_key = getattr(settings, 'GEMINI_API_KEY', None)
        if not api_key:
            st.error("Gemini API key not configured. Please set GEMINI_API_KEY in your .env file.")
            return
    else:
        api_key = getattr(settings, 'OPENAI_API_KEY', None)
        if not api_key:
            st.error("OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
            return

    # Fetch real travel data if form provided
    enriched_context = context or ""

    if trip_form and trip_form.get("destination"):
        with st.spinner("Fetching weather and flight data..."):
            try:
                from services.travel_data_service import get_travel_data_context
                api_context = get_travel_data_context(trip_form)
                if api_context:
                    enriched_context = api_context + "\n\n" + enriched_context
                    st.success("Real-time data loaded!")
            except Exception as e:
                st.warning(f"Could not fetch travel data: {e}")

        # Add preferences to question if provided
        if trip_form.get("preferences"):
            question = f"{question}\n\nPreferences: {trip_form['preferences']}"

    # Store current question and context
    st.session_state.current_question = question
    st.session_state.current_context = enriched_context
    st.session_state.processing_stage = "starting"
    st.session_state.user_selected_experts = selected_experts

    # Store destination for place enrichment
    if trip_form:
        st.session_state.current_destination = trip_form.get('destination', '')

    # Initialize service
    service = ResearchPartnerService(api_key=api_key)

    pass1_responses = None
    detected_type = question_type

    # Run research flow (Expert-First approach)
    with st.status("🔬 Consulting Experts...", expanded=True, state="running") as status:
        try:
            for event in service.run_research_flow(
                question=question,
                question_type=question_type,
                additional_context=enriched_context if enriched_context else None,
                project_id=st.session_state.get('current_project_id'),
                selected_experts=selected_experts
            ):
                stage = event.get('stage', '')
                message = event.get('message', '')
                st.session_state.processing_stage = stage

                if stage == 'parsing':
                    status.update(label="🧠 Understanding...", state="running")
                    st.write(f"✓ {message}")
                    if 'experts' in event:
                        experts = event['experts']
                        st.caption(f"Will consult: {', '.join(experts)}")
                        st.session_state.auto_selected_experts = experts
                    if 'question_type' in event:
                        detected_type = event['question_type']

                elif stage == 'searching':
                    status.update(label="📚 Background search...", state="running")

                elif stage == 'consulting':
                    status.update(label="👥 Getting expert perspectives...", state="running")
                    st.write(f"✓ {message}")
                    if event.get('pass1_complete'):
                        pass1_responses = event.get('responses', {})
                        for expert in pass1_responses:
                            st.caption(f"  ✓ {expert}: responded")
                        status.update(label="✅ Expert responses ready! Starting validation...", state="running")
                        # Continue to validation (don't break)

                elif stage == 'validating':
                    status.update(label="🔍 Validating claims against literature...", state="running")
                    st.write(f"✓ {message}")
                    if 'citations' in event:
                        count = len(event['citations'])
                        if count > 0:
                            st.caption(f"  Found {count} papers")

                elif stage == 'synthesizing':
                    status.update(label="✨ Synthesizing recommendation...", state="running")
                    st.write(f"✓ {message}")

                elif stage == 'complete':
                    result = event.get('result')
                    if result:
                        result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
                        st.session_state.research_result = result_dict
                        st.session_state.view_state = "answer"
                        status.update(label="✅ Complete!", state="complete")

                        # Save to recent questions
                        _save_to_recent_questions(
                            question=question,
                            question_type=detected_type or 'general',
                            confidence=result_dict.get('confidence', 'MEDIUM'),
                            recommendation_preview=result_dict.get('recommendation', '')[:100]
                        )
                        st.rerun()
                    return

        except Exception as e:
            st.error(user_friendly_error(e, "researching question"))
            st.session_state.processing_stage = "error"
            st.session_state.view_state = "home"
            return

    # Flow now continues automatically to validation and completion
    # No need to transition to awaiting_validation state


def _execute_quick_answer(question: str, context: Optional[str] = None):
    """Execute Quick Answer mode - single LLM call with PubMed search."""

    from services.quick_answer_service import get_quick_answer_with_search
    from ui.citation_utils import format_expert_response

    # Store question
    st.session_state.current_question = question
    st.session_state.current_context = context

    # Build scenario from context if provided
    scenario = ""
    if context:
        # Extract first sentence as scenario hint
        scenario = context[:200] if len(context) < 200 else context[:200] + "..."

    with st.spinner("Searching literature and generating quick answer..."):
        try:
            result = get_quick_answer_with_search(
                question=question,
                scenario=scenario,
                max_results=5
            )

            # Store as quick answer result
            st.session_state.quick_answer_result = {
                'question': question,
                'answer': result.answer,
                'sources_used': result.sources_used,
                'model': result.model,
                'has_context': result.has_context,
                'citations': result.citations,
            }
            st.session_state.view_state = 'quick_answer'
            st.rerun()

        except Exception as e:
            st.error(user_friendly_error(e, "generating quick answer"))


def _render_quick_answer_view():
    """Render the Quick Answer result view."""

    from ui.citation_utils import format_expert_response

    result = st.session_state.get('quick_answer_result')
    if not result:
        st.session_state.view_state = 'home'
        st.rerun()
        return

    question = result.get('question', '')
    answer = result.get('answer', '')
    citations = result.get('citations', [])
    sources_used = result.get('sources_used', 0)
    model = result.get('model', '')

    # Header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #10B981 0%, #059669 100%);
                padding: 16px 20px; border-radius: 12px; color: white; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 12px; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.5px;">Quick Answer</span>
                <h3 style="margin: 4px 0 0 0; color: white; font-size: 18px;">{question}</h3>
            </div>
            <span style="font-size: 13px; opacity: 0.9;">Based on {sources_used} sources</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Answer with citation highlighting
    formatted_answer = format_expert_response(answer)
    st.markdown(f"""
    <div style="background: #F9FAFB; padding: 20px; border-radius: 8px; border: 1px solid #E5E7EB; margin-bottom: 16px;">
        <div style="font-size: 15px; line-height: 1.7; color: #1F2937;">{formatted_answer}</div>
    </div>
    """, unsafe_allow_html=True)

    # Citations
    if citations:
        with st.expander(f"View {len(citations)} sources", expanded=False):
            for i, cit in enumerate(citations, 1):
                pmid = cit.get('pmid', '')
                title = cit.get('title', 'Unknown')
                if pmid:
                    st.markdown(f"**[{i}]** [{title}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/) (PMID: {pmid})")
                else:
                    st.markdown(f"**[{i}]** {title}")

    # Model info
    st.caption(f"Model: {model}")

    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Run Full Expert Panel", type="primary", use_container_width=True):
            # Switch to full research mode
            st.session_state['quick_answer_mode'] = False
            _execute_research(question, None, st.session_state.get('current_context'))

    with col2:
        if st.button("Ask Follow-up", use_container_width=True):
            # Go back to home with the question context preserved
            st.session_state.view_state = 'home'
            if 'quick_answer_result' in st.session_state:
                del st.session_state['quick_answer_result']
            st.rerun()

    with col3:
        if st.button("New Question", use_container_width=True):
            # Reset everything
            reset_conversational_state()
            st.session_state.view_state = 'home'
            st.rerun()


def _render_pass1_results():
    """
    LEGACY: Render Pass 1 results with manual validation buttons.

    Note: This function is no longer used in the main flow since validation
    now runs automatically after Pass 1. Kept for potential fallback/debug use.
    """

    pass1_data = st.session_state.get('_pass1_data', {})
    if not pass1_data:
        st.session_state.view_state = "home"
        st.rerun()
        return

    question = pass1_data.get('question', '')
    question_type = pass1_data.get('question_type')
    pass1_responses = pass1_data.get('responses', {})

    # Show original question
    st.info(f"**Question:** {question}")

    # Show Pass 1 expert responses
    st.markdown("### Expert Perspectives")
    st.caption("Immediate expert analysis based on domain knowledge")

    for expert_name, response in pass1_responses.items():
        content = response.get('content', '')
        with st.expander(f"**{expert_name}**", expanded=True):
            st.markdown(content[:3000])

    st.divider()

    # Action buttons (OUTSIDE any form)
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🔍 Validate Against Literature & Synthesize", type="primary", use_container_width=True):
            _execute_validation()
    with col2:
        if st.button("✏️ New Question", use_container_width=True):
            st.session_state.view_state = "home"
            if '_pass1_data' in st.session_state:
                del st.session_state['_pass1_data']
            st.rerun()
    with col3:
        if st.button("📋 Skip Validation", use_container_width=True):
            # Go straight to result view with just Pass 1 data
            _finalize_without_validation()


def _execute_validation():
    """Execute Pass 2: Literature validation and synthesis."""

    pass1_data = st.session_state.get('_pass1_data', {})
    if not pass1_data:
        st.session_state.view_state = "home"
        st.rerun()
        return

    question = pass1_data.get('question', '')
    question_type = pass1_data.get('question_type')
    context = pass1_data.get('context')
    pass1_responses = pass1_data.get('responses', {})

    # Get API key
    model = getattr(settings, 'EXPERT_MODEL', '')
    if model and model.lower().startswith('gemini'):
        api_key = getattr(settings, 'GEMINI_API_KEY', None)
    else:
        api_key = getattr(settings, 'OPENAI_API_KEY', None)

    if not api_key:
        st.error("API key not configured.")
        return

    service = ResearchPartnerService(api_key=api_key)

    # Show question context
    st.info(f"**Question:** {question}")

    # Run validation and synthesis
    result = None
    with st.status("🔍 Validating & Synthesizing...", expanded=True, state="running") as status:
        try:
            for event in service.run_research_flow(
                question=question,
                question_type=question_type,
                additional_context=context if context else None,
                project_id=st.session_state.get('current_project_id')
            ):
                stage = event.get('stage', '')
                message = event.get('message', '')

                if stage in ('parsing', 'searching', 'consulting'):
                    # Skip early stages in display (already done)
                    if stage == 'consulting' and event.get('pass1_complete'):
                        status.update(label="📚 Searching literature...", state="running")

                elif stage == 'validating':
                    status.update(label="🔍 Validating claims...", state="running")
                    st.write(f"✓ {message}")
                    if 'citations' in event:
                        count = len(event['citations'])
                        if count > 0:
                            st.caption(f"  Found {count} papers")

                elif stage == 'synthesizing':
                    status.update(label="✨ Synthesizing...", state="running")
                    st.write(f"✓ {message}")

                elif stage == 'complete':
                    result = event.get('result')
                    status.update(label="✅ Complete!", state="complete")

        except Exception as e:
            st.error(user_friendly_error(e, "validating claims"))
            return

    # Store result and transition to answer state
    if result:
        result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
        st.session_state.research_result = result_dict
        st.session_state.processing_stage = "complete"
        st.session_state.view_state = "answer"

        # Clear Pass 1 data
        if '_pass1_data' in st.session_state:
            del st.session_state['_pass1_data']

        # Save to recent questions history
        _save_to_recent_questions(
            question=question,
            question_type=question_type or 'general',
            confidence=result_dict.get('confidence', 'MEDIUM'),
            recommendation_preview=result_dict.get('recommendation', '')[:100]
        )

        st.rerun()


def _finalize_without_validation():
    """Create a basic result from Pass 1 responses only (skip validation)."""

    pass1_data = st.session_state.get('_pass1_data', {})
    if not pass1_data:
        st.session_state.view_state = "home"
        st.rerun()
        return

    question = pass1_data.get('question', '')
    question_type = pass1_data.get('question_type', 'general')
    pass1_responses = pass1_data.get('responses', {})

    # Build a simple result from Pass 1 only
    combined_findings = []
    for expert_name, response in pass1_responses.items():
        content = response.get('content', '')
        # Truncate at sentence boundary if too long
        if len(content) > 600:
            # Find last sentence end within limit
            truncate_point = content[:600].rfind('. ')
            if truncate_point > 200:
                content = content[:truncate_point + 1]
            else:
                content = content[:600].rsplit(' ', 1)[0] + "..."
        combined_findings.append(f"**{expert_name}**: {content}")

    result_dict = {
        'question': question,
        'question_type': question_type,
        'recommendation': "Expert perspectives provided (validation skipped)",
        'confidence': 'LOW',
        'key_findings': combined_findings[:5],
        'evidence_summary': {'note': 'Literature validation was skipped'},
        'expert_responses': pass1_responses,
        'validations': {},
        'follow_up_suggestions': ["Run validation to get literature-backed synthesis"],
        'dissenting_views': [],
        'metadata': {'validation_skipped': True}
    }

    st.session_state.research_result = result_dict
    st.session_state.view_state = "answer"

    # Clear Pass 1 data
    if '_pass1_data' in st.session_state:
        del st.session_state['_pass1_data']

    st.rerun()


def _save_to_recent_questions(question: str, question_type: str, confidence: str, recommendation_preview: str):
    """Save a completed question to recent history."""
    recent = st.session_state.get('recent_questions', [])

    # Add new entry at the beginning
    entry = {
        'question': question,
        'question_type': question_type,
        'timestamp': datetime.now().isoformat(),
        'confidence': confidence,
        'recommendation_preview': recommendation_preview
    }
    recent.insert(0, entry)

    # Keep only the most recent 5
    st.session_state.recent_questions = recent[:5]


def _render_processing_view():
    """Render the processing state (shown during research execution)."""
    # Show current question being processed
    question = st.session_state.get('current_question', '')
    if question:
        st.info(f"**Question:** {question}")

    # Map processing stages to user-friendly messages
    stage = st.session_state.get('processing_stage', 'starting')
    stage_messages = {
        'starting': 'Initializing research workflow...',
        'parsing': 'Understanding your question...',
        'searching': 'Searching literature databases...',
        'consulting': 'Consulting expert panel...',
        'validating': 'Validating claims against evidence...',
        'synthesizing': 'Synthesizing recommendations...',
        'complete': 'Research complete!',
        'error': 'An error occurred'
    }
    message = stage_messages.get(stage, 'Processing...')

    # Always show spinner during processing
    if stage not in ('complete', 'error'):
        with st.status(f"🔬 {message}", expanded=True, state="running") as status:
            st.write("Finding evidence, consulting experts, and synthesizing recommendations.")
            # The actual processing happens in _execute_research


def _render_recent_questions():
    """Render the recent questions history section."""
    recent = st.session_state.get('recent_questions', [])

    if not recent:
        return

    st.markdown("---")
    st.markdown("**Recent Questions**")

    for i, entry in enumerate(recent[:5]):
        question = entry.get('question', '')[:60]
        q_type = entry.get('question_type', 'general')
        confidence = entry.get('confidence', 'MEDIUM')
        preview = entry.get('recommendation_preview', '')[:80]

        # Get type info for icon
        type_info = TRAVEL_QUESTION_TYPES.get(q_type, {})
        icon = type_info.get('icon', '✈️')

        # Confidence badge color
        conf_colors = {'HIGH': '#28a745', 'MEDIUM': '#ffc107', 'LOW': '#dc3545'}
        conf_color = conf_colors.get(confidence.upper(), '#666')

        col1, col2 = st.columns([6, 1])
        with col1:
            # Clickable card
            if st.button(
                f"{icon} {question}...",
                key=f"recent_{i}",
                use_container_width=True,
                type="secondary"
            ):
                # Re-populate the question for re-running
                st.session_state.question_type = q_type
                st.session_state['home_question_input'] = entry.get('question', '')
                st.rerun()

            st.caption(f"{preview}...")
        with col2:
            # Confidence indicator
            st.markdown(f'<span style="background:{conf_color};color:white;padding:0.2rem 0.5rem;border-radius:0.25rem;font-size:0.7rem;">{confidence}</span>', unsafe_allow_html=True)


def _render_result_view():
    """Render the result with answer and evidence drawer."""
    result_dict = st.session_state.research_result

    if not result_dict:
        # No result, go back to home
        st.session_state.view_state = "home"
        st.rerun()
        return

    # Convert dict back to ResearchResult if needed
    if isinstance(result_dict, dict):
        result = ResearchResult(
            question=result_dict.get('question', ''),
            question_type=result_dict.get('question_type', 'general'),
            recommendation=result_dict.get('recommendation', ''),
            confidence=result_dict.get('confidence', 'MEDIUM'),
            key_findings=result_dict.get('key_findings', []),
            evidence_summary=result_dict.get('evidence_summary', {}),
            expert_responses=result_dict.get('expert_responses', {}),
            validations=result_dict.get('validations', {}),
            follow_up_suggestions=result_dict.get('follow_up_suggestions', []),
            dissenting_views=result_dict.get('dissenting_views', []),
            metadata=result_dict.get('metadata', {})
        )
    else:
        result = result_dict

    # Header with new question button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("New Question", type="secondary", use_container_width=True):
            reset_conversational_state()  # This now also resets view_state to "home"
            st.rerun()

    # Render the answer view component
    render_answer_view(result)

    # Render collapsible evidence drawer
    render_evidence_drawer(result)

    # Inline follow-up chat
    _render_inline_chat(result)


def _render_inline_chat(result):
    """Render inline follow-up chat interface."""
    st.markdown("---")
    st.subheader("Follow-up Questions")

    # Initialize chat history if needed
    if 'inline_chat_history' not in st.session_state:
        st.session_state.inline_chat_history = []

    # Display chat history
    chat_history = st.session_state.inline_chat_history
    for msg in chat_history:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        with st.chat_message(role):
            st.markdown(content)

    # Follow-up input
    follow_up = st.chat_input("Ask a follow-up question...")

    if follow_up:
        # 1. Add user message to history
        st.session_state.inline_chat_history.append({
            'role': 'user',
            'content': follow_up
        })
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(follow_up)

        # 2. Get API Key (copying logic from _execute_research)
        model = getattr(settings, 'EXPERT_MODEL', '')
        if model and model.lower().startswith('gemini'):
            api_key = getattr(settings, 'GEMINI_API_KEY', None)
        else:
            api_key = getattr(settings, 'OPENAI_API_KEY', None)
            
        if not api_key:
            st.error("API key not configured.")
            return

        # 3. Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Initialize service
                    from services.follow_up_service import FollowUpService
                    service = FollowUpService(api_key=api_key)
                    
                    # Convert result to dict if it's an object
                    result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
                    
                    # Generate response
                    response = service.generate_response(
                        question=follow_up,
                        chat_history=st.session_state.inline_chat_history,
                        research_context=result_dict
                    )
                    
                    st.markdown(response)
                    
                    # 4. Add assistant response to history
                    st.session_state.inline_chat_history.append({
                        'role': 'assistant',
                        'content': response
                    })

                except Exception as e:
                    st.error(user_friendly_error(e, "generating response"))

        # Get API key - use Gemini API key for Gemini models
        model = getattr(settings, 'EXPERT_MODEL', '')
        if model and model.lower().startswith('gemini'):
            api_key = getattr(settings, 'GEMINI_API_KEY', None)
            if not api_key:
                st.error("Gemini API key not configured.")
                return
        else:
            api_key = getattr(settings, 'OPENAI_API_KEY', None)
            if not api_key:
                st.error("OpenAI API key not configured.")
                return

        # Handle follow-up
        service = ResearchPartnerService(api_key=api_key)

        # Show the user's question while processing
        with st.chat_message("user"):
            st.markdown(follow_up)

        with st.chat_message("assistant"):
            with st.spinner(f"Thinking about: *{follow_up[:80]}{'...' if len(follow_up) > 80 else ''}*"):
                try:
                    response_text = ""
                    for event in service.handle_follow_up(
                        follow_up_question=follow_up,
                        previous_result=result,
                        chat_history=chat_history
                    ):
                        if event.get('stage') == 'complete':
                            response_text = event.get('response', '')
                        elif event.get('stage') == 'error':
                            response_text = event.get('message', 'Error processing follow-up')

                    # Add assistant response to history
                    st.session_state.inline_chat_history.append({
                        'role': 'assistant',
                        'content': response_text
                    })

                    st.rerun()

                except Exception as e:
                    st.error(user_friendly_error(e, "processing follow-up"))


def render_suggested_followups(result, on_select_callback=None):
    """Render suggested follow-up question buttons."""
    suggestions = result.follow_up_suggestions if hasattr(result, 'follow_up_suggestions') else result.get('follow_up_suggestions', [])

    if not suggestions:
        return

    st.markdown("**Suggested follow-ups:**")
    cols = st.columns(min(len(suggestions), 3))

    for i, suggestion in enumerate(suggestions[:3]):
        with cols[i]:
            # Truncate long suggestions for button text
            btn_text = suggestion[:50] + "..." if len(suggestion) > 50 else suggestion
            if st.button(btn_text, key=f"followup_{i}", use_container_width=True):
                # Add to chat history and trigger follow-up
                st.session_state.inline_chat_history.append({
                    'role': 'user',
                    'content': suggestion
                })
                if on_select_callback:
                    on_select_callback(suggestion)
                st.rerun()
