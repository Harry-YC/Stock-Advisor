"""
Test Question Definitions for Palliative Surgery GDG E2E Tests

Contains the clinical questions used for testing, derived from
the research report on femoral metastases and Mirels score.
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ClinicalTestCase:
    """A clinical test case with metadata for E2E testing."""
    question: str
    scenario: str
    expected_keywords: List[str]  # Keywords expected in response
    expected_citations: bool       # Whether citations are expected
    expected_experts: List[str]    # Which experts should respond
    category: str                  # Question category


# =============================================================================
# PRIMARY TEST QUESTION (From Research Report)
# =============================================================================

MIRELS_SCORE_9_QUESTION = ClinicalTestCase(
    question=(
        "What is the survival benefit of prophylactic fixation versus "
        "observation in patients with femoral metastases and Mirels score of 9?"
    ),
    scenario="Pathologic Fracture Risk",
    expected_keywords=[
        "prophylactic",
        "fixation",
        "Mirels",
        "survival",
        "metastases",
        "femoral",
        "fracture",
    ],
    expected_citations=True,
    expected_experts=[
        "Surgical Oncologist",
        "Perioperative Medicine Physician",
        "Palliative Care Physician",
    ],
    category="intervention_choice"
)


# =============================================================================
# SECONDARY TEST QUESTIONS
# =============================================================================

MALIGNANT_BOWEL_OBSTRUCTION_QUESTION = ClinicalTestCase(
    question="What is the role of palliative surgery in malignant bowel obstruction?",
    scenario="Malignant Bowel Obstruction",
    expected_keywords=[
        "bowel obstruction",
        "palliative",
        "surgery",
        "stent",
        "bypass",
        "QoL",
    ],
    expected_citations=True,
    expected_experts=[
        "Surgical Oncologist",
        "Palliative Care Physician",
        "Interventionalist",
    ],
    category="surgical_candidate"
)


SYMPTOM_CONTROL_QUESTION = ClinicalTestCase(
    question="How should intractable cancer pain be managed surgically?",
    scenario="Symptom Management",
    expected_keywords=[
        "pain",
        "intractable",
        "nerve block",
        "cordotomy",
        "palliative",
    ],
    expected_citations=True,
    expected_experts=[
        "Pain and Symptom-Control Specialist",
        "Palliative Care Physician",
    ],
    category="symptom_management"
)


PROGNOSIS_QUESTION = ClinicalTestCase(
    question="What is the expected survival after palliative resection for ovarian cancer with carcinomatosis?",
    scenario="Prognosis Assessment",
    expected_keywords=[
        "survival",
        "ovarian",
        "carcinomatosis",
        "resection",
        "prognosis",
    ],
    expected_citations=True,
    expected_experts=[
        "Surgical Oncologist",
        "GRADE Methodologist",
        "Clinical Evidence Specialist",
    ],
    category="prognosis_assessment"
)


ETHICS_QUESTION = ClinicalTestCase(
    question="When is it ethically appropriate to decline palliative surgery?",
    scenario="Ethics Review",
    expected_keywords=[
        "ethics",
        "autonomy",
        "beneficence",
        "futility",
        "informed consent",
    ],
    expected_citations=False,  # Ethics questions may rely more on principles
    expected_experts=[
        "Medical Ethicist",
        "Palliative Care Physician",
        "Patient Advocate",
    ],
    category="ethics_review"
)


# =============================================================================
# TEST QUESTION COLLECTIONS
# =============================================================================

ALL_TEST_QUESTIONS = [
    MIRELS_SCORE_9_QUESTION,
    MALIGNANT_BOWEL_OBSTRUCTION_QUESTION,
    SYMPTOM_CONTROL_QUESTION,
    PROGNOSIS_QUESTION,
    ETHICS_QUESTION,
]

QUICK_TEST_QUESTIONS = [
    MIRELS_SCORE_9_QUESTION,
    MALIGNANT_BOWEL_OBSTRUCTION_QUESTION,
]


# =============================================================================
# EXPECTED UI ELEMENTS BY FEATURE
# =============================================================================

CITATION_HIGHLIGHTING_EXPECTATIONS = {
    "pmid_pattern": r"\[PMID[:\s]*\d{7,8}\]",
    "reference_pattern": r"\[\d+\]",
    "epistemic_tags": ["EVIDENCE", "ASSUMPTION", "OPINION", "EVIDENCE GAP"],
    "badge_style": "linear-gradient(135deg, #6366F1",  # Purple gradient
}

QUICK_ANSWER_EXPECTATIONS = {
    "header_text": "Quick Answer",
    "header_style": "linear-gradient(135deg, #10B981",  # Green gradient
    "sources_text": "Based on",
    "expander_text": "sources",
}

CHALLENGER_EXPECTATIONS = {
    "button_text": ["Challenge", "Red Team", "Devil"],
    "categories": ["assumption", "evidence", "patient_selection", "threshold", "risk", "feasibility"],
    "min_questions": 3,
}

SMART_SUGGESTIONS_EXPECTATIONS = {
    "header_patterns": ["Suggested", "Follow-up", "Related"],
    "min_suggestions": 2,
}

MARK_PEN_EXPECTATIONS = {
    "button_icon": "🖊️",
    "mark_types": {
        "important_data": "📊",
        "key_finding": "⭐",
        "evidence_gap": "🔍",
        "citation_useful": "📚",
        "disagree": "❌",
        "agree": "✓",
    },
    "sidebar_text": "Your Marks",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_question_by_category(category: str) -> ClinicalTestCase:
    """Get a test question by its category."""
    for q in ALL_TEST_QUESTIONS:
        if q.category == category:
            return q
    return MIRELS_SCORE_9_QUESTION  # Default fallback


def get_quick_test_question() -> str:
    """Get the primary test question string for quick tests."""
    return MIRELS_SCORE_9_QUESTION.question


def get_expected_keywords() -> List[str]:
    """Get expected keywords for the primary test question."""
    return MIRELS_SCORE_9_QUESTION.expected_keywords
