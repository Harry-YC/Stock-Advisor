"""
Question Templates for Palliative Surgery GDG

Defines GDG clinical question types with auto-expert selection and synthesis focus areas.
Used by ResearchPartnerService to orchestrate the conversational workflow.
"""

from typing import Dict, List, Optional
from openai import OpenAI
from config import settings


QUESTION_TYPES: Dict[str, Dict] = {
    "surgical_candidate": {
        "name": "Surgical Candidacy",
        "description": "Assess whether a patient is appropriate for palliative surgical intervention",
        "icon": "🔪",
        "experts": [
            "Surgical Oncologist",
            "Perioperative Medicine Physician",
            "Palliative Care Physician",
            "GRADE Methodologist",
            "GDG Chair"
        ],
        "synthesis_focus": "surgical_recommendation",
        "output_template": """
## Recommendation
[PROCEED WITH SURGERY / DO NOT PROCEED / CONDITIONAL - requires optimization]

## Confidence
[HIGH / MEDIUM / LOW] - [rationale based on GRADE certainty]

## Patient Selection Criteria
- Indications: [bullet points with PMIDs]
- Contraindications: [bullet points with PMIDs]

## Expected Outcomes
- Symptom relief probability: [% with PMID]
- Operative mortality: [% with PMID]
- Major morbidity: [% with PMID]

## Alternatives Considered
- [Non-surgical options and why surgery preferred or not]

## Conditions for Proceeding
- [optimization requirements if conditional]

## Key Evidence Gaps
- [What data would change this recommendation]
"""
    },

    "palliative_pathway": {
        "name": "Palliative Care Pathway",
        "description": "Design comprehensive palliative care approach including goals of care",
        "icon": "🏥",
        "experts": [
            "Palliative Care Physician",
            "Pain and Symptom-Control Specialist",
            "Patient Advocate",
            "Medical Ethicist",
            "GDG Chair"
        ],
        "synthesis_focus": "care_pathway",
        "output_template": """
## Goals of Care Assessment
[Summary of patient/family priorities]

## Recommended Pathway
[COMFORT-FOCUSED / SYMPTOM CONTROL + DISEASE MODIFICATION / ACTIVE TREATMENT]

## Symptom Management Plan
| Symptom | Intervention | Expected Outcome | Timeline |
|---------|-------------|------------------|----------|

## Quality of Life Considerations
- Days at home vs hospital: [estimate if data available]
- Treatment burden: [assessment]
- Family/caregiver impact: [considerations]

## Prognosis Communication
- Estimated survival: [range with uncertainty]
- Prognostic factors: [bullet points]

## Care Coordination
- [Team members and roles]
- [Transition planning]

## Evidence Basis
- [Key PMIDs supporting recommendations]
"""
    },

    "intervention_choice": {
        "name": "Intervention Comparison",
        "description": "Compare surgical vs non-surgical interventions (stents, embolization, etc.)",
        "icon": "⚖️",
        "experts": [
            "Surgical Oncologist",
            "Interventionalist",
            "Palliative Care Physician",
            "Clinical Evidence Specialist",
            "GDG Chair"
        ],
        "synthesis_focus": "comparative_effectiveness",
        "output_template": """
## Recommended Intervention
[SURGERY / STENT / EMBOLIZATION / CONSERVATIVE / INDIVIDUALIZED]

## Comparative Assessment
| Intervention | Success Rate | Complications | Reintervention | Time to Benefit |
|--------------|--------------|---------------|----------------|-----------------|

## Clinical Scenario Considerations
- Performance status: [ECOG threshold]
- Life expectancy: [minimum for benefit]
- Tumor factors: [relevant characteristics]

## Advantages of Recommended Approach
- [bullet points with evidence tags]

## Risks and Trade-offs
- [bullet points with evidence tags]

## When to Consider Alternative
- [specific scenarios where other option preferred]

## Evidence Quality
- [GRADE certainty rating and rationale]
"""
    },

    "symptom_management": {
        "name": "Symptom Control",
        "description": "Address intractable symptoms including pain, obstruction, bleeding",
        "icon": "💊",
        "experts": [
            "Pain and Symptom-Control Specialist",
            "Palliative Care Physician",
            "Geriatric and Frailty Specialist",
            "GDG Chair"
        ],
        "synthesis_focus": "symptom_control",
        "output_template": """
## Primary Symptom Assessment
[Symptom severity, functional impact, current management]

## Recommended Management Strategy

### First-Line Approach
- [Intervention with dosing/technique]
- Expected response: [% and timeline]
- Evidence: [PMID citations]

### Escalation Options
| Step | Intervention | Indication | Evidence |
|------|-------------|------------|----------|

## Special Considerations
- Organ dysfunction: [dose adjustments]
- Drug interactions: [key concerns]
- Refractory symptoms: [when to escalate]

## Monitoring Parameters
- [What to monitor and frequency]

## When to Consider Procedural Intervention
- [Thresholds for referral to interventionalist/surgeon]

## Evidence Gaps
- [Areas needing more research]
"""
    },

    "prognosis_assessment": {
        "name": "Prognosis & Outcomes",
        "description": "Evaluate evidence on survival, quality of life, and functional outcomes",
        "icon": "📊",
        "experts": [
            "GRADE Methodologist",
            "Clinical Evidence Specialist",
            "Palliative Care Physician",
            "GDG Chair"
        ],
        "synthesis_focus": "evidence_synthesis",
        "output_template": """
## Evidence Summary

### Survival Outcomes
| Population | Intervention | Median Survival | 95% CI | Source |
|------------|-------------|-----------------|--------|--------|

### Quality of Life Outcomes
| Study | QoL Measure | Finding | GRADE Certainty |
|-------|-------------|---------|-----------------|

### Functional Outcomes
- [Performance status changes]
- [Return to function rates]

## Prognostic Factors
| Factor | Impact on Outcome | Evidence Quality |
|--------|-------------------|------------------|

## Evidence Quality Assessment
- Overall GRADE certainty: [VERY LOW / LOW / MODERATE / HIGH]
- Key limitations: [risk of bias, imprecision, etc.]

## Clinical Applicability
- Generalizability: [population considerations]
- Time period of evidence: [currency]

## Research Gaps
- [What studies would strengthen recommendations]
"""
    },

    "ethics_review": {
        "name": "Ethics & Appropriateness",
        "description": "Evaluate ethical considerations and treatment appropriateness",
        "icon": "⚖️",
        "experts": [
            "Medical Ethicist",
            "Palliative Care Physician",
            "Patient Advocate",
            "GDG Chair"
        ],
        "synthesis_focus": "ethics_assessment",
        "output_template": """
## Ethical Assessment

### Proportionality
- Benefit vs burden ratio: [assessment]
- Patient values alignment: [consideration]

### Informed Consent Considerations
- Key information for patient/family:
  - [bullet points]
- Decision-making capacity: [if relevant]
- Surrogate considerations: [if applicable]

### Appropriateness Assessment
[CLEARLY APPROPRIATE / CONDITIONALLY APPROPRIATE / POTENTIALLY INAPPROPRIATE / INAPPROPRIATE]

### Ethical Principles Analysis
| Principle | Consideration | Weight |
|-----------|---------------|--------|
| Beneficence | | |
| Non-maleficence | | |
| Autonomy | | |
| Justice | | |

### Conflict Resolution
- [If disagreement between patient/family/team]

### Documentation Recommendations
- [What should be documented for medicolegal protection]

## Professional Guidance
- [Relevant professional society positions]
"""
    },

    "resource_allocation": {
        "name": "Resource & Implementation",
        "description": "Assess cost-effectiveness, resource requirements, and implementation",
        "icon": "💰",
        "experts": [
            "Health Economist and Implementation Scientist",
            "GRADE Methodologist",
            "Palliative Care Physician",
            "GDG Chair"
        ],
        "synthesis_focus": "implementation_assessment",
        "output_template": """
## Resource Assessment

### Cost-Effectiveness
- ICER: [$/QALY if data available]
- Cost comparison vs alternatives: [table]
- Value assessment: [HIGH VALUE / REASONABLE / LOW VALUE / INSUFFICIENT DATA]

### Resource Requirements
| Resource | Requirement | Availability |
|----------|-------------|--------------|

### Implementation Considerations
- Setting: [inpatient/outpatient/ICU]
- Personnel: [required expertise]
- Equipment: [special requirements]

### System-Level Factors
- Wait times: [impact on outcomes]
- Geographic access: [considerations]
- Equity considerations: [disparities]

### Sustainability
- Short-term vs long-term costs
- Training requirements
- Infrastructure needs

### Recommendation Strength
- Strong recommendation: [conditions]
- Conditional recommendation: [conditions]
- Implementation priority: [HIGH / MEDIUM / LOW]
"""
    },

    "general": {
        "name": "General GDG Question",
        "description": "General clinical question for the guideline development group",
        "icon": "❓",
        "experts": [
            "Surgical Oncologist",
            "Palliative Care Physician",
            "GRADE Methodologist",
            "Clinical Evidence Specialist",
            "GDG Chair"
        ],
        "synthesis_focus": "general_recommendation",
        "output_template": """
## Summary of Evidence

### Key Findings
- [bullet points with evidence tags]

### Expert Consensus
- Areas of agreement: [bullet points]
- Areas of uncertainty: [bullet points]

### GRADE Assessment
| Outcome | Certainty | Direction |
|---------|-----------|-----------|

## Recommendation
[Clear statement with strength]

### Rationale
- [Supporting points]

### Applicability
- [Population considerations]
- [Setting considerations]

## Implementation Guidance
- [Practical steps]

## Research Priorities
- [What evidence is needed]
"""
    }
}


# =============================================================================
# GDG GUIDELINE SECTION MAPPING
# =============================================================================

GUIDELINE_SECTION_MAPPING: Dict[str, str] = {
    "surgical_candidate": "Patient Selection & Surgical Indications",
    "palliative_pathway": "Palliative Care Pathway",
    "intervention_choice": "Comparative Effectiveness",
    "symptom_management": "Symptom Management",
    "prognosis_assessment": "Prognosis & Outcomes",
    "ethics_review": "Ethical Considerations",
    "resource_allocation": "Implementation & Resources",
    "general": "General Recommendations"
}

# Legacy alias for backward compatibility
CDP_SECTION_MAPPING = GUIDELINE_SECTION_MAPPING


def get_guideline_section_name(question_type: str) -> str:
    """Get the guideline section name for a question type."""
    return GUIDELINE_SECTION_MAPPING.get(question_type, "General Recommendations")


# Legacy alias
def get_cdp_section_name(question_type: str) -> str:
    """Legacy alias for get_guideline_section_name."""
    return get_guideline_section_name(question_type)


# Map question types to CLINICAL_SCENARIOS in gdg_personas
QUESTION_TYPE_TO_SCENARIO = {
    "surgical_candidate": "malignant_bowel_obstruction",
    "palliative_pathway": "malignant_bowel_obstruction",
    "intervention_choice": "malignant_bowel_obstruction",
    "symptom_management": "malignant_bowel_obstruction",
    "prognosis_assessment": "malignant_bowel_obstruction",
    "ethics_review": "malignant_bowel_obstruction",
    "resource_allocation": "malignant_bowel_obstruction",
    "general": "custom"
}


def get_experts_for_question_type(question_type: str) -> List[str]:
    """
    Get auto-selected experts for a question type.

    Args:
        question_type: One of the QUESTION_TYPES keys

    Returns:
        List of expert names to consult
    """
    if question_type not in QUESTION_TYPES:
        # Default to general experts for unknown types
        base_experts = QUESTION_TYPES["general"]["experts"]
    else:
        base_experts = QUESTION_TYPES[question_type]["experts"]

    return base_experts


def get_question_type_info(question_type: str) -> Dict:
    """Get full info dict for a question type."""
    return QUESTION_TYPES.get(question_type, QUESTION_TYPES["general"])


def get_scenario_for_question_type(question_type: str) -> str:
    """Map question type to CLINICAL_SCENARIOS key."""
    return QUESTION_TYPE_TO_SCENARIO.get(question_type, "custom")


def get_synthesis_template(question_type: str) -> str:
    """Get the output template for synthesizing results."""
    info = get_question_type_info(question_type)
    return info.get("output_template", "")


def detect_question_type(question: str, api_key: Optional[str] = None) -> str:
    """
    Use LLM to auto-detect question type from free-form input.

    Args:
        question: The user's clinical question
        api_key: OpenAI API key (uses settings default if not provided)

    Returns:
        Detected question type key (e.g., "surgical_candidate", "symptom_management")
    """
    if not api_key:
        api_key = getattr(settings, 'OPENAI_API_KEY', None)

    if not api_key:
        # Fallback to keyword matching if no API key
        return _detect_question_type_keywords(question)

    try:
        client = OpenAI(api_key=api_key, timeout=30.0)

        type_descriptions = "\n".join([
            f"- {key}: {info['name']} - {info['description']}"
            for key, info in QUESTION_TYPES.items()
        ])

        response = client.chat.completions.create(
            model=getattr(settings, 'OPENAI_MODEL', 'gpt-4o-mini'),
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a palliative surgery guideline question classifier.

Given a clinical question, classify it into ONE of these types:
{type_descriptions}

Respond with ONLY the type key (e.g., "surgical_candidate", "symptom_management", etc.).
If uncertain, default to "general"."""
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=20,
            temperature=0
        )

        detected = response.choices[0].message.content.strip().lower()

        # Validate response
        if detected in QUESTION_TYPES:
            return detected

        # Try to match partial
        for key in QUESTION_TYPES:
            if key in detected:
                return key

        return "general"

    except Exception:
        return _detect_question_type_keywords(question)


def _detect_question_type_keywords(question: str) -> str:
    """
    Fallback keyword-based question type detection.

    Args:
        question: The user's clinical question

    Returns:
        Best-matching question type key
    """
    question_lower = question.lower()

    # Keyword patterns for each type
    patterns = {
        "surgical_candidate": [
            "surgery", "surgical", "candidate", "operate", "resection",
            "bypass", "procedure", "operative", "laparoscop", "palliative surgery",
            "eligible for surgery", "surgical intervention"
        ],
        "palliative_pathway": [
            "goals of care", "palliative", "hospice", "comfort care",
            "quality of life", "qol", "end of life", "prognosis discussion",
            "advance directive", "care planning"
        ],
        "intervention_choice": [
            "stent", "versus", "vs", "compare", "alternative", "embolization",
            "interventional", "endoscopic", "which approach", "better option",
            "surgical vs", "stent vs"
        ],
        "symptom_management": [
            "pain", "symptom", "nausea", "vomiting", "bowel obstruction",
            "bleeding", "dyspnea", "ascites", "management", "control",
            "intractable", "refractory"
        ],
        "prognosis_assessment": [
            "survival", "prognosis", "outcome", "evidence", "data",
            "mortality", "morbidity", "systematic review", "meta-analysis",
            "what does the evidence show"
        ],
        "ethics_review": [
            "ethics", "ethical", "appropriate", "futility", "consent",
            "decision-making", "autonomy", "should we", "right thing",
            "proportionate"
        ],
        "resource_allocation": [
            "cost", "resource", "implementation", "qaly", "icer",
            "cost-effective", "value", "budget", "healthcare system"
        ],
        "general": [
            "guideline", "recommend", "best practice", "standard"
        ]
    }

    # Score each type
    scores = {key: 0 for key in patterns}
    for key, keywords in patterns.items():
        for kw in keywords:
            if kw in question_lower:
                scores[key] += 1

    # Return highest scoring type, default to general
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def get_all_question_types() -> List[Dict]:
    """
    Get all question types with their info for UI display.

    Returns:
        List of dicts with key, name, icon, description
    """
    return [
        {
            "key": key,
            "name": info["name"],
            "icon": info["icon"],
            "description": info["description"]
        }
        for key, info in QUESTION_TYPES.items()
    ]
