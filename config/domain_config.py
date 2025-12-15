"""
Pluggable Domain Configuration System

Supports multiple guideline domains without scattering if-statements.
Each domain defines its own vocabulary, scoring weights, and LLM context.

Usage:
    from config.domain_config import get_domain_config, get_default_domain

    config = get_domain_config("palliative_surgery")
    union_filter = config.get_union_filter()
    exclusion_filter = config.get_exclusion_filter()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DomainConfig:
    """Configuration for a specific guideline domain."""

    name: str
    display_name: str

    # MeSH terms - split into verified vs candidate
    verified_mesh: List[str] = field(default_factory=list)
    candidate_mesh: List[str] = field(default_factory=list)  # Use as [tiab] if unverified

    # Keywords for scoring
    high_relevance_keywords: List[str] = field(default_factory=list)
    procedure_keywords: List[str] = field(default_factory=list)
    outcome_keywords: List[str] = field(default_factory=list)
    negative_keywords: List[str] = field(default_factory=list)

    # Query expansion terms (for union queries)
    expansion_terms: List[str] = field(default_factory=list)

    # Exclusion filter
    exclusion_terms: List[str] = field(default_factory=list)

    # Scoring weights
    domain_score_weight: float = 0.3  # Weight in composite score

    # LLM prompt context
    llm_context: str = ""

    def get_union_filter(self) -> str:
        """
        Build a UNION filter (OR-based) that preserves recall.
        Papers matching ANY of these are kept for reranking.
        """
        parts = []

        # Verified MeSH terms
        for term in self.verified_mesh:
            parts.append(term)

        # Candidate MeSH as tiab (safer)
        for term in self.candidate_mesh:
            # Strip [MeSH] if present, use [tiab]
            clean = term.replace("[MeSH]", "").replace("[mesh]", "").strip()
            parts.append(f'"{clean}"[tiab]')

        # Expansion terms as tiab
        for term in self.expansion_terms:
            parts.append(f'"{term}"[tiab]')

        if not parts:
            return ""

        return "(" + " OR ".join(parts) + ")"

    def get_exclusion_filter(self) -> str:
        """Build NOT filter for clearly irrelevant content."""
        if not self.exclusion_terms:
            return ""

        parts = [f'"{term}"[tiab]' for term in self.exclusion_terms]
        return "NOT (" + " OR ".join(parts) + ")"


# =============================================================================
# PALLIATIVE SURGERY DOMAIN CONFIG
# =============================================================================

PALLIATIVE_SURGERY_CONFIG = DomainConfig(
    name="palliative_surgery",
    display_name="Palliative Surgery",

    # VERIFIED MeSH headings (confirmed in MeSH browser)
    verified_mesh=[
        "Palliative Care[MeSH]",
        "Terminal Care[MeSH]",
        "Hospice Care[MeSH]",
        "Quality of Life[MeSH]",
        "Intestinal Obstruction[MeSH]",
        "Pleural Effusion, Malignant[MeSH]",
        "Gastric Outlet Obstruction[MeSH]",
        "Airway Obstruction[MeSH]",
        "Neoplasm Metastasis[MeSH]",
        "Pain Management[MeSH]",
        "Ascites[MeSH]",
        "Fractures, Spontaneous[MeSH]",
        "Bone Neoplasms[MeSH]",
        "Hemorrhage[MeSH]",
    ],

    # CANDIDATE MeSH (unverified - will use as [tiab])
    candidate_mesh=[
        "Palliative Surgery",  # Not a real MeSH heading
        "Symptom Control",
        "Comfort Care",
        "Surgical Palliation",
    ],

    # High relevance keywords (boost +0.08 each, max +0.4)
    high_relevance_keywords=[
        "palliative", "palliation", "palliate",
        "symptom control", "symptom relief", "symptom management",
        "quality of life", "qol",
        "end of life", "end-of-life",
        "terminal", "terminally ill",
        "advanced cancer", "metastatic",
        "inoperable", "unresectable", "incurable",
        "life expectancy", "limited prognosis",
        "prognosis", "survival",
        "functional status", "performance status",
        "hospice", "comfort care", "supportive care",
        "ecog", "karnofsky",
    ],

    # Procedure keywords (moderate boost +0.05 each, max +0.15)
    procedure_keywords=[
        "surgical palliation", "palliative surgery", "palliative procedure",
        "palliative resection", "palliative intervention",
        "bypass surgery", "surgical bypass",
        "stent placement", "stenting", "sems",
        "decompression", "drainage",
        "diversion", "colostomy", "ileostomy",
        "gastrostomy", "jejunostomy", "peg",
        "pleurodesis", "thoracentesis",
        "paracentesis", "peritoneal catheter",
        "fixation", "stabilization", "intramedullary nailing",
        "embolization", "debulking",
        "tracheostomy", "airway stent",
        "biliary stent", "ercp", "ptbd",
        "venting gastrostomy",
        "eus-ge", "eus-guided gastroenterostomy",
        "gastrojejunostomy",
    ],

    # Outcome keywords (moderate boost +0.03 each, max +0.1)
    outcome_keywords=[
        "symptom relief", "pain control", "pain relief",
        "obstruction relief",
        "functional improvement",
        "days at home", "hospital-free days", "time at home",
        "home discharge",
        "readmission", "reoperation",
        "morbidity", "mortality",
        "complication", "adverse event",
        "length of stay", "los",
        "patient satisfaction", "caregiver burden",
        "treatment burden",
    ],

    # Negative keywords (penalty -0.12 each, max -0.35)
    negative_keywords=[
        "curative", "curative intent", "curative resection",
        "adjuvant", "neoadjuvant",
        "radical resection", "radical surgery",
        "complete remission", "complete response",
        "disease-free survival", "dfs",
        "pediatric", "paediatric", "children", "infant",
        "congenital", "neonatal",
        "benign", "non-malignant", "nonmalignant",
        "prophylactic screening", "screening program",
        "cancer prevention", "chemoprevention",
        "animal model", "mouse model", "rat model",
        "in vitro", "cell line",
        "veterinary",
    ],

    # Expansion terms for union queries (preserves recall)
    expansion_terms=[
        "malignant", "unresectable", "advanced", "metastatic",
        "incurable", "carcinomatosis", "stage iv", "stage 4",
        "obstruction", "effusion", "hemorrhage", "bleeding",
        "symptom", "quality of life", "qol",
        "terminal", "prognosis", "palliat",
    ],

    # Exclusion terms for NOT filter
    exclusion_terms=[
        "pediatric", "neonatal", "infant", "child",
        "veterinary", "canine", "feline", "murine",
    ],

    # LLM prompt context for relevance scoring
    llm_context="""
CONTEXT: This search is for PALLIATIVE SURGERY clinical guidelines.

HIGHLY RELEVANT papers discuss:
- Surgical interventions for SYMPTOM PALLIATION (not curative intent)
- Management of advanced/metastatic cancer complications
- Quality of life outcomes in patients with limited life expectancy
- Comparison of surgical vs non-surgical palliation (stents, bypass, etc.)
- Malignant obstruction (bowel, gastric outlet, biliary, airway)
- Malignant effusions (pleural, pericardial, ascites)
- Pathologic fractures from bone metastases
- Bleeding control in advanced malignancy

NOT RELEVANT:
- Curative surgeries with intent to cure
- Adjuvant/neoadjuvant treatments for early cancer
- Early-stage cancer without metastases
- Pediatric populations (unless specifically asked)
- Benign conditions
- Screening/prevention programs
- Animal studies, in vitro studies
""",
)


# =============================================================================
# DOMAIN REGISTRY
# =============================================================================

DOMAIN_REGISTRY: Dict[str, DomainConfig] = {
    "palliative_surgery": PALLIATIVE_SURGERY_CONFIG,
    # Add more domains here as needed:
    # "oncology_general": ONCOLOGY_CONFIG,
    # "critical_care": CRITICAL_CARE_CONFIG,
}


def get_domain_config(domain: Optional[str]) -> Optional[DomainConfig]:
    """
    Get domain config by name, or None if not found.

    Args:
        domain: Domain name (e.g., "palliative_surgery") or None

    Returns:
        DomainConfig instance or None
    """
    if domain is None:
        return None
    return DOMAIN_REGISTRY.get(domain)


def get_default_domain() -> str:
    """Get the default domain for this application."""
    return "palliative_surgery"


def list_available_domains() -> List[str]:
    """Get list of available domain names."""
    return list(DOMAIN_REGISTRY.keys())
