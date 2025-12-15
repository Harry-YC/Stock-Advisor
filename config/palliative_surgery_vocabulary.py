"""
Palliative Surgery Domain Vocabulary

Curated MeSH terms, synonyms, procedures, and conditions
for domain-specific search enhancement.

Used by:
- core/query_parser.py - Domain context injection
- core/ranking.py - Relevance scoring boost/penalty
- services/search_service.py - Threshold filtering
"""

from typing import Dict, List

# =============================================================================
# CORE MESH HEADINGS FOR PALLIATIVE SURGERY
# =============================================================================

MESH_HEADINGS: Dict[str, List[str]] = {
    "palliative_care": [
        "Palliative Care[MeSH]",
        "Terminal Care[MeSH]",
        "Hospice Care[MeSH]",
        "Palliative Medicine[MeSH]",
    ],
    "surgical_procedures": [
        "Palliative Surgery[MeSH]",
        "Surgical Procedures, Operative[MeSH]",
        "Digestive System Surgical Procedures[MeSH]",
        "Thoracic Surgical Procedures[MeSH]",
        "Orthopedic Procedures[MeSH]",
    ],
    "symptom_management": [
        "Pain Management[MeSH]",
        "Symptom Assessment[MeSH]",
        "Quality of Life[MeSH]",
        "Patient Comfort[MeSH]",
    ],
    "oncology": [
        "Neoplasms[MeSH]",
        "Neoplasm Metastasis[MeSH]",
        "Neoplasm Staging[MeSH]",
        "Cancer Pain[MeSH]",
    ],
}

# =============================================================================
# SPECIFIC PALLIATIVE PROCEDURES BY CONDITION
# =============================================================================

PALLIATIVE_PROCEDURES: Dict[str, Dict[str, List[str]]] = {
    "malignant_bowel_obstruction": {
        "mesh": ["Intestinal Obstruction[MeSH]", "Bowel Obstruction[MeSH]"],
        "synonyms": ["MBO", "malignant intestinal obstruction", "carcinomatosis",
                     "bowel obstruction", "small bowel obstruction", "large bowel obstruction"],
        "procedures": ["gastrostomy", "jejunostomy", "colostomy", "ileostomy",
                       "stent", "bypass", "diversion", "decompression",
                       "venting gastrostomy", "PEG", "surgical bypass"],
    },
    "malignant_pleural_effusion": {
        "mesh": ["Pleural Effusion, Malignant[MeSH]", "Pleural Effusion[MeSH]"],
        "synonyms": ["MPE", "malignant effusion", "pleural metastasis"],
        "procedures": ["pleurodesis", "thoracentesis", "indwelling pleural catheter",
                       "IPC", "talc pleurodesis", "VATS", "PleurX"],
    },
    "pathologic_fracture": {
        "mesh": ["Fractures, Spontaneous[MeSH]", "Bone Neoplasms[MeSH]",
                 "Fractures, Bone[MeSH]"],
        "synonyms": ["metastatic fracture", "impending fracture", "pathological fracture",
                     "bone metastasis", "skeletal metastasis"],
        "procedures": ["intramedullary nailing", "arthroplasty", "fixation",
                       "prophylactic fixation", "stabilization", "orthopedic fixation",
                       "vertebroplasty", "kyphoplasty"],
    },
    "bleeding_control": {
        "mesh": ["Hemorrhage[MeSH]", "Hemostasis, Surgical[MeSH]"],
        "synonyms": ["tumor hemorrhage", "uncontrolled bleeding", "tumor bleeding",
                     "malignant hemorrhage", "cancer-related bleeding"],
        "procedures": ["embolization", "ligation", "cautery", "packing",
                       "interventional radiology", "TAE", "transcatheter embolization"],
    },
    "airway_obstruction": {
        "mesh": ["Airway Obstruction[MeSH]", "Tracheal Stenosis[MeSH]"],
        "synonyms": ["malignant airway obstruction", "tracheal obstruction",
                     "bronchial obstruction", "central airway obstruction"],
        "procedures": ["tracheostomy", "stent", "debulking", "laser ablation",
                       "bronchoscopy", "airway stent", "endobronchial therapy"],
    },
    "gastric_outlet_obstruction": {
        "mesh": ["Gastric Outlet Obstruction[MeSH]", "Pyloric Stenosis[MeSH]",
                 "Duodenal Obstruction[MeSH]"],
        "synonyms": ["GOO", "duodenal obstruction", "pyloric obstruction",
                     "malignant GOO", "gastric obstruction"],
        "procedures": ["gastrojejunostomy", "duodenal stent", "SEMS",
                       "EUS-guided gastroenterostomy", "EUS-GE", "surgical bypass",
                       "self-expanding metal stent"],
    },
    "biliary_obstruction": {
        "mesh": ["Cholestasis[MeSH]", "Bile Duct Obstruction[MeSH]",
                 "Jaundice, Obstructive[MeSH]"],
        "synonyms": ["malignant biliary obstruction", "jaundice", "obstructive jaundice",
                     "biliary stricture", "malignant stricture"],
        "procedures": ["biliary stent", "ERCP", "PTBD", "hepaticojejunostomy",
                       "choledochojejunostomy", "biliary drainage", "EUS-BD"],
    },
    "malignant_ascites": {
        "mesh": ["Ascites[MeSH]"],
        "synonyms": ["malignant ascites", "peritoneal carcinomatosis",
                     "refractory ascites"],
        "procedures": ["paracentesis", "peritoneal catheter", "shunt",
                       "peritoneovenous shunt", "TIPS", "PleurX"],
    },
}

# =============================================================================
# DOMAIN KEYWORDS FOR RELEVANCE SCORING
# =============================================================================

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "high_relevance": [
        # Palliative intent
        "palliative", "palliation", "palliate",
        "symptom control", "symptom relief", "symptom management",
        "quality of life", "QoL",
        "end of life", "end-of-life",
        "terminal", "terminally ill",
        # Disease context
        "advanced cancer", "metastatic",
        "inoperable", "unresectable", "incurable",
        "stage IV", "stage 4",
        "life expectancy", "limited prognosis",
        "prognosis", "survival",
        # Functional status
        "functional status", "performance status",
        "ECOG", "Karnofsky",
    ],
    "procedure_specific": [
        # Surgical management
        "surgical management", "operative intervention",
        "bypass surgery", "diversion",
        "stent placement", "stenting",
        "decompression", "drainage",
        "fixation", "stabilization",
        # Palliative procedures
        "palliative surgery", "palliative procedure",
        "palliative resection", "palliative intervention",
        "symptom-directed", "symptom-relieving",
    ],
    "outcomes": [
        # Relief outcomes
        "symptom relief", "pain control", "pain relief",
        "obstruction relief", "decompression",
        # QoL outcomes
        "quality of life", "functional improvement",
        "days at home", "hospital-free days",
        "time at home", "home discharge",
        # Clinical outcomes
        "readmission", "reoperation",
        "morbidity", "mortality",
        "complication", "adverse event",
        "length of stay", "LOS",
    ],
    "negative_indicators": [
        # Curative intent (not palliative)
        "curative", "curative intent", "curative resection",
        "adjuvant", "neoadjuvant",
        "radical resection", "radical surgery",
        "complete remission", "complete response",
        "disease-free survival", "DFS",
        # Excluded populations
        "pediatric", "paediatric", "children", "infant",
        "congenital", "neonatal",
        # Non-malignant
        "benign", "non-malignant", "nonmalignant",
        # Screening/prevention
        "prophylactic screening", "screening program",
        "cancer prevention", "chemoprevention",
        # Animal/basic science
        "animal model", "mouse model", "rat model",
        "in vitro", "cell line",
        "veterinary",
    ],
}

# =============================================================================
# DEFAULT DOMAIN FILTER (applied to PubMed queries)
# =============================================================================

DEFAULT_DOMAIN_FILTER = """
(
  "Palliative Care"[MeSH] OR
  "Terminal Care"[MeSH] OR
  palliative[tiab] OR
  palliation[tiab] OR
  "symptom control"[tiab] OR
  "symptom relief"[tiab] OR
  "end of life"[tiab] OR
  "advanced cancer"[tiab] OR
  metastatic[tiab] OR
  inoperable[tiab] OR
  unresectable[tiab]
)
"""

# Exclusion filter for clearly irrelevant content
EXCLUSION_FILTER = """
NOT (
  "Pediatrics"[MeSH] OR
  pediatric[tiab] OR
  paediatric[tiab] OR
  children[tiab] OR
  congenital[tiab] OR
  benign[tiab] OR
  veterinary[tiab] OR
  "animal model"[tiab] OR
  "in vitro"[tiab]
)
"""

# Combined filter (domain + exclusion)
FULL_DOMAIN_FILTER = f"({DEFAULT_DOMAIN_FILTER.strip()}) {EXCLUSION_FILTER.strip()}"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_high_relevance_keywords() -> List[str]:
    """Get flat list of all high-relevance keywords."""
    return (
        DOMAIN_KEYWORDS["high_relevance"] +
        DOMAIN_KEYWORDS["procedure_specific"] +
        DOMAIN_KEYWORDS["outcomes"]
    )


def get_negative_keywords() -> List[str]:
    """Get flat list of negative indicator keywords."""
    return DOMAIN_KEYWORDS["negative_indicators"]


def get_procedure_terms(condition: str) -> Dict[str, List[str]]:
    """
    Get MeSH, synonyms, and procedures for a specific condition.

    Args:
        condition: Key from PALLIATIVE_PROCEDURES (e.g., 'malignant_bowel_obstruction')

    Returns:
        Dict with 'mesh', 'synonyms', 'procedures' lists, or empty dict if not found
    """
    return PALLIATIVE_PROCEDURES.get(condition, {})


def get_all_procedure_terms() -> List[str]:
    """Get flat list of all procedure-related terms."""
    all_terms = []
    for condition, terms in PALLIATIVE_PROCEDURES.items():
        all_terms.extend(terms.get("synonyms", []))
        all_terms.extend(terms.get("procedures", []))
    return list(set(all_terms))
