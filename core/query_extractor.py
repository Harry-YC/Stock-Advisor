"""
Palliative Surgery Query Concept Extractor

Extracts clinical concepts from palliative surgery questions for intelligent search.
Ignores business words, extracts only relevant biomedical entities for palliative care.

Domain: Palliative surgery guideline development
Focus: Symptom palliation, surgical interventions, QoL outcomes
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# STOP TERMS - Words to NEVER include in PubMed queries
# =============================================================================

STOP_TERMS = {
    # Business/strategy words
    "make", "create", "plan", "strategy", "recommend", "suggest",
    "analyze", "compare", "evaluate", "assess", "discuss", "review",
    "should", "would", "could", "what", "how", "why", "when", "where",
    "help", "need", "want", "please", "can", "could", "will", "shall",

    # Common verbs
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "doing", "get", "got", "getting",

    # Connectors
    "and", "or", "but", "the", "a", "an", "of", "to", "for", "with",
    "in", "on", "at", "by", "from", "as", "about", "into", "through",

    # GDG-specific terms to ignore
    "gdg", "guideline", "recommendation", "grade", "evidence",
    "panel", "expert", "discussion", "consensus"
}

# =============================================================================
# PALLIATIVE SURGERY VOCABULARY
# =============================================================================

KNOWN_CONDITIONS = {
    # Bowel obstruction
    "malignant bowel obstruction", "mbo", "intestinal obstruction",
    "small bowel obstruction", "sbo", "large bowel obstruction",
    "colonic obstruction", "ileus",

    # Gastric/duodenal
    "gastric outlet obstruction", "goo", "duodenal obstruction",
    "pyloric obstruction", "gastroparesis",

    # Biliary
    "biliary obstruction", "malignant biliary obstruction",
    "jaundice", "cholestasis", "bile duct obstruction",

    # Airway
    "airway obstruction", "tracheal obstruction", "bronchial obstruction",
    "malignant airway obstruction", "stridor",

    # Pleural/peritoneal
    "malignant pleural effusion", "mpe", "pleural effusion",
    "malignant ascites", "peritoneal carcinomatosis", "ascites",

    # Bone metastases
    "pathologic fracture", "impending fracture", "bone metastasis",
    "bone metastases", "skeletal metastasis", "osteolytic lesion",

    # Spinal
    "spinal cord compression", "mscc", "metastatic spinal cord compression",
    "epidural compression", "vertebral metastasis",

    # Bleeding
    "tumor hemorrhage", "malignant bleeding", "gi bleeding",
    "gastrointestinal bleeding", "hemoptysis", "hematuria",

    # Pain
    "cancer pain", "intractable pain", "refractory pain",
    "neuropathic pain", "visceral pain", "bone pain",

    # Fistula
    "malignant fistula", "enterocutaneous fistula", "rectovaginal fistula",
    "tracheoesophageal fistula", "vesicovaginal fistula",

    # Other
    "dysphagia", "odynophagia", "cachexia", "anorexia",
    "nausea", "vomiting", "constipation",
}

KNOWN_ANATOMY = {
    # GI tract
    "esophagus", "esophageal", "stomach", "gastric", "duodenum", "duodenal",
    "jejunum", "jejunal", "ileum", "ileal", "colon", "colonic",
    "rectum", "rectal", "sigmoid", "cecum", "appendix",

    # Hepatobiliary
    "liver", "hepatic", "bile duct", "biliary", "gallbladder",
    "pancreas", "pancreatic", "ampulla",

    # Thoracic
    "lung", "pulmonary", "pleura", "pleural", "mediastinum",
    "trachea", "tracheal", "bronchus", "bronchial",

    # Bone
    "femur", "femoral", "humerus", "humeral", "tibia", "tibial",
    "spine", "spinal", "vertebra", "vertebral", "pelvis", "pelvic",
    "acetabulum", "rib",

    # Urologic
    "kidney", "renal", "ureter", "ureteral", "bladder", "vesical",
    "urethra", "urethral",

    # Other
    "peritoneum", "peritoneal", "retroperitoneum", "retroperitoneal",
    "mesentery", "mesenteric", "omentum",
}

KNOWN_PROCEDURES = {
    # GI surgery
    "gastrojejunostomy", "colostomy", "ileostomy", "bypass surgery",
    "intestinal bypass", "colectomy", "resection", "diverting ostomy",
    "loop colostomy", "end colostomy", "hartmann procedure",

    # Stenting
    "stent", "sems", "self-expanding metal stent",
    "duodenal stent", "esophageal stent", "biliary stent",
    "colonic stent", "rectal stent", "airway stent",
    "plastic stent", "covered stent", "uncovered stent",

    # Thoracic
    "pleurodesis", "thoracentesis", "indwelling pleural catheter",
    "ipc", "talc pleurodesis", "chest tube", "thoracostomy",
    "video-assisted thoracoscopy", "vats",

    # Orthopedic
    "surgical stabilization", "prophylactic fixation", "intramedullary nailing",
    "internal fixation", "arthroplasty", "hemiarthroplasty",
    "vertebroplasty", "kyphoplasty", "spinal decompression",
    "laminectomy", "corpectomy",

    # Interventional radiology
    "embolization", "tae", "tace", "ablation", "rfa",
    "radiofrequency ablation", "cryoablation", "microwave ablation",
    "nerve block", "celiac plexus block", "neurolysis",

    # EUS procedures
    "eus-ge", "eus-guided gastroenterostomy", "eus-bd",
    "eus-guided biliary drainage", "eus-cds", "eus-hgs",

    # Peritoneal
    "paracentesis", "peritoneal catheter", "peritoneovenous shunt",
    "denver shunt",

    # Other
    "debulking", "cytoreduction", "palliative resection",
    "tracheostomy", "peg", "gastrostomy", "jejunostomy",
    "nephrostomy", "ureteral stent",
}

KNOWN_OUTCOMES = {
    # Quality of life
    "quality of life", "qol", "hrqol", "health-related quality of life",
    "functional status", "performance status",

    # Symptom control
    "symptom control", "symptom palliation", "symptom relief",
    "pain control", "pain relief", "analgesia",

    # Survival
    "overall survival", "os", "survival", "median survival",
    "progression-free survival", "pfs",

    # Morbidity/mortality
    "morbidity", "mortality", "perioperative mortality",
    "postoperative complications", "complication rate",
    "30-day mortality", "90-day mortality",

    # Functional outcomes
    "oral intake", "diet tolerance", "gooss", "gastric outlet obstruction scoring system",
    "bowel function", "ambulation", "mobility",

    # Hospital metrics
    "length of stay", "los", "hospital stay", "readmission",
    "days at home", "hospice", "discharge disposition",

    # Procedure success
    "technical success", "clinical success", "patency",
    "reintervention", "stent migration", "stent occlusion",
}

KNOWN_SCORES = {
    # Fracture risk
    "mirels score", "mirels criteria", "mirels",
    "sin score", "spinal instability neoplastic score",

    # Performance status
    "ecog", "ecog performance status", "ecog ps",
    "karnofsky", "kps", "karnofsky performance status",
    "who performance status",

    # Frailty
    "frailty", "clinical frailty scale", "cfs",
    "frailty index", "fried frailty", "sarcopenia",

    # Prognostic scores
    "ppi", "palliative prognostic index",
    "pps", "palliative performance scale",
    "pap score", "palliative prognostic score",

    # Nutrition
    "must", "malnutrition universal screening tool",
    "pgsga", "patient-generated subjective global assessment",
    "bmi", "albumin",

    # Surgical risk
    "asa", "asa class", "asa score",
    "possum", "p-possum", "nsqip",
    "charlson comorbidity", "cci",

    # Disease-specific
    "gooss", "gastric outlet obstruction scoring system",
    "esas", "edmonton symptom assessment",
}

KNOWN_CANCERS = {
    # GI cancers (palliative context)
    "colorectal cancer", "colon cancer", "rectal cancer", "crc",
    "gastric cancer", "stomach cancer", "esophageal cancer",
    "pancreatic cancer", "pdac", "hepatocellular carcinoma", "hcc",
    "cholangiocarcinoma", "bile duct cancer",

    # Gynecologic
    "ovarian cancer", "ovarian carcinoma",
    "endometrial cancer", "uterine cancer", "cervical cancer",

    # Thoracic
    "lung cancer", "nsclc", "sclc", "mesothelioma",

    # Breast
    "breast cancer", "metastatic breast cancer",

    # Urologic
    "renal cell carcinoma", "rcc", "kidney cancer",
    "bladder cancer", "prostate cancer",

    # Other
    "melanoma", "sarcoma", "unknown primary",
    "metastatic cancer", "advanced cancer", "stage iv",
    "terminal cancer", "end-stage cancer",
}

KNOWN_COMPARISONS = {
    # Surgical vs non-surgical
    "versus", "vs", "compared to", "comparison",
    "surgical versus", "surgery versus", "operative versus",
    "conservative management", "best supportive care", "bsc",
    "watchful waiting", "observation",

    # Procedure comparisons
    "stent versus surgery", "stent vs surgery",
    "bypass versus stent", "resection versus bypass",
    "open versus laparoscopic", "minimally invasive",
}

# =============================================================================
# SYNONYM DICTIONARIES - For query expansion
# =============================================================================

CONDITION_SYNONYMS = {
    "mbo": ["malignant bowel obstruction", "MBO", "intestinal obstruction", "bowel obstruction"],
    "goo": ["gastric outlet obstruction", "GOO", "pyloric obstruction", "duodenal obstruction"],
    "mpe": ["malignant pleural effusion", "MPE", "pleural effusion", "pleural fluid"],
    "mscc": ["metastatic spinal cord compression", "MSCC", "spinal cord compression", "epidural compression"],
    "pathologic fracture": ["pathologic fracture", "pathological fracture", "metastatic fracture", "impending fracture"],
    "biliary obstruction": ["biliary obstruction", "bile duct obstruction", "malignant biliary obstruction", "obstructive jaundice"],
}

PROCEDURE_SYNONYMS = {
    "gastrojejunostomy": ["gastrojejunostomy", "GJ", "surgical bypass", "gastric bypass"],
    "sems": ["self-expanding metal stent", "SEMS", "metal stent", "duodenal stent", "enteral stent"],
    "eus-ge": ["EUS-guided gastroenterostomy", "EUS-GE", "endoscopic gastroenterostomy", "EUS-guided bypass"],
    "pleurodesis": ["pleurodesis", "talc pleurodesis", "chemical pleurodesis", "pleural fusion"],
    "ipc": ["indwelling pleural catheter", "IPC", "PleurX", "tunneled pleural catheter"],
    "prophylactic fixation": ["prophylactic fixation", "prophylactic stabilization", "preventive fixation", "impending fracture fixation"],
}

OUTCOME_SYNONYMS = {
    "qol": ["quality of life", "QoL", "HRQOL", "health-related quality of life"],
    "survival": ["overall survival", "OS", "median survival", "survival time"],
    "gooss": ["GOOSS", "gastric outlet obstruction scoring system", "oral intake score"],
    "symptom control": ["symptom palliation", "symptom relief", "symptom management", "palliative benefit"],
}

SCORE_SYNONYMS = {
    "mirels": ["Mirels score", "Mirels criteria", "Mirels", "fracture risk score"],
    "ecog": ["ECOG", "ECOG PS", "ECOG performance status", "performance status"],
    "karnofsky": ["Karnofsky", "KPS", "Karnofsky performance status"],
    "ppi": ["PPI", "palliative prognostic index", "prognostic index"],
}


def get_synonyms(term: str, term_type: str = "condition") -> List[str]:
    """
    Get synonyms for a palliative surgery term.

    Args:
        term: The term to look up
        term_type: One of "condition", "procedure", "outcome", "score"

    Returns:
        List of synonyms including the original term
    """
    key = term.lower().replace("-", "").replace(" ", "")

    synonym_dict = {
        "condition": CONDITION_SYNONYMS,
        "procedure": PROCEDURE_SYNONYMS,
        "outcome": OUTCOME_SYNONYMS,
        "score": SCORE_SYNONYMS,
    }.get(term_type, {})

    for k, v in synonym_dict.items():
        if k.replace("-", "").replace(" ", "") == key:
            return v
        for syn in v:
            if syn.lower().replace("-", "").replace(" ", "") == key:
                return v

    return [term]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExtractedConcepts:
    """Clinical concepts extracted from a palliative surgery question."""
    conditions: List[str] = field(default_factory=list)
    anatomy: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    scores: List[str] = field(default_factory=list)
    cancers: List[str] = field(default_factory=list)
    comparisons: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Check if no meaningful concepts were extracted."""
        return not (self.conditions or self.anatomy or self.procedures or
                    self.outcomes or self.cancers)

    def to_pubmed_queries(self) -> List[str]:
        """Generate multiple targeted PubMed queries for palliative surgery."""
        queries = []

        # Query 1: Condition + procedure (most specific for intervention questions)
        if self.conditions and self.procedures:
            c = " OR ".join([f'"{x}"[tiab]' for x in self.conditions[:2]])
            p = " OR ".join([f'"{x}"[tiab]' for x in self.procedures[:2]])
            queries.append(f"(({c}) AND ({p}))")

        # Query 2: Condition + anatomy + palliative
        if self.conditions and self.anatomy:
            c = self.conditions[0]
            a = self.anatomy[0]
            queries.append(f'("{c}"[tiab] AND "{a}"[tiab] AND palliative[tiab])')

        # Query 3: Cancer + condition (e.g., "ovarian cancer" + "bowel obstruction")
        if self.cancers and self.conditions:
            cancer = self.cancers[0]
            cond = self.conditions[0]
            queries.append(f'("{cancer}"[tiab] AND "{cond}"[tiab])')

        # Query 4: Procedure + outcome (for effectiveness questions)
        if self.procedures and self.outcomes:
            proc = self.procedures[0]
            out = self.outcomes[0]
            queries.append(f'("{proc}"[tiab] AND "{out}"[tiab])')

        # Query 5: Score-based queries (for risk assessment questions)
        if self.scores:
            score = self.scores[0]
            if self.conditions:
                queries.append(f'("{score}"[tiab] AND "{self.conditions[0]}"[tiab])')
            else:
                queries.append(f'("{score}"[tiab] AND (cancer[tiab] OR metastatic[tiab]))')

        # Query 6: Comparison queries
        if self.comparisons and self.procedures:
            proc1 = self.procedures[0]
            if len(self.procedures) > 1:
                proc2 = self.procedures[1]
                queries.append(f'("{proc1}"[tiab] AND "{proc2}"[tiab] AND (comparison[tiab] OR versus[tiab]))')

        # Query 7: Palliative surgery generic with condition
        if self.conditions and not queries:
            queries.append(f'("{self.conditions[0]}"[tiab] AND palliative[tiab] AND surgery[tiab])')

        # Deduplicate while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)

        return unique_queries[:5]  # Max 5 queries

    def to_trials_query(self) -> str:
        """Generate ClinicalTrials.gov query string."""
        parts = []

        # Prioritize: conditions, procedures, cancers
        for c in self.conditions[:2]:
            parts.append(c)
        for p in self.procedures[:1]:
            parts.append(p)
        for cancer in self.cancers[:1]:
            parts.append(cancer)

        # Add palliative context
        if parts:
            parts.append("palliative")

        return " AND ".join(parts) if parts else ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "conditions": self.conditions,
            "anatomy": self.anatomy,
            "procedures": self.procedures,
            "outcomes": self.outcomes,
            "scores": self.scores,
            "cancers": self.cancers,
            "comparisons": self.comparisons,
        }


# =============================================================================
# MAIN EXTRACTOR CLASS
# =============================================================================

class ClinicalQueryExtractor:
    """
    Extract clinical concepts from palliative surgery questions.

    Ignores business words like "recommend", "guideline", "discuss".
    Extracts conditions, procedures, outcomes, etc.
    """

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize extractor.

        Args:
            api_key: API key for LLM extraction
            model: Model to use (defaults to settings.EXPERT_MODEL)
        """
        self.api_key = api_key or getattr(settings, 'OPENAI_API_KEY', None) or getattr(settings, 'GOOGLE_API_KEY', None)
        self.model = model or getattr(settings, 'EXPERT_MODEL', 'gemini-3-pro-preview')

    def extract(self, question: str) -> ExtractedConcepts:
        """
        Extract concepts using LLM with pattern fallback.

        Args:
            question: The clinical question to parse

        Returns:
            ExtractedConcepts with conditions, procedures, etc.
        """
        # Try LLM extraction first
        if self.api_key:
            try:
                return self._extract_llm(question)
            except Exception as e:
                logger.warning(f"LLM extraction failed, using pattern fallback: {e}")

        # Fallback to pattern-based extraction
        return self._extract_patterns(question)

    def _extract_llm(self, question: str) -> ExtractedConcepts:
        """Use LLM for extraction."""
        from core.llm_utils import get_llm_client

        client = get_llm_client(api_key=self.api_key, model=self.model)

        response = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "system",
                "content": """Extract clinical terms from this palliative surgery question.

IMPORTANT: IGNORE business/strategy words like: recommend, guideline, discuss, analyze, compare, evaluate.

Return JSON only:
{
  "conditions": ["clinical conditions: malignant bowel obstruction, pathologic fracture, pleural effusion, etc."],
  "anatomy": ["anatomical locations: femur, colon, spine, pleura, etc."],
  "procedures": ["interventions: gastrojejunostomy, stent, pleurodesis, surgical stabilization, etc."],
  "outcomes": ["outcomes measured: quality of life, survival, symptom control, morbidity, etc."],
  "scores": ["clinical scores: Mirels, ECOG, Karnofsky, frailty, PPI, etc."],
  "cancers": ["cancer types: ovarian, colorectal, pancreatic, lung, etc."],
  "comparisons": ["comparison type: versus, compared to, vs, etc."]
}

Rules:
1. Use standard clinical terminology
2. Only include terms ACTUALLY present in the question
3. For abbreviations like "MBO", expand to full term
4. Return empty arrays if category not found
5. Focus on palliative surgery context"""
            }, {
                "role": "user",
                "content": question
            }],
            max_tokens=500,
            temperature=0.1
        )

        text = response.choices[0].message.content
        if not text:
            logger.warning("LLM returned empty response, falling back to patterns")
            return self._extract_patterns(question)
        text = text.strip()

        # Clean JSON from markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}, falling back to patterns")
            return self._extract_patterns(question)

        return ExtractedConcepts(
            conditions=data.get("conditions", []),
            anatomy=data.get("anatomy", []),
            procedures=data.get("procedures", []),
            outcomes=data.get("outcomes", []),
            scores=data.get("scores", []),
            cancers=data.get("cancers", []),
            comparisons=data.get("comparisons", []),
        )

    def _extract_patterns(self, question: str) -> ExtractedConcepts:
        """Fallback pattern-based extraction."""
        q_lower = question.lower()
        concepts = ExtractedConcepts()

        # Extract conditions
        for term in KNOWN_CONDITIONS:
            pattern = r'\b' + re.escape(term) + r'\b'
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                extracted = match.group(0)
                # Normalize short acronyms to uppercase
                if len(extracted) <= 4 and extracted.isalpha():
                    extracted = extracted.upper()
                if extracted not in concepts.conditions:
                    concepts.conditions.append(extracted)

        # Extract anatomy
        for term in KNOWN_ANATOMY:
            if term in q_lower:
                if term not in concepts.anatomy:
                    concepts.anatomy.append(term)

        # Extract procedures
        for term in KNOWN_PROCEDURES:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, question, re.IGNORECASE):
                if term not in concepts.procedures:
                    concepts.procedures.append(term)

        # Extract outcomes
        for term in KNOWN_OUTCOMES:
            if term in q_lower:
                if term not in concepts.outcomes:
                    concepts.outcomes.append(term)

        # Extract scores
        for term in KNOWN_SCORES:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, question, re.IGNORECASE):
                if term not in concepts.scores:
                    concepts.scores.append(term)

        # Extract cancers
        for term in KNOWN_CANCERS:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, question, re.IGNORECASE):
                if term not in concepts.cancers:
                    concepts.cancers.append(term)

        # Extract comparisons
        for term in KNOWN_COMPARISONS:
            if term in q_lower:
                if term not in concepts.comparisons:
                    concepts.comparisons.append(term)

        return concepts


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_search_queries(question: str, api_key: str = None) -> Dict[str, any]:
    """
    Main entry point: question -> multiple targeted queries.

    Args:
        question: Clinical question to parse
        api_key: API key for LLM extraction

    Returns:
        {
            "pubmed": ["query1", "query2", ...],
            "trials": "trials query",
            "concepts": ExtractedConcepts dict
        }
    """
    extractor = ClinicalQueryExtractor(api_key=api_key)
    concepts = extractor.extract(question)

    if concepts.is_empty():
        logger.warning(f"No concepts extracted from: {question[:100]}...")
        return {"pubmed": [], "trials": "", "concepts": {}}

    return {
        "pubmed": concepts.to_pubmed_queries(),
        "trials": concepts.to_trials_query(),
        "concepts": concepts.to_dict(),
    }


def extract_concepts(question: str, api_key: str = None) -> ExtractedConcepts:
    """
    Extract concepts from a palliative surgery question.

    Args:
        question: Clinical question to parse
        api_key: API key for LLM extraction

    Returns:
        ExtractedConcepts dataclass
    """
    extractor = ClinicalQueryExtractor(api_key=api_key)
    return extractor.extract(question)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import os

    test_questions = [
        "Should a patient with malignant bowel obstruction from ovarian cancer undergo surgery or stent placement?",
        "What is the role of prophylactic fixation for femur metastasis with Mirels score of 9?",
        "Compare gastrojejunostomy versus duodenal stent for gastric outlet obstruction",
        "Palliative pleurodesis versus indwelling pleural catheter for quality of life in malignant pleural effusion",
        "When should surgical stabilization be performed for pathologic fracture with ECOG 3?",
    ]

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
    extractor = ClinicalQueryExtractor(api_key=api_key)

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        print("-" * 60)

        concepts = extractor.extract(q)
        print(f"Conditions: {concepts.conditions}")
        print(f"Anatomy: {concepts.anatomy}")
        print(f"Procedures: {concepts.procedures}")
        print(f"Outcomes: {concepts.outcomes}")
        print(f"Scores: {concepts.scores}")
        print(f"Cancers: {concepts.cancers}")

        print("\nPubMed queries:")
        for i, query in enumerate(concepts.to_pubmed_queries(), 1):
            print(f"  {i}. {query}")

        print(f"\nTrials query: {concepts.to_trials_query()}")
