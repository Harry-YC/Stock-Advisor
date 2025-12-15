"""
Program Extractor Service

Extracts program context (target, indication, drugs, competitors) from user questions.
Uses LLM with fallback to keyword matching for reliability.
"""

import logging
import json
from typing import Dict, Optional, List

from config import settings
from core.llm_utils import get_llm_client

logger = logging.getLogger(__name__)


class ProgramExtractor:
    """
    Extract and accumulate program context from user interactions.

    Usage:
        extractor = ProgramExtractor(api_key="...")
        context = extractor.extract_from_question("What is sotorasib efficacy in NSCLC?")
        # Returns: {"target": "KRAS G12C", "indication": "NSCLC", "drug_names": ["sotorasib"], ...}
    """

    def __init__(self, api_key: str, model: str = None):
        self.api_key = api_key
        self.model = model or getattr(settings, 'EXPERT_MODEL', 'gpt-4o-mini')

    def extract_from_question(self, question: str) -> Dict:
        """
        Extract program entities from a user question.

        Args:
            question: User's research question

        Returns:
            Dict with keys: target, indication, drug_names, competitors,
                           mechanism, therapeutic_area
        """
        try:
            return self._llm_extraction(question)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}, using fallback")
            return self._fallback_extraction(question)

    def _llm_extraction(self, question: str) -> Dict:
        """Use LLM to extract entities."""
        client = get_llm_client(api_key=self.api_key, model=self.model)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """Extract drug development program entities from the question.
Return JSON only, no markdown. Use null for unknown fields.

{
  "target": "molecular target (e.g., EGFR, KRAS G12C, PD-1)",
  "indication": "disease/condition (e.g., NSCLC, breast cancer)",
  "drug_names": ["list of drug names mentioned"],
  "competitors": ["competitor drugs if mentioned"],
  "mechanism": "mechanism of action if clear",
  "therapeutic_area": "oncology/immunology/neurology/cardiovascular/metabolic/etc"
}"""
                },
                {"role": "user", "content": question}
            ],
            max_completion_tokens=300,
            temperature=0.1
        )

        text = response.choices[0].message.content.strip()

        # Clean markdown if present
        if text.startswith("```"):
            lines = text.split("```")
            if len(lines) > 1:
                text = lines[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

        result = json.loads(text)

        # Ensure lists are lists
        if not isinstance(result.get("drug_names"), list):
            result["drug_names"] = []
        if not isinstance(result.get("competitors"), list):
            result["competitors"] = []

        logger.debug(f"LLM extracted: {result}")
        return result

    def _fallback_extraction(self, question: str) -> Dict:
        """Simple keyword-based extraction as fallback."""
        question_lower = question.lower()

        result = {
            "target": None,
            "indication": None,
            "drug_names": [],
            "competitors": [],
            "mechanism": None,
            "therapeutic_area": None
        }

        # Common targets (order matters - more specific first)
        targets = {
            "kras g12c": "KRAS G12C",
            "kras g12d": "KRAS G12D",
            "kras": "KRAS",
            "egfr t790m": "EGFR T790M",
            "egfr": "EGFR",
            "her2": "HER2",
            "her-2": "HER2",
            "pd-1": "PD-1",
            "pd-l1": "PD-L1",
            "ctla-4": "CTLA-4",
            "braf v600e": "BRAF V600E",
            "braf": "BRAF",
            "alk": "ALK",
            "ros1": "ROS1",
            "met": "MET",
            "ret": "RET",
            "ntrk": "NTRK",
            "fgfr": "FGFR",
            "brca": "BRCA",
            "parp": "PARP",
            "cdk4/6": "CDK4/6",
            "cdk 4/6": "CDK4/6",
            "bcl-2": "BCL-2",
            "btk": "BTK",
            "jak": "JAK",
            "vegf": "VEGF",
            "vegfr": "VEGFR"
        }
        for key, val in targets.items():
            if key in question_lower:
                result["target"] = val
                break

        # Common indications
        indications = {
            "non-small cell lung": "NSCLC",
            "nsclc": "NSCLC",
            "small cell lung": "SCLC",
            "sclc": "SCLC",
            "breast cancer": "breast cancer",
            "triple negative breast": "TNBC",
            "tnbc": "TNBC",
            "colorectal cancer": "colorectal cancer",
            "colorectal": "colorectal cancer",
            "crc": "colorectal cancer",
            "melanoma": "melanoma",
            "prostate cancer": "prostate cancer",
            "ovarian cancer": "ovarian cancer",
            "pancreatic cancer": "pancreatic cancer",
            "pancreatic": "pancreatic cancer",
            "gastric cancer": "gastric cancer",
            "hepatocellular": "HCC",
            "hcc": "HCC",
            "renal cell": "RCC",
            "rcc": "RCC",
            "bladder cancer": "bladder cancer",
            "urothelial": "urothelial carcinoma",
            "glioblastoma": "glioblastoma",
            "gbm": "glioblastoma",
            "multiple myeloma": "multiple myeloma",
            "aml": "AML",
            "acute myeloid": "AML",
            "cll": "CLL",
            "chronic lymphocytic": "CLL",
            "dlbcl": "DLBCL",
            "non-hodgkin": "NHL",
            "hodgkin": "Hodgkin lymphoma"
        }
        for key, val in indications.items():
            if key in question_lower:
                result["indication"] = val
                break

        # Common drugs
        drugs = [
            ("sotorasib", "sotorasib"),
            ("lumakras", "sotorasib"),
            ("adagrasib", "adagrasib"),
            ("krazati", "adagrasib"),
            ("osimertinib", "osimertinib"),
            ("tagrisso", "osimertinib"),
            ("pembrolizumab", "pembrolizumab"),
            ("keytruda", "pembrolizumab"),
            ("nivolumab", "nivolumab"),
            ("opdivo", "nivolumab"),
            ("atezolizumab", "atezolizumab"),
            ("tecentriq", "atezolizumab"),
            ("durvalumab", "durvalumab"),
            ("imfinzi", "durvalumab"),
            ("trastuzumab", "trastuzumab"),
            ("herceptin", "trastuzumab"),
            ("pertuzumab", "pertuzumab"),
            ("perjeta", "pertuzumab"),
            ("olaparib", "olaparib"),
            ("lynparza", "olaparib"),
            ("palbociclib", "palbociclib"),
            ("ibrance", "palbociclib"),
            ("ribociclib", "ribociclib"),
            ("kisqali", "ribociclib"),
            ("abemaciclib", "abemaciclib"),
            ("verzenio", "abemaciclib"),
            ("venetoclax", "venetoclax"),
            ("venclexta", "venetoclax"),
            ("ibrutinib", "ibrutinib"),
            ("imbruvica", "ibrutinib"),
            ("acalabrutinib", "acalabrutinib"),
            ("calquence", "acalabrutinib"),
            ("lorlatinib", "lorlatinib"),
            ("lorbrena", "lorlatinib"),
            ("crizotinib", "crizotinib"),
            ("xalkori", "crizotinib"),
            ("alectinib", "alectinib"),
            ("alecensa", "alectinib"),
            ("dabrafenib", "dabrafenib"),
            ("tafinlar", "dabrafenib"),
            ("trametinib", "trametinib"),
            ("mekinist", "trametinib"),
            ("encorafenib", "encorafenib"),
            ("braftovi", "encorafenib"),
            ("binimetinib", "binimetinib"),
            ("mektovi", "binimetinib"),
            ("larotrectinib", "larotrectinib"),
            ("vitrakvi", "larotrectinib"),
            ("entrectinib", "entrectinib"),
            ("rozlytrek", "entrectinib"),
            ("selpercatinib", "selpercatinib"),
            ("retevmo", "selpercatinib"),
            ("pralsetinib", "pralsetinib"),
            ("gavreto", "pralsetinib"),
            ("capmatinib", "capmatinib"),
            ("tabrecta", "capmatinib"),
            ("tepotinib", "tepotinib"),
            ("tepmetko", "tepotinib")
        ]

        found_drugs = set()
        for keyword, drug_name in drugs:
            if keyword in question_lower:
                found_drugs.add(drug_name)
        result["drug_names"] = list(found_drugs)

        # Infer therapeutic area
        oncology_keywords = ["cancer", "tumor", "carcinoma", "lymphoma", "leukemia",
                            "melanoma", "sarcoma", "myeloma", "oncol"]
        immunology_keywords = ["autoimmune", "rheumatoid", "lupus", "psoriasis",
                              "inflammatory", "crohn", "colitis"]

        if any(k in question_lower for k in oncology_keywords) or result["indication"]:
            result["therapeutic_area"] = "oncology"
        elif any(k in question_lower for k in immunology_keywords):
            result["therapeutic_area"] = "immunology"

        logger.debug(f"Fallback extracted: {result}")
        return result

    def merge_profiles(self, existing: Optional[Dict], new: Dict) -> Dict:
        """
        Merge new extractions with existing profile, keeping all info.

        Args:
            existing: Existing profile (can be None)
            new: Newly extracted profile

        Returns:
            Merged profile
        """
        if not existing:
            return new

        merged = existing.copy()

        # Update scalars only if new value exists and old doesn't
        for key in ["target", "indication", "mechanism", "therapeutic_area", "development_stage"]:
            if new.get(key) and not merged.get(key):
                merged[key] = new[key]

        # Merge lists (deduplicate, preserve order)
        for key in ["drug_names", "competitors"]:
            existing_list = merged.get(key) or []
            new_list = new.get(key) or []

            # Combine and deduplicate while preserving order
            seen = set()
            combined = []
            for item in existing_list + new_list:
                if item and item.lower() not in seen:
                    seen.add(item.lower())
                    combined.append(item)

            merged[key] = combined

        return merged
