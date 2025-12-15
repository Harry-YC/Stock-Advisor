"""
Automated Evidence Table Extractor

Uses GPT to extract structured data from paper abstracts for evidence synthesis.
Extracts:
- Study design and sample size
- Interventions and comparators
- Primary/secondary endpoints
- PK parameters (Cmax, AUC, t1/2, clearance, bioavailability)
- Efficacy results
- Safety/adverse events

Outputs structured DataFrames for Excel export.
"""

import json
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


import os


class StudyDesign(Enum):
    """Common study design types"""
    RCT = "Randomized Controlled Trial"
    OBSERVATIONAL = "Observational Study"
    COHORT = "Cohort Study"
    CASE_CONTROL = "Case-Control Study"
    CROSS_SECTIONAL = "Cross-sectional Study"
    CASE_SERIES = "Case Series"
    CASE_REPORT = "Case Report"
    META_ANALYSIS = "Meta-Analysis"
    SYSTEMATIC_REVIEW = "Systematic Review"
    PHASE_1 = "Phase 1 Clinical Trial"
    PHASE_2 = "Phase 2 Clinical Trial"
    PHASE_3 = "Phase 3 Clinical Trial"
    PHASE_4 = "Phase 4 Clinical Trial"
    PRECLINICAL = "Preclinical Study"
    IN_VITRO = "In Vitro Study"
    UNKNOWN = "Unknown"


@dataclass
class PKParameters:
    """Pharmacokinetic parameters extracted from a study"""
    cmax: Optional[str] = None
    cmax_unit: Optional[str] = None
    tmax: Optional[str] = None
    tmax_unit: Optional[str] = None
    auc: Optional[str] = None
    auc_unit: Optional[str] = None
    half_life: Optional[str] = None
    half_life_unit: Optional[str] = None
    clearance: Optional[str] = None
    clearance_unit: Optional[str] = None
    volume_distribution: Optional[str] = None
    bioavailability: Optional[str] = None
    protein_binding: Optional[str] = None


@dataclass
class EfficacyResult:
    """Efficacy results extracted from a study"""
    endpoint: str
    result: str
    p_value: Optional[str] = None
    confidence_interval: Optional[str] = None
    effect_size: Optional[str] = None
    comparison: Optional[str] = None


@dataclass
class SafetyResult:
    """Safety/adverse event data"""
    event: str
    incidence: Optional[str] = None
    severity: Optional[str] = None
    comparison_to_control: Optional[str] = None


@dataclass
class ExtractedEvidence:
    """Complete extracted evidence from a paper"""
    pmid: str
    title: str
    authors: str
    year: str
    journal: str

    # Study characteristics
    study_design: str = ""
    sample_size: Optional[int] = None
    population: str = ""
    duration: str = ""

    # Interventions
    intervention: str = ""
    comparator: str = ""
    dosing: str = ""

    # Outcomes
    primary_endpoint: str = ""
    secondary_endpoints: List[str] = field(default_factory=list)

    # Results
    pk_parameters: Optional[PKParameters] = None
    efficacy_results: List[EfficacyResult] = field(default_factory=list)
    safety_results: List[SafetyResult] = field(default_factory=list)

    # Quality indicators
    extraction_confidence: float = 0.0
    notes: str = ""


class EvidenceExtractor:
    """Extracts structured evidence from paper abstracts using GPT"""

    EXTRACTION_PROMPT = """You are an expert at extracting structured data from medical/scientific paper abstracts.

Extract the following information from the abstract. Return a JSON object with these fields:

{
    "study_design": "Type of study (RCT, Phase 2, Observational, etc.)",
    "sample_size": numeric or null,
    "population": "Description of study population",
    "duration": "Study duration if mentioned",
    "intervention": "Drug/treatment being studied",
    "comparator": "Control/comparator if any",
    "dosing": "Dosing regimen if mentioned",
    "primary_endpoint": "Primary outcome measure",
    "secondary_endpoints": ["list of secondary endpoints"],
    "pk_parameters": {
        "cmax": "value or null",
        "cmax_unit": "unit or null",
        "tmax": "value or null",
        "tmax_unit": "unit or null",
        "auc": "value or null",
        "auc_unit": "unit or null",
        "half_life": "value or null",
        "half_life_unit": "unit or null",
        "clearance": "value or null",
        "clearance_unit": "unit or null",
        "bioavailability": "percentage or null",
        "protein_binding": "percentage or null"
    },
    "efficacy_results": [
        {
            "endpoint": "outcome name",
            "result": "result value/description",
            "p_value": "p-value if reported",
            "confidence_interval": "95% CI if reported",
            "effect_size": "effect size if reported"
        }
    ],
    "safety_results": [
        {
            "event": "adverse event name",
            "incidence": "percentage or count",
            "severity": "mild/moderate/severe if mentioned"
        }
    ],
    "confidence": 0.0 to 1.0 (how confident you are in the extraction),
    "notes": "Any important notes or limitations"
}

If a field is not mentioned in the abstract, use null or empty array.
Be precise with numbers and units. Extract exact values when available.

ABSTRACT:
{abstract}

Return ONLY the JSON object, no other text."""

    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-5-mini"):
        """
        Initialize the evidence extractor

        Args:
            openai_api_key: OpenAI API key (uses env var if not provided)
            model: OpenAI model to use
        """
        self.model = model
        self.client = None

        if OPENAI_AVAILABLE:
            from core.llm_utils import get_llm_client
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.client = get_llm_client(api_key=api_key, model=model)
                except Exception:
                    pass

    def extract_from_abstract(
        self,
        pmid: str,
        title: str,
        authors: List[str],
        year: str,
        journal: str,
        abstract: str
    ) -> ExtractedEvidence:
        """
        Extract structured evidence from a single paper abstract

        Args:
            pmid: PubMed ID
            title: Paper title
            authors: List of author names
            year: Publication year
            journal: Journal name
            abstract: Paper abstract text

        Returns:
            ExtractedEvidence object with extracted data
        """
        # Create base evidence object
        evidence = ExtractedEvidence(
            pmid=pmid,
            title=title,
            authors=", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
            year=year,
            journal=journal
        )

        if not abstract or not self.client:
            evidence.notes = "No abstract available or OpenAI not configured"
            return evidence

        try:
            # Call GPT for extraction
            prompt = self.EXTRACTION_PROMPT.format(abstract=abstract)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\n?', '', content)
                content = re.sub(r'\n?```$', '', content)

            data = json.loads(content)

            # Populate evidence object
            evidence.study_design = data.get("study_design", "")
            evidence.sample_size = data.get("sample_size")
            evidence.population = data.get("population", "")
            evidence.duration = data.get("duration", "")
            evidence.intervention = data.get("intervention", "")
            evidence.comparator = data.get("comparator", "")
            evidence.dosing = data.get("dosing", "")
            evidence.primary_endpoint = data.get("primary_endpoint", "")
            evidence.secondary_endpoints = data.get("secondary_endpoints", [])

            # Parse PK parameters
            pk_data = data.get("pk_parameters", {})
            if pk_data and any(v for v in pk_data.values() if v):
                evidence.pk_parameters = PKParameters(
                    cmax=pk_data.get("cmax"),
                    cmax_unit=pk_data.get("cmax_unit"),
                    tmax=pk_data.get("tmax"),
                    tmax_unit=pk_data.get("tmax_unit"),
                    auc=pk_data.get("auc"),
                    auc_unit=pk_data.get("auc_unit"),
                    half_life=pk_data.get("half_life"),
                    half_life_unit=pk_data.get("half_life_unit"),
                    clearance=pk_data.get("clearance"),
                    clearance_unit=pk_data.get("clearance_unit"),
                    bioavailability=pk_data.get("bioavailability"),
                    protein_binding=pk_data.get("protein_binding")
                )

            # Parse efficacy results
            for eff in data.get("efficacy_results", []):
                if eff.get("endpoint"):
                    evidence.efficacy_results.append(EfficacyResult(
                        endpoint=eff.get("endpoint", ""),
                        result=eff.get("result", ""),
                        p_value=eff.get("p_value"),
                        confidence_interval=eff.get("confidence_interval"),
                        effect_size=eff.get("effect_size")
                    ))

            # Parse safety results
            for safety in data.get("safety_results", []):
                if safety.get("event"):
                    evidence.safety_results.append(SafetyResult(
                        event=safety.get("event", ""),
                        incidence=safety.get("incidence"),
                        severity=safety.get("severity")
                    ))

            evidence.extraction_confidence = data.get("confidence", 0.5)
            evidence.notes = data.get("notes", "")

        except json.JSONDecodeError as e:
            evidence.notes = f"JSON parsing error: {e}"
            evidence.extraction_confidence = 0.0
        except Exception as e:
            evidence.notes = f"Extraction error: {e}"
            evidence.extraction_confidence = 0.0

        return evidence

    def batch_extract(
        self,
        papers: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> List[ExtractedEvidence]:
        """
        Extract evidence from multiple papers

        Args:
            papers: List of dicts with pmid, title, authors, year, journal, abstract
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of ExtractedEvidence objects
        """
        results = []

        for i, paper in enumerate(papers):
            evidence = self.extract_from_abstract(
                pmid=paper.get("pmid", ""),
                title=paper.get("title", ""),
                authors=paper.get("authors", []),
                year=paper.get("year", ""),
                journal=paper.get("journal", ""),
                abstract=paper.get("abstract", "")
            )
            results.append(evidence)

            if progress_callback:
                progress_callback(i + 1, len(papers))

        return results


def evidence_to_dataframe(evidence_list: List[ExtractedEvidence]) -> Optional[Any]:
    """
    Convert list of ExtractedEvidence to a pandas DataFrame

    Args:
        evidence_list: List of ExtractedEvidence objects

    Returns:
        pandas DataFrame or None if pandas not available
    """
    if not PANDAS_AVAILABLE:
        return None

    rows = []
    for ev in evidence_list:
        row = {
            "PMID": ev.pmid,
            "Title": ev.title,
            "Authors": ev.authors,
            "Year": ev.year,
            "Journal": ev.journal,
            "Study Design": ev.study_design,
            "Sample Size": ev.sample_size,
            "Population": ev.population,
            "Duration": ev.duration,
            "Intervention": ev.intervention,
            "Comparator": ev.comparator,
            "Dosing": ev.dosing,
            "Primary Endpoint": ev.primary_endpoint,
            "Secondary Endpoints": "; ".join(ev.secondary_endpoints),
            "Confidence": ev.extraction_confidence,
            "Notes": ev.notes
        }

        # Add PK parameters
        if ev.pk_parameters:
            pk = ev.pk_parameters
            row["Cmax"] = f"{pk.cmax} {pk.cmax_unit or ''}" if pk.cmax else ""
            row["Tmax"] = f"{pk.tmax} {pk.tmax_unit or ''}" if pk.tmax else ""
            row["AUC"] = f"{pk.auc} {pk.auc_unit or ''}" if pk.auc else ""
            row["Half-life"] = f"{pk.half_life} {pk.half_life_unit or ''}" if pk.half_life else ""
            row["Clearance"] = f"{pk.clearance} {pk.clearance_unit or ''}" if pk.clearance else ""
            row["Bioavailability"] = pk.bioavailability or ""
        else:
            row["Cmax"] = ""
            row["Tmax"] = ""
            row["AUC"] = ""
            row["Half-life"] = ""
            row["Clearance"] = ""
            row["Bioavailability"] = ""

        # Add efficacy summary
        if ev.efficacy_results:
            efficacy_strs = []
            for eff in ev.efficacy_results:
                s = f"{eff.endpoint}: {eff.result}"
                if eff.p_value:
                    s += f" (p={eff.p_value})"
                efficacy_strs.append(s)
            row["Efficacy Results"] = "; ".join(efficacy_strs)
        else:
            row["Efficacy Results"] = ""

        # Add safety summary
        if ev.safety_results:
            safety_strs = []
            for saf in ev.safety_results:
                s = f"{saf.event}"
                if saf.incidence:
                    s += f": {saf.incidence}"
                safety_strs.append(s)
            row["Safety/AEs"] = "; ".join(safety_strs)
        else:
            row["Safety/AEs"] = ""

        rows.append(row)

    return pd.DataFrame(rows)


def pk_parameters_to_dataframe(evidence_list: List[ExtractedEvidence]) -> Optional[Any]:
    """
    Create a focused DataFrame of PK parameters only

    Args:
        evidence_list: List of ExtractedEvidence objects

    Returns:
        pandas DataFrame with PK data or None
    """
    if not PANDAS_AVAILABLE:
        return None

    rows = []
    for ev in evidence_list:
        if not ev.pk_parameters:
            continue

        pk = ev.pk_parameters
        if not any([pk.cmax, pk.auc, pk.half_life, pk.clearance, pk.bioavailability]):
            continue

        row = {
            "PMID": ev.pmid,
            "Title": ev.title[:80] + "..." if len(ev.title) > 80 else ev.title,
            "Year": ev.year,
            "Intervention": ev.intervention,
            "Dosing": ev.dosing,
            "Cmax": pk.cmax,
            "Cmax Unit": pk.cmax_unit,
            "Tmax": pk.tmax,
            "Tmax Unit": pk.tmax_unit,
            "AUC": pk.auc,
            "AUC Unit": pk.auc_unit,
            "Half-life": pk.half_life,
            "Half-life Unit": pk.half_life_unit,
            "Clearance": pk.clearance,
            "Clearance Unit": pk.clearance_unit,
            "Bioavailability (%)": pk.bioavailability,
            "Protein Binding (%)": pk.protein_binding
        }
        rows.append(row)

    if not rows:
        return None

    return pd.DataFrame(rows)


def export_evidence_to_excel(
    evidence_list: List[ExtractedEvidence],
    filepath: str
) -> bool:
    """
    Export evidence to an Excel file with multiple sheets

    Args:
        evidence_list: List of ExtractedEvidence objects
        filepath: Output file path (.xlsx)

    Returns:
        True if successful, False otherwise
    """
    if not PANDAS_AVAILABLE:
        return False

    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main evidence table
            main_df = evidence_to_dataframe(evidence_list)
            if main_df is not None and not main_df.empty:
                main_df.to_excel(writer, sheet_name='Evidence Table', index=False)

            # PK parameters table
            pk_df = pk_parameters_to_dataframe(evidence_list)
            if pk_df is not None and not pk_df.empty:
                pk_df.to_excel(writer, sheet_name='PK Parameters', index=False)

            # Efficacy results table
            efficacy_rows = []
            for ev in evidence_list:
                for eff in ev.efficacy_results:
                    efficacy_rows.append({
                        "PMID": ev.pmid,
                        "Study": ev.title[:50] + "...",
                        "Year": ev.year,
                        "Endpoint": eff.endpoint,
                        "Result": eff.result,
                        "p-value": eff.p_value,
                        "95% CI": eff.confidence_interval,
                        "Effect Size": eff.effect_size
                    })
            if efficacy_rows:
                pd.DataFrame(efficacy_rows).to_excel(
                    writer, sheet_name='Efficacy Results', index=False
                )

            # Safety results table
            safety_rows = []
            for ev in evidence_list:
                for saf in ev.safety_results:
                    safety_rows.append({
                        "PMID": ev.pmid,
                        "Study": ev.title[:50] + "...",
                        "Year": ev.year,
                        "Adverse Event": saf.event,
                        "Incidence": saf.incidence,
                        "Severity": saf.severity
                    })
            if safety_rows:
                pd.DataFrame(safety_rows).to_excel(
                    writer, sheet_name='Safety Data', index=False
                )

        return True

    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        return False
