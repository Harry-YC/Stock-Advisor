"""
Oncology Image Analyzer

Specialized image analysis for drug development figures:
- Efficacy plots (KM curves, waterfall, spider plots)
- Preclinical data (tumor growth, viability, Western blots)
- Genomics/Biomarkers (heatmaps, volcano plots, IHC)
"""

from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
import base64
import logging
import google.generativeai as genai
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class ImageAnalysis:
    """Structured result from image analysis."""
    image_type: str           # e.g., "kaplan_meier", "tumor_growth", "heatmap"
    summary: str              # One-line summary
    key_findings: List[str]   # Bullet points
    clinical_implications: str = ""
    limitations: str = ""
    raw_description: str = ""
    extracted_data: Dict = field(default_factory=dict)

ONCOLOGY_PROMPTS = {
    "kaplan_meier": """Analyze this Kaplan-Meier survival curve:
1. Identify treatment arms and their colors/labels
2. Extract median survival times (if shown)
3. Note hazard ratio and confidence intervals
4. Identify separation of curves and timing
5. Note censoring events
6. Assess statistical significance (p-value if shown)
Output: Clinical interpretation for drug development decision-making.""",

    "tumor_growth": """Analyze this tumor growth/volume curve:
1. Identify treatment groups
2. Note starting tumor volumes
3. Calculate tumor growth inhibition (TGI) if possible
4. Identify complete/partial responses
5. Note any tumor regrowth patterns
Output: Efficacy assessment for preclinical-to-clinical translation.""",

    "waterfall": """Analyze this waterfall plot:
1. Count responders (CR, PR) vs non-responders
2. Calculate ORR (% with >30% reduction)
3. Identify any complete responses (100% reduction)
4. Note outliers (exceptional responders/progressors)
Output: Response rate summary for clinical development.""",

    "heatmap": """Analyze this gene expression/biomarker heatmap:
1. Identify sample groups (responders vs non-responders, treated vs control)
2. Note clustering patterns
3. Identify upregulated/downregulated genes or markers
4. Look for predictive biomarker signatures
Output: Biomarker insights for patient stratification.""",

    "flow_cytometry": """Analyze this flow cytometry plot:
1. Identify cell populations and gating
2. Note percentages of key populations
3. Compare between conditions if multiple panels
4. Identify immune cell infiltration or depletion
Output: Immunological assessment for I-O drug development.""",

    "western_blot": """Analyze this Western blot:
1. Identify protein targets and molecular weights
2. Compare band intensities across conditions
3. Note loading controls
4. Assess target engagement or pathway modulation
Output: Mechanism of action evidence."""
}

def detect_image_type(image_data: bytes) -> str:
    """Use Gemini to auto-detect oncology figure type."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """Classify this scientific figure into ONE category:
        - kaplan_meier (survival curves with step functions)
        - tumor_growth (line plots showing tumor volume over time)
        - waterfall (vertical bars showing % change)
        - spider (line plots showing individual patient responses)
        - heatmap (colored matrix of values)
        - flow_cytometry (scatter/density plots with quadrants)
        - western_blot (horizontal bands on gel)
        - bar_chart (grouped bars with error bars)
        - other (none of the above)

        Respond with only the category name."""
        
        image_part = {"mime_type": "image/jpeg", "data": image_data}
        response = model.generate_content([prompt, image_part])
        
        return response.text.strip().lower()
    except Exception as e:
        logger.error(f"Image type detection failed: {e}")
        return "other"

def analyze_image(
    image_data: bytes,
    filename: str,
    question_context: Optional[str] = None
) -> ImageAnalysis:
    """Analyze an oncology-related image using Gemini Vision."""
    
    # Auto-detect type
    image_type = detect_image_type(image_data)
    
    # Select prompt
    base_prompt = ONCOLOGY_PROMPTS.get(image_type, "Analyze this scientific figure and extract key findings relevant to drug development.")
    
    full_prompt = f"""
    You are an expert oncology data analyst.
    
    {base_prompt}
    
    Context: {question_context if question_context else "No specific context provided."}
    
    Provide a structured analysis in JSON format with the following keys:
    - summary: One-sentence high-level summary
    - key_findings: List of 3-5 specific observations (bullet points)
    - clinical_implications: How this data impacts the drug development program
    - limitations: Any data quality issues or missing info (e.g. no error bars, unclear labels)
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        image_part = {"mime_type": "image/jpeg", "data": image_data}
        response = model.generate_content(
            [full_prompt, image_part],
            generation_config={"response_mime_type": "application/json"}
        )
        
        import json
        result = json.loads(response.text)
        
        return ImageAnalysis(
            image_type=image_type,
            summary=result.get("summary", "Analysis unavailable"),
            key_findings=result.get("key_findings", []),
            clinical_implications=result.get("clinical_implications", ""),
            limitations=result.get("limitations", ""),
            raw_description=response.text, # Backup
            extracted_data={}
        )
        
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return ImageAnalysis(
            image_type=image_type,
            summary=f"Analysis failed: {str(e)}",
            key_findings=[],
            clinical_implications="",
            limitations="",
            raw_description=""
        )

def format_for_expert_context(analyses: List[ImageAnalysis]) -> str:
    """Format image analyses for inclusion in expert prompts."""
    if not analyses:
        return ""
        
    context_parts = ["\n[IMAGE ANALYSIS CONTEXT]"]
    
    for i, analysis in enumerate(analyses, 1):
        context_parts.append(f"Image {i} ({analysis.image_type}):")
        context_parts.append(f"Summary: {analysis.summary}")
        context_parts.append("Findings:")
        for finding in analysis.key_findings:
            context_parts.append(f"- {finding}")
        if analysis.clinical_implications:
            context_parts.append(f"Implications: {analysis.clinical_implications}")
        context_parts.append("")
        
    return "\n".join(context_parts)
