
"""
Clinical Planning Service

This module generates structured clinical development documents 
(IND Module 2.4, Clinical Development Plans) from research context.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum

from config import settings

logger = logging.getLogger("literature_review.planning_service")

class PlanType(Enum):
    IND_MODULE_2_4 = "IND Module 2.4 (Nonclinical Overview)"
    CLINICAL_DEV_PLAN = "Clinical Development Plan (CDP)"
    TARGET_PRODUCT_PROFILE = "Target Product Profile (TPP)"

@dataclass
class ClinicalDocument:
    """A generated clinical document."""
    title: str
    doc_type: str
    sections: Dict[str, str] # Section Header -> Content
    key_risks: List[str]
    missing_data: List[str] # Gaps identified

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ClinicalPlanner:
    """
    Synthesizes expert discussions and literature into regulatory documents.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key
        self.model = model or settings.REASONING_MODEL

    def generate_plan(
        self,
        topic: str,
        context_text: str,
        plan_type: PlanType
    ) -> ClinicalDocument:
        """
        Generate a structured clinical document.

        Args:
            topic: The drug/asset/indication being planned.
            context_text: Aggregated content (papers, expert discussions).
            plan_type: detailed enum of document type.

        Returns:
            ClinicalDocument object.
        """
        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=self.api_key, model=self.model)

        # distinct prompts for each doc type
        if plan_type == PlanType.IND_MODULE_2_4:
            template_instruction = """
            Structure the output EXACTLY as Common Technical Document (CTD) Module 2.4:
            1. Nonclinical Strategies (Overview)
            2. Pharmacology (Primary, Secondary, Safety)
            3. Pharmacokinetics (ADME)
            4. Toxicology (Single-dose, Repeat-dose, Geno/Carcinogenicity)
            5. Integrated Overview & Conclusions
            """
        elif plan_type == PlanType.CLINICAL_DEV_PLAN:
            template_instruction = """
            Structure the output as a Clinical Development Plan (CDP):
            1. Executive Summary
            2. Phase 1 Strategy (SAD/MAD/FE, Biomarkers)
            3. Phase 2 Proof-of-Concept (Design, Endpoints, Populations)
            4. Phase 3 Pivotal Trials (Registration Strategy)
            5. Regulatory Considerations
            """
        elif plan_type == PlanType.TARGET_PRODUCT_PROFILE:
            template_instruction = """
            Structure the output as a Target Product Profile (TPP):
            1. Indication & Usage
            2. Dosage & Administration
            3. Efficacy Claims (Primary/Secondary Endpoints)
            4. Safety & Contraindications
            5. Competitive Advantage / Differentiation
            """
        else:
            template_instruction = "Provide a structured professional summary."

        system_prompt = f"""You are an Expert Regulatory Writer and Clinical Strategist.
Your task is to draft a high-quality, professional document based ONLY on the provided context.

{template_instruction}

Rules:
1. Be formal, objective, and precise.
2. Cite specific constraints or findings from the context.
3. Explicitly list "Missing Data" where the context is insufficient.
4. Output valid JSON."""

        user_prompt = f"""Topic/Asset: {topic}

Context/Evidence Base:
{context_text[:30000]}

Task: Draft the {plan_type.value}.
Return JSON object with keys:
- title (string)
- doc_type (string)
- sections (dictionary: "Section Name" -> "Detailed Content")
- key_risks (list of strings)
- missing_data (list of strings - what crucial info is absent?)
"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.4 # Low temp for formal documentation
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            return ClinicalDocument(
                title=data.get("title", f"Draft {plan_type.value}"),
                doc_type=plan_type.value,
                sections=data.get("sections", {}),
                key_risks=data.get("key_risks", []),
                missing_data=data.get("missing_data", [])
            )

        except Exception as e:
            logger.error(f"Failed to generate plan: {e}")
            return ClinicalDocument(
                title="Error",
                doc_type="Error",
                sections={"Error": str(e)},
                key_risks=[],
                missing_data=[]
            )
