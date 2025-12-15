"""
Context-Aware Expert Enhancement System

Lightweight, data-driven rule engine that detects clinical/regulatory context
from user questions and injects targeted expertise into expert prompts.
Adapted from Virtual SAB for drug development literature review context.
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Set, Pattern, Dict
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from functools import cached_property

logger = logging.getLogger(__name__)


@dataclass
class EnhancementRule:
    """
    A single enhancement rule that fires based on pattern matching.

    Attributes:
        name: Short identifier for the rule (e.g., "LYTAC modality")
        category: Grouping dimension (e.g., "modality", "disease", "regulatory")
        patterns: List of regex patterns to match against question text
        text: Enhancement text to inject into system prompt when rule fires
        requires_all: If True, ALL patterns must match; if False, ANY pattern matches
        weight: Priority for tie-breaking (higher = more important)
    """
    name: str
    category: str
    patterns: List[str]
    text: str
    requires_all: bool = False
    weight: int = 1

    @cached_property
    def _compiled_patterns(self) -> List[Pattern]:
        """Lazy-compile patterns once on first use"""
        return [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def matches(self, question: str) -> bool:
        """Check if this rule should fire for the given question text"""
        if self.requires_all:
            return all(p.search(question) for p in self._compiled_patterns)
        return any(p.search(question) for p in self._compiled_patterns)


# =============================================================================
# SEED RULES: Drug Development Context Rules
# =============================================================================

RULES: List[EnhancementRule] = [

    # -------------------------
    # MODALITY RULES
    # -------------------------

    EnhancementRule(
        name="ADC modality",
        category="modality",
        patterns=[r"\bADC(s)?\b", r"antibody[\s-]?drug conjugate", r"\bT-DXd\b", r"trastuzumab deruxtecan"],
        text=(
            "Modality: ADC. Consider (1) DAR (drug-antibody ratio) optimization (2-8 range typical), "
            "(2) linker stability (cleavable vs non-cleavable), "
            "(3) payload selection (auristatin vs maytansinoid vs topoisomerase inhibitor), "
            "(4) bystander effect for heterogeneous tumors, "
            "(5) ILD/pneumonitis risk (especially with topoisomerase payloads), and "
            "(6) target antigen expression threshold (H-score criteria)."
        ),
        weight=3,
    ),

    EnhancementRule(
        name="Bispecific antibody",
        category="modality",
        patterns=[r"\bbispecific(s)?\b", r"\bTCE(s)?\b", r"T[\s-]?cell engager", r"\bBiTE\b"],
        text=(
            "Modality: Bispecific/TCE. Consider (1) CRS (cytokine release syndrome) mitigation with step-up dosing, "
            "(2) T-cell engagement potency and affinity balance, "
            "(3) tumor microenvironment and T-cell infiltration requirements, "
            "(4) manufacturing complexity, and "
            "(5) tocilizumab availability for CRS management."
        ),
        weight=3,
    ),

    EnhancementRule(
        name="CAR-T therapy",
        category="modality",
        patterns=[r"\bCAR[\s-]?T\b", r"chimeric antigen receptor", r"autologous.*T[\s-]?cell"],
        text=(
            "Modality: CAR-T. Consider (1) manufacturing turnaround time (14-28 days; bridging therapy needed), "
            "(2) CRS and ICANS risk management, "
            "(3) lymphodepletion pre-conditioning, "
            "(4) vein-to-vein time as competitive differentiator, and "
            "(5) cost and reimbursement challenges."
        ),
        weight=3,
    ),

    EnhancementRule(
        name="Small molecule inhibitor",
        category="modality",
        patterns=[r"\binhibitor\b", r"\bTKI\b", r"kinase inhibitor", r"small molecule"],
        text=(
            "Modality: Small molecule. Consider (1) selectivity profile and off-target effects, "
            "(2) resistance mechanisms and combination strategies, "
            "(3) oral bioavailability and food effect, "
            "(4) CYP interactions and DDI potential, and "
            "(5) CNS penetration if brain mets relevant."
        ),
        weight=2,
    ),

    EnhancementRule(
        name="Immunotherapy/IO",
        category="modality",
        patterns=[r"\bimmunotherapy\b", r"\bPD-?1\b", r"\bPD-?L1\b", r"\bCTLA-?4\b", r"checkpoint"],
        text=(
            "Modality: Immunotherapy. Consider (1) PD-L1 expression as biomarker (TPS cutoffs), "
            "(2) TMB and MSI-H status, "
            "(3) immune-related adverse events (irAEs), "
            "(4) combination sequencing with chemo or targeted therapy, and "
            "(5) durable response potential vs response rate."
        ),
        weight=3,
    ),

    # -------------------------
    # DISEASE RULES
    # -------------------------

    EnhancementRule(
        name="NSCLC",
        category="disease",
        patterns=[r"\bNSCLC\b", r"non[\s-]?small cell lung", r"lung adenocarcinoma"],
        text=(
            "Disease: NSCLC. Consider (1) driver mutation stratification (EGFR/ALK/ROS1/KRAS/MET/RET), "
            "(2) prior TKI exposure and resistance mechanisms, "
            "(3) brain metastases (40-50% prevalence; CNS penetration critical), "
            "(4) PD-L1 TPS stratification, and "
            "(5) standard backbones (carboplatin/pemetrexed ± pembrolizumab)."
        ),
        weight=2,
    ),

    EnhancementRule(
        name="Breast cancer",
        category="disease",
        patterns=[r"breast cancer", r"\bHER2\b", r"triple[\s-]?negative", r"\bTNBC\b"],
        text=(
            "Disease: Breast Cancer. Consider (1) HER2 status (IHC 3+, 2+/FISH+, HER2-low), "
            "(2) HR status and CDK4/6i combinations, "
            "(3) HER2-targeted ADCs (T-DXd precedent), "
            "(4) brain mets frequency in HER2+, and "
            "(5) neoadjuvant vs metastatic setting (pCR as surrogate)."
        ),
        weight=2,
    ),

    EnhancementRule(
        name="Colorectal cancer",
        category="disease",
        patterns=[r"\bCRC\b", r"colorectal", r"colon cancer", r"\bMSI[-\s]?H\b"],
        text=(
            "Disease: CRC. Consider (1) MSI-H vs MSS (IO response in MSI-H), "
            "(2) sidedness (right vs left prognosis), "
            "(3) RAS/BRAF mutation status, "
            "(4) liver metastases resectability, and "
            "(5) standard 1L (FOLFOX/FOLFIRI + bevacizumab or cetuximab)."
        ),
        weight=2,
    ),

    EnhancementRule(
        name="Hematologic malignancy",
        category="disease",
        patterns=[r"\bAML\b", r"\bALL\b", r"leukemia", r"lymphoma", r"myeloma", r"\bMDS\b"],
        text=(
            "Disease: Hematologic malignancy. Consider (1) cytogenetic risk stratification, "
            "(2) MRD (minimal residual disease) as endpoint, "
            "(3) transplant eligibility and bridging, "
            "(4) CAR-T and bispecific precedents, and "
            "(5) supportive care burden (transfusions, infections)."
        ),
        weight=2,
    ),

    # -------------------------
    # REGULATORY RULES
    # -------------------------

    EnhancementRule(
        name="FDA regulatory",
        category="regulatory",
        patterns=[r"\bFDA\b", r"accelerated approval", r"breakthrough therapy", r"fast track"],
        text=(
            "Regulatory: FDA. Consider (1) Accelerated Approval pathway (surrogate endpoint, confirmatory trial), "
            "(2) Breakthrough Therapy designation benefits, "
            "(3) recent FDA precedents for similar mechanisms, "
            "(4) RTOR (Real-Time Oncology Review) eligibility, and "
            "(5) post-marketing commitments."
        ),
        weight=2,
    ),

    EnhancementRule(
        name="EMA regulatory",
        category="regulatory",
        patterns=[r"\bEMA\b", r"\bCHMP\b", r"\bPRIME\b", r"European.*approval"],
        text=(
            "Regulatory: EMA. Consider (1) PRIME designation for unmet need, "
            "(2) Conditional Marketing Authorization path, "
            "(3) HTA body coordination (G-BA, NICE, HAS), "
            "(4) parallel scientific advice, and "
            "(5) PIP (pediatric investigation plan) requirements."
        ),
        weight=2,
    ),

    EnhancementRule(
        name="Japan PMDA regulatory",
        category="regulatory",
        patterns=[r"\bPMDA\b", r"\bJapan\b", r"\bsakigake\b"],
        text=(
            "Regulatory: Japan PMDA. Consider (1) ethnic sensitivity analyses (Japanese PK), "
            "(2) bridging study requirements, "
            "(3) sakigake designation criteria, "
            "(4) minimum Japanese enrollment (20%), and "
            "(5) PMDA consultation timing."
        ),
        weight=2,
    ),

    # -------------------------
    # BIOMARKER RULES
    # -------------------------

    EnhancementRule(
        name="EGFR mutations",
        category="biomarker",
        patterns=[r"\bEGFR\b.*mut", r"exon[\s-]?19", r"L858R", r"T790M"],
        text=(
            "Biomarker: EGFR. Consider (1) mutation subtype (ex19del vs L858R vs uncommon), "
            "(2) T790M resistance mechanism, "
            "(3) osimertinib as standard of care, "
            "(4) detection method (tissue vs ctDNA), and "
            "(5) co-mutations (TP53, MET amp)."
        ),
        weight=2,
    ),

    EnhancementRule(
        name="KRAS mutations",
        category="biomarker",
        patterns=[r"\bKRAS\b", r"G12C", r"G12D", r"sotorasib", r"adagrasib"],
        text=(
            "Biomarker: KRAS. Consider (1) G12C prevalence by tumor type, "
            "(2) sotorasib/adagrasib precedents (30-40% ORR in NSCLC), "
            "(3) co-mutations impacting response (STK11, KEAP1), "
            "(4) on-target resistance mechanisms, and "
            "(5) combination strategies (+ SHP2i, + chemo)."
        ),
        weight=2,
    ),

    EnhancementRule(
        name="HER2 biomarker",
        category="biomarker",
        patterns=[r"\bHER2\b", r"ERBB2", r"H-score", r"IHC.*[23]\+"],
        text=(
            "Biomarker: HER2. Consider (1) IHC/FISH testing standards, "
            "(2) HER2-low definition (IHC 1-2+/FISH-), "
            "(3) H-score thresholds for ADC eligibility, "
            "(4) heterogeneity and biopsy representativeness, and "
            "(5) HER2 amplification vs overexpression."
        ),
        weight=2,
    ),

    # -------------------------
    # CLINICAL SETTING RULES
    # -------------------------

    EnhancementRule(
        name="First-line setting",
        category="clinical",
        patterns=[r"\b1L\b", r"first[\s-]?line", r"frontline", r"treatment[\s-]?naive"],
        text=(
            "Clinical: First-line setting. Consider (1) current standard of care as comparator, "
            "(2) broader patient population (better PS, fewer comorbidities), "
            "(3) longer PFS expectations, "
            "(4) combination vs monotherapy positioning, and "
            "(5) potential for maintenance strategy."
        ),
        weight=2,
    ),

    EnhancementRule(
        name="Later-line setting",
        category="clinical",
        patterns=[r"\b[23]L\+?\b", r"second[\s-]?line", r"third[\s-]?line", r"refractory", r"relapsed"],
        text=(
            "Clinical: Later-line setting. Consider (1) limited treatment options (unmet need), "
            "(2) single-arm trial feasibility, "
            "(3) accelerated approval pathway potential, "
            "(4) prior therapy exposure effects, and "
            "(5) lower ORR thresholds acceptable for approval."
        ),
        weight=2,
    ),

    EnhancementRule(
        name="Brain metastases",
        category="clinical",
        patterns=[r"brain met", r"\bCNS\b.*met", r"leptomeningeal", r"blood[\s-]?brain barrier"],
        text=(
            "Clinical: Brain metastases. Consider (1) BBB penetration (unbound fraction in CSF), "
            "(2) prior CNS-directed therapy (WBRT, SRS), "
            "(3) CNS-specific endpoints (CNS-ORR, CNS-PFS), "
            "(4) baseline brain MRI requirements, and "
            "(5) precedents with CNS activity (tucatinib, osimertinib, lorlatinib)."
        ),
        weight=2,
    ),

    # -------------------------
    # TRIAL DESIGN RULES
    # -------------------------

    EnhancementRule(
        name="Phase 1 design",
        category="trial_design",
        patterns=[r"phase[\s-]?1", r"phase[\s-]?I\b", r"dose[\s-]?escalation", r"\bFIH\b", r"first[\s-]?in[\s-]?human"],
        text=(
            "Trial Design: Phase 1. Consider (1) 3+3 vs BOIN vs mTPI-2 dose escalation, "
            "(2) starting dose justification (MABEL, NOAEL, PAD), "
            "(3) DLT definition and evaluation window, "
            "(4) expansion cohorts by indication/biomarker, and "
            "(5) PK/PD endpoints for RP2D selection."
        ),
        weight=3,
    ),

    EnhancementRule(
        name="Randomized trial",
        category="trial_design",
        patterns=[r"randomized", r"\bRCT\b", r"phase[\s-]?3", r"phase[\s-]?III", r"pivotal"],
        text=(
            "Trial Design: Randomized/Pivotal. Consider (1) comparator selection (active vs placebo), "
            "(2) primary endpoint (PFS vs OS), "
            "(3) sample size and event-driven design, "
            "(4) stratification factors, "
            "(5) interim analysis strategy, and "
            "(6) crossover implications for OS."
        ),
        weight=3,
    ),

    EnhancementRule(
        name="Single-arm trial",
        category="trial_design",
        patterns=[r"single[\s-]?arm", r"non[\s-]?randomized", r"uncontrolled"],
        text=(
            "Trial Design: Single-arm. Consider (1) ORR as primary endpoint, "
            "(2) historical control benchmarking, "
            "(3) durability of response (DOR), "
            "(4) accelerated approval pathway eligibility, and "
            "(5) confirmatory trial requirements."
        ),
        weight=2,
    ),
]


# =============================================================================
# ENHANCEMENT ENGINE
# =============================================================================

def enhance_expert_for_question(
    base_persona: str,
    question: str,
    max_rules_per_category: int = 2,
    max_total_rules: int = 4
) -> str:
    """
    Enhance expert persona prompt with context-aware expertise.

    Args:
        base_persona: Base system prompt from expert persona
        question: User's product/scenario/question text (for context detection)
        max_rules_per_category: Maximum rules per category (prevents walls of text)
        max_total_rules: Maximum total rules that can fire

    Returns:
        Enhanced system prompt with targeted expertise injected
    """
    # Find all matching rules
    matching_rules = [rule for rule in RULES if rule.matches(question)]

    if not matching_rules:
        return base_persona

    # Sort by weight (descending) then category then name
    matching_rules.sort(key=lambda r: (-r.weight, r.category, r.name))

    # Apply per-category cap AND total cap
    selected_rules = []
    category_counts = defaultdict(int)

    for rule in matching_rules:
        if len(selected_rules) >= max_total_rules:
            break

        if category_counts[rule.category] < max_rules_per_category:
            selected_rules.append(rule)
            category_counts[rule.category] += 1

    if not selected_rules:
        return base_persona

    # Build context tags summary
    context_tags = ", ".join(sorted({r.name for r in selected_rules}))

    # Build enhancement text blocks
    enhancement_bullets = "\n\n".join(f"• {rule.text}" for rule in selected_rules)

    # Assemble enhanced prompt
    enhanced_prompt = f"""{base_persona}

CONTEXT-SPECIFIC EXPERTISE:
Context detected: {context_tags}

Use these targeted considerations in your analysis:

{enhancement_bullets}

OUTPUT GUIDANCE:
- Reference specific numbers, studies, or precedents where possible
- Cite PMIDs from the evidence corpus when making claims
- Be explicit about data gaps and uncertainties
- Provide actionable recommendations with clear rationale"""

    # Log for review
    _log_enhancement(question, selected_rules)

    return enhanced_prompt


def _log_enhancement(question: str, rules_fired: List[EnhancementRule]) -> None:
    """Log enhancement activity for review"""
    try:
        log_dir = Path(__file__).parent.parent.parent / "outputs" / "enhancement_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{today}.csv"

        if not log_file.exists():
            with open(log_file, 'w') as f:
                f.write("timestamp,question_snippet,rules_fired,categories\n")

        timestamp = datetime.now().isoformat()
        question_snippet = question[:100].replace("\n", " ").replace(",", ";")
        rules_list = "|".join(r.name for r in rules_fired)
        categories = "|".join(sorted({r.category for r in rules_fired}))

        with open(log_file, 'a') as f:
            f.write(f"{timestamp},\"{question_snippet}\",\"{rules_list}\",\"{categories}\"\n")

        logger.debug(f"Enhancement: {len(rules_fired)} rules fired ({categories})")

    except Exception as e:
        logger.warning(f"Failed to log enhancement: {e}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_rule_by_name(name: str) -> Optional[EnhancementRule]:
    """Retrieve a specific rule by name"""
    for rule in RULES:
        if rule.name == name:
            return rule
    return None


def get_rules_by_category(category: str) -> List[EnhancementRule]:
    """Get all rules in a specific category"""
    return [rule for rule in RULES if rule.category == category]


def get_all_categories() -> Set[str]:
    """Get set of all rule categories"""
    return {rule.category for rule in RULES}


def detect_context(question: str) -> Dict[str, List[str]]:
    """
    Detect what context is present in a question.

    Returns dict with category -> list of detected rule names.
    """
    detected = defaultdict(list)
    for rule in RULES:
        if rule.matches(question):
            detected[rule.category].append(rule.name)
    return dict(detected)


def debug_enhancement(question: str, verbose: bool = True) -> dict:
    """
    Test enhancement on a sample question (useful for debugging).

    Returns dict with matching_rules, selected_rules, categories, enhancement_length.
    """
    matching_rules = [rule for rule in RULES if rule.matches(question)]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Question: {question[:100]}...")
        print(f"{'='*60}")
        print(f"Matching rules: {len(matching_rules)}")
        for rule in matching_rules:
            print(f"  ✓ {rule.name} (category={rule.category}, weight={rule.weight})")

    # Apply caps
    matching_rules.sort(key=lambda r: (-r.weight, r.category, r.name))
    selected_rules = []
    category_counts = defaultdict(int)

    for rule in matching_rules:
        if len(selected_rules) >= 4:
            break
        if category_counts[rule.category] < 2:
            selected_rules.append(rule)
            category_counts[rule.category] += 1

    if verbose:
        print(f"\nSelected rules (after caps): {len(selected_rules)}")
        for rule in selected_rules:
            print(f"  → {rule.name}")

    enhancement_text = "\n\n".join(f"• {rule.text}" for rule in selected_rules)

    return {
        "matching_rules": [r.name for r in matching_rules],
        "selected_rules": [r.name for r in selected_rules],
        "categories": list({r.category for r in selected_rules}),
        "enhancement_length": len(enhancement_text)
    }
