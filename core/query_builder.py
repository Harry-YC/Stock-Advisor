"""
Component-Based PubMed Query Builder

Build PubMed queries from extracted concepts using component-based approach.

Key principles:
1. COMPONENT-BASED: Search each concept separately, not all ANDed together
2. SYNONYM EXPANSION: Use all known synonyms for each term
3. CLAIM-AWARE: Different templates for biomarker vs trial claims
4. TIERED: Multiple strictness levels for fallback
"""

import logging
from typing import List, Dict
from dataclasses import dataclass, field

from core.query_extractor import (
    ExtractedConcepts, get_synonyms,
    TARGET_SYNONYMS, INDICATION_SYNONYMS, MECHANISM_SYNONYMS, MODALITY_SYNONYMS
)

logger = logging.getLogger(__name__)


@dataclass
class QuerySet:
    """A set of queries at different strictness tiers."""
    tier1_strict: List[str] = field(default_factory=list)    # Specific, anchored queries
    tier2_relaxed: List[str] = field(default_factory=list)   # No clinical trial filters
    tier3_broad: List[str] = field(default_factory=list)     # Very broad, single concept
    metadata: Dict = field(default_factory=dict)


class ComponentQueryBuilder:
    """
    Build component-based PubMed queries.

    For a question about "FOLR1-CLEC5a-IFNa engager in 2L+ NSCLC", generates:
    - Query 1: FOLR1 + NSCLC
    - Query 2: CLEC5a + macrophage
    - Query 3: IFN-alpha + lung cancer
    - Query 4: macrophage engager + NSCLC

    NOT: FOLR1 AND CLEC5a AND IFNa AND NSCLC (would return 0 results)
    """

    def build_search_queries(self, concepts: ExtractedConcepts) -> QuerySet:
        """Build tiered query set from extracted concepts."""

        tier1 = []  # Strict: concept + indication + clinical filter
        tier2 = []  # Relaxed: concept + indication, no filters
        tier3 = []  # Broad: just the concept + cancer

        indication_clause = self._build_indication_clause(concepts.indications)

        # ====================================================================
        # TARGET QUERIES - Most important
        # ====================================================================
        for target in concepts.targets[:4]:
            target_clause = self._build_term_clause(target, "target")

            if indication_clause:
                # Tier 1: Target + Indication + Clinical relevance
                tier1.append(
                    f"(({target_clause}) AND ({indication_clause}) AND "
                    f"(clinical trial[pt] OR phase[tiab] OR patient[tiab]))"
                )

                # Tier 2: Target + Indication only
                tier2.append(f"(({target_clause}) AND ({indication_clause}))")

            # Tier 3: Target + generic cancer terms
            tier3.append(f"(({target_clause}) AND (cancer[tiab] OR tumor[tiab] OR carcinoma[tiab]))")

        # ====================================================================
        # MECHANISM QUERIES
        # ====================================================================
        for mech in concepts.mechanisms[:2]:
            mech_clause = self._build_term_clause(mech, "mechanism")

            if indication_clause:
                tier1.append(f"(({mech_clause}) AND ({indication_clause}))")
                tier2.append(f"(({mech_clause}) AND ({indication_clause}))")

            tier3.append(f"(({mech_clause}) AND (cancer[tiab] OR tumor[tiab] OR immunotherapy[tiab]))")

        # ====================================================================
        # MODALITY/PAYLOAD QUERIES (e.g., IFN-alpha)
        # ====================================================================
        for mod in concepts.modalities[:2]:
            mod_clause = self._build_term_clause(mod, "modality")

            if indication_clause:
                tier1.append(f"(({mod_clause}) AND ({indication_clause}))")
                tier2.append(f"(({mod_clause}) AND ({indication_clause}))")

            # For cytokines, also search with immunotherapy/cancer
            tier3.append(f"(({mod_clause}) AND (cancer[tiab] OR immunotherapy[tiab] OR solid tumor[tiab]))")

        # ====================================================================
        # COMPETITIVE LANDSCAPE QUERY - For clinical trial info
        # ====================================================================
        if indication_clause:
            tier1.append(
                f"(({indication_clause}) AND "
                f"(phase 2[tiab] OR phase 3[tiab] OR phase II[tiab] OR phase III[tiab]))"
            )

        # ====================================================================
        # CROSS-CONCEPT QUERIES (2 concepts together, not all)
        # ====================================================================
        if len(concepts.targets) >= 2:
            t1 = self._build_term_clause(concepts.targets[0], "target")
            t2 = self._build_term_clause(concepts.targets[1], "target")
            tier2.append(f"(({t1}) AND ({t2}))")

        if concepts.targets and concepts.mechanisms:
            t = self._build_term_clause(concepts.targets[0], "target")
            m = self._build_term_clause(concepts.mechanisms[0], "mechanism")
            tier2.append(f"(({t}) AND ({m}))")

        return QuerySet(
            tier1_strict=tier1,
            tier2_relaxed=tier2,
            tier3_broad=tier3,
            metadata={
                "concepts": concepts.to_dict(),
                "indication_clause": indication_clause
            }
        )

    def build_claim_validation_queries(
        self,
        claim_text: str,
        concepts: ExtractedConcepts,
        claim_type: str = "auto"
    ) -> List[str]:
        """
        Build queries to validate a specific claim.

        Claim types:
        - "biomarker": Expression, prevalence, IHC studies (no trial filters)
        - "efficacy": ORR, PFS, OS, trial results (with trial filters)
        - "safety": Toxicity, adverse events
        - "auto": Detect from claim text
        """
        if claim_type == "auto":
            claim_type = self._detect_claim_type(claim_text)

        queries = []
        indication_clause = self._build_indication_clause(concepts.indications)

        if claim_type == "biomarker":
            # Biomarker/expression claims - NO clinical trial filters
            for target in concepts.targets[:2]:
                target_clause = self._build_term_clause(target, "target")

                if indication_clause:
                    queries.append(
                        f"(({target_clause}) AND ({indication_clause}) AND "
                        f"(expression[tiab] OR prevalence[tiab] OR immunohistochemistry[tiab] OR IHC[tiab]))"
                    )

                # Also try without indication for broader coverage
                queries.append(
                    f"(({target_clause}) AND "
                    f"(expression[tiab] OR prevalence[tiab] OR tumor[tiab]))"
                )

        elif claim_type == "efficacy":
            # Efficacy claims - need trial data
            for target in concepts.targets[:2]:
                target_clause = self._build_term_clause(target, "target")

                if indication_clause:
                    queries.append(
                        f"(({target_clause}) AND ({indication_clause}) AND "
                        f"(ORR[tiab] OR response rate[tiab] OR survival[tiab] OR "
                        f"phase 2[tiab] OR phase 3[tiab]))"
                    )

        elif claim_type == "safety":
            # Safety claims
            for target in concepts.targets[:2]:
                target_clause = self._build_term_clause(target, "target")

                queries.append(
                    f"(({target_clause}) AND "
                    f"(toxicity[tiab] OR adverse[tiab] OR safety[tiab] OR "
                    f"cytokine release[tiab] OR dose limiting[tiab]))"
                )

            # For modalities like IFN-alpha, search toxicity specifically
            for mod in concepts.modalities[:1]:
                mod_clause = self._build_term_clause(mod, "modality")
                queries.append(
                    f"(({mod_clause}) AND (toxicity[tiab] OR adverse[tiab] OR safety[tiab]))"
                )

        else:
            # Generic - use standard component queries
            query_set = self.build_search_queries(concepts)
            queries = query_set.tier2_relaxed[:4]

        return queries

    def _build_term_clause(self, term: str, term_type: str) -> str:
        """Build a clause with all synonyms OR'd together."""
        synonyms = get_synonyms(term, term_type)

        # Deduplicate and clean
        unique = []
        seen = set()
        for s in synonyms:
            s_lower = s.lower()
            if s_lower not in seen:
                seen.add(s_lower)
                unique.append(s)

        return " OR ".join([f'"{s}"[tiab]' for s in unique])

    def _build_indication_clause(self, indications: List[str]) -> str:
        """Build indication clause with all synonyms."""
        if not indications:
            return ""

        all_terms = []
        for ind in indications[:2]:
            synonyms = get_synonyms(ind, "indication")
            all_terms.extend(synonyms)

        # Deduplicate
        unique = list(dict.fromkeys(all_terms))

        return " OR ".join([f'"{t}"[tiab]' for t in unique])

    def _detect_claim_type(self, claim_text: str) -> str:
        """Detect claim type from text."""
        claim_lower = claim_text.lower()

        biomarker_signals = ["expression", "prevalence", "ihc", "immunohistochemistry",
                            "staining", "positive", "frequency", "density"]
        efficacy_signals = ["orr", "response rate", "pfs", "os", "survival",
                           "efficacy", "phase 2", "phase 3", "trial"]
        safety_signals = ["toxicity", "adverse", "safety", "crs", "cytokine release",
                         "neurotoxicity", "dose limiting", "mtd"]

        if any(s in claim_lower for s in biomarker_signals):
            return "biomarker"
        if any(s in claim_lower for s in efficacy_signals):
            return "efficacy"
        if any(s in claim_lower for s in safety_signals):
            return "safety"

        return "generic"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_tiered_queries(concepts: ExtractedConcepts) -> QuerySet:
    """
    Build tiered PubMed queries from extracted concepts.

    Args:
        concepts: ExtractedConcepts from ClinicalQueryExtractor

    Returns:
        QuerySet with tier1_strict, tier2_relaxed, tier3_broad queries
    """
    builder = ComponentQueryBuilder()
    return builder.build_search_queries(concepts)


def build_validation_queries(
    claim_text: str,
    concepts: ExtractedConcepts,
    claim_type: str = "auto"
) -> List[str]:
    """
    Build queries to validate a specific claim.

    Args:
        claim_text: The claim text to validate
        concepts: ExtractedConcepts for context
        claim_type: "biomarker", "efficacy", "safety", or "auto"

    Returns:
        List of PubMed query strings
    """
    builder = ComponentQueryBuilder()
    return builder.build_claim_validation_queries(claim_text, concepts, claim_type)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from core.query_extractor import ExtractedConcepts

    print("=" * 60)
    print("TESTING COMPONENT QUERY BUILDER")
    print("=" * 60)

    # Create test concepts
    concepts = ExtractedConcepts(
        targets=["FOLR1", "CLEC5a"],
        indications=["NSCLC"],
        mechanisms=["macrophage engager"],
        modalities=["IFNa"]
    )

    builder = ComponentQueryBuilder()
    query_set = builder.build_search_queries(concepts)

    print("\nTier 1 (Strict) Queries:")
    for i, q in enumerate(query_set.tier1_strict, 1):
        print(f"  {i}. {q[:100]}...")

    print("\nTier 2 (Relaxed) Queries:")
    for i, q in enumerate(query_set.tier2_relaxed, 1):
        print(f"  {i}. {q[:100]}...")

    print("\nTier 3 (Broad) Queries:")
    for i, q in enumerate(query_set.tier3_broad, 1):
        print(f"  {i}. {q[:100]}...")

    print("\n" + "=" * 60)
    print("Testing claim validation queries...")
    print("=" * 60)

    claim = "FOLR1 expression is found in 40% of NSCLC patients"
    val_queries = builder.build_claim_validation_queries(claim, concepts)

    print(f"\nClaim: {claim}")
    print("Validation queries:")
    for i, q in enumerate(val_queries, 1):
        print(f"  {i}. {q[:100]}...")
