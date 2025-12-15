"""
Evidence Corpus - Single Source of Truth for All Evidence

This module provides the canonical data structures for managing evidence
in guideline development. All citations must flow through the EvidenceCorpus
to ensure GDG experts can ONLY cite from included papers.

Architecture:
    EvidenceCorpus (single source of truth)
    ├── included_pmids: Set[str]     # Papers that passed screening
    ├── excluded_pmids: Dict[str, str]  # Papers excluded with reasons
    ├── extractions: Dict[str, ExtractedEvidence]  # Structured data
    └── quality_ratings: Dict[str, QualityRating]  # Risk of bias scores
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Any

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEvidence:
    """
    Structured evidence extracted from a paper.

    Contains PICO elements and key findings in a standardized format
    for evidence tables and synthesis.
    """
    pmid: str
    title: str = ""
    study_design: str = ""  # RCT, Cohort, Case Series, Case Report, etc.
    evidence_level: str = ""  # I, II, III, IV, V (Oxford CEBM)
    population: str = ""
    sample_size: Optional[int] = None
    intervention: str = ""
    comparator: str = ""
    outcomes: List[str] = field(default_factory=list)
    key_findings: str = ""
    effect_size: Optional[str] = None
    confidence_interval: Optional[str] = None
    p_value: Optional[str] = None
    follow_up_duration: Optional[str] = None
    limitations: List[str] = field(default_factory=list)
    extracted_at: str = ""
    extracted_by: str = "ai"  # "ai" or "human"
    human_verified: bool = False
    verification_notes: str = ""

    def __post_init__(self):
        if not self.extracted_at:
            self.extracted_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractedEvidence':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_table_row(self) -> Dict[str, str]:
        """Format for evidence table display."""
        return {
            "PMID": self.pmid,
            "Title": self.title[:50] + "..." if len(self.title) > 50 else self.title,
            "Design": self.study_design,
            "N": str(self.sample_size) if self.sample_size else "NR",
            "Population": self.population[:40] + "..." if len(self.population) > 40 else self.population,
            "Intervention": self.intervention[:30] + "..." if len(self.intervention) > 30 else self.intervention,
            "Comparator": self.comparator[:30] + "..." if len(self.comparator) > 30 else self.comparator,
            "Outcomes": ", ".join(self.outcomes[:3]),
            "Effect": self.effect_size or "NR",
            "Verified": "Yes" if self.human_verified else "No"
        }


@dataclass
class ScreeningDecision:
    """Record of a screening decision with reasoning."""
    pmid: str
    decision: str  # "include", "exclude", "review"
    reason: str
    confidence: int = 0  # 0-100
    screened_by: str = "ai"  # "ai" or "human"
    screened_at: str = ""
    scenario: Optional[str] = None  # screening scenario used

    def __post_init__(self):
        if not self.screened_at:
            self.screened_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScreeningDecision':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EvidenceCorpus:
    """
    Single source of truth for all evidence in a project.

    This class enforces the rule that GDG experts can ONLY cite from
    papers that have been explicitly included in the corpus.

    Usage:
        corpus = EvidenceCorpus()
        corpus.include("12345678", "Relevant RCT on MBO outcomes")
        corpus.exclude("87654321", "Wrong population - curative intent")

        # Later, when validating expert responses:
        if corpus.can_cite("12345678"):
            # Valid citation
        else:
            # Invalid - citation not in included set
    """
    project_id: Optional[str] = None

    # Core evidence sets
    included_pmids: Set[str] = field(default_factory=set)
    excluded_pmids: Dict[str, str] = field(default_factory=dict)  # pmid -> reason
    pending_pmids: Set[str] = field(default_factory=set)  # Not yet screened

    # Structured data
    extractions: Dict[str, ExtractedEvidence] = field(default_factory=dict)
    screening_decisions: Dict[str, ScreeningDecision] = field(default_factory=dict)

    # Quality ratings (will be populated by quality_assessment module)
    quality_ratings: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: str = ""
    last_modified: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.last_modified = datetime.now().isoformat()

    # =========================================================================
    # CORE CITATION MANAGEMENT
    # =========================================================================

    def can_cite(self, pmid: str) -> bool:
        """
        Check if a PMID can be cited by GDG experts.

        This is the key enforcement point - only included papers are citable.
        """
        return pmid in self.included_pmids

    def include(self, pmid: str, reason: str = "", screened_by: str = "ai", confidence: int = 0) -> bool:
        """
        Include a paper in the evidence corpus.

        Args:
            pmid: PubMed ID
            reason: Why the paper was included
            screened_by: "ai" or "human"
            confidence: AI confidence score (0-100)

        Returns:
            True if newly included, False if already included
        """
        if pmid in self.included_pmids:
            return False

        # Remove from other sets if present
        self.excluded_pmids.pop(pmid, None)
        self.pending_pmids.discard(pmid)

        # Add to included set
        self.included_pmids.add(pmid)

        # Record decision
        self.screening_decisions[pmid] = ScreeningDecision(
            pmid=pmid,
            decision="include",
            reason=reason,
            confidence=confidence,
            screened_by=screened_by
        )

        self._touch()
        logger.info(f"Included PMID {pmid}: {reason}")
        return True

    def exclude(self, pmid: str, reason: str, screened_by: str = "ai", confidence: int = 0) -> bool:
        """
        Exclude a paper from the evidence corpus.

        Args:
            pmid: PubMed ID
            reason: Why the paper was excluded (required)
            screened_by: "ai" or "human"
            confidence: AI confidence score (0-100)

        Returns:
            True if newly excluded, False if already excluded
        """
        if not reason:
            raise ValueError("Exclusion reason is required")

        if pmid in self.excluded_pmids:
            return False

        # Remove from other sets if present
        self.included_pmids.discard(pmid)
        self.pending_pmids.discard(pmid)

        # Add to excluded set
        self.excluded_pmids[pmid] = reason

        # Record decision
        self.screening_decisions[pmid] = ScreeningDecision(
            pmid=pmid,
            decision="exclude",
            reason=reason,
            confidence=confidence,
            screened_by=screened_by
        )

        self._touch()
        logger.info(f"Excluded PMID {pmid}: {reason}")
        return True

    def mark_pending(self, pmid: str) -> None:
        """Mark a paper as pending screening."""
        if pmid not in self.included_pmids and pmid not in self.excluded_pmids:
            self.pending_pmids.add(pmid)
            self._touch()

    def get_status(self, pmid: str) -> str:
        """
        Get the screening status of a paper.

        Returns: "included", "excluded", "pending", or "unknown"
        """
        if pmid in self.included_pmids:
            return "included"
        elif pmid in self.excluded_pmids:
            return "excluded"
        elif pmid in self.pending_pmids:
            return "pending"
        return "unknown"

    def get_exclusion_reason(self, pmid: str) -> Optional[str]:
        """Get the reason a paper was excluded, if applicable."""
        return self.excluded_pmids.get(pmid)

    def get_screening_decision(self, pmid: str) -> Optional[ScreeningDecision]:
        """Get the full screening decision record for a paper."""
        return self.screening_decisions.get(pmid)

    # =========================================================================
    # EXTRACTION MANAGEMENT
    # =========================================================================

    def add_extraction(self, extraction: ExtractedEvidence) -> None:
        """Add or update extracted evidence for a paper."""
        self.extractions[extraction.pmid] = extraction
        self._touch()

    def get_extraction(self, pmid: str) -> Optional[ExtractedEvidence]:
        """Get extracted evidence for a paper."""
        return self.extractions.get(pmid)

    def has_extraction(self, pmid: str) -> bool:
        """Check if extraction exists for a paper."""
        return pmid in self.extractions

    def get_citable_evidence(self) -> List[ExtractedEvidence]:
        """
        Get all extractions for included papers.

        This is the primary method for getting evidence to show GDG experts.
        """
        return [
            self.extractions[pmid]
            for pmid in self.included_pmids
            if pmid in self.extractions
        ]

    def get_unextracted_pmids(self) -> List[str]:
        """Get PMIDs that are included but not yet extracted."""
        return [
            pmid for pmid in self.included_pmids
            if pmid not in self.extractions
        ]

    # =========================================================================
    # QUALITY RATING MANAGEMENT
    # =========================================================================

    def add_quality_rating(self, pmid: str, rating: Any) -> None:
        """Add quality/risk-of-bias rating for a paper."""
        self.quality_ratings[pmid] = rating
        self._touch()

    def get_quality_rating(self, pmid: str) -> Optional[Any]:
        """Get quality rating for a paper."""
        return self.quality_ratings.get(pmid)

    # =========================================================================
    # BULK OPERATIONS
    # =========================================================================

    def include_batch(self, decisions: List[Dict]) -> int:
        """
        Include multiple papers at once.

        Args:
            decisions: List of dicts with 'pmid', 'reason', 'confidence'

        Returns:
            Number of papers newly included
        """
        count = 0
        for d in decisions:
            if self.include(
                pmid=d['pmid'],
                reason=d.get('reason', ''),
                confidence=d.get('confidence', 0),
                screened_by=d.get('screened_by', 'ai')
            ):
                count += 1
        return count

    def exclude_batch(self, decisions: List[Dict]) -> int:
        """
        Exclude multiple papers at once.

        Args:
            decisions: List of dicts with 'pmid', 'reason', 'confidence'

        Returns:
            Number of papers newly excluded
        """
        count = 0
        for d in decisions:
            if d.get('reason'):
                if self.exclude(
                    pmid=d['pmid'],
                    reason=d['reason'],
                    confidence=d.get('confidence', 0),
                    screened_by=d.get('screened_by', 'ai')
                ):
                    count += 1
        return count

    def apply_screening_results(self, results: List[Dict]) -> Dict[str, int]:
        """
        Apply AI screening results to corpus.

        Args:
            results: List of dicts with 'pmid', 'ai_decision', 'ai_confidence', 'ai_reasoning'

        Returns:
            Dict with counts: {"included": N, "excluded": N, "review": N}
        """
        counts = {"included": 0, "excluded": 0, "review": 0}

        for r in results:
            pmid = r.get('pmid', '')
            decision = r.get('ai_decision', 'review')
            confidence = r.get('ai_confidence', 0)
            reason = r.get('ai_reasoning', '')

            if decision == "include":
                if self.include(pmid, reason, screened_by="ai", confidence=confidence):
                    counts["included"] += 1
            elif decision == "exclude":
                if self.exclude(pmid, reason, screened_by="ai", confidence=confidence):
                    counts["excluded"] += 1
            else:
                self.mark_pending(pmid)
                counts["review"] += 1

        return counts

    # =========================================================================
    # STATISTICS & REPORTING
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get corpus statistics."""
        included_with_extraction = len([
            p for p in self.included_pmids if p in self.extractions
        ])
        included_with_quality = len([
            p for p in self.included_pmids if p in self.quality_ratings
        ])

        return {
            "total_screened": len(self.included_pmids) + len(self.excluded_pmids),
            "included": len(self.included_pmids),
            "excluded": len(self.excluded_pmids),
            "pending": len(self.pending_pmids),
            "with_extraction": included_with_extraction,
            "with_quality_rating": included_with_quality,
            "extraction_coverage": included_with_extraction / len(self.included_pmids) if self.included_pmids else 0,
            "quality_coverage": included_with_quality / len(self.included_pmids) if self.included_pmids else 0
        }

    def get_evidence_summary(self) -> str:
        """Generate a text summary of the evidence corpus for GDG context."""
        stats = self.get_stats()
        extractions = self.get_citable_evidence()

        lines = [
            f"## Evidence Corpus Summary",
            f"",
            f"**{stats['included']} papers included** ({stats['excluded']} excluded)",
            f"",
        ]

        if extractions:
            # Group by study design
            by_design = {}
            for e in extractions:
                design = e.study_design or "Unknown"
                if design not in by_design:
                    by_design[design] = []
                by_design[design].append(e)

            lines.append("### By Study Design")
            for design, papers in sorted(by_design.items()):
                lines.append(f"- **{design}**: {len(papers)} papers")

            lines.append("")
            lines.append("### Key Studies")
            for e in extractions[:5]:  # Top 5
                lines.append(f"- PMID {e.pmid}: {e.title[:60]}... ({e.study_design}, n={e.sample_size or 'NR'})")

        return "\n".join(lines)

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_citations(self, pmids: List[str]) -> Dict[str, Any]:
        """
        Validate a list of PMIDs against the corpus.

        Returns dict with 'valid', 'invalid', and 'all_valid' flag.
        """
        valid = [p for p in pmids if p in self.included_pmids]
        invalid = [p for p in pmids if p not in self.included_pmids]

        return {
            "valid": valid,
            "invalid": invalid,
            "all_valid": len(invalid) == 0
        }

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize corpus to dictionary."""
        return {
            "project_id": self.project_id,
            "included_pmids": list(self.included_pmids),
            "excluded_pmids": self.excluded_pmids,
            "pending_pmids": list(self.pending_pmids),
            "extractions": {k: v.to_dict() for k, v in self.extractions.items()},
            "screening_decisions": {k: v.to_dict() for k, v in self.screening_decisions.items()},
            "quality_ratings": self.quality_ratings,  # Assumes quality ratings are already dicts
            "created_at": self.created_at,
            "last_modified": self.last_modified
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvidenceCorpus':
        """Deserialize corpus from dictionary."""
        corpus = cls(
            project_id=data.get("project_id"),
            included_pmids=set(data.get("included_pmids", [])),
            excluded_pmids=data.get("excluded_pmids", {}),
            pending_pmids=set(data.get("pending_pmids", [])),
            quality_ratings=data.get("quality_ratings", {}),
            created_at=data.get("created_at", ""),
            last_modified=data.get("last_modified", "")
        )

        # Restore extractions
        for pmid, ext_data in data.get("extractions", {}).items():
            corpus.extractions[pmid] = ExtractedEvidence.from_dict(ext_data)

        # Restore screening decisions
        for pmid, dec_data in data.get("screening_decisions", {}).items():
            corpus.screening_decisions[pmid] = ScreeningDecision.from_dict(dec_data)

        return corpus

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'EvidenceCorpus':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # =========================================================================
    # INTERNAL
    # =========================================================================

    def _touch(self) -> None:
        """Update last_modified timestamp."""
        self.last_modified = datetime.now().isoformat()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_corpus_from_session() -> Optional[EvidenceCorpus]:
    """Get evidence corpus from Streamlit session state."""
    try:
        import streamlit as st
        return st.session_state.get('evidence_corpus')
    except ImportError:
        return None


def init_corpus_in_session(project_id: Optional[str] = None) -> EvidenceCorpus:
    """Initialize or get evidence corpus in Streamlit session state."""
    try:
        import streamlit as st
        if 'evidence_corpus' not in st.session_state:
            st.session_state.evidence_corpus = EvidenceCorpus(project_id=project_id)
        return st.session_state.evidence_corpus
    except ImportError:
        return EvidenceCorpus(project_id=project_id)


def extract_pmids_from_text(text: str) -> Set[str]:
    """
    Extract PMIDs from text (for citation validation).

    Looks for patterns like:
    - PMID: 12345678
    - PMID 12345678
    - (PMID: 12345678)
    """
    import re
    pattern = r'PMID[:\s]*(\d{7,8})'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return set(matches)
