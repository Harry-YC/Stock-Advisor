"""
Two-channel search system for Palliative Surgery: Clinical + Symptom

Clinical Channel: Surgical outcomes, morbidity/mortality, comparative effectiveness
Symptom Channel: QoL, symptom palliation, functional outcomes, patient-centered

The key insight: For palliative surgery decisions, we need BOTH:
1. Clinical evidence (surgical outcomes, complications, mortality)
2. Symptom/QoL evidence (patient-reported outcomes, symptom relief, functional status)
"""

from dataclasses import dataclass, field
from typing import List
from enum import Enum

from core.query_extractor import ExtractedConcepts, get_synonyms


class SearchChannel(Enum):
    CLINICAL = "clinical"      # Surgical outcomes, morbidity/mortality
    SYMPTOM = "symptom"        # QoL, symptom palliation, functional outcomes


@dataclass
class ChannelQuery:
    """A query with its channel and purpose."""
    channel: SearchChannel
    query: str
    purpose: str  # e.g., "MBO surgical outcomes"
    priority: int = 1  # Lower = higher priority


@dataclass
class ChannelQuerySet:
    """Complete set of queries organized by channel."""
    clinical_queries: List[ChannelQuery] = field(default_factory=list)
    symptom_queries: List[ChannelQuery] = field(default_factory=list)

    def all_queries(self) -> List[ChannelQuery]:
        return self.clinical_queries + self.symptom_queries


class TwoChannelQueryBuilder:
    """
    Build separate query sets for Clinical and Symptom channels.

    Clinical Channel:
    - Surgical outcomes (morbidity, mortality)
    - Comparative effectiveness (surgery vs stent)
    - Perioperative risk factors
    - Technical success rates

    Symptom Channel:
    - Quality of life outcomes
    - Symptom palliation rates
    - Functional outcomes (oral intake, mobility)
    - Patient-reported outcomes
    """

    # Palliative anchor to ensure relevance
    PALLIATIVE_ANCHOR = "(palliative[tiab] OR symptom[tiab] OR terminal[tiab] OR advanced cancer[tiab])"

    # Cancer context anchor
    CANCER_ANCHOR = "(cancer[tiab] OR carcinoma[tiab] OR malignant[tiab] OR tumor[tiab] OR metastatic[tiab])"

    def build(self, concepts: ExtractedConcepts) -> ChannelQuerySet:
        """Build complete query set from extracted concepts."""
        result = ChannelQuerySet()

        condition_clause = self._condition_clause(concepts.conditions)
        procedure_clause = self._procedure_clause(concepts.procedures)
        cancer_clause = self._cancer_clause(concepts.cancers)

        # ================================================================
        # CLINICAL CHANNEL - Surgical outcomes, morbidity/mortality
        # ================================================================

        # Condition + procedure + outcomes
        if condition_clause and procedure_clause:
            result.clinical_queries.append(
                ChannelQuery(
                    channel=SearchChannel.CLINICAL,
                    query=f"(({condition_clause}) AND ({procedure_clause}) AND (mortality[tiab] OR morbidity[tiab] OR complication[tiab]))",
                    purpose=f"Surgical outcomes for condition",
                    priority=1
                )
            )

        # Condition + surgical outcomes
        if condition_clause:
            result.clinical_queries.extend([
                ChannelQuery(
                    channel=SearchChannel.CLINICAL,
                    query=f"(({condition_clause}) AND (surgery[tiab] OR surgical[tiab] OR operative[tiab]) AND (outcome[tiab] OR mortality[tiab]))",
                    purpose=f"Surgical management outcomes",
                    priority=1
                ),
                # Case series - PRIMARY evidence source in palliative surgery
                ChannelQuery(
                    channel=SearchChannel.CLINICAL,
                    query=f"(({condition_clause}) AND (case series[tiab] OR consecutive patients[tiab] OR clinical experience[tiab] OR institutional experience[tiab]))",
                    purpose=f"Case series evidence (supporting examples)",
                    priority=1
                ),
                ChannelQuery(
                    channel=SearchChannel.CLINICAL,
                    query=f"(({condition_clause}) AND (retrospective[tiab] OR prospective[tiab]))",
                    purpose=f"Clinical studies for condition",
                    priority=2
                ),
            ])

        # Procedure-specific outcomes
        for proc in concepts.procedures[:3]:
            proc_clause = self._term_clause(proc, "procedure")
            result.clinical_queries.extend([
                ChannelQuery(
                    channel=SearchChannel.CLINICAL,
                    query=f"(({proc_clause}) AND (outcome[tiab] OR complication[tiab] OR success[tiab] OR failure[tiab]))",
                    purpose=f"{proc} outcomes",
                    priority=2
                ),
                # Procedure-specific case series
                ChannelQuery(
                    channel=SearchChannel.CLINICAL,
                    query=f"(({proc_clause}) AND (case series[tiab] OR patients[tiab] OR experience[tiab]) AND (palliative[tiab] OR cancer[tiab] OR malignant[tiab]))",
                    purpose=f"{proc} case series",
                    priority=1
                )
            ])

        # Comparison queries (surgery vs stent, etc.)
        if len(concepts.procedures) >= 2:
            proc1_clause = self._term_clause(concepts.procedures[0], "procedure")
            proc2_clause = self._term_clause(concepts.procedures[1], "procedure")
            result.clinical_queries.append(
                ChannelQuery(
                    channel=SearchChannel.CLINICAL,
                    query=f"(({proc1_clause}) AND ({proc2_clause}) AND (comparison[tiab] OR versus[tiab] OR compared[tiab]))",
                    purpose=f"{concepts.procedures[0]} vs {concepts.procedures[1]}",
                    priority=1
                )
            )

        # Cancer-specific surgical outcomes
        if cancer_clause and condition_clause:
            result.clinical_queries.append(
                ChannelQuery(
                    channel=SearchChannel.CLINICAL,
                    query=f"(({cancer_clause}) AND ({condition_clause}) AND (surgical[tiab] OR operative[tiab]))",
                    purpose=f"Cancer-specific surgical outcomes",
                    priority=2
                )
            )

        # Risk score-based queries
        for score in concepts.scores[:2]:
            score_clause = self._term_clause(score, "score")
            result.clinical_queries.append(
                ChannelQuery(
                    channel=SearchChannel.CLINICAL,
                    query=f"(({score_clause}) AND (surgery[tiab] OR outcome[tiab] OR prognosis[tiab]))",
                    purpose=f"{score} and surgical outcomes",
                    priority=2
                )
            )

        # ================================================================
        # SYMPTOM CHANNEL - QoL, symptom palliation, functional outcomes
        # ================================================================

        # Condition + QoL
        if condition_clause:
            result.symptom_queries.extend([
                ChannelQuery(
                    channel=SearchChannel.SYMPTOM,
                    query=f"(({condition_clause}) AND (quality of life[tiab] OR QoL[tiab] OR patient-reported[tiab]))",
                    purpose=f"Quality of life in condition",
                    priority=1
                ),
                ChannelQuery(
                    channel=SearchChannel.SYMPTOM,
                    query=f"(({condition_clause}) AND (symptom[tiab] OR palliation[tiab] OR relief[tiab]))",
                    purpose=f"Symptom palliation for condition",
                    priority=1
                ),
                ChannelQuery(
                    channel=SearchChannel.SYMPTOM,
                    query=f"(({condition_clause}) AND (palliative[tiab] OR hospice[tiab] OR end of life[tiab]))",
                    purpose=f"Palliative care context",
                    priority=2
                ),
            ])

        # Procedure + QoL/symptom outcomes
        for proc in concepts.procedures[:2]:
            proc_clause = self._term_clause(proc, "procedure")
            result.symptom_queries.extend([
                ChannelQuery(
                    channel=SearchChannel.SYMPTOM,
                    query=f"(({proc_clause}) AND (quality of life[tiab] OR symptom[tiab] OR palliation[tiab]))",
                    purpose=f"{proc} QoL/symptom outcomes",
                    priority=1
                ),
                ChannelQuery(
                    channel=SearchChannel.SYMPTOM,
                    query=f"(({proc_clause}) AND (functional[tiab] OR oral intake[tiab] OR eating[tiab] OR ambulation[tiab]))",
                    purpose=f"{proc} functional outcomes",
                    priority=2
                ),
            ])

        # Outcome-specific searches
        for outcome in concepts.outcomes[:2]:
            outcome_lower = outcome.lower()
            result.symptom_queries.append(
                ChannelQuery(
                    channel=SearchChannel.SYMPTOM,
                    query=f'("{outcome}"[tiab] AND {self.CANCER_ANCHOR})',
                    purpose=f"{outcome} in cancer patients",
                    priority=2
                )
            )

        # Condition + patient-centered outcomes
        if condition_clause:
            result.symptom_queries.append(
                ChannelQuery(
                    channel=SearchChannel.SYMPTOM,
                    query=f"(({condition_clause}) AND (survival[tiab] OR prognosis[tiab] OR days at home[tiab]))",
                    purpose=f"Survival and home-based care",
                    priority=2
                )
            )

        # Score-based QoL (e.g., GOOSS for GOO)
        for score in concepts.scores[:2]:
            if score.lower() in ["gooss", "ecog", "karnofsky", "ppi"]:
                score_clause = self._term_clause(score, "score")
                result.symptom_queries.append(
                    ChannelQuery(
                        channel=SearchChannel.SYMPTOM,
                        query=f"(({score_clause}) AND (outcome[tiab] OR quality of life[tiab]))",
                        purpose=f"{score} and patient outcomes",
                        priority=2
                    )
                )

        return result

    def _term_clause(self, term: str, term_type: str) -> str:
        """Build OR clause with synonyms."""
        synonyms = get_synonyms(term, term_type)
        unique = list(dict.fromkeys(synonyms))
        return " OR ".join([f'"{s}"[tiab]' for s in unique])

    def _condition_clause(self, conditions: List[str]) -> str:
        """Build condition clause with synonyms."""
        if not conditions:
            return ""

        all_terms = []
        for cond in conditions[:2]:
            synonyms = get_synonyms(cond, "condition")
            all_terms.extend(synonyms)

        unique = list(dict.fromkeys(all_terms))
        return " OR ".join([f'"{t}"[tiab]' for t in unique])

    def _procedure_clause(self, procedures: List[str]) -> str:
        """Build procedure clause with synonyms."""
        if not procedures:
            return ""

        all_terms = []
        for proc in procedures[:2]:
            synonyms = get_synonyms(proc, "procedure")
            all_terms.extend(synonyms)

        unique = list(dict.fromkeys(all_terms))
        return " OR ".join([f'"{t}"[tiab]' for t in unique])

    def _cancer_clause(self, cancers: List[str]) -> str:
        """Build cancer clause."""
        if not cancers:
            return ""

        return " OR ".join([f'"{c}"[tiab]' for c in cancers[:2]])


# =============================================================================
# RELEVANCE FILTERS (Post-retrieval filtering)
# =============================================================================

class RelevanceFilter:
    """
    Post-retrieval filtering to exclude off-topic papers.

    Key insight: PubMed queries can be too broad. This filter removes:
    - Infection-focused papers (osteomyelitis, FRI, etc.)
    - Non-oncological fractures (osteoporosis, fragility)
    - Off-topic papers (handgrip strength, pediatric, veterinary)
    """

    # Papers must contain at least ONE of these to be relevant (oncology context)
    REQUIRED_TERMS = [
        "metastatic", "metastasis", "metastases",
        "malignant", "malignancy",
        "cancer", "carcinoma", "adenocarcinoma",
        "tumor", "tumour", "neoplasm", "neoplastic",
        "oncology", "oncologic",
        "myeloma", "plasmacytoma",
        "palliative", "terminal", "advanced disease"
    ]

    # Papers containing these in TITLE are excluded (strong signal of off-topic)
    TITLE_EXCLUSIONS = [
        "infection", "osteomyelitis", "septic",
        "fracture-related infection", "FRI",
        "osteoporosis", "osteoporotic", "fragility fracture",
        "handgrip", "grip strength", "sarcopenia",
        "pediatric", "paediatric", "child", "children", "adolescent",
        "veterinary", "canine", "feline", "equine", "murine", "mouse", "rat",
        "zebrafish", "drosophila",
        "in vitro", "cell line", "cell culture"
    ]

    # Papers containing these ANYWHERE are deprioritized (not excluded, just scored down)
    SOFT_EXCLUSIONS = [
        "curative", "adjuvant", "neoadjuvant",
        "prevention", "screening",
        "rehabilitation", "physiotherapy"
    ]

    def __init__(self, domain: str = "orthopedic_oncology"):
        """
        Initialize filter for specific domain.

        Args:
            domain: Domain name (affects which filters apply)
        """
        self.domain = domain

    def is_relevant(self, title: str, abstract: str) -> bool:
        """
        Check if paper is relevant to the palliative surgery domain.

        Args:
            title: Paper title
            abstract: Paper abstract

        Returns:
            True if paper passes relevance filter
        """
        title_lower = (title or "").lower()
        abstract_lower = (abstract or "").lower()
        combined = f"{title_lower} {abstract_lower}"

        # Check title exclusions (strong signal)
        for term in self.TITLE_EXCLUSIONS:
            if term in title_lower:
                return False

        # Check for required oncology context
        has_oncology_context = any(term in combined for term in self.REQUIRED_TERMS)
        if not has_oncology_context:
            return False

        return True

    def filter_citations(self, citations: list) -> list:
        """
        Filter a list of citations to only relevant ones.

        Args:
            citations: List of Citation objects or dicts

        Returns:
            Filtered list with only relevant papers
        """
        filtered = []

        for cit in citations:
            # Handle both Citation objects and dicts
            if hasattr(cit, 'title'):
                title = cit.title or ""
                abstract = cit.abstract or ""
            elif isinstance(cit, dict):
                title = cit.get('title', '')
                abstract = cit.get('abstract', '')
            else:
                filtered.append(cit)  # Can't filter, keep it
                continue

            if self.is_relevant(title, abstract):
                filtered.append(cit)

        return filtered

    def score_relevance(self, title: str, abstract: str) -> float:
        """
        Score relevance of a paper (0.0 to 1.0).

        Args:
            title: Paper title
            abstract: Paper abstract

        Returns:
            Relevance score (higher = more relevant)
        """
        if not self.is_relevant(title, abstract):
            return 0.0

        title_lower = (title or "").lower()
        abstract_lower = (abstract or "").lower()
        combined = f"{title_lower} {abstract_lower}"

        score = 0.5  # Base score for passing filter

        # Boost for strong oncology terms
        strong_terms = ["metastatic", "palliative", "malignant", "terminal"]
        for term in strong_terms:
            if term in combined:
                score += 0.1

        # Penalize soft exclusions
        for term in self.SOFT_EXCLUSIONS:
            if term in combined:
                score -= 0.05

        return min(1.0, max(0.0, score))


# =============================================================================
# TUMOR BIOLOGY STRATIFICATION
# =============================================================================

class TumorBiologyTagger:
    """
    Tag papers by tumor biology for stratified analysis.

    Key insight: Treatment outcomes differ significantly between:
    - Multiple myeloma (often responsive to systemic therapy, may heal with RT)
    - Solid tumor metastases (generally require fixation, progressive)

    This distinction is clinically meaningful and affects recommendations.
    """

    MYELOMA_TERMS = [
        "myeloma", "multiple myeloma", "plasmacytoma", "plasma cell",
        "light chain", "bence jones", "mm patient"
    ]

    SOLID_TUMOR_TERMS = [
        "carcinoma", "adenocarcinoma", "metastatic",
        "breast cancer", "lung cancer", "renal cell", "prostate cancer",
        "thyroid cancer", "colorectal", "hepatocellular",
        "sarcoma", "melanoma"
    ]

    TUMOR_BIOLOGY_CATEGORIES = {
        "myeloma": "Hematologic - multiple myeloma or plasmacytoma",
        "solid_tumor": "Solid tumor metastasis",
        "mixed": "Mixed tumor types or unclear",
        "unspecified": "Tumor type not specified"
    }

    def tag_tumor_biology(self, title: str, abstract: str) -> str:
        """
        Determine tumor biology category from paper text.

        Args:
            title: Paper title
            abstract: Paper abstract

        Returns:
            Tumor biology category: myeloma | solid_tumor | mixed | unspecified
        """
        text = f"{(title or '').lower()} {(abstract or '').lower()}"

        has_myeloma = any(term in text for term in self.MYELOMA_TERMS)
        has_solid = any(term in text for term in self.SOLID_TUMOR_TERMS)

        if has_myeloma and has_solid:
            return "mixed"
        elif has_myeloma:
            return "myeloma"
        elif has_solid:
            return "solid_tumor"
        else:
            return "unspecified"

    def tag_citations(self, citations: list) -> list:
        """
        Tag all citations with tumor biology category.

        Args:
            citations: List of Citation objects or dicts

        Returns:
            Same list with tumor_biology attribute/key added
        """
        for cit in citations:
            if hasattr(cit, 'title'):
                title = cit.title or ""
                abstract = cit.abstract or ""
                biology = self.tag_tumor_biology(title, abstract)
                # Add as attribute (works for dataclass)
                cit.tumor_biology = biology
            elif isinstance(cit, dict):
                title = cit.get('title', '')
                abstract = cit.get('abstract', '')
                biology = self.tag_tumor_biology(title, abstract)
                cit['tumor_biology'] = biology

        return citations

    def stratify_citations(self, citations: list) -> dict:
        """
        Stratify citations by tumor biology.

        Args:
            citations: List of citations (already tagged or will be tagged)

        Returns:
            Dict mapping category to list of citations
        """
        # Ensure all citations are tagged
        self.tag_citations(citations)

        stratified = {
            "myeloma": [],
            "solid_tumor": [],
            "mixed": [],
            "unspecified": []
        }

        for cit in citations:
            if hasattr(cit, 'tumor_biology'):
                category = cit.tumor_biology
            elif isinstance(cit, dict):
                category = cit.get('tumor_biology', 'unspecified')
            else:
                category = 'unspecified'

            if category in stratified:
                stratified[category].append(cit)
            else:
                stratified['unspecified'].append(cit)

        return stratified

    def get_stratification_summary(self, citations: list) -> str:
        """
        Get a summary of tumor biology distribution.

        Args:
            citations: List of citations

        Returns:
            Human-readable summary string
        """
        stratified = self.stratify_citations(citations)

        summary_parts = []
        total = len(citations)

        for category, cits in stratified.items():
            if cits:
                pct = (len(cits) / total * 100) if total > 0 else 0
                desc = self.TUMOR_BIOLOGY_CATEGORIES.get(category, category)
                summary_parts.append(f"- {category.replace('_', ' ').title()}: {len(cits)} papers ({pct:.0f}%)")

        if not summary_parts:
            return "No papers to stratify."

        return "**Tumor Biology Distribution:**\n" + "\n".join(summary_parts)


def tag_citations_by_tumor_biology(citations: list) -> list:
    """
    Tag citations with tumor biology category.

    Args:
        citations: List of citations

    Returns:
        Same list with tumor_biology added
    """
    tagger = TumorBiologyTagger()
    return tagger.tag_citations(citations)


def stratify_by_tumor_biology(citations: list) -> dict:
    """
    Stratify citations by tumor biology category.

    Args:
        citations: List of citations

    Returns:
        Dict mapping category to citations
    """
    tagger = TumorBiologyTagger()
    return tagger.stratify_citations(citations)


# =============================================================================
# MUST-INCLUDE QUERY BUCKETS
# =============================================================================

class MustIncludeQueries:
    """
    Additional queries to ensure critical evidence is retrieved.

    For certain topics (e.g., Mirels score), we need to ensure we find
    validation/critique literature, not just supportive papers.
    """

    # Topic-specific must-include queries
    TOPIC_QUERIES = {
        "mirels": [
            '"Mirels score" AND (specificity OR sensitivity OR validation)',
            '"Mirels score" AND (limitation OR critique OR accuracy)',
            '"impending fracture" AND (radiotherapy OR radiation) AND (fracture risk OR fracture rate)',
            '"Mirels score" AND overtreatment',
        ],
        "pathologic_fracture": [
            '"pathologic fracture" AND "prophylactic fixation" AND outcome',
            '"impending fracture" AND (observation OR conservative)',
            '"metastatic bone disease" AND "surgical decision"',
        ],
        "mbo": [  # Malignant bowel obstruction
            '"malignant bowel obstruction" AND (surgery OR surgical) AND (palliative OR outcome)',
            '"malignant bowel obstruction" AND (stent OR venting gastrostomy)',
            '"carcinomatosis peritonei" AND "bowel obstruction"',
        ],
        "goo": [  # Gastric outlet obstruction
            '"gastric outlet obstruction" AND (stent OR gastrojejunostomy)',
            '"malignant gastric obstruction" AND (EUS-GE OR endoscopic)',
        ],
    }

    @classmethod
    def get_queries_for_topic(cls, topic: str) -> List[str]:
        """Get must-include queries for a topic."""
        topic_lower = topic.lower()

        queries = []
        for key, query_list in cls.TOPIC_QUERIES.items():
            if key in topic_lower:
                queries.extend(query_list)

        return queries

    @classmethod
    def detect_topic(cls, question: str) -> List[str]:
        """Detect relevant topics from a question."""
        question_lower = question.lower()
        detected = []

        topic_keywords = {
            "mirels": ["mirels", "mirel's", "impending fracture", "prophylactic fixation"],
            "pathologic_fracture": ["pathologic fracture", "pathological fracture", "femoral metastasis", "metastatic fracture"],
            "mbo": ["bowel obstruction", "intestinal obstruction", "mbo"],
            "goo": ["gastric outlet obstruction", "goo", "pyloric obstruction"],
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in question_lower for kw in keywords):
                detected.append(topic)

        return detected


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def filter_citations_for_relevance(citations: list, domain: str = "orthopedic_oncology") -> list:
    """
    Filter citations to only include relevant papers.

    Args:
        citations: List of citations
        domain: Domain for filtering

    Returns:
        Filtered list
    """
    relevance_filter = RelevanceFilter(domain=domain)
    return relevance_filter.filter_citations(citations)


def get_must_include_queries(question: str) -> List[str]:
    """
    Get must-include queries based on the question content.

    Args:
        question: Clinical question

    Returns:
        List of additional PubMed queries to run
    """
    topics = MustIncludeQueries.detect_topic(question)
    queries = []
    for topic in topics:
        queries.extend(MustIncludeQueries.get_queries_for_topic(topic))
    return queries


def build_two_channel_queries(concepts: ExtractedConcepts) -> ChannelQuerySet:
    """Build two-channel queries from extracted concepts."""
    builder = TwoChannelQueryBuilder()
    return builder.build(concepts)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from core.query_extractor import ExtractedConcepts

    print("=" * 60)
    print("TESTING TWO-CHANNEL QUERY BUILDER (PALLIATIVE SURGERY)")
    print("=" * 60)

    concepts = ExtractedConcepts(
        conditions=["malignant bowel obstruction", "MBO"],
        anatomy=["colon", "small bowel"],
        procedures=["gastrojejunostomy", "stent"],
        outcomes=["quality of life", "symptom control"],
        scores=["GOOSS", "ECOG"],
        cancers=["ovarian cancer"],
        comparisons=["versus"]
    )

    builder = TwoChannelQueryBuilder()
    queries = builder.build(concepts)

    print("\nCLINICAL CHANNEL QUERIES:")
    for i, q in enumerate(queries.clinical_queries, 1):
        print(f"  {i}. [{q.purpose}]")
        print(f"     {q.query[:100]}...")

    print("\nSYMPTOM CHANNEL QUERIES:")
    for i, q in enumerate(queries.symptom_queries, 1):
        print(f"  {i}. [{q.purpose}]")
        print(f"     {q.query[:100]}...")
