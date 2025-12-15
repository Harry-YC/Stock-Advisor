"""
Clinical Utility Composite (CUC) ranking system.
Scores papers based on relevance, evidence quality, and recency.

Includes semantic search using BioBERT embeddings for medical domain relevance.
"""
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import os
from openai import OpenAI

# Optional imports for semantic search
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False


# Evidence hierarchy mapping (Publication Type → Score)
PT_SCORE_MAP = [
    ("practice guideline", 1.00),
    ("guideline", 1.00),
    ("systematic review", 0.95),
    ("meta-analysis", 0.95),
    ("randomized controlled trial", 0.90),
    ("clinical trial", 0.80),
    ("phase iii", 0.80),
    ("phase iv", 0.80),
    ("cohort", 0.65),
    ("case-control", 0.65),
    ("comparative study", 0.65),
    ("cross-sectional", 0.55),
    ("diagnostic", 0.55),
    ("case series", 0.25),
    ("case report", 0.25),
    ("editorial", 0.10),
    ("comment", 0.10),
    ("letter", 0.10),
]

# Problematic publication types (exclude or zero score)
PROBLEMATIC_PT = {
    "retracted publication",
    "retraction of publication",
    "expression of concern"
}


@dataclass
class RankingWeights:
    """User-tunable weights for composite scoring."""
    relevance: float = 0.40
    evidence: float = 0.60
    influence: float = 0.00  # Phase 2
    recency: float = 0.00    # Optional boost

    def __post_init__(self):
        """Validate weights sum to ~1.0."""
        total = self.relevance + self.evidence + self.influence + self.recency
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"Weights must sum to ~1.0, got {total}")


@dataclass
class ScoredCitation:
    """Citation with decomposed scoring for explainability."""
    citation: dict  # Your original citation object
    relevance_score: float
    evidence_score: float
    influence_score: float
    recency_score: float
    domain_score: float  # Domain-specific relevance (0-1)
    final_score: float
    rank_position: int
    explanation: List[str]  # Human-readable reasons
    matched_keywords: List[str] = None  # Domain keywords that matched
    penalty_keywords: List[str] = None  # Negative indicators that matched

    def __post_init__(self):
        """Set default values for optional fields."""
        if self.matched_keywords is None:
            self.matched_keywords = []
        if self.penalty_keywords is None:
            self.penalty_keywords = []


def evidence_score(publication_types: Optional[List[str]]) -> float:
    """
    Score based on study design hierarchy (0.0 to 1.0).

    Args:
        publication_types: List of PubMed publication type strings

    Returns:
        Evidence score (higher = stronger design)
    """
    if not publication_types:
        return 0.50  # Neutral when unknown

    pts = [pt.lower() for pt in publication_types]
    pts_joined = " ".join(pts)

    # Exclude problematic records
    if any(bad in pts_joined for bad in PROBLEMATIC_PT):
        return 0.0

    # Find highest-scoring publication type
    score = 0.0
    for keyword, value in PT_SCORE_MAP:
        if any(keyword in pt for pt in pts):
            score = max(score, value)

    # Default for unknown types
    return score if score > 0 else 0.50


def relevance_from_rank(rank: int, total: int) -> float:
    """
    Convert PubMed Best Match rank to relevance score.
    Uses logarithmic decay (diminishing returns by position).

    Args:
        rank: Position in PubMed results (1-indexed)
        total: Total number of results

    Returns:
        Relevance score (0.0 to 1.0)
    """
    # Avoid division by zero
    if rank < 1:
        rank = 1

    # Log-based scoring with slower decay
    return 1.0 / math.log2(rank + 1.5)


def ai_relevance_score(
    original_query: str,
    title: str,
    abstract: Optional[str],
    openai_api_key: Optional[str] = None,
    model: str = "gpt-5-mini"
) -> float:
    """
    Use AI to score semantic relevance between query and paper.
    More accurate than position-based scoring for broad/fallback queries.

    Args:
        original_query: User's original search query
        title: Paper title
        abstract: Paper abstract (optional)
        openai_api_key: OpenAI API key
        model: OpenAI model to use

    Returns:
        Relevance score (0.0 to 1.0)
    """
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        # Fall back to neutral score if no API key
        return 0.5

    try:
        from core.llm_utils import get_llm_client
        client = get_llm_client(api_key=openai_api_key, model=model)

        # Build paper summary (use abstract if available, otherwise just title)
        paper_text = f"Title: {title}"
        if abstract and len(abstract.strip()) > 0:
            paper_text += f"\nAbstract: {abstract[:500]}"  # Limit to first 500 chars

        prompt = f"""Rate how relevant this paper is to the research question on a scale of 0.0 to 1.0.
- 1.0 = Highly relevant, directly addresses the question
- 0.7-0.9 = Relevant, discusses related topics
- 0.4-0.6 = Somewhat relevant, tangentially related
- 0.0-0.3 = Not relevant, different topic

Research Question: {original_query}

Paper:
{paper_text}

Return ONLY a number between 0.0 and 1.0, nothing else."""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a research librarian who scores paper relevance."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )

        score_text = response.choices[0].message.content.strip()
        score = float(score_text)
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]

    except Exception as e:
        print(f"Warning: AI relevance scoring failed: {e}")
        return 0.5  # Neutral score on error


def recency_boost(year: Optional[str], half_life: int = 5) -> float:
    """
    Exponential decay by publication age.

    Args:
        year: Publication year (string or int)
        half_life: Years for score to decay by 50%

    Returns:
        Recency score (0.0 to 1.0)
    """
    try:
        from datetime import datetime
        current_year = datetime.now().year
        age = max(0, current_year - int(year))
        return math.exp(-math.log(2) * age / half_life)
    except (ValueError, TypeError):
        return 0.5  # Neutral when unknown


def influence_score(rcr: Optional[float] = None, jif: Optional[float] = None) -> float:
    """
    Score based on citation influence (Phase 2 - RCR preferred).

    Args:
        rcr: Relative Citation Ratio from iCite (article-level)
        jif: Journal Impact Factor (journal-level, discouraged)

    Returns:
        Influence score (0.0 to 1.0)
    """
    if rcr is not None:
        # Squash heavy tails (RCR ~1.0 → score ~0.5)
        return 1 - math.exp(-min(5.0, rcr))

    if jif is not None:
        # If user provides licensed JIF (with caveats)
        return 1 - math.exp(-min(10.0, jif / 10.0))

    return 0.5  # Neutral when unknown


def compute_domain_score(
    title: str,
    abstract: Optional[str],
    domain_config: 'DomainConfig'
) -> Tuple[float, List[str], List[str]]:
    """
    Fast heuristic domain scoring using DomainConfig (no LLM).

    Returns a domain score (0-1) and lists of matched/penalty keywords
    for explainability.

    Args:
        title: Paper title
        abstract: Paper abstract (optional)
        domain_config: DomainConfig instance with keywords

    Returns:
        Tuple of (score, matched_keywords, penalty_keywords)
        - score: float 0.0-1.0 (0.5 is neutral)
        - matched_keywords: list of positive keywords found
        - penalty_keywords: list of negative keywords found
    """
    text = f"{title or ''} {abstract or ''}".lower()

    if not text.strip():
        return (0.5, [], [])

    matched = []
    penalties = []
    score = 0.5  # Neutral baseline

    # High relevance keywords (+0.08 each, max +0.4)
    for kw in domain_config.high_relevance_keywords:
        if kw.lower() in text:
            matched.append(kw)
    score += min(0.4, len(matched) * 0.08)

    # Procedure keywords (+0.05 each, max +0.15)
    proc_matched = []
    for kw in domain_config.procedure_keywords:
        if kw.lower() in text:
            proc_matched.append(kw)
    score += min(0.15, len(proc_matched) * 0.05)
    matched.extend(proc_matched)

    # Outcome keywords (+0.03 each, max +0.1)
    for kw in domain_config.outcome_keywords:
        if kw.lower() in text:
            matched.append(kw)
            score += 0.03
    score = min(score, 1.0)  # Cap at 1.0

    # Negative keywords (-0.12 each, max -0.35)
    for kw in domain_config.negative_keywords:
        if kw.lower() in text:
            penalties.append(kw)
    score -= min(0.35, len(penalties) * 0.12)

    return (max(0.0, min(1.0, score)), matched[:10], penalties[:5])


def domain_relevance_boost(
    title: str,
    abstract: Optional[str],
    domain: str = "palliative_surgery"
) -> float:
    """
    Compute domain-specific relevance boost/penalty.

    DEPRECATED: Use compute_domain_score() with DomainConfig instead.
    This function is kept for backwards compatibility.

    Args:
        title: Paper title
        abstract: Paper abstract (optional)
        domain: Domain identifier

    Returns:
        Float between -0.3 and +0.55 to add to relevance score
    """
    if domain is None:
        return 0.0

    try:
        from config.domain_config import get_domain_config
        domain_config = get_domain_config(domain)
        if domain_config is None:
            return 0.0

        score, _, _ = compute_domain_score(title, abstract, domain_config)
        # Convert 0-1 score to -0.3 to +0.3 boost
        # 0.5 -> 0, 0.0 -> -0.3, 1.0 -> +0.3
        return (score - 0.5) * 0.6
    except ImportError:
        return 0.0


def generate_explanation(
    citation: dict,
    rel_score: float,
    ev_score: float,
    inf_score: float,
    rec_score: float
) -> List[str]:
    """
    Generate human-readable explanation chips.

    Returns:
        List of explanation strings (e.g., ["RCT", "Recent (2023)", "High relevance"])
    """
    from datetime import datetime

    chips = []

    # Evidence chips
    pub_types = citation.get("publication_types", [])
    if pub_types:
        for pt in pub_types:
            pt_lower = pt.lower()
            if "randomized controlled trial" in pt_lower:
                chips.append("🔬 RCT")
                break
            elif "systematic review" in pt_lower or "meta-analysis" in pt_lower:
                chips.append("📊 Systematic Review")
                break
            elif "guideline" in pt_lower:
                chips.append("📋 Practice Guideline")
                break

    # Recency chip
    year = citation.get("year")
    if year:
        try:
            age = datetime.now().year - int(year)
            if age <= 2:
                chips.append(f"🆕 Recent ({year})")
            elif age >= 10:
                chips.append(f"📅 {year}")
            else:
                chips.append(f"📅 {year}")
        except ValueError:
            pass

    # Relevance chip (only if high)
    if rel_score > 0.7:
        chips.append("⭐ Highly relevant")

    # Evidence quality chip
    if ev_score >= 0.90:
        chips.append("🏅 Strong evidence")
    elif ev_score <= 0.25:
        chips.append("⚠️ Weak design")

    return chips


def rank_citations(
    citations: List[dict],
    weights: RankingWeights,
    rcr_cache: Optional[Dict[str, float]] = None,
    original_query: Optional[str] = None,
    use_ai_relevance: bool = False,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-5-mini",
    domain: Optional[str] = "palliative_surgery"
) -> List[ScoredCitation]:
    """
    Rank citations using Clinical Utility Composite score.

    Args:
        citations: List of citation dicts (from PubMed fetch)
        weights: RankingWeights instance with tunable parameters
        rcr_cache: Optional dict mapping PMID → RCR score (Phase 2)
        original_query: User's original search query (for AI relevance scoring)
        use_ai_relevance: Use AI to score relevance instead of PubMed rank
        openai_api_key: OpenAI API key (for AI relevance scoring)
        model: OpenAI model to use (for AI relevance scoring)
        domain: Domain for relevance boost (default: "palliative_surgery", None to disable)

    Returns:
        List of ScoredCitation objects, sorted by final_score (descending)
    """
    if not citations:
        return []

    rcr_cache = rcr_cache or {}
    scored = []

    # Load domain config if specified
    domain_config = None
    if domain:
        try:
            from config.domain_config import get_domain_config
            domain_config = get_domain_config(domain)
        except ImportError:
            pass

    for idx, citation in enumerate(citations, start=1):
        try:
            # Compute relevance score
            if use_ai_relevance and original_query:
                # Use AI to score semantic relevance
                rel_score = ai_relevance_score(
                    original_query=original_query,
                    title=citation.get("title", ""),
                    abstract=citation.get("abstract", ""),
                    openai_api_key=openai_api_key,
                    model=model
                )
            else:
                # Use position-based relevance
                rel_score = relevance_from_rank(idx, len(citations))

            # Compute domain score and get matched/penalty keywords
            dom_score = 0.5  # Neutral default
            matched_kw = []
            penalty_kw = []
            if domain_config:
                dom_score, matched_kw, penalty_kw = compute_domain_score(
                    title=citation.get("title", ""),
                    abstract=citation.get("abstract", ""),
                    domain_config=domain_config
                )
                # Apply domain boost to relevance for backwards compatibility
                domain_boost = (dom_score - 0.5) * 0.6
                rel_score = max(0.0, min(1.0, rel_score + domain_boost))

            # Compute other subscores
            ev_score = evidence_score(citation.get("publication_types"))
            inf_score = influence_score(rcr=rcr_cache.get(citation.get("pmid")))
            rec_score = recency_boost(citation.get("year"))

            # Composite score
            final = (
                weights.relevance * rel_score +
                weights.evidence * ev_score +
                weights.influence * inf_score +
                weights.recency * rec_score
            )

            # Generate explanation
            explanation = generate_explanation(
                citation, rel_score, ev_score, inf_score, rec_score
            )

            scored.append(ScoredCitation(
                citation=citation,
                relevance_score=rel_score,
                evidence_score=ev_score,
                influence_score=inf_score,
                recency_score=rec_score,
                domain_score=dom_score,
                final_score=final,
                rank_position=idx,
                explanation=explanation,
                matched_keywords=matched_kw,
                penalty_keywords=penalty_kw
            ))
        except Exception as e:
            # Log error but continue processing
            print(f"Warning: Error scoring citation {citation.get('pmid', 'unknown')}: {e}")
            # Use neutral scores as fallback
            scored.append(ScoredCitation(
                citation=citation,
                relevance_score=0.5,
                evidence_score=0.5,
                influence_score=0.5,
                recency_score=0.5,
                domain_score=0.5,
                final_score=0.5,
                rank_position=idx,
                explanation=["⚠️ Scoring error"],
                matched_keywords=[],
                penalty_keywords=[]
            ))

    # Sort by final score (descending)
    scored.sort(key=lambda x: x.final_score, reverse=True)

    return scored


def rank_citations_with_domain(
    citations: List[dict],
    weights: RankingWeights,
    domain: str = "palliative_surgery",
    original_query: Optional[str] = None,
    use_llm_rerank: bool = True,
    top_n_for_llm: int = 50,
    composite_threshold: float = 0.0,
    domain_weight: float = 0.3,
    rcr_cache: Optional[Dict[str, float]] = None,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-5-mini",
    use_feedback_boost: bool = True,
) -> List[ScoredCitation]:
    """
    Two-pass ranking with domain awareness for efficiency and explainability.

    Pass 1: Fast heuristic domain scoring on ALL citations
    Pass 2: LLM relevance scoring only on top-N candidates

    The final composite score is:
        (1 - domain_weight) * relevance_score + domain_weight * domain_score + feedback_boost

    Args:
        citations: List of citation dicts (from PubMed fetch)
        weights: RankingWeights instance (relevance, evidence, etc.)
        domain: Domain identifier (default: "palliative_surgery")
        original_query: User's original search query (for LLM scoring)
        use_llm_rerank: Whether to use LLM for top-N re-ranking
        top_n_for_llm: Number of top candidates for LLM scoring
        composite_threshold: Minimum composite score to include (0.0-1.0)
        domain_weight: Weight of domain score in composite (default: 0.3)
        rcr_cache: Optional dict mapping PMID → RCR score
        openai_api_key: OpenAI API key (for LLM scoring)
        model: LLM model to use for relevance scoring
        use_feedback_boost: Whether to apply user feedback marks boost (default: True)

    Returns:
        List of ScoredCitation objects, sorted by final_score (descending)
    """
    if not citations:
        return []

    rcr_cache = rcr_cache or {}

    # Load domain config
    domain_config = None
    if domain:
        try:
            from config.domain_config import get_domain_config
            domain_config = get_domain_config(domain)
        except ImportError:
            pass

    # Load feedback service for user marks boost
    feedback_boost_fn = None
    if use_feedback_boost and original_query:
        try:
            from services.feedback_service import get_relevance_boost
            feedback_boost_fn = get_relevance_boost
        except ImportError:
            pass

    # =========================================================================
    # PASS 1: Fast heuristic scoring on ALL citations
    # =========================================================================
    pass1_results = []

    for idx, citation in enumerate(citations, start=1):
        try:
            title = citation.get("title", "")
            abstract = citation.get("abstract", "")

            # Compute domain score (fast heuristic)
            dom_score = 0.5
            matched_kw = []
            penalty_kw = []
            if domain_config:
                dom_score, matched_kw, penalty_kw = compute_domain_score(
                    title=title,
                    abstract=abstract,
                    domain_config=domain_config
                )

            # Use position-based relevance for Pass 1 (fast)
            rel_score = relevance_from_rank(idx, len(citations))

            # Compute other scores
            ev_score = evidence_score(citation.get("publication_types"))
            inf_score = influence_score(rcr=rcr_cache.get(citation.get("pmid")))
            rec_score = recency_boost(citation.get("year"))

            # Compute feedback boost from user marks (improves RAG over time)
            fb_boost = 0.0
            if feedback_boost_fn:
                text_to_check = f"{title} {abstract}"
                fb_boost = feedback_boost_fn(text_to_check, original_query)

            # Pass 1 composite (domain-weighted + feedback boost)
            composite = (1 - domain_weight) * rel_score + domain_weight * dom_score + fb_boost

            pass1_results.append({
                "idx": idx,
                "citation": citation,
                "rel_score": rel_score,
                "dom_score": dom_score,
                "ev_score": ev_score,
                "inf_score": inf_score,
                "rec_score": rec_score,
                "fb_boost": fb_boost,
                "composite": composite,
                "matched_kw": matched_kw,
                "penalty_kw": penalty_kw
            })
        except Exception as e:
            print(f"Warning: Pass 1 error for {citation.get('pmid', 'unknown')}: {e}")
            pass1_results.append({
                "idx": idx,
                "citation": citation,
                "rel_score": 0.5,
                "dom_score": 0.5,
                "ev_score": 0.5,
                "inf_score": 0.5,
                "rec_score": 0.5,
                "fb_boost": 0.0,
                "composite": 0.5,
                "matched_kw": [],
                "penalty_kw": []
            })

    # Sort by composite score for top-N selection
    pass1_results.sort(key=lambda x: x["composite"], reverse=True)

    # =========================================================================
    # PASS 2: LLM re-ranking on top-N candidates
    # =========================================================================
    if use_llm_rerank and original_query and openai_api_key:
        top_n = pass1_results[:top_n_for_llm]
        rest = pass1_results[top_n_for_llm:]

        for item in top_n:
            try:
                llm_rel = ai_relevance_score(
                    original_query=original_query,
                    title=item["citation"].get("title", ""),
                    abstract=item["citation"].get("abstract", ""),
                    openai_api_key=openai_api_key,
                    model=model
                )
                # Update relevance with LLM score
                item["rel_score"] = llm_rel
                # Recalculate composite with new relevance
                item["composite"] = (1 - domain_weight) * llm_rel + domain_weight * item["dom_score"]
            except Exception as e:
                print(f"Warning: LLM scoring failed for {item['citation'].get('pmid', 'unknown')}: {e}")
                # Keep Pass 1 scores

        # Re-sort top-N by updated composite
        top_n.sort(key=lambda x: x["composite"], reverse=True)
        pass1_results = top_n + rest

    # =========================================================================
    # Build final ScoredCitation objects
    # =========================================================================
    scored = []

    for item in pass1_results:
        # Calculate final score using weights
        final = (
            weights.relevance * item["rel_score"] +
            weights.evidence * item["ev_score"] +
            weights.influence * item["inf_score"] +
            weights.recency * item["rec_score"]
        )

        # Generate explanation
        explanation = generate_explanation(
            item["citation"],
            item["rel_score"],
            item["ev_score"],
            item["inf_score"],
            item["rec_score"]
        )

        # Add domain-specific explanation chips
        if item["dom_score"] > 0.7:
            explanation.append("🎯 High domain relevance")
        elif item["dom_score"] < 0.3:
            explanation.append("⚠️ Low domain relevance")

        scored.append(ScoredCitation(
            citation=item["citation"],
            relevance_score=item["rel_score"],
            evidence_score=item["ev_score"],
            influence_score=item["inf_score"],
            recency_score=item["rec_score"],
            domain_score=item["dom_score"],
            final_score=final,
            rank_position=item["idx"],
            explanation=explanation,
            matched_keywords=item["matched_kw"],
            penalty_keywords=item["penalty_kw"]
        ))

    # Apply composite threshold filtering
    if composite_threshold > 0:
        scored = [
            s for s in scored
            if (1 - domain_weight) * s.relevance_score + domain_weight * s.domain_score >= composite_threshold
        ]

    # Final sort by final_score (descending)
    scored.sort(key=lambda x: x.final_score, reverse=True)

    return scored


# =============================================================================
# SEMANTIC SEARCH WITH BioBERT
# =============================================================================

# Cache for loaded models (avoid reloading)
_MODEL_CACHE: Dict[str, any] = {}


def get_semantic_model(model_name: str = "dmis-lab/biobert-base-cased-v1.2"):
    """
    Load and cache a sentence-transformer model.

    Args:
        model_name: Model identifier. Recommended:
            - "dmis-lab/biobert-base-cased-v1.2" - BioBERT for biomedical text
            - "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb" - Fine-tuned for similarity
            - "all-mpnet-base-v2" - General purpose (good fallback)
            - "all-MiniLM-L6-v2" - Fast, smaller model

    Returns:
        SentenceTransformer model or None if not available
    """
    if not SEMANTIC_SEARCH_AVAILABLE:
        return None

    if model_name not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Warning: Failed to load model {model_name}: {e}")
            return None

    return _MODEL_CACHE[model_name]


def local_semantic_relevance(
    original_query: str,
    papers: List[Dict],
    model_name: str = "dmis-lab/biobert-base-cased-v1.2",
    max_abstract_chars: int = 500
) -> List[float]:
    """
    Fast local semantic relevance scoring using sentence-transformers.

    Uses BioBERT embeddings for medical/biomedical domain specificity.
    50-100x faster than API calls and zero cost.

    Args:
        original_query: User's search query
        papers: List of paper dicts with 'title' and 'abstract' keys
        model_name: Sentence-transformer model to use
        max_abstract_chars: Maximum abstract characters to include

    Returns:
        List of relevance scores (0.0 to 1.0), one per paper
    """
    if not SEMANTIC_SEARCH_AVAILABLE:
        # Return neutral scores if dependencies not installed
        return [0.5] * len(papers)

    if not papers:
        return []

    model = get_semantic_model(model_name)
    if model is None:
        return [0.5] * len(papers)

    try:
        # Encode query
        query_embedding = model.encode(original_query, convert_to_tensor=True)

        # Build paper texts (title + truncated abstract)
        paper_texts = []
        for paper in papers:
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')[:max_abstract_chars] if paper.get('abstract') else ''
            paper_texts.append(f"{title} {abstract}".strip())

        # Batch encode papers
        paper_embeddings = model.encode(paper_texts, convert_to_tensor=True)

        # Compute cosine similarities
        cos_scores = util.cos_sim(query_embedding, paper_embeddings)[0]

        # Normalize from [-1, 1] to [0, 1]
        scores = [(score.item() + 1) / 2 for score in cos_scores]

        return scores

    except Exception as e:
        print(f"Warning: Semantic relevance scoring failed: {e}")
        return [0.5] * len(papers)


def batch_semantic_relevance(
    queries: List[str],
    papers: List[Dict],
    model_name: str = "dmis-lab/biobert-base-cased-v1.2"
) -> List[List[float]]:
    """
    Compute semantic relevance for multiple queries against papers.

    Useful for expert-specific relevance scoring where each expert
    has different search interests.

    Args:
        queries: List of search queries (one per expert/perspective)
        papers: List of paper dicts
        model_name: Sentence-transformer model to use

    Returns:
        2D list: scores[query_idx][paper_idx]
    """
    if not SEMANTIC_SEARCH_AVAILABLE or not papers or not queries:
        return [[0.5] * len(papers) for _ in queries]

    model = get_semantic_model(model_name)
    if model is None:
        return [[0.5] * len(papers) for _ in queries]

    try:
        # Encode all queries
        query_embeddings = model.encode(queries, convert_to_tensor=True)

        # Build paper texts
        paper_texts = []
        for paper in papers:
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')[:500] if paper.get('abstract') else ''
            paper_texts.append(f"{title} {abstract}".strip())

        # Encode all papers
        paper_embeddings = model.encode(paper_texts, convert_to_tensor=True)

        # Compute all pairwise similarities
        cos_scores = util.cos_sim(query_embeddings, paper_embeddings)

        # Normalize and convert to list
        results = []
        for query_scores in cos_scores:
            normalized = [(score.item() + 1) / 2 for score in query_scores]
            results.append(normalized)

        return results

    except Exception as e:
        print(f"Warning: Batch semantic scoring failed: {e}")
        return [[0.5] * len(papers) for _ in queries]


def rank_by_semantic_relevance(
    papers: List[Dict],
    query: str,
    model_name: str = "dmis-lab/biobert-base-cased-v1.2",
    top_k: Optional[int] = None
) -> List[Tuple[Dict, float]]:
    """
    Rank papers by semantic relevance to a query.

    Args:
        papers: List of paper dicts
        query: Search query
        model_name: Model to use
        top_k: Return only top K papers (None = all)

    Returns:
        List of (paper, score) tuples, sorted by score descending
    """
    scores = local_semantic_relevance(query, papers, model_name)

    # Pair papers with scores and sort
    ranked = list(zip(papers, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)

    if top_k:
        ranked = ranked[:top_k]

    return ranked


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

RANKING_PRESETS = {
    "discovery": RankingWeights(relevance=0.60, evidence=0.30, recency=0.10),
    "clinical_appraisal": RankingWeights(relevance=0.25, evidence=0.60, recency=0.15),
    "balanced": RankingWeights(relevance=0.40, evidence=0.40, recency=0.20),
    # Preclinical/translational drug development preset
    "preclinical": RankingWeights(relevance=0.35, evidence=0.40, recency=0.25),
}
