"""
Enhanced Local Retriever with Hybrid Search and Reranking

Combines BM25 (lexical) and dense vector search with cross-encoder reranking
for improved retrieval quality in literature review contexts.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from core.ingestion.embedder import LocalEmbedder
from core.ingestion.vector_store import VectorDB

logger = logging.getLogger(__name__)


# ============================================================================
# Metrics Tracking
# ============================================================================

@dataclass
class RetrievalMetrics:
    """Collects and logs diagnostic metrics for retrieval operations."""
    query: str = ""
    timings: Dict[str, float] = field(default_factory=dict)
    score_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    result_counts: Dict[str, int] = field(default_factory=dict)

    def log_stage(
        self,
        stage: str,
        duration: float,
        results: List[Dict[str, Any]],
        score_key: str = 'score'
    ):
        """Log metrics for a retrieval stage."""
        self.timings[stage] = duration
        self.result_counts[stage] = len(results)

        if results:
            scores = [
                r.get(score_key, 0)
                for r in results
                if r.get(score_key) is not None
            ]
            if scores:
                self.score_stats[stage] = {
                    'min': min(scores),
                    'max': max(scores),
                    'avg': sum(scores) / len(scores),
                    'count': len(scores)
                }
                logger.debug(
                    f"[METRICS] {stage}: {len(results)} results in {duration:.3f}s | "
                    f"scores: min={min(scores):.3f}, max={max(scores):.3f}, "
                    f"avg={sum(scores)/len(scores):.3f}"
                )

    def summary(self) -> Dict[str, Any]:
        """Return summary of all metrics."""
        total_time = sum(self.timings.values())
        return {
            'query_preview': self.query[:50] + '...' if len(self.query) > 50 else self.query,
            'total_time': total_time,
            'timings': self.timings,
            'result_counts': self.result_counts
        }


# ============================================================================
# Optional Dependencies
# ============================================================================

# BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not installed. BM25 hybrid search disabled.")

# Cross-encoder reranking
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("CrossEncoder not available for reranking")

# NLTK for better BM25 tokenization
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    # Download required NLTK data
    for resource in ['punkt', 'punkt_tab', 'stopwords']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

    NLTK_AVAILABLE = True
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
    STEMMER = PorterStemmer()
except ImportError:
    NLTK_AVAILABLE = False
    ENGLISH_STOPWORDS = set()
    STEMMER = None
    logger.warning("NLTK not available. Using basic tokenization for BM25.")


# ============================================================================
# Utility Functions
# ============================================================================

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def maximal_marginal_relevance(
    query_embedding: List[float],
    doc_embeddings: List[List[float]],
    docs: List[Dict[str, Any]],
    lambda_param: float = 0.5,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Select diverse documents using Maximal Marginal Relevance (MMR).

    MMR balances relevance to the query with diversity among selected documents.
    Higher lambda = more relevance, lower lambda = more diversity.

    Args:
        query_embedding: Query vector
        doc_embeddings: Document vectors (same order as docs)
        docs: List of document dicts
        lambda_param: Trade-off between relevance (1) and diversity (0)
        top_k: Number of documents to select

    Returns:
        Selected documents in MMR order
    """
    if not docs or not doc_embeddings:
        return []

    if len(docs) <= top_k:
        return docs

    selected_indices = []
    remaining_indices = list(range(len(docs)))

    # Pre-compute relevance scores
    relevance_scores = [
        cosine_similarity(query_embedding, emb)
        for emb in doc_embeddings
    ]

    while len(selected_indices) < top_k and remaining_indices:
        best_idx = None
        best_mmr_score = float('-inf')

        for idx in remaining_indices:
            relevance = relevance_scores[idx]

            # Maximum similarity to already selected documents
            if selected_indices:
                max_sim_to_selected = max(
                    cosine_similarity(doc_embeddings[idx], doc_embeddings[s])
                    for s in selected_indices
                )
            else:
                max_sim_to_selected = 0

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    return [docs[i] for i in selected_indices]


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenize text for BM25 with proper preprocessing.

    Uses NLTK if available for:
    - Word tokenization
    - Stopword removal
    - Stemming (Porter stemmer)
    """
    if not text:
        return []

    text_lower = text.lower()

    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text_lower)
            tokens = [t for t in tokens if t.isalnum() and t not in ENGLISH_STOPWORDS]
            tokens = [STEMMER.stem(t) for t in tokens]
            return tokens
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {e}")

    # Fallback to basic tokenization
    tokens = text_lower.split()
    tokens = [t for t in tokens if t.isalnum() and len(t) > 1]
    return tokens


# ============================================================================
# Local Retriever
# ============================================================================

class LocalRetriever:
    """
    Enhanced retriever with hybrid search (BM25 + dense) and cross-encoder reranking.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        embedder: Optional[LocalEmbedder] = None,
        vector_db: Optional[VectorDB] = None
    ):
        """
        Initialize the Local Retriever.

        Args:
            config: Configuration dictionary (optional)
            embedder: Optional existing LocalEmbedder instance
            vector_db: Optional existing VectorDB instance
        """
        self.config = config or {}

        # Retrieval settings
        retrieval_config = self.config.get('retrieval', {})
        self.use_hybrid = retrieval_config.get('use_hybrid', True) and BM25_AVAILABLE
        self.use_reranking = retrieval_config.get('use_reranking', True) and RERANKER_AVAILABLE
        self.reranker_model = retrieval_config.get(
            'reranker_model',
            'cross-encoder/ms-marco-MiniLM-L-6-v2'
        )
        self.dense_weight = retrieval_config.get('dense_weight', 0.5)
        self.bm25_weight = retrieval_config.get('bm25_weight', 0.5)
        self.rerank_top_k = retrieval_config.get('rerank_top_k', 20)
        self.use_hyde = retrieval_config.get('use_hyde', False)
        self.hyde_model = retrieval_config.get('hyde_model', 'gpt-5-mini')

        # Use provided instances or create new ones
        self.embedder = embedder
        self.vector_db = vector_db

        # Lazy-loaded components
        self._reranker = None
        self._bm25_index = None
        self._bm25_corpus = None
        self._bm25_doc_map = None

    def _ensure_embedder(self):
        """Ensure embedder is initialized."""
        if self.embedder is None:
            from config import settings
            self.embedder = LocalEmbedder(
                model_name=getattr(settings, 'EMBEDDING_MODEL', 'nomic-ai/nomic-embed-text-v1.5'),
                device=getattr(settings, 'EMBEDDING_DEVICE', 'auto'),
            )

    def _ensure_vector_db(self):
        """Ensure vector DB is initialized."""
        if self.vector_db is None:
            from config import settings
            self._ensure_embedder()
            config = {
                "storage_path": str(getattr(settings, 'QDRANT_STORAGE_PATH', './data/vector_storage')),
                "collection_name": getattr(settings, 'QDRANT_COLLECTION_NAME', 'literature_review_v1'),
                "vector_size": self.embedder.get_embedding_dimension(),
                "distance": "Cosine"
            }
            self.vector_db = VectorDB(config)

    def _get_reranker(self):
        """Lazy load the cross-encoder reranker."""
        if self._reranker is None and self.use_reranking:
            logger.info(f"Loading reranker model: {self.reranker_model}")
            self._reranker = CrossEncoder(self.reranker_model)
        return self._reranker

    def _generate_hypothetical_doc(self, query: str) -> str:
        """
        Generate a hypothetical answer document for HyDE retrieval.

        For vague queries, generating a hypothetical answer and embedding that
        can improve semantic retrieval by matching the embedding space of
        actual answer documents rather than question embeddings.
        """
        try:
            from openai import OpenAI

            client = OpenAI(timeout=30.0)
            response = client.chat.completions.create(
                model=self.hyde_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Write a short factual paragraph (2-3 sentences) that would "
                        f"answer this question about clinical trials or pharmaceutical "
                        f"data: {query}"
                    )
                }],
                max_completion_tokens=150,
            )
            hypothetical = response.choices[0].message.content.strip()
            logger.info(f"HyDE generated: {hypothetical[:100]}...")
            return hypothetical

        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}, using original query")
            return query

    def _build_bm25_index(self, project_filter: Optional[str] = None) -> bool:
        """Build BM25 index from all documents in vector store."""
        if not BM25_AVAILABLE:
            return False

        self._ensure_vector_db()

        try:
            all_docs = self.vector_db.scroll_all_documents(project_filter=project_filter)

            if not all_docs:
                logger.warning("No documents found for BM25 index")
                return False

            self._bm25_corpus = []
            self._bm25_doc_map = []

            for doc in all_docs:
                content = doc.get('content', '')
                tokens = tokenize_for_bm25(content)
                self._bm25_corpus.append(tokens)
                self._bm25_doc_map.append(doc)

            self._bm25_index = BM25Okapi(self._bm25_corpus)
            logger.info(f"Built BM25 index with {len(self._bm25_corpus)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            return False

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search using BM25."""
        if self._bm25_index is None:
            return []

        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores = self._bm25_index.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self._bm25_doc_map[idx]
                content = doc.get('parent_content') or doc.get('content', '')

                results.append({
                    "content": content,
                    "child_content": doc.get('content', ''),
                    "source": doc.get('source', 'Unknown'),
                    "score": float(scores[idx]),
                    "chunk_index": doc.get('chunk_index'),
                    "parent_id": doc.get('parent_id'),
                    "chunk_type": doc.get('chunk_type', 'legacy'),
                    "retrieval_method": "bm25"
                })

        return results

    def _dense_search(
        self,
        query: str,
        top_k: int = 10,
        project_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search using dense vector embeddings."""
        self._ensure_embedder()
        self._ensure_vector_db()

        results = self.vector_db.search_by_text(
            query,
            self.embedder,
            top_k=top_k,
            project_filter=project_filter
        )

        formatted_results = []
        for res in results:
            payload = res['payload']
            content = payload.get('parent_content') or payload.get('content', '')

            formatted_results.append({
                "content": content,
                "child_content": payload.get('content', ''),
                "source": payload.get('source', 'Unknown'),
                "score": res['score'],
                "chunk_index": payload.get('chunk_index'),
                "parent_id": payload.get('parent_id'),
                "chunk_type": payload.get('chunk_type', 'legacy'),
                "retrieval_method": "dense"
            })

        return formatted_results

    def _reciprocal_rank_fusion(
        self,
        results_list: List[List[Dict[str, Any]]],
        weights: List[float],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """Combine multiple result lists using Reciprocal Rank Fusion (RRF)."""
        doc_scores = {}
        doc_data = {}

        for results, weight in zip(results_list, weights):
            for rank, doc in enumerate(results):
                content = doc.get('content', '')
                if not content:
                    continue

                rrf_score = weight / (k + rank + 1)

                if content in doc_scores:
                    doc_scores[content] += rrf_score
                else:
                    doc_scores[content] = rrf_score
                    doc_data[content] = doc

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for content, score in sorted_docs:
            doc = doc_data[content].copy()
            doc['rrf_score'] = score
            results.append(doc)

        return results

    def _rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rerank results using cross-encoder."""
        if not results or not self.use_reranking:
            return results[:top_k]

        reranker = self._get_reranker()
        if reranker is None:
            return results[:top_k]

        pairs = [(query, r['content']) for r in results]

        try:
            scores = reranker.predict(pairs)

            for i, score in enumerate(scores):
                results[i]['rerank_score'] = float(score)

            reranked = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)
            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:top_k]

    def _deduplicate_by_parent(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate results by parent_id."""
        if not results:
            return results

        seen_parents = {}
        deduplicated = []

        for result in results:
            parent_id = result.get('parent_id')

            if parent_id is None:
                deduplicated.append(result)
                continue

            if parent_id not in seen_parents:
                seen_parents[parent_id] = result
                deduplicated.append(result)
            else:
                existing_score = (
                    seen_parents[parent_id].get('rerank_score') or
                    seen_parents[parent_id].get('score', 0)
                )
                new_score = result.get('rerank_score') or result.get('score', 0)

                if new_score > existing_score:
                    deduplicated.remove(seen_parents[parent_id])
                    seen_parents[parent_id] = result
                    deduplicated.append(result)

        return deduplicated

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        project_filter: Optional[str] = None,
        force_rebuild_bm25: bool = False,
        use_hyde: bool = False,
        use_query_expansion: bool = False,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid search with optional reranking.

        Args:
            query: User's question
            top_k: Number of final results to return
            project_filter: Optional project name to filter results
            force_rebuild_bm25: Force rebuild of BM25 index
            use_hyde: Use HyDE for dense search
            use_query_expansion: Use LLM to expand query with synonyms
            use_mmr: Use MMR for diverse results
            mmr_lambda: MMR trade-off (0=diversity, 1=relevance)

        Returns:
            List of results with content and metadata
        """
        metrics = RetrievalMetrics(query=query)
        retrieval_methods = []

        # Determine candidates to fetch for reranking
        candidate_k = self.rerank_top_k if self.use_reranking else top_k

        # Query expansion
        query_variants = [query]
        if use_query_expansion:
            from .query_expansion import expand_query
            expansion_start = time.time()
            query_variants = expand_query(query)
            if len(query_variants) > 1:
                retrieval_methods.append("query_expansion")
                logger.info(f"Query expanded to {len(query_variants)} variants")

        # HyDE
        dense_query = query
        if use_hyde or self.use_hyde:
            hyde_start = time.time()
            dense_query = self._generate_hypothetical_doc(query)
            if dense_query != query:
                retrieval_methods.append("hyde")

        logger.info(
            f"Retrieving top {top_k} for: '{query[:50]}...' "
            f"(hybrid={self.use_hybrid}, rerank={self.use_reranking})"
        )

        # 1. Dense search
        dense_start = time.time()

        if len(query_variants) > 1:
            all_dense_results = []
            for variant in query_variants:
                variant_results = self._dense_search(
                    variant,
                    top_k=candidate_k,
                    project_filter=project_filter
                )
                all_dense_results.extend(variant_results)

            # Deduplicate
            seen_content = {}
            for result in all_dense_results:
                key = result.get('content', '')[:200]
                if key not in seen_content or result.get('score', 0) > seen_content[key].get('score', 0):
                    seen_content[key] = result

            dense_results = sorted(
                seen_content.values(),
                key=lambda x: x.get('score', 0),
                reverse=True
            )[:candidate_k]
        else:
            dense_results = self._dense_search(
                dense_query,
                top_k=candidate_k,
                project_filter=project_filter
            )

        dense_duration = time.time() - dense_start
        metrics.log_stage("dense", dense_duration, dense_results)
        retrieval_methods.append("dense")

        # 2. BM25 search (if enabled)
        bm25_results = []
        if self.use_hybrid:
            if self._bm25_index is None or force_rebuild_bm25:
                self._build_bm25_index(project_filter=project_filter)

            if self._bm25_index is not None:
                bm25_start = time.time()
                bm25_results = self._bm25_search(query, top_k=candidate_k)
                bm25_duration = time.time() - bm25_start
                metrics.log_stage("bm25", bm25_duration, bm25_results)
                retrieval_methods.append("bm25")

                # Combine with RRF
                combined_results = self._reciprocal_rank_fusion(
                    [dense_results, bm25_results],
                    [self.dense_weight, self.bm25_weight]
                )
            else:
                combined_results = dense_results
        else:
            combined_results = dense_results

        # 3. Rerank
        if self.use_reranking and len(combined_results) > 0:
            rerank_start = time.time()
            final_results = self._rerank(
                query,
                combined_results,
                top_k * 2 if use_mmr else top_k
            )
            rerank_duration = time.time() - rerank_start
            metrics.log_stage("rerank", rerank_duration, final_results, score_key='rerank_score')
            retrieval_methods.append("rerank")
        else:
            final_results = combined_results[:top_k * 2 if use_mmr else top_k]

        # 4. MMR for diversity
        if use_mmr and len(final_results) > top_k:
            self._ensure_embedder()
            try:
                query_embedding = self.embedder.embed_query(query)
                doc_contents = [r.get('child_content') or r.get('content', '') for r in final_results]
                doc_embeddings = self.embedder.embed_documents(doc_contents, show_progress=False)

                final_results = maximal_marginal_relevance(
                    query_embedding=query_embedding,
                    doc_embeddings=doc_embeddings,
                    docs=final_results,
                    lambda_param=mmr_lambda,
                    top_k=top_k
                )
                retrieval_methods.append("mmr")
            except Exception as e:
                logger.warning(f"MMR failed: {e}")
                final_results = final_results[:top_k]
        else:
            final_results = final_results[:top_k]

        # Deduplicate by parent
        final_results = self._deduplicate_by_parent(final_results)

        # Add retrieval info
        for result in final_results:
            result['retrieval_methods'] = retrieval_methods

        logger.info(
            f"Retrieved {len(final_results)} results using: {', '.join(retrieval_methods)}"
        )

        return final_results

    def invalidate_bm25_index(self):
        """Invalidate BM25 index (call after document changes)."""
        self._bm25_index = None
        self._bm25_corpus = None
        self._bm25_doc_map = None
        logger.info("BM25 index invalidated")
