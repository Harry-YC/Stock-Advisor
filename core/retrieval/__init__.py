"""
Retrieval Package for Literature Review

Provides hybrid retrieval with:
- Dense vector search (sentence-transformers)
- BM25 sparse search
- RRF fusion
- Cross-encoder reranking
- HyDE (Hypothetical Document Embeddings)
- Query expansion
- MMR diversity
"""

from .retriever import LocalRetriever, maximal_marginal_relevance
from .query_expansion import expand_query

__all__ = [
    "LocalRetriever",
    "maximal_marginal_relevance",
    "expand_query",
]
