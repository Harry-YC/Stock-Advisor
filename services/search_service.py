"""
Search Service Module

Handles business logic for literature search, including:
- Query refinement and suggestions using LLMs
- Search execution and orchestration (future)
- Result ranking and post-processing (future)
"""

import logging
from config import settings

import hashlib
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

from core.pubmed_client import PubMedClient, Citation
from core.query_parser import AdaptiveQueryParser
from core.ranking import rank_citations, rank_citations_with_domain, RANKING_PRESETS
from core.utils import extract_simple_query
from core.database import PaperSignalDAO

logger = logging.getLogger(__name__)

class SearchService:
    """Service for handling literature search operations."""

    def __init__(self, openai_api_key: str = None, model: str = None):
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MODEL

    def generate_search_refinements(
        self,
        query: str,
        total_results: int,
        citations: list,
        max_suggestions: int = 4
    ) -> list:
        """
        Generate search refinement suggestions using AI.

        Args:
            query: Original search query
            total_results: Number of results found
            citations: Sample of top citations
            max_suggestions: Maximum number of suggestions

        Returns:
            List of suggested refinement queries
        """
        try:
            from core.llm_utils import get_llm_client

            client = get_llm_client(api_key=self.openai_api_key, model=self.model)

            # Build context from top citations
            citation_context = ""
            for i, c in enumerate(citations[:5], 1):
                # Handle both dict and object access safely
                title = c.title if hasattr(c, 'title') else c.get('title', 'Untitled')
                citation_context += f"{i}. {title[:100]}\n"

            prompt = f"""Based on this literature search, suggest {max_suggestions} refined search queries.

Original query: {query}
Results found: {total_results}

Top papers:
{citation_context}

Suggest refinements that:
1. Narrow down if too many results (>500)
2. Broaden if too few results (<10)
3. Add specific filters (disease subtypes, patient populations, study types)
4. Explore related angles not covered

Return ONLY the refined queries, one per line, no numbering or explanation."""

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical literature search expert."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=500,
                temperature=0.7
            )

            text = response.choices[0].message.content.strip()
            suggestions = [line.strip() for line in text.split('\n') if line.strip()]
            return suggestions[:max_suggestions]

        except Exception as e:
            logger.error(f"Failed to generate search refinements: {e}")
            return []

    def execute_search(
        self,
        query: str,
        project_id: str,
        citation_dao,
        search_dao,
        query_cache_dao,
        search_context_dao=None,  # Optional: for persisting search context
        db=None,  # Optional: for paper signal boosts
        max_results: int = 100,
        ranking_mode: str = "Balanced",
        filters: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = "palliative_surgery",
        relevance_threshold: float = 0.3,
        use_two_pass: bool = True,
        domain_weight: float = 0.3,
        composite_threshold: float = 0.0,
        use_union_filter: bool = True,
        top_n_for_llm: int = 50
    ) -> Dict[str, Any]:
        """
        Execute a full literature search with AI optimization, fallback, and ranking.

        Args:
            query: Search query string
            project_id: Project ID
            citation_dao: CitationDAO instance
            search_dao: SearchHistoryDAO instance
            query_cache_dao: QueryCacheDAO instance
            search_context_dao: Optional SearchContextDAO for persisting metadata
            db: Optional DatabaseManager for paper signal boosts
            max_results: Maximum results to retrieve
            ranking_mode: Ranking preset name
            filters: Optional search filters
            domain: Domain for relevance boost (default: "palliative_surgery", None to disable)
            relevance_threshold: Minimum relevance score to include (default: 0.3, 0 to disable)
            use_two_pass: Use two-pass ranking (fast heuristic + LLM on top-N)
            domain_weight: Weight of domain score in composite (0.3 = 30% domain, 70% relevance)
            composite_threshold: Minimum composite score to include (0.0-1.0)
            use_union_filter: Apply domain union filter to search query
            top_n_for_llm: Number of top candidates for LLM re-ranking (if use_two_pass=True)

        Returns:
            Dict with search results and metadata
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        filters = filters or {}

        # Load domain config for union filter
        domain_config = None
        union_filter = ""
        exclusion_filter = ""
        if domain:
            try:
                from config.domain_config import get_domain_config
                domain_config = get_domain_config(domain)
                if domain_config:
                    union_filter = domain_config.get_union_filter()
                    exclusion_filter = domain_config.get_exclusion_filter()
            except ImportError:
                logger.warning("Could not import domain_config")

        # 1. Query Optimization & Caching
        # Include domain settings in cache key to prevent wrong cached results
        optimized_query = query
        cache_key_data = f"{query}|domain={domain}|union={use_union_filter}|dw={domain_weight}"
        query_hash = hashlib.sha256(cache_key_data.encode()).hexdigest()
        query_type = "DIRECT"
        query_explanation = ""
        query_confidence = "high"
        is_cached = False

        cached_query = query_cache_dao.get_cached_query(query_hash)

        if cached_query:
            optimized_query = cached_query['optimized_query']
            query_type = cached_query.get('query_type', 'DIRECT')
            is_cached = True
        else:
            try:
                parser = AdaptiveQueryParser(openai_api_key=self.openai_api_key, model=self.model, domain=domain)
                result = parser.parse(query)
                if result.optimized_query:
                    optimized_query = result.optimized_query
                    query_type = result.query_type
                    query_explanation = getattr(result, 'explanation', "")
                    query_confidence = getattr(result, 'confidence', "high")
                    
                    # Cache it
                    query_cache_dao.save_query(
                        query_hash=query_hash, 
                        original_query=query, 
                        optimized_query=optimized_query, 
                        query_type=result.query_type
                    )
            except Exception as e:
                logger.warning(f"Query optimization failed: {e}")
                optimized_query = query

        # 2. Apply Filters (Query augmentation)
        final_query = optimized_query
        applied_filters_list = []
        
        clinical_category = filters.get('clinical_category', 'None')
        clinical_scope = filters.get('clinical_scope', 'Broad')
        
        if clinical_category != "None":
            try:
                parser_for_filters = AdaptiveQueryParser(openai_api_key=self.openai_api_key)
                scope_key = clinical_scope.split()[0]
                final_query = parser_for_filters.add_clinical_filter(final_query, clinical_category, scope_key)
                applied_filters_list.append(f"{clinical_category}/{scope_key}")
            except Exception:
                pass

        if filters.get('quality_gate', False):
            try:
                parser_for_filters = AdaptiveQueryParser(openai_api_key=self.openai_api_key)
                final_query = parser_for_filters.add_quality_gate(final_query)
                applied_filters_list.append("Quality Gate")
            except Exception:
                pass

        # 2b. Apply Domain Union Filter (preserves recall with OR-based filter)
        union_filter_applied = False
        if use_union_filter and union_filter:
            # Use AND with union filter - requires query + any domain term
            final_query = f"({final_query}) AND {union_filter}"
            union_filter_applied = True
            applied_filters_list.append(f"Domain Union ({domain})")
            logger.info(f"Applied union filter for domain: {domain}")

        # 2c. Apply Domain Exclusion Filter (removes clearly irrelevant)
        if use_union_filter and exclusion_filter:
            final_query = f"({final_query}) {exclusion_filter}"
            applied_filters_list.append("Exclusion Filter")

        # 3. Execute Search (PubMed)
        client = PubMedClient(email=settings.PUBMED_EMAIL, api_key=settings.PUBMED_API_KEY)
        client_filters = {}
        if filters.get('date_from'): client_filters["date_from"] = filters['date_from']
        if filters.get('date_to'): client_filters["date_to"] = filters['date_to']
        if filters.get('pub_types'): client_filters["article_types"] = filters['pub_types']

        search_result = client.search(
            query=final_query, 
            max_results=max_results, 
            filters=client_filters if client_filters else None
        )

        # 4. Fallback Strategies
        used_fallback = False
        fallback_from_query = None
        
        if search_result["count"] == 0 and optimized_query != query:
            # Fallback 1: Simple query
            simpler_query = extract_simple_query(query)
            fallback_from_query = optimized_query
            search_result = client.search(query=simpler_query, max_results=max_results, filters=client_filters if client_filters else None)
            
            if search_result["count"] > 0:
                optimized_query = simpler_query
                used_fallback = True
            elif search_result["count"] == 0:
                # Fallback 2: AI Keyword Extraction
                try:
                    parser_ai = AdaptiveQueryParser(openai_api_key=self.openai_api_key, model=self.model)
                    keyword_prompt = f"Extract ONLY the 2-4 core MEDICAL CONCEPTS from this clinical question... Query: {query}"
                    # Hacky access to internal method, assumes it exists or we replicate logic
                    # In true refactor we should expose a method on AdaptiveQueryParser
                    # For now using the logic copied from UI
                    # But we don't have _call_openai exposed publicly on parser usually?
                    # Let's assume we can instantiate OpenAI client locally if needed or use public method.
                    # Actually AdaptiveQueryParser likely has methods.
                    # Let's use a simpler approach for now to avoid breaking if private method used.
                    # We will reuse generate_search_refinements logic concept or similar.
                    # Or just rely on direct client usage since we are in Service.
                    from core.llm_utils import get_llm_client
                    tmp_client = get_llm_client(api_key=self.openai_api_key, model=self.model)
                    response = tmp_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": "You extract key medical search terms."}, {"role": "user", "content": keyword_prompt}]
                    )
                    ultra_simple = response.choices[0].message.content.strip()
                    search_result = client.search(query=ultra_simple, max_results=max_results, filters=None)
                    
                    if search_result["count"] > 0:
                        optimized_query = ultra_simple
                        used_fallback = True
                except Exception:
                    # Fallback 3: Regex/Rule based
                    import re
                    words = re.findall(r'\b\w+\b', query.lower())
                    medical_keywords = {'cancer', 'tumor', 'therapy', 'treatment', 'drug', 'clinical', 'patient'}
                    found = [w for w in words if w in medical_keywords]
                    if found:
                        ultra_simple = ' '.join(found[:3])
                        search_result = client.search(query=ultra_simple, max_results=max_results, filters=None)
                        if search_result["count"] > 0:
                            optimized_query = ultra_simple
                            used_fallback = True

        # 5. Fetch Citations (Cache + Network)
        citations = []
        if search_result["pmids"]:
            cached_map = citation_dao.get_citations_batch(search_result["pmids"])
            cached_pmids = set(cached_map.keys())
            new_pmids = [p for p in search_result["pmids"] if p not in cached_pmids]
            
            new_citations_map = {}
            if new_pmids:
                new_fetched, failed_batches = client.fetch_citations(new_pmids)
                if failed_batches:
                    import logging
                    logger = logging.getLogger("literature_review.search_service")
                    logger.warning(f"Some citation batches failed to fetch: {len(failed_batches)} batch(es)")
                for c in new_fetched:
                    new_citations_map[c.pmid] = c
            
            # Reconstruct list respecting order
            for pmid in search_result["pmids"]:
                if pmid in cached_map:
                    c_dict = cached_map[pmid]
                    citations.append(Citation(
                        pmid=c_dict['pmid'], title=c_dict['title'], authors=c_dict.get('authors',[]),
                        journal=c_dict.get('journal',''), year=str(c_dict.get('year','')), 
                        abstract=c_dict.get('abstract',''), doi=c_dict.get('doi',''), 
                        fetched_at=c_dict.get('fetched_at',''),
                        publication_types=c_dict.get('publication_types', None)
                    ))
                elif pmid in new_citations_map:
                    citations.append(new_citations_map[pmid])

        # 6. Rank Results
        preset_key = ranking_mode.lower().replace(" ", "_")
        ranking_weights = RANKING_PRESETS.get(preset_key, RANKING_PRESETS["balanced"])

        citation_dicts = [
            {
                "pmid": c.pmid, "title": c.title, "authors": c.authors,
                "journal": c.journal, "year": c.year, "abstract": c.abstract,
                "doi": c.doi, "publication_types": getattr(c, 'publication_types', None),
                "fetched_at": getattr(c, 'fetched_at', '')
            } for c in citations
        ]

        # Use two-pass ranking (fast heuristic + LLM on top-N) if enabled
        if use_two_pass and domain:
            scored_citations = rank_citations_with_domain(
                citations=citation_dicts,
                weights=ranking_weights,
                domain=domain,
                original_query=query,
                use_llm_rerank=True,
                top_n_for_llm=top_n_for_llm,
                composite_threshold=composite_threshold,
                domain_weight=domain_weight,
                openai_api_key=self.openai_api_key,
                model=self.model
            )
        else:
            # Use original ranking function
            scored_citations = rank_citations(
                citations=citation_dicts,
                weights=ranking_weights,
                original_query=query,
                use_ai_relevance=True,
                openai_api_key=self.openai_api_key,
                model=self.model,
                domain=domain
            )

        # 6b. Apply Paper Signal Boosts (from user's prior selections)
        boosts_applied = 0
        if db and scored_citations:
            try:
                paper_signal_dao = PaperSignalDAO(db)
                project_id_int = int(project_id) if isinstance(project_id, str) else project_id
                pmids = [sc.citation.get('pmid') for sc in scored_citations]
                boosts = paper_signal_dao.get_boosts(project_id_int, pmids)

                if boosts:
                    for scored_cit in scored_citations:
                        pmid = scored_cit.citation.get('pmid')
                        if pmid in boosts:
                            boost = boosts[pmid]
                            scored_cit.final_score += boost
                            boosts_applied += 1
                            logger.debug(f"Applied boost {boost:.2f} to PMID {pmid}")

                    # Re-sort by final_score after applying boosts
                    scored_citations.sort(key=lambda x: x.final_score, reverse=True)
                    logger.info(f"Applied boosts to {boosts_applied} papers")
            except Exception as e:
                logger.warning(f"Failed to apply paper boosts: {e}")

        # 6c. Apply Relevance Threshold Filtering
        filtered_count = 0
        domain_filter_applied = domain is not None
        if relevance_threshold > 0 and scored_citations:
            original_count = len(scored_citations)
            scored_citations = [
                sc for sc in scored_citations
                if sc.relevance_score >= relevance_threshold
            ]
            filtered_count = original_count - len(scored_citations)
            if filtered_count > 0:
                logger.info(f"Filtered {filtered_count} papers below relevance threshold {relevance_threshold}")

        # 7. Tracking & Persistence
        search_id = search_dao.add_search(
            project_id=project_id,
            query=optimized_query,
            filters=filters,
            total_results=search_result["count"],
            retrieved_count=len(citations)
        )

        # Save search context if DAO provided
        context_id = None
        if search_context_dao and search_id:
            try:
                context_id = search_context_dao.save_context(
                    project_id=int(project_id) if isinstance(project_id, str) else project_id,
                    search_id=search_id,
                    context={
                        'ranking_mode': ranking_mode,
                        'ranking_weights': {
                            'relevance': ranking_weights.relevance,
                            'evidence': ranking_weights.evidence,
                            'recency': ranking_weights.recency,
                            'influence': ranking_weights.influence
                        },
                        'query_explanation': query_explanation,
                        'query_confidence': query_confidence,
                        'query_type': query_type,
                        'selected_pmids': []  # Empty initially, updated by UI
                    }
                )
                logger.info(f"Saved search context {context_id} for project {project_id}")
            except Exception as e:
                logger.warning(f"Failed to save search context: {e}")

        for c in citations:
            c_dict = {
                "pmid": c.pmid, "title": c.title, "authors": c.authors, 
                "journal": c.journal, "year": c.year, "abstract": c.abstract, 
                "doi": c.doi, "publication_types": getattr(c, 'publication_types', None),
                "keywords": None
            }
            citation_dao.upsert_citation(c_dict)
            citation_dao.add_citation_to_project(project_id, c.pmid)

        # 8. Return Result Object
        return {
            "query": query,
            "optimized_query": optimized_query,
            "final_query": final_query,
            "applied_filters": applied_filters_list,
            "used_fallback": used_fallback,
            "fallback_from_query": fallback_from_query,
            "total_count": search_result["count"],
            "retrieved_count": len(citations),
            "query_translation": search_result.get("query_translation", ""),
            "citations": citations,
            "scored_citations": scored_citations,
            "ranking_mode": ranking_mode,
            "ranking_weights": {
                "relevance": ranking_weights.relevance,
                "evidence": ranking_weights.evidence,
                "recency": ranking_weights.recency,
                "influence": ranking_weights.influence
            },
            "search_date": datetime.now().isoformat(),
            "query_type": query_type,
            "query_explanation": query_explanation,
            "query_confidence": query_confidence,
            "is_cached": is_cached,
            "search_id": search_id,
            "context_id": context_id,  # For tracking selected papers
            "domain": domain,
            "domain_filter_applied": domain_filter_applied,
            "relevance_threshold": relevance_threshold,
            "filtered_count": filtered_count,
            # v2 additions
            "use_two_pass": use_two_pass,
            "domain_weight": domain_weight,
            "composite_threshold": composite_threshold,
            "union_filter_applied": union_filter_applied,
            "union_filter": union_filter if union_filter_applied else None,
            "top_n_for_llm": top_n_for_llm
        }
