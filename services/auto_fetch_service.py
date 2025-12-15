"""
Auto-Fetch Service

Automatically extracts clinical context from user questions
and fetches relevant knowledge from PubMed/ClinicalTrials.gov.

Triggered on each user question:
1. Extract condition/procedure/outcome from question
2. Update clinical context
3. Fetch relevant papers/trials in background
4. Store for injection into future expert prompts
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)


class AutoFetchService:
    """
    Automatically extract clinical context and fetch relevant knowledge.

    Triggered on each user question to:
    1. Extract concepts (condition, procedure, outcome)
    2. Update clinical context (merge with existing)
    3. Fetch relevant papers/trials in background
    4. Store in trusted_knowledge table for prompt injection
    """

    def __init__(self, api_key: str = None, db=None):
        """
        Initialize service.

        Args:
            api_key: API key for LLM extraction
            db: DatabaseManager instance (creates default if not provided)
        """
        self.api_key = api_key or getattr(settings, 'OPENAI_API_KEY', None) or getattr(settings, 'GOOGLE_API_KEY', None)
        self.db = db or self._get_db()
        self._executor = ThreadPoolExecutor(max_workers=2)

    def _get_db(self):
        """Get default database manager."""
        from core.database import DatabaseManager
        return DatabaseManager(settings.OUTPUTS_DIR / "literature_review.db")

    def on_question(
        self,
        project_id: int,
        question: str,
        background: bool = True
    ) -> Optional[Dict]:
        """
        Called when user asks a question.

        Extracts concepts, updates clinical context, triggers background fetch.

        Args:
            project_id: Project ID
            question: User's question
            background: Whether to fetch in background (True) or blocking (False)

        Returns:
            Updated clinical context dict, or None if extraction failed
        """
        from core.query_extractor import ClinicalQueryExtractor
        from core.database import ProgramProfileDAO

        # 1. Extract concepts
        extractor = ClinicalQueryExtractor(api_key=self.api_key)
        concepts = extractor.extract(question)

        if concepts.is_empty():
            logger.debug(f"No concepts extracted from question: {question[:50]}...")
            return None

        # 2. Build context update from extracted concepts (palliative surgery fields)
        context_update = {
            "condition": concepts.conditions[0] if concepts.conditions else None,
            "anatomy": concepts.anatomy[0] if concepts.anatomy else None,
            "procedure": concepts.procedures[0] if concepts.procedures else None,
            "outcome": concepts.outcomes[0] if concepts.outcomes else None,
            "cancer": concepts.cancers[0] if concepts.cancers else None,
            "procedures": concepts.procedures,  # Full list
            "outcomes": concepts.outcomes,  # Full list
        }

        # 3. Get existing context and merge
        profile_dao = ProgramProfileDAO(self.db)
        existing = profile_dao.get(project_id) or {}

        # Merge: only update fields that are None or empty in existing
        merged = {}
        for key in ["condition", "anatomy", "procedure", "outcome", "cancer"]:
            if context_update.get(key) and not existing.get(key):
                merged[key] = context_update[key]
            else:
                merged[key] = existing.get(key)

        # For list fields, merge unique values
        for key in ["procedures", "outcomes"]:
            existing_list = existing.get(key, [])
            if isinstance(existing_list, str):
                try:
                    import json
                    existing_list = json.loads(existing_list)
                except:
                    existing_list = []
            new_list = context_update.get(key, [])
            merged[key] = list(set(existing_list + new_list))

        # 4. Save updated context
        profile_dao.upsert(project_id, merged)
        logger.info(f"Updated context for project {project_id}: condition={merged.get('condition')}, procedure={merged.get('procedure')}")

        # 5. Trigger background fetch
        if background:
            self._executor.submit(self._fetch_and_store, project_id, merged)
        else:
            self._fetch_and_store(project_id, merged)

        return merged

    def _fetch_and_store(self, project_id: int, context: Dict) -> int:
        """
        Fetch from sources and store in trusted_knowledge DB.

        Args:
            project_id: Project ID
            context: Clinical context dict

        Returns:
            Number of entries saved
        """
        from core.database import TrustedKnowledgeDAO, ProgramProfileDAO
        from core.pubmed_client import PubMedClient

        knowledge_dao = TrustedKnowledgeDAO(self.db)
        saved = 0

        try:
            # Build search queries from context
            queries = self._build_queries(context)

            if not queries:
                logger.debug("No queries to execute for context")
                return 0

            # Fetch from PubMed
            pubmed = PubMedClient(
                email=getattr(settings, 'PUBMED_EMAIL', 'user@example.com'),
                api_key=getattr(settings, 'PUBMED_API_KEY', None)
            )

            for query in queries[:3]:  # Max 3 queries
                try:
                    citations = self._fetch_pubmed(pubmed, query)
                    for cit in citations[:8]:  # Max 8 per query
                        confidence = self._compute_confidence(cit, context)
                        knowledge_dao.add(
                            project_id=project_id,
                            source="pubmed",
                            source_id=cit.pmid,
                            title=cit.title,
                            insight=cit.abstract[:500] if cit.abstract else "",
                            confidence=confidence,
                            entry_type="paper",
                            metadata={
                                "journal": cit.journal,
                                "year": cit.year,
                                "authors": cit.authors[:3] if cit.authors else []
                            }
                        )
                        saved += 1
                except Exception as e:
                    logger.error(f"PubMed fetch failed for query '{query[:50]}': {e}")

            # Fetch from ClinicalTrials.gov
            trials = self._fetch_trials(context)
            for trial in trials[:5]:
                try:
                    knowledge_dao.add(
                        project_id=project_id,
                        source="clinicaltrials",
                        source_id=trial.get("nct_id", ""),
                        title=trial.get("title", ""),
                        insight=trial.get("description", "")[:500],
                        confidence=0.8,  # Trials are generally high confidence
                        entry_type="trial",
                        metadata={
                            "phase": trial.get("phase", []),
                            "status": trial.get("status", "")
                        }
                    )
                    saved += 1
                except Exception as e:
                    logger.error(f"Failed to save trial: {e}")

            # Sync to KnowledgeStore for prompt injection
            if saved > 0:
                try:
                    knowledge_dao.sync_to_knowledge_store(project_id)
                except Exception as e:
                    logger.warning(f"Failed to sync to KnowledgeStore: {e}")

            logger.info(f"Auto-fetched {saved} entries for project {project_id}")

        except Exception as e:
            logger.error(f"Auto-fetch failed: {e}")
            import traceback
            traceback.print_exc()

        return saved

    def _build_queries(self, context: Dict) -> List[str]:
        """Build PubMed queries from clinical context."""
        queries = []
        condition = context.get("condition")
        procedure = context.get("procedure")
        cancer = context.get("cancer")
        outcome = context.get("outcome")

        # Palliative surgery focused queries
        if condition and procedure:
            queries.append(f'("{condition}"[tiab]) AND ("{procedure}"[tiab]) AND (palliative[tiab] OR palliation[tiab])')
        if condition:
            queries.append(f'("{condition}"[tiab]) AND (surgery[tiab] OR surgical[tiab]) AND (palliative[tiab])')
        if procedure and cancer:
            queries.append(f'("{procedure}"[tiab]) AND ("{cancer}"[tiab]) AND (outcomes[tiab] OR survival[tiab])')
        if condition and outcome:
            queries.append(f'("{condition}"[tiab]) AND ("{outcome}"[tiab])')

        return queries[:3]  # Max 3 queries

    def _fetch_pubmed(self, client, query: str, max_results: int = 15) -> list:
        """Execute PubMed search and fetch citations."""
        try:
            # Add quality filter for palliative surgery
            quality_query = f'({query}) AND ("retrospective" OR "prospective" OR "cohort" OR "systematic review"[pt])'
            result = client.search(quality_query, max_results=max_results)
            pmids = result.get("pmids", [])

            # Fallback to unfiltered
            if not pmids:
                result = client.search(query, max_results=max_results)
                pmids = result.get("pmids", [])

            if pmids:
                citations, _ = client.fetch_citations(pmids)
                return citations

        except Exception as e:
            logger.error(f"PubMed fetch error: {e}")

        return []

    def _fetch_trials(self, context: Dict) -> List[Dict]:
        """Fetch from ClinicalTrials.gov."""
        import requests

        parts = []
        if context.get("condition"):
            parts.append(context["condition"])
        if context.get("procedure"):
            parts.append(context["procedure"])
        if context.get("cancer"):
            parts.append(context["cancer"])

        # Add palliative context
        parts.append("palliative")

        if len(parts) < 2:  # Need at least condition + palliative
            return []

        try:
            response = requests.get(
                "https://clinicaltrials.gov/api/v2/studies",
                params={
                    "query.term": " AND ".join(parts),
                    "pageSize": 10,
                    "format": "json"
                },
                timeout=30
            )
            response.raise_for_status()

            studies = response.json().get("studies", [])
            return [{
                "nct_id": s.get("protocolSection", {}).get(
                    "identificationModule", {}
                ).get("nctId", ""),
                "title": s.get("protocolSection", {}).get(
                    "identificationModule", {}
                ).get("briefTitle", ""),
                "description": s.get("protocolSection", {}).get(
                    "descriptionModule", {}
                ).get("briefSummary", ""),
                "phase": s.get("protocolSection", {}).get(
                    "designModule", {}
                ).get("phases", []),
                "status": s.get("protocolSection", {}).get(
                    "statusModule", {}
                ).get("overallStatus", "")
            } for s in studies]

        except Exception as e:
            logger.error(f"ClinicalTrials fetch failed: {e}")

        return []

    def _compute_confidence(self, cit, context: Dict) -> float:
        """Compute confidence score for a citation."""
        score = 0.5  # Base score

        text = f"{cit.title} {cit.abstract or ''}".lower()

        # Boost for study type
        if "systematic review" in text or "meta-analysis" in text:
            score += 0.25
        elif "randomized" in text or "rct" in text:
            score += 0.2
        elif "prospective" in text:
            score += 0.15
        elif "retrospective" in text:
            score += 0.1

        # Boost for high-impact journals
        top_journals = [
            "ann surg", "jama surg", "br j surg", "j clin oncol",
            "lancet oncol", "palliat med", "support care cancer"
        ]
        if cit.journal and any(j in cit.journal.lower() for j in top_journals):
            score += 0.15

        # Boost for matching condition/procedure
        if context.get("condition") and context["condition"].lower() in text:
            score += 0.1
        if context.get("procedure") and context["procedure"].lower() in text:
            score += 0.1

        return min(1.0, score)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def auto_fetch_on_question(
    project_id: int,
    question: str,
    api_key: str = None,
    background: bool = True
) -> Optional[Dict]:
    """
    Convenience function to trigger auto-fetch on a question.

    Args:
        project_id: Project ID
        question: User's question
        api_key: API key for LLM extraction
        background: Whether to fetch in background

    Returns:
        Updated clinical context or None
    """
    service = AutoFetchService(api_key=api_key)
    return service.on_question(project_id, question, background=background)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import os

    test_questions = [
        "Should a patient with malignant bowel obstruction undergo gastrojejunostomy?",
        "What is the role of palliative surgery for pathologic fracture in breast cancer?",
    ]

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
    service = AutoFetchService(api_key=api_key)

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        print("-" * 60)

        # Use project_id=1 for testing
        context = service.on_question(project_id=1, question=q, background=False)

        if context:
            print(f"Context updated:")
            print(f"  Condition: {context.get('condition')}")
            print(f"  Procedure: {context.get('procedure')}")
            print(f"  Cancer: {context.get('cancer')}")
            print(f"  Outcome: {context.get('outcome')}")
        else:
            print("No context update (no concepts extracted)")
