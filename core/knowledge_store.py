"""
Knowledge Store for Literature Review Platform

Provides persistent storage for learned knowledge from expert panel discussions
and literature analysis. Enables the system to "remember" insights across sessions.

Features:
- Persona-organized fact storage
- Knowledge graph triples (Subject, Predicate, Object)
- Relevance scoring for expert selection
- Search across stored knowledge

Adapted from CI-RAG for preclinical/translational drug development context.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from config import settings


# Default storage path
DEFAULT_KNOWLEDGE_FILE = settings.OUTPUTS_DIR / "learned_knowledge.json"


class KnowledgeStore:
    """
    Manages persistent storage of learned knowledge from expert discussions.

    Knowledge is organized by expert persona and stored as JSON. Each entry
    includes the source, extraction date, and extracted facts/insights.

    Example usage:
        store = KnowledgeStore()

        # Add learned facts
        store.add_knowledge(
            persona="DMPK Scientist",
            source="Expert Panel: Compound X PK Analysis",
            facts=[
                "CYP3A4 inhibition IC50 = 2.3 µM",
                "Half-life in rat = 8.2 hours",
                "Oral bioavailability ~40% in preclinical species"
            ]
        )

        # Add knowledge graph triple
        store.add_triple(
            subject="Compound X",
            predicate="inhibits",
            object_val="CYP3A4",
            source="Expert Panel Discussion",
            context="In vitro hepatocyte study"
        )

        # Query for relevant knowledge
        relevant = store.get_relevant_knowledge_for_query(
            "What are the ADME properties of Compound X?",
            entities=["Compound X", "CYP3A4"]
        )
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the knowledge store.

        Args:
            storage_path: Path to JSON storage file. Defaults to outputs/learned_knowledge.json
        """
        self.storage_path = storage_path or DEFAULT_KNOWLEDGE_FILE
        self._ensure_storage_exists()

    def _ensure_storage_exists(self) -> None:
        """Create storage file and directory if they don't exist."""
        storage_dir = self.storage_path.parent
        if not storage_dir.exists():
            storage_dir.mkdir(parents=True, exist_ok=True)

        if not self.storage_path.exists():
            self._save_data({})

    def _load_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load knowledge data from storage file."""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_data(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        """Save knowledge data to storage file with atomic write."""
        temp_path = self.storage_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_path.replace(self.storage_path)  # Atomic on POSIX
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    # =========================================================================
    # FACT-BASED KNOWLEDGE
    # =========================================================================

    def add_knowledge(
        self,
        persona: str,
        source: str,
        facts: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add new knowledge entry for a persona.

        Args:
            persona: Expert persona (e.g., "DMPK Scientist", "Toxicology Expert")
            source: Source description (e.g., "Expert Panel: Target Validation")
            facts: List of extracted facts/insights
            metadata: Optional additional metadata

        Returns:
            Unique ID for the added knowledge entry
        """
        data = self._load_data()

        if persona not in data:
            data[persona] = []

        entry_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(data[persona])}"

        entry = {
            "id": entry_id,
            "source": source,
            "extracted": datetime.now().isoformat(),
            "facts": facts,
            "metadata": metadata or {}
        }

        data[persona].append(entry)
        self._save_data(data)

        return entry_id

    def get_knowledge(self, persona: str) -> List[Dict[str, Any]]:
        """Get all knowledge entries for a specific persona."""
        data = self._load_data()
        return data.get(persona, [])

    def get_all_knowledge(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all stored knowledge across all personas."""
        return self._load_data()

    def get_knowledge_facts(self, persona: str) -> List[str]:
        """Get just the facts (flat list) for a persona."""
        entries = self.get_knowledge(persona)
        facts = []
        for entry in entries:
            facts.extend(entry.get("facts", []))
        return facts

    def delete_knowledge(self, persona: str, entry_id: str) -> bool:
        """Delete a specific knowledge entry."""
        data = self._load_data()

        if persona not in data:
            return False

        original_length = len(data[persona])
        data[persona] = [e for e in data[persona] if e.get("id") != entry_id]

        if len(data[persona]) < original_length:
            self._save_data(data)
            return True

        return False

    def clear_persona_knowledge(self, persona: str) -> int:
        """Delete all knowledge for a specific persona. Returns count deleted."""
        data = self._load_data()

        if persona not in data:
            return 0

        count = len(data[persona])
        del data[persona]
        self._save_data(data)

        return count

    def list_personas_with_knowledge(self) -> List[str]:
        """List all personas that have stored knowledge."""
        data = self._load_data()
        return [p for p, entries in data.items() if entries and not p.startswith("_")]

    def get_knowledge_summary(self) -> Dict[str, int]:
        """Get summary of stored knowledge counts by persona."""
        data = self._load_data()
        return {
            persona: len(entries)
            for persona, entries in data.items()
            if not persona.startswith("_")
        }

    def search_knowledge(self, query: str, persona: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search knowledge entries for matching text.

        Args:
            query: Search query (case-insensitive substring match)
            persona: Optional persona to limit search to

        Returns:
            List of matching entries with persona info
        """
        data = self._load_data()
        query_lower = query.lower()
        results = []

        personas_to_search = [persona] if persona else [
            p for p in data.keys() if not p.startswith("_")
        ]

        for p in personas_to_search:
            if p not in data:
                continue
            for entry in data[p]:
                for fact in entry.get("facts", []):
                    if query_lower in fact.lower():
                        results.append({**entry, "persona": p})
                        break
                else:
                    if query_lower in entry.get("source", "").lower():
                        results.append({**entry, "persona": p})

        return results

    # =========================================================================
    # KNOWLEDGE GRAPH (TRIPLE) FUNCTIONALITY
    # =========================================================================

    def add_triple(
        self,
        subject: str,
        predicate: str,
        object_val: str,
        source: str,
        context: Optional[str] = None,
        confidence: float = 1.0
    ) -> str:
        """
        Add a knowledge triple (Subject, Predicate, Object).

        Enables cross-document reasoning by storing structured relationships.

        Args:
            subject: Entity (e.g., "Compound X", "Target Y")
            predicate: Relationship (e.g., "inhibits", "has_IC50_of", "causes_toxicity")
            object_val: Value or related entity (e.g., "CYP3A4", "2.3 µM", "hepatotoxicity")
            source: Source document/discussion
            context: Optional context (e.g., "in vitro", "rat 28-day study")
            confidence: Confidence score (0-1)

        Returns:
            Unique ID for the triple
        """
        data = self._load_data()

        if "_triples" not in data:
            data["_triples"] = []

        triple_id = f"triple_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(data['_triples'])}"

        triple = {
            "id": triple_id,
            "subject": subject.strip(),
            "predicate": predicate.strip(),
            "object": object_val.strip(),
            "source": source,
            "context": context,
            "confidence": confidence,
            "created": datetime.now().isoformat()
        }

        data["_triples"].append(triple)
        self._save_data(data)

        return triple_id

    def add_triples_batch(self, triples: List[Dict[str, Any]], source: str) -> int:
        """
        Add multiple triples at once.

        Args:
            triples: List of dicts with keys: subject, predicate, object, context (optional)
            source: Source document for all triples

        Returns:
            Number of triples added
        """
        data = self._load_data()

        if "_triples" not in data:
            data["_triples"] = []

        count = 0
        for t in triples:
            if not all(k in t for k in ["subject", "predicate", "object"]):
                continue

            triple_id = f"triple_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(data['_triples'])}"

            triple = {
                "id": triple_id,
                "subject": t["subject"].strip(),
                "predicate": t["predicate"].strip(),
                "object": t["object"].strip(),
                "source": source,
                "context": t.get("context"),
                "confidence": t.get("confidence", 1.0),
                "created": datetime.now().isoformat()
            }

            data["_triples"].append(triple)
            count += 1

        self._save_data(data)
        return count

    def get_all_triples(self) -> List[Dict[str, Any]]:
        """Get all stored triples."""
        data = self._load_data()
        return data.get("_triples", [])

    def query_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_val: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query triples by any combination of subject, predicate, object.

        All matches are case-insensitive substring matches.
        """
        triples = self.get_all_triples()
        results = []

        for t in triples:
            match = True

            if subject and subject.lower() not in t.get("subject", "").lower():
                match = False
            if predicate and predicate.lower() not in t.get("predicate", "").lower():
                match = False
            if object_val and object_val.lower() not in t.get("object", "").lower():
                match = False

            if match:
                results.append(t)

        return results

    def get_related_entities(self, entity: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all relationships involving an entity (as subject or object).

        Returns dict with 'as_subject' and 'as_object' lists.
        """
        triples = self.get_all_triples()
        entity_lower = entity.lower()

        as_subject = []
        as_object = []

        for t in triples:
            if entity_lower in t.get("subject", "").lower():
                as_subject.append(t)
            if entity_lower in t.get("object", "").lower():
                as_object.append(t)

        return {
            "as_subject": as_subject,
            "as_object": as_object
        }

    def get_triple_summary(self) -> Dict[str, Any]:
        """Get summary statistics for stored triples."""
        triples = self.get_all_triples()

        subjects = set(t.get("subject", "") for t in triples)
        predicates = set(t.get("predicate", "") for t in triples)
        objects = set(t.get("object", "") for t in triples)

        return {
            "total_triples": len(triples),
            "unique_subjects": len(subjects),
            "unique_predicates": len(predicates),
            "unique_objects": len(objects),
            "predicate_types": list(predicates)[:20]
        }

    def format_triples_for_context(self, entity: str, max_triples: int = 10) -> str:
        """
        Format relevant triples as context for LLM prompts.

        Args:
            entity: Entity to get context for
            max_triples: Maximum number of triples to include

        Returns:
            Formatted string of triples
        """
        related = self.get_related_entities(entity)
        all_related = related["as_subject"] + related["as_object"]

        if not all_related:
            return ""

        all_related.sort(
            key=lambda t: (t.get("confidence", 0), t.get("created", "")),
            reverse=True
        )
        top_triples = all_related[:max_triples]

        lines = [f"Known facts about {entity}:"]
        for t in top_triples:
            if t.get("context"):
                lines.append(f"- {t['subject']} {t['predicate']} {t['object']} ({t['context']})")
            else:
                lines.append(f"- {t['subject']} {t['predicate']} {t['object']}")

        return "\n".join(lines)

    def delete_triple(self, triple_id: str) -> bool:
        """Delete a specific triple by ID."""
        data = self._load_data()

        if "_triples" not in data:
            return False

        original_length = len(data["_triples"])
        data["_triples"] = [t for t in data["_triples"] if t.get("id") != triple_id]

        if len(data["_triples"]) < original_length:
            self._save_data(data)
            return True

        return False

    def clear_all_triples(self) -> int:
        """Delete all triples. Returns count deleted."""
        data = self._load_data()
        count = len(data.get("_triples", []))
        data["_triples"] = []
        self._save_data(data)
        return count

    # =========================================================================
    # LEARNING FROM USER INTERACTIONS
    # =========================================================================

    def add_user_correction(
        self,
        persona: str,
        original_response: str,
        corrected_response: str,
        question: str,
        project_id: Optional[int] = None
    ) -> str:
        """
        Record when a user corrects an expert's response.

        This enables the system to learn from user feedback and improve
        future responses for similar questions.

        Args:
            persona: Expert persona that was corrected
            original_response: The original expert response
            corrected_response: The user-corrected/regenerated response
            question: The clinical question being answered
            project_id: Optional project ID for project-specific learning

        Returns:
            Unique ID for the correction entry
        """
        data = self._load_data()

        if "_corrections" not in data:
            data["_corrections"] = []

        correction_id = f"correction_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(data['_corrections'])}"

        # Extract what changed (simple diff summary)
        original_summary = original_response[:200] if len(original_response) > 200 else original_response
        corrected_summary = corrected_response[:200] if len(corrected_response) > 200 else corrected_response

        entry = {
            "id": correction_id,
            "persona": persona,
            "question": question,
            "original_summary": original_summary,
            "corrected_summary": corrected_summary,
            "project_id": project_id,
            "created": datetime.now().isoformat()
        }

        data["_corrections"].append(entry)
        self._save_data(data)

        return correction_id

    def add_query_effectiveness(
        self,
        query: str,
        question_type: str,
        paper_count: int,
        selected_count: int,
        project_id: Optional[int] = None
    ) -> str:
        """
        Track the effectiveness of a search query.

        Helps the system learn which queries produce useful results.

        Args:
            query: The search query
            question_type: Type of question (Go/No-Go, Dosing, etc.)
            paper_count: Total papers returned
            selected_count: Papers selected by user as relevant
            project_id: Optional project ID

        Returns:
            Unique ID for the entry
        """
        data = self._load_data()

        if "_query_effectiveness" not in data:
            data["_query_effectiveness"] = []

        entry_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(data['_query_effectiveness'])}"

        # Calculate effectiveness score
        effectiveness = selected_count / paper_count if paper_count > 0 else 0

        entry = {
            "id": entry_id,
            "query": query,
            "question_type": question_type,
            "paper_count": paper_count,
            "selected_count": selected_count,
            "effectiveness": round(effectiveness, 3),
            "project_id": project_id,
            "created": datetime.now().isoformat()
        }

        data["_query_effectiveness"].append(entry)
        self._save_data(data)

        return entry_id

    def get_corrections_for_persona(self, persona: str) -> List[Dict[str, Any]]:
        """Get all corrections recorded for a specific persona."""
        data = self._load_data()
        corrections = data.get("_corrections", [])
        return [c for c in corrections if c.get("persona") == persona]

    def get_effective_query_patterns(
        self,
        question_type: Optional[str] = None,
        min_effectiveness: float = 0.3,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get queries that were effective (high selection rate).

        Args:
            question_type: Optional filter by question type
            min_effectiveness: Minimum effectiveness threshold
            limit: Maximum results to return

        Returns:
            List of effective query entries, sorted by effectiveness
        """
        data = self._load_data()
        queries = data.get("_query_effectiveness", [])

        # Filter by question type if specified
        if question_type:
            queries = [q for q in queries if q.get("question_type") == question_type]

        # Filter by minimum effectiveness
        queries = [q for q in queries if q.get("effectiveness", 0) >= min_effectiveness]

        # Sort by effectiveness descending
        queries.sort(key=lambda q: q.get("effectiveness", 0), reverse=True)

        return queries[:limit]

    def get_project_learning_summary(self, project_id: int) -> Dict[str, Any]:
        """
        Get a summary of all learning for a specific project.

        Args:
            project_id: Project ID to get learning for

        Returns:
            Dict with corrections, effective queries, and extracted facts
        """
        data = self._load_data()

        # Project corrections
        corrections = data.get("_corrections", [])
        project_corrections = [c for c in corrections if c.get("project_id") == project_id]

        # Project queries
        queries = data.get("_query_effectiveness", [])
        project_queries = [q for q in queries if q.get("project_id") == project_id]

        # Calculate summary stats
        avg_effectiveness = 0
        if project_queries:
            avg_effectiveness = sum(q.get("effectiveness", 0) for q in project_queries) / len(project_queries)

        # Personas that were corrected
        corrected_personas = list(set(c.get("persona") for c in project_corrections))

        return {
            "project_id": project_id,
            "total_corrections": len(project_corrections),
            "corrected_personas": corrected_personas,
            "total_searches": len(project_queries),
            "avg_query_effectiveness": round(avg_effectiveness, 3),
            "effective_queries": [
                q for q in project_queries if q.get("effectiveness", 0) >= 0.3
            ][:5],
            "recent_corrections": project_corrections[-5:] if project_corrections else []
        }

    def format_learning_for_prompt(
        self,
        persona: str,
        question_type: Optional[str] = None,
        project_id: Optional[int] = None
    ) -> str:
        """
        Format learned insights for injection into expert prompts.

        Args:
            persona: Expert persona
            question_type: Optional question type for query hints
            project_id: Optional project for project-specific learning

        Returns:
            Formatted string for prompt injection
        """
        lines = []

        # Get corrections for this persona
        corrections = self.get_corrections_for_persona(persona)
        if corrections:
            recent_corrections = corrections[-3:]  # Last 3
            lines.append("**User Feedback History:**")
            for c in recent_corrections:
                lines.append(f"- Question: '{c.get('question', '')[:100]}...'")
                lines.append(f"  Correction: {c.get('corrected_summary', '')[:150]}...")
            lines.append("")

        # Get effective query patterns for this question type
        if question_type:
            effective = self.get_effective_query_patterns(question_type, limit=3)
            if effective:
                lines.append("**Effective Search Strategies:**")
                for q in effective:
                    lines.append(f"- Query: '{q.get('query', '')}' ({int(q.get('effectiveness', 0)*100)}% selection rate)")
                lines.append("")

        if not lines:
            return ""

        header = ["=" * 50, "LEARNED INSIGHTS", "=" * 50, ""]
        return "\n".join(header + lines)

    # =========================================================================
    # PROGRAM CONCLUSIONS (INSTITUTIONAL MEMORY)
    # =========================================================================

    def add_program_conclusion(
        self,
        project_id: int,
        question: str,
        conclusion: str,
        citations: Optional[List[str]] = None,
        persona: str = "Consensus"
    ) -> str:
        """
        Save a program conclusion for institutional memory.

        Program conclusions are key findings from expert panel discussions
        that should inform future research within the same project.

        Args:
            project_id: Project ID this conclusion belongs to
            question: The research question that was answered
            conclusion: The synthesized conclusion/recommendation
            citations: Optional list of PMIDs supporting this conclusion
            persona: Source persona (default "Consensus" for panel conclusions)

        Returns:
            Unique ID for the conclusion entry
        """
        data = self._load_data()

        if "_program_conclusions" not in data:
            data["_program_conclusions"] = []

        conclusion_id = f"conclusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(data['_program_conclusions'])}"

        entry = {
            "id": conclusion_id,
            "project_id": project_id,
            "question": question,
            "conclusion": conclusion,
            "citations": citations or [],
            "persona": persona,
            "created": datetime.now().isoformat()
        }

        data["_program_conclusions"].append(entry)
        self._save_data(data)

        return conclusion_id

    def get_program_conclusions(
        self,
        project_id: int,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get stored conclusions for a project.

        Args:
            project_id: Project ID to get conclusions for
            limit: Maximum conclusions to return (most recent first)

        Returns:
            List of conclusion entries, most recent first
        """
        data = self._load_data()
        conclusions = data.get("_program_conclusions", [])

        # Filter by project
        project_conclusions = [
            c for c in conclusions
            if c.get("project_id") == project_id
        ]

        # Sort by creation date descending and limit
        project_conclusions.sort(
            key=lambda c: c.get("created", ""),
            reverse=True
        )

        return project_conclusions[:limit]

    def format_conclusions_for_prompt(
        self,
        project_id: int,
        max_conclusions: int = 3
    ) -> str:
        """
        Format prior conclusions as context for expert prompts.

        Args:
            project_id: Project ID to get conclusions for
            max_conclusions: Maximum conclusions to include

        Returns:
            Formatted string for prompt injection, or empty string if none
        """
        conclusions = self.get_program_conclusions(project_id, max_conclusions)

        if not conclusions:
            return ""

        lines = [
            "=" * 50,
            "PRIOR PROGRAM CONCLUSIONS",
            "=" * 50,
            "",
            "The following conclusions have been established in previous research:",
            ""
        ]

        for i, c in enumerate(conclusions, 1):
            lines.append(f"**Prior Finding {i}:**")
            lines.append(f"- Question: {c.get('question', 'N/A')[:100]}")
            lines.append(f"- Conclusion: {c.get('conclusion', 'N/A')[:300]}")
            if c.get('citations'):
                lines.append(f"- Supporting PMIDs: {', '.join(c['citations'][:5])}")
            lines.append("")

        lines.append("Please consider these prior conclusions in your analysis.")
        return "\n".join(lines)

    def delete_conclusion(self, conclusion_id: str) -> bool:
        """Delete a specific conclusion by ID."""
        data = self._load_data()

        if "_program_conclusions" not in data:
            return False

        original_length = len(data["_program_conclusions"])
        data["_program_conclusions"] = [
            c for c in data["_program_conclusions"]
            if c.get("id") != conclusion_id
        ]

        if len(data["_program_conclusions"]) < original_length:
            self._save_data(data)
            return True

        return False

    # =========================================================================
    # RELEVANCE SCORING FOR EXPERT SELECTION
    # =========================================================================

    def score_persona_knowledge_relevance(
        self,
        query: str,
        entities: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Score personas by how much relevant knowledge they have for a query.

        Used to boost expert selection based on accumulated knowledge.

        Args:
            query: The research question
            entities: Optional list of extracted entities (compounds, targets, etc.)

        Returns:
            Dict mapping persona names to relevance scores (0.0 to 1.0)
        """
        data = self._load_data()
        query_lower = query.lower()
        entities_lower = [e.lower() for e in (entities or [])]

        # Extract query terms
        stop_words = {
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are',
            'was', 'were', 'what', 'how', 'why', 'which', 'when', 'where'
        }
        query_words = set(query_lower.split()) - stop_words

        scores: Dict[str, float] = {}

        for persona, entries in data.items():
            if persona.startswith("_"):
                continue

            persona_score = 0.0
            total_facts = 0

            for entry in entries:
                facts = entry.get("facts", [])
                for fact in facts:
                    fact_lower = fact.lower()
                    total_facts += 1

                    # Entity matches (high weight)
                    for entity in entities_lower:
                        if entity in fact_lower:
                            persona_score += 2.0

                    # Query word matches (lower weight)
                    word_matches = sum(
                        1 for word in query_words
                        if len(word) > 3 and word in fact_lower
                    )
                    persona_score += word_matches * 0.5

            # Normalize score
            if total_facts > 0:
                normalized_score = min(1.0, persona_score / (total_facts * 0.5 + 5))
                scores[persona] = round(normalized_score, 3)

        return scores

    def get_relevant_knowledge_for_query(
        self,
        query: str,
        entities: Optional[List[str]] = None,
        max_facts_per_persona: int = 5
    ) -> Dict[str, List[str]]:
        """
        Get relevant knowledge facts for a query, organized by persona.

        Args:
            query: The research question
            entities: Optional list of extracted entities
            max_facts_per_persona: Maximum facts to return per persona

        Returns:
            Dict mapping persona names to lists of relevant facts
        """
        data = self._load_data()
        query_lower = query.lower()
        entities_lower = [e.lower() for e in (entities or [])]

        relevant: Dict[str, List[str]] = {}

        for persona, entries in data.items():
            if persona.startswith("_"):
                continue

            persona_facts = []
            for entry in entries:
                for fact in entry.get("facts", []):
                    fact_lower = fact.lower()

                    is_relevant = False

                    # Entity match
                    for entity in entities_lower:
                        if entity in fact_lower:
                            is_relevant = True
                            break

                    # Query word match (need 2+ matches)
                    if not is_relevant:
                        query_words = query_lower.split()
                        matches = sum(
                            1 for w in query_words
                            if len(w) > 3 and w in fact_lower
                        )
                        if matches >= 2:
                            is_relevant = True

                    if is_relevant:
                        persona_facts.append(fact)

            if persona_facts:
                relevant[persona] = persona_facts[:max_facts_per_persona]

        return relevant

    def format_knowledge_for_prompt(
        self,
        query: str,
        persona: str,
        entities: Optional[List[str]] = None,
        max_facts: int = 5
    ) -> str:
        """
        Format relevant knowledge for injection into expert prompts.

        Args:
            query: The research question
            persona: The expert persona
            entities: Optional entities to search for
            max_facts: Maximum facts to include

        Returns:
            Formatted string for prompt injection, or empty string if no relevant knowledge
        """
        # Get persona-specific facts
        relevant = self.get_relevant_knowledge_for_query(query, entities, max_facts)
        persona_facts = relevant.get(persona, [])

        # Get relevant triples
        triple_context = ""
        if entities:
            for entity in entities[:3]:  # Limit to 3 entities
                entity_triples = self.format_triples_for_context(entity, max_triples=3)
                if entity_triples:
                    triple_context += entity_triples + "\n"

        if not persona_facts and not triple_context:
            return ""

        lines = ["=" * 50, "PRIOR KNOWLEDGE (from previous sessions)", "=" * 50, ""]

        if persona_facts:
            lines.append("**Relevant Facts:**")
            for fact in persona_facts:
                lines.append(f"- {fact}")
            lines.append("")

        if triple_context:
            lines.append("**Known Relationships:**")
            lines.append(triple_context)

        return "\n".join(lines)


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

_default_store: Optional[KnowledgeStore] = None


def get_default_store() -> KnowledgeStore:
    """Get or create the default knowledge store instance."""
    global _default_store
    if _default_store is None:
        _default_store = KnowledgeStore()
    return _default_store


def add_knowledge(
    persona: str,
    source: str,
    facts: List[str],
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Add knowledge to the default store."""
    return get_default_store().add_knowledge(persona, source, facts, metadata)


def get_knowledge(persona: str) -> List[Dict[str, Any]]:
    """Get knowledge from the default store."""
    return get_default_store().get_knowledge(persona)


def get_all_knowledge() -> Dict[str, List[Dict[str, Any]]]:
    """Get all knowledge from the default store."""
    return get_default_store().get_all_knowledge()


def get_knowledge_summary() -> Dict[str, int]:
    """Get knowledge summary from the default store."""
    return get_default_store().get_knowledge_summary()


def delete_knowledge(persona: str, entry_id: str) -> bool:
    """Delete knowledge from the default store."""
    return get_default_store().delete_knowledge(persona, entry_id)


def search_knowledge(query: str, persona: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search knowledge in the default store."""
    return get_default_store().search_knowledge(query, persona)


def add_triple(
    subject: str,
    predicate: str,
    object_val: str,
    source: str,
    context: Optional[str] = None,
    confidence: float = 1.0
) -> str:
    """Add a triple to the default store."""
    return get_default_store().add_triple(
        subject, predicate, object_val, source, context, confidence
    )


def get_all_triples() -> List[Dict[str, Any]]:
    """Get all triples from the default store."""
    return get_default_store().get_all_triples()


def query_triples(
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object_val: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Query triples from the default store."""
    return get_default_store().query_triples(subject, predicate, object_val)


def format_triples_for_context(entity: str, max_triples: int = 10) -> str:
    """Format relevant triples as context for LLM prompts."""
    return get_default_store().format_triples_for_context(entity, max_triples)


# =============================================================================
# LEARNING CONVENIENCE FUNCTIONS
# =============================================================================

def add_user_correction(
    persona: str,
    original_response: str,
    corrected_response: str,
    question: str,
    project_id: Optional[int] = None
) -> str:
    """Record a user correction to the default store."""
    return get_default_store().add_user_correction(
        persona, original_response, corrected_response, question, project_id
    )


def add_query_effectiveness(
    query: str,
    question_type: str,
    paper_count: int,
    selected_count: int,
    project_id: Optional[int] = None
) -> str:
    """Track query effectiveness in the default store."""
    return get_default_store().add_query_effectiveness(
        query, question_type, paper_count, selected_count, project_id
    )


def get_project_learning_summary(project_id: int) -> Dict[str, Any]:
    """Get learning summary for a project from the default store."""
    return get_default_store().get_project_learning_summary(project_id)


def format_learning_for_prompt(
    persona: str,
    question_type: Optional[str] = None,
    project_id: Optional[int] = None
) -> str:
    """Format learned insights for injection into expert prompts."""
    return get_default_store().format_learning_for_prompt(persona, question_type, project_id)


# =============================================================================
# PROGRAM CONCLUSIONS CONVENIENCE FUNCTIONS
# =============================================================================

def add_program_conclusion(
    project_id: int,
    question: str,
    conclusion: str,
    citations: Optional[List[str]] = None,
    persona: str = "Consensus"
) -> str:
    """Save a program conclusion to the default store."""
    return get_default_store().add_program_conclusion(
        project_id, question, conclusion, citations, persona
    )


def get_program_conclusions(project_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    """Get program conclusions from the default store."""
    return get_default_store().get_program_conclusions(project_id, limit)


def format_conclusions_for_prompt(project_id: int, max_conclusions: int = 3) -> str:
    """Format prior conclusions for prompt injection."""
    return get_default_store().format_conclusions_for_prompt(project_id, max_conclusions)

