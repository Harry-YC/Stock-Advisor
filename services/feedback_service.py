"""
Feedback Service - Mark Pen Feature for RAG Improvement

Allows users to mark/highlight important text passages in expert responses
and search results. This feedback is used to:
1. Boost relevance of similar content in future searches
2. Train better prompts for expert responses
3. Identify high-value evidence patterns

Usage:
    from services.feedback_service import FeedbackService, save_mark, get_similar_marks

    # Save a user mark
    save_mark(
        text="Mortality rate was 15% at 30 days",
        source_type="expert_response",
        source_id="Surgical Oncologist",
        mark_type="important_data",
        question_context="surgical bypass vs stenting",
        project_id="project_123"
    )

    # Get marks to boost search relevance
    marks = get_similar_marks(query="mortality outcomes", limit=10)
"""

import json
import logging
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class FeedbackMark:
    """A single user-marked text passage."""
    id: str
    text: str
    source_type: str  # expert_response, search_result, recommendation
    source_id: str    # Expert name, PMID, etc.
    mark_type: str    # important_data, key_finding, evidence_gap, disagree, agree
    question_context: str
    project_id: Optional[str] = None
    user_note: Optional[str] = None  # User's annotation/feedback
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'text': self.text,
            'source_type': self.source_type,
            'source_id': self.source_id,
            'mark_type': self.mark_type,
            'question_context': self.question_context,
            'project_id': self.project_id,
            'user_note': self.user_note,
            'created_at': self.created_at,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FeedbackMark':
        return cls(
            id=data.get('id', ''),
            text=data.get('text', ''),
            source_type=data.get('source_type', ''),
            source_id=data.get('source_id', ''),
            mark_type=data.get('mark_type', ''),
            question_context=data.get('question_context', ''),
            project_id=data.get('project_id'),
            user_note=data.get('user_note'),
            created_at=data.get('created_at', datetime.now().isoformat()),
            metadata=data.get('metadata', {}),
        )


class FeedbackService:
    """
    Service for storing and retrieving user feedback marks.

    Uses JSON file storage for simplicity. Can be upgraded to SQLite
    or vector store for semantic similarity search.
    """

    MARK_TYPES = {
        'important_data': {'icon': '📊', 'color': '#10B981', 'boost': 0.15},
        'key_finding': {'icon': '⭐', 'color': '#F59E0B', 'boost': 0.12},
        'evidence_gap': {'icon': '🔍', 'color': '#6366F1', 'boost': 0.05},
        'disagree': {'icon': '❌', 'color': '#EF4444', 'boost': -0.10},
        'agree': {'icon': '✓', 'color': '#10B981', 'boost': 0.08},
        'citation_useful': {'icon': '📚', 'color': '#8B5CF6', 'boost': 0.10},
    }

    def __init__(self, storage_path: Path = None):
        """Initialize the feedback service."""
        self.storage_path = storage_path or (settings.OUTPUTS_DIR / "feedback_marks.json")
        self._ensure_storage()

    def _ensure_storage(self):
        """Ensure storage file exists."""
        if not self.storage_path.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_marks([])

    def _load_marks(self) -> List[Dict]:
        """Load all marks from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_marks(self, marks: List[Dict]):
        """Save marks to storage with atomic write."""
        temp_path = self.storage_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(marks, f, indent=2)
        temp_path.replace(self.storage_path)

    def save_mark(
        self,
        text: str,
        source_type: str,
        source_id: str,
        mark_type: str,
        question_context: str,
        project_id: Optional[str] = None,
        user_note: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> FeedbackMark:
        """
        Save a new feedback mark.

        Args:
            text: The highlighted text
            source_type: Type of source (expert_response, search_result, recommendation)
            source_id: Identifier (expert name, PMID, etc.)
            mark_type: Type of mark (important_data, key_finding, evidence_gap, etc.)
            question_context: The question being researched
            project_id: Optional project ID
            user_note: Optional user annotation/feedback
            metadata: Optional additional metadata

        Returns:
            The created FeedbackMark
        """
        # Generate deterministic ID based on content
        content_hash = hashlib.md5(f"{text}{source_id}{question_context}".encode()).hexdigest()[:12]
        mark_id = f"mark_{content_hash}"

        mark = FeedbackMark(
            id=mark_id,
            text=text,
            source_type=source_type,
            source_id=source_id,
            mark_type=mark_type,
            question_context=question_context,
            project_id=project_id,
            user_note=user_note,
            metadata=metadata or {},
        )

        # Load, append, save
        marks = self._load_marks()

        # Check for duplicate
        existing_ids = {m.get('id') for m in marks}
        if mark_id not in existing_ids:
            marks.append(mark.to_dict())
            self._save_marks(marks)
            logger.info(f"Saved feedback mark: {mark_id}")
        else:
            logger.info(f"Mark already exists: {mark_id}")

        return mark

    def get_marks(
        self,
        project_id: Optional[str] = None,
        source_type: Optional[str] = None,
        mark_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[FeedbackMark]:
        """
        Retrieve marks with optional filtering.

        Args:
            project_id: Filter by project
            source_type: Filter by source type
            mark_type: Filter by mark type
            limit: Maximum number of marks to return

        Returns:
            List of FeedbackMark objects
        """
        marks = self._load_marks()

        # Apply filters
        if project_id:
            marks = [m for m in marks if m.get('project_id') == project_id]
        if source_type:
            marks = [m for m in marks if m.get('source_type') == source_type]
        if mark_type:
            marks = [m for m in marks if m.get('mark_type') == mark_type]

        # Sort by date (newest first)
        marks.sort(key=lambda m: m.get('created_at', ''), reverse=True)

        return [FeedbackMark.from_dict(m) for m in marks[:limit]]

    def get_relevance_boost(self, text: str, query: str) -> float:
        """
        Calculate relevance boost based on similar marks.

        Args:
            text: Text to check for similarity
            query: Current search query

        Returns:
            Relevance boost value (can be negative for disagreement marks)
        """
        marks = self._load_marks()

        # Simple keyword overlap scoring
        text_lower = text.lower()
        query_words = set(query.lower().split())

        total_boost = 0.0
        match_count = 0

        for mark in marks:
            mark_text = mark.get('text', '').lower()
            mark_type = mark.get('mark_type', '')

            # Check for overlap
            mark_words = set(mark_text.split())
            overlap = len(mark_words & query_words) / max(len(query_words), 1)

            # Also check if marked text appears in current text
            text_match = mark_text in text_lower or any(
                word in text_lower for word in list(mark_words)[:5]
            )

            if overlap > 0.3 or text_match:
                boost_config = self.MARK_TYPES.get(mark_type, {})
                boost = boost_config.get('boost', 0.05)
                total_boost += boost
                match_count += 1

        # Normalize by match count (diminishing returns)
        if match_count > 0:
            return min(total_boost / (1 + match_count * 0.1), 0.3)  # Cap at 0.3

        return 0.0

    def get_mark_stats(self, project_id: Optional[str] = None) -> Dict:
        """Get statistics about marks."""
        marks = self.get_marks(project_id=project_id, limit=1000)

        stats = {
            'total': len(marks),
            'by_type': {},
            'by_source': {},
        }

        for mark in marks:
            # By type
            mt = mark.mark_type
            stats['by_type'][mt] = stats['by_type'].get(mt, 0) + 1

            # By source
            st = mark.source_type
            stats['by_source'][st] = stats['by_source'].get(st, 0) + 1

        return stats

    def delete_mark(self, mark_id: str) -> bool:
        """Delete a mark by ID."""
        marks = self._load_marks()
        original_count = len(marks)
        marks = [m for m in marks if m.get('id') != mark_id]

        if len(marks) < original_count:
            self._save_marks(marks)
            logger.info(f"Deleted mark: {mark_id}")
            return True

        return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_service = None

def _get_service() -> FeedbackService:
    """Get singleton service instance."""
    global _service
    if _service is None:
        _service = FeedbackService()
    return _service


def save_mark(
    text: str,
    source_type: str,
    source_id: str,
    mark_type: str,
    question_context: str,
    project_id: Optional[str] = None,
    user_note: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> FeedbackMark:
    """Convenience function to save a mark."""
    return _get_service().save_mark(
        text=text,
        source_type=source_type,
        source_id=source_id,
        mark_type=mark_type,
        question_context=question_context,
        project_id=project_id,
        user_note=user_note,
        metadata=metadata,
    )


def get_marks(
    project_id: Optional[str] = None,
    source_type: Optional[str] = None,
    mark_type: Optional[str] = None,
    limit: int = 100,
) -> List[FeedbackMark]:
    """Convenience function to get marks."""
    return _get_service().get_marks(
        project_id=project_id,
        source_type=source_type,
        mark_type=mark_type,
        limit=limit,
    )


def get_relevance_boost(text: str, query: str) -> float:
    """Convenience function to get relevance boost."""
    return _get_service().get_relevance_boost(text, query)
