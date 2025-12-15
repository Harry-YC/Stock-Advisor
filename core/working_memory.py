"""
Persistent Working Memory for Expert Panel

Stores constraints and facts that must be remembered across ALL rounds.
Injected into every expert's system prompt.

Features:
- File-based persistence (survives browser refresh)
- Per-project isolation
- Atomic writes (temp file + rename)
- Clear/reset functionality
"""

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any


@dataclass
class MemoryEntry:
    """Single entry in working memory."""
    content: str
    category: str  # "constraint", "fact", "correction"
    source: str  # "human", "expert:Name"
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(**data)


class WorkingMemory:
    """
    Session-persistent memory that survives across rounds and browser refreshes.

    Usage:
        memory = WorkingMemory(project_id="my_project")
        memory.add("This drug is IV only, not oral", category="constraint")
        memory.add("Half-life is 8.2 hours in rat", category="fact")

        # Inject into prompts
        prompt += memory.format_for_prompt()

        # Clear when done
        memory.clear()
    """

    MEMORY_DIR = "outputs/working_memory"

    def __init__(self, project_id: str = "default"):
        """
        Initialize working memory for a project.

        Args:
            project_id: Project identifier for isolation
        """
        self.project_id = project_id
        self.entries: List[MemoryEntry] = []
        self._ensure_dir()
        self._load()

    @property
    def _file_path(self) -> str:
        """Get file path for this project's memory."""
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.project_id)
        return os.path.join(self.MEMORY_DIR, f"{safe_id}_memory.json")

    def _ensure_dir(self):
        """Ensure memory directory exists."""
        os.makedirs(self.MEMORY_DIR, exist_ok=True)

    def add(
        self,
        content: str,
        category: str = "constraint",
        source: str = "human"
    ) -> int:
        """
        Add an entry to working memory.

        Args:
            content: The constraint, fact, or correction text
            category: One of "constraint", "fact", "correction"
            source: Who added this ("human" or "expert:Name")

        Returns:
            Index of the added entry
        """
        if category not in ("constraint", "fact", "correction"):
            category = "constraint"

        entry = MemoryEntry(
            content=content.strip(),
            category=category,
            source=source
        )
        self.entries.append(entry)
        self._save()
        return len(self.entries) - 1

    def remove(self, index: int) -> bool:
        """
        Remove an entry by index.

        Args:
            index: Entry index to remove

        Returns:
            True if removed, False if index invalid
        """
        if 0 <= index < len(self.entries):
            self.entries.pop(index)
            self._save()
            return True
        return False

    def clear(self):
        """Clear all entries and save."""
        self.entries = []
        self._save()

    def get_entries_by_category(self, category: str) -> List[MemoryEntry]:
        """Get all entries of a specific category."""
        return [e for e in self.entries if e.category == category]

    def format_for_prompt(self) -> str:
        """
        Format working memory for injection into expert prompts.

        Returns:
            Formatted string to append to system prompt, or empty string if no entries
        """
        if not self.entries:
            return ""

        lines = [
            "",
            "=" * 60,
            "WORKING MEMORY (must respect in ALL responses):",
            "=" * 60
        ]

        # Group by category
        by_category: Dict[str, List[MemoryEntry]] = {}
        for entry in self.entries:
            by_category.setdefault(entry.category, []).append(entry)

        # Format constraints first (highest priority)
        if "constraint" in by_category:
            lines.append("\n**CONSTRAINTS (you MUST follow these):**")
            for entry in by_category["constraint"]:
                lines.append(f"- {entry.content}")

        # Format established facts
        if "fact" in by_category:
            lines.append("\n**ESTABLISHED FACTS (treat as ground truth):**")
            for entry in by_category["fact"]:
                lines.append(f"- {entry.content}")

        # Format corrections (errors to avoid)
        if "correction" in by_category:
            lines.append("\n**CORRECTIONS (do NOT repeat these errors):**")
            for entry in by_category["correction"]:
                lines.append(f"- {entry.content}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "project_id": self.project_id,
            "entries": [e.to_dict() for e in self.entries],
            "updated_at": datetime.now().isoformat()
        }

    def _load(self):
        """Load memory from file if exists."""
        if not os.path.exists(self._file_path):
            self.entries = []
            return

        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.entries = [
                MemoryEntry.from_dict(e)
                for e in data.get("entries", [])
            ]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Could not load working memory: {e}")
            self.entries = []

    def _save(self):
        """Save memory to file using atomic write pattern."""
        self._ensure_dir()

        data = self.to_dict()

        # Write to temp file first, then rename (atomic on most systems)
        try:
            fd, temp_path = tempfile.mkstemp(
                dir=self.MEMORY_DIR,
                suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic rename
            os.replace(temp_path, self._file_path)

        except Exception as e:
            print(f"Warning: Could not save working memory: {e}")
            # Clean up temp file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return len(self.entries) > 0

    def __repr__(self) -> str:
        return f"WorkingMemory(project_id='{self.project_id}', entries={len(self.entries)})"


# Convenience function for getting project-specific memory
_memory_cache: Dict[str, WorkingMemory] = {}


def get_working_memory(project_id: str = "default") -> WorkingMemory:
    """
    Get or create working memory for a project.

    Uses a simple cache to avoid reloading from disk on every call.

    Args:
        project_id: Project identifier

    Returns:
        WorkingMemory instance for the project
    """
    if project_id not in _memory_cache:
        _memory_cache[project_id] = WorkingMemory(project_id)
    return _memory_cache[project_id]


def clear_memory_cache():
    """Clear the memory cache (useful for testing)."""
    global _memory_cache
    _memory_cache = {}
