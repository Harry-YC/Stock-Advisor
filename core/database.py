"""
SQLite Database Layer for Literature Review Platform

Manages all structured data including:
- Projects and citations
- Screening decisions and dual review tracking
- Search history and PRISMA flow data
- Tags and metadata

Architecture: File-based SQLite database with Dropbox sync support
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, asdict

# Set up logger for this module
logger = logging.getLogger("literature_review.database")

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Project:
    id: Optional[int]
    name: str
    description: str
    created_at: str
    updated_at: str

@dataclass
class Citation:
    pmid: str
    title: str
    authors: List[str]  # Will be stored as JSON string
    journal: str
    year: int
    abstract: str
    doi: Optional[str]
    publication_types: Optional[str]
    keywords: Optional[str]
    fetched_at: str

@dataclass
class ScreeningDecision:
    id: Optional[int]
    project_id: int
    pmid: str
    reviewer: str  # "user1" or "user2"
    decision: str  # "include", "exclude", "maybe", "conflict"
    stage: str  # "title", "abstract", "fulltext"
    reason: Optional[str]
    notes: Optional[str]
    decided_at: str

@dataclass
class SearchHistory:
    id: Optional[int]
    project_id: int
    query: str
    filters: str  # JSON string
    total_results: int
    retrieved_count: int
    executed_at: str

# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """Manages SQLite connection and schema"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise  # Preserves full traceback
        finally:
            conn.close()

    def _initialize_schema(self):
        """Create all tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Projects table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Citations table (shared across projects)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS citations (
                    pmid TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors TEXT,  -- JSON array
                    journal TEXT,
                    year INTEGER,
                    abstract TEXT,
                    doi TEXT,
                    publication_types TEXT,
                    keywords TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Project-Citation many-to-many relationship
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS project_citations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    pmid TEXT NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    FOREIGN KEY (pmid) REFERENCES citations(pmid) ON DELETE CASCADE,
                    UNIQUE(project_id, pmid)
                )
            """)

            # Screening decisions (dual review support)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS screening_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    pmid TEXT NOT NULL,
                    reviewer TEXT NOT NULL,  -- "user1" or "user2"
                    decision TEXT NOT NULL CHECK(decision IN ('include', 'exclude', 'maybe', 'conflict')),
                    stage TEXT NOT NULL CHECK(stage IN ('title', 'abstract', 'fulltext')),
                    reason TEXT,
                    notes TEXT,
                    decided_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    FOREIGN KEY (pmid) REFERENCES citations(pmid) ON DELETE CASCADE,
                    UNIQUE(project_id, pmid, reviewer, stage)
                )
            """)

            # Search history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    query TEXT NOT NULL,
                    filters TEXT,  -- JSON string
                    total_results INTEGER,
                    retrieved_count INTEGER,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """)

            # Tags for papers
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    color TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Paper-Tag many-to-many relationship
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS paper_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    pmid TEXT NOT NULL,
                    tag_id INTEGER NOT NULL,
                    tagged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    FOREIGN KEY (pmid) REFERENCES citations(pmid) ON DELETE CASCADE,
                    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE,
                    UNIQUE(project_id, pmid, tag_id)
                )
            """)

            # Query cache (for AI-optimized queries)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT NOT NULL UNIQUE,
                    original_query TEXT NOT NULL,
                    optimized_query TEXT NOT NULL,
                    query_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # AI Screening results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_screening (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    pmid TEXT NOT NULL,
                    decision TEXT NOT NULL CHECK(decision IN ('include', 'exclude', 'review')),
                    confidence INTEGER,
                    reasoning TEXT,
                    screened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    FOREIGN KEY (pmid) REFERENCES citations(pmid) ON DELETE CASCADE,
                    UNIQUE(project_id, pmid)
                )
            """)

            # Expert discussions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS expert_discussions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    clinical_question TEXT NOT NULL,
                    scenario TEXT,
                    selected_experts TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """)

            # Discussion entries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS discussion_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    discussion_id INTEGER NOT NULL,
                    round INTEGER NOT NULL,
                    expert_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    raw_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (discussion_id) REFERENCES expert_discussions(id) ON DELETE CASCADE
                )
            """)

            # Documents table (for uploaded/pasted content)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    project_id INTEGER NOT NULL,
                    source_type TEXT NOT NULL,  -- 'pdf', 'docx', 'text', 'url', 'pubmed', 'preprint'
                    title TEXT NOT NULL,
                    content TEXT,
                    authors TEXT,  -- JSON array
                    year TEXT,
                    journal TEXT,
                    doi TEXT,
                    pmid TEXT,
                    metadata TEXT,  -- JSON object
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """)

            # Chat conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_conversations (
                    id TEXT PRIMARY KEY,
                    project_id INTEGER NOT NULL,
                    mode TEXT NOT NULL,  -- 'single_expert' or 'panel_router'
                    expert_name TEXT,  -- For single expert mode
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """)
            
            # CDP Data table (JSON store per project)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS project_cdp (
                    project_id INTEGER PRIMARY KEY,
                    cdp_json TEXT NOT NULL,  -- JSON string of cdp_sections
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """)

            # Search context table (for persisting search metadata and selections)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    search_id INTEGER NOT NULL,
                    ranking_mode TEXT,
                    ranking_weights TEXT,  -- JSON: {relevance, evidence, recency, influence}
                    query_explanation TEXT,
                    query_confidence TEXT,
                    query_type TEXT,
                    selected_pmids TEXT,  -- JSON array of user-selected PMIDs
                    is_active INTEGER DEFAULT 1,  -- Most recent search for project
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    FOREIGN KEY (search_id) REFERENCES search_history(id) ON DELETE CASCADE
                )
            """)

            # Paper signals table (for learning from user selections)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS paper_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    pmid TEXT NOT NULL,
                    signal_type TEXT NOT NULL CHECK(signal_type IN ('selected', 'cited', 'tagged', 'rejected')),
                    query_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    UNIQUE(project_id, pmid, signal_type)
                )
            """)

            # Expert corrections table (for learning from user feedback)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS expert_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    expert_name TEXT NOT NULL,
                    question_snippet TEXT,
                    critique TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """)

            # Program profile table (for auto-detected program context)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS program_profile (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER UNIQUE NOT NULL,
                    target TEXT,
                    indication TEXT,
                    drug_names TEXT,
                    competitors TEXT,
                    mechanism TEXT,
                    therapeutic_area TEXT,
                    development_stage TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """)

            # Trusted knowledge table (auto-fetched from sources)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trusted_knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    insight TEXT,
                    confidence REAL DEFAULT 0.5,
                    entry_type TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    UNIQUE(project_id, source, source_id)
                )
            """)

            # Chat messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,  -- 'user', 'assistant', 'system'
                    expert_name TEXT,  -- Which expert responded (for panel mode)
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES chat_conversations(id) ON DELETE CASCADE
                )
            """)

            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_citations_year ON citations(year)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_citations_journal ON citations(journal)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screening_project ON screening_decisions(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screening_decision ON screening_decisions(decision)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_project_citations_project ON project_citations(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_history_project ON search_history(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_screening_project ON ai_screening(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_expert_discussions_project ON expert_discussions(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_conversations_project ON chat_conversations(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation ON chat_messages(conversation_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_context_project ON search_context(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_context_active ON search_context(project_id, is_active)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_signals_project ON paper_signals(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_signals_pmid ON paper_signals(project_id, pmid)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_expert_corrections_expert ON expert_corrections(expert_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_expert_corrections_project ON expert_corrections(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_program_profile_project ON program_profile(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trusted_knowledge_project ON trusted_knowledge(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trusted_knowledge_confidence ON trusted_knowledge(project_id, confidence)")

# =============================================================================
# DATA ACCESS OBJECTS (DAOs)
# =============================================================================

class ProjectDAO:
    """Manages project CRUD operations"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_project(self, name: str, description: str = "") -> int:
        """Create a new project and return its ID"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO projects (name, description) VALUES (?, ?)",
                (name, description)
            )
            return cursor.lastrowid

    def get_all_projects(self) -> List[Project]:
        """Get all projects"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            return [Project(**dict(row)) for row in rows]

    def get_project(self, project_id: int) -> Optional[Project]:
        """Get project by ID"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            row = cursor.fetchone()
            return Project(**dict(row)) if row else None

    def get_project_by_name(self, name: str) -> Optional[Project]:
        """Get project by name"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects WHERE name = ?", (name,))
            row = cursor.fetchone()
            return Project(**dict(row)) if row else None

    def update_project(self, project_id: int, name: str = None, description: str = None):
        """Update project details"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if name:
                cursor.execute(
                    "UPDATE projects SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (name, project_id)
                )
            if description is not None:
                cursor.execute(
                    "UPDATE projects SET description = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (description, project_id)
                )

    def delete_project(self, project_id: int):
        """Delete project (cascades to related data)"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))


class CitationDAO:
    """Manages citation CRUD operations"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def upsert_citation(self, citation: Dict):
        """Insert or update a citation"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Convert list fields to JSON strings
            authors_json = json.dumps(citation.get('authors', []))
            pub_types_json = json.dumps(citation.get('publication_types') or [])
            keywords_json = json.dumps(citation.get('keywords') or [])

            cursor.execute("""
                INSERT INTO citations (pmid, title, authors, journal, year, abstract, doi, publication_types, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pmid) DO UPDATE SET
                    title = excluded.title,
                    authors = excluded.authors,
                    journal = excluded.journal,
                    year = excluded.year,
                    abstract = excluded.abstract,
                    doi = excluded.doi,
                    publication_types = excluded.publication_types,
                    keywords = excluded.keywords,
                    fetched_at = CURRENT_TIMESTAMP
            """, (
                citation.get('pmid'),
                citation.get('title'),
                authors_json,
                citation.get('journal'),
                citation.get('year'),
                citation.get('abstract'),
                citation.get('doi'),
                pub_types_json,
                keywords_json
            ))

    def add_citation_to_project(self, project_id: int, pmid: str):
        """Link a citation to a project"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO project_citations (project_id, pmid) VALUES (?, ?)",
                    (project_id, pmid)
                )
            except sqlite3.IntegrityError:
                # Citation already in project - this is expected during re-imports
                logger.debug(f"Citation PMID {pmid} already linked to project {project_id}")

    def get_citations_by_project(self, project_id: int) -> List[Dict]:
        """Get all citations for a project"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.* FROM citations c
                JOIN project_citations pc ON c.pmid = pc.pmid
                WHERE pc.project_id = ?
                ORDER BY c.year DESC, c.title
            """, (project_id,))

            citations = []
            for row in cursor.fetchall():
                citation = dict(row)
                # Parse authors JSON back to list
                citation['authors'] = json.loads(citation['authors']) if citation['authors'] else []
                citations.append(citation)

            return citations

    def get_citations_by_project_paginated(
        self, project_id: int, limit: int = 100, offset: int = 0
    ) -> Tuple[List[Dict], int]:
        """
        Get citations for a project with pagination.

        Args:
            project_id: Project ID
            limit: Maximum citations to return (default: 100)
            offset: Number of citations to skip (default: 0)

        Returns:
            Tuple of (citations_list, total_count)
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get total count first
            cursor.execute("""
                SELECT COUNT(*) as count FROM project_citations
                WHERE project_id = ?
            """, (project_id,))
            total_count = cursor.fetchone()['count']

            # Get paginated results
            cursor.execute("""
                SELECT c.* FROM citations c
                JOIN project_citations pc ON c.pmid = pc.pmid
                WHERE pc.project_id = ?
                ORDER BY c.year DESC, c.title
                LIMIT ? OFFSET ?
            """, (project_id, limit, offset))

            citations = []
            for row in cursor.fetchall():
                citation = dict(row)
                citation['authors'] = json.loads(citation['authors']) if citation['authors'] else []
                citations.append(citation)

            return citations, total_count

    def get_citation(self, pmid: str) -> Optional[Dict]:
        """Get a single citation by PMID"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM citations WHERE pmid = ?", (pmid,))
            row = cursor.fetchone()
            if row:
                citation = dict(row)
                citation['authors'] = json.loads(citation['authors']) if citation['authors'] else []
                return citation
            return None

    def get_citations_batch(self, pmids: List[str]) -> Dict[str, Dict]:
        """
        Get multiple citations by PMIDs (for caching)

        Args:
            pmids: List of PMID strings (must be numeric)

        Returns:
            Dict mapping PMID -> citation dict for found citations only

        Raises:
            ValueError: If any PMID is not a valid numeric string
        """
        if not pmids:
            return {}

        # Validate PMIDs are numeric strings to prevent SQL issues
        validated_pmids = []
        for pmid in pmids:
            if not isinstance(pmid, str):
                pmid = str(pmid)
            # PMIDs should be numeric (1 to 99999999)
            if not pmid.isdigit():
                raise ValueError(f"Invalid PMID format: {pmid}. PMIDs must be numeric.")
            validated_pmids.append(pmid)

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            # Use IN clause for efficient batch retrieval
            # Build query safely - placeholders are just '?' characters
            placeholders = ','.join(['?' for _ in validated_pmids])
            query = "SELECT * FROM citations WHERE pmid IN ({})".format(placeholders)
            cursor.execute(query, validated_pmids)

            citations_map = {}
            for row in cursor.fetchall():
                citation = dict(row)
                citation['authors'] = json.loads(citation['authors']) if citation['authors'] else []
                citations_map[citation['pmid']] = citation

            return citations_map

    def remove_citation_from_project(self, project_id: int, pmid: str):
        """Remove citation from project (doesn't delete citation itself)"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM project_citations WHERE project_id = ? AND pmid = ?",
                (project_id, pmid)
            )


class ScreeningDAO:
    """Manages screening decisions and dual review"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def add_decision(self, project_id: int, pmid: str, reviewer: str,
                     decision: str, stage: str, reason: str = None, notes: str = None):
        """Add or update a screening decision"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO screening_decisions (project_id, pmid, reviewer, decision, stage, reason, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, pmid, reviewer, stage) DO UPDATE SET
                    decision = excluded.decision,
                    reason = excluded.reason,
                    notes = excluded.notes,
                    decided_at = CURRENT_TIMESTAMP
            """, (project_id, pmid, reviewer, decision, stage, reason, notes))

    def get_decisions_by_project(self, project_id: int, stage: str = None) -> List[Dict]:
        """Get all screening decisions for a project"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if stage:
                cursor.execute(
                    "SELECT * FROM screening_decisions WHERE project_id = ? AND stage = ? ORDER BY decided_at DESC",
                    (project_id, stage)
                )
            else:
                cursor.execute(
                    "SELECT * FROM screening_decisions WHERE project_id = ? ORDER BY decided_at DESC",
                    (project_id,)
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_decision(self, project_id: int, pmid: str, reviewer: str, stage: str) -> Optional[Dict]:
        """Get a specific screening decision"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM screening_decisions
                WHERE project_id = ? AND pmid = ? AND reviewer = ? AND stage = ?
            """, (project_id, pmid, reviewer, stage))
            row = cursor.fetchone()
            return dict(row) if row else None

    def detect_conflicts(self, project_id: int, stage: str) -> List[Dict]:
        """Detect conflicts between two reviewers at a given stage"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    d1.pmid,
                    d1.decision as reviewer1_decision,
                    d2.decision as reviewer2_decision,
                    d1.decided_at as reviewer1_date,
                    d2.decided_at as reviewer2_date
                FROM screening_decisions d1
                JOIN screening_decisions d2
                    ON d1.project_id = d2.project_id
                    AND d1.pmid = d2.pmid
                    AND d1.stage = d2.stage
                WHERE d1.project_id = ?
                    AND d1.stage = ?
                    AND d1.reviewer < d2.reviewer  -- Avoid duplicates
                    AND d1.decision != d2.decision
            """, (project_id, stage))
            return [dict(row) for row in cursor.fetchall()]

    def get_prisma_counts(self, project_id: int) -> Dict[str, int]:
        """Get counts for PRISMA flow diagram"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Total citations in project
            cursor.execute(
                "SELECT COUNT(*) as count FROM project_citations WHERE project_id = ?",
                (project_id,)
            )
            total = cursor.fetchone()['count']

            # Title screening
            cursor.execute("""
                SELECT decision, COUNT(*) as count
                FROM screening_decisions
                WHERE project_id = ? AND stage = 'title'
                GROUP BY decision
            """, (project_id,))
            title_counts = {row['decision']: row['count'] for row in cursor.fetchall()}

            # Abstract screening
            cursor.execute("""
                SELECT decision, COUNT(*) as count
                FROM screening_decisions
                WHERE project_id = ? AND stage = 'abstract'
                GROUP BY decision
            """, (project_id,))
            abstract_counts = {row['decision']: row['count'] for row in cursor.fetchall()}

            # Full-text screening
            cursor.execute("""
                SELECT decision, COUNT(*) as count
                FROM screening_decisions
                WHERE project_id = ? AND stage = 'fulltext'
                GROUP BY decision
            """, (project_id,))
            fulltext_counts = {row['decision']: row['count'] for row in cursor.fetchall()}

            return {
                'total_citations': total,
                'title_include': title_counts.get('include', 0),
                'title_exclude': title_counts.get('exclude', 0),
                'abstract_include': abstract_counts.get('include', 0),
                'abstract_exclude': abstract_counts.get('exclude', 0),
                'fulltext_include': fulltext_counts.get('include', 0),
                'fulltext_exclude': fulltext_counts.get('exclude', 0),
            }


class SearchHistoryDAO:
    """Manages search history tracking"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def add_search(self, project_id: int, query: str, filters: Dict,
                   total_results: int, retrieved_count: int) -> int:
        """Record a search execution"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            filters_json = json.dumps(filters)
            cursor.execute("""
                INSERT INTO search_history (project_id, query, filters, total_results, retrieved_count)
                VALUES (?, ?, ?, ?, ?)
            """, (project_id, query, filters_json, total_results, retrieved_count))
            return cursor.lastrowid

    def get_search_history(self, project_id: int) -> List[Dict]:
        """Get all searches for a project"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM search_history
                WHERE project_id = ?
                ORDER BY executed_at DESC
            """, (project_id,))

            searches = []
            for row in cursor.fetchall():
                search = dict(row)
                search['filters'] = json.loads(search['filters']) if search['filters'] else {}
                searches.append(search)

            return searches


class QueryCacheDAO:
    """Manages AI-optimized query cache"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def get_cached_query(self, query_hash: str) -> Optional[Dict]:
        """Get cached optimized query by hash"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM query_cache WHERE query_hash = ?",
                (query_hash,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def save_query(self, query_hash: str, original_query: str,
                   optimized_query: str, query_type: str = None):
        """Save optimized query to cache"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO query_cache
                (query_hash, original_query, optimized_query, query_type)
                VALUES (?, ?, ?, ?)
            """, (query_hash, original_query, optimized_query, query_type))

    def clear_cache(self):
        """Clear all cached queries"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM query_cache")


class TagDAO:
    """Manages tags and paper tagging"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_tag(self, name: str, color: str = None) -> int:
        """Create a new tag"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO tags (name, color) VALUES (?, ?)", (name, color))
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Tag already exists, get its ID
                cursor.execute("SELECT id FROM tags WHERE name = ?", (name,))
                return cursor.fetchone()['id']

    def get_all_tags(self) -> List[Dict]:
        """Get all tags"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tags ORDER BY name")
            return [dict(row) for row in cursor.fetchall()]

    def tag_paper(self, project_id: int, pmid: str, tag_id: int):
        """Add a tag to a paper"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO paper_tags (project_id, pmid, tag_id) VALUES (?, ?, ?)",
                    (project_id, pmid, tag_id)
                )
            except sqlite3.IntegrityError:
                # Tag already applied to this paper - this is expected behavior
                logger.debug(f"Tag {tag_id} already applied to PMID {pmid} in project {project_id}")

    def untag_paper(self, project_id: int, pmid: str, tag_id: int):
        """Remove a tag from a paper"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM paper_tags WHERE project_id = ? AND pmid = ? AND tag_id = ?",
                (project_id, pmid, tag_id)
            )


class CdpDAO:
    """Manages Clinical Development Plan (CDP) persistence"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def save_cdp(self, project_id: int, cdp_sections: Dict):
        """Save CDP sections for a project"""
        cdp_json = json.dumps(cdp_sections)
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO project_cdp (project_id, cdp_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(project_id) DO UPDATE SET
                    cdp_json = excluded.cdp_json,
                    updated_at = CURRENT_TIMESTAMP
            """, (project_id, cdp_json))

    def get_cdp(self, project_id: int) -> Dict:
        """Get CDP sections for a project"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT cdp_json FROM project_cdp WHERE project_id = ?", (project_id,))
            row = cursor.fetchone()
            if row and row['cdp_json']:
                return json.loads(row['cdp_json'])
            return {}

    def get_paper_tags(self, project_id: int, pmid: str) -> List[Dict]:
        """Get all tags for a paper"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT t.* FROM tags t
                JOIN paper_tags pt ON t.id = pt.tag_id
                WHERE pt.project_id = ? AND pt.pmid = ?
            """, (project_id, pmid))
            return [dict(row) for row in cursor.fetchall()]


class AIScreeningDAO:
    """Manages AI-powered paper screening results"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def save_screening(self, project_id: int, pmid: str, decision: str,
                       confidence: int, reasoning: str):
        """Save or update an AI screening result"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ai_screening (project_id, pmid, decision, confidence, reasoning)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project_id, pmid) DO UPDATE SET
                    decision = excluded.decision,
                    confidence = excluded.confidence,
                    reasoning = excluded.reasoning,
                    screened_at = CURRENT_TIMESTAMP
            """, (project_id, pmid, decision, confidence, reasoning))

    def get_screening_results(self, project_id: int) -> List[Dict]:
        """Get all AI screening results for a project"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM ai_screening
                WHERE project_id = ?
                ORDER BY screened_at DESC
            """, (project_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_screening_by_pmid(self, project_id: int, pmid: str) -> Optional[Dict]:
        """Get AI screening result for a specific paper"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM ai_screening
                WHERE project_id = ? AND pmid = ?
            """, (project_id, pmid))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_screening_summary(self, project_id: int) -> Dict[str, int]:
        """Get summary counts of screening decisions"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT decision, COUNT(*) as count
                FROM ai_screening
                WHERE project_id = ?
                GROUP BY decision
            """, (project_id,))
            summary = {row['decision']: row['count'] for row in cursor.fetchall()}
            return {
                'include': summary.get('include', 0),
                'exclude': summary.get('exclude', 0),
                'review': summary.get('review', 0),
                'total': sum(summary.values())
            }

    def update_decision(self, project_id: int, pmid: str, decision: str):
        """Manually override an AI screening decision"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE ai_screening
                SET decision = ?, screened_at = CURRENT_TIMESTAMP
                WHERE project_id = ? AND pmid = ?
            """, (decision, project_id, pmid))


class ExpertDiscussionDAO:
    """Manages expert panel discussions"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_discussion(self, project_id: int, clinical_question: str,
                          scenario: str = None, experts: List[str] = None) -> int:
        """Create a new expert panel discussion"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            experts_json = json.dumps(experts) if experts else None
            cursor.execute("""
                INSERT INTO expert_discussions (project_id, clinical_question, scenario, selected_experts)
                VALUES (?, ?, ?, ?)
            """, (project_id, clinical_question, scenario, experts_json))
            return cursor.lastrowid

    def add_entry(self, discussion_id: int, round_num: int, expert_name: str,
                  content: str, raw_response: str = None):
        """Add an expert response entry to a discussion"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO discussion_entries (discussion_id, round, expert_name, content, raw_response)
                VALUES (?, ?, ?, ?, ?)
            """, (discussion_id, round_num, expert_name, content, raw_response))

    def get_discussion(self, discussion_id: int) -> Optional[Dict]:
        """Get a discussion by ID"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM expert_discussions WHERE id = ?", (discussion_id,))
            row = cursor.fetchone()
            if row:
                discussion = dict(row)
                discussion['selected_experts'] = json.loads(discussion['selected_experts']) if discussion['selected_experts'] else []
                return discussion
            return None

    def get_discussions_by_project(self, project_id: int) -> List[Dict]:
        """Get all discussions for a project"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM expert_discussions
                WHERE project_id = ?
                ORDER BY created_at DESC
            """, (project_id,))
            discussions = []
            for row in cursor.fetchall():
                discussion = dict(row)
                discussion['selected_experts'] = json.loads(discussion['selected_experts']) if discussion['selected_experts'] else []
                discussions.append(discussion)
            return discussions

    def get_entries(self, discussion_id: int) -> List[Dict]:
        """Get all entries for a discussion"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM discussion_entries
                WHERE discussion_id = ?
                ORDER BY round, created_at
            """, (discussion_id,))
            entries = []
            for row in cursor.fetchall():
                entry = dict(row)
                if entry['raw_response']:
                    try:
                        entry['raw_response'] = json.loads(entry['raw_response'])
                    except json.JSONDecodeError:
                        pass
                entries.append(entry)
            return entries

    def get_latest_discussion(self, project_id: int) -> Optional[Dict]:
        """Get the most recent discussion for a project"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM expert_discussions
                WHERE project_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (project_id,))
            row = cursor.fetchone()
            if row:
                discussion = dict(row)
                discussion['selected_experts'] = json.loads(discussion['selected_experts']) if discussion['selected_experts'] else []
                return discussion
            return None


class DocumentDAO:
    """Manages uploaded/ingested documents"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def add_document(self, project_id: int, doc_id: str, source_type: str,
                     title: str, content: str, authors: List[str] = None,
                     year: str = None, journal: str = None, doi: str = None,
                     pmid: str = None, metadata: Dict = None) -> str:
        """Add a document to the project"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents (id, project_id, source_type, title, content,
                                       authors, year, journal, doi, pmid, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    content = excluded.content,
                    authors = excluded.authors,
                    metadata = excluded.metadata
            """, (
                doc_id, project_id, source_type, title, content,
                json.dumps(authors) if authors else None,
                year, journal, doi, pmid,
                json.dumps(metadata) if metadata else None
            ))
            return doc_id

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get a document by ID"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            if row:
                doc = dict(row)
                doc['authors'] = json.loads(doc['authors']) if doc['authors'] else []
                doc['metadata'] = json.loads(doc['metadata']) if doc['metadata'] else {}
                return doc
            return None

    def get_documents_by_project(self, project_id: int) -> List[Dict]:
        """Get all documents for a project"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM documents
                WHERE project_id = ?
                ORDER BY created_at DESC
            """, (project_id,))
            documents = []
            for row in cursor.fetchall():
                doc = dict(row)
                doc['authors'] = json.loads(doc['authors']) if doc['authors'] else []
                doc['metadata'] = json.loads(doc['metadata']) if doc['metadata'] else {}
                documents.append(doc)
            return documents

    def delete_document(self, doc_id: str):
        """Delete a document"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def get_document_count(self, project_id: int) -> int:
        """Get count of documents in a project"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM documents WHERE project_id = ?",
                (project_id,)
            )
            return cursor.fetchone()[0]


class ChatDAO:
    """Manages expert chat conversations and messages"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_conversation(self, project_id: int, conversation_id: str,
                            mode: str, expert_name: str = None,
                            title: str = None) -> str:
        """Create a new chat conversation"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_conversations (id, project_id, mode, expert_name, title)
                VALUES (?, ?, ?, ?, ?)
            """, (conversation_id, project_id, mode, expert_name, title))
            return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get a conversation by ID"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chat_conversations WHERE id = ?",
                (conversation_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_conversations_by_project(self, project_id: int) -> List[Dict]:
        """Get all conversations for a project"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM chat_conversations
                WHERE project_id = ?
                ORDER BY updated_at DESC
            """, (project_id,))
            return [dict(row) for row in cursor.fetchall()]

    def add_message(self, conversation_id: str, role: str, content: str,
                    expert_name: str = None) -> int:
        """Add a message to a conversation"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_messages (conversation_id, role, content, expert_name)
                VALUES (?, ?, ?, ?)
            """, (conversation_id, role, content, expert_name))

            # Update conversation timestamp
            cursor.execute("""
                UPDATE chat_conversations
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (conversation_id,))

            return cursor.lastrowid

    def get_messages(self, conversation_id: str) -> List[Dict]:
        """Get all messages for a conversation"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM chat_messages
                WHERE conversation_id = ?
                ORDER BY created_at
            """, (conversation_id,))
            return [dict(row) for row in cursor.fetchall()]

    def update_conversation_title(self, conversation_id: str, title: str):
        """Update conversation title"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE chat_conversations
                SET title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (title, conversation_id))

    def delete_conversation(self, conversation_id: str):
        """Delete a conversation and its messages"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM chat_conversations WHERE id = ?",
                (conversation_id,)
            )

    def get_recent_conversations(self, project_id: int, limit: int = 10) -> List[Dict]:
        """Get recent conversations with message count"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.*, COUNT(m.id) as message_count
                FROM chat_conversations c
                LEFT JOIN chat_messages m ON c.id = m.conversation_id
                WHERE c.project_id = ?
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?
            """, (project_id, limit))
            return [dict(row) for row in cursor.fetchall()]


class SearchContextDAO:
    """Manages search context persistence for literature searches"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def save_context(
        self,
        project_id: int,
        search_id: int,
        context: Dict
    ) -> int:
        """
        Save full search context. Deactivates previous contexts for this project.

        Args:
            project_id: Project ID
            search_id: Search history ID
            context: Dict with ranking_mode, ranking_weights, query_explanation, etc.

        Returns:
            New context ID
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Deactivate previous contexts for this project
            cursor.execute("""
                UPDATE search_context
                SET is_active = 0
                WHERE project_id = ?
            """, (project_id,))

            # Insert new context
            cursor.execute("""
                INSERT INTO search_context (
                    project_id, search_id, ranking_mode, ranking_weights,
                    query_explanation, query_confidence, query_type, selected_pmids
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                project_id,
                search_id,
                context.get('ranking_mode'),
                json.dumps(context.get('ranking_weights', {})),
                context.get('query_explanation'),
                context.get('query_confidence'),
                context.get('query_type'),
                json.dumps(context.get('selected_pmids', []))
            ))

            return cursor.lastrowid

    def get_active_context(self, project_id: int) -> Optional[Dict]:
        """Get the most recent active search context for a project."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM search_context
                WHERE project_id = ? AND is_active = 1
                ORDER BY created_at DESC
                LIMIT 1
            """, (project_id,))

            row = cursor.fetchone()
            if row:
                result = dict(row)
                # Parse JSON fields
                if result.get('ranking_weights'):
                    try:
                        result['ranking_weights'] = json.loads(result['ranking_weights'])
                    except (json.JSONDecodeError, TypeError):
                        result['ranking_weights'] = {}
                if result.get('selected_pmids'):
                    try:
                        result['selected_pmids'] = json.loads(result['selected_pmids'])
                    except (json.JSONDecodeError, TypeError):
                        result['selected_pmids'] = []
                return result
            return None

    def update_selected_pmids(self, context_id: int, pmids: List[str]) -> bool:
        """Update selected papers for a search context."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE search_context
                SET selected_pmids = ?
                WHERE id = ?
            """, (json.dumps(pmids), context_id))
            return cursor.rowcount > 0

    def get_context_by_search(self, search_id: int) -> Optional[Dict]:
        """Get context for a specific search."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM search_context
                WHERE search_id = ?
            """, (search_id,))

            row = cursor.fetchone()
            if row:
                result = dict(row)
                # Parse JSON fields
                if result.get('ranking_weights'):
                    try:
                        result['ranking_weights'] = json.loads(result['ranking_weights'])
                    except (json.JSONDecodeError, TypeError):
                        result['ranking_weights'] = {}
                if result.get('selected_pmids'):
                    try:
                        result['selected_pmids'] = json.loads(result['selected_pmids'])
                    except (json.JSONDecodeError, TypeError):
                        result['selected_pmids'] = []
                return result
            return None

    def get_all_contexts_for_project(self, project_id: int, limit: int = 10) -> List[Dict]:
        """Get all search contexts for a project (for history)."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT sc.*, sh.query as search_query
                FROM search_context sc
                LEFT JOIN search_history sh ON sc.search_id = sh.id
                WHERE sc.project_id = ?
                ORDER BY sc.created_at DESC
                LIMIT ?
            """, (project_id, limit))

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                # Parse JSON fields
                if result.get('ranking_weights'):
                    try:
                        result['ranking_weights'] = json.loads(result['ranking_weights'])
                    except (json.JSONDecodeError, TypeError):
                        result['ranking_weights'] = {}
                if result.get('selected_pmids'):
                    try:
                        result['selected_pmids'] = json.loads(result['selected_pmids'])
                    except (json.JSONDecodeError, TypeError):
                        result['selected_pmids'] = []
                results.append(result)
            return results


class PaperSignalDAO:
    """Manages paper signals for learning from user selections.

    Signal types and their boost weights:
    - cited: +0.25 (highest - user explicitly referenced this paper)
    - selected: +0.15 (user selected for review)
    - tagged: +0.10 (user tagged as relevant)
    - rejected: -0.10 (user explicitly rejected)
    """

    BOOST_WEIGHTS = {
        'cited': 0.25,
        'selected': 0.15,
        'tagged': 0.10,
        'rejected': -0.10
    }

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def record_signal(
        self,
        project_id: int,
        pmid: str,
        signal_type: str,
        query_hash: str = None
    ) -> int:
        """
        Record a user signal for a paper.

        Args:
            project_id: Project ID
            pmid: Paper PMID
            signal_type: One of 'selected', 'cited', 'tagged', 'rejected'
            query_hash: Optional hash of the search query that returned this paper

        Returns:
            Signal ID (or existing ID if duplicate)
        """
        if signal_type not in self.BOOST_WEIGHTS:
            raise ValueError(f"Invalid signal_type: {signal_type}. Must be one of {list(self.BOOST_WEIGHTS.keys())}")

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO paper_signals (project_id, pmid, signal_type, query_hash)
                    VALUES (?, ?, ?, ?)
                """, (project_id, pmid, signal_type, query_hash))
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Signal already exists - get existing ID
                cursor.execute("""
                    SELECT id FROM paper_signals
                    WHERE project_id = ? AND pmid = ? AND signal_type = ?
                """, (project_id, pmid, signal_type))
                row = cursor.fetchone()
                return row['id'] if row else None

    def remove_signal(self, project_id: int, pmid: str, signal_type: str) -> bool:
        """Remove a signal (e.g., when user deselects a paper)."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM paper_signals
                WHERE project_id = ? AND pmid = ? AND signal_type = ?
            """, (project_id, pmid, signal_type))
            return cursor.rowcount > 0

    def get_signals_for_paper(self, project_id: int, pmid: str) -> List[Dict]:
        """Get all signals for a specific paper."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM paper_signals
                WHERE project_id = ? AND pmid = ?
                ORDER BY created_at DESC
            """, (project_id, pmid))
            return [dict(row) for row in cursor.fetchall()]

    def get_boosts(self, project_id: int, pmids: List[str]) -> Dict[str, float]:
        """
        Calculate boost scores for a list of PMIDs based on recorded signals.

        Args:
            project_id: Project ID
            pmids: List of PMIDs to get boosts for

        Returns:
            Dict mapping PMID -> total boost score
        """
        if not pmids:
            return {}

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join(['?' for _ in pmids])
            cursor.execute(f"""
                SELECT pmid, signal_type FROM paper_signals
                WHERE project_id = ? AND pmid IN ({placeholders})
            """, [project_id] + list(pmids))

            boosts = {}
            for row in cursor.fetchall():
                pmid = row['pmid']
                signal_type = row['signal_type']
                weight = self.BOOST_WEIGHTS.get(signal_type, 0)
                boosts[pmid] = boosts.get(pmid, 0) + weight

            return boosts

    def get_all_signals(self, project_id: int) -> List[Dict]:
        """Get all signals for a project."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM paper_signals
                WHERE project_id = ?
                ORDER BY created_at DESC
            """, (project_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_signal_stats(self, project_id: int) -> Dict[str, int]:
        """Get signal counts by type for a project."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT signal_type, COUNT(*) as count
                FROM paper_signals
                WHERE project_id = ?
                GROUP BY signal_type
            """, (project_id,))
            return {row['signal_type']: row['count'] for row in cursor.fetchall()}


class ExpertCorrectionDAO:
    """Manages expert corrections for learning from user feedback."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def add_correction(
        self,
        expert_name: str,
        critique: str,
        project_id: int = None,
        question_snippet: str = None
    ) -> int:
        """
        Add a correction/feedback for an expert's response.

        Args:
            expert_name: Name of the expert (e.g., "DMPK Scientist")
            critique: User's feedback/correction text
            project_id: Optional project ID for context
            question_snippet: Optional snippet of the original question

        Returns:
            Correction ID
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO expert_corrections (project_id, expert_name, question_snippet, critique)
                VALUES (?, ?, ?, ?)
            """, (project_id, expert_name, question_snippet, critique))
            return cursor.lastrowid

    def get_corrections(
        self,
        expert_name: str,
        project_id: int = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Get corrections for an expert, optionally filtered by project.

        Args:
            expert_name: Name of the expert
            project_id: Optional project ID filter
            limit: Maximum corrections to return

        Returns:
            List of correction dicts
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            if project_id is not None:
                cursor.execute("""
                    SELECT * FROM expert_corrections
                    WHERE expert_name = ? AND project_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (expert_name, project_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM expert_corrections
                    WHERE expert_name = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (expert_name, limit))

            return [dict(row) for row in cursor.fetchall()]

    def get_all_corrections(self, project_id: int = None, limit: int = 50) -> List[Dict]:
        """Get all corrections, optionally filtered by project."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            if project_id is not None:
                cursor.execute("""
                    SELECT * FROM expert_corrections
                    WHERE project_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (project_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM expert_corrections
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def format_for_prompt(self, expert_name: str, project_id: int = None, limit: int = 3) -> str:
        """
        Format corrections as text for prompt injection.

        Args:
            expert_name: Name of the expert
            project_id: Optional project ID filter
            limit: Maximum corrections to include

        Returns:
            Formatted string for prompt injection, or empty string if none
        """
        corrections = self.get_corrections(expert_name, project_id, limit)

        if not corrections:
            return ""

        lines = ["## Previous User Feedback", "The user has provided the following feedback on your previous responses:", ""]

        for i, c in enumerate(corrections, 1):
            lines.append(f"**Feedback {i}:**")
            if c.get('question_snippet'):
                lines.append(f"- Context: {c['question_snippet'][:100]}...")
            lines.append(f"- Correction: {c['critique']}")
            lines.append("")

        lines.append("Please incorporate this feedback in your responses.")
        return "\n".join(lines)


class ProgramProfileDAO:
    """Manages program profile for auto-detected program context.

    Stores target, indication, drugs, competitors, etc. per project.
    Auto-populated from user questions via ProgramExtractor.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def upsert(self, project_id: int, profile: Dict) -> None:
        """
        Create or update program profile for a project.

        Args:
            project_id: Project ID
            profile: Dict with target, indication, drug_names, etc.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO program_profile
                    (project_id, target, indication, drug_names, competitors,
                     mechanism, therapeutic_area, development_stage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    target = excluded.target,
                    indication = excluded.indication,
                    drug_names = excluded.drug_names,
                    competitors = excluded.competitors,
                    mechanism = excluded.mechanism,
                    therapeutic_area = excluded.therapeutic_area,
                    development_stage = excluded.development_stage,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                project_id,
                profile.get("target"),
                profile.get("indication"),
                json.dumps(profile.get("drug_names", [])),
                json.dumps(profile.get("competitors", [])),
                profile.get("mechanism"),
                profile.get("therapeutic_area"),
                profile.get("development_stage")
            ))

    def get(self, project_id: int) -> Optional[Dict]:
        """
        Get program profile for a project.

        Args:
            project_id: Project ID

        Returns:
            Profile dict with parsed JSON fields, or None if not found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM program_profile WHERE project_id = ?",
                (project_id,)
            )
            row = cursor.fetchone()
            if row:
                result = dict(row)
                # Parse JSON fields
                try:
                    result["drug_names"] = json.loads(result.get("drug_names") or "[]")
                except (json.JSONDecodeError, TypeError):
                    result["drug_names"] = []
                try:
                    result["competitors"] = json.loads(result.get("competitors") or "[]")
                except (json.JSONDecodeError, TypeError):
                    result["competitors"] = []
                return result
            return None

    def delete(self, project_id: int) -> bool:
        """Delete program profile for a project."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM program_profile WHERE project_id = ?",
                (project_id,)
            )
            return cursor.rowcount > 0

    def format_for_search(self, project_id: int) -> Optional[str]:
        """
        Format profile as search query augmentation.

        Returns:
            String like "KRAS G12C NSCLC" or None if no profile
        """
        profile = self.get(project_id)
        if not profile:
            return None

        parts = []
        if profile.get("target"):
            parts.append(profile["target"])
        if profile.get("indication"):
            parts.append(profile["indication"])
        if profile.get("drug_names"):
            parts.extend(profile["drug_names"][:2])  # Limit to first 2 drugs

        return " ".join(parts) if parts else None


class TrustedKnowledgeDAO:
    """Store and retrieve auto-fetched trusted knowledge.

    Stores high-confidence information from PubMed, ClinicalTrials.gov, FDA, etc.
    Auto-populated by AutoFetchService when users ask questions.
    Used for prompt injection to give experts prior knowledge.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def add(
        self,
        project_id: int,
        source: str,
        source_id: str,
        title: str,
        insight: str,
        confidence: float = 0.5,
        entry_type: str = None,
        metadata: Dict = None
    ) -> None:
        """
        Add or update trusted knowledge entry.

        Args:
            project_id: Project ID
            source: Source type ('pubmed', 'clinicaltrials', 'fda')
            source_id: Source identifier (PMID, NCT ID, etc.)
            title: Entry title
            insight: Key insight/summary
            confidence: Confidence score 0-1
            entry_type: Entry type ('paper', 'trial', 'approval')
            metadata: Additional metadata as dict
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trusted_knowledge
                    (project_id, source, source_id, title, insight,
                     confidence, entry_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, source, source_id) DO UPDATE SET
                    title = excluded.title,
                    insight = excluded.insight,
                    confidence = excluded.confidence,
                    entry_type = excluded.entry_type,
                    metadata = excluded.metadata
            """, (
                project_id, source, source_id, title, insight,
                confidence, entry_type, json.dumps(metadata or {})
            ))

    def get_for_project(
        self,
        project_id: int,
        min_confidence: float = 0.5,
        limit: int = 20
    ) -> List[Dict]:
        """
        Get trusted knowledge entries for a project.

        Args:
            project_id: Project ID
            min_confidence: Minimum confidence threshold
            limit: Max entries to return

        Returns:
            List of knowledge entry dicts
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM trusted_knowledge
                WHERE project_id = ? AND confidence >= ?
                ORDER BY confidence DESC, created_at DESC
                LIMIT ?
            """, (project_id, min_confidence, limit))

            results = []
            for row in cursor.fetchall():
                entry = dict(row)
                # Parse metadata JSON
                try:
                    entry["metadata"] = json.loads(entry.get("metadata") or "{}")
                except (json.JSONDecodeError, TypeError):
                    entry["metadata"] = {}
                results.append(entry)
            return results

    def get_context_for_question(
        self,
        project_id: int,
        question: str,
        max_entries: int = 8
    ) -> str:
        """
        Get relevant knowledge formatted for prompt injection.

        Scores entries by keyword overlap with question,
        then formats as markdown for injection into expert prompts.

        Args:
            project_id: Project ID
            question: User's question for relevance scoring
            max_entries: Max entries to include

        Returns:
            Formatted markdown string, or empty string if no knowledge
        """
        entries = self.get_for_project(project_id, limit=50)
        if not entries:
            return ""

        # Score by keyword overlap
        q_words = set(question.lower().split())
        scored = []
        for e in entries:
            text = f"{e.get('title', '')} {e.get('insight', '')}".lower()
            overlap = len(q_words & set(text.split()))
            score = overlap * e.get('confidence', 0.5)
            scored.append((score, e))

        scored.sort(reverse=True)
        top = [e for _, e in scored[:max_entries]]

        if not top:
            return ""

        # Format as markdown
        lines = ["## Prior Knowledge (Auto-Fetched):\n"]
        for e in top:
            icon = {
                "pubmed": "📄",
                "clinicaltrials": "🏥",
                "fda": "✅"
            }.get(e.get("source", ""), "📌")

            lines.append(f"{icon} **{e.get('title', 'Unknown')[:60]}**")
            insight = e.get('insight', '')[:150]
            if insight:
                lines.append(f"   {insight}...")
            lines.append(f"   _[{e.get('source', 'unknown')}: {e.get('source_id', '')}]_\n")

        return "\n".join(lines)

    def delete_for_project(self, project_id: int) -> int:
        """Delete all trusted knowledge for a project."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM trusted_knowledge WHERE project_id = ?",
                (project_id,)
            )
            return cursor.rowcount

    def sync_to_knowledge_store(self, project_id: int) -> int:
        """
        Sync trusted knowledge to JSON KnowledgeStore for prompt injection.

        Returns:
            Number of entries synced
        """
        from core.knowledge_store import get_default_store

        store = get_default_store()
        entries = self.get_for_project(project_id)

        for entry in entries:
            store.add_knowledge(
                persona="Trusted Sources",
                source=f"{entry.get('source', 'unknown')}:{entry.get('source_id', '')}",
                facts=[entry.get('insight', '')[:200]]
            )

        return len(entries)


# =============================================================================
# MIGRATION HELPERS
# =============================================================================

def migrate_json_session_to_sqlite(json_path: Path, db_manager: DatabaseManager, project_name: str):
    """
    Migrate legacy JSON session files to SQLite database

    Args:
        json_path: Path to JSON session file
        db_manager: DatabaseManager instance
        project_name: Name for the new project
    """
    with open(json_path, 'r') as f:
        session_data = json.load(f)

    # Create DAOs
    project_dao = ProjectDAO(db_manager)
    citation_dao = CitationDAO(db_manager)
    search_dao = SearchHistoryDAO(db_manager)

    # Create project
    project_id = project_dao.create_project(
        name=project_name,
        description=f"Migrated from {json_path.name}"
    )

    # Import citations
    if 'search_results' in session_data and 'citations' in session_data['search_results']:
        for citation in session_data['search_results']['citations']:
            # Insert citation
            citation_dao.upsert_citation(citation)
            # Link to project
            citation_dao.add_citation_to_project(project_id, citation['pmid'])

    # Import search history
    if 'search_results' in session_data:
        sr = session_data['search_results']
        search_dao.add_search(
            project_id=project_id,
            query=sr.get('search_query', ''),
            filters={},
            total_results=sr.get('total_results', 0),
            retrieved_count=len(sr.get('citations', []))
        )

    print(f"✅ Migrated {json_path.name} to project '{project_name}' (ID: {project_id})")
    return project_id


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Literature Review Database")
    print("=" * 70)

    # Create test database
    test_db_path = Path(__file__).parent.parent / "outputs" / "test_literature.db"
    db = DatabaseManager(test_db_path)

    # Test ProjectDAO
    print("\n1. Testing ProjectDAO...")
    project_dao = ProjectDAO(db)
    pid = project_dao.create_project("Test Systematic Review", "Testing database layer")
    print(f"   Created project ID: {pid}")

    projects = project_dao.get_all_projects()
    print(f"   Retrieved {len(projects)} projects")

    # Test CitationDAO
    print("\n2. Testing CitationDAO...")
    citation_dao = CitationDAO(db)

    test_citation = {
        'pmid': '12345678',
        'title': 'Machine Learning in Medicine',
        'authors': ['Smith J', 'Doe A', 'Johnson B'],
        'journal': 'Nature Medicine',
        'year': 2024,
        'abstract': 'This is a test abstract about ML in medicine.',
        'doi': '10.1038/test.2024',
        'publication_types': 'Journal Article',
        'keywords': 'machine learning; medicine; AI'
    }

    citation_dao.upsert_citation(test_citation)
    citation_dao.add_citation_to_project(pid, '12345678')
    print(f"   Added citation PMID: 12345678")

    citations = citation_dao.get_citations_by_project(pid)
    print(f"   Retrieved {len(citations)} citations for project")

    # Test ScreeningDAO
    print("\n3. Testing ScreeningDAO...")
    screening_dao = ScreeningDAO(db)

    screening_dao.add_decision(pid, '12345678', 'user1', 'include', 'title', 'Relevant topic')
    screening_dao.add_decision(pid, '12345678', 'user2', 'exclude', 'title', 'Out of scope')
    print(f"   Added 2 conflicting decisions")

    conflicts = screening_dao.detect_conflicts(pid, 'title')
    print(f"   Detected {len(conflicts)} conflicts")

    prisma = screening_dao.get_prisma_counts(pid)
    print(f"   PRISMA counts: {prisma}")

    # Test SearchHistoryDAO
    print("\n4. Testing SearchHistoryDAO...")
    search_dao = SearchHistoryDAO(db)

    search_id = search_dao.add_search(
        pid,
        'machine learning AND medicine',
        {'max_results': 100, 'date_from': '2020/01/01'},
        1523,
        100
    )
    print(f"   Recorded search ID: {search_id}")

    history = search_dao.get_search_history(pid)
    print(f"   Retrieved {len(history)} search records")

    # Test TagDAO
    print("\n5. Testing TagDAO...")
    tag_dao = TagDAO(db)

    tag_id = tag_dao.create_tag("Systematic Review", "#3C3A36")
    tag_dao.tag_paper(pid, '12345678', tag_id)
    print(f"   Created and applied tag ID: {tag_id}")

    paper_tags = tag_dao.get_paper_tags(pid, '12345678')
    print(f"   Paper has {len(paper_tags)} tags")

    print("\n" + "=" * 70)
    print(f"✅ All tests passed! Database created at: {test_db_path}")
    print(f"   File size: {test_db_path.stat().st_size} bytes")
