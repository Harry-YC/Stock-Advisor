"""
Test search context persistence functionality.

This test verifies that:
1. Search context is saved when a search is executed
2. Selected papers are persisted when toggled
3. Search context is restored when switching projects
"""

import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import DatabaseManager, SearchContextDAO, ProjectDAO, SearchHistoryDAO, PaperSignalDAO, ExpertCorrectionDAO, ProgramProfileDAO


@pytest.fixture
def test_db(tmp_path):
    """Create a test database."""
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)
    return db


@pytest.fixture
def search_context_dao(test_db):
    """Create SearchContextDAO instance."""
    return SearchContextDAO(test_db)


@pytest.fixture
def project_dao(test_db):
    """Create ProjectDAO instance."""
    return ProjectDAO(test_db)


@pytest.fixture
def search_dao(test_db):
    """Create SearchHistoryDAO instance."""
    return SearchHistoryDAO(test_db)


class TestSearchContextDAO:
    """Test SearchContextDAO functionality."""

    def test_save_context(self, search_context_dao, project_dao, search_dao):
        """Test saving a search context."""
        # Create a project first
        project_id = project_dao.create_project("Test Project", "Test description")

        # Create a search history entry
        search_id = search_dao.add_search(
            project_id=project_id,
            query="test query",
            filters={},
            total_results=100,
            retrieved_count=50
        )

        # Save search context
        context = {
            'ranking_mode': 'balanced',
            'ranking_weights': {'relevance': 0.4, 'evidence': 0.3, 'recency': 0.2, 'influence': 0.1},
            'query_explanation': 'Test explanation',
            'query_confidence': 'high',
            'query_type': 'DIRECT',
            'selected_pmids': []
        }

        context_id = search_context_dao.save_context(project_id, search_id, context)

        assert context_id is not None
        assert context_id > 0

    def test_get_active_context(self, search_context_dao, project_dao, search_dao):
        """Test retrieving active context for a project."""
        # Create project and search
        project_id = project_dao.create_project("Test Project 2", "")
        search_id = search_dao.add_search(
            project_id=project_id,
            query="cancer treatment",
            filters={},
            total_results=200,
            retrieved_count=100
        )

        # Save context
        context = {
            'ranking_mode': 'clinical',
            'ranking_weights': {'relevance': 0.5, 'evidence': 0.3, 'recency': 0.1, 'influence': 0.1},
            'query_explanation': 'Clinical focus',
            'query_confidence': 'medium',
            'query_type': 'OPTIMIZED',
            'selected_pmids': ['12345', '67890']
        }

        context_id = search_context_dao.save_context(project_id, search_id, context)

        # Get active context
        active = search_context_dao.get_active_context(project_id)

        assert active is not None
        assert active['ranking_mode'] == 'clinical'
        assert active['query_confidence'] == 'medium'
        assert active['selected_pmids'] == ['12345', '67890']

    def test_update_selected_pmids(self, search_context_dao, project_dao, search_dao):
        """Test updating selected PMIDs."""
        # Create project and search
        project_id = project_dao.create_project("Test Project 3", "")
        search_id = search_dao.add_search(
            project_id=project_id,
            query="drug efficacy",
            filters={},
            total_results=50,
            retrieved_count=30
        )

        # Save context with empty selection
        context = {
            'ranking_mode': 'balanced',
            'ranking_weights': {},
            'query_explanation': '',
            'query_confidence': 'high',
            'query_type': 'DIRECT',
            'selected_pmids': []
        }

        context_id = search_context_dao.save_context(project_id, search_id, context)

        # Update selected PMIDs
        new_pmids = ['11111', '22222', '33333']
        success = search_context_dao.update_selected_pmids(context_id, new_pmids)

        assert success is True

        # Verify update
        active = search_context_dao.get_active_context(project_id)
        assert active['selected_pmids'] == new_pmids

    def test_multiple_contexts_only_one_active(self, search_context_dao, project_dao, search_dao):
        """Test that only the most recent context is active."""
        # Create project
        project_id = project_dao.create_project("Test Project 4", "")

        # Create two searches
        search_id_1 = search_dao.add_search(project_id, "query 1", {}, 10, 5)
        search_id_2 = search_dao.add_search(project_id, "query 2", {}, 20, 10)

        # Save first context
        context_1 = {
            'ranking_mode': 'balanced',
            'ranking_weights': {},
            'query_explanation': 'First',
            'query_confidence': 'high',
            'query_type': 'DIRECT',
            'selected_pmids': ['old1']
        }
        search_context_dao.save_context(project_id, search_id_1, context_1)

        # Save second context (should deactivate first)
        context_2 = {
            'ranking_mode': 'clinical',
            'ranking_weights': {},
            'query_explanation': 'Second',
            'query_confidence': 'medium',
            'query_type': 'OPTIMIZED',
            'selected_pmids': ['new1', 'new2']
        }
        search_context_dao.save_context(project_id, search_id_2, context_2)

        # Get active context - should be the second one
        active = search_context_dao.get_active_context(project_id)

        assert active is not None
        assert active['query_explanation'] == 'Second'
        assert active['selected_pmids'] == ['new1', 'new2']


class TestKnowledgeStoreLearning:
    """Test knowledge store learning functionality."""

    def test_add_user_correction(self):
        """Test adding a user correction."""
        from core.knowledge_store import KnowledgeStore
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(storage_path=Path(tmpdir) / "test_knowledge.json")

            correction_id = store.add_user_correction(
                persona="DMPK Scientist",
                original_response="The half-life is 2 hours.",
                corrected_response="The half-life is 8 hours based on recent data.",
                question="What is the half-life of compound X?",
                project_id=1
            )

            assert correction_id is not None
            assert correction_id.startswith("correction_")

            # Verify correction was saved
            corrections = store.get_corrections_for_persona("DMPK Scientist")
            assert len(corrections) == 1
            assert corrections[0]['question'] == "What is the half-life of compound X?"

    def test_add_query_effectiveness(self):
        """Test adding query effectiveness tracking."""
        from core.knowledge_store import KnowledgeStore
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(storage_path=Path(tmpdir) / "test_knowledge.json")

            entry_id = store.add_query_effectiveness(
                query="EGFR inhibitor lung cancer",
                question_type="Go/No-Go",
                paper_count=100,
                selected_count=25,
                project_id=1
            )

            assert entry_id is not None

            # Verify effectiveness was calculated
            effective = store.get_effective_query_patterns("Go/No-Go", min_effectiveness=0.2)
            assert len(effective) == 1
            assert effective[0]['effectiveness'] == 0.25  # 25/100

    def test_format_learning_for_prompt(self):
        """Test formatting learning for prompt injection."""
        from core.knowledge_store import KnowledgeStore
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(storage_path=Path(tmpdir) / "test_knowledge.json")

            # Add some corrections
            store.add_user_correction(
                persona="Toxicology Expert",
                original_response="No hepatotoxicity concerns.",
                corrected_response="There are hepatotoxicity concerns based on ALT elevation.",
                question="Safety profile of compound Y?"
            )

            # Add effective query
            store.add_query_effectiveness(
                query="hepatotoxicity drug-induced liver injury",
                question_type="Safety",
                paper_count=50,
                selected_count=20
            )

            # Format for prompt
            prompt_text = store.format_learning_for_prompt(
                persona="Toxicology Expert",
                question_type="Safety"
            )

            assert "LEARNED INSIGHTS" in prompt_text
            assert "User Feedback History" in prompt_text
            assert "Effective Search Strategies" in prompt_text


class TestPaperSignalDAO:
    """Test PaperSignalDAO functionality (v4.2)."""

    def test_record_signal(self, test_db, project_dao):
        """Test recording a paper signal."""
        project_id = project_dao.create_project("Signal Test Project", "")
        paper_signal_dao = PaperSignalDAO(test_db)

        # Record a 'selected' signal
        signal_id = paper_signal_dao.record_signal(project_id, "12345678", "selected")
        assert signal_id is not None
        assert signal_id > 0

    def test_record_duplicate_signal(self, test_db, project_dao):
        """Test that duplicate signals are handled (UNIQUE constraint)."""
        project_id = project_dao.create_project("Duplicate Test", "")
        paper_signal_dao = PaperSignalDAO(test_db)

        # Record first signal
        signal_id_1 = paper_signal_dao.record_signal(project_id, "12345678", "selected")

        # Record duplicate - should return existing ID, not raise error
        signal_id_2 = paper_signal_dao.record_signal(project_id, "12345678", "selected")

        assert signal_id_1 == signal_id_2

    def test_remove_signal(self, test_db, project_dao):
        """Test removing a signal."""
        project_id = project_dao.create_project("Remove Signal Test", "")
        paper_signal_dao = PaperSignalDAO(test_db)

        # Record and then remove
        paper_signal_dao.record_signal(project_id, "12345678", "selected")
        removed = paper_signal_dao.remove_signal(project_id, "12345678", "selected")

        assert removed is True

        # Verify removal
        signals = paper_signal_dao.get_signals_for_paper(project_id, "12345678")
        assert len(signals) == 0

    def test_get_boosts(self, test_db, project_dao):
        """Test calculating boost scores."""
        project_id = project_dao.create_project("Boost Test", "")
        paper_signal_dao = PaperSignalDAO(test_db)

        # Record various signals
        paper_signal_dao.record_signal(project_id, "11111", "selected")  # +0.15
        paper_signal_dao.record_signal(project_id, "22222", "cited")     # +0.25
        paper_signal_dao.record_signal(project_id, "33333", "rejected")  # -0.10
        paper_signal_dao.record_signal(project_id, "44444", "selected")  # +0.15
        paper_signal_dao.record_signal(project_id, "44444", "cited")     # +0.25 (same paper, both signals)

        # Get boosts
        boosts = paper_signal_dao.get_boosts(project_id, ["11111", "22222", "33333", "44444", "55555"])

        assert boosts.get("11111") == 0.15
        assert boosts.get("22222") == 0.25
        assert boosts.get("33333") == -0.10
        assert boosts.get("44444") == 0.40  # selected + cited
        assert "55555" not in boosts  # No signal

    def test_invalid_signal_type(self, test_db, project_dao):
        """Test that invalid signal types raise ValueError."""
        project_id = project_dao.create_project("Invalid Signal Test", "")
        paper_signal_dao = PaperSignalDAO(test_db)

        with pytest.raises(ValueError):
            paper_signal_dao.record_signal(project_id, "12345678", "invalid_type")


class TestExpertCorrectionDAO:
    """Test ExpertCorrectionDAO functionality (v4.2)."""

    def test_add_correction(self, test_db, project_dao):
        """Test adding an expert correction."""
        project_id = project_dao.create_project("Correction Test", "")
        correction_dao = ExpertCorrectionDAO(test_db)

        correction_id = correction_dao.add_correction(
            expert_name="DMPK Scientist",
            critique="Please include more PK parameters",
            project_id=project_id,
            question_snippet="What is the half-life?"
        )

        assert correction_id is not None
        assert correction_id > 0

    def test_get_corrections(self, test_db, project_dao):
        """Test retrieving corrections for an expert."""
        project_id = project_dao.create_project("Get Corrections Test", "")
        correction_dao = ExpertCorrectionDAO(test_db)

        # Add multiple corrections
        correction_dao.add_correction("Toxicology Expert", "Add safety data", project_id)
        correction_dao.add_correction("Toxicology Expert", "Include ALT/AST", project_id)
        correction_dao.add_correction("DMPK Scientist", "Add bioavailability", project_id)

        # Get corrections for Toxicology Expert
        tox_corrections = correction_dao.get_corrections("Toxicology Expert", project_id)
        assert len(tox_corrections) == 2

        # Get corrections for DMPK
        dmpk_corrections = correction_dao.get_corrections("DMPK Scientist", project_id)
        assert len(dmpk_corrections) == 1

    def test_format_for_prompt(self, test_db, project_dao):
        """Test formatting corrections for prompt injection."""
        project_id = project_dao.create_project("Format Test", "")
        correction_dao = ExpertCorrectionDAO(test_db)

        # Add corrections
        correction_dao.add_correction(
            expert_name="Clinical Pharmacologist",
            critique="Always cite relevant drug interactions",
            project_id=project_id,
            question_snippet="Dosing for combo therapy?"
        )

        # Format for prompt
        prompt_text = correction_dao.format_for_prompt("Clinical Pharmacologist", project_id)

        assert "Previous User Feedback" in prompt_text
        assert "Always cite relevant drug interactions" in prompt_text
        assert "incorporate this feedback" in prompt_text

    def test_format_for_prompt_empty(self, test_db, project_dao):
        """Test that format_for_prompt returns empty string when no corrections."""
        project_id = project_dao.create_project("Empty Format Test", "")
        correction_dao = ExpertCorrectionDAO(test_db)

        prompt_text = correction_dao.format_for_prompt("Nonexistent Expert", project_id)
        assert prompt_text == ""


class TestProgramConclusions:
    """Test program conclusions functionality in KnowledgeStore (v4.2)."""

    def test_add_program_conclusion(self):
        """Test adding a program conclusion."""
        from core.knowledge_store import KnowledgeStore
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(storage_path=Path(tmpdir) / "test_knowledge.json")

            conclusion_id = store.add_program_conclusion(
                project_id=1,
                question="Should we proceed with compound X?",
                conclusion="Proceed with caution due to hepatotoxicity signals.",
                citations=["12345678", "87654321"],
                persona="Consensus"
            )

            assert conclusion_id is not None
            assert conclusion_id.startswith("conclusion_")

    def test_get_program_conclusions(self):
        """Test retrieving program conclusions."""
        from core.knowledge_store import KnowledgeStore
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(storage_path=Path(tmpdir) / "test_knowledge.json")

            # Add multiple conclusions
            store.add_program_conclusion(1, "Q1", "A1", ["111"])
            store.add_program_conclusion(1, "Q2", "A2", ["222"])
            store.add_program_conclusion(2, "Q3", "A3", ["333"])  # Different project

            # Get conclusions for project 1
            conclusions = store.get_program_conclusions(1)
            assert len(conclusions) == 2

            # Get conclusions for project 2
            conclusions_2 = store.get_program_conclusions(2)
            assert len(conclusions_2) == 1

    def test_format_conclusions_for_prompt(self):
        """Test formatting conclusions for prompt injection."""
        from core.knowledge_store import KnowledgeStore
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(storage_path=Path(tmpdir) / "test_knowledge.json")

            # Add a conclusion
            store.add_program_conclusion(
                project_id=1,
                question="What is the recommended dose?",
                conclusion="100mg QD based on PK modeling",
                citations=["12345678"]
            )

            # Format for prompt
            prompt_text = store.format_conclusions_for_prompt(1)

            assert "PRIOR PROGRAM CONCLUSIONS" in prompt_text
            assert "100mg QD" in prompt_text
            assert "12345678" in prompt_text

    def test_delete_conclusion(self):
        """Test deleting a conclusion."""
        from core.knowledge_store import KnowledgeStore
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(storage_path=Path(tmpdir) / "test_knowledge.json")

            conclusion_id = store.add_program_conclusion(1, "Q1", "A1", [])

            # Delete
            deleted = store.delete_conclusion(conclusion_id)
            assert deleted is True

            # Verify deletion
            conclusions = store.get_program_conclusions(1)
            assert len(conclusions) == 0


class TestProgramProfileDAO:
    """Test ProgramProfileDAO functionality (Auto-Fetch feature)."""

    def test_upsert_and_get(self, test_db, project_dao):
        """Test upserting and retrieving a program profile."""
        project_id = project_dao.create_project("Profile Test Project", "")
        profile_dao = ProgramProfileDAO(test_db)

        # Create profile
        profile = {
            "target": "KRAS G12C",
            "indication": "NSCLC",
            "drug_names": ["sotorasib", "adagrasib"],
            "competitors": ["pembrolizumab"],
            "mechanism": "KRAS G12C inhibitor",
            "therapeutic_area": "oncology"
        }
        profile_dao.upsert(project_id, profile)

        # Retrieve profile
        result = profile_dao.get(project_id)

        assert result is not None
        assert result["target"] == "KRAS G12C"
        assert result["indication"] == "NSCLC"
        assert result["drug_names"] == ["sotorasib", "adagrasib"]
        assert result["competitors"] == ["pembrolizumab"]
        assert result["therapeutic_area"] == "oncology"

    def test_upsert_updates_existing(self, test_db, project_dao):
        """Test that upsert updates existing profile."""
        project_id = project_dao.create_project("Update Test Project", "")
        profile_dao = ProgramProfileDAO(test_db)

        # Create initial profile
        profile_dao.upsert(project_id, {"target": "EGFR", "indication": "NSCLC"})

        # Update profile
        profile_dao.upsert(project_id, {"target": "KRAS G12C", "indication": "NSCLC"})

        # Verify update
        result = profile_dao.get(project_id)
        assert result["target"] == "KRAS G12C"

    def test_get_nonexistent(self, test_db, project_dao):
        """Test getting a nonexistent profile returns None."""
        project_id = project_dao.create_project("Nonexistent Test", "")
        profile_dao = ProgramProfileDAO(test_db)

        result = profile_dao.get(project_id)
        assert result is None

    def test_delete(self, test_db, project_dao):
        """Test deleting a profile."""
        project_id = project_dao.create_project("Delete Test", "")
        profile_dao = ProgramProfileDAO(test_db)

        # Create and delete
        profile_dao.upsert(project_id, {"target": "BRAF"})
        deleted = profile_dao.delete(project_id)

        assert deleted is True
        assert profile_dao.get(project_id) is None

    def test_format_for_search(self, test_db, project_dao):
        """Test formatting profile for search augmentation."""
        project_id = project_dao.create_project("Format Test", "")
        profile_dao = ProgramProfileDAO(test_db)

        profile_dao.upsert(project_id, {
            "target": "KRAS G12C",
            "indication": "NSCLC",
            "drug_names": ["sotorasib"]
        })

        search_str = profile_dao.format_for_search(project_id)
        assert search_str is not None
        assert "KRAS G12C" in search_str
        assert "NSCLC" in search_str
        assert "sotorasib" in search_str


class TestProgramExtractor:
    """Test ProgramExtractor fallback extraction."""

    def test_fallback_extraction_kras(self):
        """Test fallback extraction for KRAS question."""
        from services.program_extractor import ProgramExtractor

        # Use fallback directly (no API key needed)
        extractor = ProgramExtractor(api_key="dummy")
        result = extractor._fallback_extraction(
            "What is the efficacy of sotorasib in 2L NSCLC patients with KRAS G12C mutation?"
        )

        assert result["target"] == "KRAS G12C"
        assert result["indication"] == "NSCLC"
        assert "sotorasib" in result["drug_names"]
        assert result["therapeutic_area"] == "oncology"

    def test_fallback_extraction_egfr(self):
        """Test fallback extraction for EGFR question."""
        from services.program_extractor import ProgramExtractor

        extractor = ProgramExtractor(api_key="dummy")
        result = extractor._fallback_extraction(
            "How does osimertinib compare to erlotinib in EGFR-mutant lung cancer?"
        )

        assert result["target"] == "EGFR"
        assert "osimertinib" in result["drug_names"]

    def test_fallback_extraction_no_entities(self):
        """Test fallback extraction when no entities found."""
        from services.program_extractor import ProgramExtractor

        extractor = ProgramExtractor(api_key="dummy")
        result = extractor._fallback_extraction("What is the weather today?")

        assert result["target"] is None
        assert result["indication"] is None
        assert result["drug_names"] == []

    def test_merge_profiles_new(self):
        """Test merging with empty existing profile."""
        from services.program_extractor import ProgramExtractor

        extractor = ProgramExtractor(api_key="dummy")

        new_profile = {
            "target": "KRAS",
            "indication": "NSCLC",
            "drug_names": ["sotorasib"]
        }

        result = extractor.merge_profiles(None, new_profile)
        assert result == new_profile

    def test_merge_profiles_existing(self):
        """Test merging with existing profile."""
        from services.program_extractor import ProgramExtractor

        extractor = ProgramExtractor(api_key="dummy")

        existing = {
            "target": "KRAS G12C",
            "indication": None,
            "drug_names": ["sotorasib"],
            "competitors": []
        }

        new = {
            "target": None,  # Should not overwrite existing
            "indication": "NSCLC",  # Should fill empty field
            "drug_names": ["adagrasib"],  # Should merge
            "competitors": ["pembrolizumab"]
        }

        result = extractor.merge_profiles(existing, new)

        assert result["target"] == "KRAS G12C"  # Kept existing
        assert result["indication"] == "NSCLC"  # Filled empty
        assert "sotorasib" in result["drug_names"]  # Merged
        assert "adagrasib" in result["drug_names"]  # Merged
        assert "pembrolizumab" in result["competitors"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
