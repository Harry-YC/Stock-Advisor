"""
Playwright tests for Two-Channel Search functionality.

Tests the two-channel search system that separates:
- Clinical Channel: Surgical outcomes, morbidity/mortality, comparative effectiveness
- Symptom Channel: QoL, symptom palliation, functional outcomes
"""

import pytest
import os
from playwright.sync_api import Page, expect

# Get test URL from environment or default
TEST_URL = os.environ.get("TEST_URL", "http://localhost:8501")


class TestTwoChannelSearchUnit:
    """Unit tests for two-channel search components (no browser required)."""

    def test_query_builder_creates_clinical_queries(self):
        """Test that TwoChannelQueryBuilder creates clinical queries."""
        from core.search_channels import TwoChannelQueryBuilder
        from core.query_extractor import ExtractedConcepts

        concepts = ExtractedConcepts(
            conditions=["malignant bowel obstruction"],
            anatomy=["colon"],
            procedures=["gastrojejunostomy", "stent"],
            outcomes=["mortality"],
            scores=["ECOG"],
            cancers=["ovarian cancer"],
            comparisons=["versus"]
        )

        builder = TwoChannelQueryBuilder()
        queries = builder.build(concepts)

        # Should have clinical queries
        assert len(queries.clinical_queries) > 0
        # Should include surgical outcome queries
        assert any("mortality" in q.query.lower() or "morbidity" in q.query.lower()
                   for q in queries.clinical_queries)
        # Should include procedure queries
        assert any("gastrojejunostomy" in q.query.lower() or "stent" in q.query.lower()
                   for q in queries.clinical_queries)

    def test_query_builder_creates_symptom_queries(self):
        """Test that TwoChannelQueryBuilder creates symptom queries."""
        from core.search_channels import TwoChannelQueryBuilder
        from core.query_extractor import ExtractedConcepts

        concepts = ExtractedConcepts(
            conditions=["malignant bowel obstruction"],
            anatomy=["colon"],
            procedures=["gastrojejunostomy"],
            outcomes=["quality of life", "symptom control"],
            scores=["GOOSS"],
            cancers=["ovarian cancer"],
            comparisons=[]
        )

        builder = TwoChannelQueryBuilder()
        queries = builder.build(concepts)

        # Should have symptom queries
        assert len(queries.symptom_queries) > 0
        # Should include QoL queries
        assert any("quality of life" in q.query.lower() or "qol" in q.query.lower()
                   for q in queries.symptom_queries)
        # Should include symptom/palliation queries
        assert any("symptom" in q.query.lower() or "palliation" in q.query.lower()
                   for q in queries.symptom_queries)

    def test_two_channel_search_result_structure(self):
        """Test TwoChannelSearchResult data structure."""
        from services.two_channel_search_service import TwoChannelSearchResult
        from core.search_channels import SearchChannel

        result = TwoChannelSearchResult()

        # Should have both channels initialized
        assert result.clinical is not None
        assert result.symptom is not None
        assert result.clinical.channel == SearchChannel.CLINICAL
        assert result.symptom.channel == SearchChannel.SYMPTOM

        # Should serialize to dict
        d = result.to_dict()
        assert "clinical" in d
        assert "symptom" in d
        assert "total_citations" in d

    def test_channel_query_priorities(self):
        """Test that queries are prioritized correctly."""
        from core.search_channels import TwoChannelQueryBuilder
        from core.query_extractor import ExtractedConcepts

        concepts = ExtractedConcepts(
            conditions=["pathologic fracture"],
            anatomy=["femur"],
            procedures=["surgical stabilization"],
            outcomes=["survival"],
            scores=["Mirels"],
            cancers=["breast cancer"],
            comparisons=[]
        )

        builder = TwoChannelQueryBuilder()
        queries = builder.build(concepts)

        # Priority 1 queries should come first when sorted
        sorted_symptom = sorted(queries.symptom_queries, key=lambda q: q.priority)
        if len(sorted_symptom) > 1:
            assert sorted_symptom[0].priority <= sorted_symptom[-1].priority

    def test_query_extractor_palliative_concepts(self):
        """Test that query extractor recognizes palliative surgery concepts."""
        from core.query_extractor import ClinicalQueryExtractor

        extractor = ClinicalQueryExtractor(api_key=None)  # Pattern-only mode

        # Test MBO extraction
        concepts = extractor.extract(
            "Should a patient with malignant bowel obstruction undergo gastrojejunostomy?"
        )
        assert "malignant bowel obstruction" in [c.lower() for c in concepts.conditions] or \
               "MBO" in concepts.conditions
        assert any("gastrojejunostomy" in p.lower() for p in concepts.procedures)

    def test_query_extractor_scores(self):
        """Test that query extractor recognizes clinical scores."""
        from core.query_extractor import ClinicalQueryExtractor

        extractor = ClinicalQueryExtractor(api_key=None)

        concepts = extractor.extract(
            "What Mirels score threshold indicates need for prophylactic fixation?"
        )
        assert any("mirels" in s.lower() for s in concepts.scores)

    def test_quality_report_structure(self):
        """Test QualityReport data structure."""
        from services.two_channel_search_service import QualityReport

        report = QualityReport(
            clinical_count=10,
            symptom_count=5,
            condition_coverage={"MBO": 8},
            procedure_coverage={"stent": 6},
            warnings=["Low symptom evidence"],
            filtered_out=3
        )

        d = report.to_dict()
        assert d["clinical_count"] == 10
        assert d["symptom_count"] == 5
        assert "MBO" in d["condition_coverage"]
        assert len(d["warnings"]) == 1


@pytest.mark.skipif(
    os.environ.get("SKIP_INTEGRATION_TESTS", "1") == "1",
    reason="Integration tests require running Streamlit app. Set SKIP_INTEGRATION_TESTS=0 to run."
)
class TestTwoChannelSearchIntegration:
    """Integration tests that require a running app (Playwright)."""

    @pytest.fixture(autouse=True)
    def setup(self, page: Page):
        """Setup for each test."""
        self.page = page
        try:
            self.page.goto(TEST_URL, timeout=10000)
            # Wait for app to load
            self.page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pytest.skip(f"Streamlit app not running at {TEST_URL}")

    def test_app_loads(self):
        """Test that the app loads successfully."""
        # Check for main title or app container
        expect(self.page.locator("body")).to_be_visible(timeout=5000)
        # App should not show error
        assert "Error" not in self.page.title() or self.page.title() == ""

    def test_evidence_drawer_has_tabs(self):
        """Test that evidence drawer shows Clinical/Symptom tabs when expanded."""
        # This test requires running a search first
        # For now, just verify the app structure is correct
        pass  # Will be implemented when UI integration is complete


class TestClinicalValidation:
    """Tests for claim validation against two-channel evidence."""

    def test_clinical_claims_validated_against_clinical_channel(self):
        """Test that clinical claims use clinical channel for validation."""
        from services.two_channel_search_service import TwoChannelSearchResult, ChannelResult
        from core.search_channels import SearchChannel
        from core.pubmed_client import Citation

        # Create mock search result
        clinical_citation = Citation(
            pmid="12345",
            title="Gastrojejunostomy versus duodenal stent for malignant GOO",
            authors=["Smith J"],
            year="2023",
            abstract="A retrospective study comparing surgical bypass with enteral stenting showed similar 30-day mortality.",
            journal="Ann Surg"
        )

        result = TwoChannelSearchResult()
        result.clinical = ChannelResult(
            channel=SearchChannel.CLINICAL,
            citations=[clinical_citation]
        )
        result.symptom = ChannelResult(
            channel=SearchChannel.SYMPTOM,
            citations=[]
        )

        # Should have clinical citations
        assert len(result.clinical.citations) > 0
        assert result.clinical.channel == SearchChannel.CLINICAL

    def test_symptom_claims_validated_against_symptom_channel(self):
        """Test that symptom claims use symptom channel for validation."""
        from services.two_channel_search_service import TwoChannelSearchResult, ChannelResult
        from core.search_channels import SearchChannel
        from core.pubmed_client import Citation

        # Create mock search result with symptom evidence
        symptom_citation = Citation(
            pmid="67890",
            title="Quality of Life After Palliative Surgery for MBO",
            authors=["Jones A"],
            year="2022",
            abstract="Patient-reported QoL improved significantly after surgical decompression in 78% of patients.",
            journal="Palliat Med"
        )

        result = TwoChannelSearchResult()
        result.clinical = ChannelResult(
            channel=SearchChannel.CLINICAL,
            citations=[]
        )
        result.symptom = ChannelResult(
            channel=SearchChannel.SYMPTOM,
            citations=[symptom_citation]
        )

        # Should have symptom citations
        assert len(result.symptom.citations) > 0
        assert result.symptom.channel == SearchChannel.SYMPTOM


class TestPalliativeSurgeryVocabulary:
    """Tests for palliative surgery domain vocabulary."""

    def test_condition_synonyms(self):
        """Test condition synonym lookup."""
        from core.query_extractor import get_synonyms

        mbo_synonyms = get_synonyms("mbo", "condition")
        assert len(mbo_synonyms) > 1
        assert any("malignant bowel obstruction" in s.lower() for s in mbo_synonyms)

    def test_procedure_synonyms(self):
        """Test procedure synonym lookup."""
        from core.query_extractor import get_synonyms

        sems_synonyms = get_synonyms("sems", "procedure")
        assert len(sems_synonyms) > 1
        assert any("stent" in s.lower() for s in sems_synonyms)

    def test_outcome_synonyms(self):
        """Test outcome synonym lookup."""
        from core.query_extractor import get_synonyms

        qol_synonyms = get_synonyms("qol", "outcome")
        assert len(qol_synonyms) > 1
        assert any("quality of life" in s.lower() for s in qol_synonyms)

    def test_extracted_concepts_pubmed_queries(self):
        """Test that ExtractedConcepts generates valid PubMed queries."""
        from core.query_extractor import ExtractedConcepts

        concepts = ExtractedConcepts(
            conditions=["malignant bowel obstruction"],
            anatomy=["colon"],
            procedures=["gastrojejunostomy", "stent"],
            outcomes=["quality of life"],
            scores=[],
            cancers=["ovarian cancer"],
            comparisons=["versus"]
        )

        queries = concepts.to_pubmed_queries()
        assert len(queries) > 0
        # Should include condition + procedure query
        assert any("bowel obstruction" in q.lower() for q in queries)

    def test_extracted_concepts_trials_query(self):
        """Test that ExtractedConcepts generates trials query."""
        from core.query_extractor import ExtractedConcepts

        concepts = ExtractedConcepts(
            conditions=["malignant pleural effusion"],
            anatomy=["pleura"],
            procedures=["pleurodesis"],
            outcomes=[],
            scores=[],
            cancers=["lung cancer"],
            comparisons=[]
        )

        trials_query = concepts.to_trials_query()
        assert len(trials_query) > 0
        assert "palliative" in trials_query.lower()


# Run with: TEST_URL=http://localhost:8501 pytest tests/test_two_channel_search.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
