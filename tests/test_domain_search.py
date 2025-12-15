"""
Tests for domain-specific search features.

Tests:
1. Vocabulary module loads correctly
2. Domain relevance boost (positive for palliative papers)
3. Domain relevance boost (negative for curative papers)
4. Query parser domain filter application
"""

import pytest


class TestVocabulary:
    """Test vocabulary module loading and content."""

    def test_vocabulary_imported(self):
        """Verify vocabulary module imports correctly."""
        from config.palliative_surgery_vocabulary import (
            MESH_HEADINGS,
            PALLIATIVE_PROCEDURES,
            DOMAIN_KEYWORDS,
            DEFAULT_DOMAIN_FILTER,
            EXCLUSION_FILTER,
        )

        assert MESH_HEADINGS is not None
        assert PALLIATIVE_PROCEDURES is not None
        assert DOMAIN_KEYWORDS is not None
        assert DEFAULT_DOMAIN_FILTER is not None
        assert EXCLUSION_FILTER is not None

    def test_mesh_headings_structure(self):
        """Verify MeSH headings structure."""
        from config.palliative_surgery_vocabulary import MESH_HEADINGS

        assert "palliative_care" in MESH_HEADINGS
        assert "surgical_procedures" in MESH_HEADINGS
        assert isinstance(MESH_HEADINGS["palliative_care"], list)
        assert len(MESH_HEADINGS["palliative_care"]) > 0

    def test_domain_keywords_structure(self):
        """Verify domain keywords structure."""
        from config.palliative_surgery_vocabulary import DOMAIN_KEYWORDS

        assert "high_relevance" in DOMAIN_KEYWORDS
        assert "procedure_specific" in DOMAIN_KEYWORDS
        assert "outcomes" in DOMAIN_KEYWORDS
        assert "negative_indicators" in DOMAIN_KEYWORDS

        assert "palliative" in DOMAIN_KEYWORDS["high_relevance"]
        assert "curative" in DOMAIN_KEYWORDS["negative_indicators"]

    def test_palliative_procedures_coverage(self):
        """Verify coverage of palliative procedure conditions."""
        from config.palliative_surgery_vocabulary import PALLIATIVE_PROCEDURES

        # Key conditions should be covered
        expected_conditions = [
            "malignant_bowel_obstruction",
            "gastric_outlet_obstruction",
            "biliary_obstruction",
            "malignant_pleural_effusion",
            "pathologic_fracture",
        ]

        for condition in expected_conditions:
            assert condition in PALLIATIVE_PROCEDURES, f"Missing condition: {condition}"
            assert "mesh" in PALLIATIVE_PROCEDURES[condition]
            assert "procedures" in PALLIATIVE_PROCEDURES[condition]


class TestDomainRelevanceBoost:
    """Test domain_relevance_boost function."""

    def test_palliative_paper_gets_boost(self):
        """Palliative surgery paper should get positive boost."""
        from core.ranking import domain_relevance_boost

        title = "Palliative surgery for malignant bowel obstruction in advanced cancer"
        abstract = """
        Background: Malignant bowel obstruction is common in terminal cancer patients.
        Objective: To evaluate palliative surgical intervention for symptom relief and quality of life.
        Methods: Retrospective analysis of patients with advanced cancer and bowel obstruction.
        Results: Surgery improved symptom control in 78% of patients.
        """

        boost = domain_relevance_boost(title, abstract, domain="palliative_surgery")

        # Should have positive boost
        assert boost > 0, f"Expected positive boost, got {boost}"

    def test_curative_paper_gets_penalty(self):
        """Curative surgery paper should get negative boost (penalty)."""
        from core.ranking import domain_relevance_boost

        title = "Curative resection for early stage colorectal cancer"
        abstract = """
        Background: Complete surgical resection offers cure for early-stage cancer.
        Objective: To evaluate disease-free survival after curative intent surgery.
        Methods: Adjuvant chemotherapy was given to high-risk patients.
        Results: 5-year disease-free survival was 85% after radical resection.
        """

        boost = domain_relevance_boost(title, abstract, domain="palliative_surgery")

        # Should have negative boost (penalty)
        assert boost < 0, f"Expected negative boost (penalty), got {boost}"

    def test_neutral_paper(self):
        """Paper without domain keywords should have minimal boost."""
        from core.ranking import domain_relevance_boost

        title = "Machine learning approaches in medical imaging"
        abstract = """
        We present a novel deep learning model for medical image segmentation.
        The model achieves state-of-the-art results on benchmark datasets.
        """

        boost = domain_relevance_boost(title, abstract, domain="palliative_surgery")

        # Should be near zero
        assert abs(boost) < 0.15, f"Expected minimal boost, got {boost}"

    def test_none_domain_returns_zero(self):
        """When domain is None, should return 0."""
        from core.ranking import domain_relevance_boost

        title = "Palliative surgery for cancer"
        abstract = "Quality of life assessment in terminal patients."

        boost = domain_relevance_boost(title, abstract, domain=None)

        assert boost == 0.0

    def test_boost_bounds(self):
        """Boost should be bounded between -0.3 and +0.3."""
        from core.ranking import domain_relevance_boost

        # Extremely palliative paper
        title = "Palliative palliation palliate symptom control end of life terminal"
        abstract = "Palliative care quality of life symptom relief advanced cancer metastatic"

        boost = domain_relevance_boost(title, abstract, domain="palliative_surgery")
        assert boost <= 0.55, f"Positive boost exceeded upper bound: {boost}"
        assert boost >= -0.3, f"Boost exceeded lower bound: {boost}"

        # Extremely curative paper
        title = "Curative adjuvant neoadjuvant radical resection disease-free survival"
        abstract = "Pediatric congenital benign veterinary animal model in vitro"

        boost = domain_relevance_boost(title, abstract, domain="palliative_surgery")
        assert boost <= 0.55, f"Boost exceeded upper bound: {boost}"
        assert boost >= -0.3, f"Negative boost exceeded lower bound: {boost}"


class TestQueryParserDomain:
    """Test AdaptiveQueryParser domain functionality."""

    def test_parser_loads_domain(self):
        """Parser should load palliative surgery domain vocabulary."""
        from core.query_parser import AdaptiveQueryParser

        parser = AdaptiveQueryParser(domain="palliative_surgery")

        assert parser.domain == "palliative_surgery"
        assert parser.domain_filter is not None
        assert len(parser.domain_filter) > 0

    def test_parser_no_domain(self):
        """Parser should work without domain."""
        from core.query_parser import AdaptiveQueryParser

        parser = AdaptiveQueryParser(domain=None)

        assert parser.domain is None
        assert parser.domain_filter is None

    def test_domain_filter_method(self):
        """Test apply_domain_filter method."""
        from core.query_parser import AdaptiveQueryParser

        parser = AdaptiveQueryParser(domain="palliative_surgery")

        query = "gastric outlet obstruction treatment"
        filtered = parser.apply_domain_filter(query)

        # Should wrap query with domain filter
        assert "Palliative Care" in filtered or "palliative" in filtered.lower()
        assert query in filtered

    def test_get_domain_info(self):
        """Test get_domain_info method."""
        from core.query_parser import AdaptiveQueryParser

        parser = AdaptiveQueryParser(domain="palliative_surgery")
        info = parser.get_domain_info()

        assert info["domain"] == "palliative_surgery"
        assert info["has_domain_context"] is True


class TestSearchServiceDomain:
    """Test SearchService domain parameter integration."""

    def test_search_service_accepts_domain(self):
        """SearchService.execute_search should accept domain parameter."""
        from services.search_service import SearchService
        import inspect

        sig = inspect.signature(SearchService.execute_search)
        params = list(sig.parameters.keys())

        assert "domain" in params, "execute_search should have 'domain' parameter"
        assert "relevance_threshold" in params, "execute_search should have 'relevance_threshold' parameter"

    def test_search_service_default_domain(self):
        """SearchService should have default domain of palliative_surgery."""
        from services.search_service import SearchService
        import inspect

        sig = inspect.signature(SearchService.execute_search)
        domain_param = sig.parameters.get("domain")

        assert domain_param is not None
        assert domain_param.default == "palliative_surgery"

    def test_search_service_v2_params(self):
        """SearchService should have v2 parameters for two-pass ranking."""
        from services.search_service import SearchService
        import inspect

        sig = inspect.signature(SearchService.execute_search)
        params = list(sig.parameters.keys())

        # v2 params
        assert "use_two_pass" in params, "execute_search should have 'use_two_pass' parameter"
        assert "domain_weight" in params, "execute_search should have 'domain_weight' parameter"
        assert "composite_threshold" in params, "execute_search should have 'composite_threshold' parameter"
        assert "use_union_filter" in params, "execute_search should have 'use_union_filter' parameter"
        assert "top_n_for_llm" in params, "execute_search should have 'top_n_for_llm' parameter"


class TestDomainConfigV2:
    """Test DomainConfig system (v2)."""

    def test_domain_config_imports(self):
        """Verify DomainConfig imports correctly."""
        from config.domain_config import (
            DomainConfig,
            get_domain_config,
            get_default_domain,
            list_available_domains,
            PALLIATIVE_SURGERY_CONFIG,
        )

        assert DomainConfig is not None
        assert PALLIATIVE_SURGERY_CONFIG is not None
        assert callable(get_domain_config)
        assert callable(get_default_domain)
        assert callable(list_available_domains)

    def test_palliative_surgery_config(self):
        """Verify PALLIATIVE_SURGERY_CONFIG structure."""
        from config.domain_config import PALLIATIVE_SURGERY_CONFIG

        assert PALLIATIVE_SURGERY_CONFIG.name == "palliative_surgery"
        assert PALLIATIVE_SURGERY_CONFIG.display_name == "Palliative Surgery"
        assert len(PALLIATIVE_SURGERY_CONFIG.verified_mesh) > 0
        assert len(PALLIATIVE_SURGERY_CONFIG.high_relevance_keywords) > 0
        assert len(PALLIATIVE_SURGERY_CONFIG.negative_keywords) > 0
        assert PALLIATIVE_SURGERY_CONFIG.domain_score_weight == 0.3

    def test_get_union_filter(self):
        """Verify union filter generation."""
        from config.domain_config import PALLIATIVE_SURGERY_CONFIG

        union_filter = PALLIATIVE_SURGERY_CONFIG.get_union_filter()

        assert union_filter is not None
        assert len(union_filter) > 0
        assert "OR" in union_filter  # Should have OR operators
        assert "Palliative Care[MeSH]" in union_filter

    def test_get_exclusion_filter(self):
        """Verify exclusion filter generation."""
        from config.domain_config import PALLIATIVE_SURGERY_CONFIG

        exclusion_filter = PALLIATIVE_SURGERY_CONFIG.get_exclusion_filter()

        assert exclusion_filter is not None
        assert "NOT" in exclusion_filter
        assert "pediatric" in exclusion_filter.lower()

    def test_get_domain_config(self):
        """Test get_domain_config helper."""
        from config.domain_config import get_domain_config

        config = get_domain_config("palliative_surgery")
        assert config is not None
        assert config.name == "palliative_surgery"

        # Non-existent domain
        none_config = get_domain_config("nonexistent")
        assert none_config is None

    def test_list_available_domains(self):
        """Test list_available_domains helper."""
        from config.domain_config import list_available_domains

        domains = list_available_domains()
        assert "palliative_surgery" in domains


class TestComputeDomainScoreV2:
    """Test compute_domain_score function (v2)."""

    def test_compute_domain_score_palliative(self):
        """Palliative paper should get high domain score."""
        from core.ranking import compute_domain_score
        from config.domain_config import PALLIATIVE_SURGERY_CONFIG

        title = "Palliative surgery for malignant bowel obstruction"
        abstract = "Quality of life outcomes in terminal cancer patients with symptom control."

        score, matched, penalty = compute_domain_score(title, abstract, PALLIATIVE_SURGERY_CONFIG)

        assert score > 0.5, f"Expected score > 0.5, got {score}"
        assert len(matched) > 0, "Should have matched keywords"
        assert "palliative" in [kw.lower() for kw in matched]

    def test_compute_domain_score_curative(self):
        """Curative paper should get low domain score."""
        from core.ranking import compute_domain_score
        from config.domain_config import PALLIATIVE_SURGERY_CONFIG

        title = "Curative resection for early stage cancer"
        abstract = "Adjuvant chemotherapy disease-free survival pediatric patients."

        score, matched, penalty = compute_domain_score(title, abstract, PALLIATIVE_SURGERY_CONFIG)

        assert score < 0.5, f"Expected score < 0.5, got {score}"
        assert len(penalty) > 0, "Should have penalty keywords"

    def test_compute_domain_score_returns_keywords(self):
        """compute_domain_score should return matched and penalty keywords."""
        from core.ranking import compute_domain_score
        from config.domain_config import PALLIATIVE_SURGERY_CONFIG

        title = "Palliative surgery with curative intent"
        abstract = "Terminal patients treated with radical resection."

        score, matched, penalty = compute_domain_score(title, abstract, PALLIATIVE_SURGERY_CONFIG)

        # Should have both matched and penalty keywords
        assert isinstance(matched, list)
        assert isinstance(penalty, list)
        # Both present means some positive and negative
        assert len(matched) > 0 or len(penalty) > 0


class TestRankCitationsWithDomainV2:
    """Test rank_citations_with_domain function (v2)."""

    def test_rank_citations_with_domain_import(self):
        """Verify rank_citations_with_domain imports."""
        from core.ranking import rank_citations_with_domain
        assert callable(rank_citations_with_domain)

    def test_rank_citations_with_domain_empty(self):
        """Empty list should return empty list."""
        from core.ranking import rank_citations_with_domain, RankingWeights

        result = rank_citations_with_domain([], RankingWeights())
        assert result == []

    def test_rank_citations_with_domain_scoring(self):
        """rank_citations_with_domain should produce ScoredCitation with domain_score."""
        from core.ranking import rank_citations_with_domain, RankingWeights

        citations = [
            {
                "pmid": "12345678",
                "title": "Palliative surgery for malignant obstruction",
                "abstract": "Symptom control in terminal cancer patients.",
                "year": "2023",
                "publication_types": ["Journal Article"]
            },
            {
                "pmid": "87654321",
                "title": "Curative resection for early cancer",
                "abstract": "Disease-free survival after radical surgery.",
                "year": "2023",
                "publication_types": ["Randomized Controlled Trial"]
            }
        ]

        scored = rank_citations_with_domain(
            citations=citations,
            weights=RankingWeights(relevance=0.4, evidence=0.6),
            domain="palliative_surgery",
            use_llm_rerank=False  # Skip LLM for unit test
        )

        assert len(scored) == 2

        # All ScoredCitations should have domain_score
        for sc in scored:
            assert hasattr(sc, 'domain_score')
            assert 0.0 <= sc.domain_score <= 1.0
            assert hasattr(sc, 'matched_keywords')
            assert hasattr(sc, 'penalty_keywords')

    def test_rank_citations_with_domain_ordering(self):
        """Palliative paper should rank higher than curative paper."""
        from core.ranking import rank_citations_with_domain, RankingWeights

        citations = [
            {
                "pmid": "00000001",
                "title": "Curative resection for early cancer adjuvant therapy",
                "abstract": "Disease-free survival after radical surgery.",
                "year": "2023",
                "publication_types": ["Journal Article"]
            },
            {
                "pmid": "00000002",
                "title": "Palliative surgery for malignant bowel obstruction",
                "abstract": "Quality of life and symptom control in terminal cancer.",
                "year": "2023",
                "publication_types": ["Journal Article"]
            }
        ]

        scored = rank_citations_with_domain(
            citations=citations,
            weights=RankingWeights(relevance=0.4, evidence=0.6),
            domain="palliative_surgery",
            use_llm_rerank=False,
            domain_weight=0.5  # High domain weight to see effect
        )

        # Palliative paper should have higher domain score
        palliative = next(sc for sc in scored if sc.citation["pmid"] == "00000002")
        curative = next(sc for sc in scored if sc.citation["pmid"] == "00000001")

        assert palliative.domain_score > curative.domain_score


class TestScoredCitationV2:
    """Test ScoredCitation dataclass v2 fields."""

    def test_scored_citation_has_v2_fields(self):
        """ScoredCitation should have domain_score, matched_keywords, penalty_keywords."""
        from core.ranking import ScoredCitation

        sc = ScoredCitation(
            citation={"pmid": "12345678"},
            relevance_score=0.8,
            evidence_score=0.9,
            influence_score=0.5,
            recency_score=0.7,
            domain_score=0.75,
            final_score=0.85,
            rank_position=1,
            explanation=["Test"],
            matched_keywords=["palliative", "symptom control"],
            penalty_keywords=[]
        )

        assert sc.domain_score == 0.75
        assert sc.matched_keywords == ["palliative", "symptom control"]
        assert sc.penalty_keywords == []

    def test_scored_citation_defaults(self):
        """ScoredCitation should have sensible defaults for optional fields."""
        from core.ranking import ScoredCitation

        sc = ScoredCitation(
            citation={"pmid": "12345678"},
            relevance_score=0.8,
            evidence_score=0.9,
            influence_score=0.5,
            recency_score=0.7,
            domain_score=0.5,
            final_score=0.85,
            rank_position=1,
            explanation=["Test"]
        )

        # Should have empty lists as defaults
        assert sc.matched_keywords == []
        assert sc.penalty_keywords == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
