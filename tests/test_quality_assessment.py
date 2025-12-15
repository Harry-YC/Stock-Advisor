"""
Unit tests for core/quality_assessment.py

Tests GRADE-based quality assessment including:
- Evidence level classification
- Risk of bias assessment
- Certainty calculation
"""

import pytest

from core.quality_assessment import (
    StudyDesign,
    EvidenceLevel,
    RiskOfBias,
    Certainty,
    DESIGN_TO_LEVEL,
    ROB_DOMAINS,
    DomainRating,
    QualityRating,
    get_evidence_level,
    get_rob_domains,
    calculate_overall_rob,
    calculate_certainty,
    create_quality_rating,
    summarize_evidence_quality,
    format_certainty_badge,
)


class TestStudyDesignEnum:
    """Tests for StudyDesign enum."""

    def test_all_designs_exist(self):
        """Test all expected study designs are defined."""
        expected = [
            "Systematic Review", "Meta-Analysis", "RCT",
            "Prospective Cohort", "Retrospective Cohort",
            "Case-Control", "Case Series", "Case Report",
            "Expert Opinion", "Other"
        ]
        actual = [d.value for d in StudyDesign]
        for design in expected:
            assert design in actual


class TestEvidenceLevelEnum:
    """Tests for EvidenceLevel enum."""

    def test_all_levels_exist(self):
        """Test all Oxford CEBM levels are defined."""
        levels = [e.value for e in EvidenceLevel]
        assert "I" in levels
        assert "II" in levels
        assert "III" in levels
        assert "IV" in levels
        assert "V" in levels


class TestDesignToLevelMapping:
    """Tests for DESIGN_TO_LEVEL mapping."""

    def test_sr_maps_to_level_i(self):
        """Systematic reviews map to Level I."""
        assert DESIGN_TO_LEVEL["Systematic Review"] == "I"
        assert DESIGN_TO_LEVEL["Meta-Analysis"] == "I"

    def test_rct_maps_to_level_ii(self):
        """RCTs map to Level II."""
        assert DESIGN_TO_LEVEL["RCT"] == "II"

    def test_cohort_maps_to_level_iii(self):
        """Cohort studies map to Level III."""
        assert DESIGN_TO_LEVEL["Prospective Cohort"] == "III"
        assert DESIGN_TO_LEVEL["Retrospective Cohort"] == "III"

    def test_case_series_maps_to_level_iv(self):
        """Case series/control map to Level IV."""
        assert DESIGN_TO_LEVEL["Case Series"] == "IV"
        assert DESIGN_TO_LEVEL["Case-Control"] == "IV"

    def test_case_report_maps_to_level_v(self):
        """Case reports and expert opinion map to Level V."""
        assert DESIGN_TO_LEVEL["Case Report"] == "V"
        assert DESIGN_TO_LEVEL["Expert Opinion"] == "V"


class TestRobDomains:
    """Tests for ROB_DOMAINS mapping."""

    def test_rct_domains(self):
        """Test RCT has correct ROB domains."""
        domains = [d[0] for d in ROB_DOMAINS["RCT"]]
        assert "randomization" in domains
        assert "missing_data" in domains
        assert "selective_reporting" in domains

    def test_cohort_domains(self):
        """Test cohort studies have confounding domain."""
        domains = [d[0] for d in ROB_DOMAINS["Cohort"]]
        assert "confounding" in domains
        assert "selection" in domains

    def test_case_series_domains(self):
        """Test case series has appropriate domains."""
        domains = [d[0] for d in ROB_DOMAINS["Case Series"]]
        assert "selection" in domains
        assert "causality" in domains


class TestDomainRating:
    """Tests for DomainRating dataclass."""

    def test_create_basic(self):
        """Test creating a domain rating."""
        rating = DomainRating(
            domain_id="randomization",
            domain_name="Randomization process",
            rating="Low"
        )
        assert rating.domain_id == "randomization"
        assert rating.rating == "Low"
        assert rating.rationale == ""

    def test_create_with_rationale(self):
        """Test creating rating with rationale."""
        rating = DomainRating(
            domain_id="randomization",
            domain_name="Randomization process",
            rating="High",
            rationale="No allocation concealment described"
        )
        assert rating.rationale == "No allocation concealment described"


class TestQualityRating:
    """Tests for QualityRating dataclass."""

    def test_create_basic(self):
        """Test creating a quality rating."""
        rating = QualityRating(
            pmid="12345678",
            study_design="RCT",
            evidence_level="II",
            risk_of_bias="Low",
            certainty="High"
        )
        assert rating.pmid == "12345678"
        assert rating.assessed_at != ""

    def test_get_certainty_symbol(self):
        """Test certainty symbol generation."""
        rating = QualityRating(
            pmid="12345678",
            study_design="RCT",
            evidence_level="II",
            risk_of_bias="Low",
            certainty="High"
        )
        assert rating.get_certainty_symbol() == "⊕⊕⊕⊕"

        rating.certainty = "Moderate"
        assert rating.get_certainty_symbol() == "⊕⊕⊕○"

        rating.certainty = "Low"
        assert rating.get_certainty_symbol() == "⊕⊕○○"

        rating.certainty = "Very Low"
        assert rating.get_certainty_symbol() == "⊕○○○"

    def test_get_rob_color(self):
        """Test risk of bias color coding."""
        rating = QualityRating(
            pmid="12345678",
            study_design="RCT",
            evidence_level="II",
            risk_of_bias="Low",
            certainty="High"
        )
        assert rating.get_rob_color() == "#28a745"  # Green

        rating.risk_of_bias = "Critical"
        assert rating.get_rob_color() == "#dc3545"  # Red

    def test_to_dict_and_from_dict(self):
        """Test serialization roundtrip."""
        original = QualityRating(
            pmid="12345678",
            study_design="RCT",
            evidence_level="II",
            risk_of_bias="Low",
            certainty="High",
            domain_ratings=[
                DomainRating("randomization", "Randomization", "Low")
            ]
        )
        d = original.to_dict()
        restored = QualityRating.from_dict(d)
        assert restored.pmid == original.pmid
        assert len(restored.domain_ratings) == 1


class TestGetEvidenceLevel:
    """Tests for get_evidence_level function."""

    def test_systematic_review(self):
        """Test systematic review detection."""
        assert get_evidence_level("Systematic Review") == "I"
        assert get_evidence_level("systematic review with meta-analysis") == "I"
        assert get_evidence_level("Meta-Analysis") == "I"

    def test_rct(self):
        """Test RCT detection."""
        assert get_evidence_level("RCT") == "II"
        assert get_evidence_level("Randomized Controlled Trial") == "II"
        assert get_evidence_level("randomized trial") == "II"

    def test_cohort(self):
        """Test cohort study detection."""
        assert get_evidence_level("Cohort Study") == "III"
        assert get_evidence_level("Prospective Cohort") == "III"
        assert get_evidence_level("retrospective cohort analysis") == "III"

    def test_case_control(self):
        """Test case-control detection."""
        assert get_evidence_level("Case-Control Study") == "IV"
        assert get_evidence_level("case control") == "IV"

    def test_case_series(self):
        """Test case series/report detection."""
        assert get_evidence_level("Case Series") == "IV"
        assert get_evidence_level("Case Report") == "IV"

    def test_unknown_defaults_to_v(self):
        """Test unknown designs default to Level V."""
        assert get_evidence_level("Unknown") == "V"
        assert get_evidence_level("") == "V"


class TestGetRobDomains:
    """Tests for get_rob_domains function."""

    def test_rct_domains(self):
        """Test RCT gets correct domains."""
        domains = get_rob_domains("RCT")
        domain_ids = [d[0] for d in domains]
        assert "randomization" in domain_ids

    def test_cohort_domains(self):
        """Test cohort gets correct domains."""
        domains = get_rob_domains("Prospective Cohort Study")
        domain_ids = [d[0] for d in domains]
        assert "confounding" in domain_ids

    def test_unknown_defaults_to_case_series(self):
        """Test unknown design gets case series domains."""
        domains = get_rob_domains("Unknown Study Type")
        assert len(domains) > 0


class TestCalculateOverallRob:
    """Tests for calculate_overall_rob function."""

    def test_all_low(self):
        """Test all low ratings result in low overall."""
        ratings = [
            DomainRating("d1", "Domain 1", "Low"),
            DomainRating("d2", "Domain 2", "Low"),
            DomainRating("d3", "Domain 3", "Low"),
        ]
        assert calculate_overall_rob(ratings) == "Low"

    def test_one_concern(self):
        """Test one concern results in some concerns overall."""
        ratings = [
            DomainRating("d1", "Domain 1", "Low"),
            DomainRating("d2", "Domain 2", "Some Concerns"),
            DomainRating("d3", "Domain 3", "Low"),
        ]
        assert calculate_overall_rob(ratings) == "Some Concerns"

    def test_two_concerns_equals_high(self):
        """Test two concerns results in high overall."""
        ratings = [
            DomainRating("d1", "Domain 1", "Some Concerns"),
            DomainRating("d2", "Domain 2", "Some Concerns"),
            DomainRating("d3", "Domain 3", "Low"),
        ]
        assert calculate_overall_rob(ratings) == "High"

    def test_any_high_equals_high(self):
        """Test any high rating results in high overall."""
        ratings = [
            DomainRating("d1", "Domain 1", "Low"),
            DomainRating("d2", "Domain 2", "High"),
            DomainRating("d3", "Domain 3", "Low"),
        ]
        assert calculate_overall_rob(ratings) == "High"

    def test_any_critical_equals_critical(self):
        """Test any critical rating results in critical overall."""
        ratings = [
            DomainRating("d1", "Domain 1", "Low"),
            DomainRating("d2", "Domain 2", "Critical"),
        ]
        assert calculate_overall_rob(ratings) == "Critical"

    def test_empty_defaults_to_high(self):
        """Test empty ratings default to high."""
        assert calculate_overall_rob([]) == "High"


class TestCalculateCertainty:
    """Tests for calculate_certainty function."""

    def test_rct_starts_high(self):
        """Test RCT with low ROB starts at high."""
        certainty = calculate_certainty(
            study_design="RCT",
            risk_of_bias="Low",
            downgrade_reasons=[]
        )
        assert certainty == "High"

    def test_observational_starts_low(self):
        """Test observational study starts at low."""
        certainty = calculate_certainty(
            study_design="Cohort",
            risk_of_bias="Low",
            downgrade_reasons=[]
        )
        assert certainty == "Low"

    def test_downgrade_for_rob(self):
        """Test downgrade for risk of bias."""
        certainty = calculate_certainty(
            study_design="RCT",
            risk_of_bias="High",
            downgrade_reasons=[]
        )
        assert certainty in ["Moderate", "Low"]

    def test_explicit_downgrades(self):
        """Test explicit downgrade reasons."""
        certainty = calculate_certainty(
            study_design="RCT",
            risk_of_bias="Low",
            downgrade_reasons=["imprecision", "indirectness"]
        )
        assert certainty in ["Low", "Very Low"]

    def test_upgrades_for_observational(self):
        """Test upgrades for observational studies."""
        certainty = calculate_certainty(
            study_design="Cohort",
            risk_of_bias="Low",
            downgrade_reasons=[],
            upgrade_reasons=["large_effect"]
        )
        assert certainty == "Moderate"

    def test_max_two_upgrades(self):
        """Test that only 2 upgrades are applied."""
        certainty = calculate_certainty(
            study_design="Cohort",
            risk_of_bias="Low",
            downgrade_reasons=[],
            upgrade_reasons=["large_effect", "dose_response", "confounding"]
        )
        # Should be High (Low + 2 upgrades)
        assert certainty == "High"


class TestCreateQualityRating:
    """Tests for create_quality_rating function."""

    def test_create_basic(self):
        """Test creating a basic quality rating."""
        rating = create_quality_rating(
            pmid="12345678",
            study_design="RCT"
        )
        assert rating.pmid == "12345678"
        assert rating.evidence_level == "II"
        assert rating.assessed_by == "ai"

    def test_create_with_assessments(self):
        """Test creating rating with domain assessments."""
        rating = create_quality_rating(
            pmid="12345678",
            study_design="RCT",
            domain_assessments={
                "randomization": "Low",
                "deviations": "Low",
                "missing_data": "Some Concerns",
                "outcome_measurement": "Low",
                "selective_reporting": "Low"
            }
        )
        # With 1 Some Concerns and 4 Low, overall is Some Concerns
        assert rating.risk_of_bias == "Some Concerns"
        assert len(rating.domain_ratings) > 0

    def test_create_with_downgrades(self):
        """Test creating rating with downgrade reasons."""
        rating = create_quality_rating(
            pmid="12345678",
            study_design="RCT",
            downgrade_reasons=["imprecision"]
        )
        assert "imprecision" in rating.downgrade_reasons
        assert rating.certainty != "High"


class TestSummarizeEvidenceQuality:
    """Tests for summarize_evidence_quality function."""

    def test_empty_list(self):
        """Test summarizing empty list."""
        summary = summarize_evidence_quality([])
        assert summary["total_studies"] == 0
        assert summary["overall_certainty"] == "Very Low"

    def test_multiple_studies(self):
        """Test summarizing multiple studies."""
        ratings = [
            QualityRating(pmid="1", study_design="RCT", evidence_level="II",
                         risk_of_bias="Low", certainty="High"),
            QualityRating(pmid="2", study_design="Cohort", evidence_level="III",
                         risk_of_bias="Some Concerns", certainty="Low"),
        ]
        summary = summarize_evidence_quality(ratings)
        assert summary["total_studies"] == 2
        assert summary["by_certainty"]["High"] == 1
        assert summary["by_certainty"]["Low"] == 1
        assert summary["overall_certainty"] == "High"


class TestFormatCertaintyBadge:
    """Tests for format_certainty_badge function."""

    def test_high_certainty(self):
        """Test high certainty badge."""
        badge = format_certainty_badge("High")
        assert "⊕⊕⊕⊕" in badge
        assert "#28a745" in badge  # Green

    def test_low_certainty(self):
        """Test low certainty badge."""
        badge = format_certainty_badge("Low")
        assert "⊕⊕○○" in badge
        assert "#ffc107" in badge  # Yellow/gold

    def test_very_low_certainty(self):
        """Test very low certainty badge."""
        badge = format_certainty_badge("Very Low")
        assert "⊕○○○" in badge
        assert "#dc3545" in badge  # Red
