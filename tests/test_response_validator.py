"""
Unit tests for gdg/response_validator.py

Tests GDG response validation including:
- PMID extraction and validation
- Epistemic tag detection
- Hallucination detection
- Compliance scoring
"""

import pytest

from gdg.response_validator import (
    ResponseValidator,
    ValidationResult,
    validate_all_responses,
    get_compliance_summary,
    check_claims_have_citations,
)


class TestResponseValidator:
    """Tests for ResponseValidator class."""

    def test_create_without_corpus(self):
        """Test creating validator without corpus."""
        validator = ResponseValidator()
        assert validator.corpus_pmids == set()

    def test_create_with_corpus(self):
        """Test creating validator with corpus."""
        corpus = {"12345678", "87654321"}
        validator = ResponseValidator(corpus_pmids=corpus)
        assert len(validator.corpus_pmids) == 2

    def test_extract_pmids_standard_format(self):
        """Test extracting PMIDs in standard format."""
        validator = ResponseValidator()
        text = "As shown in EVIDENCE (PMID: 12345678), the results were..."
        pmids = validator._extract_pmids(text)
        assert "12345678" in pmids

    def test_extract_pmids_multiple(self):
        """Test extracting multiple PMIDs."""
        validator = ResponseValidator()
        text = "Studies (PMID: 11111111) and (PMID: 22222222) showed..."
        pmids = validator._extract_pmids(text)
        assert len(pmids) == 2
        assert "11111111" in pmids
        assert "22222222" in pmids

    def test_extract_pmids_case_insensitive(self):
        """Test case-insensitive PMID extraction."""
        validator = ResponseValidator()
        text = "See pmid: 12345678 and Pmid:87654321"
        pmids = validator._extract_pmids(text)
        assert len(pmids) == 2

    def test_count_epistemic_tags_evidence(self):
        """Test counting EVIDENCE tags."""
        validator = ResponseValidator()
        text = "EVIDENCE (PMID: 12345678) shows 30% response rate. EVIDENCE (PMID: 87654321) confirms this."
        counts = validator._count_epistemic_tags(text)
        assert counts["EVIDENCE"] == 2

    def test_count_epistemic_tags_assumption(self):
        """Test counting ASSUMPTION tags."""
        validator = ResponseValidator()
        text = "ASSUMPTION: Patients with good PS would tolerate surgery. ASSUMPTION: Similar results expected."
        counts = validator._count_epistemic_tags(text)
        assert counts["ASSUMPTION"] == 2

    def test_count_epistemic_tags_opinion(self):
        """Test counting OPINION tags."""
        validator = ResponseValidator()
        text = "OPINION: Surgery should be considered. OPINION: Medical management is preferable."
        counts = validator._count_epistemic_tags(text)
        assert counts["OPINION"] == 2

    def test_count_epistemic_tags_evidence_gap(self):
        """Test counting EVIDENCE GAP tags."""
        validator = ResponseValidator()
        text = "EVIDENCE GAP → No RCTs compare these approaches. EVIDENCE GAP → QoL data lacking."
        counts = validator._count_epistemic_tags(text)
        assert counts["EVIDENCE GAP"] == 2

    def test_count_bracketed_tags(self):
        """Test counting bracketed style tags."""
        validator = ResponseValidator()
        text = "[EVIDENCE] Response rate was 30%. [OPINION] This is promising."
        counts = validator._count_epistemic_tags(text)
        assert counts["EVIDENCE"] >= 1
        assert counts["OPINION"] >= 1


class TestValidation:
    """Tests for validate method."""

    def test_validate_valid_response(self):
        """Test validating a well-formed response."""
        corpus = {"12345678", "87654321"}
        validator = ResponseValidator(corpus_pmids=corpus)

        response = """
        Based on the evidence, surgery shows benefit for selected patients.

        EVIDENCE (PMID: 12345678): A retrospective study of 150 patients showed
        30-day mortality of 8% and symptom relief in 75% of cases.

        EVIDENCE (PMID: 87654321): A prospective cohort found median survival
        of 4.2 months post-surgery.

        ASSUMPTION: These results would apply to our patient population.

        OPINION: Given limited prognosis, goals of care discussion essential.

        EVIDENCE GAP → No RCTs comparing surgery to medical management exist.
        """

        result = validator.validate(response)
        assert result.is_valid is True
        assert result.compliance_score >= 0.5
        assert len(result.cited_pmids) == 2
        assert len(result.hallucinated_pmids) == 0

    def test_validate_detects_hallucinations(self):
        """Test that hallucinated PMIDs are detected."""
        corpus = {"12345678"}  # Only one valid PMID
        validator = ResponseValidator(corpus_pmids=corpus)

        response = """
        EVIDENCE (PMID: 12345678): Valid citation.
        EVIDENCE (PMID: 99999999): This PMID is not in corpus!
        """

        result = validator.validate(response)
        assert result.is_valid is False
        assert "99999999" in result.hallucinated_pmids
        assert "12345678" not in result.hallucinated_pmids
        assert len(result.errors) > 0

    def test_validate_warns_no_tags(self):
        """Test warning when no epistemic tags found."""
        validator = ResponseValidator()

        response = "Surgery may benefit some patients. Consider medical management."

        result = validator.validate(response)
        assert any("epistemic" in w.lower() for w in result.warnings)

    def test_validate_warns_short_response(self):
        """Test warning for short responses."""
        validator = ResponseValidator()
        response = "Surgery is an option."
        result = validator.validate(response)
        assert any("short" in w.lower() for w in result.warnings)

    def test_validate_warns_long_response(self):
        """Test warning for very long responses."""
        validator = ResponseValidator()
        response = "Word " * 3500  # Over 3000 words
        result = validator.validate(response)
        assert any("long" in w.lower() for w in result.warnings)


class TestUntaggedClaims:
    """Tests for untagged claim detection."""

    def test_detect_untagged_percentage(self):
        """Test detecting untagged percentage claims."""
        validator = ResponseValidator()
        text = "The response rate was 45%. Surgery showed 30% mortality."
        untagged = validator._find_untagged_claims(text)
        assert len(untagged) > 0

    def test_detect_untagged_statistics(self):
        """Test detecting untagged statistical claims."""
        validator = ResponseValidator()
        text = "The hazard ratio was HR=0.65 and p<0.05."
        untagged = validator._find_untagged_claims(text)
        assert len(untagged) > 0

    def test_tagged_claims_not_flagged(self):
        """Test that properly tagged claims are not flagged."""
        validator = ResponseValidator()
        text = "EVIDENCE (PMID: 12345678): Response rate was 45%."
        untagged = validator._find_untagged_claims(text)
        assert len(untagged) == 0


class TestComplianceScore:
    """Tests for compliance score calculation."""

    def test_perfect_score(self):
        """Test achieving high compliance score."""
        corpus = {"12345678", "87654321"}
        validator = ResponseValidator(corpus_pmids=corpus)

        response = """
        Based on the evidence, I recommend considering surgery.

        EVIDENCE (PMID: 12345678): The retrospective study showed 30-day
        mortality of 8% with symptom relief in 75% of patients.

        EVIDENCE (PMID: 87654321): Median survival was 4.2 months.

        ASSUMPTION: These results generalize to similar patients.

        OPINION: Goals of care discussion should precede any intervention.
        """ + " Additional text." * 20  # Make it substantial

        result = validator.validate(response)
        assert result.compliance_score >= 0.7

    def test_low_score_for_hallucinations(self):
        """Test low score when hallucinations present."""
        corpus = {"12345678"}
        validator = ResponseValidator(corpus_pmids=corpus)

        response = """
        EVIDENCE (PMID: 99999999): Invalid citation.
        EVIDENCE (PMID: 88888888): Also invalid.
        """ + " Additional text." * 10

        result = validator.validate(response)
        assert result.compliance_score < 0.6

    def test_partial_credit_no_citations(self):
        """Test partial credit when no citations expected."""
        validator = ResponseValidator()  # No corpus
        response = "OPINION: Based on clinical experience, surgery may help." * 5
        result = validator.validate(response)
        assert result.compliance_score > 0


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_result_contains_all_fields(self):
        """Test that ValidationResult has all expected fields."""
        result = ValidationResult(
            is_valid=True,
            compliance_score=0.85,
            errors=[],
            warnings=["Test warning"],
            cited_pmids={"12345678"},
            hallucinated_pmids=set(),
            epistemic_tags_found={"EVIDENCE": 2, "OPINION": 1}
        )
        assert result.is_valid is True
        assert result.compliance_score == 0.85
        assert "12345678" in result.cited_pmids
        assert result.epistemic_tags_found["EVIDENCE"] == 2


class TestValidateAllResponses:
    """Tests for validate_all_responses function."""

    def test_validate_multiple_experts(self):
        """Test validating responses from multiple experts."""
        corpus = {"12345678", "87654321"}
        responses = {
            "Expert A": "EVIDENCE (PMID: 12345678): Study shows benefit." * 10,
            "Expert B": "EVIDENCE (PMID: 87654321): Similar findings." * 10,
            "Expert C": "EVIDENCE (PMID: 99999999): Invalid citation." * 10,
        }

        results = validate_all_responses(responses, corpus)

        assert "Expert A" in results
        assert "Expert B" in results
        assert "Expert C" in results
        assert results["Expert A"].is_valid is True
        assert results["Expert C"].is_valid is False


class TestGetComplianceSummary:
    """Tests for get_compliance_summary function."""

    def test_empty_results(self):
        """Test summary of empty results."""
        summary = get_compliance_summary({})
        assert summary["average_score"] == 0.0
        assert summary["experts_valid"] == 0

    def test_summary_calculations(self):
        """Test summary statistics calculation."""
        results = {
            "Expert A": ValidationResult(
                is_valid=True, compliance_score=0.9, errors=[], warnings=[],
                cited_pmids={"12345678"}, hallucinated_pmids=set(),
                epistemic_tags_found={"EVIDENCE": 2}
            ),
            "Expert B": ValidationResult(
                is_valid=False, compliance_score=0.4, errors=["Error"],
                warnings=[], cited_pmids={"87654321"},
                hallucinated_pmids={"99999999"},
                epistemic_tags_found={"EVIDENCE": 1}
            ),
        }

        summary = get_compliance_summary(results)
        assert summary["average_score"] == 0.65
        assert summary["experts_valid"] == 1
        assert summary["experts_total"] == 2
        assert summary["total_hallucinations"] == 1
        assert summary["total_citations"] == 2


class TestCheckClaimsHaveCitations:
    """Tests for check_claims_have_citations function."""

    def test_all_claims_cited(self):
        """Test when all claims have citations."""
        text = """
        EVIDENCE (PMID: 12345678): Response rate was 45%.
        EVIDENCE (PMID: 87654321): Survival was 4.2 months.
        """
        result = check_claims_have_citations(text)
        assert result["citation_rate"] > 0

    def test_uncited_claims(self):
        """Test detecting uncited claims."""
        text = """
        Response rate was 45% in the treatment arm.
        Median survival was 4.2 months with p=0.03 confidence.
        """
        result = check_claims_have_citations(text)
        # The function uses heuristics to detect claims with numbers
        assert result["claims_found"] >= 1
        assert result["claims_cited"] == 0
        # May or may not find uncited claims depending on sentence parsing
        # The main check is that citation_rate is 0 when no PMIDs present
        assert result["citation_rate"] == 0.0 or result["claims_found"] == 0


class TestFormatValidationSummary:
    """Tests for format_validation_summary method."""

    def test_format_valid_response(self):
        """Test formatting a valid validation result."""
        validator = ResponseValidator()
        result = ValidationResult(
            is_valid=True,
            compliance_score=0.85,
            errors=[],
            warnings=[],
            cited_pmids={"12345678", "87654321"},
            hallucinated_pmids=set(),
            epistemic_tags_found={"EVIDENCE": 2, "OPINION": 1, "ASSUMPTION": 0, "EVIDENCE GAP": 0}
        )

        summary = validator.format_validation_summary(result)
        assert "VALID" in summary
        assert "85%" in summary
        assert "2 PMIDs cited" in summary

    def test_format_invalid_response(self):
        """Test formatting an invalid validation result."""
        validator = ResponseValidator()
        result = ValidationResult(
            is_valid=False,
            compliance_score=0.35,
            errors=["Hallucinated PMIDs not in corpus: 99999999"],
            warnings=["Response is very short"],
            cited_pmids={"99999999"},
            hallucinated_pmids={"99999999"},
            epistemic_tags_found={"EVIDENCE": 1, "OPINION": 0, "ASSUMPTION": 0, "EVIDENCE GAP": 0}
        )

        summary = validator.format_validation_summary(result)
        assert "NEEDS REVIEW" in summary
        assert "1 hallucinated" in summary
        assert "Errors:" in summary
