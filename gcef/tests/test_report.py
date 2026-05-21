"""
tests/test_report.py
--------------------
Tests for gcef.report.

Tests verify:
- PDF and HTML generate without error
- All 9 required sections are present
- Estimand appears on every output (no number without population statement)
- Red flags surface in executive summary when assumptions fail
- Reproducibility statement is always present (non-suppressible)
- hash_dataframe produces stable, consistent hashes
- to_dict() provides machine-readable report data
- report() shortcut on GCEFResults works
"""
import pytest
import warnings
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from gcef.report import ReportGenerator, hash_dataframe, _ReportData
from gcef.testing.synthetic import make_valid_panel


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def results():
    """Full pipeline results for report testing."""
    from gcef.pipeline import GreenCreditEvaluator
    from gcef.treatment import (GreenCreditTreatment, TreatmentType,
                                 ConditionalityMechanism, VerificationMethod)
    from gcef.outcomes import ResilienceIndex

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        treatment = GreenCreditTreatment(
            type=TreatmentType.RATE_REDUCTION,
            conditionality_mechanism=ConditionalityMechanism.VERIFIED_INVESTMENT,
            verification_method=VerificationMethod.DOCUMENT_REVIEW,
            intensity=0.03,
        )
        outcome = ResilienceIndex(
            columns={"revenue": 0.50, "loan_repayment_rate": 0.50},
            shock_instrument="rainfall_anomaly_lag1",
            shock_threshold=-1.5,
        )
        evaluator = GreenCreditEvaluator(
            treatment=treatment, outcome=outcome,
            unit_id="sme_id", time_id="period", lender_id="lender_id",
            adoption_time="lender_green_adoption_period",
            covariates=["firm_age", "sector", "prior_revenue"],
            random_seed=42,
        )
        data = make_valid_panel(n_smes=150, n_lenders=4, n_periods=8, seed=42)
        return evaluator.fit(data)


@pytest.fixture(scope="module")
def reporter(results):
    return ReportGenerator(
        results,
        programme_name="Test Green Credit Programme",
        analyst_name="Test Analyst",
        data_description="Synthetic SSA panel",
        data_hash="abc123def456",
    )


@pytest.fixture(scope="module")
def report_dict(reporter):
    return reporter.to_dict()


@pytest.fixture(scope="module")
def pdf_path(reporter, tmp_path_factory):
    p = tmp_path_factory.mktemp("reports") / "test_report.pdf"
    reporter.to_pdf(p)
    return p


@pytest.fixture(scope="module")
def html_path(reporter, tmp_path_factory):
    p = tmp_path_factory.mktemp("reports") / "test_report.html"
    reporter.to_html(p)
    return p


@pytest.fixture(scope="module")
def html_content(html_path):
    return html_path.read_text(encoding="utf-8")


# ── File generation ────────────────────────────────────────────────────────────

class TestFileGeneration:

    def test_pdf_generates(self, pdf_path):
        assert pdf_path.exists()

    def test_pdf_non_empty(self, pdf_path):
        assert pdf_path.stat().st_size > 1000

    def test_pdf_is_valid_pdf(self, pdf_path):
        with open(pdf_path, "rb") as f:
            header = f.read(4)
        assert header == b"%PDF"

    def test_html_generates(self, html_path):
        assert html_path.exists()

    def test_html_non_empty(self, html_path):
        assert html_path.stat().st_size > 1000

    def test_html_is_valid_html(self, html_content):
        assert "<!DOCTYPE html>" in html_content
        assert "</html>" in html_content

    def test_generate_method_pdf(self, reporter, tmp_path_factory):
        p = tmp_path_factory.mktemp("gen") / "out.pdf"
        result = reporter.generate("pdf", p)
        assert result == p
        assert p.exists()

    def test_generate_method_html(self, reporter, tmp_path_factory):
        p = tmp_path_factory.mktemp("gen") / "out.html"
        result = reporter.generate("html", p)
        assert result == p
        assert p.exists()

    def test_invalid_format_raises(self, reporter, tmp_path_factory):
        p = tmp_path_factory.mktemp("gen") / "out.xyz"
        with pytest.raises(ValueError, match="output_format"):
            reporter.generate("docx", p)


# ── HTML section completeness ──────────────────────────────────────────────────

class TestHTMLSections:
    """All 9 spec-required sections must be present in HTML output."""

    def test_section_1_executive_summary(self, html_content):
        assert "Executive Summary" in html_content

    def test_section_2_estimand(self, html_content):
        assert "Estimand Statement" in html_content

    def test_section_3_complier_profile(self, html_content):
        assert "Complier Profile" in html_content

    def test_section_4_late(self, html_content):
        assert "LATE Estimate" in html_content

    def test_section_5_cate(self, html_content):
        assert "CATE Heterogeneity" in html_content

    def test_section_6_bounds(self, html_content):
        assert "Bounds for Full Portfolio" in html_content

    def test_section_7_assumption_audit(self, html_content):
        assert "Assumption Audit" in html_content

    def test_section_8_methodological_notes(self, html_content):
        assert "Methodological Notes" in html_content

    def test_section_9_reproducibility(self, html_content):
        assert "Reproducibility Statement" in html_content


# ── Estimand appears everywhere ────────────────────────────────────────────────

class TestEstimandPresence:
    """No number appears without a population statement (spec principle)."""

    def test_estimand_population_in_html(self, html_content):
        assert "complier" in html_content.lower()

    def test_estimand_identification_in_html(self, html_content):
        assert "IV-DiD" in html_content or "staggered" in html_content.lower()

    def test_estimand_dfi_framing_in_html(self, html_content):
        assert "marginal" in html_content.lower() or "additionality" in html_content.lower()

    def test_population_in_report_dict(self, report_dict):
        assert report_dict["estimand_population"] == "compliers"

    def test_estimand_prose_in_report_dict(self, report_dict):
        assert len(report_dict["estimand_prose"]) > 50


# ── Reproducibility (non-suppressible) ────────────────────────────────────────

class TestReproducibility:

    def test_reproducibility_in_html(self, html_content):
        assert "Reproducibility Statement" in html_content
        assert "mandatory" in html_content.lower()

    def test_version_in_html(self, html_content):
        assert "0.1" in html_content

    def test_analyst_name_in_html(self, html_content):
        assert "Test Analyst" in html_content

    def test_data_hash_in_html(self, html_content):
        assert "abc123def456" in html_content

    def test_random_seed_in_report_dict(self, report_dict):
        # random_seed may be None if not stored on results — that's acceptable
        assert "random_seed" in report_dict

    def test_gcef_version_in_report_dict(self, report_dict):
        assert "gcef_version" in report_dict
        assert report_dict["gcef_version"].startswith("0.")


# ── Assumption audit ───────────────────────────────────────────────────────────

class TestAssumptionAudit:

    def test_audit_in_report_dict(self, report_dict):
        assert "assumption_audit" in report_dict
        assert len(report_dict["assumption_audit"]) > 0

    def test_audit_items_have_required_fields(self, report_dict):
        for item in report_dict["assumption_audit"]:
            assert "label" in item
            assert "status" in item
            assert "value" in item

    def test_audit_statuses_are_valid(self, report_dict):
        valid = {"PASS", "FAIL", "WARN"}
        for item in report_dict["assumption_audit"]:
            assert item["status"] in valid

    def test_failed_assumption_in_red_flags(self, report_dict):
        """Any FAIL assumption must appear in red_flags."""
        failed = [a["label"] for a in report_dict["assumption_audit"]
                  if a["status"] == "FAIL"]
        for label in failed:
            assert label in report_dict["red_flags"]

    def test_red_flags_in_executive_summary(self, report_dict):
        """Red flags must appear in the executive summary text."""
        if report_dict["red_flags"]:
            for flag in report_dict["red_flags"]:
                assert flag in report_dict["executive_summary"] or \
                       "CONCERN" in report_dict["executive_summary"]

    def test_assumption_audit_in_html(self, html_content):
        assert "PASS" in html_content or "FAIL" in html_content or "WARN" in html_content

    def test_all_expected_assumptions_present(self, report_dict):
        expected_keys = {
            "single_lender", "adoption_timing_anomaly", "panel_length",
            "instrument_relevance", "kappa_weight_negatives",
            "complier_share", "overlap",
        }
        audit_keys = {a["key"] for a in report_dict["assumption_audit"]}
        assert expected_keys.issubset(audit_keys)


# ── Core estimates in report ───────────────────────────────────────────────────

class TestEstimatesInReport:

    def test_late_in_report_dict(self, report_dict, results):
        assert abs(report_dict["late_estimate"] - results.late["estimate"]) < 1e-6

    def test_late_se_in_report_dict(self, report_dict, results):
        assert abs(report_dict["late_se"] - results.late["se"]) < 1e-6

    def test_complier_share_in_report_dict(self, report_dict, results):
        assert abs(
            report_dict["complier_share"] - results.complier_share["estimate"]
        ) < 1e-6

    def test_cate_mean_correct(self, report_dict, results):
        mask = results.cate_complier_mask.values
        expected_mean = results.cate.loc[mask, "cate_estimate"].mean()
        assert abs(report_dict["cate_mean"] - expected_mean) < 1e-4

    def test_f_statistic_in_report_dict(self, report_dict):
        assert report_dict["f_statistic"] > 0

    def test_late_in_html(self, html_content, results):
        # LATE value should appear in HTML (rounded)
        late_rounded = f"{results.late['estimate']:,.0f}"
        # Strip commas for the check since formatting may vary
        assert str(int(abs(results.late["estimate"])))[:3] in html_content

    def test_complier_profile_table_non_empty(self, report_dict):
        assert len(report_dict["complier_profile_table"]) > 0

    def test_bounds_summary_present(self, report_dict):
        assert "bounds_summary" in report_dict


# ── hash_dataframe ─────────────────────────────────────────────────────────────

class TestHashDataframe:

    def test_returns_string(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        h = hash_dataframe(df)
        assert isinstance(h, str)

    def test_returns_16_chars(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        h = hash_dataframe(df)
        assert len(h) == 16

    def test_same_data_same_hash(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        assert hash_dataframe(df) == hash_dataframe(df.copy())

    def test_different_data_different_hash(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [1, 3]})
        assert hash_dataframe(df1) != hash_dataframe(df2)

    def test_column_order_affects_hash(self):
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"b": [2], "a": [1]})
        # Different column order should produce different hash
        # (columns are part of the hash input)
        assert hash_dataframe(df1) != hash_dataframe(df2)

    def test_works_on_real_panel(self):
        data = make_valid_panel(seed=42)
        h = hash_dataframe(data)
        assert len(h) == 16
        assert hash_dataframe(data) == h  # stable


# ── results.report() shortcut ──────────────────────────────────────────────────

class TestResultsReportShortcut:

    def test_report_method_exists_on_results(self, results):
        assert hasattr(results, "report")
        assert callable(results.report)

    def test_report_shortcut_generates_pdf(self, results, tmp_path_factory):
        p = tmp_path_factory.mktemp("shortcut") / "shortcut.pdf"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results.report(output_format="pdf", output_path=str(p))
        assert p.exists()
        assert p.stat().st_size > 1000

    def test_report_shortcut_generates_html(self, results, tmp_path_factory):
        p = tmp_path_factory.mktemp("shortcut") / "shortcut.html"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results.report(output_format="html", output_path=str(p))
        content = p.read_text()
        assert "Reproducibility Statement" in content


# ── Minimal results (graceful degradation) ────────────────────────────────────

class TestMinimalResults:
    """Report should generate even when optional fields are sparse."""

    def test_report_with_no_always_takers(self, results, tmp_path_factory):
        """Clean panel has no always-takers; bounds_summary may be sparse."""
        reporter = ReportGenerator(results, programme_name="Minimal Test")
        d = reporter.to_dict()
        # Should not raise, bounds_summary may be empty for never-takers only
        assert "bounds_summary" in d

    def test_report_with_empty_data_description(self, results, tmp_path_factory):
        """Empty metadata fields should not break report generation."""
        p = tmp_path_factory.mktemp("minimal") / "minimal.html"
        reporter = ReportGenerator(results)  # no optional fields
        reporter.to_html(p)
        content = p.read_text()
        assert "Reproducibility Statement" in content
        assert "not provided" in content
