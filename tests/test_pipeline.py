"""
tests/test_pipeline.py
----------------------
Integration tests for GreenCreditEvaluator and GCEFResults.

Structured in two tiers:

Tier 1 — Pre-Stage 2 (runnable now):
    Tests that verify the pipeline's pre-flight checks, Stage 1 execution,
    and results object assembly up to the point where Stage 2 raises
    NotImplementedError. These run against real Stage 1 output.

Tier 2 — Post-Stage 2 contract (marked xfail until stage2 implemented):
    Tests that define what the complete results object must contain.
    Written now so Stage 2 implementation has a clear target. These will
    be un-xfailed one by one as Stage 2 components are completed.
"""
import pytest
import warnings
import unittest.mock as mock
import numpy as np
import pandas as pd

from gcef.pipeline import GreenCreditEvaluator, GCEFResults, COMPLIER_KAPPA_THRESHOLD
from gcef.treatment import GreenCreditTreatment, TreatmentType, ConditionalityMechanism, VerificationMethod
from gcef.outcomes import ResilienceIndex
from gcef.estimand import Estimand
from gcef.exceptions import (
    SingleLenderError,
    SelfReportedTreatmentWarning,
    AdoptionTimingAnomalyWarning,
    NuisanceModelOverrideWarning,
)
from gcef.testing.synthetic import (
    make_valid_panel,
    make_single_lender,
    make_adoption_anomaly,
    make_self_reported_treatment as make_self_reported,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def standard_treatment():
    return GreenCreditTreatment(
        type=TreatmentType.RATE_REDUCTION,
        conditionality_mechanism=ConditionalityMechanism.VERIFIED_INVESTMENT,
        verification_method=VerificationMethod.DOCUMENT_REVIEW,
        intensity=0.03,
    )


@pytest.fixture(scope="module")
def standard_outcome():
    return ResilienceIndex(
        columns={
            "revenue": 0.40,
            "loan_repayment_rate": 0.30,
            "employment": 0.20,
            "adaptation_investment": 0.10,
        },
        shock_instrument="rainfall_anomaly_lag1",
        shock_threshold=-1.5,
    )


@pytest.fixture(scope="module")
def standard_evaluator(standard_treatment, standard_outcome):
    return GreenCreditEvaluator(
        treatment=standard_treatment,
        outcome=standard_outcome,
        unit_id="sme_id",
        time_id="period",
        lender_id="lender_id",
        adoption_time="lender_green_adoption_period",
        covariates=["firm_age", "sector", "firm_size", "prior_revenue"],
        random_seed=42,
    )


@pytest.fixture(scope="module")
def valid_data():
    return make_valid_panel(n_smes=200, n_lenders=5, n_periods=10, seed=42)


def _make_stub_results(evaluator, data, suppress_warnings=True):
    """
    Runs the pipeline with Stage 2 and bounds stubbed out.
    Returns a complete GCEFResults object with real Stage 1 outputs.
    """
    stub_stage2 = {
        "cate": pd.DataFrame({
            "sme_id": data["sme_id"].unique()[:5],
            "cate_estimate": [0.1] * 5,
            "cate_se": [0.02] * 5,
        }),
        "propensity_scores": np.full(len(data), 0.5),
    }
    stub_bounds = pd.DataFrame(
        columns=["unit_id", "lower_bound", "upper_bound", "population"]
    )
    ctx = warnings.catch_warnings()
    with mock.patch("gcef.stage2.run_stage2", return_value=stub_stage2), \
         mock.patch("gcef.bounds.compute_manski_bounds", return_value=stub_bounds):
        if suppress_warnings:
            with ctx:
                warnings.simplefilter("ignore")
                return evaluator.fit(data)
        else:
            return evaluator.fit(data)


@pytest.fixture(scope="module")
def stage1_results(standard_evaluator, valid_data):
    """
    Full end-to-end results from a real evaluator.fit() call.
    No mocks — ForestDRIV runs for real. Used by all Tier 1 and Tier 2 tests.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return standard_evaluator.fit(valid_data)


# ── Tier 1: Pre-flight checks ──────────────────────────────────────────────────

class TestPreflightChecks:

    def test_single_lender_raises_before_stage1(self, standard_evaluator):
        data = make_single_lender()
        with pytest.raises(SingleLenderError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _make_stub_results(standard_evaluator, data)

    def test_adoption_anomaly_warns(self, standard_evaluator):
        data = make_adoption_anomaly()
        with pytest.warns(AdoptionTimingAnomalyWarning):
            _make_stub_results(standard_evaluator, data, suppress_warnings=False)

    def test_self_reported_warns_during_fit(self, standard_outcome):
        """SelfReportedTreatmentWarning fires during fit()."""
        evaluator = GreenCreditEvaluator(
            treatment=GreenCreditTreatment(
                type=TreatmentType.RATE_REDUCTION,
                conditionality_mechanism=ConditionalityMechanism.SELF_REPORTED,
                verification_method=VerificationMethod.SELF_REPORTED,
            ),
            outcome=standard_outcome,
            unit_id="sme_id",
            time_id="period",
            lender_id="lender_id",
            adoption_time="lender_green_adoption_period",
            covariates=[],
        )
        data = make_valid_panel(n_smes=100, n_lenders=3, n_periods=8, seed=42)
        with pytest.warns(SelfReportedTreatmentWarning):
            _make_stub_results(evaluator, data, suppress_warnings=False)

    def test_nuisance_override_warns(self, standard_treatment, standard_outcome):
        """Passing any model_propensity triggers NuisanceModelOverrideWarning."""
        from sklearn.linear_model import LogisticRegression
        with pytest.warns(NuisanceModelOverrideWarning):
            GreenCreditEvaluator(
                treatment=standard_treatment,
                outcome=standard_outcome,
                unit_id="sme_id",
                time_id="period",
                lender_id="lender_id",
                adoption_time="lender_green_adoption_period",
                covariates=[],
                model_propensity=LogisticRegression(),
            )


# ── Tier 1: assumptions_tested schema ─────────────────────────────────────────

class TestAssumptionsTested:

    def test_assumptions_tested_is_dict(self, stage1_results):
        assert isinstance(stage1_results.assumptions_tested, dict)

    def test_required_keys_present(self, stage1_results):
        required = {
            "single_lender", "adoption_timing_anomaly", "panel_length",
            "instrument_relevance", "kappa_weight_negatives", "complier_share",
        }
        assert required.issubset(stage1_results.assumptions_tested.keys())

    def test_each_result_has_schema_fields(self, stage1_results):
        for key, result in stage1_results.assumptions_tested.items():
            assert "passed" in result, f"{key} missing 'passed'"
            assert "value" in result, f"{key} missing 'value'"
            assert "test" in result, f"{key} missing 'test'"

    def test_instrument_relevance_passes(self, stage1_results):
        ir = stage1_results.assumptions_tested["instrument_relevance"]
        assert ir["passed"] is True
        assert ir["value"] > 10

    def test_single_lender_passes(self, stage1_results):
        sl = stage1_results.assumptions_tested["single_lender"]
        assert sl["passed"] is True

    def test_kappa_negatives_uses_staggered_did_thresholds(self, stage1_results):
        kn = stage1_results.assumptions_tested["kappa_weight_negatives"]
        assert kn["design"] == "staggered_did"
        assert kn["warning_threshold"] == 0.30
        assert kn["error_threshold"] == 0.45


# ── Tier 1: Stage 1 outputs ───────────────────────────────────────────────────

class TestStage1OutputsInResults:

    def test_late_has_required_keys(self, stage1_results):
        for key in ("estimate", "se", "ci_lower", "ci_upper"):
            assert key in stage1_results.late

    def test_late_estimate_positive(self, stage1_results):
        assert stage1_results.late["estimate"] > 0

    def test_complier_share_has_required_keys(self, stage1_results):
        for key in ("estimate", "ci_lower", "ci_upper"):
            assert key in stage1_results.complier_share

    def test_complier_share_in_unit_interval(self, stage1_results):
        assert 0.0 < stage1_results.complier_share["estimate"] < 1.0

    def test_complier_profile_is_dataframe(self, stage1_results):
        assert isinstance(stage1_results.complier_profile, pd.DataFrame)
        assert len(stage1_results.complier_profile) > 0

    def test_complier_profile_covers_firm_age(self, stage1_results):
        covs = stage1_results.complier_profile["covariate"].tolist()
        assert any("firm_age" in c for c in covs)

    def test_estimand_attached_and_correct(self, stage1_results):
        assert isinstance(stage1_results.estimand, Estimand)
        assert stage1_results.estimand.population == "compliers"
        assert stage1_results.estimand.extrapolation_flag is False

    def test_cate_complier_mask_is_boolean_series(self, stage1_results, valid_data):
        mask = stage1_results.cate_complier_mask
        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool
        assert len(mask) == len(valid_data)

    def test_complier_kappa_threshold_is_0_1(self):
        assert COMPLIER_KAPPA_THRESHOLD == 0.1


# ── Tier 2: Post-Stage 2 contract ─────────────────────────────────────────────

class TestStage2Contract:
    """Stage 2 is implemented — these are now standard assertions."""

    def test_cate_is_full_length_dataframe(self, stage1_results, valid_data):
        assert isinstance(stage1_results.cate, pd.DataFrame)
        assert len(stage1_results.cate) == len(valid_data)

    def test_cate_has_required_columns(self, stage1_results):
        assert {"cate_estimate", "cate_se", "kappa_weight",
                "cate_ci_lower", "cate_ci_upper", "in_shock_period"}.issubset(
            stage1_results.cate.columns
        )

    def test_complier_cate_positive_on_average(self, stage1_results):
        """DGP has positive treatment effect — complier CATEs positive on average."""
        mask = stage1_results.cate_complier_mask.values  # use .values to avoid index mismatch
        complier_cate = stage1_results.cate.loc[mask, "cate_estimate"]
        assert complier_cate.mean() > 0

    def test_cate_se_all_positive(self, stage1_results):
        assert (stage1_results.cate["cate_se"] > 0).all()

    def test_cate_ci_lower_leq_upper(self, stage1_results):
        cate = stage1_results.cate
        assert (cate["cate_ci_lower"] <= cate["cate_ci_upper"]).all()

    def test_cate_estimate_inside_ci(self, stage1_results):
        cate = stage1_results.cate
        assert (cate["cate_ci_lower"] <= cate["cate_estimate"]).all()
        assert (cate["cate_estimate"] <= cate["cate_ci_upper"]).all()

    def test_cate_bounds_has_lower_upper_leq(self, stage1_results):
        bounds = stage1_results.cate_bounds
        assert isinstance(bounds, pd.DataFrame)
        non_null = bounds.dropna(subset=["lower_bound", "upper_bound"])
        assert (non_null["lower_bound"] <= non_null["upper_bound"]).all()

    def test_overlap_check_in_assumptions(self, stage1_results):
        assert "overlap" in stage1_results.assumptions_tested

    def test_overlap_result_has_schema(self, stage1_results):
        overlap = stage1_results.assumptions_tested["overlap"]
        assert "passed" in overlap
        assert "value" in overlap
        assert 0.0 <= overlap["value"] <= 1.0

    def test_cate_kappa_weight_col_matches_complier_mask(self, stage1_results):
        """kappa_weight column in cate aligns with cate_complier_mask."""
        cate = stage1_results.cate
        mask = stage1_results.cate_complier_mask.values
        # Complier rows should have higher kappa than non-complier rows
        complier_kappa = cate.loc[mask, "kappa_weight"].mean()
        non_complier_kappa = cate.loc[~mask, "kappa_weight"].mean()
        assert complier_kappa > non_complier_kappa

    def test_estimand_identification_string_present(self, stage1_results):
        assert "IV-DiD" in stage1_results.estimand.identification
        assert "staggered" in stage1_results.estimand.identification
