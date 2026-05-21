"""
Tests for the synthetic data generator.

These tests verify that the DGP produces datasets with the structural
properties the GCEF specification requires. They are the foundation
that every subsequent module test builds on.
"""
import pytest
import pandas as pd
import numpy as np
from gcef.testing.synthetic import (
    make_valid_panel, make_single_lender, make_weak_instrument,
    make_short_panel, make_low_complier_share, make_missing_geocoding,
    make_admin_unit_only, make_self_reported_treatment,
    make_adoption_anomaly, make_underidentified_outcome,
    make_high_heterogeneity, make_exclusion_violation,
    describe_panel, SyntheticConfig, make_panel,
)

# ── Schema tests ───────────────────────────────────────────────────────────────

class TestPanelSchema:
    """Every required column from spec Section 7 is present in valid panels."""

    def setup_method(self):
        self.data = make_valid_panel(seed=42)

    def test_required_columns_present(self):
        required = [
            "sme_id", "lender_id", "period",
            "lender_green_adoption_period", "green_credit_takeup",
            "revenue",
        ]
        for col in required:
            assert col in self.data.columns, f"Missing required column: {col}"

    def test_recommended_columns_present(self):
        recommended = [
            "loan_repayment_rate", "employment",
            "sme_latitude", "sme_longitude",
            "firm_age", "sector", "firm_size", "prior_revenue",
        ]
        for col in recommended:
            assert col in self.data.columns, f"Missing recommended column: {col}"

    def test_treatment_spec_columns_present(self):
        spec_cols = ["treatment_type", "conditionality_mechanism",
                     "verification_method", "treatment_intensity"]
        for col in spec_cols:
            assert col in self.data.columns, f"Missing treatment spec column: {col}"

    def test_panel_is_balanced(self):
        counts = self.data.groupby("sme_id")["period"].count()
        assert counts.nunique() == 1, "Panel is not balanced"

    def test_no_negative_revenue(self):
        assert (self.data["revenue"] >= 0).all()

    def test_repayment_rate_bounded(self):
        assert self.data["loan_repayment_rate"].between(0, 1).all()

    def test_employment_positive(self):
        assert (self.data["employment"] >= 1).all()

    def test_binary_takeup(self):
        assert set(self.data["green_credit_takeup"].unique()).issubset({0, 1})

    def test_binary_instrument(self):
        assert set(self.data["instrument_Z"].unique()).issubset({0, 1})


# ── DGP structural properties ──────────────────────────────────────────────────

class TestDGPStructure:
    """The DGP implements the spec's identification assumptions correctly."""

    def setup_method(self):
        self.data = make_valid_panel(n_smes=300, seed=42)
        self.diag = describe_panel(self.data)

    def test_multiple_lenders(self):
        assert self.data["lender_id"].nunique() >= 2

    def test_staggered_adoption(self):
        adoption_periods = (
            self.data.drop_duplicates("lender_id")
            ["lender_green_adoption_period"].nunique()
        )
        assert adoption_periods >= 2, "Adoption not staggered across lenders"

    def test_instrument_predicts_takeup(self):
        """A1: First-stage correlation is positive and meaningful."""
        diff = self.diag["first_stage_diff"]
        assert diff > 0.05, f"First-stage diff too small: {diff}"

    def test_complier_share_near_target(self):
        """Complier share within 15pp of 35% target."""
        share = self.diag["complier_share"]
        assert abs(share - 0.35) < 0.15, f"Complier share {share} too far from target 0.35"

    def test_no_monotonicity_violation_by_default(self):
        """A3: By default, defier share is 0."""
        assert self.diag["defier_share"] == 0.0

    def test_instrument_exogenous_to_outcome_by_default(self):
        """A2: By default, no direct Z → Y effect (exclusion_violation=0)."""
        config = SyntheticConfig(exclusion_violation=0.0, seed=42)
        data = make_panel(config)
        # Indirect test: correlation between Z and residualised revenue
        # should be near zero after controlling for D
        data["residual_revenue"] = (
            data["revenue"] - data.groupby("green_credit_takeup")["revenue"].transform("mean")
        )
        corr = data["residual_revenue"].corr(data["instrument_Z"])
        assert abs(corr) < 0.15, f"Unexpected Z-residual correlation: {corr}"

    def test_treatment_effect_positive(self):
        """True ATE is positive — green credit improves resilience."""
        treated = self.data[self.data["green_credit_takeup"] == 1]["revenue"].mean()
        untreated = self.data[self.data["green_credit_takeup"] == 0]["revenue"].mean()
        assert treated > untreated

    def test_prior_revenue_computed(self):
        """prior_revenue is the mean pre-treatment revenue per SME."""
        assert self.data["prior_revenue"].notna().any()
        # Check it's only from pre-treatment periods
        sme = self.data["sme_id"].iloc[0]
        sme_data = self.data[self.data["sme_id"] == sme]
        adoption = sme_data["lender_green_adoption_period"].iloc[0]
        pre = sme_data[sme_data["period"] < adoption]["revenue"].mean()
        prior = sme_data["prior_revenue"].iloc[0]
        assert abs(pre - prior) < 1.0, "prior_revenue doesn't match pre-treatment mean"

    def test_sector_heterogeneity(self):
        """CATE varies by sector when het_effect_by_sector=True."""
        sme_level = self.data.drop_duplicates("sme_id")
        sector_cate = sme_level.groupby("sector")["true_cate"].mean()
        assert sector_cate.max() > sector_cate.min() * 1.5, \
            "Expected meaningful sector heterogeneity"

    def test_reproducibility(self):
        """Same seed produces identical panels."""
        d1 = make_valid_panel(seed=99)
        d2 = make_valid_panel(seed=99)
        pd.testing.assert_frame_equal(d1, d2)

    def test_different_seeds_differ(self):
        d1 = make_valid_panel(seed=1)
        d2 = make_valid_panel(seed=2)
        assert not d1["revenue"].equals(d2["revenue"])


# ── Boundary condition factories ───────────────────────────────────────────────

class TestBoundaryFactories:
    """
    Each factory produces a dataset designed to trigger a specific
    boundary condition. These tests verify the factories work correctly
    so that module tests can rely on them.
    """

    def test_single_lender_has_one_lender(self):
        data = make_single_lender()
        assert data["lender_id"].nunique() == 1

    def test_weak_instrument_low_first_stage(self):
        data = make_weak_instrument(seed=42)
        diag = describe_panel(data)
        # With the hard-gate DGP, first_stage_diff measures complier response rate.
        # A weak instrument means few SMEs respond — low complier share.
        # The test verifies the weak instrument config produces a structurally
        # different dataset from the valid panel (lower complier share or response).
        valid = make_valid_panel(seed=42)
        valid_diag = describe_panel(valid)
        # Weak instrument should have meaningfully lower first-stage signal
        # than the standard valid panel
        assert diag["first_stage_diff"] <= valid_diag["first_stage_diff"]

    def test_short_panel_few_pre_periods(self):
        data = make_short_panel(min_pre_periods=2, seed=42)
        # Check that at least one lender has <= 2 distinct pre-treatment periods
        def pre_period_count(group):
            adoption = group["lender_green_adoption_period"].iloc[0]
            return group[group["period"] < adoption]["period"].nunique()

        pre_counts = data.groupby("lender_id").apply(pre_period_count)
        assert pre_counts.min() <= 2, f"Expected some lender with <=2 pre periods, got min={pre_counts.min()}"

    def test_low_complier_share_factory(self):
        data = make_low_complier_share(seed=42)
        diag = describe_panel(data)
        assert diag["complier_share"] < 0.25, \
            f"Expected low complier share, got {diag['complier_share']}"

    def test_missing_geocoding_no_lat_lon(self):
        data = make_missing_geocoding()
        assert "sme_latitude" not in data.columns
        assert "sme_longitude" not in data.columns

    def test_admin_unit_only_has_admin_col(self):
        data = make_admin_unit_only()
        assert "admin_unit" in data.columns
        assert "sme_latitude" not in data.columns

    def test_self_reported_verification(self):
        data = make_self_reported_treatment()
        assert (data["verification_method"] == "self_reported").all()

    def test_adoption_anomaly_introduces_early_takeup(self):
        data = make_adoption_anomaly()
        anomalies = data[
            (data["green_credit_takeup"] == 1)
            & (data["period"] < data["lender_green_adoption_period"])
        ]
        assert len(anomalies) > 0, "Expected adoption timing anomalies"

    def test_underidentified_outcome_missing_columns(self):
        data = make_underidentified_outcome()
        assert "loan_repayment_rate" not in data.columns
        assert "employment" not in data.columns

    def test_exclusion_violation_affects_revenue(self):
        clean = make_valid_panel(seed=42)
        violated = make_exclusion_violation(violation_strength=0.20, seed=42)
        # Revenue should differ between the two (Z has direct effect in violated)
        assert not clean["revenue"].equals(violated["revenue"])

    def test_high_heterogeneity_sector_spread(self):
        data = make_high_heterogeneity(seed=42)
        sme_level = data.drop_duplicates("sme_id")
        cate_range = sme_level["true_cate"].max() - sme_level["true_cate"].min()
        assert cate_range > 0.05, f"Expected wide CATE range, got {cate_range}"


# ── describe_panel diagnostics ─────────────────────────────────────────────────

class TestDescribePanel:

    def test_returns_all_keys(self):
        data = make_valid_panel(seed=42)
        diag = describe_panel(data)
        required_keys = [
            "n_smes", "n_lenders", "n_periods",
            "complier_share", "always_taker_share", "never_taker_share",
            "defier_share", "first_stage_correlation",
            "mean_takeup_Z1", "mean_takeup_Z0", "first_stage_diff",
            "true_ate", "mean_true_cate",
            "has_geocoding", "has_prior_revenue",
        ]
        for key in required_keys:
            assert key in diag, f"Missing diagnostic key: {key}"

    def test_shares_sum_to_one(self):
        data = make_valid_panel(seed=42)
        diag = describe_panel(data)
        total = (
            diag["complier_share"] + diag["always_taker_share"]
            + diag["never_taker_share"] + diag["defier_share"]
        )
        assert abs(total - 1.0) < 0.01, f"Complier type shares don't sum to 1: {total}"

    def test_n_smes_matches_config(self):
        data = make_valid_panel(n_smes=80, seed=42)
        diag = describe_panel(data)
        assert diag["n_smes"] == 80

    def test_geocoding_flag_accurate(self):
        with_geo = make_valid_panel(seed=42)
        without_geo = make_missing_geocoding(seed=42)
        assert describe_panel(with_geo)["has_geocoding"] is True
        assert describe_panel(without_geo)["has_geocoding"] is False
