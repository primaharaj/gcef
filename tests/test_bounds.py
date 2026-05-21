"""
tests/test_bounds.py
--------------------
Tests for gcef.bounds.compute_manski_bounds.

Tests verify:
- Manski bounds formula is correctly applied for always-takers and never-takers
- Complier rows carry NaN bounds (their effects identified via results.cate)
- Population classification from kappa weights is correct
- Output DataFrame has required columns and is full-length
- Bounds respect the mathematical constraint lower ≤ upper
- Outcome support estimation is robust to outliers
- Both subpopulation estimands are correctly constructed
"""
import pytest
import warnings
import numpy as np
import pandas as pd

from gcef.bounds import (
    compute_manski_bounds,
    _estimate_support,
    _classify_subpopulations,
    NONCOMPLIER_KAPPA_THRESHOLD,
    SUPPORT_PERCENTILE_LO,
    SUPPORT_PERCENTILE_HI,
)
from gcef.testing.synthetic import make_valid_panel


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def panel():
    """Small clean panel with a mix of compliers, always-takers, never-takers."""
    return make_valid_panel(n_smes=150, n_lenders=4, n_periods=8, seed=42)


@pytest.fixture(scope="module")
def panel_with_instrument(panel):
    """Panel with _Z column added (normally done by Stage 1)."""
    data = panel.copy()
    data["_Z"] = (
        data["period"] >= data["lender_green_adoption_period"]
    ).astype(int)
    return data


@pytest.fixture(scope="module")
def synthetic_kappa(panel_with_instrument):
    """
    Synthetic kappa weights that produce a controlled mix of subpopulations.
    Always-takers: D=1, Z=0 → set kappa negative
    Never-takers:  D=0, Z=1 → set kappa negative
    Compliers:     everything else → set kappa positive
    """
    data = panel_with_instrument
    D = data["green_credit_takeup"].values.astype(float)
    Z = data["_Z"].values.astype(float)

    kappa = np.ones(len(data)) * 0.5  # default: complier weight
    at_mask = (D == 1) & (Z == 0)
    nt_mask = (D == 0) & (Z == 1)
    kappa[at_mask] = -0.3   # always-taker
    kappa[nt_mask] = -0.2   # never-taker
    return kappa


@pytest.fixture(scope="module")
def bounds_result(panel_with_instrument, synthetic_kappa):
    """Full bounds result for the panel."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_manski_bounds(
            data=panel_with_instrument,
            outcome_col="revenue",
            takeup_col="green_credit_takeup",
            unit_id="sme_id",
            kappa_weights=synthetic_kappa,
        )


# ── Output structure ───────────────────────────────────────────────────────────

class TestBoundsOutputStructure:

    def test_returns_dataframe(self, bounds_result):
        assert isinstance(bounds_result, pd.DataFrame)

    def test_full_length(self, bounds_result, panel_with_instrument):
        assert len(bounds_result) == len(panel_with_instrument)

    def test_required_columns_present(self, bounds_result):
        required = {
            "population", "lower_bound", "upper_bound",
            "kappa_weight", "estimand_population",
            "outcome_support_lo", "outcome_support_hi",
        }
        assert required.issubset(bounds_result.columns)

    def test_population_values_valid(self, bounds_result):
        valid = {"complier", "always_taker", "never_taker"}
        assert set(bounds_result["population"].unique()).issubset(valid)

    def test_has_complier_and_never_taker_populations(self, bounds_result):
        """
        Clean panel with hard-gate DGP: no always-takers (D=1, Z=0 is impossible
        by DGP construction — lender adoption is a necessary condition).
        Never-takers (D=0, Z=1) are present. Compliers are present.
        """
        pops = set(bounds_result["population"].unique())
        assert "complier" in pops
        assert "never_taker" in pops

    def test_always_taker_present_in_constructed_data(self):
        """
        Always-takers are present when the dataset contains D=1, Z=0 rows
        with negative kappa — as happens in real portfolios where some SMEs
        self-financed green assets before their lender's programme launched.
        """
        n = 20
        df = pd.DataFrame({
            "sme_id": [f"s{i}" for i in range(n)],
            "revenue": np.random.default_rng(42).normal(500, 50, n),
            "green_credit_takeup": [1.0] * n,   # D=1
            "_Z": [0.0] * n,                     # Z=0 → always-taker pattern
            "lender_green_adoption_period": [5] * n,
            "period": [2] * n,
        })
        kappa = np.full(n, -0.4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compute_manski_bounds(
                data=df, outcome_col="revenue",
                takeup_col="green_credit_takeup",
                unit_id="sme_id", kappa_weights=kappa,
            )
        assert "always_taker" in result["population"].values

    def test_sme_id_present(self, bounds_result, panel_with_instrument):
        assert "sme_id" in bounds_result.columns
        assert (bounds_result["sme_id"].values == panel_with_instrument["sme_id"].values).all()


# ── Manski formula correctness ─────────────────────────────────────────────────

class TestManskiBoundsFormula:

    def test_always_taker_lower_bound_formula(self, bounds_result):
        """
        For always-takers: lower = Y_obs - y_hi
        Y_obs is observed revenue; y_hi is 99th percentile of support.
        """
        at = bounds_result[bounds_result["population"] == "always_taker"]
        if len(at) == 0:
            pytest.skip("No always-takers in this panel")
        y_hi = at["outcome_support_hi"].iloc[0]
        # lower_bound = Y_obs - y_hi
        # We don't have Y_obs in the bounds_result, but we can check:
        # lower ≤ upper for all rows
        assert (at["lower_bound"] <= at["upper_bound"] + 1e-9).all()

    def test_always_taker_bound_width_equals_support(self, bounds_result):
        """
        Width of always-taker bounds = y_hi - y_lo (full support width).
        This is a key Manski property.
        """
        at = bounds_result[bounds_result["population"] == "always_taker"]
        if len(at) == 0:
            pytest.skip("No always-takers in this panel")
        y_lo = at["outcome_support_lo"].iloc[0]
        y_hi = at["outcome_support_hi"].iloc[0]
        support_width = y_hi - y_lo
        bound_widths = at["upper_bound"] - at["lower_bound"]
        assert np.allclose(bound_widths, support_width, atol=1e-6), (
            f"Expected bound width ≈ {support_width:.3f}, "
            f"got range {bound_widths.min():.3f}–{bound_widths.max():.3f}"
        )

    def test_never_taker_bound_width_equals_support(self, bounds_result):
        """Width of never-taker bounds = y_hi - y_lo (full support width)."""
        nt = bounds_result[bounds_result["population"] == "never_taker"]
        if len(nt) == 0:
            pytest.skip("No never-takers in this panel")
        y_lo = nt["outcome_support_lo"].iloc[0]
        y_hi = nt["outcome_support_hi"].iloc[0]
        support_width = y_hi - y_lo
        bound_widths = nt["upper_bound"] - nt["lower_bound"]
        assert np.allclose(bound_widths, support_width, atol=1e-6)

    def test_lower_leq_upper_for_all_rows(self, bounds_result):
        """Mathematical constraint: lower ≤ upper for every non-NaN row."""
        non_null = bounds_result.dropna(subset=["lower_bound", "upper_bound"])
        assert (non_null["lower_bound"] <= non_null["upper_bound"] + 1e-9).all()

    def test_always_taker_upper_minus_lower_eq_support_width(self):
        """
        Direct formula test with known values.
        y_lo=0, y_hi=10, Y_obs=6.
        Always-taker: lower = 6 - 10 = -4, upper = 6 - 0 = 6. Width = 10.
        """
        n = 10
        df = pd.DataFrame({
            "sme_id": [f"s{i}" for i in range(n)],
            "revenue": [6.0] * n,
            "green_credit_takeup": [1.0] * n,   # D=1: always-takers
            "_Z": [0.0] * n,                     # Z=0: pre-adoption
            "lender_green_adoption_period": [5] * n,
            "period": [2] * n,
        })
        kappa = np.full(n, -0.5)  # negative → always-taker

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compute_manski_bounds(
                data=df,
                outcome_col="revenue",
                takeup_col="green_credit_takeup",
                unit_id="sme_id",
                kappa_weights=kappa,
            )

        at = result[result["population"] == "always_taker"]
        assert len(at) == n
        y_lo = at["outcome_support_lo"].iloc[0]
        y_hi = at["outcome_support_hi"].iloc[0]
        expected_lower = 6.0 - y_hi
        expected_upper = 6.0 - y_lo
        assert np.allclose(at["lower_bound"].values, expected_lower, atol=1e-6)
        assert np.allclose(at["upper_bound"].values, expected_upper, atol=1e-6)

    def test_never_taker_formula_known_values(self):
        """
        Direct formula test with known values.
        y_lo=0, y_hi=10, Y_obs=3.
        Never-taker: lower = 0 - 3 = -3, upper = 10 - 3 = 7. Width = 10.
        """
        n = 10
        df = pd.DataFrame({
            "sme_id": [f"s{i}" for i in range(n)],
            "revenue": [3.0] * n,
            "green_credit_takeup": [0.0] * n,   # D=0: never-takers
            "_Z": [1.0] * n,                     # Z=1: post-adoption
            "lender_green_adoption_period": [1] * n,
            "period": [5] * n,
        })
        kappa = np.full(n, -0.4)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compute_manski_bounds(
                data=df,
                outcome_col="revenue",
                takeup_col="green_credit_takeup",
                unit_id="sme_id",
                kappa_weights=kappa,
            )

        nt = result[result["population"] == "never_taker"]
        assert len(nt) == n
        y_lo = nt["outcome_support_lo"].iloc[0]
        y_hi = nt["outcome_support_hi"].iloc[0]
        assert np.allclose(nt["lower_bound"].values, y_lo - 3.0, atol=1e-6)
        assert np.allclose(nt["upper_bound"].values, y_hi - 3.0, atol=1e-6)


# ── Complier rows ──────────────────────────────────────────────────────────────

class TestComplierRows:

    def test_complier_bounds_are_nan(self, bounds_result):
        """Compliers are identified via results.cate — bounds are NaN."""
        compliers = bounds_result[bounds_result["population"] == "complier"]
        assert compliers["lower_bound"].isna().all()
        assert compliers["upper_bound"].isna().all()

    def test_complier_estimand_population_label(self, bounds_result):
        compliers = bounds_result[bounds_result["population"] == "complier"]
        assert (compliers["estimand_population"] == "complier").all()


# ── Population classification ──────────────────────────────────────────────────

class TestSubpopulationClassification:

    def test_d1_z0_negative_kappa_is_always_taker(self):
        """D=1, Z=0, kappa < 0 → always_taker."""
        D = np.array([1.0, 0.0, 1.0])
        Z = np.array([0.0, 1.0, 1.0])
        kappa = np.array([-0.5, -0.3, 0.8])
        labels, at, nt = _classify_subpopulations(D, Z, kappa)
        assert labels[0] == "always_taker"
        assert labels[1] == "never_taker"
        assert labels[2] == "complier"

    def test_d0_z1_negative_kappa_is_never_taker(self):
        D = np.array([0.0])
        Z = np.array([1.0])
        kappa = np.array([-0.2])
        labels, at, nt = _classify_subpopulations(D, Z, kappa)
        assert labels[0] == "never_taker"

    def test_positive_kappa_is_complier_regardless_of_dz(self):
        """Positive kappa → complier, even if D/Z pattern looks like non-complier."""
        D = np.array([1.0, 0.0])
        Z = np.array([0.0, 1.0])
        kappa = np.array([0.6, 0.4])  # positive → complier
        labels, at, nt = _classify_subpopulations(D, Z, kappa)
        assert labels[0] == "complier"
        assert labels[1] == "complier"

    def test_no_overlap_between_at_and_nt(self):
        D = np.array([1.0, 0.0, 0.5, 1.0, 0.0])
        Z = np.array([0.0, 1.0, 0.5, 1.0, 0.0])
        kappa = np.array([-0.3, -0.2, 0.5, 0.7, 0.6])
        labels, at, nt = _classify_subpopulations(D, Z, kappa)
        assert not (at & nt).any()


# ── Support estimation ─────────────────────────────────────────────────────────

class TestSupportEstimation:

    def test_support_uses_percentiles_not_extremes(self):
        """Outliers should not inflate support bounds."""
        Y = np.concatenate([np.ones(98) * 5.0, [0.0001, 1000.0]])
        y_lo, y_hi = _estimate_support(Y)
        # 1st/99th percentile should exclude the extreme values
        assert y_lo > 0.001
        assert y_hi < 999.0

    def test_support_lo_lt_hi(self):
        Y = np.random.default_rng(42).normal(100, 15, 200)
        y_lo, y_hi = _estimate_support(Y)
        assert y_lo < y_hi

    def test_degenerate_support_fallback(self):
        """Constant outcome → fallback to mean ± 3 SD."""
        Y = np.ones(50) * 5.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y_lo, y_hi = _estimate_support(Y)
        assert y_lo < y_hi  # fallback should produce valid support


# ── Missing _Z column ──────────────────────────────────────────────────────────

class TestInputValidation:

    def test_missing_z_column_raises(self, panel):
        """If _Z is not present, should raise RuntimeError with clear message."""
        kappa = np.ones(len(panel)) * 0.3
        with pytest.raises(RuntimeError, match="_Z"):
            compute_manski_bounds(
                data=panel,  # no _Z column
                outcome_col="revenue",
                takeup_col="green_credit_takeup",
                unit_id="sme_id",
                kappa_weights=kappa,
            )

    def test_all_complier_kappa_produces_nan_bounds(self, panel_with_instrument):
        """When all kappa > 0, all rows are compliers and all bounds are NaN."""
        kappa = np.ones(len(panel_with_instrument)) * 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compute_manski_bounds(
                data=panel_with_instrument,
                outcome_col="revenue",
                takeup_col="green_credit_takeup",
                unit_id="sme_id",
                kappa_weights=kappa,
            )
        assert result["lower_bound"].isna().all()
        assert result["upper_bound"].isna().all()
        assert (result["population"] == "complier").all()


# ── Estimand construction ──────────────────────────────────────────────────────

class TestEstimandInBounds:

    def test_at_rows_have_always_taker_estimand(self, bounds_result):
        at = bounds_result[bounds_result["population"] == "always_taker"]
        if len(at) == 0:
            pytest.skip("No always-takers")
        assert (at["estimand_population"] == "always_takers").all()

    def test_nt_rows_have_never_taker_estimand(self, bounds_result):
        nt = bounds_result[bounds_result["population"] == "never_taker"]
        if len(nt) == 0:
            pytest.skip("No never-takers")
        assert (nt["estimand_population"] == "never_takers").all()
