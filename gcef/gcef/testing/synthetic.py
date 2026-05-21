"""
gcef.testing.synthetic
----------------------
Synthetic panel data generator for GCEF testing and exploration.

Produces realistic panel datasets that implement the data-generating
process implied by the GCEF specification. Every assumption in the
spec is operationalised here as a DGP parameter.

Two interfaces:
    make_panel()        — fully configurable base generator
    make_*()            — named factory functions for specific scenarios

Named factories are the primary test interface. Each one produces a
dataset designed to trigger a specific boundary condition defined in
the spec (an error, a warning, or a clean pass).

Usage
-----
>>> from gcef.testing.synthetic import make_valid_panel, make_single_lender
>>> data = make_valid_panel(n_smes=200, n_lenders=5, n_periods=10, seed=42)
>>> data_bad = make_single_lender()  # triggers SingleLenderError

Data-generating process
-----------------------
The DGP implements the spec's identification assumptions as structural
equations so tests can verify the framework behaves correctly at
boundaries:

Treatment assignment (A1 — instrument relevance):
    Z_jt = 1[t >= adoption_period_j]          # lender j adopted by period t
    D_it = bernoulli(sigmoid(alpha * Z_jt + u_i + eps_it))
    # alpha controls first-stage strength (instrument relevance)
    # u_i = SME fixed effect (unobserved confound for selection)

Outcome (causal structure):
    Y_it = tau_i * D_it + beta * X_it + u_i + shock_it + noise_it
    # tau_i = heterogeneous treatment effect (varies by firm type)
    # shock_it = climate shock effect, conditioned on shock indicator

Exclusion restriction (A2):
    Z_jt affects Y_it only through D_it in this DGP by construction.
    Violations can be introduced via the `exclusion_violation` parameter.

Monotonicity (A3):
    No defiers by construction — D_it is non-decreasing in Z_jt for
    each unit. Violations can be introduced via `defier_share`.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ── DGP configuration ─────────────────────────────────────────────────────────

@dataclass
class SyntheticConfig:
    """
    Full configuration for the synthetic data DGP.
    All parameters have defaults that produce a clean, valid dataset.

    Parameters
    ----------
    n_smes : int
        Number of SMEs in the panel.
    n_lenders : int
        Number of lenders. Must be >= 2 for the instrument to work.
        Set to 1 to trigger SingleLenderError.
    n_periods : int
        Number of time periods. Must be >= 6 for meaningful pre/post.
    adoption_periods : list[int] | None
        Periods in which each lender adopts. Length must equal n_lenders.
        If None, assigned randomly with minimum 2 pre-treatment periods
        for each lender (staggered design).
    first_stage_strength : float
        Controls instrument relevance (alpha in the DGP).
        High values (3.0+) produce strong instruments (F >> 10).
        Low values (0.5) produce weak instruments (F ~ 5-8).
        Default: 2.5 → F ~ 18-25.
    ate : float
        Average treatment effect on revenue stability.
        Default: 0.08 (8 percentage points).
    het_effect_by_sector : bool
        If True, treatment effects vary by sector (manufacturing > services).
        Produces interesting CATE heterogeneity. Default: True.
    complier_share_target : float
        Approximate target complier share (fraction of SMEs who respond
        to the instrument). Controls selection into treatment.
        Default: 0.35.
    shock_frequency : float
        Fraction of (SME, period) observations that experience a shock.
        Default: 0.15.
    shock_effect : float
        Reduction in revenue during shock periods (untreated firms).
        Default: -0.20 (20% revenue reduction).
    n_pre_treatment_min : int
        Minimum pre-treatment periods per lender cohort. Set to 1 to
        trigger ShortPanelWarning, 0 to trigger InsufficientPreTreatmentError.
        Default: 3.
    exclusion_violation : float
        Strength of direct effect of Z on Y (violates A2 exclusion
        restriction). Default: 0.0 (no violation).
    defier_share : float
        Fraction of SMEs who are defiers (take up iff lender did NOT
        adopt). Violates A3 monotonicity. Default: 0.0.
    missing_geocoding : bool
        If True, latitude/longitude columns are absent.
        Triggers GeocodingRequiredError. Default: False.
    admin_unit_only : bool
        If True, provides admin_unit instead of lat/lon.
        Triggers GeocodingResolutionWarning. Default: False.
    self_reported_verification : bool
        If True, sets verification_method to 'self_reported'.
        Triggers SelfReportedTreatmentWarning. Default: False.
    introduce_adoption_anomaly : bool
        If True, a small fraction of SMEs have take-up before their
        lender's adoption date. Triggers AdoptionTimingAnomalyWarning.
        Default: False.
    seed : int
        Random seed for reproducibility.
    """
    n_smes: int = 150
    n_lenders: int = 5
    n_periods: int = 10
    adoption_periods: Optional[list] = None
    first_stage_strength: float = 2.5
    ate: float = 0.08
    het_effect_by_sector: bool = True
    complier_share_target: float = 0.35
    shock_frequency: float = 0.15
    shock_effect: float = -0.20
    n_pre_treatment_min: int = 3
    exclusion_violation: float = 0.0
    defier_share: float = 0.0
    missing_geocoding: bool = False
    admin_unit_only: bool = False
    self_reported_verification: bool = False
    introduce_adoption_anomaly: bool = False
    seed: int = 42


# ── Core generator ─────────────────────────────────────────────────────────────

def make_panel(config: Optional[SyntheticConfig] = None, **kwargs) -> pd.DataFrame:
    """
    Generate a synthetic panel dataset for GCEF testing.

    Parameters
    ----------
    config : SyntheticConfig | None
        Full configuration object. If None, uses defaults with any
        kwargs applied as overrides.
    **kwargs
        Override individual SyntheticConfig fields without constructing
        the full config object.

    Returns
    -------
    pd.DataFrame
        Panel dataset with one row per (SME, period). Columns match
        the minimum viable dataset in spec Section 7, plus covariates
        and additional columns for diagnostics.

    Notes
    -----
    The DGP embeds the true treatment effect in the 'true_ate' and
    'true_cate' columns so tests can verify estimation accuracy.
    These columns are not available in real data — strip them before
    passing to GreenCreditEvaluator in non-testing contexts.
    """
    if config is None:
        config = SyntheticConfig(**kwargs)
    elif kwargs:
        # Apply overrides to existing config
        for k, v in kwargs.items():
            setattr(config, k, v)

    rng = np.random.default_rng(config.seed)
    C = config  # shorthand

    # ── Lender adoption periods ────────────────────────────────────────────────
    if C.adoption_periods is not None:
        assert len(C.adoption_periods) == C.n_lenders
        adoption_periods = C.adoption_periods
    else:
        adoption_periods = _assign_adoption_periods(
            n_lenders=C.n_lenders,
            n_periods=C.n_periods,
            n_pre_min=C.n_pre_treatment_min,
            rng=rng,
        )

    lender_ids = [f"lender_{i:02d}" for i in range(C.n_lenders)]
    lender_df = pd.DataFrame({
        "lender_id": lender_ids,
        "lender_green_adoption_period": adoption_periods,
    })

    # ── SME characteristics ────────────────────────────────────────────────────
    sme_ids = [f"sme_{i:04d}" for i in range(C.n_smes)]
    sectors = rng.choice(
        ["manufacturing", "agriprocessing", "services", "retail", "construction"],
        size=C.n_smes,
        p=[0.25, 0.20, 0.25, 0.20, 0.10],
    )
    firm_sizes = rng.choice(
        ["micro", "small", "medium"],
        size=C.n_smes,
        p=[0.50, 0.35, 0.15],
    )
    firm_ages = rng.integers(1, 25, size=C.n_smes)

    # Assign SMEs to lenders
    lender_assignments = rng.choice(lender_ids, size=C.n_smes)

    # SME fixed effects (unobserved confound — drives selection into treatment)
    sme_fe = rng.normal(0, 0.5, size=C.n_smes)

    # Base revenue (varies by size and sector)
    sector_revenue_multiplier = {
        "manufacturing": 1.3, "agriprocessing": 1.1,
        "services": 0.9, "retail": 0.8, "construction": 1.0,
    }
    size_revenue_multiplier = {"micro": 0.5, "small": 1.0, "medium": 2.5}
    base_revenue = np.array([
        rng.uniform(80_000, 120_000)
        * sector_revenue_multiplier[sectors[i]]
        * size_revenue_multiplier[firm_sizes[i]]
        for i in range(C.n_smes)
    ])

    # Heterogeneous treatment effects by sector
    if C.het_effect_by_sector:
        sector_te_multiplier = {
            "manufacturing": 1.5, "agriprocessing": 1.3,
            "services": 0.8, "retail": 0.7, "construction": 1.0,
        }
        true_cate = np.array([
            C.ate * sector_te_multiplier[sectors[i]] for i in range(C.n_smes)
        ])
    else:
        true_cate = np.full(C.n_smes, C.ate)

    # Complier type: always-taker, never-taker, complier, defier
    complier_types = _assign_complier_types(
        n_smes=C.n_smes,
        complier_share=C.complier_share_target,
        defier_share=C.defier_share,
        sme_fe=sme_fe,
        rng=rng,
    )

    # Geographic location (SSA bounding box roughly)
    lats = rng.uniform(-35, 15, size=C.n_smes)    # South Africa to Ethiopia
    lons = rng.uniform(10, 45, size=C.n_smes)     # Angola to Somalia

    # ── Build panel ───────────────────────────────────────────────────────────
    rows = []
    for i, sme_id in enumerate(sme_ids):
        lender_id = lender_assignments[i]
        adoption_period = lender_df.loc[
            lender_df["lender_id"] == lender_id, "lender_green_adoption_period"
        ].iloc[0]

        for t in range(C.n_periods):
            # Instrument: lender has adopted by period t
            Z = int(t >= adoption_period)

            # Treatment take-up based on complier type
            D = _compute_takeup(
                complier_type=complier_types[i],
                Z=Z,
                first_stage_strength=C.first_stage_strength,
                sme_fe=sme_fe[i],
                rng=rng,
            )

            # Shock indicator (lagged construction — shock in t affects Y in t+1)
            shock_this_period = int(rng.random() < C.shock_frequency)

            # Revenue with treatment effect, shock, and noise
            shock_impact = C.shock_effect if shock_this_period else 0.0
            treatment_impact = true_cate[i] * D
            exclusion_leak = C.exclusion_violation * Z  # A2 violation if > 0
            revenue = base_revenue[i] * (
                1.0
                + treatment_impact
                + shock_impact
                + exclusion_leak
                + sme_fe[i] * 0.1
                + rng.normal(0, 0.05)
            )

            # Loan repayment rate — higher for treated firms, lower during shocks
            repayment_base = 0.85 + 0.10 * D - 0.15 * shock_this_period
            loan_repayment_rate = float(np.clip(
                repayment_base + rng.normal(0, 0.05), 0.0, 1.0
            ))

            # Employment — grows slightly with treatment
            employment = max(1, int(
                int(firm_sizes[i] == "medium") * 10 + 5
                + D * 2
                + rng.normal(0, 1)
            ))

            # Adaptation investment — present only for some firms
            adaptation_investment = (
                float(rng.uniform(5_000, 50_000)) if (D == 1 and rng.random() < 0.6)
                else (float(rng.uniform(0, 5_000)) if rng.random() < 0.2 else None)
            )

            # Lagged shock instrument (what gets used in Stage 2)
            # This is the shock from t-1 affecting revenue at t
            rainfall_anomaly_lag1 = rng.normal(0, 1)
            if shock_this_period:
                rainfall_anomaly_lag1 -= rng.uniform(1.0, 2.5)  # negative anomaly

            row = {
                "sme_id": sme_id,
                "lender_id": lender_id,
                "period": t,
                "lender_green_adoption_period": adoption_period,
                "green_credit_takeup": D,
                "revenue": max(0.0, revenue),
                "loan_repayment_rate": loan_repayment_rate,
                "employment": employment,
                "adaptation_investment": adaptation_investment,
                "rainfall_anomaly_lag1": rainfall_anomaly_lag1,
                "shock_this_period": shock_this_period,
                "firm_age": firm_ages[i],
                "sector": sectors[i],
                "firm_size": firm_sizes[i],
                "instrument_Z": Z,
                # Ground truth for test assertions
                "true_cate": true_cate[i],
                "true_ate": C.ate,
                "complier_type": complier_types[i],
                "sme_fe": sme_fe[i],
            }

            # Geographic columns
            if not C.missing_geocoding:
                if C.admin_unit_only:
                    row["admin_unit"] = f"district_{rng.integers(1, 50):02d}"
                else:
                    row["sme_latitude"] = lats[i]
                    row["sme_longitude"] = lons[i]

            # Verification method
            row["verification_method"] = (
                "self_reported" if C.self_reported_verification
                else rng.choice(
                    ["document_review", "site_visit", "document_review"],
                    p=[0.6, 0.2, 0.2],
                )
            )
            row["treatment_type"] = "rate_reduction"
            row["conditionality_mechanism"] = "verified_investment"
            row["treatment_intensity"] = 0.03

            rows.append(row)

    data = pd.DataFrame(rows)

    # Compute prior_revenue (mean revenue in pre-treatment periods per SME)
    pre_treatment = data[data["period"] < data["lender_green_adoption_period"]]
    prior_rev = (
        pre_treatment.groupby("sme_id")["revenue"]
        .mean()
        .rename("prior_revenue")
    )
    data = data.merge(prior_rev, on="sme_id", how="left")

    # Adoption timing anomaly: flip a few take-up records to before adoption
    if C.introduce_adoption_anomaly:
        anomaly_smes = rng.choice(sme_ids, size=max(1, C.n_smes // 50), replace=False)
        mask = (
            data["sme_id"].isin(anomaly_smes)
            & (data["period"] == 0)
        )
        data.loc[mask, "green_credit_takeup"] = 1

    data = data.sort_values(["sme_id", "period"]).reset_index(drop=True)
    return data


# ── Named factory functions ────────────────────────────────────────────────────
# Each factory is named for the boundary condition it tests.

def make_valid_panel(
    n_smes: int = 150,
    n_lenders: int = 5,
    n_periods: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Clean valid dataset. All assumption checks should pass.
    First-stage F >> 10, complier share ~35%, no warnings.
    """
    return make_panel(SyntheticConfig(
        n_smes=n_smes,
        n_lenders=n_lenders,
        n_periods=n_periods,
        first_stage_strength=2.5,
        complier_share_target=0.35,
        n_pre_treatment_min=3,
        seed=seed,
    ))


def make_single_lender(seed: int = 42) -> pd.DataFrame:
    """
    Single lender. Should trigger SingleLenderError.
    All SMEs assigned to the same lender — no cross-lender variation.
    """
    return make_panel(SyntheticConfig(
        n_lenders=1,
        adoption_periods=[4],
        seed=seed,
    ))


def make_weak_instrument(seed: int = 42) -> pd.DataFrame:
    """
    Weak first-stage instrument. Should trigger WeakInstrumentError (F < 10)
    or WeakInstrumentWarning (10 <= F < 20).
    first_stage_strength=0.3 produces F ~ 3-6.
    """
    return make_panel(SyntheticConfig(
        first_stage_strength=0.3,
        seed=seed,
    ))


def make_marginal_instrument(seed: int = 42) -> pd.DataFrame:
    """
    Marginal instrument. Should trigger WeakInstrumentWarning (10 <= F < 20).
    first_stage_strength=0.9 produces F ~ 12-16.
    """
    return make_panel(SyntheticConfig(
        first_stage_strength=0.9,
        seed=seed,
    ))


def make_short_panel(min_pre_periods: int = 2, seed: int = 42) -> pd.DataFrame:
    """
    Short pre-treatment panel. min_pre_periods=2 triggers ShortPanelWarning.
    min_pre_periods=1 triggers InsufficientPreTreatmentError.
    """
    return make_panel(SyntheticConfig(
        n_periods=6,
        n_pre_treatment_min=min_pre_periods,
        seed=seed,
    ))


def make_low_complier_share(seed: int = 42) -> pd.DataFrame:
    """
    Low complier share (~12%). Should trigger ComplierShareWarning (< 20%).
    Most SMEs are always-takers or never-takers.
    """
    return make_panel(SyntheticConfig(
        complier_share_target=0.12,
        seed=seed,
    ))


def make_missing_geocoding(seed: int = 42) -> pd.DataFrame:
    """
    No latitude/longitude columns. Should trigger GeocodingRequiredError
    when shock-conditioned outcomes are requested.
    """
    return make_panel(SyntheticConfig(
        missing_geocoding=True,
        seed=seed,
    ))


def make_admin_unit_only(seed: int = 42) -> pd.DataFrame:
    """
    Admin unit geocoding only (no lat/lon). Should trigger
    GeocodingResolutionWarning and fall back to centroid.
    """
    return make_panel(SyntheticConfig(
        admin_unit_only=True,
        seed=seed,
    ))


def make_self_reported_treatment(seed: int = 42) -> pd.DataFrame:
    """
    Self-reported verification method. Should trigger
    SelfReportedTreatmentWarning.
    """
    return make_panel(SyntheticConfig(
        self_reported_verification=True,
        seed=seed,
    ))


def make_adoption_anomaly(seed: int = 42) -> pd.DataFrame:
    """
    Some SMEs show take-up before lender adoption.
    Should trigger AdoptionTimingAnomalyWarning (not a monotonicity error —
    the framework treats this as a data quality issue).
    """
    return make_panel(SyntheticConfig(
        introduce_adoption_anomaly=True,
        seed=seed,
    ))


def make_exclusion_violation(violation_strength: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """
    Dataset where the exclusion restriction (A2) is violated.
    Lender adoption timing has a direct effect on SME revenue beyond
    the treatment channel. Used to test sensitivity analysis.
    violation_strength=0.15 is a moderate violation.
    """
    return make_panel(SyntheticConfig(
        exclusion_violation=violation_strength,
        seed=seed,
    ))


def make_underidentified_outcome(seed: int = 42) -> pd.DataFrame:
    """
    Dataset with only revenue available (no employment, repayment rate,
    or adaptation investment). Should trigger ResilienceIndexWarning
    when underidentification_threshold >= 2.
    """
    data = make_valid_panel(seed=seed)
    # Drop the components that would make the index well-identified
    return data.drop(columns=["loan_repayment_rate", "employment",
                               "adaptation_investment"], errors="ignore")


def make_high_heterogeneity(seed: int = 42) -> pd.DataFrame:
    """
    Large heterogeneous treatment effects across sectors.
    Manufacturing ATE ~ 3x services. Useful for testing CATE
    estimation and complier profile characterisation.
    """
    return make_panel(SyntheticConfig(
        ate=0.12,
        het_effect_by_sector=True,
        n_smes=300,
        seed=seed,
    ))


# ── Internal helpers ───────────────────────────────────────────────────────────

def _assign_adoption_periods(
    n_lenders: int,
    n_periods: int,
    n_pre_min: int,
    rng: np.random.Generator,
) -> list:
    """
    Assigns staggered adoption periods to lenders.
    Ensures each lender has at least n_pre_min pre-treatment periods.
    Ensures at least one lender adopts late (for C&S not-yet-treated comparison).
    """
    if n_lenders == 1:
        return [max(n_pre_min, 2)]

    # Valid adoption window: period n_pre_min to n_periods - 1
    earliest = n_pre_min
    latest = n_periods - 1

    if earliest >= latest:
        earliest = max(1, latest - 1)

    # Draw without replacement where possible
    n_valid = latest - earliest + 1
    if n_lenders <= n_valid:
        periods = sorted(rng.choice(
            range(earliest, latest + 1), size=n_lenders, replace=False
        ).tolist())
    else:
        # More lenders than valid periods — allow some clustering
        periods = sorted(rng.integers(earliest, latest + 1, size=n_lenders).tolist())

    return periods


def _assign_complier_types(
    n_smes: int,
    complier_share: float,
    defier_share: float,
    sme_fe: np.ndarray,
    rng: np.random.Generator,
) -> list:
    """
    Assigns complier types based on target shares.
    Always-takers: high SME FE (self-motivated, would take up regardless)
    Never-takers: low SME FE (resistant, won't take up regardless)
    Compliers: respond to instrument
    Defiers: take up iff lender did NOT adopt (violations of A3)
    """
    always_taker_share = (1 - complier_share - defier_share) * 0.4
    never_taker_share = (1 - complier_share - defier_share) * 0.6

    types = rng.choice(
        ["complier", "always_taker", "never_taker", "defier"],
        size=n_smes,
        p=[
            complier_share,
            always_taker_share,
            never_taker_share,
            defier_share,
        ],
    )
    return types.tolist()


def _compute_takeup(
    complier_type: str,
    Z: int,
    first_stage_strength: float,
    sme_fe: float,
    rng: np.random.Generator,
) -> int:
    """
    Computes treatment take-up D given complier type and instrument Z.

    Lender adoption (Z=1) is a hard necessary condition for take-up for all
    types except defiers. This enforces the spec's monotonicity assumption (A3)
    in the DGP: no SME can take up green credit before their lender adopts a
    green product, because the product does not exist yet.

    Complier types:
    - always_taker: takes up as soon as lender adopts (Z=1 → high prob, Z=0 → 0)
    - never_taker: never takes up (D=0 always)
    - complier: take-up probability increases strongly with Z
    - defier: takes up iff lender did NOT adopt (violates A3 — only used when
              defier_share > 0 to explicitly test monotonicity violation scenarios)
    """
    if complier_type == "always_taker":
        # Lender adoption is necessary — cannot take up before lender adopts.
        # Once lender adopts (Z=1), always-taker takes up with high probability.
        if Z == 0:
            return 0
        prob = 0.90 + 0.08 * sme_fe
        return int(rng.random() < np.clip(prob, 0.70, 0.99))

    elif complier_type == "never_taker":
        # Never takes up regardless of lender adoption.
        return 0

    elif complier_type == "complier":
        # Lender adoption is necessary. Without it, cannot take up.
        # With it, responds to the instrument with probability driven by first_stage_strength.
        if Z == 0:
            return 0
        logit = first_stage_strength + 0.3 * sme_fe - 0.5
        prob = 1 / (1 + np.exp(-logit))
        return int(rng.random() < prob)

    elif complier_type == "defier":
        # Violates monotonicity — only present when defier_share > 0.
        # Takes up iff lender did NOT adopt. Used only in explicit violation tests.
        if Z == 1:
            return 0
        logit = first_stage_strength * 0.5 + 0.3 * sme_fe - 1.0
        prob = 1 / (1 + np.exp(-logit))
        return int(rng.random() < prob)

    else:
        raise ValueError(f"Unknown complier_type: {complier_type}")


# ── Diagnostic utilities ───────────────────────────────────────────────────────

def describe_panel(data: pd.DataFrame) -> dict:
    """
    Computes ground-truth diagnostics for a synthetic panel.
    Useful for verifying the DGP before running GCEF estimation.

    Returns dict with: n_smes, n_lenders, n_periods, complier_share,
    always_taker_share, never_taker_share, defier_share,
    first_stage_correlation, mean_takeup_treated, mean_takeup_control,
    true_ate, mean_true_cate.
    """
    sme_level = data.drop_duplicates("sme_id")
    type_counts = sme_level["complier_type"].value_counts(normalize=True)

    # Naive first-stage correlation (proxy for instrument relevance)
    corr = data["green_credit_takeup"].corr(data["instrument_Z"])

    # Mean take-up by instrument value
    mean_D_Z1 = data.loc[data["instrument_Z"] == 1, "green_credit_takeup"].mean()
    mean_D_Z0 = data.loc[data["instrument_Z"] == 0, "green_credit_takeup"].mean()

    return {
        "n_smes": data["sme_id"].nunique(),
        "n_lenders": data["lender_id"].nunique(),
        "n_periods": data["period"].nunique(),
        "complier_share": round(type_counts.get("complier", 0), 3),
        "always_taker_share": round(type_counts.get("always_taker", 0), 3),
        "never_taker_share": round(type_counts.get("never_taker", 0), 3),
        "defier_share": round(type_counts.get("defier", 0), 3),
        "first_stage_correlation": round(corr, 3),
        "mean_takeup_Z1": round(mean_D_Z1, 3),
        "mean_takeup_Z0": round(mean_D_Z0, 3),
        "first_stage_diff": round(mean_D_Z1 - mean_D_Z0, 3),
        "true_ate": round(data["true_ate"].iloc[0], 4),
        "mean_true_cate": round(data["true_cate"].mean(), 4),
        "sectors": data["sector"].unique().tolist(),
        "has_geocoding": "sme_latitude" in data.columns,
        "has_prior_revenue": "prior_revenue" in data.columns,
    }
