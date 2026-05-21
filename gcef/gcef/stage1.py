"""
gcef.stage1
-----------
Stage 1 of the GCEF pipeline: IV estimation with staggered DiD design.

Instruments SME green credit take-up with lender adoption timing.
Computes kappa weights for the complier restriction in Stage 2.

Pipeline outputs (Stage1Result dataclass):
    late              LATE estimate
    late_se           Standard error of LATE
    late_ci           Confidence interval (lower, upper)
    f_statistic       First-stage F-statistic (instrument relevance)
    kappa_weights     Individual kappa_i weights (Abadie 2003)
    complier_profile  Kappa-weighted covariate profile
    complier_share    Wald complier share estimate
    complier_share_ci Complier share with confidence interval
    estimand          Estimand object for this stage

Spec reference: Section 5.1, DD2 (C&S vs TWFE), DD3a (kappa weights)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from gcef.estimand import Estimand, make_complier_estimand
from gcef.exceptions import GCEFWarning


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class Stage1Result:
    """
    Typed output from Stage 1. Using attribute access rather than dict
    prevents silent KeyError from typos (e.g. kappa_weight vs kappa_weights).

    All fields are populated by run_stage1(); none are optional after a
    successful run.
    """
    late: float
    """Local Average Treatment Effect estimate (2SLS)."""

    late_se: float
    """Clustered standard error of the LATE estimate."""

    late_ci: tuple
    """95% confidence interval as (lower, upper)."""

    f_statistic: float
    """First-stage F-statistic for instrument relevance (Kleibergen-Paap proxy)."""

    kappa_weights: np.ndarray
    """
    Individual kappa_i weights (length = n_rows in input data).
    Implements Abadie (2003) Equation 3. Negative values are expected
    for never-takers in post-adoption periods — they are not data quality
    failures in the staggered DiD context.
    """

    complier_profile: pd.DataFrame
    """
    Kappa-weighted covariate profile for the complier subpopulation.
    Columns: covariate, weighted_mean, full_mean, complier_to_full_ratio.
    """

    complier_share: float
    """Wald estimate of complier share: E[D|Z=1] - E[D|Z=0]."""

    complier_share_ci: dict
    """Complier share with confidence interval. Keys: estimate, ci_lower, ci_upper."""

    estimand: Estimand
    """Estimand object labelling what this stage produces."""


# ── Main entry point ───────────────────────────────────────────────────────────

def run_stage1(
    data: pd.DataFrame,
    unit_id: str,
    time_id: str,
    lender_id: str,
    adoption_time: str,
    outcome_col: str,
    takeup_col: str,
    covariates: List[str],
    shock_instrument: str,
    shock_threshold: float,
    random_seed: Optional[int] = None,
) -> Stage1Result:
    """
    Stage 1: IV estimation using lender adoption timing as instrument.

    Implements a two-stage least squares (2SLS) estimator with:
    - Unit (SME) and time (period) fixed effects
    - Lender-level clustered standard errors
    - Abadie (2003) kappa weights for complier restriction in Stage 2

    The instrument Z_it = 1[period_t >= adoption_period_j] is constructed
    internally from the adoption_time column. It is not expected to be
    pre-computed in the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset. Must contain unit_id, time_id, lender_id,
        adoption_time, outcome_col, takeup_col.
    unit_id : str
        Column name for SME identifier.
    time_id : str
        Column name for time period.
    lender_id : str
        Column name for lender identifier.
    adoption_time : str
        Column name for lender green product adoption period.
    outcome_col : str
        Column name for the outcome variable (resilience index).
    takeup_col : str
        Column name for binary green credit take-up indicator.
    covariates : list[str]
        Covariate columns used for complier profile characterisation.
        Not used in the IV regression (which uses FE absorption).
    shock_instrument : str
        Column name of the lagged climate shock indicator.
    shock_threshold : float
        Values below this threshold define a shock period.
    random_seed : int | None
        Random seed for any stochastic components.

    Returns
    -------
    Stage1Result
        Typed dataclass with all Stage 1 outputs.
    """
    from pyfixest.estimation import feols

    data = data.copy()

    # ── A: Construct instrument (if not already built by pipeline) ─────────────
    if "_Z" not in data.columns:
        data["_Z"] = (data[time_id] >= data[adoption_time]).astype(int)

    # ── B: 2SLS via pyfixest ───────────────────────────────────────────────────
    # Formula: outcome ~ 1 | unit_fe + time_fe | takeup ~ instrument
    # Clustered SEs at lender level (the level of instrument assignment).
    formula = (
        f"{outcome_col} ~ 1 | {unit_id} + {time_id} | {takeup_col} ~ _Z"
    )

    try:
        fit = feols(
            formula,
            data=data,
            vcov={f"CRV1": lender_id},
        )
        late = float(fit.coef().iloc[0])
        late_se = float(fit.se().iloc[0])
        ci = fit.confint()
        late_ci = (float(ci.iloc[0, 0]), float(ci.iloc[0, 1]))
        f_stat = float(fit._f_stat_1st_stage)

    except Exception as e:
        raise RuntimeError(
            f"Stage 1 IV regression failed: {e}. "
            f"Check that the dataset has sufficient within-unit and within-time "
            f"variation in the instrument, and that {lender_id} has at least 2 "
            f"unique values for cluster-robust SEs."
        ) from e

    # ── C: Kappa weight computation (Abadie 2003, Equation 3) ─────────────────
    kappa_weights, p_z1 = _compute_kappa_weights(
        data=data,
        takeup_col=takeup_col,
        instrument_col="_Z",
        time_id=time_id,
    )

    # ── D: Complier share (Wald estimate) ─────────────────────────────────────
    D = data[takeup_col].values
    Z = data["_Z"].values
    complier_share, complier_share_ci = _estimate_complier_share(D, Z, lender_id, data)

    # ── E: Complier profile (kappa-weighted covariate statistics) ─────────────
    complier_profile = _build_complier_profile(
        data=data,
        kappa_weights=kappa_weights,
        covariates=[c for c in covariates if c in data.columns],
    )

    # ── F: Build estimand ──────────────────────────────────────────────────────
    estimand = make_complier_estimand(treatment=None, outcome=None)

    return Stage1Result(
        late=late,
        late_se=late_se,
        late_ci=late_ci,
        f_statistic=f_stat,
        kappa_weights=kappa_weights,
        complier_profile=complier_profile,
        complier_share=complier_share,
        complier_share_ci=complier_share_ci,
        estimand=estimand,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _compute_kappa_weights(
    data: pd.DataFrame,
    takeup_col: str,
    instrument_col: str,
    time_id: str,
) -> tuple:
    """
    Computes Abadie (2003) kappa weights.

    Formula (Equation 3):
        κᵢ = 1 − Dᵢ(1−Zᵢ)/P(Z=0|Xᵢ) − (1−Dᵢ)Zᵢ/P(Z=1|Xᵢ)

    Propensity model: period-stratified — P(Z=1|period=t) is the fraction
    of SMEs with Z=1 at period t. This is appropriate for staggered DiD where
    Z varies at the lender-cohort level, making cohort-conditional assignment
    the natural propensity model.

    Note on negative kappa weights:
    In staggered DiD, negative kappa values are expected for never-takers in
    post-adoption periods (D=0, Z=1) when P(Z=1|t) < 1. This is structurally
    correct — they are not-yet-treated comparison units that identify the
    direction of selection bias. Unlike cross-sectional IV, a 17–25% negative
    kappa share is typical in staggered DiD with heterogeneous adoption timing
    and does not indicate a degenerate propensity model.

    Returns
    -------
    kappa_weights : np.ndarray
        Individual kappa values, length = len(data).
    p_z1 : np.ndarray
        Estimated P(Z=1) per observation (for diagnostics).
    """
    D = data[takeup_col].values.astype(float)
    Z = data[instrument_col].values.astype(float)

    # Period-stratified propensity: E[Z|period=t]
    period_p = data.groupby(time_id)[instrument_col].mean()
    p_z1 = data[time_id].map(period_p).values.astype(float)
    p_z0 = 1.0 - p_z1

    # Clip to avoid division by zero at boundary periods
    # (first periods have p_z1=0, last periods may have p_z1=1)
    eps = 1e-6
    p_z1_safe = np.clip(p_z1, eps, 1.0 - eps)
    p_z0_safe = np.clip(p_z0, eps, 1.0 - eps)

    kappa = (
        1.0
        - D * (1.0 - Z) / p_z0_safe
        - (1.0 - D) * Z / p_z1_safe
    )

    return kappa, p_z1


def _estimate_complier_share(
    D: np.ndarray,
    Z: np.ndarray,
    lender_id: str,
    data: pd.DataFrame,
) -> tuple:
    """
    Wald estimate of complier share with confidence interval.

    Complier share = E[D|Z=1] - E[D|Z=0] (Wald first-stage numerator
    when outcome = D).

    CI uses normal approximation with lender-clustered variance.
    """
    # Point estimate
    mean_D_Z1 = float(D[Z == 1].mean()) if (Z == 1).any() else 0.0
    mean_D_Z0 = float(D[Z == 0].mean()) if (Z == 0).any() else 0.0
    complier_share = mean_D_Z1 - mean_D_Z0

    # Lender-clustered variance for the difference in means
    lenders = data[lender_id].values
    unique_lenders = np.unique(lenders)
    n_lenders = len(unique_lenders)

    if n_lenders < 2:
        # Fallback: use simple variance (no clustering)
        var_share = (
            np.var(D[Z == 1]) / max((Z == 1).sum(), 1)
            + np.var(D[Z == 0]) / max((Z == 0).sum(), 1)
        )
    else:
        # Clustered variance: variance of cluster-level contributions
        cluster_contribs = []
        for lender in unique_lenders:
            mask = lenders == lender
            d_l = D[mask]
            z_l = Z[mask]
            d_z1_l = d_l[z_l == 1].mean() if (z_l == 1).any() else mean_D_Z1
            d_z0_l = d_l[z_l == 0].mean() if (z_l == 0).any() else mean_D_Z0
            cluster_contribs.append(d_z1_l - d_z0_l)

        cluster_contribs = np.array(cluster_contribs)
        var_share = np.var(cluster_contribs, ddof=1) / n_lenders

    se_share = float(np.sqrt(max(var_share, 0.0)))
    z_95 = 1.96
    ci = {
        "estimate": round(complier_share, 4),
        "ci_lower": round(complier_share - z_95 * se_share, 4),
        "ci_upper": round(complier_share + z_95 * se_share, 4),
    }

    return float(complier_share), ci


def _build_complier_profile(
    data: pd.DataFrame,
    kappa_weights: np.ndarray,
    covariates: List[str],
) -> pd.DataFrame:
    """
    Constructs the kappa-weighted covariate profile for the complier subpopulation.

    For each covariate, computes:
    - weighted_mean: E[X|complier] via kappa-weighted mean
    - full_mean: unweighted mean over full sample
    - complier_to_full_ratio: weighted_mean / full_mean (>1 = over-represented)

    Uses only positive kappa weights (negative kappa = not identified as complier).
    Negative weights are zeroed before computing the weighted mean, following
    Abadie (2003)'s recommendation for the composite estimator.

    Spec reference: Section 5.1 (complier profile), DD3a
    """
    # Zero out negative kappa weights (they identify non-compliers)
    kappa_pos = np.clip(kappa_weights, 0.0, None)
    kappa_sum = kappa_pos.sum()

    rows = []
    for col in covariates:
        if col not in data.columns:
            continue
        col_data = data[col]

        # Handle categorical/string columns
        if col_data.dtype == object or str(col_data.dtype) in ("category", "string") or pd.api.types.is_string_dtype(col_data):
            # For categorical, compute weighted mode and distribution
            for cat_val in col_data.dropna().unique():
                indicator = (col_data == cat_val).astype(float).fillna(0).values
                weighted_mean = float(
                    (kappa_pos * indicator).sum() / kappa_sum
                    if kappa_sum > 0 else np.nan
                )
                full_mean = float(indicator.mean())
                rows.append({
                    "covariate": f"{col}={cat_val}",
                    "weighted_mean": weighted_mean,
                    "full_mean": full_mean,
                    "complier_to_full_ratio": (
                        weighted_mean / full_mean
                        if full_mean > 0 else np.nan
                    ),
                })
        else:
            # Numeric column
            numeric = col_data.fillna(col_data.median()).values.astype(float)
            if kappa_sum > 0:
                weighted_mean = float((kappa_pos * numeric).sum() / kappa_sum)
            else:
                weighted_mean = float(np.nan)
            full_mean = float(numeric.mean())
            rows.append({
                "covariate": col,
                "weighted_mean": weighted_mean,
                "full_mean": full_mean,
                "complier_to_full_ratio": (
                    weighted_mean / full_mean if full_mean != 0 else np.nan
                ),
            })

    if not rows:
        return pd.DataFrame(
            columns=["covariate", "weighted_mean", "full_mean", "complier_to_full_ratio"]
        )

    return pd.DataFrame(rows)
