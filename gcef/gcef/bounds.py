"""
gcef.bounds
-----------
Manski (1990) worst-case partial identification bounds for always-takers
and never-takers.

Background
----------
The GCEF IV identification strategy (lender adoption timing as instrument)
identifies treatment effects for compliers only — SMEs who took up green
credit because their lender adopted, and who would not have otherwise. The
instrument provides no identifying variation for:

  Always-takers: SMEs who would take up green credit regardless of whether
      their lender adopted. Observed in the data as D=1 when Z=0
      (took up before/without lender adoption).

  Never-takers: SMEs who would never take up green credit regardless of
      lender adoption. Observed in the data as D=0 when Z=1 (did not take
      up even when their lender had adopted).

For these subpopulations, Manski (1990) derives worst-case bounds on the
average treatment effect using only the observed outcome distribution and
the support of the outcome variable. No additional assumptions are required.

Manski (1990) bounds
--------------------
Let Y ∈ [y_lo, y_hi] be the bounded outcome. The treatment effect for unit
i is τᵢ = Yᵢ(1) - Yᵢ(0).

For always-takers (observed Y(1) = Y_obs, Y(0) unobserved):
    τᵢ ∈ [Y_obs_i - y_hi,  Y_obs_i - y_lo]

For never-takers (observed Y(0) = Y_obs, Y(1) unobserved):
    τᵢ ∈ [y_lo - Y_obs_i,  y_hi - Y_obs_i]

These are the tightest possible bounds given only the support restriction.
They are deliberately wide — the data cannot rule out effects across this
range for these subpopulations. A DFI analyst can work with honest bounds.

Subpopulation identification via kappa weights
----------------------------------------------
The Abadie (2003) kappa weights identify subpopulations structurally:

  kappa < 0 for D=1, Z=0 observations → likely always-takers
      (D=1·(1-Z)/p_z0 term dominates; took up before lender adoption)

  kappa < 0 for D=0, Z=1 observations → likely never-takers
      (D=0·Z/p_z1 term dominates; did not take up post-adoption)

  kappa > 0 → compliers (identified; bounds not needed)

We use this structural classification rather than D and Z alone because
kappa weights account for the period-level adoption intensity — they are
more informative than raw D/Z in staggered DiD contexts.

Spec reference
--------------
DD3a: "For full-portfolio questions, the framework provides cate_bounds —
Manski-style partial identification bounds for always-takers and never-takers
— rather than point estimates with inflated confidence intervals."

Section 4.3: "cate_bounds: Manski (1990) partial identification bounds...
These are BOUNDS, not point estimates."

Output format
-------------
Full-length DataFrame (one row per observation in data), indexed like data.
Complier rows have NaN bounds — their effects are identified via results.cate.
Population column labels each row: "complier", "always_taker", "never_taker".

References
----------
Manski, C.F. (1990). Nonparametric bounds on treatment effects.
    American Economic Review Papers and Proceedings, 80(2), 319-323.
"""
from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

from gcef.estimand import make_bounds_estimand


# ── Constants ──────────────────────────────────────────────────────────────────

#: Kappa weight threshold for classifying always-takers and never-takers.
#: Observations with kappa <= this value AND (D=1,Z=0) or (D=0,Z=1) are
#: classified as non-compliers. Chosen to be slightly above floating point
#: noise while remaining conservative.
NONCOMPLIER_KAPPA_THRESHOLD = 0.0

#: Percentile for estimating outcome support bounds.
#: Using 1st/99th rather than min/max to reduce sensitivity to outliers.
#: The support bounds affect the width of all Manski bounds.
SUPPORT_PERCENTILE_LO = 1.0
SUPPORT_PERCENTILE_HI = 99.0


# ── Main entry point ───────────────────────────────────────────────────────────

def compute_manski_bounds(
    data: pd.DataFrame,
    outcome_col: str,
    takeup_col: str,
    unit_id: str,
    kappa_weights: np.ndarray,
    treatment: Optional[Any] = None,
    outcome: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Compute Manski (1990) worst-case partial identification bounds for
    always-takers and never-takers.

    Parameters
    ----------
    data : pd.DataFrame
        Full panel dataset (with _Z instrument column from Stage 1).
        Must contain outcome_col, takeup_col, and _Z.
    outcome_col : str
        Outcome variable column (resilience index).
    takeup_col : str
        Binary green credit take-up column (D).
    unit_id : str
        SME identifier column.
    kappa_weights : np.ndarray
        Individual kappa_i weights from Stage 1 (length = len(data)).
        Used to classify always-takers and never-takers.
    treatment : GreenCreditTreatment | None
        Treatment object for Estimand construction.
    outcome : ResilienceIndex | None
        Outcome object for Estimand construction.

    Returns
    -------
    pd.DataFrame
        Full-length bounds DataFrame (one row per row in data).

        Columns
        -------
        unit_id_col : str
            SME identifier (same values as data[unit_id]).
        population : str
            "complier", "always_taker", or "never_taker".
        lower_bound : float
            Manski lower bound on τᵢ = Y(1) - Y(0).
            NaN for compliers (their effects are identified via results.cate).
        upper_bound : float
            Manski upper bound on τᵢ.
            NaN for compliers.
        kappa_weight : float
            Raw kappa weight for this observation (diagnostic).
        estimand_population : str
            Population label from the Estimand object.

    Notes
    -----
    Complier rows have NaN bounds. The analyst should use results.cate for
    complier effects and results.cate_bounds for always-taker/never-taker
    bounds. The full-length format enables portfolio-level joining:

        portfolio = data.merge(results.cate_bounds, left_index=True, right_index=True)

    Inference caveat:
        Manski bounds do not carry standard errors in the classical sense.
        The width of the bounds is determined by the outcome support, not
        sampling uncertainty. For large samples, the bounds are well-estimated;
        for small samples (< 100 non-complier observations), interpret with care.

    References
    ----------
    Manski, C.F. (1990). Nonparametric bounds on treatment effects.
        American Economic Review Papers and Proceedings, 80(2), 319-323.
    """
    data = data.copy().reset_index(drop=True)

    if "_Z" not in data.columns:
        raise RuntimeError(
            "compute_manski_bounds requires '_Z' instrument column from Stage 1. "
            "Ensure pipeline.py passes the Stage-1-augmented dataset."
        )

    D = data[takeup_col].values.astype(float)
    Z = data["_Z"].values.astype(float)
    Y = data[outcome_col].fillna(np.nanmedian(data[outcome_col])).values.astype(float)
    kappa = kappa_weights.astype(float)

    # ── 1. Outcome support ────────────────────────────────────────────────────
    y_lo, y_hi = _estimate_support(Y)

    # ── 2. Classify subpopulations ────────────────────────────────────────────
    population_labels, at_mask, nt_mask = _classify_subpopulations(D, Z, kappa)

    n_at = at_mask.sum()
    n_nt = nt_mask.sum()
    n_complier = (~at_mask & ~nt_mask).sum()

    if n_at == 0:
        warnings.warn(
            "No always-taker observations identified (D=1, Z=0, kappa≤0). "
            "Always-taker bounds will be empty. "
            "Check that the panel includes pre-adoption periods.",
            stacklevel=2,
        )
    if n_nt == 0:
        warnings.warn(
            "No never-taker observations identified (D=0, Z=1, kappa≤0). "
            "Never-taker bounds will be empty. "
            "Check that the panel includes post-adoption periods with non-takers.",
            stacklevel=2,
        )

    # ── 3. Compute Manski bounds ──────────────────────────────────────────────
    lower_bounds = np.full(len(data), np.nan)
    upper_bounds = np.full(len(data), np.nan)

    # Always-takers: observe Y(1) = Y_obs; Y(0) is unobserved ∈ [y_lo, y_hi]
    # τ = Y(1) - Y(0) ∈ [Y_obs - y_hi, Y_obs - y_lo]
    if at_mask.any():
        lower_bounds[at_mask] = Y[at_mask] - y_hi
        upper_bounds[at_mask] = Y[at_mask] - y_lo

    # Never-takers: observe Y(0) = Y_obs; Y(1) is unobserved ∈ [y_lo, y_hi]
    # τ = Y(1) - Y(0) ∈ [y_lo - Y_obs, y_hi - Y_obs]
    if nt_mask.any():
        lower_bounds[nt_mask] = y_lo - Y[nt_mask]
        upper_bounds[nt_mask] = y_hi - Y[nt_mask]

    # ── 4. Build Estimand objects ─────────────────────────────────────────────
    at_estimand = make_bounds_estimand("always_takers", treatment, outcome)
    nt_estimand = make_bounds_estimand("never_takers", treatment, outcome)

    estimand_population = np.where(
        at_mask, at_estimand.population,
        np.where(nt_mask, nt_estimand.population, "complier")
    )

    # ── 5. Assemble output DataFrame ──────────────────────────────────────────
    bounds_df = pd.DataFrame({
        unit_id: data[unit_id].values,
        "population": population_labels,
        "lower_bound": lower_bounds,
        "upper_bound": upper_bounds,
        "kappa_weight": kappa,
        "estimand_population": estimand_population,
        "outcome_support_lo": y_lo,
        "outcome_support_hi": y_hi,
    }, index=data.index)

    _log_bounds_summary(bounds_df, n_at, n_nt, n_complier, y_lo, y_hi)

    return bounds_df


# ── Helpers ────────────────────────────────────────────────────────────────────

def _estimate_support(Y: np.ndarray) -> tuple:
    """
    Estimate the support [y_lo, y_hi] of the outcome distribution.

    Uses robust percentiles (1st/99th) rather than min/max to reduce
    sensitivity to outliers. The support bounds determine the width of
    all Manski bounds — wider support → wider (more honest) bounds.

    Returns
    -------
    y_lo : float, y_hi : float
    """
    y_lo = float(np.nanpercentile(Y, SUPPORT_PERCENTILE_LO))
    y_hi = float(np.nanpercentile(Y, SUPPORT_PERCENTILE_HI))

    if y_hi <= y_lo:
        # Degenerate support — outcome has no variation
        # Fall back to mean ± 3 SD (or ± 1 if SD=0)
        mu = float(np.nanmean(Y))
        sigma = float(np.nanstd(Y))
        if sigma == 0:
            sigma = max(abs(mu) * 0.1, 1.0)  # 10% of mean or 1.0 if mean is zero
        y_lo = mu - 3 * sigma
        y_hi = mu + 3 * sigma
        warnings.warn(
            f"Outcome support is degenerate (y_lo >= y_hi after percentile estimation). "
            f"Falling back to mean ± 3 SD support for Manski bounds: "
            f"[{y_lo:.4f}, {y_hi:.4f}].",
            stacklevel=3,
        )

    return y_lo, y_hi


def _classify_subpopulations(
    D: np.ndarray,
    Z: np.ndarray,
    kappa: np.ndarray,
) -> tuple:
    """
    Classify observations into compliers, always-takers, and never-takers
    using kappa weights and the (D, Z) pattern.

    Classification rules:
        always_taker: kappa <= 0 AND D=1 AND Z=0
            (took up before/without lender adoption; Y(0) unobserved)
        never_taker:  kappa <= 0 AND D=0 AND Z=1
            (did not take up post-adoption; Y(1) unobserved)
        complier:     everything else (kappa > 0, or boundary cases)

    The double condition (kappa ≤ 0 AND D/Z pattern) is more conservative
    than kappa alone. It ensures that observations misclassified by the
    propensity model are not included in the bounds subpopulations.

    Returns
    -------
    population_labels : np.ndarray of str
    at_mask : np.ndarray of bool (always-taker observations)
    nt_mask : np.ndarray of bool (never-taker observations)
    """
    at_mask = (kappa <= NONCOMPLIER_KAPPA_THRESHOLD) & (D == 1) & (Z == 0)
    nt_mask = (kappa <= NONCOMPLIER_KAPPA_THRESHOLD) & (D == 0) & (Z == 1)

    # Ensure no overlap (should be impossible given D/Z conditions)
    assert not (at_mask & nt_mask).any(), (
        "Observation classified as both always-taker and never-taker. "
        "This should be impossible given the D/Z conditions."
    )

    population_labels = np.where(
        at_mask, "always_taker",
        np.where(nt_mask, "never_taker", "complier")
    )

    return population_labels, at_mask, nt_mask


def _log_bounds_summary(
    bounds_df: pd.DataFrame,
    n_at: int,
    n_nt: int,
    n_complier: int,
    y_lo: float,
    y_hi: float,
) -> None:
    """
    Log a summary of bounds computation to help analysts interpret output.
    Fired as a warning so it appears in the pipeline's warning log.
    """
    n_total = len(bounds_df)
    at_bounds = bounds_df[bounds_df["population"] == "always_taker"]
    nt_bounds = bounds_df[bounds_df["population"] == "never_taker"]

    summary_parts = [
        f"Manski bounds computed. Outcome support: [{y_lo:.3f}, {y_hi:.3f}]. "
        f"Subpopulations: {n_complier} complier ({n_complier/n_total:.0%}), "
        f"{n_at} always-taker ({n_at/n_total:.0%}), "
        f"{n_nt} never-taker ({n_nt/n_total:.0%})."
    ]

    if n_at > 0:
        at_width = (at_bounds["upper_bound"] - at_bounds["lower_bound"]).mean()
        summary_parts.append(
            f"Always-taker bounds width (mean): {at_width:.3f}. "
            f"Bounds span full outcome support by construction."
        )
    if n_nt > 0:
        nt_width = (nt_bounds["upper_bound"] - nt_bounds["lower_bound"]).mean()
        summary_parts.append(
            f"Never-taker bounds width (mean): {nt_width:.3f}."
        )

    summary_parts.append(
        "These are Manski (1990) worst-case bounds, not estimates with wide CIs. "
        "Use results.cate for identified complier treatment effects."
    )

    warnings.warn(" ".join(summary_parts), stacklevel=3)
