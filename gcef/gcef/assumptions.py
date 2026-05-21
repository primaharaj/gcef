"""
gcef.assumptions
----------------
All assumption tests for GCEF. Produces the assumptions_tested dictionary
attached to every results object.

Schema (Option A from spec Section 4.3):
    {
        "test_name": {
            "passed": bool,
            "value": float | int,
            "threshold": float | int | None,
            "test": str,
            "warning": str | None,
            ... (test-specific extra fields)
        }
    }
"""
from __future__ import annotations
import warnings
from typing import Optional
import pandas as pd
import numpy as np
from gcef.exceptions import (
    WeakInstrumentError, WeakInstrumentWarning,
    SingleLenderError, InsufficientPreTreatmentError,
    ShortPanelWarning, AdoptionTimingAnomalyWarning,
    KappaWeightDegeneracyError, KappaWeightWarning,
    SmallComplierShareWarning,
)


def check_single_lender(data: pd.DataFrame, lender_id: str) -> dict:
    """A5 proxy: instrument requires cross-lender variation."""
    n_lenders = data[lender_id].nunique()
    passed = n_lenders > 1
    result = {
        "passed": passed,
        "value": n_lenders,
        "threshold": 2,
        "test": "unique_lender_count",
        "warning": None,
    }
    if not passed:
        raise SingleLenderError(
            f"Dataset contains only 1 unique lender_id. The staggered DiD "
            f"instrument requires variation in adoption timing across lenders. "
            f"If you have branch-level rollout data, use branch_id as lender_id."
        )
    return result


def check_instrument_relevance(first_stage_f: float) -> dict:
    """A1: First-stage F-statistic for instrument relevance."""
    passed = first_stage_f >= 10
    result = {
        "passed": passed,
        "value": round(first_stage_f, 3),
        "threshold": 10,
        "warning_threshold": 20,
        "test": "first_stage_F",
        "warning": None,
    }
    if first_stage_f < 10:
        raise WeakInstrumentError(
            f"First-stage F-statistic is {first_stage_f:.2f} (threshold: 10). "
            f"Weak instrument produces biased LATE estimates. "
            f"Consider: (a) additional instruments, (b) restricting to lenders "
            f"with more staggered adoption, (c) LIML estimation."
        )
    elif first_stage_f < 20:
        msg = (
            f"First-stage F-statistic is {first_stage_f:.2f} (warning threshold: 20). "
            f"Instrument is marginally relevant. Interpret LATE with caution."
        )
        result["warning"] = msg
        result["passed"] = False  # amber — passes error threshold but not warning threshold
        warnings.warn(msg, WeakInstrumentWarning, stacklevel=3)
    return result


def check_adoption_timing_anomaly(
    data: pd.DataFrame,
    unit_id: str,
    takeup_col: str,
    adoption_period_col: str,
    period_col: str,
) -> dict:
    """A3 empirical check: no SME takes up before their lender adopted."""
    anomalies = data[
        (data[takeup_col] == 1) & (data[period_col] < data[adoption_period_col])
    ]
    n_anomalies = anomalies[unit_id].nunique()
    passed = n_anomalies == 0
    result = {
        "passed": passed,
        "value": n_anomalies,
        "threshold": 0,
        "test": "sme_takeup_precedes_lender_adoption",
        "warning": None,
    }
    if not passed:
        msg = (
            f"ADOPTION_TIMING_ANOMALY: {n_anomalies} SME(s) show green credit "
            f"take-up before their lender's recorded adoption date. This is likely "
            f"a data quality issue (mismatch in adoption timing records), not a "
            f"monotonicity violation. Review these observations before proceeding."
        )
        result["warning"] = msg
        warnings.warn(msg, AdoptionTimingAnomalyWarning, stacklevel=3)
    return result


def check_panel_length(
    data: pd.DataFrame,
    lender_id: str,
    period_col: str,
    adoption_period_col: str,
) -> dict:
    """L7: Minimum pre-treatment periods per cohort."""
    cohort_pre_periods = (
        data.groupby(lender_id)
        .apply(lambda g: (g[period_col] < g[adoption_period_col].iloc[0]).sum())
    )
    min_periods = int(cohort_pre_periods.min())
    short_cohorts = cohort_pre_periods[cohort_pre_periods < 3].index.tolist()
    passed = min_periods >= 3
    result = {
        "passed": passed,
        "value": min_periods,
        "threshold": 3,
        "test": "min_pre_treatment_periods",
        "cohorts_affected": short_cohorts if short_cohorts else [],
        "warning": None,
    }
    if min_periods < 2:
        raise InsufficientPreTreatmentError(
            f"Cohort(s) {short_cohorts} have fewer than 2 pre-treatment periods. "
            f"Parallel trends is untestable. Restrict sample to cohorts with "
            f"at least 2 pre-treatment periods, or aggregate to longer time periods."
        )
    elif min_periods < 3:
        msg = (
            f"ShortPanelWarning: cohort(s) {short_cohorts} have {min_periods} "
            f"pre-treatment period(s) (minimum 3 recommended for credible parallel "
            f"trends test). Consider Rambachan & Roth (2023) sensitivity analysis."
        )
        result["warning"] = msg
        warnings.warn(msg, ShortPanelWarning, stacklevel=3)
    return result


def check_kappa_weights(
    kappa_weights: "np.ndarray",
    design: str = "staggered_did",
) -> dict:
    """
    Checks the share of non-positive kappa weights.

    Thresholds are design-dependent. In staggered DiD, negative kappa values
    are structurally expected for never-takers in post-adoption periods
    (D=0, Z=1 when P(Z=1|t) < 1). A 17–25% negative share is typical and
    does not indicate a degenerate propensity model — it reflects comparison
    units identifying the direction of selection bias.

    In cross-sectional IV, negative kappa values are more anomalous and warrant
    tighter thresholds.

    Parameters
    ----------
    kappa_weights : np.ndarray
        Individual kappa_i weights from Abadie (2003) Equation 3.
    design : str
        "staggered_did" (default) or "cross_sectional".
        Controls warning and error thresholds.

    Thresholds
    ----------
    staggered_did:  warning > 30%, error > 45%
    cross_sectional: warning > 5%, error > 15%
    """
    _THRESHOLDS = {
        "staggered_did":   {"warning": 0.30, "error": 0.45},
        "cross_sectional": {"warning": 0.05, "error": 0.15},
    }
    if design not in _THRESHOLDS:
        raise ValueError(
            f"design must be 'staggered_did' or 'cross_sectional'. Got '{design}'."
        )
    thresh = _THRESHOLDS[design]

    share_nonpositive = float((kappa_weights <= 0).mean())
    passed = share_nonpositive <= thresh["warning"]
    result = {
        "passed": passed,
        "value": round(share_nonpositive, 4),
        "warning_threshold": thresh["warning"],
        "error_threshold": thresh["error"],
        "design": design,
        "test": "share_nonpositive_kappa_weights",
        "warning": None,
    }
    if share_nonpositive > thresh["error"]:
        raise KappaWeightDegeneracyError(
            f"{share_nonpositive:.1%} of kappa weights are non-positive "
            f"(error threshold for {design}: {thresh['error']:.0%}). "
            f"This exceeds the expected range for this design. "
            f"Review instrument specification and adoption timing data."
        )
    elif share_nonpositive > thresh["warning"]:
        msg = (
            f"KappaWeightWarning: {share_nonpositive:.1%} of kappa weights are "
            f"non-positive (warning threshold for {design}: {thresh['warning']:.0%}). "
            f"Complier profile estimates should be interpreted with caution."
        )
        result["warning"] = msg
        warnings.warn(msg, KappaWeightWarning, stacklevel=3)
    return result


def check_complier_share(complier_share: float) -> dict:
    """L3: Complier share below 20% produces noisy CATE estimates."""
    passed = complier_share >= 0.20
    result = {
        "passed": passed,
        "value": round(complier_share, 4),
        "threshold": 0.20,
        "test": "iv_complier_share",
        "warning": None,
    }
    if not passed:
        msg = (
            f"SmallComplierShareWarning: complier share is {complier_share:.1%} "
            f"(threshold: 20%). CATE estimates will be noisy. Report this as "
            f"a substantive finding about market structure: approximately "
            f"{complier_share:.0%} of SMEs would change their green credit "
            f"behaviour if their lender changed its product offering."
        )
        result["warning"] = msg
        warnings.warn(msg, SmallComplierShareWarning, stacklevel=3)
    return result


def check_overlap(propensity_scores: "np.ndarray", trim_threshold: float = 0.10) -> dict:
    """A6: Overlap check via propensity score trimming share."""
    trimmed_share = float(
        ((propensity_scores < trim_threshold) | (propensity_scores > 1 - trim_threshold)).mean()
    )
    passed = trimmed_share <= 0.10
    result = {
        "passed": passed,
        "value": round(trimmed_share, 4),
        "threshold": 0.10,
        "test": "propensity_trimming_share",
        "warning": None,
    }
    if not passed:
        msg = (
            f"OverlapWarning: {trimmed_share:.1%} of observations fall outside "
            f"the propensity score overlap region (threshold: 10%). "
            f"CATE estimates in sparse covariate regions are unreliable."
        )
        result["warning"] = msg
    return result
