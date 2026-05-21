"""
gcef.stage2
-----------
Stage 2 of the GCEF pipeline: ForestDRIV causal forest restricted to
compliers via kappa-weighted sample weights.

econml API notes (v0.16)
------------------------
ForestDRIV constructor parameters relevant to GCEF:
    model_y_xw  : outcome nuisance model E[Y|X,W]
    model_t_xw  : treatment propensity nuisance model P(T|X,W)
    n_estimators: number of trees (default 1000; use 200 for speed)
    random_state : random seed
    cv           : cross-fitting folds (default 2)
    honest       : honest splitting (default True — keep True)

ForestDRIV.fit(Y, T, *, Z, X, W, sample_weight)
    sample_weight : kappa weights (complier restriction)

ForestDRIV.effect_inference(X) → NormalInferenceResults
    .point_estimate : np.ndarray shape (n,) — CATE point estimates
    .conf_int()     : tuple (lower, upper) each shape (n,)
    NB: .mean_pred_stderr is None for ForestDRIV — use conf_int() for
    per-observation uncertainty. Population-level SE via .population_summary().

Spec reference: Section 5.2, DD3, DD3a, L8
"""
from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────

MIN_SHOCK_PERIOD_OBS = 50
MIN_SHOCK_PERIOD_COMPLIERS = 20


# ── Main entry point ───────────────────────────────────────────────────────────

def run_stage2(
    data: pd.DataFrame,
    unit_id: str,
    outcome_col: str,
    takeup_col: str,
    shock_instrument: str,
    shock_threshold: float,
    covariates: List[str],
    kappa_weights: np.ndarray,
    model_propensity,
    model_outcome,
    random_seed: Optional[int] = None,
) -> dict:
    """
    Stage 2: ForestDRIV causal forest restricted to compliers via kappa weights.

    Parameters
    ----------
    data : pd.DataFrame
        Full panel with _Z column (constructed in pipeline.py).
    unit_id, outcome_col, takeup_col, shock_instrument, shock_threshold,
    covariates, kappa_weights, model_propensity, model_outcome, random_seed:
        See pipeline.py docstring.

    Returns
    -------
    dict:
        cate : pd.DataFrame — full-length CATE output
        propensity_scores : np.ndarray — P(D=1|X) for overlap check
    """
    from econml.iv.dr import ForestDRIV

    data = data.copy().reset_index(drop=True)

    if "_Z" not in data.columns:
        raise RuntimeError(
            "Stage 2 requires '_Z' instrument column. "
            "It should be constructed in pipeline.py before calling run_stage2."
        )

    # ── 1. Encode covariates ──────────────────────────────────────────────────
    available_covariates = [c for c in covariates if c in data.columns]
    X_full = _encode_covariates(data, available_covariates)

    # ── 2. Shock-period subsetting ────────────────────────────────────────────
    if shock_instrument in data.columns:
        shock_mask = (data[shock_instrument] < shock_threshold).values
    else:
        warnings.warn(
            f"Shock instrument '{shock_instrument}' not found. "
            f"Falling back to full sample (no shock conditioning).",
            stacklevel=2,
        )
        shock_mask = np.ones(len(data), dtype=bool)

    if shock_mask.sum() < MIN_SHOCK_PERIOD_OBS:
        warnings.warn(
            f"Only {shock_mask.sum()} shock-period observations "
            f"(min: {MIN_SHOCK_PERIOD_OBS}). Using full sample.",
            stacklevel=2,
        )
        shock_mask = np.ones(len(data), dtype=bool)

    # ── 3. Build shock-period arrays ──────────────────────────────────────────
    shock_idx = np.where(shock_mask)[0]
    Y = data.loc[shock_idx, outcome_col].fillna(0).values.astype(float)
    T = data.loc[shock_idx, takeup_col].values.astype(float)
    Z = data.loc[shock_idx, "_Z"].values.astype(float)
    X = X_full.iloc[shock_idx].values.astype(float)

    # Zero negative kappa weights — negative = not identified as complier
    kappa_shock = np.clip(kappa_weights[shock_idx], 0.0, None)

    n_complier_shock = (kappa_shock > 0).sum()
    if n_complier_shock < MIN_SHOCK_PERIOD_COMPLIERS:
        warnings.warn(
            f"Only {n_complier_shock} complier-identified observations in "
            f"shock periods (min: {MIN_SHOCK_PERIOD_COMPLIERS}). "
            f"CATE estimates will be unreliable.",
            stacklevel=2,
        )

    # ── 4. Fit ForestDRIV ─────────────────────────────────────────────────────
    # econml v0.16: nuisance model parameters are model_y_xw and model_t_xw
    forest = ForestDRIV(
        model_y_xw=model_outcome,
        model_t_xw=model_propensity,
        n_estimators=200,
        min_samples_leaf=10,
        random_state=random_seed,
        cv=2,
        verbose=0,
    )

    try:
        forest.fit(
            Y=Y,
            T=T,
            Z=Z,
            X=X,
            W=None,
            sample_weight=kappa_shock,
        )
    except Exception as e:
        raise RuntimeError(
            f"ForestDRIV fitting failed: {e}. "
            f"Check that shock-period subset has variation in T and Z, "
            f"and covariates have no NaN values."
        ) from e

    # ── 5. Predict CATE and CI for full dataset ───────────────────────────────
    # effect() on full X so results.cate is full-length (one row per input row)
    X_all = X_full.values.astype(float)

    try:
        cate_point = forest.effect(X_all)               # shape (n,)
        inf = forest.effect_inference(X_all)
        ci = inf.conf_int()                              # tuple (lower, upper)
        cate_ci_lower = ci[0]                            # shape (n,)
        cate_ci_upper = ci[1]                            # shape (n,)
        # Derive SE from CI: SE ≈ (upper - lower) / (2 * 1.96)
        cate_se = (cate_ci_upper - cate_ci_lower) / (2 * 1.96)
    except Exception as e:
        raise RuntimeError(f"ForestDRIV inference failed: {e}.") from e

    # ── 6. Propensity scores for overlap check ────────────────────────────────
    propensity_scores = _estimate_propensity(
        T_full=data[takeup_col].values.astype(float),
        X_full=X_all,
        model_propensity=model_propensity,
        random_seed=random_seed,
    )

    # ── 7. Assemble CATE DataFrame (full-length) ──────────────────────────────
    cate_df = pd.DataFrame({
        unit_id: data[unit_id].values,
        "cate_estimate": cate_point,
        "cate_se": cate_se,
        "cate_ci_lower": cate_ci_lower,
        "cate_ci_upper": cate_ci_upper,
        "kappa_weight": kappa_weights,
        "in_shock_period": shock_mask,
    }, index=data.index)

    return {
        "cate": cate_df,
        "propensity_scores": propensity_scores,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _encode_covariates(data: pd.DataFrame, covariates: List[str]) -> pd.DataFrame:
    """One-hot encode categoricals, fill numeric NaNs with median."""
    if not covariates:
        return pd.DataFrame({"_intercept": np.ones(len(data))}, index=data.index)

    frames = []
    for col in covariates:
        if col not in data.columns:
            continue
        series = data[col]
        if (
            series.dtype == object
            or str(series.dtype) in ("category", "string")
            or pd.api.types.is_string_dtype(series)
        ):
            dummies = pd.get_dummies(series, prefix=col, drop_first=True)
            frames.append(dummies)
        else:
            frames.append(series.fillna(series.median()).rename(col))

    if not frames:
        return pd.DataFrame({"_intercept": np.ones(len(data))}, index=data.index)

    result = pd.concat(frames, axis=1)
    return result.apply(pd.to_numeric, errors="coerce").fillna(0)


def _estimate_propensity(
    T_full: np.ndarray,
    X_full: np.ndarray,
    model_propensity,
    random_seed: Optional[int],
) -> np.ndarray:
    """
    Cross-fitted P(D=1|X) for the overlap check.

    This is separate from the ForestDRIV nuisance model. ForestDRIV uses
    E[T|X,W] as a continuous regression (model_t_xw). The overlap check
    needs P(D=1|X) as a probability — so we always use a GradientBoostingClassifier
    here, regardless of what model_propensity was passed to ForestDRIV.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict

    # Always use a classifier for the overlap propensity score
    clf = GradientBoostingClassifier(
        n_estimators=100,
        random_state=random_seed,
    )
    T_binary = (T_full > 0.5).astype(int)

    try:
        propensity = cross_val_predict(
            clf, X_full, T_binary, cv=3, method="predict_proba"
        )[:, 1]
    except Exception:
        clf.fit(X_full, T_binary)
        propensity = clf.predict_proba(X_full)[:, 1]

    return np.clip(propensity, 1e-6, 1 - 1e-6)
