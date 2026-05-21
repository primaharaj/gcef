"""
gcef.pipeline
-------------
GreenCreditEvaluator: the main pipeline object. Orchestrates the full
two-stage IV/DiD → causal forest pipeline as a single .fit() call.

Stage 1 → Stage 2 data flow (spec Section 5.2):
    Stage 1 outputs: Stage1Result dataclass (late, kappa_weights, ...)
    Stage 2 consumes: kappa_weights as sample_weight to ForestDRIV

The complier restriction is implemented by passing kappa_weights as
sample_weight. Always-takers and never-takers receive near-zero weight.

Spec reference: Section 4.3 (GreenCreditEvaluator), Section 5 (pipeline)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, List

import pandas as pd
import numpy as np

from gcef.treatment import GreenCreditTreatment
from gcef.outcomes import ResilienceIndex
from gcef.estimand import Estimand, make_complier_estimand
from gcef.exceptions import SelfReportedTreatmentWarning, NuisanceModelOverrideWarning
from gcef import assumptions as checks


#: Threshold above which a kappa weight is considered "complier-identified".
#: Used to produce results.cate_complier_mask.
COMPLIER_KAPPA_THRESHOLD = 0.1


@dataclass
class GCEFResults:
    """
    Results object returned by GreenCreditEvaluator.fit().

    All estimates carry an explicit estimand. Never interpret a number
    from this object without reading results.estimand first.
    """

    late: Optional[dict] = None
    """
    LATE estimate with standard errors — defined for compliers only.
    Keys: 'estimate', 'se', 'ci_lower', 'ci_upper', 'estimand'
    """

    cate: Optional[pd.DataFrame] = None
    """
    CATE per observation — full-length array (one row per input row).
    κ-weighted so always-takers and never-takers receive near-zero weight,
    but all rows are present for portfolio join convenience.
    Columns: unit_id, cate_estimate, cate_se, kappa_weight, estimand (str).
    """

    cate_complier_mask: Optional[pd.Series] = None
    """
    Boolean mask — True where kappa_weight > COMPLIER_KAPPA_THRESHOLD.

    Usage:
        complier_cate = results.cate[results.cate_complier_mask]
        full_portfolio = results.cate.copy()
        full_portfolio['is_complier'] = results.cate_complier_mask
    """

    complier_profile: Optional[pd.DataFrame] = None
    """
    Covariate distribution (κ-weighted) for complier subpopulation.
    Columns: covariate, weighted_mean, weighted_median, iqr_25, iqr_75,
             complier_to_full_ratio
    """

    complier_share: Optional[dict] = None
    """
    Share of sample classified as compliers, with confidence interval.
    Keys: 'estimate', 'ci_lower', 'ci_upper'
    """

    cate_bounds: Optional[pd.DataFrame] = None
    """
    Manski (1990) partial identification bounds for always-takers and
    never-takers. BOUNDS, not point estimates.
    Columns: unit_id, lower_bound, upper_bound, population, estimand
    """

    estimand: Optional[Estimand] = None
    """Non-optional structured estimand. Read before interpreting any field."""

    assumptions_tested: dict = field(default_factory=dict)
    """
    Structured assumption test results. Schema: Option A from spec.
    Every key maps to one assumption check with: passed, value,
    threshold, test, warning.
    """

    def report(
        self,
        output_format: str = "pdf",
        output_path: str = "./gcef_report.pdf",
    ) -> None:
        """Generate a structured PDF or HTML report for DFI analysts."""
        from gcef.report import ReportGenerator
        ReportGenerator(results=self).generate(
            output_format=output_format,
            output_path=output_path,
        )


class GreenCreditEvaluator:
    """
    Main pipeline object. Takes treatment and outcome specifications,
    runs the full two-stage IV/DiD → causal forest pipeline.

    Parameters
    ----------
    treatment : GreenCreditTreatment
        Structured treatment specification.
    outcome : ResilienceIndex
        Outcome index specification.
    unit_id : str
        Column name for SME identifier.
    time_id : str
        Column name for time period.
    lender_id : str
        Column name for lender identifier.
    adoption_time : str
        Column name for lender green product adoption period.
    covariates : list[str]
        Covariate column names for Stage 2 CATE estimation.
    model_propensity : optional
        Override for the P(D=1|X) nuisance model in ForestDRIV.
        Default: GradientBoostingClassifier.
    model_outcome : optional
        Override for E[Y|X,Z] nuisance model in ForestDRIV.
        Default: GradientBoostingRegressor.
    random_seed : Optional[int]
        Random seed for reproducibility.

    Examples
    --------
    >>> evaluator = GreenCreditEvaluator(
    ...     treatment=treatment, outcome=outcome,
    ...     unit_id="sme_id", time_id="period",
    ...     lender_id="lender_id", adoption_time="lender_green_adoption_period",
    ...     covariates=["firm_age", "sector", "region", "firm_size", "prior_revenue"],
    ... )
    >>> results = evaluator.fit(data)
    """

    def __init__(
        self,
        treatment: GreenCreditTreatment,
        outcome: ResilienceIndex,
        unit_id: str,
        time_id: str,
        lender_id: str,
        adoption_time: str,
        covariates: List[str],
        model_propensity=None,
        model_outcome=None,
        random_seed: Optional[int] = None,
    ):
        self.treatment = treatment
        self.outcome = outcome
        self.unit_id = unit_id
        self.time_id = time_id
        self.lender_id = lender_id
        self.adoption_time = adoption_time
        self.covariates = covariates
        self.random_seed = random_seed

        self._model_propensity_overridden = model_propensity is not None
        self._model_outcome_overridden = model_outcome is not None

        if self._model_propensity_overridden or self._model_outcome_overridden:
            warnings.warn(
                "Default nuisance model(s) have been overridden. "
                "This will be logged in results.estimand and the report "
                "reproducibility statement.",
                NuisanceModelOverrideWarning,
                stacklevel=2,
            )

        from sklearn.ensemble import GradientBoostingRegressor
        # ForestDRIV with continuous treatment (discrete_treatment=False, the default)
        # requires regressors — not classifiers — for both nuisance models.
        # model_propensity here is the E[T|X,W] nuisance model (treatment regression),
        # not a probability model. The overlap check uses a separate cross-fitted
        # propensity score estimated in stage2._estimate_propensity().
        self.model_propensity = model_propensity or GradientBoostingRegressor(
            random_state=random_seed
        )
        self.model_outcome = model_outcome or GradientBoostingRegressor(
            random_state=random_seed
        )

    def fit(self, data: pd.DataFrame) -> GCEFResults:
        """
        Run the full two-stage pipeline on the provided dataset.

        Stage 1: IV/DiD with staggered adoption → Stage1Result (LATE + kappa weights)
        Stage 2: ForestDRIV with kappa-weighted sample_weight → CATE

        Parameters
        ----------
        data : pd.DataFrame
            Panel dataset meeting the minimum data requirements in
            spec Section 7.

        Returns
        -------
        GCEFResults
        """
        from gcef.stage1 import run_stage1
        from gcef.stage2 import run_stage2
        from gcef.bounds import compute_manski_bounds

        results = GCEFResults()

        # ── Pre-flight checks ──────────────────────────────────────────────────
        results.assumptions_tested["single_lender"] = checks.check_single_lender(
            data, self.lender_id
        )

        if self.treatment.issues_self_reported_warning:
            warnings.warn(
                "Treatment uses self-reported verification or conditionality. "
                "Treatment assignment may be endogenous. Run sensitivity analysis "
                "with a conservative treatment definition.",
                SelfReportedTreatmentWarning,
                stacklevel=2,
            )

        results.assumptions_tested["adoption_timing_anomaly"] = (
            checks.check_adoption_timing_anomaly(
                data,
                unit_id=self.unit_id,
                takeup_col="green_credit_takeup",
                adoption_period_col=self.adoption_time,
                period_col=self.time_id,
            )
        )

        results.assumptions_tested["panel_length"] = checks.check_panel_length(
            data,
            lender_id=self.lender_id,
            period_col=self.time_id,
            adoption_period_col=self.adoption_time,
        )

        # ── Derive stability columns and build resilience index ────────────────
        data = self.outcome.derive_stability_columns(data, unit_id=self.unit_id)
        data["_resilience_index"] = self.outcome.build(data, unit_id=self.unit_id)

        # ── Construct instrument column (shared by Stage 1 and Stage 2) ───────
        # _Z = 1 if the SME's lender had adopted a green product by period t.
        # Built here so both stages receive the same column without duplication.
        data["_Z"] = (data[self.time_id] >= data[self.adoption_time]).astype(int)

        # ── Stage 1: IV/DiD → LATE + kappa weights ────────────────────────────
        stage1_result = run_stage1(
            data=data,
            unit_id=self.unit_id,
            time_id=self.time_id,
            lender_id=self.lender_id,
            adoption_time=self.adoption_time,
            outcome_col="_resilience_index",
            takeup_col="green_credit_takeup",
            covariates=self.covariates,
            shock_instrument=self.outcome.shock_instrument,
            shock_threshold=self.outcome.shock_threshold,
            random_seed=self.random_seed,
        )

        results.assumptions_tested["instrument_relevance"] = (
            checks.check_instrument_relevance(stage1_result.f_statistic)
        )
        results.assumptions_tested["kappa_weight_negatives"] = (
            checks.check_kappa_weights(
                stage1_result.kappa_weights, design="staggered_did"
            )
        )
        results.assumptions_tested["complier_share"] = (
            checks.check_complier_share(stage1_result.complier_share)
        )

        results.late = {
            "estimate": stage1_result.late,
            "se": stage1_result.late_se,
            "ci_lower": stage1_result.late_ci[0],
            "ci_upper": stage1_result.late_ci[1],
        }
        results.complier_share = stage1_result.complier_share_ci
        results.complier_profile = stage1_result.complier_profile

        # ── Stage 2: ForestDRIV (kappa-weighted) → CATE ───────────────────────
        stage2_output = run_stage2(
            data=data,
            unit_id=self.unit_id,
            outcome_col="_resilience_index",
            takeup_col="green_credit_takeup",
            shock_instrument=self.outcome.shock_instrument,
            shock_threshold=self.outcome.shock_threshold,
            covariates=self.covariates,
            kappa_weights=stage1_result.kappa_weights,
            model_propensity=self.model_propensity,
            model_outcome=self.model_outcome,
            random_seed=self.random_seed,
        )

        results.assumptions_tested["overlap"] = checks.check_overlap(
            stage2_output["propensity_scores"]
        )
        results.cate = stage2_output["cate"]

        # Populate complier mask from kappa weights (spec v0.7 Section 4.3)
        results.cate_complier_mask = pd.Series(
            stage1_result.kappa_weights > COMPLIER_KAPPA_THRESHOLD,
            index=data.index,
            name="is_complier",
        )

        # ── Manski bounds for always-takers / never-takers ────────────────────
        results.cate_bounds = compute_manski_bounds(
            data=data,
            outcome_col="_resilience_index",
            takeup_col="green_credit_takeup",
            unit_id=self.unit_id,
            kappa_weights=stage1_result.kappa_weights,
            treatment=self.treatment,
            outcome=self.outcome,
        )

        # ── Build estimand ─────────────────────────────────────────────────────
        nuisance_overrides = {}
        if self._model_propensity_overridden or self._model_outcome_overridden:
            nuisance_overrides = {
                "model_propensity": type(self.model_propensity).__name__,
                "model_outcome": type(self.model_outcome).__name__,
            }

        results.estimand = make_complier_estimand(
            treatment=self.treatment,
            outcome=self.outcome,
            nuisance_model_overrides=nuisance_overrides,
            random_seed=self.random_seed,
        )

        return results
