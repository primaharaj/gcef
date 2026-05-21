"""
gcef.exceptions
---------------
All custom warnings and errors for the Green Credit Evaluation Framework.

Design principles:
- All warnings inherit from GCEFWarning (UserWarning)
- All errors inherit from GCEFError (Exception)
- Warnings and errors accept keyword metadata arguments stored as instance
  attributes, enabling structured assumption_tested schema (spec Option A)
- Old names are preserved as aliases for backward compatibility

Usage:
    # Simple raise/warn (existing style)
    raise WeakInstrumentError("F too low")

    # Structured (new style — metadata accessible downstream)
    raise WeakInstrumentError("F too low", f_statistic=7.3)
    w = ShortPanelWarning("short", cohorts_affected=["2021_cohort"])
"""


# ── Base classes with metadata support ────────────────────────────────────────

class GCEFError(Exception):
    """Base class for all GCEF errors. Accepts keyword metadata as attributes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        for k, v in kwargs.items():
            setattr(self, k, v)


class GCEFWarning(UserWarning):
    """Base class for all GCEF warnings. Accepts keyword metadata as attributes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        for k, v in kwargs.items():
            setattr(self, k, v)


# ── Errors ─────────────────────────────────────────────────────────────────────

class SingleLenderError(GCEFError):
    """
    Dataset contains only one unique lender_id.
    The staggered DiD instrument requires cross-lender variation.
    A single-lender dataset collapses the instrument.
    """
    pass


class WeakInstrumentError(GCEFError):
    """
    First-stage F-statistic falls below 10.
    A weak instrument produces biased LATE estimates.
    Metadata: f_statistic (float)
    """
    pass


class ShortPanelError(GCEFError):
    """
    A cohort has fewer than 2 pre-treatment periods.
    Parallel trends is untestable and estimation is unreliable.
    Metadata: cohorts_affected (list[str])
    """
    pass


class GeocodingRequiredError(GCEFError):
    """
    Shock-conditioned outcomes requested but SME location data is absent.
    """
    pass


class BlendedTreatmentNotImplemented(GCEFError, NotImplementedError):
    """
    TreatmentType.BLENDED is not implemented in v0.1.
    Decompose the instrument into constituent treatment types and run
    separate evaluations, or contribute via the GitHub repository.

    The error message contains the wording required by spec Section 4.1.
    """
    def __init__(self, message: str = ""):
        if not message:
            message = (
                "Blended treatment types require multi-pathway DAG specification, "
                "which is not implemented in GCEF v0.1. "
                "Decompose the instrument into constituent treatment types and run "
                "separate GreenCreditEvaluator instances, or contribute a blended "
                "treatment implementation via the GitHub repository."
            )
        super().__init__(message)


class KappaWeightError(GCEFError):
    """
    More than 15% of kappa weights are non-positive.
    The complier restriction is unreliable.
    Metadata: share_nonpositive (float)
    """
    pass


class InsufficientRevenueHistoryError(GCEFError):
    """
    One or more firms have fewer than stability_window periods of revenue data.
    Cannot compute rolling CV for these firms.
    Metadata: affected_sme_ids (list[str])
    """
    pass


class AdoptionTimingAnomalyError(GCEFError):
    """
    SMEs show green credit take-up before their lender's adoption date.
    Likely a data quality issue — check lender adoption timing records.
    Metadata: anomaly_count (int)
    """
    pass


# ── Backward-compatible aliases for existing code ─────────────────────────────

# Old name → kept as alias so existing imports continue to work
BlendedTreatmentNotImplementedError = BlendedTreatmentNotImplemented
InsufficientPreTreatmentError = ShortPanelError
KappaWeightDegeneracyError = KappaWeightError


# ── Warnings ──────────────────────────────────────────────────────────────────

class ShortPanelWarning(GCEFWarning):
    """
    A cohort has fewer than 3 but at least 2 pre-treatment periods.
    Parallel trends assumption is testable but marginal.
    Analyst should consider Rambachan & Roth (2023) sensitivity analysis.
    Metadata: cohorts_affected (list[str])
    """
    pass


class WeakInstrumentWarning(GCEFWarning):
    """
    First-stage F-statistic is between 10 and 20.
    Estimation proceeds but results should be interpreted with caution.
    Metadata: f_statistic (float)
    """
    pass


class ResilienceIndexWarning(GCEFWarning):
    """
    Fewer than underidentification_threshold components available.
    The resilience index is underidentified.
    """
    pass


class UserDerivedStabilityWarning(GCEFWarning):
    """
    Both a level column (e.g. 'revenue') and its derived counterpart
    ('revenue_stability') are present. Pre-computed column takes precedence.
    Analyst must document their derivation in the reproducibility statement.
    """
    pass


class InsufficientRevenueHistoryWarning(GCEFWarning):
    """
    A firm has fewer than stability_window periods of data.
    Excluded from the resilience index for incomplete rolling window periods.
    """
    pass


class SmallComplierShareWarning(GCEFWarning):
    """
    Complier share falls below 20% of the sample.
    CATE estimates will be noisy.
    Report as a substantive finding about market structure, not a data quality flag.
    Metadata: complier_share (float)
    """
    pass


class KappaWeightWarning(GCEFWarning):
    """
    Between 5% and 15% of kappa weights are non-positive.
    Complier profile estimates should be interpreted with caution.
    Metadata: share_nonpositive (float)
    """
    pass


class GeocodeResolutionWarning(GCEFWarning):
    """
    Geocoding at administrative unit level only (not lat/lon).
    Framework falls back to administrative unit centroid.
    Shock resolution degrades; confidence intervals expand.
    """
    pass


class SelfReportedTreatmentWarning(GCEFWarning):
    """
    verification_method or conditionality_mechanism is 'self_reported'.
    Treatment assignment may be endogenous even conditional on lender adoption.
    Sensitivity analysis with a conservative treatment definition is recommended.
    """
    pass


class NuisanceModelOverrideWarning(GCEFWarning):
    """
    Analyst has overridden default nuisance model(s).
    Override is logged in results.estimand and the report reproducibility statement.
    """
    pass


class AdoptionTimingAnomalyWarning(GCEFWarning):
    """
    SME take-up precedes lender adoption date.
    Likely a data quality issue (adoption timing mismatch), not a monotonicity violation.
    """
    pass


class ComplierShareWarning(GCEFWarning):
    """Backward-compatible alias for SmallComplierShareWarning."""
    pass


class GeocodingResolutionWarning(GCEFWarning):
    """Backward-compatible alias for GeocodeResolutionWarning."""
    pass
