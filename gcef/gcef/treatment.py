"""
gcef.treatment
--------------
GreenCreditTreatment: represents the treatment as a structured vector,
not a binary scalar. The package selects estimator components based on
treatment specification rather than forcing one DAG on all use cases.

Design decision DD1 (spec): Treatment is a vector, not a scalar.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from gcef.exceptions import BlendedTreatmentNotImplementedError


class TreatmentType(str, Enum):
    """
    The type of green credit instrument.

    Each type has a distinct causal pathway from lender product design
    to SME behaviour. Do not pool across types without justification —
    they require different identification strategies.
    """
    RATE_REDUCTION = "rate_reduction"
    """
    Interest rate reduced conditional on green use of funds.
    Pathway: lender prices green behaviour → SME cost of capital reduced
    → SME more likely to invest in green assets.
    """
    EQUIPMENT_LOAN = "equipment_loan"
    """
    Loan specifically for renewable or green equipment purchase.
    Pathway: lender restricts use of funds → SME investment directed
    toward green capital.
    """
    GREEN_COVENANT = "green_covenant"
    """
    Standard loan with environmental covenant attached.
    Pathway: lender monitors compliance → SME maintains green
    practices to avoid covenant breach.
    """
    SECTOR_RESTRICTED = "sector_restricted"
    """
    Loan restricted to firms in designated green sectors.
    Pathway: lender selects into a sector, not changing SME behaviour.
    This type is most susceptible to selection effects — the lender
    is not intervening in investment decisions, only restricting
    eligibility. Treat with extra caution.
    """
    BLENDED = "blended"
    """
    Composite instrument combining multiple treatment types.
    NOT IMPLEMENTED in v0.1. Raises BlendedTreatmentNotImplementedError.
    """


class ConditionalityMechanism(str, Enum):
    """How the green conditionality is enforced."""
    VERIFIED_INVESTMENT = "verified_investment"
    """Disbursement or rate conditional on verified green investment."""
    SECTOR_CLASSIFICATION = "sector_classification"
    """Eligibility based on sector classification only."""
    COVENANT_COMPLIANCE = "covenant_compliance"
    """Ongoing compliance monitoring."""
    SELF_REPORTED = "self_reported"
    """SME self-reports green use; not independently verified."""


class VerificationMethod(str, Enum):
    """How green use is verified."""
    SITE_VISIT = "site_visit"
    """Physical verification by lender or third party."""
    DOCUMENT_REVIEW = "document_review"
    """Invoice/receipt verification."""
    SATELLITE_REMOTE_SENSING = "satellite_remote_sensing"
    """Remote sensing verification (e.g. solar installation detection)."""
    SELF_REPORTED = "self_reported"
    """No independent verification."""


@dataclass
class GreenCreditTreatment:
    """
    Structured representation of a green credit treatment instrument.

    Captures the treatment as a vector: type, conditionality mechanism,
    verification method, and intensity. The GreenCreditEvaluator selects
    pipeline components based on this specification.

    Parameters
    ----------
    type : TreatmentType
        The category of green credit instrument. Determines the
        causal pathway and DAG structure.
    conditionality_mechanism : ConditionalityMechanism
        How the green conditionality is enforced.
    verification_method : VerificationMethod
        How green use is verified by the lender.
    intensity : Optional[float]
        Quantitative measure of treatment intensity where applicable.
        For rate_reduction: the interest rate differential (e.g. 0.03
        for a 3% reduction). None if not applicable or unknown.

    Raises
    ------
    BlendedTreatmentNotImplementedError
        If type is TreatmentType.BLENDED. Deferred to v0.2.

    Notes
    -----
    A rate_reduction + verified_investment + site_visit treatment has a
    fundamentally different causal pathway than sector_restricted +
    self_reported. Do not pool without explicit justification.

    Examples
    --------
    >>> treatment = GreenCreditTreatment(
    ...     type=TreatmentType.RATE_REDUCTION,
    ...     conditionality_mechanism=ConditionalityMechanism.VERIFIED_INVESTMENT,
    ...     verification_method=VerificationMethod.DOCUMENT_REVIEW,
    ...     intensity=0.03
    ... )
    """

    type: TreatmentType
    conditionality_mechanism: ConditionalityMechanism
    verification_method: VerificationMethod
    intensity: Optional[float] = None

    def __post_init__(self):
        if self.type == TreatmentType.BLENDED:
            raise BlendedTreatmentNotImplementedError(
                "Blended treatment types require multi-pathway DAG specification, "
                "which is not implemented in GCEF v0.1. "
                "Decompose the instrument into constituent treatment types and run "
                "separate GreenCreditEvaluator instances, or contribute a blended "
                "treatment implementation at https://github.com/[repo]/gcef."
            )

    @property
    def issues_self_reported_warning(self) -> bool:
        """True if this treatment spec should trigger SelfReportedTreatmentWarning."""
        return (
            self.verification_method == VerificationMethod.SELF_REPORTED
            or self.conditionality_mechanism == ConditionalityMechanism.SELF_REPORTED
        )

    @property
    def is_selection_dominant(self) -> bool:
        """
        True for treatment types where selection effects dominate the
        causal pathway. Sector-restricted loans select into a sector
        rather than changing SME investment behaviour.
        """
        return self.type == TreatmentType.SECTOR_RESTRICTED

    def to_description(self) -> str:
        """Human-readable description for Estimand and report module."""
        parts = [f"Type: {self.type.value}"]
        parts.append(f"Conditionality: {self.conditionality_mechanism.value}")
        parts.append(f"Verification: {self.verification_method.value}")
        if self.intensity is not None:
            parts.append(f"Intensity: {self.intensity}")
        return " | ".join(parts)
