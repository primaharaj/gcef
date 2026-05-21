"""
gcef — Green Credit Evaluation Framework
-----------------------------------------
Causal impact evaluation of green credit instruments on SME resilience
in Sub-Saharan African lending markets.

Specification: https://github.com/[repo]/gcef/blob/main/SPECIFICATION.md
Version: 0.1.0 (pre-release — specification complete, implementation in progress)

Quick start
-----------
>>> from gcef import GreenCreditEvaluator, GreenCreditTreatment, ResilienceIndex
>>> from gcef.treatment import TreatmentType, ConditionalityMechanism, VerificationMethod

>>> treatment = GreenCreditTreatment(
...     type=TreatmentType.RATE_REDUCTION,
...     conditionality_mechanism=ConditionalityMechanism.VERIFIED_INVESTMENT,
...     verification_method=VerificationMethod.DOCUMENT_REVIEW,
...     intensity=0.03,
... )
>>> outcome = ResilienceIndex(
...     columns={"revenue": 0.40, "loan_repayment_rate": 0.30,
...              "employment": 0.20, "adaptation_investment": 0.10},
...     shock_instrument="rainfall_anomaly_lag1",
... )
>>> evaluator = GreenCreditEvaluator(
...     treatment=treatment, outcome=outcome,
...     unit_id="sme_id", time_id="period",
...     lender_id="lender_id", adoption_time="lender_green_adoption_period",
...     covariates=["firm_age", "sector", "region", "firm_size", "prior_revenue"],
... )
>>> results = evaluator.fit(data)
>>> print(results.estimand.to_prose())
>>> results.report(output_format="pdf", output_path="./gcef_report.pdf")
"""

__version__ = "0.1.0"

from gcef.treatment import GreenCreditTreatment, TreatmentType, ConditionalityMechanism, VerificationMethod
from gcef.outcomes import ResilienceIndex
from gcef.estimand import Estimand
from gcef.pipeline import GreenCreditEvaluator, GCEFResults

__all__ = [
    "GreenCreditEvaluator",
    "GCEFResults",
    "GreenCreditTreatment",
    "TreatmentType",
    "ConditionalityMechanism",
    "VerificationMethod",
    "ResilienceIndex",
    "Estimand",
]
