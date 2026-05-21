"""
gcef.estimand
-------------
The Estimand dataclass is attached to every output object produced by GCEF.
It is non-optional and machine-readable.

Kept in its own module to avoid circular imports: report.py, stage1.py,
and stage2.py all need Estimand without pulling in the full pipeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING


# ── Constants ──────────────────────────────────────────────────────────────────

VALID_POPULATIONS = ["compliers", "always_takers", "never_takers", "full_sample"]

#: Populations for which the instrument provides no identifying variation.
#: These receive extrapolation_flag=True automatically.
_UNIDENTIFIED_POPULATIONS = {"always_takers", "never_takers"}

DFI_FRAMING_DEFAULTS = {
    "compliers": (
        "Marginal SMEs — those whose green credit behaviour changes with programme "
        "availability. Relevant for additionality assessment. These estimates describe "
        "the population for whom programme expansion would change access."
    ),
    "always_takers": (
        "SMEs who would access green credit regardless of lender product availability. "
        "The instrument provides no identifying variation for this subpopulation. "
        "These are Manski (1990) partial identification bounds, not point estimates."
    ),
    "never_takers": (
        "SMEs who would not access green credit regardless of lender product availability. "
        "The instrument provides no identifying variation for this subpopulation. "
        "These are Manski (1990) partial identification bounds, not point estimates."
    ),
    "full_sample": (
        "Full portfolio including compliers, always-takers, and never-takers. "
        "Point estimates apply only to the complier subpopulation. "
        "Full-portfolio figures combine identified estimates with partial bounds."
    ),
}


# ── Estimand dataclass ─────────────────────────────────────────────────────────

@dataclass
class Estimand:
    """
    Structured description of what a GCEF estimate measures, who it applies
    to, and what identification strategy produced it.

    Appears on every results object: results.late, results.cate,
    results.complier_profile, results.cate_bounds.

    Parameters
    ----------
    population : str
        One of VALID_POPULATIONS: "compliers", "always_takers",
        "never_takers", "full_sample".
    population_description : str
        Plain-language description of the population.
    identification : str
        Description of the identification strategy used.
    dfi_framing : str | None
        Description in DFI vocabulary. If None, defaults to
        DFI_FRAMING_DEFAULTS[population].
    treatment : any | None
        Treatment object. Serialised via to_dict() if available, else repr().
    outcome : any | None
        Outcome object. Serialised via to_dict() if available, else repr().
    nuisance_model_overrides : dict
        Analyst-supplied nuisance model overrides. Empty dict = using defaults.
    gcef_version : str
        Package version that produced this estimate.
    random_seed : int | None
        Random seed used. None if not set.
    data_hash : str | None
        SHA-256 hash of input dataframe for reproducibility.
    notes : list[str]
        Any analyst-added notes.
    """

    population: str
    population_description: str
    identification: str
    dfi_framing: Optional[str] = None
    treatment: Optional[Any] = None
    outcome: Optional[Any] = None
    nuisance_model_overrides: dict = field(default_factory=dict)
    gcef_version: str = "0.1.0"
    random_seed: Optional[int] = None
    data_hash: Optional[str] = None
    notes: list = field(default_factory=list)

    # Set post-init — not a constructor parameter
    extrapolation_flag: bool = field(init=False)

    def __post_init__(self):
        if self.population not in VALID_POPULATIONS:
            raise ValueError(
                f"population must be one of {VALID_POPULATIONS}. "
                f"Got: '{self.population}'"
            )
        # Auto-set extrapolation flag for unidentified subpopulations
        self.extrapolation_flag = self.population in _UNIDENTIFIED_POPULATIONS

        # Apply default DFI framing if not supplied
        if self.dfi_framing is None:
            self.dfi_framing = DFI_FRAMING_DEFAULTS.get(self.population, "")

    # ── Equality and hashing ──────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        """
        Equality based on population and identification only.
        Two estimands describing the same population with the same
        identification strategy are considered identical even if their
        prose descriptions differ.
        """
        if not isinstance(other, Estimand):
            return False
        return (
            self.population == other.population
            and self.identification == other.identification
        )

    def __hash__(self) -> int:
        return hash((self.population, self.identification))

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Serialise the estimand to a plain dictionary.
        Treatment and outcome objects are serialised via their own to_dict()
        if available; otherwise via repr().
        """
        def _serialise(obj: Any) -> Any:
            if obj is None:
                return None
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            return repr(obj)

        return {
            "population": self.population,
            "population_description": self.population_description,
            "dfi_framing": self.dfi_framing,
            "identification": self.identification,
            "treatment": _serialise(self.treatment),
            "outcome": _serialise(self.outcome),
            "nuisance_model_overrides": self.nuisance_model_overrides,
            "extrapolation_flag": self.extrapolation_flag,
            "gcef_version": self.gcef_version,
            "random_seed": self.random_seed,
            "data_hash": self.data_hash,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_prose(self) -> str:
        """
        Human-readable prose rendering, suitable for the report module's
        estimand statement section.
        """
        lines = [
            f"Population ({self.population}): {self.population_description}",
            "",
            f"For DFI decision-making: {self.dfi_framing}",
            "",
            f"Identification strategy: {self.identification}",
        ]
        if self.treatment is not None:
            lines += ["", f"Treatment: {self.treatment}"]
        if self.outcome is not None:
            lines += ["", f"Outcome: {self.outcome}"]
        if self.extrapolation_flag:
            lines += [
                "",
                "NOTE: These are partial identification bounds, not identified "
                "point estimates. The instrument provides no variation for this "
                "subpopulation. These results are not identified from the data.",
            ]
        if self.nuisance_model_overrides:
            lines += [
                "",
                f"Nuisance model override(s) in effect — see reproducibility "
                f"statement: {self.nuisance_model_overrides}",
            ]
        return "\n".join(lines)

    def to_checklist_row(self) -> dict:
        """
        Returns a compact dict for the report module's assumption audit section.
        The 'overrides' key is True if any nuisance model was overridden.
        """
        return {
            "population": self.population,
            "identification": self.identification,
            "extrapolation_flag": self.extrapolation_flag,
            "overrides": bool(self.nuisance_model_overrides),
        }


# ── Factory helpers ────────────────────────────────────────────────────────────

def make_complier_estimand(
    treatment: Any,
    outcome: Any,
    nuisance_model_overrides: Optional[dict] = None,
    **kwargs,
) -> Estimand:
    """
    Returns the canonical estimand for complier-restricted CATE estimates.
    This is the primary estimand produced by GreenCreditEvaluator.fit().
    """
    return Estimand(
        population="compliers",
        population_description=(
            "SMEs who accessed green credit because their lender offered it, "
            "and who would not have accessed it otherwise."
        ),
        identification=(
            "IV-DiD two-stage pipeline with staggered lender adoption as instrument. "
            "Stage 1: Callaway & Sant'Anna (2021) staggered DiD → LATE + kappa weights. "
            "Stage 2: ForestDRIV causal forest restricted to compliers via kappa-weighted "
            "sample weights."
        ),
        treatment=treatment,
        outcome=outcome,
        nuisance_model_overrides=nuisance_model_overrides or {},
        **kwargs,
    )


def make_bounds_estimand(
    population: str,
    treatment: Any,
    outcome: Any,
    **kwargs,
) -> Estimand:
    """
    Returns an estimand for Manski partial identification bounds.

    Only valid for 'always_takers' and 'never_takers'. These are unidentified
    subpopulations — bounds, not point estimates.

    Raises
    ------
    ValueError
        If population is 'compliers' or 'full_sample'. Use make_complier_estimand
        for compliers; full_sample bounds are computed as the union.
    """
    _valid = {"always_takers", "never_takers"}
    if population not in _valid:
        raise ValueError(
            f"make_bounds_estimand only constructs estimands for {_valid}. "
            f"Got: '{population}'. "
            f"Use make_complier_estimand() for compliers, or Estimand() directly "
            f"for full_sample."
        )

    descriptions = {
        "always_takers": (
            "SMEs who would access green credit regardless of lender adoption. "
            "The instrument provides no identifying variation for this subpopulation."
        ),
        "never_takers": (
            "SMEs who would not access green credit regardless of lender adoption. "
            "The instrument provides no identifying variation for this subpopulation."
        ),
    }

    return Estimand(
        population=population,
        population_description=descriptions[population],
        identification="Manski (1990) worst-case partial identification bounds.",
        treatment=treatment,
        outcome=outcome,
        **kwargs,
    )
