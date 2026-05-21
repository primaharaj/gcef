"""
gcef.outcomes
-------------
ResilienceIndex: constructs a composite outcome index from available
columns with explicit, documented weights.

Revenue stability and employment stability are derived internally from
their respective level columns using a rolling coefficient of variation.
Analysts may supply pre-computed columns; see UserDerivedStabilityWarning.

Design decisions DD1, and rolling CV derivation from spec Section 4.2.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
from gcef.exceptions import ResilienceIndexWarning, UserDerivedStabilityWarning, InsufficientRevenueHistoryWarning

DEFAULT_WEIGHTS = {
    "revenue": 0.40,            # derived to revenue_stability internally
    "loan_repayment_rate": 0.30,
    "employment": 0.20,         # derived to employment_stability internally
    "adaptation_investment": 0.10,
}

# Columns that are derived internally rather than used directly
DERIVED_COLUMNS = {
    "revenue": "revenue_stability",
    "employment": "employment_stability",
}


@dataclass
class ResilienceIndex:
    """
    Composite SME resilience outcome index.

    Constructs the index from whatever outcome columns are available in
    the dataset. Revenue stability and employment stability are derived
    internally from level columns via rolling coefficient of variation.

    Parameters
    ----------
    columns : dict[str, float]
        Mapping of column names to weights. Weights must sum to 1.0.
        Use 'revenue' (not 'revenue_stability') — stability is derived
        internally. Likewise use 'employment', not 'employment_stability'.
    shock_instrument : str
        Column name of the lagged climate shock indicator (e.g.
        'rainfall_anomaly_lag1'). Used by Stage 2 to subset observations
        to shock periods.
    shock_threshold : float
        Values below this threshold define a shock event.
        Default: -1.5 standard deviations.
    stability_window : int
        Rolling window length (in periods) for CV computation.
        Minimum periods of revenue/employment data required per firm.
        Default: 4.
    underidentification_threshold : int
        Raise ResilienceIndexWarning if fewer than this many components
        are available. Default: 2.

    Notes
    -----
    Shock conditioning in Stage 2:
    Revenue stability is computed across ALL periods using the full
    revenue history (rolling CV over stability_window). The causal forest
    then subsets to shock periods by filtering on shock_instrument. The
    stability scores are pre-computed; only the observation subsetting
    is shock-conditional. Do not compute CV within shock windows only —
    this would produce noisier estimates in thin shock-event data.

    Examples
    --------
    >>> outcome = ResilienceIndex(
    ...     columns={"revenue": 0.40, "loan_repayment_rate": 0.30,
    ...              "employment": 0.20, "adaptation_investment": 0.10},
    ...     shock_instrument="rainfall_anomaly_lag1",
    ...     shock_threshold=-1.5,
    ...     stability_window=4,
    ... )
    """

    columns: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    shock_instrument: str = "rainfall_anomaly_lag1"
    shock_threshold: float = -1.5
    stability_window: int = 4
    underidentification_threshold: int = 2

    def __post_init__(self):
        total = sum(self.columns.values())
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(
                f"ResilienceIndex weights must sum to 1.0. Got {total:.4f}. "
                f"Normalise your weights before passing them in."
            )

    def derive_stability_columns(
        self,
        data: pd.DataFrame,
        unit_id: str,
    ) -> pd.DataFrame:
        """
        Derives revenue_stability and employment_stability columns from
        their level counterparts using rolling 1 - CV, grouped by unit_id.

        Parameters
        ----------
        data : pd.DataFrame
            Panel dataset sorted by (unit_id, period).
        unit_id : str
            Column name for the SME identifier.

        Returns
        -------
        pd.DataFrame
            Input dataframe with derived stability columns added.
            Firms with insufficient history are flagged and their
            stability values set to NaN.
        """
        data = data.copy()

        for level_col, stability_col in DERIVED_COLUMNS.items():
            if level_col not in data.columns:
                continue

            # Analyst has supplied pre-computed stability — warn and skip
            if stability_col in data.columns:
                warnings.warn(
                    f"Both '{level_col}' and '{stability_col}' found in data. "
                    f"Pre-computed '{stability_col}' takes precedence. "
                    f"Document your derivation method in the reproducibility statement.",
                    UserDerivedStabilityWarning,
                    stacklevel=2,
                )
                continue

            def _rolling_stability(group: pd.Series) -> pd.Series:
                rolling = group.rolling(
                    window=self.stability_window,
                    min_periods=self.stability_window,
                )
                cv = rolling.std() / rolling.mean()
                return 1 - cv

            data[stability_col] = (
                data.groupby(unit_id)[level_col]
                .transform(_rolling_stability)
            )

            # Flag firms with insufficient history
            insufficient = data[stability_col].isna() & data[level_col].notna()
            n_insufficient = data.loc[insufficient, unit_id].nunique()
            if n_insufficient > 0:
                warnings.warn(
                    f"{n_insufficient} firm(s) have fewer than {self.stability_window} "
                    f"periods of '{level_col}' data and cannot have '{stability_col}' "
                    f"computed. These firms are excluded from the resilience index "
                    f"for periods where the rolling window is incomplete.",
                    InsufficientRevenueHistoryWarning,
                    stacklevel=2,
                )

        return data

    def build(
        self,
        data: pd.DataFrame,
        unit_id: str,
    ) -> pd.Series:
        """
        Constructs the composite resilience index for each observation.

        Parameters
        ----------
        data : pd.DataFrame
            Panel dataset with stability columns already derived
            (call derive_stability_columns first).
        unit_id : str
            Column name for the SME identifier.

        Returns
        -------
        pd.Series
            Composite resilience index, indexed like data.

        Raises
        ------
        ResilienceIndexWarning
            If fewer than underidentification_threshold components
            are available. Analyst must acknowledge to proceed.
        """
        # Map input column names to derived/direct column names
        available = {}
        for col, weight in self.columns.items():
            derived = DERIVED_COLUMNS.get(col, col)  # revenue → revenue_stability
            if derived in data.columns:
                available[derived] = weight
            elif col in data.columns:
                available[col] = weight

        if len(available) < self.underidentification_threshold:
            warnings.warn(
                f"ResilienceIndex is underidentified: only {len(available)} of "
                f"{len(self.columns)} components are available "
                f"(threshold: {self.underidentification_threshold}). "
                f"Available: {list(available.keys())}. "
                f"Index estimates are unreliable. Acknowledge this warning to proceed.",
                ResilienceIndexWarning,
                stacklevel=2,
            )

        # Normalise weights to available components
        total_weight = sum(available.values())
        index = pd.Series(0.0, index=data.index)
        for col, weight in available.items():
            normalised_weight = weight / total_weight
            index += data[col].fillna(0) * normalised_weight

        return index

    def to_description(self) -> str:
        """Human-readable description for Estimand and report module."""
        parts = [f"Composite resilience index ({len(self.columns)} components):"]
        for col, w in self.columns.items():
            derived = DERIVED_COLUMNS.get(col, col)
            note = f" [derived from {col} via rolling 1-CV, window={self.stability_window}]" \
                   if col in DERIVED_COLUMNS else ""
            parts.append(f"  {derived}: weight={w:.2f}{note}")
        parts.append(f"Shock instrument: {self.shock_instrument} < {self.shock_threshold}")
        return "\n".join(parts)
