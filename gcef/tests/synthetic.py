"""
tests/synthetic
---------------
GCEFDataGenerator — class-based synthetic panel data generator for the
GCEF test suite.

This module provides a stable, class-based interface to the synthetic data
generation logic in gcef.testing.synthetic. The GCEFDataGenerator class
is the canonical interface used by tests that need to exercise specific
boundary conditions.

Defaults (n_smes=300, n_lenders=8, n_periods=8) are chosen to be large
enough for meaningful statistical tests while remaining fast enough for
a full test suite run.

Usage
-----
    from tests.synthetic import GCEFDataGenerator

    gen = GCEFDataGenerator(seed=42)
    data = gen.make_clean_panel()
    summary = gen.summary(data)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gcef.testing.synthetic import (
    make_panel,
    SyntheticConfig,
)


class GCEFDataGenerator:
    """
    Class-based interface to the GCEF synthetic data generator.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_smes : int
        Number of SMEs. Default: 300.
    n_lenders : int
        Number of lenders. Default: 8.
    n_periods : int
        Number of time periods. Default: 8.
    """

    def __init__(
        self,
        seed: int = 42,
        n_smes: int = 300,
        n_lenders: int = 8,
        n_periods: int = 8,
    ):
        self.seed = seed
        self.n_smes = n_smes
        self.n_lenders = n_lenders
        self.n_periods = n_periods

    # ── Factory methods ────────────────────────────────────────────────────────

    def make_clean_panel(self) -> pd.DataFrame:
        """
        A valid panel satisfying all spec identification assumptions.
        - Multiple lenders with staggered adoption timing
        - No adoption timing anomalies
        - Full geocoding (lat/lon)
        - Standard verification method (not self-reported)
        - Sufficient pre-treatment periods per cohort
        """
        return make_panel(SyntheticConfig(
            n_smes=self.n_smes,
            n_lenders=self.n_lenders,
            n_periods=self.n_periods,
            first_stage_strength=2.5,
            complier_share_target=0.35,
            n_pre_treatment_min=3,
            missing_geocoding=False,
            self_reported_verification=False,
            introduce_adoption_anomaly=False,
            seed=self.seed,
        ))

    def make_single_lender_panel(self) -> pd.DataFrame:
        """
        All SMEs assigned to one lender. Triggers SingleLenderError.
        """
        return make_panel(SyntheticConfig(
            n_smes=self.n_smes,
            n_lenders=1,
            n_periods=self.n_periods,
            adoption_periods=[max(2, self.n_periods // 3)],
            seed=self.seed,
        ))

    def make_missing_geocode_panel(self) -> pd.DataFrame:
        """
        Lat/lon columns absent. Triggers GeocodingRequiredError.
        Returns NaN in sme_latitude and sme_longitude columns
        (columns present but filled with NaN for schema consistency).
        """
        data = make_panel(SyntheticConfig(
            n_smes=self.n_smes,
            n_lenders=self.n_lenders,
            n_periods=self.n_periods,
            missing_geocoding=True,
            seed=self.seed,
        ))
        # Add lat/lon columns filled with NaN so schema is consistent
        # (allows tests to check .isna() rather than KeyError)
        data["sme_latitude"] = np.nan
        data["sme_longitude"] = np.nan
        return data

    def make_adoption_timing_anomaly_panel(self) -> pd.DataFrame:
        """
        Some SMEs have take-up before their lender's adoption date.
        Triggers AdoptionTimingAnomalyWarning/Error.
        """
        return make_panel(SyntheticConfig(
            n_smes=self.n_smes,
            n_lenders=self.n_lenders,
            n_periods=self.n_periods,
            introduce_adoption_anomaly=True,
            seed=self.seed,
        ))

    def make_short_panel(self, min_pre_periods: int = 2) -> pd.DataFrame:
        """
        Short pre-treatment panel.
        min_pre_periods=2 → ShortPanelWarning
        min_pre_periods=1 → ShortPanelError
        """
        return make_panel(SyntheticConfig(
            n_smes=self.n_smes,
            n_lenders=self.n_lenders,
            n_periods=6,
            n_pre_treatment_min=min_pre_periods,
            seed=self.seed,
        ))

    def make_weak_instrument_panel(self) -> pd.DataFrame:
        """Low first-stage strength. Triggers WeakInstrumentError (F < 10)."""
        return make_panel(SyntheticConfig(
            n_smes=self.n_smes,
            n_lenders=self.n_lenders,
            n_periods=self.n_periods,
            first_stage_strength=0.3,
            seed=self.seed,
        ))

    def make_low_complier_share_panel(self) -> pd.DataFrame:
        """Complier share ~12%. Triggers SmallComplierShareWarning."""
        return make_panel(SyntheticConfig(
            n_smes=self.n_smes,
            n_lenders=self.n_lenders,
            n_periods=self.n_periods,
            complier_share_target=0.12,
            seed=self.seed,
        ))

    def make_self_reported_panel(self) -> pd.DataFrame:
        """Self-reported verification. Triggers SelfReportedTreatmentWarning."""
        return make_panel(SyntheticConfig(
            n_smes=self.n_smes,
            n_lenders=self.n_lenders,
            n_periods=self.n_periods,
            self_reported_verification=True,
            seed=self.seed,
        ))

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def summary(self, data: pd.DataFrame) -> dict:
        """
        Returns basic diagnostics for a generated panel.

        Parameters
        ----------
        data : pd.DataFrame
            Panel produced by any make_* method.

        Returns
        -------
        dict with keys: n_smes, n_lenders, n_periods, plus additional
        diagnostics when ground-truth columns are present.
        """
        result = {
            "n_smes": data["sme_id"].nunique(),
            "n_lenders": data["lender_id"].nunique(),
            "n_periods": data["period"].nunique(),
            "has_geocoding": "sme_latitude" in data.columns
                and data["sme_latitude"].notna().any(),
            "has_prior_revenue": "prior_revenue" in data.columns,
        }

        if "instrument_Z" in data.columns:
            mean_D_Z1 = data.loc[data["instrument_Z"] == 1, "green_credit_takeup"].mean()
            mean_D_Z0 = data.loc[data["instrument_Z"] == 0, "green_credit_takeup"].mean()
            result["first_stage_diff"] = round(float(mean_D_Z1 - mean_D_Z0), 3)

        if "complier_type" in data.columns:
            sme_level = data.drop_duplicates("sme_id")
            type_counts = sme_level["complier_type"].value_counts(normalize=True)
            result["complier_share"] = round(float(type_counts.get("complier", 0)), 3)

        if "true_ate" in data.columns:
            result["true_ate"] = round(float(data["true_ate"].iloc[0]), 4)

        return result
