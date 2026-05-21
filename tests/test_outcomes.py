"""Tests for ResilienceIndex and rolling CV derivation."""
import pytest
import warnings
import pandas as pd
import numpy as np
from gcef.outcomes import ResilienceIndex
from gcef.exceptions import ResilienceIndexWarning, UserDerivedStabilityWarning


def make_panel(n_firms=5, n_periods=8, seed=42):
    rng = np.random.default_rng(seed)
    firms = [f"firm_{i}" for i in range(n_firms)]
    periods = list(range(n_periods))
    rows = []
    for firm in firms:
        base = rng.uniform(100, 1000)
        for p in periods:
            rows.append({
                "sme_id": firm,
                "period": p,
                "revenue": base + rng.normal(0, base * 0.1),
                "loan_repayment_rate": rng.uniform(0.7, 1.0),
                "employment": int(rng.uniform(5, 50)),
                "adaptation_investment": rng.uniform(0, 100),
                "rainfall_anomaly_lag1": rng.normal(0, 1),
            })
    return pd.DataFrame(rows)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        ResilienceIndex(columns={"revenue": 0.5, "loan_repayment_rate": 0.3})


def test_stability_derived_from_revenue():
    outcome = ResilienceIndex(stability_window=4)
    data = make_panel()
    data = outcome.derive_stability_columns(data, unit_id="sme_id")
    assert "revenue_stability" in data.columns
    assert "employment_stability" in data.columns


def test_user_supplied_stability_warns():
    outcome = ResilienceIndex(stability_window=4)
    data = make_panel()
    data["revenue_stability"] = 0.9  # pre-computed by analyst
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        data = outcome.derive_stability_columns(data, unit_id="sme_id")
        assert any(issubclass(x.category, UserDerivedStabilityWarning) for x in w)


def test_underidentification_warns():
    outcome = ResilienceIndex(
        columns={"revenue": 1.0},
        underidentification_threshold=2,
    )
    data = make_panel()
    data = outcome.derive_stability_columns(data, unit_id="sme_id")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        outcome.build(data, unit_id="sme_id")
        assert any(issubclass(x.category, ResilienceIndexWarning) for x in w)


def test_index_builds_with_sufficient_columns():
    outcome = ResilienceIndex()
    data = make_panel()
    data = outcome.derive_stability_columns(data, unit_id="sme_id")
    index = outcome.build(data, unit_id="sme_id")
    assert len(index) == len(data)
    assert index.notna().any()
