"""Tests for assumption checks."""
import pytest
import warnings
import pandas as pd
import numpy as np
from gcef import assumptions as checks
from gcef.exceptions import (
    SingleLenderError, WeakInstrumentError, WeakInstrumentWarning,
    InsufficientPreTreatmentError, ShortPanelWarning,
    KappaWeightDegeneracyError, KappaWeightWarning,
)


def test_single_lender_raises():
    data = pd.DataFrame({"lender_id": ["A", "A", "A"]})
    with pytest.raises(SingleLenderError):
        checks.check_single_lender(data, "lender_id")


def test_multiple_lenders_passes():
    data = pd.DataFrame({"lender_id": ["A", "B", "C"]})
    result = checks.check_single_lender(data, "lender_id")
    assert result["passed"] is True


def test_weak_instrument_below_10_raises():
    with pytest.raises(WeakInstrumentError):
        checks.check_instrument_relevance(9.5)


def test_weak_instrument_between_10_20_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = checks.check_instrument_relevance(15.0)
        assert any(issubclass(x.category, WeakInstrumentWarning) for x in w)
    assert result["value"] == 15.0


def test_strong_instrument_passes():
    result = checks.check_instrument_relevance(25.0)
    assert result["passed"] is True


def test_kappa_above_45pct_raises_staggered_did():
    """staggered_did error threshold is 45%."""
    weights = np.array([-1] * 50 + [1] * 50, dtype=float)  # 50% nonpositive
    with pytest.raises(KappaWeightDegeneracyError):
        checks.check_kappa_weights(weights, design="staggered_did")


def test_kappa_above_15pct_raises_cross_sectional():
    """cross_sectional error threshold is 15%."""
    weights = np.array([-1] * 20 + [1] * 80, dtype=float)  # 20% nonpositive
    with pytest.raises(KappaWeightDegeneracyError):
        checks.check_kappa_weights(weights, design="cross_sectional")


def test_kappa_between_30_45pct_warns_staggered_did():
    """staggered_did warning threshold is 30%."""
    weights = np.array([-1] * 35 + [1] * 65, dtype=float)  # 35% nonpositive
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        checks.check_kappa_weights(weights, design="staggered_did")
        assert any(issubclass(x.category, KappaWeightWarning) for x in w)


def test_kappa_between_5_15pct_warns_cross_sectional():
    """cross_sectional warning threshold is 5%."""
    weights = np.array([-1] * 10 + [1] * 90, dtype=float)  # 10% nonpositive
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        checks.check_kappa_weights(weights, design="cross_sectional")
        assert any(issubclass(x.category, KappaWeightWarning) for x in w)


def test_kappa_below_30pct_passes_staggered_did():
    """25% nonpositive is typical and passes staggered_did threshold."""
    weights = np.array([-1] * 25 + [1] * 75, dtype=float)
    result = checks.check_kappa_weights(weights, design="staggered_did")
    assert result["passed"] is True


def test_kappa_below_5pct_passes_cross_sectional():
    weights = np.array([-1] * 3 + [1] * 97, dtype=float)
    result = checks.check_kappa_weights(weights, design="cross_sectional")
    assert result["passed"] is True


def test_kappa_design_recorded_in_result():
    weights = np.ones(100)
    result = checks.check_kappa_weights(weights, design="staggered_did")
    assert result["design"] == "staggered_did"
    assert result["warning_threshold"] == 0.30
    assert result["error_threshold"] == 0.45


def test_kappa_invalid_design_raises():
    with pytest.raises(ValueError, match="design must be"):
        checks.check_kappa_weights(np.ones(10), design="invalid_design")
