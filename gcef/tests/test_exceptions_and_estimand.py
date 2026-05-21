"""
tests/test_exceptions_and_estimand.py
--------------------------------------
Test suite for gcef.exceptions and gcef.estimand.
"""
import json
import pytest
import warnings

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gcef.exceptions import (
    GCEFWarning, ResilienceIndexWarning, UserDerivedStabilityWarning,
    ShortPanelWarning, WeakInstrumentWarning, SmallComplierShareWarning,
    KappaWeightWarning, GeocodeResolutionWarning, SelfReportedTreatmentWarning,
    GCEFError, WeakInstrumentError, KappaWeightError, SingleLenderError,
    ShortPanelError, InsufficientRevenueHistoryError, AdoptionTimingAnomalyError,
    BlendedTreatmentNotImplemented,
)
from gcef.estimand import (
    Estimand, VALID_POPULATIONS, DFI_FRAMING_DEFAULTS,
    make_complier_estimand, make_bounds_estimand,
)


@pytest.fixture
def complier_estimand():
    return Estimand(
        population="compliers",
        population_description=(
            "SMEs who accessed green credit because their lender offered it, "
            "and who would not have accessed it otherwise"
        ),
        identification="IV-DiD with staggered lender adoption as instrument",
    )


@pytest.fixture
def always_taker_estimand():
    return Estimand(
        population="always_takers",
        population_description=(
            "SMEs who would access green credit regardless of lender adoption"
        ),
        identification="Manski (1990) partial identification bounds",
    )


class TestExceptionHierarchy:

    def test_gcef_warning_is_user_warning(self):
        assert issubclass(GCEFWarning, UserWarning)

    def test_gcef_error_is_exception(self):
        assert issubclass(GCEFError, Exception)

    @pytest.mark.parametrize("warning_cls", [
        ResilienceIndexWarning, UserDerivedStabilityWarning, ShortPanelWarning,
        WeakInstrumentWarning, SmallComplierShareWarning, KappaWeightWarning,
        GeocodeResolutionWarning, SelfReportedTreatmentWarning,
    ])
    def test_all_warnings_inherit_gcef_warning(self, warning_cls):
        assert issubclass(warning_cls, GCEFWarning)

    @pytest.mark.parametrize("error_cls", [
        WeakInstrumentError, KappaWeightError, SingleLenderError, ShortPanelError,
        InsufficientRevenueHistoryError, AdoptionTimingAnomalyError,
    ])
    def test_all_errors_inherit_gcef_error(self, error_cls):
        assert issubclass(error_cls, GCEFError)

    def test_blended_treatment_is_not_implemented_error(self):
        assert issubclass(BlendedTreatmentNotImplemented, NotImplementedError)


class TestStructuredWarnings:

    def test_short_panel_warning_carries_cohorts_affected(self):
        w = ShortPanelWarning("msg", cohorts_affected=["2021_cohort"])
        assert w.cohorts_affected == ["2021_cohort"]

    def test_weak_instrument_warning_carries_f_statistic(self):
        w = WeakInstrumentWarning("msg", f_statistic=14.2)
        assert w.f_statistic == 14.2

    def test_small_complier_share_warning_carries_share(self):
        w = SmallComplierShareWarning("msg", complier_share=0.15)
        assert w.complier_share == 0.15

    def test_kappa_weight_warning_carries_share(self):
        w = KappaWeightWarning("msg", share_nonpositive=0.07)
        assert w.share_nonpositive == 0.07

    def test_warnings_can_be_caught_as_gcef_warning(self):
        with pytest.warns(GCEFWarning):
            warnings.warn("test", ResilienceIndexWarning)


class TestStructuredErrors:

    def test_weak_instrument_error_carries_f_statistic(self):
        err = WeakInstrumentError("F too low", f_statistic=7.3)
        assert err.f_statistic == 7.3
        assert "F too low" in str(err)

    def test_kappa_weight_error_carries_share(self):
        err = KappaWeightError("too many negative weights", share_nonpositive=0.20)
        assert err.share_nonpositive == 0.20

    def test_short_panel_error_carries_cohorts_affected(self):
        err = ShortPanelError("short panel", cohorts_affected=["2019_cohort"])
        assert err.cohorts_affected == ["2019_cohort"]

    def test_adoption_timing_anomaly_carries_count(self):
        err = AdoptionTimingAnomalyError("anomaly", anomaly_count=12)
        assert err.anomaly_count == 12

    def test_insufficient_revenue_history_carries_ids(self):
        err = InsufficientRevenueHistoryError("short history", affected_sme_ids=["SME_001"])
        assert "SME_001" in err.affected_sme_ids

    def test_blended_treatment_message_matches_spec(self):
        err = BlendedTreatmentNotImplemented()
        assert "v0.1" in str(err)
        assert "multi-pathway DAG" in str(err)
        assert "GitHub repository" in str(err)

    def test_errors_can_be_caught_as_gcef_error(self):
        with pytest.raises(GCEFError):
            raise WeakInstrumentError("test", f_statistic=5.0)

    def test_blended_can_be_caught_as_not_implemented_error(self):
        with pytest.raises(NotImplementedError):
            raise BlendedTreatmentNotImplemented()


class TestEstimandConstruction:

    def test_valid_complier_estimand(self, complier_estimand):
        assert complier_estimand.population == "compliers"
        assert complier_estimand.extrapolation_flag is False

    def test_always_taker_sets_extrapolation_flag(self, always_taker_estimand):
        assert always_taker_estimand.extrapolation_flag is True

    def test_never_taker_sets_extrapolation_flag(self):
        e = Estimand(
            population="never_takers",
            population_description="desc",
            identification="Manski bounds",
        )
        assert e.extrapolation_flag is True

    def test_invalid_population_raises_value_error(self):
        with pytest.raises(ValueError, match="population must be one of"):
            Estimand(
                population="non_compliers",
                population_description="desc",
                identification="IV-DiD",
            )

    def test_default_dfi_framing_populated_from_lookup(self, complier_estimand):
        assert complier_estimand.dfi_framing == DFI_FRAMING_DEFAULTS["compliers"]
        assert len(complier_estimand.dfi_framing) > 0

    def test_custom_dfi_framing_not_overwritten(self):
        custom = "Custom DFI framing for this analysis."
        e = Estimand(
            population="compliers",
            population_description="desc",
            identification="IV-DiD",
            dfi_framing=custom,
        )
        assert e.dfi_framing == custom

    def test_all_valid_populations_accepted(self):
        for pop in VALID_POPULATIONS:
            e = Estimand(
                population=pop,
                population_description="desc",
                identification="IV-DiD",
            )
            assert e.population == pop

    def test_full_sample_population_without_extrapolation_flag(self):
        e = Estimand(
            population="full_sample",
            population_description="desc",
            identification="IV-DiD",
        )
        assert e.extrapolation_flag is False

    def test_nuisance_model_overrides_defaults_to_empty(self, complier_estimand):
        assert complier_estimand.nuisance_model_overrides == {}

    def test_gcef_version_set(self, complier_estimand):
        assert complier_estimand.gcef_version == "0.1.0"


class TestEstimandSerialisation:

    def test_to_dict_returns_dict(self, complier_estimand):
        d = complier_estimand.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_contains_required_keys(self, complier_estimand):
        d = complier_estimand.to_dict()
        required = {
            "population", "population_description", "dfi_framing",
            "identification", "nuisance_model_overrides",
            "extrapolation_flag", "gcef_version",
        }
        assert required.issubset(d.keys())

    def test_to_json_is_valid_json(self, complier_estimand):
        json_str = complier_estimand.to_json()
        parsed = json.loads(json_str)
        assert parsed["population"] == "compliers"

    def test_to_json_round_trips(self, complier_estimand):
        d = json.loads(complier_estimand.to_json())
        assert d["extrapolation_flag"] is False
        assert d["identification"] == "IV-DiD with staggered lender adoption as instrument"

    def test_to_dict_with_no_treatment_outcome(self, complier_estimand):
        d = complier_estimand.to_dict()
        assert d["treatment"] is None
        assert d["outcome"] is None

    def test_to_dict_with_mock_treatment(self):
        class MockTreatment:
            def to_dict(self):
                return {"type": "rate_reduction", "intensity": 0.03}

        e = Estimand(
            population="compliers",
            population_description="desc",
            identification="IV-DiD",
            treatment=MockTreatment(),
        )
        d = e.to_dict()
        assert d["treatment"]["type"] == "rate_reduction"

    def test_to_dict_with_mock_treatment_no_to_dict(self):
        class MockTreatment:
            def __repr__(self):
                return "MockTreatment()"

        e = Estimand(
            population="compliers",
            population_description="desc",
            identification="IV-DiD",
            treatment=MockTreatment(),
        )
        d = e.to_dict()
        assert d["treatment"] == "MockTreatment()"


class TestEstimandRendering:

    def test_to_prose_contains_population(self, complier_estimand):
        prose = complier_estimand.to_prose()
        assert "complier" in prose.lower()

    def test_to_prose_contains_identification(self, complier_estimand):
        prose = complier_estimand.to_prose()
        assert "IV-DiD" in prose

    def test_to_prose_contains_dfi_framing(self, complier_estimand):
        prose = complier_estimand.to_prose()
        assert "additionality" in prose.lower() or "marginal" in prose.lower()

    def test_to_prose_flags_extrapolation(self, always_taker_estimand):
        prose = always_taker_estimand.to_prose()
        assert "bounds" in prose.lower()
        assert "not identified" in prose.lower()

    def test_to_prose_flags_nuisance_overrides(self):
        e = Estimand(
            population="compliers",
            population_description="desc",
            identification="IV-DiD",
            nuisance_model_overrides={"model_t": "RandomForestClassifier"},
        )
        prose = e.to_prose()
        assert "override" in prose.lower() or "reproducibility" in prose.lower()

    def test_to_checklist_row_structure(self, complier_estimand):
        row = complier_estimand.to_checklist_row()
        assert "population" in row
        assert "identification" in row
        assert "extrapolation_flag" in row
        assert "overrides" in row
        assert row["extrapolation_flag"] is False
        assert row["overrides"] is False


class TestEstimandEquality:

    def test_identical_estimands_are_equal(self, complier_estimand):
        e2 = Estimand(
            population="compliers",
            population_description="different description",
            identification="IV-DiD with staggered lender adoption as instrument",
        )
        assert complier_estimand == e2

    def test_different_populations_not_equal(self, complier_estimand, always_taker_estimand):
        assert complier_estimand != always_taker_estimand

    def test_estimands_are_hashable(self, complier_estimand, always_taker_estimand):
        s = {complier_estimand, always_taker_estimand}
        assert len(s) == 2

    def test_estimand_not_equal_to_non_estimand(self, complier_estimand):
        assert complier_estimand != "compliers"
        assert complier_estimand != 42


class TestEstimandFactories:

    def test_make_complier_estimand_returns_correct_population(self):
        e = make_complier_estimand(treatment=None, outcome=None)
        assert e.population == "compliers"
        assert e.extrapolation_flag is False

    def test_make_complier_estimand_uses_iv_did_identification(self):
        e = make_complier_estimand(treatment=None, outcome=None)
        assert "IV-DiD" in e.identification
        assert "staggered" in e.identification

    def test_make_complier_estimand_stores_overrides(self):
        overrides = {"model_t": "LogisticRegression"}
        e = make_complier_estimand(
            treatment=None, outcome=None,
            nuisance_model_overrides=overrides,
        )
        assert e.nuisance_model_overrides == overrides

    def test_make_bounds_estimand_always_takers(self):
        e = make_bounds_estimand("always_takers", treatment=None, outcome=None)
        assert e.population == "always_takers"
        assert e.extrapolation_flag is True
        assert "Manski" in e.identification

    def test_make_bounds_estimand_never_takers(self):
        e = make_bounds_estimand("never_takers", treatment=None, outcome=None)
        assert e.population == "never_takers"
        assert e.extrapolation_flag is True

    def test_make_bounds_estimand_rejects_invalid_population(self):
        with pytest.raises(ValueError, match="only constructs estimands for"):
            make_bounds_estimand("compliers", treatment=None, outcome=None)

    def test_make_bounds_estimand_rejects_full_sample(self):
        with pytest.raises(ValueError):
            make_bounds_estimand("full_sample", treatment=None, outcome=None)


class TestSyntheticDataGenerator:

    def test_clean_panel_has_required_columns(self):
        from tests.synthetic import GCEFDataGenerator
        gen = GCEFDataGenerator(seed=42)
        data = gen.make_clean_panel()
        required = [
            "sme_id", "lender_id", "period",
            "green_credit_takeup", "lender_green_adoption_period",
            "revenue", "loan_repayment_rate", "employment",
            "sme_latitude", "sme_longitude",
            "firm_age", "sector", "firm_size", "prior_revenue",
        ]
        for col in required:
            assert col in data.columns, f"Missing required column: {col}"

    def test_clean_panel_has_multiple_lenders(self):
        from tests.synthetic import GCEFDataGenerator
        gen = GCEFDataGenerator(seed=42)
        data = gen.make_clean_panel()
        assert data["lender_id"].nunique() >= 3

    def test_clean_panel_has_staggered_adoption(self):
        from tests.synthetic import GCEFDataGenerator
        gen = GCEFDataGenerator(seed=42)
        data = gen.make_clean_panel()
        adoption_periods = (
            data.groupby("lender_id")["lender_green_adoption_period"]
            .first().unique()
        )
        assert len(adoption_periods) > 1

    def test_clean_panel_has_no_adoption_timing_anomalies(self):
        from tests.synthetic import GCEFDataGenerator
        gen = GCEFDataGenerator(seed=42)
        data = gen.make_clean_panel()
        anomalies = data[
            (data["green_credit_takeup"] == 1)
            & (data["period"] < data["lender_green_adoption_period"])
        ]
        assert len(anomalies) == 0

    def test_single_lender_panel_has_one_lender(self):
        from tests.synthetic import GCEFDataGenerator
        gen = GCEFDataGenerator(seed=42)
        data = gen.make_single_lender_panel()
        assert data["lender_id"].nunique() == 1

    def test_missing_geocode_panel_has_nan_coordinates(self):
        from tests.synthetic import GCEFDataGenerator
        gen = GCEFDataGenerator(seed=42)
        data = gen.make_missing_geocode_panel()
        assert data["sme_latitude"].isna().all()
        assert data["sme_longitude"].isna().all()

    def test_adoption_timing_anomaly_panel_has_anomalies(self):
        from tests.synthetic import GCEFDataGenerator
        gen = GCEFDataGenerator(seed=42)
        data = gen.make_adoption_timing_anomaly_panel()
        anomalies = data[
            (data["green_credit_takeup"] == 1)
            & (data["period"] < data["lender_green_adoption_period"])
        ]
        assert len(anomalies) > 0

    def test_generator_is_reproducible(self):
        from tests.synthetic import GCEFDataGenerator
        gen1 = GCEFDataGenerator(seed=42)
        gen2 = GCEFDataGenerator(seed=42)
        data1 = gen1.make_clean_panel()
        data2 = gen2.make_clean_panel()
        assert data1["revenue"].equals(data2["revenue"])

    def test_different_seeds_produce_different_data(self):
        from tests.synthetic import GCEFDataGenerator
        gen1 = GCEFDataGenerator(seed=42)
        gen2 = GCEFDataGenerator(seed=99)
        data1 = gen1.make_clean_panel()
        data2 = gen2.make_clean_panel()
        assert not data1["revenue"].equals(data2["revenue"])

    def test_summary_returns_expected_shape(self):
        from tests.synthetic import GCEFDataGenerator
        gen = GCEFDataGenerator(seed=42)
        data = gen.make_clean_panel()
        summary = gen.summary(data)
        assert summary["n_smes"] == 300
        assert summary["n_lenders"] == 8
        assert summary["n_periods"] == 8
