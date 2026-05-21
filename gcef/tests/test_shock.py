"""
tests/test_shock.py
-------------------
Tests for gcef.shock.

All tests run without network access. CHIRPS downloads are mocked via the
_downloader injection point. Raster operations use in-memory synthetic tiles.

Tests cover:
- SPI formula correctness on known precipitation sequences
- Spatial raster sampling
- Lag construction
- Geocoding fallback (admin unit centroid)
- Missing location data handling
- Synthetic shock instrument
- period_to_yearmonth validation
- NaN handling throughout
"""
import io
import gzip
import pytest
import warnings
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

from gcef.shock import (
    attach_chirps_anomaly,
    make_synthetic_shock_instrument,
    _compute_spi_from_precip,
    _sample_raster_at_points,
    _get_unique_locations,
    _resolve_locations,
    _join_and_lag_spi,
    _get_yearmonths_needed,
    ShockInstrumentWarning,
    DEFAULT_SPI_WINDOW,
)
from gcef.exceptions import GeocodeResolutionWarning
from gcef.testing.synthetic import make_valid_panel


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def panel():
    return make_valid_panel(n_smes=50, n_lenders=3, n_periods=8, seed=42)


@pytest.fixture
def period_to_yearmonth():
    """Map 8 quarterly periods to calendar year-months."""
    return {
        1: (2019, 3),
        2: (2019, 6),
        3: (2019, 9),
        4: (2019, 12),
        5: (2020, 3),
        6: (2020, 6),
        7: (2020, 9),
        8: (2020, 12),
    }


@pytest.fixture
def synthetic_tile_factory(tmp_path):
    """
    Returns a factory that creates a synthetic CHIRPS-like GeoTIFF.
    Used to inject into attach_chirps_anomaly via _downloader.
    """
    def _make_tile(year, month, value=80.0, with_nodata=False):
        """Create a synthetic 0.05° resolution raster covering SSA."""
        west, south, east, north = 10.0, -35.0, 50.0, 15.0
        width = int((east - west) / 0.05)
        height = int((north - south) / 0.05)
        transform = from_bounds(west, south, east, north, width, height)

        rng = np.random.default_rng(year * 100 + month)
        data = rng.normal(value, value * 0.3, (1, height, width)).astype(np.float32)
        data = np.clip(data, 0, None)

        if with_nodata:
            data[0, :5, :5] = -9999.0

        tile_path = tmp_path / f"chirps-v2.0.{year}.{month:02d}.tif"
        with rasterio.open(
            tile_path, 'w', driver='GTiff',
            height=height, width=width, count=1,
            dtype='float32', crs='EPSG:4326', transform=transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(data)
        return tile_path

    return _make_tile


@pytest.fixture
def mock_downloader(synthetic_tile_factory):
    """Inject into attach_chirps_anomaly to avoid network calls."""
    def _download(year, month, cache_dir):
        return synthetic_tile_factory(year, month)
    return _download


# ── SPI formula correctness ────────────────────────────────────────────────────

class TestSPIFormula:

    def test_spi_length_matches_input(self):
        precip = np.array([80.0] * 24)
        spi = _compute_spi_from_precip(precip, window=3)
        assert len(spi) == 24

    def test_spi_first_valid_index(self):
        """
        SPI-3 with expanding variance requires at least 2 history points.
        First rolling sum is at index 2; second at index 3 (first with variance).
        First valid SPI is at index 3.
        """
        precip = np.array([80.0] * 12)
        spi = _compute_spi_from_precip(precip, window=3)
        # First 3 values are NaN (insufficient history for variance)
        assert np.isnan(spi[0])
        assert np.isnan(spi[1])
        assert np.isnan(spi[2])
        # Index 3 is the first valid value
        assert not np.isnan(spi[3])

    def test_constant_precip_gives_zero_spi(self):
        """Constant precipitation → zero SPI (no anomaly)."""
        precip = np.array([80.0] * 24)
        spi = _compute_spi_from_precip(precip, window=3)
        valid = spi[~np.isnan(spi)]
        assert np.allclose(valid, 0.0, atol=1e-6)

    def test_drought_gives_negative_spi(self):
        """Well below normal precipitation → negative SPI."""
        # 20 normal months then a very dry month
        precip = np.array([80.0] * 20 + [5.0, 5.0, 5.0])
        spi = _compute_spi_from_precip(precip, window=3)
        assert spi[-1] < -1.0

    def test_wet_gives_positive_spi(self):
        """Well above normal precipitation → positive SPI."""
        precip = np.array([80.0] * 20 + [200.0, 200.0, 200.0])
        spi = _compute_spi_from_precip(precip, window=3)
        assert spi[-1] > 1.0

    def test_nan_in_precip_propagates(self):
        """NaN precipitation → NaN SPI for that window."""
        precip = np.array([80.0, np.nan, 80.0, 80.0, 80.0])
        spi = _compute_spi_from_precip(precip, window=3)
        assert np.isnan(spi[2])  # window containing the NaN

    def test_spi_window_1_has_one_nan(self):
        """SPI-1: first value is NaN, rest are valid."""
        precip = np.array([80.0] * 6)
        spi = _compute_spi_from_precip(precip, window=1)
        assert np.isnan(spi[0])
        assert not np.isnan(spi[1])

    def test_spi_threshold_correctly_identifies_drought(self):
        """SPI < -1.5 correctly flags drought month."""
        precip = np.array([80.0] * 24 + [2.0, 2.0, 2.0])
        spi = _compute_spi_from_precip(precip, window=3)
        assert spi[-1] < -1.5, f"Expected drought SPI, got {spi[-1]:.2f}"


# ── Spatial sampling ───────────────────────────────────────────────────────────

class TestRasterSampling:

    def test_sample_returns_correct_length(self, synthetic_tile_factory):
        tile = synthetic_tile_factory(2020, 3)
        lons = np.array([20.0, 25.0, 30.0, 35.0])
        lats = np.array([-5.0, -10.0, 0.0, 5.0])
        result = _sample_raster_at_points(tile, lons, lats)
        assert len(result) == 4

    def test_sample_values_are_finite_in_bounds(self, synthetic_tile_factory):
        tile = synthetic_tile_factory(2020, 3)
        lons = np.array([20.0, 25.0])
        lats = np.array([-5.0, -10.0])
        result = _sample_raster_at_points(tile, lons, lats)
        assert np.isfinite(result).all()

    def test_sample_nodata_becomes_nan(self, synthetic_tile_factory):
        """Points coinciding with nodata (-9999) values become NaN."""
        tile = synthetic_tile_factory(2020, 3, with_nodata=True)
        # The nodata region is the top-left corner near (10, 15)
        lons = np.array([10.1])
        lats = np.array([14.9])
        result = _sample_raster_at_points(tile, lons, lats)
        assert np.isnan(result[0])

    def test_sample_consistent_with_rasterio_direct(self, synthetic_tile_factory):
        """Results match direct rasterio.sample call."""
        tile = synthetic_tile_factory(2020, 6)
        lons = np.array([30.0, 35.0])
        lats = np.array([-20.0, 0.0])
        result = _sample_raster_at_points(tile, lons, lats)
        with rasterio.open(tile) as src:
            expected = np.array(
                [v[0] for v in src.sample(zip(lons, lats))], dtype=float
            )
        assert np.allclose(result, expected, equal_nan=True)


# ── Unique location deduplication ─────────────────────────────────────────────

class TestUniqueLocations:

    def test_unique_locations_deduplicates(self):
        lats = np.array([1.0, 2.0, 1.0, 3.0])
        lons = np.array([10.0, 20.0, 10.0, 30.0])
        unique, idx = _get_unique_locations(lats, lons)
        assert len(unique) == 3  # (1,10), (2,20), (3,30)

    def test_loc_index_maps_back_correctly(self):
        lats = np.array([1.0, 2.0, 1.0])
        lons = np.array([10.0, 20.0, 10.0])
        unique, idx = _get_unique_locations(lats, lons)
        # Reconstruct original pairs
        reconstructed = unique[idx]
        np.testing.assert_array_equal(
            np.column_stack([lats, lons]),
            reconstructed,
        )

    def test_all_unique_points(self):
        lats = np.array([1.0, 2.0, 3.0])
        lons = np.array([10.0, 20.0, 30.0])
        unique, idx = _get_unique_locations(lats, lons)
        assert len(unique) == 3


# ── Lag construction ───────────────────────────────────────────────────────────

class TestLagConstruction:

    def test_lag_1_first_period_is_nan(self):
        """Period 1 with lag=1 has no lagged source → NaN."""
        n = 3
        periods = [1, 2, 3]
        spi_by_period = {
            1: np.array([1.0, 1.0, 1.0]),
            2: np.array([-2.0, -2.0, -2.0]),
            3: np.array([0.5, 0.5, 0.5]),
        }
        df = pd.DataFrame({
            "period": [1, 2, 3],
            "sme_id": ["a", "a", "a"],
        })
        result = _join_and_lag_spi(df, "period", spi_by_period, periods, lag=1,
                                   output_col="spi_lag1")
        assert np.isnan(result.loc[result["period"] == 1, "spi_lag1"].values[0])

    def test_lag_1_second_period_gets_first_period_value(self):
        """Period 2 with lag=1 receives period 1's SPI."""
        periods = [1, 2, 3]
        spi_by_period = {
            1: np.array([1.5, 1.5, 1.5]),
            2: np.array([-2.0, -2.0, -2.0]),
            3: np.array([0.5, 0.5, 0.5]),
        }
        df = pd.DataFrame({
            "period": [1, 2, 3],
            "sme_id": ["a", "b", "c"],
        })
        result = _join_and_lag_spi(df, "period", spi_by_period, periods, lag=1,
                                   output_col="spi_lag1")
        # Row at period=2 (index 1) → receives spi_by_period[1][1] = 1.5
        period2_val = result.loc[result["period"] == 2, "spi_lag1"].values[0]
        assert period2_val == pytest.approx(1.5)


# ── Geocoding fallback ─────────────────────────────────────────────────────────

class TestGeocodingFallback:

    def test_latlon_present_used_directly(self):
        df = pd.DataFrame({
            "sme_latitude": [1.0, 2.0],
            "sme_longitude": [10.0, 20.0],
        })
        lats, lons, used_centroid = _resolve_locations(
            df, "sme_latitude", "sme_longitude", None, None
        )
        assert not used_centroid
        np.testing.assert_array_equal(lats, [1.0, 2.0])

    def test_missing_latlon_uses_centroid(self):
        df = pd.DataFrame({
            "sme_latitude": [np.nan, np.nan],
            "sme_longitude": [np.nan, np.nan],
            "admin_unit": ["REGION_A", "REGION_B"],
        })
        centroids = {"REGION_A": (-1.3, 36.8), "REGION_B": (5.5, 1.2)}
        lats, lons, used_centroid = _resolve_locations(
            df, "sme_latitude", "sme_longitude",
            "admin_unit", centroids
        )
        assert used_centroid
        assert lats[0] == pytest.approx(-1.3)
        assert lons[0] == pytest.approx(36.8)

    def test_missing_latlon_no_fallback_raises(self):
        df = pd.DataFrame({
            "sme_latitude": [np.nan],
            "sme_longitude": [np.nan],
        })
        with pytest.raises(ValueError, match="admin_unit_col"):
            _resolve_locations(df, "sme_latitude", "sme_longitude", None, None)

    def test_centroid_fallback_warns(self, panel, period_to_yearmonth, mock_downloader):
        """GeocodeResolutionWarning fires when centroid fallback is used."""
        data = panel.copy()
        data["sme_latitude"] = np.nan
        data["sme_longitude"] = np.nan
        data["admin_unit"] = "REGION_A"
        centroids = {"REGION_A": (-1.3, 36.8)}

        with pytest.warns(GeocodeResolutionWarning):
            try:
                attach_chirps_anomaly(
                    data, lat_col="sme_latitude", lon_col="sme_longitude",
                    period_col="period",
                    period_to_yearmonth=period_to_yearmonth,
                    admin_unit_col="admin_unit",
                    admin_centroids=centroids,
                    _downloader=mock_downloader,
                )
            except Exception:
                pass  # May fail on SPI computation with single centroid; warning is what matters


# ── period_to_yearmonth validation ─────────────────────────────────────────────

class TestPeriodToYearmonth:

    def test_none_raises_value_error(self, panel, mock_downloader):
        with pytest.raises(ValueError, match="period_to_yearmonth"):
            attach_chirps_anomaly(
                panel,
                period_to_yearmonth=None,
                _downloader=mock_downloader,
            )

    def test_get_yearmonths_includes_baseline(self):
        """SPI-3 needs 2 extra months before first period."""
        periods = [1, 2, 3]
        p2ym = {1: (2020, 3), 2: (2020, 6), 3: (2020, 9)}
        yms = _get_yearmonths_needed(periods, p2ym, extra_baseline_months=2)
        assert (2020, 1) in yms   # 2 months before March
        assert (2020, 2) in yms
        assert (2020, 3) in yms
        assert (2020, 6) in yms
        assert (2020, 9) in yms

    def test_empty_period_mapping_raises(self):
        with pytest.raises(ValueError, match="None of the panel periods"):
            _get_yearmonths_needed([1, 2], {99: (2020, 1)}, extra_baseline_months=2)


# ── Full pipeline integration ──────────────────────────────────────────────────

class TestAttachChirpsAnomaly:

    def test_output_column_added(self, panel, period_to_yearmonth, mock_downloader):
        result = attach_chirps_anomaly(
            panel,
            period_to_yearmonth=period_to_yearmonth,
            _downloader=mock_downloader,
        )
        assert "rainfall_anomaly_lag1" in result.columns

    def test_output_same_length(self, panel, period_to_yearmonth, mock_downloader):
        result = attach_chirps_anomaly(
            panel,
            period_to_yearmonth=period_to_yearmonth,
            _downloader=mock_downloader,
        )
        assert len(result) == len(panel)

    def test_first_period_is_nan(self, panel, period_to_yearmonth, mock_downloader):
        """Lag=1: first period has no lagged value → NaN."""
        result = attach_chirps_anomaly(
            panel,
            period_to_yearmonth=period_to_yearmonth,
            lag=1,
            _downloader=mock_downloader,
        )
        first_period = result["period"].min()
        first_period_vals = result.loc[result["period"] == first_period, "rainfall_anomaly_lag1"]
        assert first_period_vals.isna().all()

    def test_later_periods_have_values(self, panel, period_to_yearmonth, mock_downloader):
        """Non-first periods should have non-NaN SPI values."""
        result = attach_chirps_anomaly(
            panel,
            period_to_yearmonth=period_to_yearmonth,
            lag=1,
            _downloader=mock_downloader,
        )
        last_period = result["period"].max()
        last_vals = result.loc[result["period"] == last_period, "rainfall_anomaly_lag1"]
        assert last_vals.notna().any()

    def test_custom_lag_produces_correct_column_name(
        self, panel, period_to_yearmonth, mock_downloader
    ):
        result = attach_chirps_anomaly(
            panel,
            period_to_yearmonth=period_to_yearmonth,
            lag=2,
            _downloader=mock_downloader,
        )
        assert "rainfall_anomaly_lag2" in result.columns

    def test_does_not_modify_input_data(self, panel, period_to_yearmonth, mock_downloader):
        original_cols = set(panel.columns)
        attach_chirps_anomaly(
            panel,
            period_to_yearmonth=period_to_yearmonth,
            _downloader=mock_downloader,
        )
        assert set(panel.columns) == original_cols  # original unchanged

    def test_download_error_warns_not_raises(self, panel, period_to_yearmonth):
        """Download failure produces warning + NaN, not an exception."""
        def failing_downloader(year, month, cache_dir):
            raise ConnectionError("Server unreachable")

        with pytest.warns(ShockInstrumentWarning):
            result = attach_chirps_anomaly(
                panel,
                period_to_yearmonth=period_to_yearmonth,
                _downloader=failing_downloader,
            )
        assert "rainfall_anomaly_lag1" in result.columns


# ── Synthetic shock instrument ─────────────────────────────────────────────────

class TestSyntheticShockInstrument:

    def test_column_added(self, panel):
        result = make_synthetic_shock_instrument(panel, seed=42)
        assert "rainfall_anomaly_lag1" in result.columns

    def test_length_preserved(self, panel):
        result = make_synthetic_shock_instrument(panel, seed=42)
        assert len(result) == len(panel)

    def test_first_period_is_nan(self, panel):
        result = make_synthetic_shock_instrument(panel, lag=1, seed=42)
        first_period = result["period"].min()
        vals = result.loc[result["period"] == first_period, "rainfall_anomaly_lag1"]
        assert vals.isna().all()

    def test_shock_probability_respected(self, panel):
        """With high shock probability, most periods should be below -1.5."""
        result = make_synthetic_shock_instrument(
            panel, shock_probability=0.9, lag=1, seed=42
        )
        shock_mask = result["rainfall_anomaly_lag1"] < -1.5
        non_nan = result["rainfall_anomaly_lag1"].notna()
        assert shock_mask[non_nan].mean() > 0.5

    def test_zero_shock_probability_no_shocks(self, panel):
        result = make_synthetic_shock_instrument(
            panel, shock_probability=0.0, lag=1, seed=42
        )
        valid = result["rainfall_anomaly_lag1"].dropna()
        assert (valid >= 0).all()

    def test_reproducible_with_seed(self, panel):
        r1 = make_synthetic_shock_instrument(panel, seed=42)
        r2 = make_synthetic_shock_instrument(panel, seed=42)
        pd.testing.assert_series_equal(
            r1["rainfall_anomaly_lag1"], r2["rainfall_anomaly_lag1"]
        )

    def test_does_not_modify_input(self, panel):
        original_cols = set(panel.columns)
        make_synthetic_shock_instrument(panel, seed=42)
        assert set(panel.columns) == original_cols

    def test_custom_output_col(self, panel):
        result = make_synthetic_shock_instrument(
            panel, output_col="drought_index", seed=42
        )
        assert "drought_index" in result.columns


# ── Pipeline integration: synthetic shock feeds into evaluator ─────────────────

class TestShockIntegrationWithPipeline:

    def test_synthetic_shock_compatible_with_evaluator(self):
        """
        make_synthetic_shock_instrument output is usable as ResilienceIndex
        shock_instrument in a full pipeline run.
        """
        import warnings as _warnings
        from gcef.pipeline import GreenCreditEvaluator
        from gcef.treatment import (GreenCreditTreatment, TreatmentType,
                                     ConditionalityMechanism, VerificationMethod)
        from gcef.outcomes import ResilienceIndex

        data = make_valid_panel(n_smes=80, n_lenders=3, n_periods=8, seed=42)
        data = make_synthetic_shock_instrument(data, seed=42)

        treatment = GreenCreditTreatment(
            type=TreatmentType.RATE_REDUCTION,
            conditionality_mechanism=ConditionalityMechanism.VERIFIED_INVESTMENT,
            verification_method=VerificationMethod.DOCUMENT_REVIEW,
        )
        outcome = ResilienceIndex(
            columns={"revenue": 0.60, "loan_repayment_rate": 0.40},
            shock_instrument="rainfall_anomaly_lag1",
            shock_threshold=-1.5,
        )
        evaluator = GreenCreditEvaluator(
            treatment=treatment, outcome=outcome,
            unit_id="sme_id", time_id="period", lender_id="lender_id",
            adoption_time="lender_green_adoption_period",
            covariates=["firm_age", "sector", "prior_revenue"],
            random_seed=42,
        )
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            results = evaluator.fit(data)

        assert results.late is not None
        assert results.cate is not None
        assert "rainfall_anomaly_lag1" in data.columns
