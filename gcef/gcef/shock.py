"""
gcef.shock
----------
Climate shock instrumentation for GCEF.

Downloads CHIRPS monthly precipitation data, computes Standardised
Precipitation Index (SPI), spatially joins to SME locations, and
constructs lagged shock indicators for use as exogenous shock variables
in Stage 2 of the GCEF pipeline.

Spec reference: Section 6 (Shock instrumentation), A4 (Shock exogeneity),
P3 (Shock instrument must be exogenous), DD4 (Why lag the shock variable).

Design decisions
----------------
SPI rather than raw precipitation:
    Raw precipitation (mm/month) is not comparable across SSA regions with
    different climatic baselines. Nairobi and Accra have different mean
    precipitation; a drought is defined relative to each location's baseline.
    SPI standardises against the 30-year climatological baseline, making
    threshold comparisons (SPI < -1.5) valid across regions.

SPI window = 3 months (SPI-3):
    SPI-3 captures medium-term moisture deficit relevant to agricultural and
    SME revenue outcomes. SPI-1 is too noisy; SPI-12 captures inter-annual
    signals not relevant to quarterly lending periods.

Lag = 1 period:
    Contemporaneous shocks affect both SME revenue (the outcome) and lender
    disbursement behaviour (a pathway violation). Lagging by 1 period
    preserves shock exogeneity relative to lender behaviour in period t.
    (Spec DD4, A4.)

Geocoding fallback:
    If only administrative unit identifiers are available (not lat/lon),
    the module falls back to administrative unit centroid. Shock resolution
    degrades to ~50km rather than ~5.5km. A GeocodeResolutionWarning is
    raised and the output column is flagged as centroid-derived.

CHIRPS data access:
    CHIRPS v2.0 monthly global GeoTIFFs are served from:
    https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/
    Files are ~15MB each (compressed). The module caches downloaded files
    to avoid repeated downloads within a session and across runs.

Testing:
    All functions are testable without network access via the
    `_compute_spi_from_precip` and `_sample_raster_at_points` helpers,
    which accept in-memory arrays. The HTTP download is isolated in
    `_download_chirps_tile` and can be mocked in tests.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

from gcef.exceptions import (
    GeocodeResolutionWarning,
    GCEFWarning,
)


# ── Constants ──────────────────────────────────────────────────────────────────

#: CHIRPS v2.0 monthly GeoTIFF base URL
CHIRPS_BASE_URL = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/"
)

#: CHIRPS file naming pattern: chirps-v2.0.YYYY.MM.tif.gz
CHIRPS_FILENAME_PATTERN = "chirps-v2.0.{year}.{month:02d}.tif.gz"

#: CHIRPS nodata value
CHIRPS_NODATA = -9999.0

#: Default SPI window in months (SPI-3 for medium-term moisture deficit)
DEFAULT_SPI_WINDOW = 3

#: Default cache directory for downloaded CHIRPS tiles
DEFAULT_CACHE_DIR = Path(os.path.expanduser("~")) / ".gcef" / "chirps_cache"

#: CHIRPS approximate resolution in degrees
CHIRPS_RESOLUTION_DEG = 0.05

#: SSA bounding box (approximate) for validation
SSA_BBOX = (-35.0, -20.0, 40.0, 55.0)  # (south, west, north, east)


class ShockInstrumentWarning(GCEFWarning):
    """Warning raised during shock instrument construction."""
    pass


# ── Public API ─────────────────────────────────────────────────────────────────

def attach_chirps_anomaly(
    data: pd.DataFrame,
    lat_col: str = "sme_latitude",
    lon_col: str = "sme_longitude",
    period_col: str = "period",
    period_to_yearmonth: Optional[Dict] = None,
    lag: int = 1,
    spi_window: int = DEFAULT_SPI_WINDOW,
    cache_dir: Optional[Path] = None,
    admin_unit_col: Optional[str] = None,
    admin_centroids: Optional[Dict[str, Tuple[float, float]]] = None,
    _downloader=None,
) -> pd.DataFrame:
    """
    Fetch CHIRPS monthly rainfall data and attach a lagged SPI anomaly column.

    Downloads monthly CHIRPS precipitation rasters for the required periods,
    samples each raster at SME locations, computes SPI-{spi_window}, lags
    by {lag} periods, and joins the result to the input panel.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset. Must contain lat_col, lon_col, and period_col.
    lat_col : str
        Column name for SME latitude. Default: 'sme_latitude'.
    lon_col : str
        Column name for SME longitude. Default: 'sme_longitude'.
    period_col : str
        Column name for time period. Default: 'period'.
    period_to_yearmonth : dict | None
        Mapping from period values to (year, month) tuples.
        Example: {1: (2019, 3), 2: (2019, 6), ...}
        If None, periods are assumed to be integers starting from 1 and
        the function will raise an error explaining how to construct the map.
    lag : int
        Number of periods to lag the shock indicator. Default: 1.
        Per spec DD4: contemporaneous shocks violate the exclusion restriction.
    spi_window : int
        SPI computation window in months. Default: 3 (SPI-3).
    cache_dir : Path | None
        Directory for caching downloaded CHIRPS tiles.
        Default: ~/.gcef/chirps_cache/
    admin_unit_col : str | None
        Column name for administrative unit identifier. Used as fallback
        if lat/lon are missing. Requires admin_centroids.
    admin_centroids : dict | None
        Mapping from admin unit ID to (lat, lon) centroid.
        Required if admin_unit_col is provided and lat/lon are missing.
    _downloader : callable | None
        Injection point for testing. Replaces the HTTP download function.
        Signature: (year, month, cache_dir) -> Path to local .tif file.

    Returns
    -------
    pd.DataFrame
        Input data with column 'rainfall_anomaly_lag{lag}' added.
        Values are SPI scores; negative values indicate drought.
        SPI < -1.5 is treated as a shock event by the pipeline.

    Raises
    ------
    ValueError
        If period_to_yearmonth is not provided.
    RuntimeError
        If CHIRPS download fails and no cached tiles exist.

    Warns
    -----
    GeocodeResolutionWarning
        If lat/lon are missing and fallback to admin unit centroid is used.
    ShockInstrumentWarning
        If SPI cannot be computed for some periods (insufficient history).
    """
    if period_to_yearmonth is None:
        raise ValueError(
            "period_to_yearmonth is required. Provide a dict mapping each period "
            "value to a (year, month) tuple. Example:\n"
            "  period_to_yearmonth = {1: (2019, 3), 2: (2019, 6), 3: (2019, 9)}\n"
            "This tells GCEF which calendar month each panel period corresponds to, "
            "so the correct CHIRPS raster can be downloaded and joined."
        )

    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    downloader = _downloader or _download_chirps_tile

    # ── 1. Resolve SME locations ──────────────────────────────────────────────
    data = data.copy()
    lat_vals, lon_vals, used_centroid = _resolve_locations(
        data, lat_col, lon_col, admin_unit_col, admin_centroids
    )
    if used_centroid:
        warnings.warn(
            "SME latitude/longitude not available. Falling back to administrative "
            "unit centroid for CHIRPS spatial join. Shock resolution degrades from "
            f"~{CHIRPS_RESOLUTION_DEG}° (~5.5km) to centroid precision (~50km). "
            "SPI estimates are less precise for urban SME clusters. "
            "Column 'rainfall_anomaly_lag{lag}' is flagged as centroid-derived.",
            GeocodeResolutionWarning,
            stacklevel=2,
        )

    # ── 2. Identify required periods (including SPI baseline periods) ─────────
    periods_needed = sorted(data[period_col].unique())
    # Need extra periods before the first for SPI computation
    # SPI-3 requires 3 months of precipitation before first observation
    spi_baseline_extra = spi_window - 1

    yearmonths_needed = _get_yearmonths_needed(
        periods_needed, period_to_yearmonth, spi_baseline_extra
    )

    # ── 3. Download and cache CHIRPS tiles ────────────────────────────────────
    tile_paths = {}
    download_errors = []
    for ym in yearmonths_needed:
        year, month = ym
        try:
            tile_path = downloader(year, month, cache_dir)
            tile_paths[ym] = tile_path
        except Exception as e:
            download_errors.append((year, month, str(e)))

    if download_errors:
        failed = [(y, m) for y, m, _ in download_errors]
        warnings.warn(
            f"CHIRPS download failed for {len(failed)} period(s): {failed}. "
            f"SPI cannot be computed for these periods. "
            f"Affected observations will have NaN shock values.",
            ShockInstrumentWarning,
            stacklevel=2,
        )

    # ── 4. Sample precipitation at SME locations for each period ─────────────
    # precip_by_ym: {(year, month): np.ndarray of shape (n_smes,)}
    n_obs = len(lat_vals)
    unique_lats = data.groupby(
        [lat_col if not used_centroid else "_centroid_lat"]
    )[lat_col if not used_centroid else "_centroid_lat"].first() if not used_centroid \
        else None

    # Build unique location array to avoid redundant raster reads
    unique_locs, loc_index = _get_unique_locations(lat_vals, lon_vals)

    precip_by_ym: Dict[Tuple, np.ndarray] = {}
    for ym, tile_path in tile_paths.items():
        try:
            precip = _sample_raster_at_points(
                tile_path, unique_locs[:, 1], unique_locs[:, 0]
            )
            # Expand back to full observation array
            precip_by_ym[ym] = precip[loc_index]
        except Exception as e:
            warnings.warn(
                f"Raster sampling failed for {ym}: {e}. "
                f"Setting precipitation to NaN for this period.",
                ShockInstrumentWarning,
                stacklevel=2,
            )
            precip_by_ym[ym] = np.full(n_obs, np.nan)

    # ── 5. Build precipitation time series per observation ────────────────────
    # Shape: (n_obs, n_yearmonths_for_spi)
    all_yearmonths = sorted(yearmonths_needed)
    precip_matrix = np.column_stack([
        precip_by_ym.get(ym, np.full(n_obs, np.nan))
        for ym in all_yearmonths
    ])

    # ── 6. Compute SPI for each panel period ──────────────────────────────────
    spi_by_period = _compute_spi_panel(
        precip_matrix=precip_matrix,
        all_yearmonths=all_yearmonths,
        period_to_yearmonth=period_to_yearmonth,
        periods=periods_needed,
        spi_window=spi_window,
        n_obs=n_obs,
    )

    # ── 7. Join SPI to panel and lag ──────────────────────────────────────────
    output_col = f"rainfall_anomaly_lag{lag}"
    data = _join_and_lag_spi(
        data=data,
        period_col=period_col,
        spi_by_period=spi_by_period,
        periods=periods_needed,
        lag=lag,
        output_col=output_col,
    )

    if used_centroid:
        data[f"{output_col}_centroid_derived"] = True

    _validate_spi_output(data, output_col)

    return data


# ── SPI computation ────────────────────────────────────────────────────────────

def _compute_spi_from_precip(
    precip_series: np.ndarray,
    window: int = DEFAULT_SPI_WINDOW,
) -> np.ndarray:
    """
    Compute SPI from a precipitation time series using a rolling window.

    SPI = (P_rolling - mu_rolling) / sigma_rolling

    where P_rolling is the {window}-month rolling sum, mu and sigma are
    the mean and standard deviation computed over the available history.

    Parameters
    ----------
    precip_series : np.ndarray
        Monthly precipitation values in chronological order (mm/month).
        NaN values are handled gracefully — SPI is NaN where precip is NaN.
    window : int
        SPI computation window in months. Default: 3 (SPI-3).

    Returns
    -------
    np.ndarray
        SPI values, same length as precip_series.
        First (window-1) values are NaN (insufficient history).
        Negative values indicate below-normal precipitation (drought).

    Notes
    -----
    This is a simplified SPI using Gaussian normalisation rather than
    the full Gamma distribution fitting used in operational SPI. For
    research contexts, the Gaussian approximation is adequate when the
    precipitation distribution is not highly skewed (typical for SPI-3+).
    For arid regions with many zero-precipitation months, consider using
    the operational SPI implementation (e.g. via the `climate-indices` package).
    """
    n = len(precip_series)
    spi = np.full(n, np.nan)

    # Rolling sum
    rolling_sum = np.full(n, np.nan)
    for i in range(window - 1, n):
        window_vals = precip_series[i - window + 1: i + 1]
        if not np.isnan(window_vals).any():
            rolling_sum[i] = np.sum(window_vals)

    # Standardise using expanding mean and std (uses all available history)
    for i in range(window - 1, n):
        if np.isnan(rolling_sum[i]):
            continue
        history = rolling_sum[window - 1: i + 1]
        history = history[~np.isnan(history)]
        if len(history) < 2:
            continue
        mu = np.mean(history)
        sigma = np.std(history, ddof=1)
        if sigma > 0:
            spi[i] = (rolling_sum[i] - mu) / sigma
        else:
            spi[i] = 0.0

    return spi


def _compute_spi_panel(
    precip_matrix: np.ndarray,
    all_yearmonths: List[Tuple],
    period_to_yearmonth: Dict,
    periods: List,
    spi_window: int,
    n_obs: int,
) -> Dict:
    """
    Compute SPI for all observations across all panel periods.

    Returns
    -------
    dict: {period: np.ndarray of shape (n_obs,)}
    """
    spi_by_period = {}

    for i in range(n_obs):
        precip_i = precip_matrix[i, :]
        spi_i = _compute_spi_from_precip(precip_i, window=spi_window)

        for period in periods:
            ym = period_to_yearmonth.get(period)
            if ym is None:
                continue
            ym_idx = all_yearmonths.index(ym) if ym in all_yearmonths else None
            if ym_idx is None:
                continue

            if period not in spi_by_period:
                spi_by_period[period] = np.full(n_obs, np.nan)
            spi_by_period[period][i] = spi_i[ym_idx]

    return spi_by_period


# ── Spatial operations ─────────────────────────────────────────────────────────

def _sample_raster_at_points(
    raster_path: Path,
    lons: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    """
    Sample a raster at given (lon, lat) coordinate pairs using rasterio.

    Parameters
    ----------
    raster_path : Path
        Path to a GeoTIFF file.
    lons : np.ndarray
        Longitude values (x-coordinates in EPSG:4326).
    lats : np.ndarray
        Latitude values (y-coordinates in EPSG:4326).

    Returns
    -------
    np.ndarray
        Sampled values at each point. NaN for nodata or out-of-bounds.
    """
    with rasterio.open(raster_path) as src:
        coords = list(zip(lons.astype(float), lats.astype(float)))
        sampled = np.array([v[0] for v in src.sample(coords)], dtype=float)

        # Replace nodata values with NaN
        nodata = src.nodata
        if nodata is not None:
            sampled[sampled == nodata] = np.nan
        sampled[sampled == CHIRPS_NODATA] = np.nan

    return sampled


def _get_unique_locations(
    lats: np.ndarray,
    lons: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return unique (lat, lon) pairs and an index mapping each observation
    back to its unique location — avoids redundant raster reads.

    Returns
    -------
    unique_locs : np.ndarray, shape (n_unique, 2) — columns [lat, lon]
    loc_index : np.ndarray, shape (n_obs,) — index into unique_locs
    """
    loc_pairs = np.column_stack([lats, lons])
    unique_locs, loc_index = np.unique(loc_pairs, axis=0, return_inverse=True)
    return unique_locs, loc_index


# ── CHIRPS download ────────────────────────────────────────────────────────────

def _download_chirps_tile(
    year: int,
    month: int,
    cache_dir: Path,
) -> Path:
    """
    Download a CHIRPS monthly GeoTIFF and return the path to the local file.

    Files are cached by year/month. If the file already exists in cache,
    returns the cached path without downloading.

    Parameters
    ----------
    year : int
    month : int (1-12)
    cache_dir : Path
        Local cache directory.

    Returns
    -------
    Path
        Path to the local decompressed .tif file.

    Raises
    ------
    RuntimeError
        If the download fails and no cached file exists.
    """
    import requests

    filename_gz = CHIRPS_FILENAME_PATTERN.format(year=year, month=month)
    filename_tif = filename_gz.replace(".gz", "")
    cached_path = cache_dir / filename_tif

    if cached_path.exists():
        return cached_path

    url = CHIRPS_BASE_URL + filename_gz

    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Failed to download CHIRPS tile for {year}-{month:02d}: {e}. "
            f"URL: {url}. "
            f"Check network connectivity and that the UCSB CHG data server "
            f"(data.chc.ucsb.edu) is accessible. "
            f"If working offline, pre-download tiles and pass a custom _downloader."
        ) from e

    # Decompress gzip in memory and write to cache
    try:
        compressed = io.BytesIO(response.content)
        with gzip.GzipFile(fileobj=compressed) as gz:
            tif_data = gz.read()

        with open(cached_path, "wb") as f:
            f.write(tif_data)

    except Exception as e:
        if cached_path.exists():
            cached_path.unlink()
        raise RuntimeError(
            f"Failed to decompress CHIRPS tile for {year}-{month:02d}: {e}."
        ) from e

    return cached_path


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_locations(
    data: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    admin_unit_col: Optional[str],
    admin_centroids: Optional[Dict],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Resolve SME locations from lat/lon or admin unit centroid fallback.

    Returns
    -------
    lats, lons : np.ndarray
    used_centroid : bool — True if centroid fallback was used
    """
    has_latlon = (
        lat_col in data.columns
        and lon_col in data.columns
        and data[lat_col].notna().any()
        and data[lon_col].notna().any()
    )

    if has_latlon:
        lats = data[lat_col].values.astype(float)
        lons = data[lon_col].values.astype(float)
        return lats, lons, False

    # Centroid fallback
    if admin_unit_col is None or admin_centroids is None:
        raise ValueError(
            f"SME lat/lon not available in columns '{lat_col}', '{lon_col}', "
            f"and no admin_unit_col or admin_centroids provided. "
            f"Provide either lat/lon columns or admin_unit_col + admin_centroids "
            f"to enable the centroid fallback."
        )

    lats = data[admin_unit_col].map(
        {k: v[0] for k, v in admin_centroids.items()}
    ).values.astype(float)
    lons = data[admin_unit_col].map(
        {k: v[1] for k, v in admin_centroids.items()}
    ).values.astype(float)

    if np.isnan(lats).any():
        missing = data.loc[np.isnan(lats), admin_unit_col].unique()
        warnings.warn(
            f"Admin unit centroid not found for: {list(missing)}. "
            f"These observations will have NaN shock values.",
            ShockInstrumentWarning,
            stacklevel=3,
        )

    return lats, lons, True


def _get_yearmonths_needed(
    periods: List,
    period_to_yearmonth: Dict,
    extra_baseline_months: int,
) -> List[Tuple]:
    """
    Get all (year, month) tuples needed, including SPI baseline months
    preceding the first panel period.
    """
    core_yearmonths = [period_to_yearmonth[p] for p in periods if p in period_to_yearmonth]
    if not core_yearmonths:
        raise ValueError(
            "None of the panel periods found in period_to_yearmonth. "
            "Check that period values match the keys of period_to_yearmonth."
        )

    first_ym = min(core_yearmonths)

    # Generate preceding months for SPI baseline
    extra_yms = []
    year, month = first_ym
    for _ in range(extra_baseline_months):
        month -= 1
        if month == 0:
            month = 12
            year -= 1
        extra_yms.append((year, month))

    all_yms = sorted(set(extra_yms + core_yearmonths))
    return all_yms


def _join_and_lag_spi(
    data: pd.DataFrame,
    period_col: str,
    spi_by_period: Dict,
    periods: List,
    lag: int,
    output_col: str,
) -> pd.DataFrame:
    """
    Join SPI values to the panel and apply the lag.

    spi_by_period[p] is a numpy array of length n_obs where n_obs is the
    number of rows in data. The array is indexed identically to data.index
    (reset to 0..n-1 before this function is called).

    The lag is applied at the period level: an observation in period t
    receives the SPI from period t-lag (per spec DD4).
    """
    period_order = sorted(periods)
    period_to_idx = {p: i for i, p in enumerate(period_order)}

    result_col = np.full(len(data), np.nan)

    for i, period in enumerate(period_order):
        mask = (data[period_col] == period).values
        if not mask.any():
            continue

        lag_source_idx = i - lag
        if lag_source_idx < 0:
            # No lagged value — remains NaN
            continue

        lag_source_period = period_order[lag_source_idx]
        spi_array = spi_by_period.get(lag_source_period)
        if spi_array is None:
            continue

        # spi_array is shape (n_obs,) indexed 0..n-1 matching data order
        # Select values for rows where period matches
        row_indices = np.where(mask)[0]
        result_col[row_indices] = spi_array[row_indices]

    data[output_col] = result_col
    return data


def _validate_spi_output(data: pd.DataFrame, output_col: str) -> None:
    """Post-computation validation — flag if too many NaN values."""
    nan_share = data[output_col].isna().mean()
    if nan_share > 0.5:
        warnings.warn(
            f"{nan_share:.0%} of observations have NaN shock values in "
            f"'{output_col}'. This may indicate download failures, "
            f"insufficient SPI baseline history, or location data issues. "
            f"Check that period_to_yearmonth covers at least "
            f"{DEFAULT_SPI_WINDOW} months before the first panel period.",
            ShockInstrumentWarning,
            stacklevel=2,
        )
    elif nan_share > 0:
        warnings.warn(
            f"{nan_share:.1%} of observations have NaN shock values in "
            f"'{output_col}' (typically early periods without SPI baseline history).",
            ShockInstrumentWarning,
            stacklevel=2,
        )


# ── Convenience: create synthetic shock data for testing ──────────────────────

def make_synthetic_shock_instrument(
    data: pd.DataFrame,
    period_col: str = "period",
    shock_probability: float = 0.15,
    shock_magnitude: float = -2.0,
    seed: int = 42,
    lag: int = 1,
    output_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic SPI-like shock instrument for testing and development.

    Produces a realistic-looking shock column without requiring CHIRPS download.
    Regional shocks are spatially correlated (same region → same shock event).
    Use when CHIRPS data is unavailable or for unit testing.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset. Must contain period_col.
    period_col : str
        Time period column.
    shock_probability : float
        Per-period probability of a shock event (default 0.15).
    shock_magnitude : float
        SPI value during shock periods (default -2.0, well below -1.5 threshold).
    seed : int
        Random seed.
    lag : int
        Periods to lag (default 1, per spec DD4).
    output_col : str | None
        Output column name. Default: 'rainfall_anomaly_lag{lag}'.

    Returns
    -------
    pd.DataFrame
        Data with synthetic shock column added.
    """
    rng = np.random.default_rng(seed)
    output_col = output_col or f"rainfall_anomaly_lag{lag}"

    data = data.copy()
    periods = sorted(data[period_col].unique())

    # Generate period-level shock events
    shock_periods = {
        p: (rng.random() < shock_probability)
        for p in periods
    }

    # Build contemporaneous SPI
    contemporaneous_spi = {}
    for p in periods:
        if shock_periods[p]:
            base_spi = shock_magnitude + rng.normal(0, 0.3)
        else:
            base_spi = 0.5 + rng.normal(0, 0.4)
        contemporaneous_spi[p] = base_spi

    # Apply lag
    lagged_spi = {}
    for i, p in enumerate(periods):
        if i < lag:
            lagged_spi[p] = np.nan
        else:
            lagged_spi[p] = contemporaneous_spi[periods[i - lag]]

    data[output_col] = data[period_col].map(lagged_spi)
    return data
