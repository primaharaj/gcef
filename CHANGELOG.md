# Changelog

All notable changes to GCEF are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-05-21

First public release. All core modules implemented; zero stubs remaining.

### Added

**Core pipeline**
- `GreenCreditEvaluator.fit()` — end-to-end two-stage causal pipeline from raw SME panel data to identified treatment effect estimates
- `GCEFResults` — structured results object; every output carries a non-optional `Estimand` with explicit population statement
- `COMPLIER_KAPPA_THRESHOLD = 0.1` — module constant for complier mask construction; documented and testable

**Treatment specification (`treatment.py`)**
- `GreenCreditTreatment` — structured treatment vector with `TreatmentType`, `ConditionalityMechanism`, `VerificationMethod` enums
- `BlendedTreatmentNotImplemented` — deferred to v0.2 with a clear error message and contribution guidance
- DAG routing by treatment type — different identification strategies for different treatment archetypes

**Outcome specification (`outcomes.py`)**
- `ResilienceIndex` — composite outcome index with explicitly documented, theory-driven weights
- Rolling CV derivation of `revenue_stability` from `revenue` — internally derived, consistent across analyses
- `ResilienceIndexWarning` — fires when fewer than `underidentification_threshold` components are available
- `stability_window` parameter — number of periods for rolling CV computation (default: 4)

**Estimand (`estimand.py`)**
- `Estimand` dataclass — structured, machine-readable population statement attached to every output
- `make_complier_estimand()` and `make_bounds_estimand()` factory helpers
- `to_prose()`, `to_dict()`, `to_json()`, `to_checklist_row()` renderers
- `VALID_POPULATIONS`, `DFI_FRAMING_DEFAULTS` constants — framing in DFI vocabulary (additionality, marginal targeting)

**Stage 1 — IV/DiD (`stage1.py`)**
- Callaway & Sant'Anna (2021) staggered DiD via `pyfixest`, with lender adoption timing as instrument
- Abadie (2003) kappa weight computation using period-stratified propensity (logistic regression)
- `Stage1Result` dataclass — typed output with `late`, `late_se`, `late_ci`, `f_statistic`, `kappa_weights`, `complier_profile`, `complier_share`, `complier_share_ci`
- `ADOPTION_TIMING_ANOMALY` empirical check for A3 monotonicity support
- Complier covariate profile — weighted mean, full sample mean, complier-to-full ratio per covariate

**Stage 2 — Causal forest (`stage2.py`)**
- `econml.iv.dr.ForestDRIV` with kappa-weighted `sample_weight` — complier restriction without individual identification
- Shock-period subsetting — causal forest fitted on shock-period observations, CATE predicted for full dataset (full-length output)
- SE derived from `conf_int()` width — documented approximation per spec limitation L8
- Separate `GradientBoostingClassifier` for overlap propensity score (independent of ForestDRIV nuisance model)
- `_encode_covariates()` — one-hot encoding with median NaN imputation

**Bounds (`bounds.py`)**
- Manski (1990) worst-case partial identification bounds for always-takers and never-takers
- Double condition classification (kappa ≤ 0 AND D/Z pattern) — conservative, avoids propensity model misclassification
- Full-length output — one row per input observation; complier rows have NaN bounds
- `outcome_support_lo`/`_hi` columns — 1st/99th percentile with degenerate support fallback
- `_log_bounds_summary()` — summary warning with subpopulation counts and mean bound widths

**Assumptions (`assumptions.py`)**
- All seven identification assumptions tested where possible (A1–A7)
- Structured `assumptions_tested` schema — `passed`, `value`, `threshold`, `test`, `warning` per assumption
- `check_kappa_weights(design=)` — design-aware thresholds; staggered DiD tolerates 17–30% non-positive kappa (structurally expected)
- `WeakInstrumentError` at F < 10; `WeakInstrumentWarning` at F < 20
- `SingleLenderError` on single-lender datasets
- `ShortPanelWarning` at < 3 pre-treatment periods; `ShortPanelError` at < 2

**Shock instrumentation (`shock.py`)**
- `attach_chirps_anomaly()` — downloads CHIRPS v2.0 monthly GeoTIFFs, computes SPI-3, spatial join to SME locations, applies 1-period lag
- `_download_chirps_tile()` — HTTP download with local caching; injectable via `_downloader` for testing
- `_compute_spi_from_precip()` — Gaussian SPI using expanding history; documented approximation vs Gamma-distribution operational SPI
- `_sample_raster_at_points()` — `rasterio`-based spatial sampling with nodata → NaN handling
- `_get_unique_locations()` — deduplicates lat/lon before raster reads to avoid redundant I/O
- Admin unit centroid fallback with `GeocodeResolutionWarning`
- `make_synthetic_shock_instrument()` — realistic SPI-like shock column for development and testing without CHIRPS access

**Report generation (`report.py`)**
- `ReportGenerator` — PDF (reportlab) and HTML (Jinja2) output
- Nine report sections per spec: executive summary, estimand statement, complier profile, LATE, CATE heterogeneity, Manski bounds, assumption audit, methodological notes, reproducibility statement
- Red flags in assumption audit propagate to executive summary — cannot be buried
- Reproducibility statement mandatory and non-suppressible — GCEF version, random seed, data hash, analyst, nuisance overrides
- `hash_dataframe()` — SHA-256 hash of input data for reproducibility; call before `fit()`
- `_ReportData` assembly layer — both PDF and HTML renderers consume the same data object
- `results.report()` shortcut on `GCEFResults`

**Exceptions (`exceptions.py`)**
- Full hierarchy of structured warnings and errors with metadata attributes
- All errors carry actionable context (F-statistic value, affected cohort IDs, anomaly counts)
- `BlendedTreatmentNotImplemented` — `NotImplementedError` subclass with v0.2 guidance

**Testing infrastructure (`gcef/testing/synthetic.py`)**
- `GCEFDataGenerator` / factory functions — synthetic SSA SME panel with structured DGP
- Hard-gate instrument: lender adoption is a necessary condition for take-up (enforces monotonicity)
- DGP truth values accessible for estimate validation (`expected_true_late()`, `expected_complier_share()`)
- Scenario factories for every spec failure mode: `make_weak_instrument`, `make_single_lender`, `make_short_panel`, `make_adoption_anomaly`, `make_missing_geocoding`, `make_self_reported_treatment`, `make_low_complier_share`, `make_exclusion_violation`

### Known limitations (v0.1)

- **Blended treatment types** — deferred to v0.2; `BlendedTreatmentNotImplemented` raised with guidance
- **Lee (2009) tighter bounds** — requires monotone treatment selection; deferred to v0.2
- **MODIS flood extent** — CHIRPS implemented; MODIS deferred to v0.2
- **ForestDRIV CI approximation** — CIs derived from `conf_int()` width; may undercover with highly heterogeneous kappa weights (spec L8)
- **Always-taker synthetic data** — hard-gate DGP makes D=1, Z=0 impossible in clean panels; always-taker bounds tests skip on synthetic data (pass on real portfolio data with pre-adoption records)

### Design decisions resolved in this version

- **`_Z` constructed in pipeline** — instrument column built in `pipeline.py` before either stage; both stages receive it via `data` rather than Stage 1 returning augmented data
- **`ForestDRIV` nuisance models are regressors** — `model_t_xw` in `ForestDRIV` is E[T|X] (continuous regression), not P(T=1|X); overlap check uses a separate `GradientBoostingClassifier`
- **Kappa weight thresholds are design-dependent** — staggered DiD tolerates higher non-positive shares than cross-sectional IV; `check_kappa_weights(design=)` selects the correct threshold
- **`cate_complier_mask` is full-length** — one boolean per input row; analysts join to portfolio data directly
- **Manski bounds not extrapolated estimates** — `cate_bounds` contains bounds for non-identified subpopulations, never point estimates with inflated CIs

---

## [Unreleased] — v0.2 planned items

- Blended treatment types with multi-pathway DAG specification
- Lee (2009) tighter bounds with monotone selection assumption
- MODIS flood extent integration in `shock.py`
- Native Callaway & Sant'Anna Python implementation (remove `pyfixest` dependency for C&S)
- `make_with_always_takers()` DGP factory for bounds integration testing
- Sensitivity analysis module — E-values, Rosenbaum bounds
- Rambachan & Roth (2023) honest parallel trends tests
