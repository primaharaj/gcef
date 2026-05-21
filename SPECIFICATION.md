# Green Credit Causal Evaluation Framework (GCEF)
## Framework Specification — v0.7
### Working document · Priya Maharaj · May 2026

---

## 1. Overview

GCEF is an open-source Python framework for causal impact evaluation of green credit instruments on SME resilience in Sub-Saharan African lending markets.

It fills a specific and documented gap: most published impact evaluations of green credit programmes in SSA are correlational. They show that SMEs who received green credit improved on outcome measures — but do not credibly separate genuine causal impact from selection effects. SMEs that seek and access green credit are systematically different from those that do not: better managed, more climate-aware, more likely to invest in adaptation regardless of the loan. Without controlling for this selection, impact reports overstate causal effects and misdirect capital.

GCEF provides a rigorous, reproducible, openly documented methodology for separating selection from causation in this context. It is designed for:

- DFI analysts evaluating green credit portfolios without needing to be causal inference specialists
- Researchers conducting academic impact evaluation in SSA climate finance
- Lenders building internal impact measurement capacity

The framework produces four outputs: a Local Average Treatment Effect (LATE) estimate at the SME level, Conditional Average Treatment Effects (CATE) across the complier subpopulation, a transparent composite resilience index with documented weights, and a complier covariate profile that characterises the marginal SME population in terms actionable for DFI targeting and additionality assessment. The complier profile is a primary output, not a diagnostic.

---

## 2. The problem this solves

Green credit impact reports in SSA typically show:

> "SMEs that received green credit showed X% improvement in revenue stability compared to the control group."

This claim is almost never causally identified. The comparison group is typically:
- SMEs that did not apply (selection on unobservables)
- SMEs that applied but were rejected (selection on observables, but endogenous)
- A before/after comparison for the same SMEs (confounded by time trends)

None of these comparisons isolates the causal effect of the credit instrument. The result is a literature in which every programme appears effective, capital allocation decisions are made on inflated estimates, and the genuinely impactful interventions cannot be distinguished from ineffective ones.

GCEF's identification strategy exploits variation in *lender adoption timing* — the fact that different lenders introduced green credit products at different points in time — as an instrument for SME take-up. This variation is plausibly exogenous to individual SME characteristics, making it a valid instrument for estimating the causal effect of access to green credit on SME outcomes.

---

## 3. Design principles

**P1 — Treatment is a vector, not a scalar.**
Green credit is not one thing. A rate-reduced loan for solar panel installation has a different causal pathway than a green covenant attached to working capital. GCEF represents treatment as a structured object that captures type, conditionality mechanism, verification method, and intensity. The framework selects estimator components based on treatment specification rather than forcing one DAG on all use cases.

**P2 — Outcomes are composite and context-dependent.**
No single outcome variable captures SME resilience across all contexts and datasets. GCEF constructs a composite resilience index from whatever outcome columns are available, with explicitly documented weights. Analysts can customise weights; the framework flags when the composite is underidentified.

**P3 — The shock instrument must be exogenous.**
Revenue stability during shocks is the primary outcome variable for the causal forest stage. Identifying this requires an exogenous shock variable orthogonal to lender behaviour. GCEF uses climate event data (CHIRPS rainfall anomalies, MODIS flood extent) at sub-national resolution. **The shock variable should be lagged** — contemporaneous shocks may affect both SME outcomes and lender behaviour in the same period, violating the exclusion restriction.

**P4 — The pipeline is two-stage but runs as one.**
Stage 1 estimates LATE using IV/DiD with staggered adoption. Stage 2 estimates CATE across the complier subpopulation using a causal forest. The analyst runs a single pipeline; the stages execute sequentially with intermediate outputs available for inspection.

**P5 — Assumptions are explicit and testable.**
Every identification assumption is documented in the framework and, where possible, tested programmatically. The framework will not silently proceed when assumptions are likely violated — it flags and warns.

**P6 — Reproducibility is non-negotiable.**
All random seeds are set and documented. All data transformations are logged. All intermediate outputs are saved. A GCEF analysis should be fully reproducible from raw data inputs.

---

## 4. Core objects

### 4.1 `GreenCreditTreatment`

Represents the treatment as a structured vector rather than a binary indicator.

```python
from gceval import GreenCreditTreatment

treatment = GreenCreditTreatment(
    type="rate_reduction",              # see TreatmentType enum
    conditionality_mechanism="verified_investment",   # see CondMechanism enum
    verification_method="document_review",            # see VerificationMethod enum
    intensity=0.03                      # e.g. 3% rate reduction; None if not applicable
)
```

**`type` options (TreatmentType enum):**
- `rate_reduction` — interest rate reduced conditional on green use
- `equipment_loan` — loan specifically for renewable/green equipment purchase
- `green_covenant` — standard loan with environmental covenant attached
- `sector_restricted` — loan restricted to firms in designated green sectors
- `blended` — composite instrument combining multiple treatment types. **Deferred to v0.2.** In v0.1, passing `type="blended"` raises `NotImplementedError` with the message: "Blended treatment types require multi-pathway DAG specification, which is not implemented in v0.1. Decompose the instrument into constituent treatment types and run separate evaluations, or contribute a blended treatment implementation via the GitHub repository." Implementing blended treatment DAGs correctly requires specifying how pathway-specific effects are partially identified — shipping it half-built is worse than deferring it.

**`conditionality_mechanism` options:**
- `verified_investment` — disbursement or rate conditional on verified green investment
- `sector_classification` — eligibility based on sector classification
- `covenant_compliance` — ongoing compliance monitoring
- `self_reported` — SME self-reports green use; not independently verified

**`verification_method` options:**
- `site_visit` — physical verification by lender or third party
- `document_review` — invoice/receipt verification
- `satellite_remote_sensing` — remote sensing verification (e.g. solar installation)
- `self_reported` — no independent verification

**Why this matters for the DAG:**
A `rate_reduction` + `verified_investment` + `site_visit` treatment has a fundamentally different causal pathway than `sector_restricted` + `self_reported`. In the first case, the lender actively intervenes in the investment decision. In the second, the lender is selecting into a sector, not changing SME behaviour. These require different identification strategies and should not be pooled without justification.

---

### 4.2 `ResilienceIndex`

Constructs a composite outcome index from available columns with explicit, documented weights.

```python
from gceval import ResilienceIndex

outcome = ResilienceIndex(
    columns={
        "revenue": 0.40,              # framework derives revenue_stability internally
        "loan_repayment_rate": 0.30,
        "employment": 0.20,           # framework derives employment_stability internally
        "adaptation_investment": 0.10
    },
    shock_instrument="rainfall_anomaly_lag1",   # lagged CHIRPS anomaly column
    shock_threshold=-1.5,                        # standard deviations below mean
    stability_window=4,                          # periods for rolling CV computation
    underidentification_threshold=2              # flag if fewer than N columns available
)
```

**Revenue stability derivation (Option A — computed internally):**
`revenue_stability` is not a required analyst-supplied column. It is derived internally from the `revenue` column using a rolling coefficient of variation:

`revenue_stability_it = 1 − CV(revenue_i, window=stability_window)`

where CV = σ/μ computed over a rolling window of `stability_window` periods (default: 4). The result is dimensionless and comparable across firms of different sizes. Higher values indicate greater stability. A value of 1.0 would indicate zero variation (impossible in practice); values below 0 indicate that the standard deviation exceeds the mean, signalling extreme volatility.

**Why this derivation:**
- Dimensionless — comparable across firms of different revenue scales
- Requires `stability_window` periods of revenue data per firm (default 4); firms with fewer periods are flagged as `INSUFFICIENT_REVENUE_HISTORY` and excluded from the composite with a warning
- Computed before shock conditioning — the stability measure captures baseline revenue behaviour, which is then evaluated during shock periods by the causal forest

**Employment stability** is derived analogously from the `employment` column: `employment_stability_it = 1 − CV(employment_i, window=stability_window)`.

**Analyst override:** If an analyst has a theoretically motivated alternative derivation of revenue stability (e.g. deviation from pre-shock trend per Option B), they may supply `revenue_stability` as a pre-computed column directly. If both `revenue` and `revenue_stability` are present in the data, the pre-computed column takes precedence and a `UserDerivedStabilityWarning` is logged, requiring the analyst to document their derivation method in the report module's reproducibility statement.

**Default weights (theory-driven, not data-driven):**

| Component | Input column | Derived as | Default weight | Rationale |
|---|---|---|---|---|
| Revenue stability during shocks | `revenue` | Rolling 1−CV | 0.40 | Most direct measure; continuous; causally downstream |
| Loan repayment rate | `loan_repayment_rate` | Direct | 0.30 | Financial resilience; available in lender data |
| Employment stability | `employment` | Rolling 1−CV | 0.20 | Community-level resilience via employment |
| Adaptation investment | `adaptation_investment` | Direct | 0.10 | Forward-looking; often missing; lowest weight |

**Underidentification flag:**
When fewer than `underidentification_threshold` (default: 2) components are available, the framework raises a `ResilienceIndexWarning` and includes an explicit warning in all outputs. The analyst must acknowledge this flag to proceed.

**Revenue stability as the primary causal forest outcome:**
For Stage 2 (causal forest), GCEF uses revenue stability during shocks as the primary outcome rather than the composite index. Reasons:
1. It is continuous — causal forest performs better with continuous outcomes
2. It is causally downstream of the treatment mechanism (green investment → reduced climate exposure → revenue stability during climate events)
3. Heterogeneous effects across firm type and geographic climate exposure are substantively interesting for policy
4. It requires an identified shock variable, which forces explicit shock instrumentation

---

### 4.3 `GreenCreditEvaluator`

The main pipeline object. Takes a treatment specification, outcome specification, and dataset. Runs the full two-stage pipeline.

```python
from gceval import GreenCreditEvaluator

evaluator = GreenCreditEvaluator(
    treatment=treatment,
    outcome=outcome,
    unit_id="sme_id",
    time_id="period",
    lender_id="lender_id",
    adoption_time="lender_green_adoption_period",   # staggered adoption column
    covariates=["firm_age", "sector", "region", "firm_size", "prior_revenue"]
)

results = evaluator.fit(data)

# Results object — full structure
results.late                # LATE estimate with SE — defined for compliers
results.cate                # CATE per observation — full-length array, κ-weighted
                            #   (one row per input row; always-takers/never-takers
                            #    receive near-zero weight but are not filtered out)
results.cate_complier_mask  # Boolean Series — True where κ > 0.1 (complier rows)
                            #   Use: complier_cate = results.cate[results.cate_complier_mask]
results.complier_profile    # Covariate distribution (κ-weighted) for complier subpopulation
results.complier_share      # Share of sample classified as compliers, with CI
results.cate_bounds         # Manski (1990) partial identification bounds for always-takers + never-takers
results.estimand            # Structured estimand object — non-optional, machine-readable
results.assumptions_tested  # Structured dictionary — see schema below

**`assumptions_tested` schema (Option A — machine-readable, consistent across deployments):**
```python
results.assumptions_tested = {
    "instrument_relevance": {
        "passed": True,
        "value": 18.4,
        "threshold": 10,
        "test": "first_stage_F",
        "warning": None
    },
    "panel_length": {
        "passed": False,
        "value": 2,
        "threshold": 3,
        "test": "min_pre_treatment_periods",
        "cohorts_affected": ["2021_cohort"],
        "warning": "ShortPanelWarning: cohort 2021 has 2 pre-treatment periods (minimum 3 required for credible parallel trends test)"
    },
    "overlap": {
        "passed": True,
        "value": 0.12,
        "threshold": 0.10,
        "test": "propensity_trimming_share",
        "warning": None
    },
    "adoption_timing_anomaly": {
        "passed": True,
        "value": 0,
        "test": "sme_takeup_precedes_lender_adoption",
        "warning": None
    },
    "complier_share": {
        "passed": True,
        "value": 0.34,
        "threshold": 0.20,
        "test": "iv_complier_share",
        "warning": None
    },
    "kappa_weight_negatives": {
        "passed": True,
        "value": 0.03,
        "warning_threshold": 0.05,
        "error_threshold": 0.15,
        "test": "share_nonpositive_kappa_weights",
        "warning": None
        # warning triggered at > 5% non-positive weights
        # error raised at > 15% non-positive weights (conventional IV reweighting thresholds)
    }
}
```
Every key maps to a single assumption or check. `passed` is always a boolean. `value` is always the computed statistic. `threshold` is included where a threshold exists. `warning` is `None` if passed, a string message if failed. The schema is extensible — additional tests add new keys without breaking existing ones.

**Note on `cate_bounds`:**
v0.1 implements Manski (1990) worst-case bounds only. Lee (2009) bounds are tighter but require a monotone treatment selection assumption (treatment take-up is monotone in potential outcomes) that is distinct from and additional to the IV monotonicity in A3. Adding A8 to justify Lee bounds is out of scope for v0.1. Lee bounds are deferred to v0.2 and noted in the roadmap. The Lee (2009) citation in Section 12 is retained with a clarifying note marking it as v0.2.
```

**Note on `cate_bounds` vs `cate_extrapolated`:**
The results object deliberately does not include `cate_extrapolated` as point estimates with inflated confidence intervals. For always-takers and never-takers, the identification strategy provides no variation — wider CIs would imply precision that does not exist. Instead, `cate_bounds` implements Manski-style partial identification bounds: honest bounds on what the data can and cannot rule out, not estimates with fictional uncertainty. A DFI analyst can work with honest bounds. Discovering that "wide CI estimates" lacked identification destroys trust in the framework. This distinction — bounds vs. estimates — must be surfaced clearly in all outputs.

**`estimand` as a structured object (non-optional):**
```python
results.estimand = Estimand(
    population="compliers",
    population_description="SMEs who accessed green credit because their lender offered it, and who would not have accessed it otherwise",
    dfi_framing="Marginal SMEs — those whose green credit behaviour changes with programme availability. Relevant for additionality assessment.",
    treatment=treatment,           # the GreenCreditTreatment object
    outcome=outcome,               # the ResilienceIndex object
    identification="IV-DiD with staggered lender adoption as instrument"
)
```
The `estimand` field is non-optional and appears on every output object. This is the mechanism through which the framework enforces transparency in downstream publications — if every GCEF output carries a machine-readable estimand, analysts using the package propagate that standard into their reporting automatically.

---

## 5. Pipeline architecture

### 5.1 Stage 1 — IV with staggered DiD (LATE estimation)

**Goal:** Estimate the Local Average Treatment Effect (LATE) of green credit take-up on the resilience index, instrumenting take-up with lender adoption timing.

**Instrument:** Lender adoption timing — when a lender introduced a green credit product. This creates variation in SME *access* to green credit that is plausibly exogenous to individual SME characteristics. The instrument is constructed as a binary indicator of whether the SME's lender had adopted a green product by period t.

**Identification assumption (relevance):** Lender adoption timing is correlated with SME green credit take-up. An SME is more likely to take up green credit if their lender has a product available. *Testable: first-stage F-statistic, threshold F > 10.*

**Identification assumption (exclusion):** Lender adoption timing affects SME resilience *only through* green credit take-up — not through any other channel. This is the critical assumption. Potential violations:
- Lenders that adopt green products may also improve other lending practices simultaneously
- Lender adoption timing may be correlated with regional economic conditions that independently affect SME resilience

**Mitigation:** Include lender fixed effects and time fixed effects. Include region × time interaction terms. Use Callaway & Sant'Anna (2021) group-time ATT estimator for the staggered adoption design to avoid contamination of treatment effect estimates by already-treated units.

**Implementation:** `pyfixest` for the IV regression with fixed effects; `csdid` (or direct implementation) for the Callaway & Sant'Anna staggered DiD.

**Output:** LATE estimate with standard errors, first-stage F-statistic, complier share with confidence interval, and complier covariate profile.

**Complier covariate profile (first-class output, not a footnote):**
The framework computes the full covariate distribution of the complier subpopulation using Abadie (2003) individual-level importance weights (κ weights). The mean estimator alone:

E[X|complier] = (E[X·D|Z=1] − E[X·D|Z=0]) / (P(D=1|Z=1) − P(D=1|Z=0))

gives reweighted means only — it cannot produce medians, IQRs, or complier-to-full-sample ratios. To obtain distributional statistics, the framework computes individual κ weights:

κᵢ = 1 − Dᵢ(1−Zᵢ)/P(Z=0|Xᵢ) − (1−Dᵢ)Zᵢ/P(Z=1|Xᵢ)

where P(Z|X) is the instrument propensity score estimated via logistic regression. Logistic regression is used here rather than gradient boosting because Z is a lender-level binary variable with a limited covariate set — gradient boosting would overfit in this thin-feature context where the instrument propensity has little to model. This is distinct from the `ForestDRIV` nuisance propensity P(D=1|X) in Stage 2, which models individual SME take-up and appropriately uses gradient boosting for its richer covariate space. The two estimators serve different propensity models and the inconsistency is intentional and documented here.

**The profile reports for each covariate:** weighted mean, weighted median, weighted IQR (25th–75th percentile of the weighted CDF), and the complier-to-full-sample ratio (ratio of weighted mean to unweighted mean — values above 1 indicate over-representation in the complier population, below 1 indicate under-representation).

Negative or near-zero κ weights are set to zero and flagged — they indicate observations that are not consistent with complier status. The share of observations with non-positive weights is reported as a diagnostic.

The profile is framed operationally:
> "Green credit access, if expanded, would causally benefit SMEs with the following profile: median firm age 6 years (IQR: 3–11), predominantly manufacturing and agriprocessing sectors (62% of compliers vs. 41% of full sample, complier ratio 1.51), located in regions with above-average historical rainfall variability. This characterises the *marginal SME* — the population for which programme availability determines access."

---

### 5.2 Stage 2 — Causal forest on compliers (CATE estimation)

**Goal:** Estimate heterogeneous treatment effects across the complier subpopulation — specifically, which SMEs benefit most from access to green credit.

**Why compliers only:** The LATE from Stage 1 applies to the complier subpopulation — SMEs who took up green credit because their lender adopted, and would not have taken up otherwise. Running the causal forest on the full population would include always-takers and never-takers, for whom the instrument is irrelevant. CATE estimates restricted to compliers are more policy-relevant: they answer "for SMEs who would respond to a green credit programme, which ones benefit most?"

**Treatment variable:** Predicted take-up from Stage 1 (instrumented take-up), not observed take-up. This removes the endogeneity of self-selection.

**Outcome variable:** Revenue stability during identified climate shocks (primary); resilience index composite (secondary). Revenue stability is computed across all periods via the rolling CV derivation in Section 4.2 — the CV uses the full revenue history, not only shock-period observations. The causal forest then conditions on shock periods by subsetting observations to those where `shock_instrument < shock_threshold` in period t-1. The stability scores are pre-computed; only the observation subsetting is shock-conditional. This preserves sufficient data for the rolling CV while ensuring that the causal inference is conditioned on shock exposure. A reader should not infer that the CV is computed within shock windows only — that interpretation would require a different derivation and would produce noisier estimates in thin shock-event data.

**Covariates:** Firm characteristics (age, size, sector), geographic climate exposure (historical rainfall variability, flood zone classification), prior financial performance.

**Implementation:** `econml.iv.dr.ForestDRIV` (primary target) — doubly robust forest IV, which handles the instrument natively, gains efficiency over manual two-stage residualisation, and is more auditable than passing instrumented treatment into `CausalForestDML` manually. `econml.iv.ForestIV` is the fallback if `ForestDRIV` presents compatibility issues in the implementation phase. `CausalForestDML` with manually instrumented treatment is explicitly not the implementation path — it introduces first-stage residualisation risk and makes the pipeline harder to audit. The specific class is confirmed during implementation, not locked in this specification, but the family (`econml.iv`) is committed.

**Implementing the complier restriction via κ-weighted estimation:**
`ForestDRIV` estimates treatment effects for the full sample by default. The complier restriction is implemented by passing the κ weights computed in Stage 1 as `sample_weight` to `ForestDRIV.fit()`. Always-takers and never-takers receive near-zero weight (κᵢ ≈ 0); compliers receive weight proportional to their κᵢ, driving the estimation. This achieves the restriction without requiring individual complier identification — which the framework correctly holds is impossible.

**Stage 1 → Stage 2 data flow:**
Stage 1 outputs two objects that Stage 2 consumes:
1. `late_result` — the LATE estimate with standard errors (for the results object)
2. `kappa_weights` — the array of individual κᵢ values (passed as `sample_weight` to Stage 2)

The pipeline sequencing is: compute LATE and κ weights in Stage 1 → pass κ weights to `ForestDRIV` in Stage 2 → CATE estimates are κ-weighted and therefore complier-restricted. This data flow should be visible in the `GreenCreditEvaluator.fit()` implementation as two explicitly named intermediate outputs, not embedded silently in the pipeline.

**Nuisance model defaults (documented for cross-analyst consistency):**
`ForestDRIV` is doubly robust — it fits two nuisance models whose defaults must be specified to ensure consistent results across analysts using the package:

| Nuisance model | Default estimator | Rationale |
|---|---|---|
| Propensity score P(D=1\|X) | `sklearn.ensemble.GradientBoostingClassifier` | Handles non-linearity; performs well in thin-data contexts |
| Outcome model E[Y\|X,Z] | `sklearn.ensemble.GradientBoostingRegressor` | Same rationale; consistent with propensity estimator |
| Cross-fitting | k=5 | Standard; avoids overfitting nuisance models on full sample |

Analysts may override these defaults by passing `model_t` and `model_y` arguments to the evaluator. Any override is logged in `results.estimand` and flagged in the report module. Undocumented nuisance model choices undermine reproducibility — this is an infrastructure standard, not an aesthetic preference.

**Output:** CATE estimates per SME, heterogeneity analysis across firm type and geographic exposure, policy-relevant subgroup effects.

---

## 6. Shock instrumentation

**Why shocks matter:** Revenue stability *during shocks* is a better outcome variable than unconditional revenue stability. Without a shock, a green investment may not yet have had time to demonstrate resilience value. Conditioning on shock periods forces the counterfactual question: "Did green credit help this SME weather this specific climate event?"

**Primary shock data sources:**

| Source | Variable | Resolution | Access |
|---|---|---|---|
| CHIRPS | Rainfall anomaly (SPI) | 0.05° (~5.5km), monthly | Free, NASA |
| MODIS | Flood extent | 500m, event-based | Free, NASA Earthdata |
| SPEI Global Drought Monitor | Drought index | 0.5°, monthly | Free |

**Operationalisation:**
- Rainfall anomaly: Standardised Precipitation Index (SPI) below -1.5 standard deviations in period t defines a drought shock
- Flood extent: MODIS-derived flood indicator intersected with SME location (requires geocoded SME data)
- Shock variable is **lagged by one period** to ensure exogeneity to lender behaviour in the same period

**Data requirement:** SME geocoding (latitude/longitude or administrative unit minimum) is required to link climate event data to firm outcomes. This is a binding constraint — if SME location data is unavailable, the causal forest stage cannot run on shock-conditioned outcomes. The framework will flag this and fall back to unconditional revenue as the outcome variable, with an explicit warning.

---

## 7. Data requirements

### Minimum viable dataset

| Column | Type | Required | Notes |
|---|---|---|---|
| `sme_id` | string | Yes | Unique firm identifier |
| `lender_id` | string | Yes | Links to lender adoption timing |
| `period` | int/date | Yes | Time period of observation |
| `green_credit_takeup` | binary | Yes | Did SME take up green credit in period t |
| `lender_green_adoption_period` | int/date | Yes | When did lender introduce green product |
| `revenue` | float | Yes (primary outcome) | Revenue in period t. `revenue_stability` is derived internally via rolling CV (window = `stability_window`, default 4 periods). Do not supply `revenue_stability` unless using a custom derivation. |
| `loan_repayment_rate` | float | Recommended | Repayment performance |
| `employment` | int | Recommended | Number of employees. `employment_stability` derived internally via rolling CV. |
| `sme_latitude` | float | Required for shock | For CHIRPS/MODIS linkage |
| `sme_longitude` | float | Required for shock | For CHIRPS/MODIS linkage |
| `firm_age` | int | Recommended | Years since establishment |
| `sector` | categorical | Recommended | ISIC/SIC sector code |
| `firm_size` | categorical | Recommended | Micro/small/medium |
| `prior_revenue` | float | Recommended | Mean revenue in pre-treatment periods; used as a covariate in Stage 2 to control for pre-existing revenue levels |

### Treatment specification columns (mapped from GreenCreditTreatment object)

| Column | Type | Required |
|---|---|---|
| `treatment_type` | categorical | Yes if using treatment vector |
| `conditionality_mechanism` | categorical | Recommended |
| `verification_method` | categorical | Recommended |
| `treatment_intensity` | float | Recommended |

---

## 8. Identification assumptions — full list

| # | Assumption | Stage | Testable | Test |
|---|---|---|---|---|
| A1 | Instrument relevance: lender adoption timing predicts SME take-up | Stage 1 | Yes | First-stage F > 10 |
| A2 | Exclusion restriction: adoption timing affects resilience only through take-up | Stage 1 | Partial | Lender FE, time FE, over-ID test if multiple instruments |
| A3 | Monotonicity: no defiers (no SME takes up *because* lender did NOT adopt) | Stage 1 | Partial | Theoretical argument + empirical check (see below) |
| A4 | Shock exogeneity: climate events are orthogonal to lender behaviour | Stage 2 | Partial | Test lender disbursement behaviour in shock periods |
| A5 | Shock lag: lagged shocks do not predict current lender adoption | Stage 2 | Yes | Regression test |
| A6 | Overlap: sufficient treated and untreated compliers across covariate space | Stage 2 | Yes | Propensity score distribution check |
| A7 | Composite index weights are pre-specified, not data-driven | Outcome | By design | Weights documented before data ingestion |

**Monotonicity argument (A3):**
Defiers would be SMEs that take up green credit specifically *because* their lender has not adopted a green product — a behavioural mechanism in which product unavailability drives demand. No plausible account of this mechanism exists in SSA lending markets. Green credit take-up requires that a lender offers a green product; lender adoption is a necessary condition for SME access, not an inverse cause of it.

The monotonicity assumption is also partially verifiable from the data: if any SME in the dataset took up green credit before their lender's recorded adoption date, this is evidence of a data quality problem or an adoption timing mismatch — not a defier. The framework checks for this directly and flags cases where SME take-up precedes lender adoption as `ADOPTION_TIMING_ANOMALY`, which triggers a data quality warning rather than a monotonicity failure (since the more likely explanation is measurement error in adoption timing).

---

## 9. Limitations and failure modes

**L1 — Geocoding requirement.**
CHIRPS/MODIS shock linkage requires SME latitude/longitude. In many SSA lender datasets, location data is at administrative unit level only (district, region) or missing entirely. If geocoding is unavailable at sub-district level, shock resolution degrades significantly. The framework falls back to administrative unit centroid with an explicit warning and a confidence interval expansion.

**L2 — Thin first stage.**
In smaller datasets or contexts where lender adoption is not sufficiently staggered (e.g., all lenders adopted in the same year), the instrument is weak. Weak instrument produces biased LATE estimates. The framework checks F-statistic and raises an error if F < 10 and a warning if F < 20.

**L3 — Complier subpopulation may be small.**
In contexts where most SMEs are always-takers (would take green credit regardless of lender adoption) or never-takers (would not take up regardless), the complier population is small and CATE estimates are noisy. The framework reports complier share and flags if < 20% of the sample.

**L4 — Treatment heterogeneity within type.**
A `rate_reduction` treatment at 1% intensity is different from one at 5% intensity. The framework handles intensity as a continuous moderator but does not automatically interact treatment type with intensity. Analysts should specify this interaction manually in the covariate set.

**L5 — Data from a single lender.**
The instrument requires variation in adoption timing *across* lenders. If the dataset covers only one lender, there is no cross-lender variation and the staggered DiD instrument collapses. A within-lender instrument (e.g., branch-level rollout) is required in this case. The framework raises an error if `lender_id` has only one unique value.

**L6 — Self-reported green use.**
When `verification_method = "self_reported"`, treatment assignment may be endogenous even conditional on lender adoption — firms may misreport green use of funds. The framework flags this and recommends sensitivity analysis with a conservative treatment definition.

**L7 — Short pre-treatment panels.**
The Callaway & Sant'Anna (2021) estimator requires pre-treatment periods to test parallel trends — the identifying assumption for the DiD component. In SSA lender portfolios, panel data is commonly 3–5 years. With staggered adoption, late-adopting lender cohorts may have only 1–2 pre-treatment periods, which is insufficient for a credible parallel trends test. Early-adopting cohorts have more pre-treatment data but constitute a decreasing share of the not-yet-treated comparison group over time.

This interacts with L2 (thin first stage): in short panels a dataset may simultaneously face a weak instrument problem and an untestable parallel trends assumption. The framework checks minimum pre-treatment period count per cohort and raises `ShortPanelWarning` if any cohort has fewer than 3 pre-treatment periods; raises an error if fewer than 2. Analysts working with short panels should consider: (a) aggregating to longer periods if the data permits, (b) Rambachan & Roth (2023) sensitivity analysis for parallel trends, or (c) restricting to cohorts with sufficient pre-treatment history with explicit documentation. `panel_length_check` is added to `assumptions_tested`.

**L8 — κ-weighted ForestDRIV confidence interval coverage.**
`ForestDRIV` confidence intervals computed with `sample_weight` are approximate. The variance estimation in `econml`'s forest implementation uses the κ weights for point estimation, but the honest confidence interval construction (based on the infinitesimal jackknife) treats the weights as fixed rather than estimated. In practice this means reported CIs may undercover slightly when weights are highly heterogeneous — which they will be when complier share is low (below 30%). Analysts using GCEF in publications should note this caveat and consider reporting sensitivity bounds alongside the CIs, particularly in contexts where complier share triggers a `ComplierShareWarning`. This is a known limitation of the current `econml` implementation, not a flaw in the identification strategy. It does not change the implementation.

---

## 10. Version roadmap

Items deferred from v0.1 are collected here. They are referenced throughout the specification at their points of deferral.

**v0.2 — Lee (2009) bounds for `cate_bounds`**
Lee bounds are tighter than Manski (1990) bounds but require a monotone treatment selection assumption (A8: treatment take-up is monotone in potential outcomes). A8 is a separate and additional assumption from the IV monotonicity in A3. Adding A8 requires its own theoretical argument and empirical check. Deferred to v0.2 to avoid overloading the v0.1 assumption set.

**v0.2 — Blended treatment types**
`TreatmentType.BLENDED` raises `BlendedTreatmentNotImplementedError` in v0.1. Implementing blended treatment DAGs requires specifying how pathway-specific effects are partially identified when multiple treatment mechanisms are bundled in one instrument. This is a significant scope expansion. Deferred to v0.2. Contribution welcome via the GitHub repository.

**v0.2 — `rpy2` elimination**
The Callaway & Sant'Anna (2021) staggered DiD estimator currently relies on optional `rpy2` interop with the R `did` package. A pure Python implementation (`csdid`) is preferred for dependency clarity. If `csdid` reaches sufficient maturity between v0.1 and v0.2, `rpy2` is dropped as a dependency.

**v0.3 — MODIS flood extent integration**
v0.1 implements CHIRPS rainfall anomaly as the shock instrument. MODIS flood extent integration requires additional spatial processing (event-based rather than continuous, event boundary detection) and is deferred to v0.3. Analysts with flood-exposed portfolios should flag this as a limitation in v0.1 publications.

**v0.3 — Multi-outcome CATE**
v0.1 runs the causal forest on revenue stability as the primary outcome. Secondary analyses on other resilience components (loan repayment rate, employment stability) are described but not implemented as a formal multi-outcome pipeline. v0.3 will implement a structured multi-outcome CATE pipeline with multiple testing corrections.

---

## 11. Report module

The report module produces the output a DFI analyst actually receives — not Python objects, but a structured document. It is the interface between the framework's methodology and the decisions made by people who will not read the code.

**Output format:** PDF (primary) and HTML (secondary). The PDF is designed for inclusion in programme evaluation reports and funding applications. The HTML is for interactive review and web publishing.

**Report structure:**

| Section | Content | Audience |
|---|---|---|
| Executive summary | LATE estimate in plain language; complier profile summary; top 3 CATE findings; key caveats | Non-technical decision-maker |
| Estimand statement | Full `estimand` object rendered in prose; explicit statement of who the estimates apply to and who they do not | All |
| Complier profile | Covariate table with weighted mean, median, IQR, complier ratio; operational framing as marginal SME characterisation | DFI analyst |
| LATE estimate | Point estimate, confidence interval, interpretation; first-stage F-statistic prominently displayed | DFI analyst / researcher |
| CATE heterogeneity | Top covariate dimensions driving treatment effect heterogeneity; policy subgroup estimates; visualisation of CATE distribution | DFI analyst |
| Bounds for full portfolio | Manski bounds for always-takers and never-takers; explicit labelling as bounds not estimates | DFI analyst |
| Assumption audit | `assumptions_tested` rendered as a checklist — green/amber/red per assumption; plain-language explanation of any failures | All |
| Methodological notes | Condensed methodology; key citations; link to full specification on GitHub | Researcher / reviewer |
| Reproducibility statement | Data hash, random seed, GCEF version, nuisance model overrides (if any) | Researcher / reviewer |

**Design principles for the report module:**
- Every estimate carries its estimand. No number appears without an explicit statement of what population it applies to.
- Assumption failures are prominent, not buried. A red flag in the assumption audit appears in the executive summary, not only in the technical appendix.
- The reproducibility statement is mandatory and non-suppressible. It cannot be removed by the analyst.
- The report does not include raw data or individual SME identifiers.

**Generation:** `results.report(output_format="pdf", output_path="./gcef_report.pdf")`. The report module uses the `reportlab` library for PDF generation and Jinja2 for HTML templating.

---

## 12. Design decisions and rationale

**DD1 — Why not use the composite index as the causal forest outcome?**
The composite index is a weighted average of multiple outcomes. Causal forest estimates CATE by finding covariate subgroups where treatment effects differ. Running it on a composite index obscures *which component* is driving heterogeneity — a firm that benefits on employment but not revenue looks the same as one that benefits on revenue but not employment. The primary causal forest outcome is revenue stability during shocks; secondary analyses on other components are run separately.

**DD2 — Why Callaway & Sant'Anna (2021) rather than two-way fixed effects?**
Standard two-way fixed effects DiD with staggered adoption is biased when treatment effects are heterogeneous across cohorts — and they are in this context, because early adopters (lenders who introduced green products first) likely differ from late adopters. C&S avoids contaminating treatment effect estimates by using only not-yet-treated units as the comparison group. This is especially important in SSA where lender adoption was not random across time.

**DD3 — Why restrict causal forest to compliers?**
The LATE from Stage 1 is defined for compliers — SMEs who respond to the instrument (lender adoption). Running an unrestricted causal forest on the full sample would estimate CATE for always-takers and never-takers, for whom the instrument provides no variation. The complier restriction ensures that the HTE estimates are relevant to the population that would actually respond to a green credit expansion policy.

DD3 is not a choice. It is a consequence of the identification strategy. The tension it creates is not methodological — it is a communication design problem, and it is resolved in DD3a below, not by weakening DD3.

**DD3a — Complier characterisation is a primary output, not a caveat.**
Because CATE estimates are restricted to the complier subpopulation, the framework produces a complier covariate profile as a standard first-class output using the Abadie (2003) IV-based covariate mean estimator:

E[X|complier] = (E[X·D|Z=1] − E[X·D|Z=0]) / (P(D=1|Z=1) − P(D=1|Z=0))

This gives the distribution of firm characteristics within the complier subpopulation without requiring individual-level complier identification. The profile reports distributional statistics — mean, median, IQR, complier-to-full-sample ratio — not just means, because means alone are insufficient for targeting.

All CATE outputs carry a non-optional `estimand` object framing compliers as the *marginal SME* population for additionality assessment. This connects the identification concept to existing DFI vocabulary: compliers are the additional SMEs — the ones whose behaviour the programme changes. Complier share is reported with a confidence interval and framed as a substantive finding about market structure, not a data quality flag.

For full-portfolio questions, the framework provides `cate_bounds` — Manski-style partial identification bounds for always-takers and never-takers — rather than point estimates with inflated confidence intervals. Wider CIs on unidentified subpopulations imply a level of precision the identification strategy does not support. The distinction — bounds for unidentified subpopulations, estimates for identified ones — is enforced by the API design and cannot be bypassed.

**DD4 — Why lag the shock variable?**
A contemporaneous climate shock (period t) may affect both SME revenue (the outcome) and lender behaviour (portfolio quality deteriorates, lender tightens lending) in the same period. This creates a pathway from the shock to the instrument (lender adoption patterns) that violates the exclusion restriction for the shock as an exogenous event. Lagging by one period (shock in t-1 affects revenue measured in t) preserves the exogeneity of the shock relative to lender behaviour in t.

---

## 13. Implementation dependencies

| Package | Version | Purpose |
|---|---|---|
| `econml` | ≥ 0.15 | `iv.dr.ForestDRIV` (primary), `iv.ForestIV` (fallback) |
| `pyfixest` | ≥ 0.18 | IV regression with fixed effects |
| `pandas` | ≥ 2.0 | Data handling |
| `numpy` | ≥ 1.24 | Numerical operations |
| `scikit-learn` | ≥ 1.3 | Nuisance models (GradientBoosting); base ML components |
| `statsmodels` | ≥ 0.14 | Diagnostic tests |
| `geopandas` | ≥ 0.14 | Geospatial operations for shock linkage |
| `rasterio` | ≥ 1.3 | CHIRPS/MODIS raster data ingestion |
| `requests` / `chirps` | latest | CHIRPS API access |
| `matplotlib` / `seaborn` | latest | CATE visualisations |
| `reportlab` | ≥ 4.0 | PDF report generation |
| `jinja2` | ≥ 3.0 | HTML report templating |

**Optional (R interop for Callaway & Sant'Anna):**
| Package | Purpose |
|---|---|
| `rpy2` | R interoperability |
| `did` (R) | Callaway & Sant'Anna (2021) estimator |

Note: A pure Python implementation of the C&S estimator is preferred for dependency clarity. If `csdid` matures sufficiently, `rpy2` dependency is dropped.

---

## 14. Versioning and open research commitment

This specification is version-controlled on GitHub. All changes to design decisions, identification assumptions, and weighting schemes are documented with rationale. The package and this specification are released under MIT licence.

All published analyses using GCEF should cite:
- This specification document (DOI forthcoming)
- Abadie, A. (2003). "Semiparametric instrumental variable estimation of treatment response models." *Journal of Econometrics*, 113(2), 231–263. — Complier covariate characterisation estimator
- Callaway, B. & Sant'Anna, P.H.C. (2021). "Difference-in-differences with multiple time periods." *Journal of Econometrics*, 225(2), 200–230. — Staggered DiD estimator
- Wager, S. & Athey, S. (2018). "Estimation and inference of heterogeneous treatment effects using random forests." *Journal of the American Statistical Association*, 113(523), 1228–1242. — Causal forest estimator
- Manski, C.F. (1990). "Nonparametric Bounds on Treatment Effects." *American Economic Review*, 80(2), 319–323. — Partial identification bounds for unidentified subpopulations
- Lee, D.S. (2009). "Training, Wages, and Sample Selection: Estimating Sharp Bounds on Treatment Effects." *Review of Economic Studies*, 76(3), 1071–1102. — Tighter bounds exploiting monotonicity for selection problems. **Deferred to v0.2 of the package** — requires A8 (monotone treatment selection), a separate and additional assumption not included in v0.1.
- Rambachan, A. & Roth, J. (2023). "A More Credible Approach to Parallel Trends." *Review of Economic Studies*, 90(5), 2555–2591. — Parallel trends sensitivity analysis for short panels.
- CHIRPS: Funk, C. et al. (2015). "The climate hazards infrared precipitation with stations — a new environmental record for monitoring extremes." *Scientific Data*, 2, 150066.

---

*GCEF v0.7 — Working specification. Not yet peer reviewed. Feedback and critique actively solicited.*
*Contact: priya@pemigloans.co.ke*

