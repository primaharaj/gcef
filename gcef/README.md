# GCEF — Green Credit Causal Evaluation Framework

**Causal impact evaluation of green credit programmes on SME resilience in Sub-Saharan Africa.**

Most published impact evaluations of green credit programmes in SSA are correlational. They show that SMEs who received green credit improved — but they do not separate genuine causal impact from selection effects. SMEs that seek and access green credit are systematically better managed, more climate-aware, and more likely to improve regardless of the loan. Without controlling for this selection, impact reports overstate causal effects and misdirect capital.

GCEF provides a rigorous, reproducible methodology for separating selection from causation. It is designed for DFI analysts evaluating green credit portfolios, researchers conducting academic impact evaluation, and lenders building internal impact measurement capacity.

---

## Quick start

```python
pip install gcef
```

```python
from gcef import GreenCreditEvaluator, GreenCreditTreatment, ResilienceIndex
from gcef.treatment import TreatmentType, ConditionalityMechanism, VerificationMethod
from gcef.report import hash_dataframe

# 1. Define the treatment instrument
treatment = GreenCreditTreatment(
    type=TreatmentType.RATE_REDUCTION,
    conditionality_mechanism=ConditionalityMechanism.VERIFIED_INVESTMENT,
    verification_method=VerificationMethod.DOCUMENT_REVIEW,
    intensity=0.03,  # 3% rate reduction
)

# 2. Define the outcome index
outcome = ResilienceIndex(
    columns={
        "revenue":              0.40,
        "loan_repayment_rate":  0.30,
        "employment":           0.20,
        "adaptation_investment": 0.10,
    },
    shock_instrument="rainfall_anomaly_lag1",
    shock_threshold=-1.5,
)

# 3. Hash your data before fitting (for the reproducibility statement)
data_hash = hash_dataframe(data)

# 4. Run the pipeline
evaluator = GreenCreditEvaluator(
    treatment=treatment,
    outcome=outcome,
    unit_id="sme_id",
    time_id="period",
    lender_id="lender_id",
    adoption_time="lender_green_adoption_period",
    covariates=["firm_age", "sector", "firm_size", "prior_revenue"],
    random_seed=42,
)

results = evaluator.fit(data)

# 5. Inspect results
print(results.late)
# {'estimate': 1578.4, 'se': 56.8, 'ci_lower': 1420.7, 'ci_upper': 1736.0}

print(results.estimand.to_prose())
# These estimates apply to the complier subpopulation — SMEs who accessed
# green credit because their lender offered it, and who would not have
# accessed it otherwise. In DFI terms: Marginal SMEs...

# 6. Generate a report
results.report(output_format="pdf", output_path="./gcef_report.pdf")
```

---

## What the framework produces

GCEF runs a two-stage causal pipeline and produces four primary outputs.

**Local Average Treatment Effect (LATE).** The causal effect of green credit access on the resilience index for the complier subpopulation, identified by variation in lender adoption timing. Estimated with clustered standard errors and a first-stage F-statistic.

**Conditional Average Treatment Effects (CATE).** Heterogeneous treatment effects across the complier subpopulation — which SME types benefit most. Estimated using a kappa-weighted causal forest (`ForestDRIV`), restricted to compliers without requiring individual complier identification.

**Complier profile.** A characterisation of the marginal SME population — the subgroup whose green credit behaviour changes with programme availability. Reported as a covariate distribution table with complier-to-full-sample ratios. This is the primary output for DFI targeting decisions.

**Manski bounds.** Partial identification bounds for always-takers and never-takers — the subpopulations for which the instrument provides no identifying variation. Honest worst-case bounds rather than point estimates with inflated confidence intervals.

---

## The identification strategy

GCEF exploits variation in **lender adoption timing** — the fact that different lenders introduced green credit products at different points in time — as an instrument for SME take-up.

```
Instrument (Z):   Lender adopted green product by period t
                  ↓ (relevance: SMEs take up when product is available)
Treatment (D):    SME took up green credit
                  ↓ (exclusion: adoption timing affects resilience only through take-up)
Outcome (Y):      SME resilience index during climate shocks
```

This variation is plausibly exogenous to individual SME characteristics — whether a lender adopted in Q3 2019 versus Q1 2020 is driven by DFI capital availability and institutional capacity, not by the characteristics of any individual SME.

**Stage 1** uses the Callaway & Sant'Anna (2021) staggered DiD estimator to recover the LATE for compliers, and computes Abadie (2003) kappa weights to characterise the complier subpopulation without requiring individual identification.

**Stage 2** uses `econml`'s `ForestDRIV` with kappa-weighted sample weights to estimate CATE across the complier subpopulation, conditioned on shock periods identified by lagged CHIRPS rainfall anomaly data.

---

## Why compliers, not the full sample

The LATE from Stage 1 is defined for **compliers** — SMEs who took up green credit *because* their lender adopted, and who would not have taken up otherwise. This is the marginal population: the SMEs whose behaviour the programme actually changes.

Running an unrestricted CATE analysis on the full sample would estimate effects for always-takers (who would take up regardless) and never-takers (who would not take up regardless), for whom the instrument provides no identifying variation. Those estimates would be extrapolation without identification.

GCEF calls this the **marginal SME** framing, connecting the methodological concept to DFI vocabulary around additionality. The complier profile characterises this population operationally, without requiring knowledge of which specific SMEs are compliers.

---

## What your data needs

```python
# Minimum required columns
data.columns = [
    "sme_id",                        # unique firm identifier
    "lender_id",                     # links to lender adoption timing
    "period",                        # time period (int or date)
    "green_credit_takeup",           # binary: did SME take up in period t
    "lender_green_adoption_period",  # when did lender introduce green product
    "revenue",                       # firm revenue in period t
]

# Recommended for a complete analysis
# "loan_repayment_rate", "employment", "adaptation_investment"
# "sme_latitude", "sme_longitude"    # for CHIRPS shock linkage
# "firm_age", "sector", "firm_size", "prior_revenue"  # CATE covariates
```

**Multiple lenders required.** The instrument requires variation in adoption timing *across* lenders. Single-lender datasets will raise `SingleLenderError`. If your data covers one lender, a within-lender instrument (e.g. branch-level rollout) is required — contact us on GitHub.

**Minimum panel length.** Each adoption cohort needs at least 3 pre-treatment periods for the Callaway & Sant'Anna parallel trends test. GCEF raises `ShortPanelWarning` at 2 periods and `ShortPanelError` at 1.

**Revenue stability is derived internally.** Do not pre-compute `revenue_stability`. GCEF derives it from `revenue` as a rolling coefficient of variation (window=4 periods by default), ensuring consistent computation across analyses.

---

## Shock instrumentation

GCEF conditions on climate shocks to sharpen the outcome measure. Revenue stability *during shocks* is a better causal outcome than unconditional revenue stability — it forces the counterfactual question: did green credit help this SME weather this specific climate event?

**If you have geocoded SME data** (lat/lon), attach CHIRPS rainfall anomaly data:

```python
from gcef.shock import attach_chirps_anomaly

# Map your panel periods to calendar months
period_to_yearmonth = {
    1: (2019, 3), 2: (2019, 6), 3: (2019, 9), 4: (2019, 12),
    5: (2020, 3), 6: (2020, 6), 7: (2020, 9), 8: (2020, 12),
}

data = attach_chirps_anomaly(
    data,
    period_to_yearmonth=period_to_yearmonth,
    lag=1,   # spec DD4: lag to preserve shock exogeneity
)
# Adds "rainfall_anomaly_lag1" column (SPI-3 score, lagged 1 period)
```

**If you don't have geocoded data** or want to test without CHIRPS access:

```python
from gcef.shock import make_synthetic_shock_instrument

data = make_synthetic_shock_instrument(data, shock_probability=0.15)
# Adds a realistic synthetic shock column for development and testing
```

The shock variable is lagged by one period because contemporaneous shocks affect both SME revenue (the outcome) and lender disbursement behaviour (a potential exclusion restriction violation). Lagged shocks are orthogonal to lender behaviour in period t.

---

## Interpreting the output

```python
results = evaluator.fit(data)

# ── LATE ──────────────────────────────────────────────────────────────────────
results.late
# {'estimate': 1578.4, 'se': 56.8, 'ci_lower': 1420.7, 'ci_upper': 1736.0}
# Causal effect of green credit access on the resilience index.
# Applies to compliers only. First-stage F = 206 (strong instrument).

# ── Estimand — every number carries its population ───────────────────────────
results.estimand.population          # "compliers"
results.estimand.dfi_framing         # "Marginal SMEs — those whose green credit
                                     #  behaviour changes with programme availability..."
results.estimand.to_prose()          # full prose paragraph for reports

# ── Complier profile — who are the marginal SMEs? ────────────────────────────
results.complier_share
# {'estimate': 0.49, 'ci_lower': 0.43, 'ci_upper': 0.55}

results.complier_profile
#              covariate  weighted_mean  full_mean  complier_to_full_ratio
# 0             firm_age          12.4       12.3                    1.00
# 1  sector=construction           0.09       0.09                    1.03
# ...

# ── CATE — which SMEs benefit most? ──────────────────────────────────────────
complier_cate = results.cate[results.cate_complier_mask.values]
complier_cate["cate_estimate"].describe()
# mean    1605.3    # average CATE among compliers
# std      191.1
# 25%     1420.2    # some SMEs benefit much more than others
# 75%     1746.7

# ── Bounds — honest limits for non-identified subpopulations ─────────────────
results.cate_bounds[results.cate_bounds["population"] == "never_taker"].head()
# lower_bound and upper_bound are Manski (1990) worst-case bounds,
# not estimates with wide CIs. The width equals the outcome support.

# ── Assumption audit ──────────────────────────────────────────────────────────
results.assumptions_tested["instrument_relevance"]
# {'passed': True, 'value': 205.8, 'threshold': 10, 'test': 'first_stage_F'}

results.assumptions_tested["overlap"]
# {'passed': False, 'value': 0.35, ...}
# If any assumption fails, it appears in the executive summary of the report.
```

---

## Generating reports

```python
from gcef.report import ReportGenerator, hash_dataframe

# Hash the data before fitting for the reproducibility statement
data_hash = hash_dataframe(data)
results = evaluator.fit(data)

reporter = ReportGenerator(
    results,
    programme_name="AFDB Solar SME Facility — Kenya 2019-2021",
    analyst_name="Priya Maharaj",
    data_description="FSD Africa SME panel, Kenya 2019-2021",
    data_hash=data_hash,
)

reporter.to_pdf("gcef_report.pdf")    # 9-section structured PDF
reporter.to_html("gcef_report.html")  # HTML equivalent

# Or via the shortcut on the results object
results.report(output_format="pdf", output_path="gcef_report.pdf")
```

The report contains nine sections: executive summary, estimand statement, complier profile, LATE estimate, CATE heterogeneity, Manski bounds for the full portfolio, assumption audit (green/amber/red checklist), methodological notes, and a reproducibility statement. The reproducibility statement is mandatory and non-suppressible — it cannot be removed from the output.

---

## Identification assumptions

GCEF tests every assumption it can and documents those it cannot.

| # | Assumption | Testable | Test |
|---|---|---|---|
| A1 | Instrument relevance: lender adoption predicts SME take-up | Yes | First-stage F > 10 |
| A2 | Exclusion restriction: adoption timing affects resilience only through take-up | Partial | Lender + time FE, over-ID test |
| A3 | Monotonicity: no SME takes up *because* their lender did not adopt | Partial | Empirical check + theoretical argument |
| A4 | Shock exogeneity: climate events are orthogonal to lender behaviour | Partial | Test disbursement in shock periods |
| A5 | Shock lag: lagged shocks do not predict current lender adoption | Yes | Regression test |
| A6 | Overlap: sufficient treated and untreated compliers across covariates | Yes | Propensity score distribution |
| A7 | Index weights are pre-specified, not data-driven | By design | Documented before data ingestion |

The framework will not silently proceed when assumptions are likely violated. `WeakInstrumentError` halts the pipeline if F < 10. `SingleLenderError` halts if there is only one lender. Warnings fire at intermediate thresholds. Every assumption result is available in `results.assumptions_tested` with a structured schema.

---

## Known limitations

**L1 — Geocoding.** CHIRPS shock linkage requires SME latitude/longitude. If only administrative unit data is available, GCEF falls back to centroid with `GeocodeResolutionWarning` and degraded resolution (~50km vs ~5.5km).

**L2 — Thin first stage.** In small datasets or when all lenders adopted in the same period, the instrument is weak. GCEF warns at F < 20 and errors at F < 10.

**L3 — Small complier share.** If most SMEs are always-takers or never-takers, the complier population is small and CATE estimates are noisy. GCEF flags complier share below 20%.

**L4 — ForestDRIV confidence intervals are approximate.** The infinitesimal jackknife variance estimator in `econml` treats kappa sample weights as fixed. CIs may undercover slightly when weights are highly heterogeneous (common with low complier share). This is documented in every CATE output.

**L5 — Single lender.** The staggered DiD instrument requires cross-lender variation. Single-lender datasets raise `SingleLenderError`.

**L6 — Self-reported treatment.** When `verification_method=SELF_REPORTED`, treatment assignment may be endogenous. GCEF flags this and recommends sensitivity analysis.

**L7 — Short panels.** The Callaway & Sant'Anna estimator requires pre-treatment periods for parallel trends testing. GCEF warns at < 3 pre-treatment periods per cohort and errors at < 2.

---

## What is not yet implemented (v0.2)

- **Blended treatment types** — instruments combining multiple conditionality mechanisms require multi-pathway DAG specification.
- **Lee (2009) tighter bounds** — requires monotone treatment selection, an additional assumption beyond IV monotonicity.
- **MODIS flood extent** — the shock module implements CHIRPS; MODIS flood data integration is planned.
- **Native Callaway & Sant'Anna implementation** — currently uses `pyfixest`'s built-in; a standalone pure-Python implementation is planned.

---

## Citation

If you use GCEF in published work, please cite:

```
Maharaj, P. (2026). Green Credit Causal Evaluation Framework (GCEF), version 0.1.0.
https://github.com/priya-maharaj/gcef. Specification DOI: forthcoming.
```

And the underlying methodological work:

- Abadie, A. (2003). Semiparametric instrumental variable estimation of treatment response models. *Journal of Econometrics*, 113(2), 231–263.
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200–230.
- Manski, C. F. (1990). Nonparametric bounds on treatment effects. *American Economic Review Papers and Proceedings*, 80(2), 319–323.
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523), 1228–1242.

---

## Development

```bash
git clone https://github.com/priya-maharaj/gcef
cd gcef
pip install -e ".[dev]"
pytest tests/
```

311 tests. All modules implemented. Zero stubs remaining.

```
gcef/
├── treatment.py      # GreenCreditTreatment, TreatmentType enums
├── outcomes.py       # ResilienceIndex, rolling CV derivation
├── estimand.py       # Estimand dataclass — attached to every output
├── exceptions.py     # All custom warnings and errors
├── assumptions.py    # Identification assumption tests
├── stage1.py         # IV/DiD, kappa weights, complier profile
├── stage2.py         # ForestDRIV with kappa-weighted sample weights
├── bounds.py         # Manski (1990) partial identification bounds
├── pipeline.py       # GreenCreditEvaluator — orchestrates the full pipeline
├── shock.py          # CHIRPS ingestion, SPI computation, spatial join
├── report.py         # PDF and HTML report generation
└── testing/
    └── synthetic.py  # Synthetic panel data generator for testing
```

---

## Licence

MIT. See [LICENCE](LICENCE).

Feedback and critique actively solicited — open an issue or email priya@pemigloans.co.ke.
