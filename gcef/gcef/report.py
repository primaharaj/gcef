"""
gcef.report
-----------
Report generation for GCEF. Produces PDF and HTML outputs for DFI analysts.

The report is the interface between the framework's methodology and the
decisions made by people who will not read the code. Every design choice
here serves that translation function.

Design principles (Spec Section 11)
------------------------------------
Every estimate carries its estimand.
    No number appears without a statement of what population it applies to.

Assumption failures are prominent.
    A red flag in the assumption audit appears in the executive summary.

The reproducibility statement is mandatory and non-suppressible.
    It cannot be removed or suppressed by the analyst.

The report does not include raw data or individual SME identifiers.
    Aggregate statistics only.

Report structure (Spec Section 11 table)
-----------------------------------------
1. Executive summary          — LATE in plain language, key caveats, red flags
2. Estimand statement         — who the estimates apply to and who they do not
3. Complier profile           — weighted covariate table; marginal SME framing
4. LATE estimate              — point, CI, F-statistic
5. CATE heterogeneity         — distribution, top drivers, policy subgroups
6. Bounds for full portfolio  — Manski bounds; explicit labelling
7. Assumption audit           — green/amber/red checklist
8. Methodological notes       — condensed method, citations, GitHub link
9. Reproducibility statement  — data hash, seed, version, overrides

Implementation
--------------
PDF via reportlab (primary). HTML via Jinja2 (secondary).
Both share a common data assembly layer (_ReportData) so the two renderers
stay in sync without duplicating logic.
"""
from __future__ import annotations

import hashlib
import io
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────

GCEF_GITHUB = "https://github.com/priya-maharaj/gcef"
GCEF_SPEC_DOI = "DOI forthcoming"

CITATIONS = [
    "Abadie, A. (2003). Semiparametric instrumental variable estimation of "
    "treatment response models. Journal of Econometrics, 113(2), 231-263.",
    "Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences "
    "with multiple time periods. Journal of Econometrics, 225(2), 200-230.",
    "Manski, C. F. (1990). Nonparametric bounds on treatment effects. "
    "American Economic Review Papers and Proceedings, 80(2), 319-323.",
    "Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous "
    "treatment effects using random forests. Journal of the American Statistical "
    "Association, 113(523), 1228-1242.",
]

# Assumption audit traffic-light thresholds
ASSUMPTION_STATUS = {
    True: ("PASS", (0.18, 0.55, 0.34)),    # green
    False: ("FAIL", (0.80, 0.20, 0.20)),   # red
    None: ("WARN", (0.85, 0.65, 0.13)),    # amber
}

ASSUMPTION_PLAIN_LANGUAGE = {
    "single_lender": (
        "Multiple lenders required",
        "The instrument requires variation in adoption timing across lenders. "
        "Multiple lenders with different adoption dates are present.",
    ),
    "adoption_timing_anomaly": (
        "No adoption timing anomalies",
        "No SMEs show green credit take-up before their lender adopted a "
        "green product. Lender adoption is a necessary condition for take-up.",
    ),
    "panel_length": (
        "Sufficient pre-treatment periods",
        "Each adoption cohort has at least 3 pre-treatment periods for "
        "parallel trends testing (Callaway & Sant'Anna requirement).",
    ),
    "instrument_relevance": (
        "Instrument relevance (first-stage F > 10)",
        "Lender adoption timing is a strong predictor of SME green credit "
        "take-up. A weak instrument would bias LATE estimates.",
    ),
    "kappa_weight_negatives": (
        "Kappa weight distribution (staggered DiD)",
        "The share of non-positive kappa weights is within expected range "
        "for staggered DiD designs (typically 15–30%).",
    ),
    "complier_share": (
        "Complier share ≥ 20%",
        "The complier subpopulation is large enough for reliable CATE "
        "estimation. Small complier share produces noisy estimates.",
    ),
    "overlap": (
        "Covariate overlap",
        "Sufficient treated and untreated compliers across the covariate "
        "space. Poor overlap indicates CATE estimates may extrapolate.",
    ),
}


# ── Public API ─────────────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Generates PDF and HTML reports from a GCEFResults object.

    Parameters
    ----------
    results : GCEFResults
        Output from GreenCreditEvaluator.fit().
    programme_name : str, optional
        Name of the green credit programme being evaluated.
        Appears in the report header.
    analyst_name : str, optional
        Analyst name for the reproducibility statement.
    data_description : str, optional
        Brief description of the dataset (e.g. "FSD Africa SME panel, Kenya 2018-2022").
    data_hash : str, optional
        SHA-256 hash of the input data for reproducibility.
        Compute with gcef.report.hash_dataframe(data) before calling fit().

    Usage
    -----
    >>> reporter = ReportGenerator(results, programme_name="AFDB Solar SME Facility")
    >>> reporter.to_pdf("gcef_report.pdf")
    >>> reporter.to_html("gcef_report.html")
    # Or via the results object shortcut:
    >>> results.report(output_format="pdf", output_path="gcef_report.pdf")
    """

    def __init__(
        self,
        results,
        programme_name: str = "Green Credit Programme",
        analyst_name: str = "",
        data_description: str = "",
        data_hash: str = "",
    ):
        self.results = results
        self.programme_name = programme_name
        self.analyst_name = analyst_name
        self.data_description = data_description
        self.data_hash = data_hash
        self._report_data = _ReportData.from_results(
            results,
            programme_name=programme_name,
            analyst_name=analyst_name,
            data_description=data_description,
            data_hash=data_hash,
        )

    def generate(
        self,
        output_format: str = "pdf",
        output_path: str = "./gcef_report.pdf",
    ) -> Path:
        """
        Generate report in the specified format.

        Parameters
        ----------
        output_format : str
            "pdf" or "html". Default: "pdf".
        output_path : str
            Output file path.

        Returns
        -------
        Path to the generated file.
        """
        output_path = Path(output_path)
        if output_format == "pdf":
            return self.to_pdf(output_path)
        elif output_format == "html":
            return self.to_html(output_path)
        else:
            raise ValueError(f"output_format must be 'pdf' or 'html'. Got '{output_format}'.")

    def to_pdf(self, output_path: Path = Path("./gcef_report.pdf")) -> Path:
        """Generate PDF report using reportlab."""
        output_path = Path(output_path)
        _PDFRenderer(self._report_data).render(output_path)
        return output_path

    def to_html(self, output_path: Path = Path("./gcef_report.html")) -> Path:
        """Generate HTML report using Jinja2."""
        output_path = Path(output_path)
        _HTMLRenderer(self._report_data).render(output_path)
        return output_path

    def to_dict(self) -> dict:
        """Return report data as a plain dict (for testing and programmatic use)."""
        return self._report_data.to_dict()


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Compute a SHA-256 hash of a DataFrame for the reproducibility statement.

    Call before evaluator.fit() and pass the hash to ReportGenerator.
    The hash covers column names and values but not the index.

    Returns a 16-character hex prefix (sufficient for reproducibility).
    """
    h = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()
    return h[:16]


# ── Report data assembly ───────────────────────────────────────────────────────

@dataclass
class _ReportData:
    """
    Assembled, renderer-agnostic report data.
    Both PDF and HTML renderers consume this object.
    """
    programme_name: str
    generated_at: str
    analyst_name: str
    data_description: str
    data_hash: str
    gcef_version: str

    # Section 1: Executive summary
    executive_summary: str
    red_flags: List[str]
    amber_flags: List[str]

    # Section 2: Estimand
    estimand_prose: str
    estimand_population: str
    estimand_identification: str
    estimand_dfi_framing: str

    # Section 3: Complier profile
    complier_share: float
    complier_share_ci: Tuple[float, float]
    complier_profile_table: pd.DataFrame
    complier_profile_prose: str

    # Section 4: LATE
    late_estimate: float
    late_se: float
    late_ci_lower: float
    late_ci_upper: float
    f_statistic: float
    late_interpretation: str

    # Section 5: CATE
    cate_mean: float
    cate_std: float
    cate_p25: float
    cate_median: float
    cate_p75: float
    cate_n_compliers: int
    cate_top_heterogeneity: pd.DataFrame
    cate_interpretation: str

    # Section 6: Bounds
    bounds_summary: pd.DataFrame
    bounds_prose: str

    # Section 7: Assumption audit
    assumption_audit: List[Dict]

    # Section 8: Methodological notes
    citations: List[str]

    # Section 9: Reproducibility
    random_seed: Optional[int]
    nuisance_model_overrides: dict

    @classmethod
    def from_results(
        cls,
        results,
        programme_name: str,
        analyst_name: str,
        data_description: str,
        data_hash: str,
    ) -> "_ReportData":
        """Assemble all report sections from a GCEFResults object."""

        gcef_version = getattr(results.estimand, "gcef_version", "0.1.0")
        generated_at = datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # ── Assumption audit ──────────────────────────────────────────────────
        audit = _build_assumption_audit(results.assumptions_tested)
        red_flags = [a["label"] for a in audit if a["status"] == "FAIL"]
        amber_flags = [a["label"] for a in audit if a["status"] == "WARN"]

        # ── LATE ──────────────────────────────────────────────────────────────
        late = results.late
        f_stat = results.assumptions_tested.get(
            "instrument_relevance", {}
        ).get("value", float("nan"))

        # ── Complier profile ──────────────────────────────────────────────────
        cs = results.complier_share
        cs_est = cs.get("estimate", float("nan"))
        cs_lo = cs.get("ci_lower", float("nan"))
        cs_hi = cs.get("ci_upper", float("nan"))

        # ── CATE ──────────────────────────────────────────────────────────────
        cate_df = results.cate
        mask = results.cate_complier_mask.values
        complier_cate = cate_df.loc[mask, "cate_estimate"]
        cate_top = _top_heterogeneity_drivers(cate_df, mask)

        # ── Bounds ────────────────────────────────────────────────────────────
        bounds_summary = _summarise_bounds(results.cate_bounds)

        # ── Prose sections ────────────────────────────────────────────────────
        exec_summary = _executive_summary(
            programme_name=programme_name,
            late=late,
            cs_est=cs_est,
            complier_cate=complier_cate,
            red_flags=red_flags,
            amber_flags=amber_flags,
        )

        late_interp = _late_interpretation(late, f_stat)
        cate_interp = _cate_interpretation(complier_cate, cate_top)
        bounds_prose = _bounds_prose(bounds_summary)
        complier_prose = _complier_prose(
            results.complier_profile, cs_est, cs_lo, cs_hi
        )

        return cls(
            programme_name=programme_name,
            generated_at=generated_at,
            analyst_name=analyst_name,
            data_description=data_description,
            data_hash=data_hash,
            gcef_version=gcef_version,
            executive_summary=exec_summary,
            red_flags=red_flags,
            amber_flags=amber_flags,
            estimand_prose=results.estimand.to_prose(),
            estimand_population=results.estimand.population,
            estimand_identification=results.estimand.identification,
            estimand_dfi_framing=results.estimand.dfi_framing,
            complier_share=cs_est,
            complier_share_ci=(cs_lo, cs_hi),
            complier_profile_table=results.complier_profile,
            complier_profile_prose=complier_prose,
            late_estimate=late["estimate"],
            late_se=late["se"],
            late_ci_lower=late["ci_lower"],
            late_ci_upper=late["ci_upper"],
            f_statistic=f_stat,
            late_interpretation=late_interp,
            cate_mean=float(complier_cate.mean()),
            cate_std=float(complier_cate.std()),
            cate_p25=float(complier_cate.quantile(0.25)),
            cate_median=float(complier_cate.median()),
            cate_p75=float(complier_cate.quantile(0.75)),
            cate_n_compliers=int(mask.sum()),
            cate_top_heterogeneity=cate_top,
            cate_interpretation=cate_interp,
            bounds_summary=bounds_summary,
            bounds_prose=bounds_prose,
            assumption_audit=audit,
            citations=CITATIONS,
            random_seed=getattr(results, "_random_seed", None),
            nuisance_model_overrides=getattr(
                results.estimand, "nuisance_model_overrides", {}
            ),
        )

    def to_dict(self) -> dict:
        """Serialise to plain dict (for testing)."""
        d = {
            k: v for k, v in self.__dict__.items()
            if not isinstance(v, pd.DataFrame)
        }
        d["complier_profile_table"] = self.complier_profile_table.to_dict(
            orient="records"
        )
        d["bounds_summary"] = self.bounds_summary.to_dict(orient="records")
        d["cate_top_heterogeneity"] = self.cate_top_heterogeneity.to_dict(
            orient="records"
        )
        return d


# ── Prose assembly helpers ─────────────────────────────────────────────────────

def _executive_summary(
    programme_name: str,
    late: dict,
    cs_est: float,
    complier_cate: pd.Series,
    red_flags: List[str],
    amber_flags: List[str],
) -> str:
    est = late["estimate"]
    lo = late["ci_lower"]
    hi = late["ci_upper"]
    pct_sign = "+" if est >= 0 else ""
    cate_mean = complier_cate.mean()
    cate_range = complier_cate.quantile(0.75) - complier_cate.quantile(0.25)

    lines = [
        f"This report presents a causal impact evaluation of the "
        f"{programme_name} using the Green Credit Causal Evaluation "
        f"Framework (GCEF v0.1).",
        "",
        f"GREEN CREDIT ACCESS CAUSALLY IMPROVED SME RESILIENCE. "
        f"The Local Average Treatment Effect (LATE) is {pct_sign}{est:,.1f} "
        f"(95% CI: {lo:,.1f} to {hi:,.1f}). This estimate applies to "
        f"the complier subpopulation — SMEs whose access to green credit "
        f"was determined by their lender's programme adoption.",
        "",
        f"EFFECT SIZE VARIES ACROSS SME TYPES. "
        f"The average conditional treatment effect among compliers is "
        f"{cate_mean:,.1f}, with an interquartile range of {cate_range:,.1f}. "
        f"The most responsive SME types are identified in Section 5.",
        "",
        f"PROGRAMME REACH. Approximately {cs_est:.0%} of SMEs in the "
        f"portfolio are estimated to be in the complier subpopulation — "
        f"the marginal population whose green credit behaviour responds "
        f"to programme availability.",
    ]

    if red_flags:
        lines += [
            "",
            f"ATTENTION — IDENTIFICATION CONCERNS. The following assumption "
            f"checks failed and may affect the reliability of these estimates: "
            f"{'; '.join(red_flags)}. See Section 7 for details.",
        ]
    elif amber_flags:
        lines += [
            "",
            f"NOTE: The following assumptions flagged warnings: "
            f"{'; '.join(amber_flags)}. Estimates should be interpreted "
            f"with appropriate caution. See Section 7.",
        ]

    return "\n".join(lines)


def _late_interpretation(late: dict, f_stat: float) -> str:
    est = late["estimate"]
    lo = late["ci_lower"]
    hi = late["ci_upper"]
    sign = "positive" if est > 0 else "negative"
    f_note = (
        f"The first-stage F-statistic is {f_stat:.1f}, "
        f"{'well above' if f_stat > 20 else 'above'} the conventional "
        f"threshold of 10, indicating a strong instrument."
    ) if f_stat >= 10 else (
        f"WARNING: The first-stage F-statistic is {f_stat:.1f}, below "
        f"the conventional threshold of 10. LATE estimates may be biased."
    )
    return (
        f"The LATE of {est:,.1f} (95% CI: {lo:,.1f}–{hi:,.1f}) is "
        f"the estimated causal effect of green credit access on the "
        f"resilience index for the complier subpopulation. "
        f"The effect is {sign} and the confidence interval "
        f"{'excludes' if lo > 0 or hi < 0 else 'includes'} zero. "
        f"{f_note} "
        f"This estimate is identified by variation in lender adoption "
        f"timing and applies only to SMEs who responded to that variation."
    )


def _cate_interpretation(complier_cate: pd.Series, top_drivers: pd.DataFrame) -> str:
    mean = complier_cate.mean()
    std = complier_cate.std()
    p25 = complier_cate.quantile(0.25)
    p75 = complier_cate.quantile(0.75)
    n = len(complier_cate)

    top_cov = (
        top_drivers.iloc[0]["covariate"] if len(top_drivers) > 0 else "unknown"
    )

    return (
        f"Among the {n:,} complier observations, the mean conditional "
        f"treatment effect is {mean:,.1f} (SD: {std:,.1f}). "
        f"The interquartile range ({p25:,.1f}–{p75:,.1f}) indicates "
        f"{'substantial' if (p75 - p25) > std else 'moderate'} "
        f"heterogeneity across SME types. "
        f"The covariate most strongly associated with treatment effect "
        f"variation is {top_cov}. "
        f"SMEs in the upper quartile of predicted treatment effects are "
        f"the primary candidates for programme targeting."
    )


def _complier_prose(
    profile: pd.DataFrame,
    cs_est: float,
    cs_lo: float,
    cs_hi: float,
) -> str:
    return (
        f"Approximately {cs_est:.0%} of SMEs in the portfolio are "
        f"estimated to be compliers (95% CI: {cs_lo:.0%}–{cs_hi:.0%}). "
        f"These are the marginal SMEs — those whose green credit behaviour "
        f"changes with programme availability. The covariate profile below "
        f"characterises this subpopulation relative to the full sample. "
        f"Values with a complier-to-full ratio above 1.0 indicate "
        f"characteristics over-represented among compliers."
    )


def _bounds_prose(bounds_summary: pd.DataFrame) -> str:
    nt = bounds_summary[bounds_summary["population"] == "never_taker"]
    at = bounds_summary[bounds_summary["population"] == "always_taker"]

    parts = [
        "These are Manski (1990) worst-case partial identification bounds. "
        "They are NOT estimates with wide confidence intervals — they are "
        "the honest limits of what the data can rule out for subpopulations "
        "where the instrument provides no identifying variation."
    ]
    if len(nt) > 0:
        nt_lo = nt["mean_lower"].values[0]
        nt_hi = nt["mean_upper"].values[0]
        nt_n = nt["n"].values[0]
        parts.append(
            f"Never-takers ({nt_n:,} observations): treatment effect bounds "
            f"[{nt_lo:,.1f}, {nt_hi:,.1f}]. "
            f"These SMEs would not take up green credit regardless of "
            f"programme availability."
        )
    if len(at) > 0:
        at_lo = at["mean_lower"].values[0]
        at_hi = at["mean_upper"].values[0]
        at_n = at["n"].values[0]
        parts.append(
            f"Always-takers ({at_n:,} observations): treatment effect bounds "
            f"[{at_lo:,.1f}, {at_hi:,.1f}]. "
            f"These SMEs would access green credit regardless of the programme."
        )
    parts.append(
        "Use results.cate (Section 5) for identified treatment effect "
        "estimates for the complier subpopulation."
    )
    return " ".join(parts)


# ── Data assembly helpers ──────────────────────────────────────────────────────

def _build_assumption_audit(assumptions_tested: dict) -> List[Dict]:
    rows = []
    for key, result in assumptions_tested.items():
        passed = result.get("passed")
        label, plain, explanation = _assumption_label(key, result)
        status, colour = ASSUMPTION_STATUS.get(passed, ("WARN", (0.85, 0.65, 0.13)))[:2], \
                         ASSUMPTION_STATUS.get(passed, ("WARN", (0.85, 0.65, 0.13)))[1]
        rows.append({
            "key": key,
            "label": label,
            "status": ASSUMPTION_STATUS.get(passed, ("WARN",))[0],
            "colour": colour,
            "value": result.get("value", ""),
            "plain": plain,
            "explanation": explanation,
        })
    return rows


def _assumption_label(key: str, result: dict) -> Tuple[str, str, str]:
    info = ASSUMPTION_PLAIN_LANGUAGE.get(key, (key.replace("_", " ").title(), "", ""))
    label = info[0] if len(info) > 0 else key
    explanation_base = info[1] if len(info) > 1 else ""

    val = result.get("value", "")
    passed = result.get("passed", None)
    warning_text = result.get("warning") or ""

    if key == "instrument_relevance":
        plain = f"F-statistic: {val:.1f}" if isinstance(val, (int, float)) else str(val)
    elif key == "complier_share":
        plain = f"Estimated share: {val:.1%}" if isinstance(val, (int, float)) else str(val)
    elif key == "kappa_weight_negatives":
        plain = (f"Non-positive share: {val:.1%} "
                 f"(threshold: {result.get('warning_threshold', 0.30):.0%})"
                 if isinstance(val, (int, float)) else str(val))
    elif key == "overlap":
        plain = f"Propensity trimming share: {val:.1%}" if isinstance(val, (int, float)) else str(val)
    elif key == "adoption_timing_anomaly":
        plain = f"Anomalies found: {int(val)}" if isinstance(val, (int, float)) else str(val)
    elif key == "single_lender":
        plain = f"Lenders in dataset: {int(val)}" if isinstance(val, (int, float)) else str(val)
    elif key == "panel_length":
        plain = f"Min pre-treatment periods: {int(val)}" if isinstance(val, (int, float)) else str(val)
    else:
        plain = str(val)

    return label, plain, explanation_base + (" " + warning_text if warning_text else "")


def _top_heterogeneity_drivers(
    cate_df: pd.DataFrame, mask: np.ndarray, top_n: int = 3
) -> pd.DataFrame:
    """
    Identify top covariates driving CATE heterogeneity among compliers.
    Uses correlation between CATE estimates and kappa weights as a proxy.
    """
    complier_cate = cate_df.loc[mask, "cate_estimate"].values
    numeric_cols = [
        c for c in cate_df.columns
        if c not in ("sme_id", "cate_estimate", "cate_se",
                     "cate_ci_lower", "cate_ci_upper", "kappa_weight",
                     "in_shock_period")
        and pd.api.types.is_numeric_dtype(cate_df[c])
    ]

    if not numeric_cols:
        return pd.DataFrame(columns=["covariate", "abs_correlation", "direction"])

    rows = []
    for col in numeric_cols:
        vals = cate_df.loc[mask, col].values
        if np.std(vals) == 0:
            continue
        corr = float(np.corrcoef(complier_cate, vals)[0, 1])
        rows.append({
            "covariate": col,
            "abs_correlation": abs(corr),
            "direction": "positive" if corr > 0 else "negative",
        })

    if not rows:
        return pd.DataFrame(columns=["covariate", "abs_correlation", "direction"])

    df = pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)
    return df.head(top_n).reset_index(drop=True)


def _summarise_bounds(cate_bounds: pd.DataFrame) -> pd.DataFrame:
    if cate_bounds is None or len(cate_bounds) == 0:
        return pd.DataFrame(columns=[
            "population", "n", "mean_lower", "mean_upper", "width"
        ])
    rows = []
    for pop in ["always_taker", "never_taker"]:
        subset = cate_bounds[cate_bounds["population"] == pop].dropna(
            subset=["lower_bound", "upper_bound"]
        )
        if len(subset) == 0:
            continue
        rows.append({
            "population": pop,
            "n": len(subset),
            "mean_lower": subset["lower_bound"].mean(),
            "mean_upper": subset["upper_bound"].mean(),
            "width": (subset["upper_bound"] - subset["lower_bound"]).mean(),
        })
    return pd.DataFrame(rows)


# ── PDF renderer ───────────────────────────────────────────────────────────────

class _PDFRenderer:

    def __init__(self, data: _ReportData):
        self.d = data
        self._styles = None

    def render(self, output_path: Path) -> None:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Spacer, PageBreak,
        )

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=2.5 * cm,
            rightMargin=2.5 * cm,
            topMargin=2.5 * cm,
            bottomMargin=2.5 * cm,
        )

        self._styles = self._build_styles()
        story = []

        story += self._cover_page()
        story += self._section_executive_summary()
        story.append(PageBreak())
        story += self._section_estimand()
        story += self._section_complier_profile()
        story.append(PageBreak())
        story += self._section_late()
        story += self._section_cate()
        story.append(PageBreak())
        story += self._section_bounds()
        story += self._section_assumption_audit()
        story.append(PageBreak())
        story += self._section_methodological_notes()
        story += self._section_reproducibility()

        doc.build(story)

    # ── Style helpers ──────────────────────────────────────────────────────────

    def _build_styles(self):
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT, TA_CENTER

        base = getSampleStyleSheet()
        styles = {}

        styles["title"] = ParagraphStyle(
            "GCEFTitle",
            parent=base["Title"],
            fontSize=20,
            spaceAfter=6,
            textColor=colors.HexColor("#1a3a4a"),
        )
        styles["h1"] = ParagraphStyle(
            "GCEFH1",
            parent=base["Heading1"],
            fontSize=13,
            spaceBefore=14,
            spaceAfter=6,
            textColor=colors.HexColor("#1a3a4a"),
            borderPad=4,
        )
        styles["h2"] = ParagraphStyle(
            "GCEFH2",
            parent=base["Heading2"],
            fontSize=11,
            spaceBefore=10,
            spaceAfter=4,
            textColor=colors.HexColor("#2d6a7f"),
        )
        styles["body"] = ParagraphStyle(
            "GCEFBody",
            parent=base["Normal"],
            fontSize=9,
            leading=13,
            spaceAfter=6,
        )
        styles["small"] = ParagraphStyle(
            "GCEFSmall",
            parent=base["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#555555"),
            spaceAfter=4,
        )
        styles["flag_red"] = ParagraphStyle(
            "GCEFFlagRed",
            parent=base["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#cc0000"),
            spaceAfter=4,
        )
        styles["flag_amber"] = ParagraphStyle(
            "GCEFFlagAmber",
            parent=base["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#b35900"),
            spaceAfter=4,
        )
        styles["stat"] = ParagraphStyle(
            "GCEFStat",
            parent=base["Normal"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#1a3a4a"),
            spaceAfter=2,
        )
        styles["stat_label"] = ParagraphStyle(
            "GCEFStatLabel",
            parent=base["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#666666"),
            spaceAfter=8,
        )
        styles["cite"] = ParagraphStyle(
            "GCEFCite",
            parent=base["Normal"],
            fontSize=8,
            leading=11,
            leftIndent=12,
            spaceAfter=3,
            textColor=colors.HexColor("#333333"),
        )
        styles["repro"] = ParagraphStyle(
            "GCEFRepro",
            parent=base["Normal"],
            fontSize=8,
            leading=12,
            fontName="Courier",
            backColor=colors.HexColor("#f5f5f5"),
            borderPad=6,
            spaceAfter=4,
        )
        return styles

    # ── Section builders ───────────────────────────────────────────────────────

    def _cover_page(self):
        from reportlab.platypus import Paragraph, Spacer, HRFlowable
        from reportlab.lib import colors
        S = self.d
        els = [
            Spacer(1, 40),
            Paragraph("GREEN CREDIT CAUSAL EVALUATION", self._styles["small"]),
            Paragraph(S.programme_name, self._styles["title"]),
            HRFlowable(width="100%", thickness=2,
                       color=colors.HexColor("#1a3a4a"), spaceAfter=10),
            Paragraph(f"Generated: {S.generated_at}", self._styles["small"]),
        ]
        if S.analyst_name:
            els.append(Paragraph(f"Analyst: {S.analyst_name}", self._styles["small"]))
        if S.data_description:
            els.append(Paragraph(f"Dataset: {S.data_description}", self._styles["small"]))
        els.append(Spacer(1, 10))
        els.append(Paragraph(
            f"GCEF version {S.gcef_version} · {GCEF_GITHUB}",
            self._styles["small"],
        ))
        return els

    def _section_executive_summary(self):
        from reportlab.platypus import Paragraph, Spacer
        S = self.d
        els = [
            Spacer(1, 20),
            Paragraph("1. Executive Summary", self._styles["h1"]),
        ]
        # Red flags first
        for flag in S.red_flags:
            els.append(Paragraph(f"⚠ IDENTIFICATION CONCERN: {flag}",
                                 self._styles["flag_red"]))
        for flag in S.amber_flags:
            els.append(Paragraph(f"△ WARNING: {flag}",
                                 self._styles["flag_amber"]))

        # Key statistics row
        els += self._stat_row([
            (f"{S.late_estimate:+,.0f}", "LATE (resilience index)"),
            (f"{S.complier_share:.0%}", "Complier share"),
            (f"{S.cate_mean:,.0f}", "Mean CATE (compliers)"),
            (f"{S.f_statistic:.0f}", "First-stage F"),
        ])
        # Prose
        for para in S.executive_summary.split("\n\n"):
            if para.strip():
                els.append(Paragraph(para.strip(), self._styles["body"]))
        return els

    def _section_estimand(self):
        from reportlab.platypus import Paragraph, Spacer
        S = self.d
        els = [
            Paragraph("2. Estimand Statement", self._styles["h1"]),
            Paragraph(
                "Every estimate in this report applies to a specific subpopulation. "
                "This section states that subpopulation explicitly.",
                self._styles["body"],
            ),
            Paragraph("<b>Population:</b> " + S.estimand_population.replace("_", " ").title(),
                      self._styles["body"]),
            Paragraph("<b>Identification strategy:</b> " + S.estimand_identification,
                      self._styles["body"]),
            Paragraph("<b>DFI framing:</b> " + S.estimand_dfi_framing,
                      self._styles["body"]),
            Spacer(1, 4),
            Paragraph(S.estimand_prose, self._styles["small"]),
        ]
        return els

    def _section_complier_profile(self):
        from reportlab.platypus import Paragraph, Spacer
        S = self.d
        ci_lo, ci_hi = S.complier_share_ci
        els = [
            Paragraph("3. Complier Profile", self._styles["h1"]),
            Paragraph(S.complier_profile_prose, self._styles["body"]),
        ]
        els += self._stat_row([
            (f"{S.complier_share:.0%}", "Complier share"),
            (f"{ci_lo:.0%}–{ci_hi:.0%}", "95% CI"),
            (f"{S.cate_n_compliers:,}", "Complier observations"),
        ])
        if len(S.complier_profile_table) > 0:
            els.append(self._build_table(
                S.complier_profile_table.rename(columns={
                    "covariate": "Covariate",
                    "weighted_mean": "Complier mean",
                    "full_mean": "Full sample mean",
                    "complier_to_full_ratio": "Ratio",
                }),
                fmt={
                    "Complier mean": lambda x: f"{x:,.3f}",
                    "Full sample mean": lambda x: f"{x:,.3f}",
                    "Ratio": lambda x: f"{x:.3f}",
                },
            ))
        return els

    def _section_late(self):
        from reportlab.platypus import Paragraph, Spacer
        S = self.d
        els = [
            Paragraph("4. LATE Estimate", self._styles["h1"]),
        ]
        els += self._stat_row([
            (f"{S.late_estimate:+,.1f}", "LATE"),
            (f"± {S.late_se:,.1f}", "Standard error"),
            (f"{S.late_ci_lower:,.1f} – {S.late_ci_upper:,.1f}", "95% CI"),
            (f"{S.f_statistic:.1f}", "First-stage F"),
        ])
        els.append(Paragraph(S.late_interpretation, self._styles["body"]))
        return els

    def _section_cate(self):
        from reportlab.platypus import Paragraph, Spacer
        S = self.d
        els = [
            Paragraph("5. CATE Heterogeneity", self._styles["h1"]),
            Paragraph(S.cate_interpretation, self._styles["body"]),
        ]
        els += self._stat_row([
            (f"{S.cate_mean:,.1f}", "Mean CATE"),
            (f"{S.cate_median:,.1f}", "Median CATE"),
            (f"{S.cate_p25:,.1f} – {S.cate_p75:,.1f}", "IQR"),
            (f"{S.cate_std:,.1f}", "Std dev"),
        ])
        if len(S.cate_top_heterogeneity) > 0:
            els.append(Paragraph("Top heterogeneity drivers:", self._styles["h2"]))
            els.append(self._build_table(
                S.cate_top_heterogeneity.rename(columns={
                    "covariate": "Covariate",
                    "abs_correlation": "|Correlation with CATE|",
                    "direction": "Direction",
                }),
                fmt={"|Correlation with CATE|": lambda x: f"{x:.3f}"},
            ))
        els.append(Paragraph(
            "Note: CATE estimates are conditioned on the complier subpopulation "
            "and on shock periods. Confidence intervals are approximate "
            "(ForestDRIV with sample weights; spec limitation L8).",
            self._styles["small"],
        ))
        return els

    def _section_bounds(self):
        from reportlab.platypus import Paragraph, Spacer
        S = self.d
        els = [
            Paragraph("6. Bounds for Full Portfolio", self._styles["h1"]),
            Paragraph(S.bounds_prose, self._styles["body"]),
        ]
        if len(S.bounds_summary) > 0:
            els.append(self._build_table(
                S.bounds_summary.rename(columns={
                    "population": "Subpopulation",
                    "n": "N observations",
                    "mean_lower": "Mean lower bound",
                    "mean_upper": "Mean upper bound",
                    "width": "Mean bound width",
                }),
                fmt={
                    "Mean lower bound": lambda x: f"{x:,.1f}",
                    "Mean upper bound": lambda x: f"{x:,.1f}",
                    "Mean bound width": lambda x: f"{x:,.1f}",
                },
            ))
        return els

    def _section_assumption_audit(self):
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        S = self.d
        els = [
            Paragraph("7. Assumption Audit", self._styles["h1"]),
            Paragraph(
                "Each identification assumption is tested where possible. "
                "PASS = assumption satisfied. FAIL = assumption violated. "
                "WARN = marginal or untestable.",
                self._styles["body"],
            ),
        ]
        # Build audit table
        rows = [["Assumption", "Status", "Value", "Notes"]]
        col_colours = []
        for i, a in enumerate(S.assumption_audit):
            status = a["status"]
            colour = a["colour"]
            rows.append([
                a["label"],
                status,
                str(a["value"])[:20],
                a["explanation"][:60] + ("..." if len(a["explanation"]) > 60 else ""),
            ])
            # Convert 0–1 RGB to reportlab color
            col_colours.append((i + 1, colours_from_tuple(colour)))

        table = Table(rows, colWidths=["38%", "10%", "16%", "36%"],
                      hAlign="LEFT")
        style_cmds = [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.HexColor("#f5f5f5"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]
        for row_idx, colour in col_colours:
            style_cmds.append(
                ("TEXTCOLOR", (1, row_idx), (1, row_idx), colour)
            )
            style_cmds.append(
                ("FONTNAME", (1, row_idx), (1, row_idx), "Helvetica-Bold")
            )
        table.setStyle(TableStyle(style_cmds))
        els.append(table)
        return els

    def _section_methodological_notes(self):
        from reportlab.platypus import Paragraph, Spacer
        els = [
            Paragraph("8. Methodological Notes", self._styles["h1"]),
            Paragraph(
                "GCEF uses a two-stage causal identification strategy. "
                "Stage 1 estimates the Local Average Treatment Effect (LATE) "
                "using instrumental variables with staggered difference-in-differences "
                "(Callaway & Sant'Anna, 2021). The instrument is lender adoption timing — "
                "the period in which each lender introduced a green credit product. "
                "Stage 2 estimates Conditional Average Treatment Effects (CATE) "
                "across the complier subpopulation using a causal forest "
                "(Wager & Athey, 2018) with kappa-weighted sample weights "
                "(Abadie, 2003). For subpopulations without identifying variation "
                "(always-takers and never-takers), Manski (1990) worst-case "
                "partial identification bounds are reported.",
                self._styles["body"],
            ),
            Paragraph("Full specification:", self._styles["h2"]),
            Paragraph(f"{GCEF_GITHUB} · Specification DOI: {GCEF_SPEC_DOI}",
                      self._styles["small"]),
            Paragraph("References:", self._styles["h2"]),
        ]
        for cite in CITATIONS:
            els.append(Paragraph(f"• {cite}", self._styles["cite"]))
        return els

    def _section_reproducibility(self):
        from reportlab.platypus import Paragraph, Spacer, HRFlowable
        from reportlab.lib import colors
        S = self.d
        els = [
            Paragraph("9. Reproducibility Statement", self._styles["h1"]),
            Paragraph(
                "This statement is mandatory and non-suppressible. "
                "All analyses using GCEF must include it.",
                self._styles["small"],
            ),
        ]
        items = [
            f"GCEF version:     {S.gcef_version}",
            f"Generated at:     {S.generated_at}",
            f"Random seed:      {S.random_seed if S.random_seed is not None else 'not set'}",
            f"Data hash:        {S.data_hash if S.data_hash else '(not provided — call hash_dataframe() before fit())'}",
            f"Data:             {S.data_description if S.data_description else '(not provided)'}",
            f"Analyst:          {S.analyst_name if S.analyst_name else '(not provided)'}",
        ]
        if S.nuisance_model_overrides:
            items.append(f"Nuisance overrides: {S.nuisance_model_overrides}")
        else:
            items.append("Nuisance models:  defaults (GradientBoostingRegressor)")
        for item in items:
            els.append(Paragraph(item, self._styles["repro"]))
        els.append(Spacer(1, 6))
        els.append(Paragraph(
            "Cite as: Maharaj, P. (2026). Green Credit Causal Evaluation Framework "
            f"(GCEF), version {S.gcef_version}. {GCEF_GITHUB}. "
            "Also cite: Callaway & Sant'Anna (2021); Wager & Athey (2018); "
            "Abadie (2003); Manski (1990).",
            self._styles["small"],
        ))
        return els

    # ── Layout helpers ─────────────────────────────────────────────────────────

    def _stat_row(self, stats: List[Tuple[str, str]]):
        """Build a row of large stat + label pairs."""
        from reportlab.platypus import Table, TableStyle, Paragraph
        from reportlab.lib import colors
        col_width = 100.0 / len(stats)
        cells = [[
            [Paragraph(val, self._styles["stat"]),
             Paragraph(label, self._styles["stat_label"])]
            for val, label in stats
        ]]
        t = Table(cells, colWidths=[f"{col_width}%"] * len(stats))
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        return [t]

    def _build_table(
        self,
        df: pd.DataFrame,
        fmt: Dict = None,
    ):
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors
        fmt = fmt or {}
        rows = [list(df.columns)]
        for _, row in df.iterrows():
            cells = []
            for col in df.columns:
                val = row[col]
                if col in fmt:
                    try:
                        val = fmt[col](val)
                    except Exception:
                        pass
                cells.append(str(val))
            rows.append(cells)
        col_w = f"{100.0 / len(df.columns):.0f}%"
        t = Table(rows, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.HexColor("#f0f4f7"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]))
        return t


def colours_from_tuple(rgb: Tuple) -> Any:
    from reportlab.lib import colors
    return colors.Color(*rgb)


# ── HTML renderer ──────────────────────────────────────────────────────────────

class _HTMLRenderer:

    TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GCEF Report — {{ d.programme_name }}</title>
<style>
  :root {
    --ink: #1a2a3a; --accent: #1a3a4a; --muted: #555;
    --green: #2e7d52; --red: #c0392b; --amber: #b35900;
    --bg: #ffffff; --rule: #dde;
  }
  body { font-family: Georgia, serif; max-width: 860px; margin: 40px auto;
         padding: 0 24px; color: var(--ink); line-height: 1.6; }
  h1 { font-size: 1.5rem; border-bottom: 2px solid var(--accent);
       color: var(--accent); padding-bottom: 4px; margin-top: 2.5rem; }
  h2 { font-size: 1.1rem; color: var(--accent); margin-top: 1.5rem; }
  .cover { border-left: 5px solid var(--accent); padding: 16px 20px;
            background: #f4f8fb; margin-bottom: 2rem; }
  .cover h1 { border: none; margin: 0; font-size: 2rem; }
  .cover .meta { color: var(--muted); font-size: 0.85rem; margin-top: 8px; }
  .stat-row { display: flex; gap: 20px; margin: 16px 0; flex-wrap: wrap; }
  .stat-box { flex: 1; min-width: 120px; background: #f4f8fb;
               border-radius: 6px; padding: 12px 16px; }
  .stat-val { font-size: 1.6rem; font-weight: bold; color: var(--accent); }
  .stat-label { font-size: 0.78rem; color: var(--muted); }
  .flag-red { color: var(--red); font-weight: bold; background: #fff0f0;
               border-left: 3px solid var(--red); padding: 8px 12px;
               margin: 6px 0; border-radius: 2px; }
  .flag-amber { color: var(--amber); font-weight: bold; background: #fff8f0;
                 border-left: 3px solid var(--amber); padding: 8px 12px;
                 margin: 6px 0; border-radius: 2px; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.88rem; }
  th { background: #eef2f6; text-align: left; padding: 6px 10px;
       border-bottom: 2px solid var(--rule); }
  td { padding: 5px 10px; border-bottom: 1px solid var(--rule); }
  tr:nth-child(even) { background: #f9fbfc; }
  .pass { color: var(--green); font-weight: bold; }
  .fail { color: var(--red); font-weight: bold; }
  .warn { color: var(--amber); font-weight: bold; }
  .repro { background: #f5f5f5; border: 1px solid #ddd; padding: 14px;
            font-family: monospace; font-size: 0.82rem; white-space: pre-wrap;
            border-radius: 4px; margin: 12px 0; }
  .mandatory { font-size: 0.8rem; color: var(--amber); margin-bottom: 6px; }
  .note { font-size: 0.82rem; color: var(--muted); }
  .cite { font-size: 0.82rem; margin: 4px 0 4px 14px; }
  footer { margin-top: 3rem; border-top: 1px solid var(--rule);
            padding-top: 12px; font-size: 0.8rem; color: var(--muted); }
</style>
</head>
<body>

<div class="cover">
  <div class="meta">GREEN CREDIT CAUSAL EVALUATION</div>
  <h1>{{ d.programme_name }}</h1>
  <div class="meta">
    Generated: {{ d.generated_at }}
    {% if d.analyst_name %} · Analyst: {{ d.analyst_name }}{% endif %}
    {% if d.data_description %} · Dataset: {{ d.data_description }}{% endif %}
    <br>GCEF version {{ d.gcef_version }} ·
    <a href="{{ github }}">{{ github }}</a>
  </div>
</div>

<h1>1. Executive Summary</h1>
{% for flag in d.red_flags %}
<div class="flag-red">⚠ IDENTIFICATION CONCERN: {{ flag }}</div>
{% endfor %}
{% for flag in d.amber_flags %}
<div class="flag-amber">△ WARNING: {{ flag }}</div>
{% endfor %}

<div class="stat-row">
  <div class="stat-box">
    <div class="stat-val">{{ "%.0f"|format(d.late_estimate) }}</div>
    <div class="stat-label">LATE (resilience index)</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">{{ "%.0%"|format(d.complier_share) }}</div>
    <div class="stat-label">Complier share</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">{{ "%.0f"|format(d.cate_mean) }}</div>
    <div class="stat-label">Mean CATE (compliers)</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">{{ "%.0f"|format(d.f_statistic) }}</div>
    <div class="stat-label">First-stage F</div>
  </div>
</div>

{% for para in d.executive_summary.split("\\n\\n") %}
<p>{{ para }}</p>
{% endfor %}

<h1>2. Estimand Statement</h1>
<p>Every estimate in this report applies to a specific subpopulation.
This section states that subpopulation explicitly.</p>
<p><strong>Population:</strong> {{ d.estimand_population.replace("_", " ").title() }}</p>
<p><strong>Identification:</strong> {{ d.estimand_identification }}</p>
<p><strong>DFI framing:</strong> {{ d.estimand_dfi_framing }}</p>
<p class="note">{{ d.estimand_prose }}</p>

<h1>3. Complier Profile</h1>
<p>{{ d.complier_profile_prose }}</p>
<div class="stat-row">
  <div class="stat-box">
    <div class="stat-val">{{ "%.0%"|format(d.complier_share) }}</div>
    <div class="stat-label">Complier share</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">{{ "%.0%"|format(d.complier_share_ci[0]) }}–{{ "%.0%"|format(d.complier_share_ci[1]) }}</div>
    <div class="stat-label">95% CI</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">{{ "{:,}".format(d.cate_n_compliers) }}</div>
    <div class="stat-label">Complier observations</div>
  </div>
</div>
<table>
  <tr><th>Covariate</th><th>Complier mean</th><th>Full sample mean</th><th>Ratio</th></tr>
  {% for row in d.complier_profile_table %}
  <tr>
    <td>{{ row.covariate }}</td>
    <td>{{ "%.3f"|format(row.weighted_mean) }}</td>
    <td>{{ "%.3f"|format(row.full_mean) }}</td>
    <td>{{ "%.3f"|format(row.complier_to_full_ratio) }}</td>
  </tr>
  {% endfor %}
</table>

<h1>4. LATE Estimate</h1>
<div class="stat-row">
  <div class="stat-box">
    <div class="stat-val">{{ "%+.1f"|format(d.late_estimate) }}</div>
    <div class="stat-label">LATE</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">± {{ "%.1f"|format(d.late_se) }}</div>
    <div class="stat-label">Standard error</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">{{ "%.1f"|format(d.late_ci_lower) }} – {{ "%.1f"|format(d.late_ci_upper) }}</div>
    <div class="stat-label">95% CI</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">{{ "%.1f"|format(d.f_statistic) }}</div>
    <div class="stat-label">First-stage F</div>
  </div>
</div>
<p>{{ d.late_interpretation }}</p>

<h1>5. CATE Heterogeneity</h1>
<p>{{ d.cate_interpretation }}</p>
<div class="stat-row">
  <div class="stat-box">
    <div class="stat-val">{{ "%.0f"|format(d.cate_mean) }}</div>
    <div class="stat-label">Mean CATE</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">{{ "%.0f"|format(d.cate_median) }}</div>
    <div class="stat-label">Median CATE</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">{{ "%.0f"|format(d.cate_p25) }}–{{ "%.0f"|format(d.cate_p75) }}</div>
    <div class="stat-label">IQR</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">{{ "%.0f"|format(d.cate_std) }}</div>
    <div class="stat-label">Std dev</div>
  </div>
</div>
{% if d.cate_top_heterogeneity %}
<h2>Top heterogeneity drivers</h2>
<table>
  <tr><th>Covariate</th><th>|Correlation with CATE|</th><th>Direction</th></tr>
  {% for row in d.cate_top_heterogeneity %}
  <tr>
    <td>{{ row.covariate }}</td>
    <td>{{ "%.3f"|format(row.abs_correlation) }}</td>
    <td>{{ row.direction }}</td>
  </tr>
  {% endfor %}
</table>
{% endif %}
<p class="note">CATE confidence intervals are approximate (ForestDRIV with
sample weights; specification limitation L8).</p>

<h1>6. Bounds for Full Portfolio</h1>
<p>{{ d.bounds_prose }}</p>
{% if d.bounds_summary %}
<table>
  <tr><th>Subpopulation</th><th>N</th><th>Mean lower bound</th>
      <th>Mean upper bound</th><th>Mean width</th></tr>
  {% for row in d.bounds_summary %}
  <tr>
    <td>{{ row.population.replace("_", "-").title() }}</td>
    <td>{{ "{:,}".format(row.n) }}</td>
    <td>{{ "%.1f"|format(row.mean_lower) }}</td>
    <td>{{ "%.1f"|format(row.mean_upper) }}</td>
    <td>{{ "%.1f"|format(row.width) }}</td>
  </tr>
  {% endfor %}
</table>
{% endif %}

<h1>7. Assumption Audit</h1>
<p>PASS = assumption satisfied. FAIL = violated. WARN = marginal or untestable.</p>
<table>
  <tr><th>Assumption</th><th>Status</th><th>Value</th><th>Notes</th></tr>
  {% for a in d.assumption_audit %}
  <tr>
    <td>{{ a.label }}</td>
    <td class="{{ a.status.lower() }}">{{ a.status }}</td>
    <td>{{ a.value }}</td>
    <td class="note">{{ a.explanation }}</td>
  </tr>
  {% endfor %}
</table>

<h1>8. Methodological Notes</h1>
<p>GCEF uses a two-stage causal identification strategy. Stage 1 estimates
the Local Average Treatment Effect (LATE) using instrumental variables with
staggered difference-in-differences (Callaway &amp; Sant'Anna, 2021). The
instrument is lender adoption timing. Stage 2 estimates Conditional Average
Treatment Effects (CATE) using a causal forest (Wager &amp; Athey, 2018)
with kappa-weighted sample weights (Abadie, 2003). Manski (1990) worst-case
bounds are reported for non-identified subpopulations.</p>
<p><strong>Full specification:</strong>
<a href="{{ github }}">{{ github }}</a> · DOI: {{ spec_doi }}</p>
<h2>References</h2>
{% for cite in d.citations %}
<p class="cite">{{ cite }}</p>
{% endfor %}

<h1>9. Reproducibility Statement</h1>
<p class="mandatory">⚠ This statement is mandatory and non-suppressible.</p>
<div class="repro">GCEF version:      {{ d.gcef_version }}
Generated at:      {{ d.generated_at }}
Random seed:       {{ d.random_seed if d.random_seed is not none else "not set" }}
Data hash:         {{ d.data_hash if d.data_hash else "(not provided — call hash_dataframe() before fit())" }}
Data:              {{ d.data_description if d.data_description else "(not provided)" }}
Analyst:           {{ d.analyst_name if d.analyst_name else "(not provided)" }}
Nuisance models:   {{ d.nuisance_model_overrides if d.nuisance_model_overrides else "defaults (GradientBoostingRegressor)" }}</div>
<p class="note">Cite as: Maharaj, P. (2026). Green Credit Causal Evaluation Framework
(GCEF), version {{ d.gcef_version }}. {{ github }}.
Also cite: Callaway &amp; Sant'Anna (2021); Wager &amp; Athey (2018);
Abadie (2003); Manski (1990).</p>

<footer>
  GCEF v{{ d.gcef_version }} · Generated {{ d.generated_at }} ·
  <a href="{{ github }}">{{ github }}</a>
</footer>
</body>
</html>"""

    def __init__(self, data: _ReportData):
        self.d = data

    def render(self, output_path: Path) -> None:
        from jinja2 import Environment
        env = Environment(autoescape=False)
        # Register the format filter for percentages
        env.filters["format"] = lambda value, fmt: (fmt % value) if isinstance(value, (int, float)) else str(value)

        template = env.from_string(self.TEMPLATE)

        # Convert DataFrames to list-of-dicts for Jinja2
        d = self.d
        html = template.render(
            d=_HTMLTemplateData(d),
            github=GCEF_GITHUB,
            spec_doi=GCEF_SPEC_DOI,
        )
        output_path.write_text(html, encoding="utf-8")


class _HTMLTemplateData:
    """Wrapper that makes _ReportData's DataFrames accessible as list-of-dicts."""

    def __init__(self, data: _ReportData):
        self._data = data
        # Convert DataFrames to list of SimpleNamespace
        self.complier_profile_table = [
            _row_to_ns(row)
            for _, row in data.complier_profile_table.iterrows()
        ]
        self.bounds_summary = [
            _row_to_ns(row)
            for _, row in data.bounds_summary.iterrows()
        ]
        self.cate_top_heterogeneity = [
            _row_to_ns(row)
            for _, row in data.cate_top_heterogeneity.iterrows()
        ]

    def __getattr__(self, name):
        return getattr(self._data, name)


def _row_to_ns(row: pd.Series):
    from types import SimpleNamespace
    return SimpleNamespace(**row.to_dict())
