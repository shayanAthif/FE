"""
Statistical analysis module for Phase 5.

Computes:
  - Pearson and Spearman correlations (risk score vs market returns/volatility)
  - OLS regression results
  - Descriptive group comparisons (high-risk vs low-risk)
  - Volatility analysis
  - Guidance/outcome analysis

Anti-leakage:
  - Market outcomes are measured AFTER the call date.
  - Risk scores are computed only from transcript text (no future data).
  - Train/val/test splits use chronological cutoffs, not random assignment.
  - No regression coefficient is used for forward prediction without the split.

Causal language is avoided. All results use associative framing:
  "associated with", "observed relationship", "correlated with"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _pearson_with_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000) -> Dict:
    """Pearson r with 95% bootstrap CI."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return {"r": np.nan, "p": np.nan, "n": n, "ci_lo": np.nan, "ci_hi": np.nan}

    r, p = stats.pearsonr(x, y)
    rng = np.random.default_rng(42)
    boot_rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        br, _ = stats.pearsonr(x[idx], y[idx])
        boot_rs.append(br)
    return {
        "r": float(r),
        "p": float(p),
        "n": int(n),
        "ci_lo": float(np.percentile(boot_rs, 2.5)),
        "ci_hi": float(np.percentile(boot_rs, 97.5)),
    }


def _spearman_with_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000) -> Dict:
    """Spearman rho with 95% bootstrap CI."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return {"rho": np.nan, "p": np.nan, "n": n, "ci_lo": np.nan, "ci_hi": np.nan}

    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(42)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        brho, _ = stats.spearmanr(x[idx], y[idx])
        boot_rhos.append(brho)
    return {
        "rho": float(rho),
        "p": float(p),
        "n": int(n),
        "ci_lo": float(np.percentile(boot_rhos, 2.5)),
        "ci_hi": float(np.percentile(boot_rhos, 97.5)),
    }


def market_correlation_analysis(df: pd.DataFrame) -> Dict:
    """
    Compute Pearson and Spearman correlations between risk scores and
    post-call market metrics.

    Only uses observations where both risk score and market metric are finite.
    Analysis is observational — no causal claims.

    Returns nested dict: metric_name → {pearson, spearman} → {r/rho, p, n, ci_lo, ci_hi}
    """
    risk_cols = {
        "overall_hidden_risk": "Proposed: Hidden Risk Score",
        "lm_lexicon_score":    "Baseline 1: LM Lexicon",
        "finbert_only_score":  "Baseline 2: FinBERT (tone shift)",
        "hedging_only_score":  "Baseline 3: Hedging Only",
    }
    market_cols = {
        "return_1d":           "1-Day Return",
        "return_5d":           "5-Day Return",
        "return_10d":          "10-Day Return",
        "return_20d":          "20-Day Return",
        "volatility_5d":       "5-Day Volatility",
        "volatility_10d":      "10-Day Volatility",
        "abnormal_return_5d":  "Abnormal 5-Day Return",
    }

    results = {}
    for rcol, rlabel in risk_cols.items():
        if rcol not in df.columns:
            continue
        x = pd.to_numeric(df[rcol], errors="coerce").values
        results[rlabel] = {}
        for mcol, mlabel in market_cols.items():
            if mcol not in df.columns:
                continue
            y = pd.to_numeric(df[mcol], errors="coerce").values
            results[rlabel][mlabel] = {
                "pearson": _pearson_with_ci(x, y),
                "spearman": _spearman_with_ci(x, y),
            }

    return results


def regression_analysis(df: pd.DataFrame) -> Dict:
    """
    Simple OLS regression: market_outcome ~ risk_score + controls.

    Controls: year (time trend), ticker one-hot not used (too few obs).
    Fits separate regressions for each risk score × market outcome.

    Returns dict with regression statistics for each combination.

    Note: With N=103 observations these regressions have very low power.
    Results are descriptive, not inferential.
    """
    from scipy.stats import t as t_dist

    risk_cols = ["overall_hidden_risk", "lm_lexicon_score", "finbert_only_score", "hedging_only_score"]
    market_cols = ["return_5d", "return_20d", "volatility_5d", "volatility_10d"]

    df_num = df.copy()
    df_num["year_numeric"] = pd.to_datetime(df_num["date"]).dt.year.astype(float)

    results = {}
    for rcol in risk_cols:
        if rcol not in df_num.columns:
            continue
        results[rcol] = {}
        for mcol in market_cols:
            if mcol not in df_num.columns:
                continue

            sub = df_num[[rcol, mcol, "year_numeric"]].dropna()
            if len(sub) < 10:
                continue

            # Design matrix: intercept + risk_score + year
            X = np.column_stack([
                np.ones(len(sub)),
                sub[rcol].values,
                sub["year_numeric"].values,
            ])
            y = sub[mcol].values

            try:
                beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
                y_pred = X @ beta
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

                # Standard errors via normal equations
                n, k = X.shape
                df_resid = n - k
                if df_resid <= 0:
                    continue
                s2 = ss_res / df_resid
                XtX_inv = np.linalg.pinv(X.T @ X)
                se = np.sqrt(s2 * np.diag(XtX_inv))
                t_stats = beta / se
                p_vals = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=df_resid))

                results[rcol][mcol] = {
                    "n": int(n),
                    "r_squared": float(r2),
                    "beta_risk": float(beta[1]),
                    "se_risk": float(se[1]),
                    "t_risk": float(t_stats[1]),
                    "p_risk": float(p_vals[1]),
                    "beta_year": float(beta[2]),
                    "intercept": float(beta[0]),
                }
            except Exception as e:
                results[rcol][mcol] = {"error": str(e)}

    return results


def volatility_analysis(df: pd.DataFrame) -> Dict:
    """
    Compare 5-day and 10-day post-call volatility for high-risk vs low-risk calls.

    High-risk: overall_hidden_risk >= 75th percentile
    Low-risk:  overall_hidden_risk <= 25th percentile

    Uses Mann-Whitney U test (non-parametric) given small N.
    Returns test statistics and group descriptives.
    """
    risk = pd.to_numeric(df["overall_hidden_risk"], errors="coerce")
    q25 = risk.quantile(0.25)
    q75 = risk.quantile(0.75)

    high_mask = risk >= q75
    low_mask = risk <= q25

    results = {}
    for vcol in ["volatility_5d", "volatility_10d", "return_5d", "return_20d"]:
        if vcol not in df.columns:
            continue
        col = pd.to_numeric(df[vcol], errors="coerce")
        high_vals = col[high_mask].dropna().values
        low_vals = col[low_mask].dropna().values

        if len(high_vals) < 3 or len(low_vals) < 3:
            continue

        stat, p = stats.mannwhitneyu(high_vals, low_vals, alternative="two-sided")
        results[vcol] = {
            "high_risk_n": int(len(high_vals)),
            "high_risk_mean": float(np.mean(high_vals)),
            "high_risk_median": float(np.median(high_vals)),
            "high_risk_std": float(np.std(high_vals)),
            "low_risk_n": int(len(low_vals)),
            "low_risk_mean": float(np.mean(low_vals)),
            "low_risk_median": float(np.median(low_vals)),
            "low_risk_std": float(np.std(low_vals)),
            "mannwhitney_U": float(stat),
            "p_value": float(p),
            "risk_threshold_q25": float(q25),
            "risk_threshold_q75": float(q75),
        }

    return results


def guidance_outcome_analysis(df: pd.DataFrame) -> Dict:
    """
    Test whether high-risk calls are associated with:
      - negative earnings surprise
      - guidance reduction

    Returns descriptive statistics and Mann-Whitney U test results.
    Language: observational, not causal.
    """
    risk = pd.to_numeric(df["overall_hidden_risk"], errors="coerce")
    q75 = risk.quantile(0.75)
    q25 = risk.quantile(0.25)

    high_mask = risk >= q75
    low_mask = risk <= q25

    results = {}

    for ocol in ["earnings_surprise", "actual_revenue", "guidance_change"]:
        if ocol not in df.columns:
            continue
        col = pd.to_numeric(df[ocol], errors="coerce")
        high_vals = col[high_mask].dropna().values
        low_vals = col[low_mask].dropna().values

        if len(high_vals) < 3 or len(low_vals) < 3:
            results[ocol] = {
                "note": f"Insufficient non-null data: high_n={len(high_vals)}, low_n={len(low_vals)}"
            }
            continue

        stat, p = stats.mannwhitneyu(high_vals, low_vals, alternative="two-sided")
        results[ocol] = {
            "high_risk_n": int(len(high_vals)),
            "high_risk_mean": float(np.mean(high_vals)),
            "high_risk_median": float(np.median(high_vals)),
            "low_risk_n": int(len(low_vals)),
            "low_risk_mean": float(np.mean(low_vals)),
            "low_risk_median": float(np.median(low_vals)),
            "mannwhitney_U": float(stat),
            "p_value": float(p),
        }

    return results


def descriptive_statistics(df: pd.DataFrame) -> Dict:
    """
    Descriptive statistics for the full corpus.

    Returns dict with score distributions, split counts, verification summary,
    and QA vs prepared comparison.
    """
    numeric_cols = [
        "overall_hidden_risk", "average_risk", "hedging_avg",
        "evasiveness_avg", "tone_shift_avg", "qa_risk", "prepared_risk",
        "lm_lexicon_score", "finbert_only_score",
        "return_1d", "return_5d", "return_10d", "return_20d",
        "volatility_5d", "volatility_10d",
    ]

    desc = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        desc[col] = {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
        }

    split_counts = df.get("split", pd.Series()).value_counts().to_dict()

    # QA vs Prepared comparison
    qa_prep = {}
    if "qa_risk" in df.columns and "prepared_risk" in df.columns:
        qa = pd.to_numeric(df["qa_risk"], errors="coerce").dropna()
        prep = pd.to_numeric(df["prepared_risk"], errors="coerce").dropna()
        if len(qa) >= 5 and len(prep) >= 5:
            stat, p = stats.wilcoxon(
                qa.values[:min(len(qa), len(prep))],
                prep.values[:min(len(qa), len(prep))],
                alternative="two-sided",
            )
            qa_prep = {
                "qa_mean": float(qa.mean()),
                "prepared_mean": float(prep.mean()),
                "qa_median": float(qa.median()),
                "prepared_median": float(prep.median()),
                "wilcoxon_stat": float(stat),
                "p_value": float(p),
                "interpretation": (
                    "Q&A sections show higher risk scores than Prepared Remarks, "
                    "consistent with spontaneous responses revealing more uncertainty."
                    if qa.mean() > prep.mean()
                    else "Prepared Remarks show equal or higher risk scores than Q&A sections."
                ),
            }

    return {
        "n_transcripts": len(df),
        "split_counts": split_counts,
        "score_distributions": desc,
        "qa_vs_prepared": qa_prep,
    }


def build_correlation_table(corr_results: Dict) -> pd.DataFrame:
    """
    Flatten correlation results into a tidy DataFrame for CSV export.
    """
    rows = []
    for system, market_dict in corr_results.items():
        for market_metric, stat_dict in market_dict.items():
            row = {
                "system": system,
                "market_metric": market_metric,
                "pearson_r": stat_dict["pearson"]["r"],
                "pearson_p": stat_dict["pearson"]["p"],
                "pearson_ci_lo": stat_dict["pearson"]["ci_lo"],
                "pearson_ci_hi": stat_dict["pearson"]["ci_hi"],
                "spearman_rho": stat_dict["spearman"]["rho"],
                "spearman_p": stat_dict["spearman"]["p"],
                "spearman_ci_lo": stat_dict["spearman"]["ci_lo"],
                "spearman_ci_hi": stat_dict["spearman"]["ci_hi"],
                "n": stat_dict["pearson"]["n"],
            }
            rows.append(row)
    return pd.DataFrame(rows)
