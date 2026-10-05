"""
Phase 5 — Research Validation Main Script

Usage:
    python scripts/run_phase5.py [--limit N] [--seed S] [--skip-annotation]

Options:
    --limit N           Limit transcripts analyzed to first N (default: all available)
    --seed S            Random seed (default: 42)
    --skip-annotation   Skip annotation export/generation step
    --annotation-csv P  Path to completed human annotation CSV (if available)

Outputs (all written to outputs/):
    research_results.csv     — transcript-level risk + market + outcome data
    evaluation_metrics.csv   — precision/recall/F1 per baseline and proposed system
    correlation_results.csv  — Pearson/Spearman correlations with 95% CI
    verification_results.csv — claim verification status breakdown

Annotation export (written to data/annotations/):
    annotation_export_<timestamp>.csv  — QA pairs ready for human labeling

Anti-leakage enforcement:
  - Train: date <= 2018-12-31
  - Val:   2019-01-01 to 2021-12-31
  - Test:  date >= 2022-01-01
  - Market outcomes used ONLY for post-hoc correlation, never for risk scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.validation.annotation import (
    build_synthetic_annotations,
    export_annotation_csv,
    import_annotation_csv,
    sample_qa_for_annotation,
)
from src.validation.baselines import apply_time_split, load_baseline_data
from src.validation.metrics import evaluate_baselines_vs_annotations
from src.validation.statistics import (
    build_correlation_table,
    descriptive_statistics,
    guidance_outcome_analysis,
    market_correlation_analysis,
    regression_analysis,
    volatility_analysis,
)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# -- Verification stats --------------------------------------------------------

def load_verification_stats() -> pd.DataFrame:
    """Load verification status breakdown from DB."""
    from src.database import get_db_manager
    db_path = PROJECT_ROOT / "database" / "hidden_risk.db"
    mgr = get_db_manager(db_path)
    if not db_path.exists():
        mgr.init_database()
    conn = mgr.get_connection()
    try:
        df = pd.read_sql_query("""
            SELECT
                v.status,
                COUNT(*) AS n_claims,
                ROUND(AVG(v.confidence), 4) AS avg_confidence,
                t.ticker,
                t.date
            FROM verification v
            JOIN claims c ON v.claim_id = c.claim_id
            JOIN transcripts t ON c.transcript_id = t.transcript_id
            GROUP BY t.ticker, t.date, v.status
            ORDER BY t.date, t.ticker
        """, conn)
    finally:
        conn.close()
    return df


def load_verification_summary() -> pd.DataFrame:
    """Aggregated verification summary."""
    from src.database import get_db_manager
    db_path = PROJECT_ROOT / "database" / "hidden_risk.db"
    mgr = get_db_manager(db_path)
    if not db_path.exists():
        mgr.init_database()
    conn = mgr.get_connection()
    try:
        df = pd.read_sql_query("""
            SELECT
                v.status,
                COUNT(*) AS n_claims,
                ROUND(AVG(v.confidence), 4) AS avg_confidence,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_total
            FROM verification v
            GROUP BY v.status
            ORDER BY n_claims DESC
        """, conn)
    finally:
        conn.close()
    return df


# -- Main ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5 — Research Validation")
    p.add_argument("--limit", type=int, default=None,
                   help="Max transcripts to analyze (default: all available)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-annotation", action="store_true")
    p.add_argument("--annotation-csv", type=str, default=None,
                   help="Path to completed human annotation CSV")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    print("=" * 65)
    print("PHASE 5 — RESEARCH VALIDATION")
    print("=" * 65)
    print(f"  Limit       : {args.limit or 'ALL'}")
    print(f"  Seed        : {args.seed}")
    print(f"  Time split  : train <=2018 | val 2019-2021 | test >=2022")
    print("=" * 65)

    # -- 1. Load all transcript-level data ----------------------------------
    print("\n[1/7] Loading transcript-level data from database...")
    df = load_baseline_data()
    if args.limit:
        df = df.head(args.limit)
    df = apply_time_split(df)

    print(f"      Transcripts loaded: {len(df)}")
    print(f"      Split: {df['split'].value_counts().to_dict()}")
    print(f"      Date range: {df['date'].min()[:10]} ? {df['date'].max()[:10]}")

    # -- 2. Export research results CSV ------------------------------------
    print("\n[2/7] Writing outputs/research_results.csv...")
    research_cols = [
        "transcript_id", "ticker", "date", "year", "quarter", "split",
        "overall_hidden_risk", "average_risk", "hedging_avg",
        "evasiveness_avg", "tone_shift_avg", "qa_risk", "prepared_risk",
        "lm_lexicon_score", "finbert_only_score", "hedging_only_score",
        "return_1d", "return_5d", "return_10d", "return_20d",
        "volatility_5d", "volatility_10d", "abnormal_return_5d",
        "actual_revenue", "earnings_surprise", "guidance_change",
    ]
    res_df = df[[c for c in research_cols if c in df.columns]]
    res_df.to_csv(OUTPUTS_DIR / "research_results.csv", index=False)
    print(f"      Written {len(res_df)} rows.")

    # -- 3. Annotation workflow ---------------------------------------------
    print("\n[3/7] Annotation workflow...")
    annotated_data: list = []

    if args.annotation_csv:
        print(f"      Loading human annotations from: {args.annotation_csv}")
        annotated_data = import_annotation_csv(Path(args.annotation_csv))
    else:
        print("      No human annotation CSV provided.")
        print("      Sampling QA pairs and generating annotation export...")
        sampled_qa = sample_qa_for_annotation(n=500, seed=args.seed)
        print(f"      Sampled {len(sampled_qa)} substantive Q&A pairs.")

        if not args.skip_annotation:
            export_path = export_annotation_csv(sampled_qa)
            print(f"      Annotation template written to: {export_path}")
            print("      NOTICE: No real human labels available.")
            print("              Using synthetic labels as evaluation proxy.")
            print("              (Clearly documented in report; not true ground truth.)")

        annotated_data = build_synthetic_annotations(sampled_qa)

    print(f"      Annotation records ready: {len(annotated_data)}")

    # -- 4. Classification metrics -----------------------------------------
    print("\n[4/7] Computing classification metrics (P/R/F1, confusion matrix)...")
    eval_results = evaluate_baselines_vs_annotations(annotated_data)

    # Flatten to DataFrame
    eval_rows = []
    for system, metrics in eval_results.items():
        row = {
            "system": system,
            "n_samples": metrics.get("n_samples"),
            "accuracy": metrics.get("accuracy"),
            "macro_precision": metrics.get("macro_precision"),
            "macro_recall": metrics.get("macro_recall"),
            "macro_f1": metrics.get("macro_f1"),
            "ci_f1_macro_95_lo": metrics.get("ci_f1_macro_95_lo"),
            "ci_f1_macro_95_hi": metrics.get("ci_f1_macro_95_hi"),
            "confusion_matrix": str(metrics.get("confusion_matrix")),
            "per_class_f1": str(metrics.get("per_class_f1")),
            "note": metrics.get("note", ""),
        }
        eval_rows.append(row)
        print(
            f"      {system:40s} | F1={row['macro_f1']:.3f} "
            f"[{row['ci_f1_macro_95_lo']:.3f}, {row['ci_f1_macro_95_hi']:.3f}]"
        )

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(OUTPUTS_DIR / "evaluation_metrics.csv", index=False)
    print(f"      Written outputs/evaluation_metrics.csv")

    # -- 5. Market correlation analysis ------------------------------------
    print("\n[5/7] Market correlation analysis (Pearson + Spearman)...")
    corr_results = market_correlation_analysis(df)
    corr_df = build_correlation_table(corr_results)
    corr_df.to_csv(OUTPUTS_DIR / "correlation_results.csv", index=False)
    print(f"      Written {len(corr_df)} correlation rows.")

    # Print key correlations for proposed system
    prop_corr = corr_df[corr_df["system"] == "Proposed: Hidden Risk Score"]
    if not prop_corr.empty:
        print("\n      Proposed system correlations (risk ? market):")
        for _, row in prop_corr.iterrows():
            r = row['pearson_r']
            p = row['pearson_p']
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
            print(f"        {row['market_metric']:30s} r={r:.3f} p={p:.3f} {sig} n={int(row['n'])}")

    # -- 6. Verification results CSV ---------------------------------------
    print("\n[6/7] Exporting verification_results.csv...")
    verif_df = load_verification_stats()
    verif_df.to_csv(OUTPUTS_DIR / "verification_results.csv", index=False)
    verif_summary = load_verification_summary()
    print("      Verification summary:")
    for _, row in verif_summary.iterrows():
        print(f"        {row['status']:22s} {row['n_claims']:5d} ({row['pct_total']:.1f}%)")

    # -- 7. Statistical reporting ------------------------------------------
    print("\n[7/7] Statistical reporting...")

    desc = descriptive_statistics(df)
    vol = volatility_analysis(df)
    outcome = guidance_outcome_analysis(df)
    reg = regression_analysis(df)

    # Print key descriptives
    risk_dist = desc["score_distributions"].get("overall_hidden_risk", {})
    if risk_dist:
        print(f"\n      Hidden Risk Score: mean={risk_dist['mean']:.1f}  "
              f"median={risk_dist['median']:.1f}  std={risk_dist['std']:.1f}")
    qa_vs_prep = desc.get("qa_vs_prepared", {})
    if qa_vs_prep:
        print(f"      Q&A risk: mean={qa_vs_prep.get('qa_mean', 0):.1f}  "
              f"Prepared: mean={qa_vs_prep.get('prepared_mean', 0):.1f}  "
              f"p={qa_vs_prep.get('p_value', 1):.3f}")

    for vcol, vstats in vol.items():
        print(f"      Volatility [{vcol}]: high-risk mean={vstats['high_risk_mean']:.4f}  "
              f"low-risk mean={vstats['low_risk_mean']:.4f}  p={vstats['p_value']:.3f}")

    # Save full stats as JSON for report generation
    stats_dump = {
        "descriptive": desc,
        "volatility": vol,
        "guidance_outcome": outcome,
        "regression": reg,
    }
    stats_path = OUTPUTS_DIR / "phase5_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_dump, f, indent=2, default=str)
    print(f"\n      Full stats written to outputs/phase5_stats.json")

    # -- Summary -----------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'=' * 65}")
    print("PHASE 5 COMPLETE")
    print(f"{'=' * 65}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Outputs:")
    for f in ["research_results.csv", "evaluation_metrics.csv",
              "correlation_results.csv", "verification_results.csv",
              "phase5_stats.json"]:
        fp = OUTPUTS_DIR / f
        size = fp.stat().st_size if fp.exists() else 0
        print(f"    outputs/{f:40s} {size:>8,} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
