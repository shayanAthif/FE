"""
Phase 5 test suite — Research Validation.

Tests:
  1. Annotation workflow (sample, export, import, synthetic)
  2. Baseline models (LM lexicon, data loading, time splits)
  3. Classification metrics (P/R/F1, confusion matrix, bootstrap CI)
  4. Statistical analysis (correlations, regression, volatility, guidance)
  5. Integration (full Phase 5 pipeline on real DB)
"""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.validation.annotation import (
    build_synthetic_annotations,
    export_annotation_csv,
    import_annotation_csv,
    sample_qa_for_annotation,
    score_to_label,
    EVASIVE_THRESHOLDS,
)
from src.validation.baselines import (
    apply_time_split,
    load_baseline_data,
    lm_negative_score,
    compute_baseline1_scores,
)
from src.validation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    evaluate_baselines_vs_annotations,
    scores_to_labels,
)
from src.validation.statistics import (
    build_correlation_table,
    descriptive_statistics,
    guidance_outcome_analysis,
    market_correlation_analysis,
    regression_analysis,
    volatility_analysis,
)


# ── Annotation ────────────────────────────────────────────────────────────────

class TestAnnotation:

    def test_score_to_label_boundaries(self):
        lo, hi = EVASIVE_THRESHOLDS
        assert score_to_label(0.0) == 0
        assert score_to_label(lo - 0.1) == 0
        assert score_to_label(lo) == 1
        assert score_to_label((lo + hi) / 2) == 1
        assert score_to_label(hi) == 2
        assert score_to_label(100.0) == 2

    def test_sample_qa_for_annotation_returns_list(self):
        rows = sample_qa_for_annotation(n=50)
        assert isinstance(rows, list)
        assert len(rows) <= 50
        if rows:
            assert "qa_id" in rows[0]
            assert "question_text" in rows[0]
            assert "answer_text" in rows[0]

    def test_sample_qa_reproducible(self):
        r1 = sample_qa_for_annotation(n=20, seed=1)
        r2 = sample_qa_for_annotation(n=20, seed=1)
        ids1 = [r["qa_id"] for r in r1]
        ids2 = [r["qa_id"] for r in r2]
        assert ids1 == ids2

    def test_export_import_roundtrip(self, tmp_path):
        rows = sample_qa_for_annotation(n=10)
        annotated = build_synthetic_annotations(rows)

        export_path = tmp_path / "test_annotation.csv"
        export_annotation_csv(annotated, output_path=export_path)

        assert export_path.exists()
        imported = import_annotation_csv(export_path)
        assert len(imported) == len(annotated)
        assert all("label_evasive" in r for r in imported)

    def test_synthetic_annotations_valid_labels(self):
        rows = sample_qa_for_annotation(n=30)
        annotated = build_synthetic_annotations(rows)
        for record in annotated:
            assert record["label_evasive"] in (0, 1, 2)
            assert record["label_hedging"] in (0, 1)

    def test_import_skips_unannotated_rows(self, tmp_path):
        csv_path = tmp_path / "partial.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "qa_id", "transcript_id", "label_evasive",
                "auto_evasiveness_score", "auto_hedging_score", "auto_hidden_risk_score",
                "label_hedging", "label_topic_avoidance", "label_perceived_uncertainty",
                "annotator_notes",
            ])
            writer.writeheader()
            writer.writerow({"qa_id": "q1", "transcript_id": "t1", "label_evasive": "2",
                             "auto_evasiveness_score": "60.0", "auto_hedging_score": "30.0",
                             "auto_hidden_risk_score": "55.0"})
            writer.writerow({"qa_id": "q2", "transcript_id": "t1", "label_evasive": "",  # blank
                             "auto_evasiveness_score": "10.0"})

        imported = import_annotation_csv(csv_path)
        assert len(imported) == 1
        assert imported[0]["qa_id"] == "q1"


# ── Baselines ─────────────────────────────────────────────────────────────────

class TestBaselines:

    def test_lm_negative_score_empty(self):
        assert lm_negative_score("") == 0.0

    def test_lm_negative_score_neutral(self):
        score = lm_negative_score("The company reported quarterly results today.")
        assert 0.0 <= score <= 100.0

    def test_lm_negative_score_high_for_negative_text(self):
        neg_text = "bankruptcy decline failure loss adverse risk warning poor weak"
        score = lm_negative_score(neg_text)
        assert score > 30.0, f"Expected high negative score, got {score}"

    def test_lm_negative_score_low_for_positive_text(self):
        pos_text = "We grew revenue significantly and beat analyst expectations comfortably."
        score = lm_negative_score(pos_text)
        assert score < 30.0, f"Expected low negative score, got {score}"

    def test_compute_baseline1_scores(self):
        df = pd.DataFrame({
            "sentence_id": ["s1", "s2"],
            "text": ["Strong growth and record revenue.", "Bankruptcy risk failure decline."]
        })
        scores = compute_baseline1_scores(df)
        assert len(scores) == 2
        assert scores["s2"] > scores["s1"]

    def test_load_baseline_data_structure(self):
        df = load_baseline_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        required = ["transcript_id", "ticker", "date", "overall_hidden_risk"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_apply_time_split(self):
        df = pd.DataFrame({
            "date": ["2015-01-01", "2020-06-15", "2023-03-01"],
            "overall_hidden_risk": [30.0, 45.0, 60.0],
        })
        result = apply_time_split(df)
        assert result["split"].tolist() == ["train", "val", "test"]

    def test_apply_time_split_no_leakage(self):
        # Test dates: split boundaries must be strictly chronological
        df = load_baseline_data()
        df = apply_time_split(df)
        train = pd.to_datetime(df[df["split"] == "train"]["date"])
        val = pd.to_datetime(df[df["split"] == "val"]["date"])
        if not train.empty and not val.empty:
            assert train.max() < val.min() or val.empty, "Data leakage: train and val overlap"


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_confusion_matrix_perfect(self):
        y = [0, 1, 2, 0, 1, 2]
        cm = compute_confusion_matrix(y, y)
        assert cm[0, 0] == 2
        assert cm[1, 1] == 2
        assert cm[2, 2] == 2
        assert np.sum(cm) - np.trace(cm) == 0  # no off-diagonal

    def test_confusion_matrix_all_wrong(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [2, 2, 0, 0, 1, 1]
        cm = compute_confusion_matrix(y_true, y_pred)
        assert np.trace(cm) == 0

    def test_compute_classification_metrics_output_keys(self):
        y = [0, 0, 1, 1, 2, 2, 0, 1, 2]
        result = compute_classification_metrics(y, y)
        assert "macro_f1" in result
        assert "confusion_matrix" in result
        assert "ci_f1_macro_95_lo" in result
        assert result["macro_f1"] == pytest.approx(1.0, abs=1e-6)

    def test_scores_to_labels(self):
        # pd.cut bins are (lo, hi] with right=True.
        # thresholds=(20, 50) → bins: (-0.001, 20] → 0, (20, 50] → 1, (50, 100.001] → 2
        scores = pd.Series([0.0, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0])
        labels = scores_to_labels(scores)
        assert labels.tolist() == [0, 0, 0, 1, 1, 2, 2]

    def test_evaluate_baselines_returns_all_systems(self):
        rows = sample_qa_for_annotation(n=50)
        annotated = build_synthetic_annotations(rows)
        results = evaluate_baselines_vs_annotations(annotated)
        expected_keys = [
            "baseline_1_lm_lexicon",
            "baseline_2_finbert_only",
            "baseline_3_hedging_only",
            "proposed_system",
        ]
        for k in expected_keys:
            assert k in results, f"Missing key: {k}"
            assert "macro_f1" in results[k]

    def test_bootstrap_ci_bounds(self):
        y = [0] * 30 + [1] * 20 + [2] * 10
        pred = [0] * 25 + [1] * 20 + [2] * 15
        result = compute_classification_metrics(y, pred)
        lo = result["ci_f1_macro_95_lo"]
        hi = result["ci_f1_macro_95_hi"]
        f1 = result["macro_f1"]
        assert lo <= f1 <= hi


# ── Statistics ────────────────────────────────────────────────────────────────

class TestStatistics:

    @pytest.fixture
    def sample_df(self):
        df = load_baseline_data()
        return apply_time_split(df)

    def test_market_correlation_analysis_structure(self, sample_df):
        result = market_correlation_analysis(sample_df)
        assert isinstance(result, dict)
        assert len(result) >= 1
        for system, mkt_dict in result.items():
            for metric, stat in mkt_dict.items():
                assert "pearson" in stat
                assert "spearman" in stat
                assert "r" in stat["pearson"]
                assert "rho" in stat["spearman"]

    def test_correlation_values_in_range(self, sample_df):
        result = market_correlation_analysis(sample_df)
        for system, mkt_dict in result.items():
            for metric, stat in mkt_dict.items():
                r = stat["pearson"]["r"]
                rho = stat["spearman"]["rho"]
                if not np.isnan(r):
                    assert -1.0 <= r <= 1.0
                if not np.isnan(rho):
                    assert -1.0 <= rho <= 1.0

    def test_build_correlation_table_output(self, sample_df):
        corr = market_correlation_analysis(sample_df)
        df_corr = build_correlation_table(corr)
        assert isinstance(df_corr, pd.DataFrame)
        assert "pearson_r" in df_corr.columns
        assert "spearman_rho" in df_corr.columns
        assert len(df_corr) > 0

    def test_volatility_analysis_structure(self, sample_df):
        result = volatility_analysis(sample_df)
        for col, stats in result.items():
            assert "high_risk_mean" in stats
            assert "low_risk_mean" in stats
            assert "p_value" in stats
            assert 0.0 <= stats["p_value"] <= 1.0

    def test_descriptive_statistics_has_risk_distribution(self, sample_df):
        result = descriptive_statistics(sample_df)
        assert "n_transcripts" in result
        assert "score_distributions" in result
        assert "overall_hidden_risk" in result["score_distributions"]
        dist = result["score_distributions"]["overall_hidden_risk"]
        assert dist["mean"] >= 0.0

    def test_regression_analysis_runs(self, sample_df):
        result = regression_analysis(sample_df)
        assert isinstance(result, dict)
        for rcol, mkt_dict in result.items():
            for mcol, stats in mkt_dict.items():
                if "error" not in stats:
                    assert "r_squared" in stats
                    assert 0.0 <= stats["r_squared"] <= 1.0 or np.isnan(stats["r_squared"])

    def test_guidance_outcome_analysis_runs(self, sample_df):
        result = guidance_outcome_analysis(sample_df)
        assert isinstance(result, dict)
        # May have empty results if outcome data is sparse


# ── Integration ───────────────────────────────────────────────────────────────

class TestPhase5Integration:

    def test_full_pipeline_produces_outputs(self, tmp_path, monkeypatch):
        """Smoke test: run Phase 5 main script with small limit."""
        import subprocess
        result = subprocess.run(
            ["python", "scripts/run_phase5.py", "--limit", "20", "--skip-annotation"],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Phase 5 script failed:\n{result.stderr}"

        out_dir = Path(__file__).resolve().parent.parent / "outputs"
        for fname in [
            "research_results.csv",
            "evaluation_metrics.csv",
            "correlation_results.csv",
            "verification_results.csv",
            "phase5_stats.json",
        ]:
            assert (out_dir / fname).exists(), f"Missing output: {fname}"

    def test_research_results_csv_no_leakage(self):
        """Verify train transcripts predate val transcripts."""
        out_path = Path(__file__).resolve().parent.parent / "outputs" / "research_results.csv"
        if not out_path.exists():
            pytest.skip("research_results.csv not yet generated")
        df = pd.read_csv(out_path, parse_dates=["date"])
        train = df[df["split"] == "train"]["date"]
        val = df[df["split"] == "val"]["date"]
        if not train.empty and not val.empty:
            assert train.max() <= pd.Timestamp("2018-12-31"), "Train split date leakage"
            assert val.min() >= pd.Timestamp("2019-01-01"), "Val split date leakage"
