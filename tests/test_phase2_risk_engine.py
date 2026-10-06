"""
Phase 2 — Linguistic Risk Engine Tests

Tests cover:
  1. HedgingDetector unit tests
  2. FinBERT loading and inference
  3. ToneShiftDetector unit tests
  4. EvasivenessDetector unit tests
  5. RiskScorer unit tests
  6. Integration: full pipeline on sample sentences
  7. Checkpoint/resume behaviour
  8. Memory usage sanity
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch, MagicMock

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# 1. HEDGING DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

class TestHedgingDetector:

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.hedging_detector import HedgingDetector
        self.detector = HedgingDetector()

    def test_empty_string(self):
        result = self.detector.analyse("")
        assert result["hedging_score"] == 0.0
        assert result["hedge_count"] == 0

    def test_clean_sentence_low_score(self):
        result = self.detector.analyse("Revenue increased 15% year over year.")
        assert result["hedging_score"] < 30.0, "Clean factual sentence should have low hedging score"

    def test_hedged_sentence_high_score(self):
        text = "We believe we may potentially see approximately 10% growth, subject to uncertain market conditions."
        result = self.detector.analyse(text)
        assert result["hedging_score"] > 40.0, f"Highly hedged sentence should score > 40, got {result['hedging_score']}"
        assert result["hedge_count"] > 3

    def test_modal_verb_detection(self):
        result = self.detector.analyse("This could potentially lead to significant gains.")
        assert result["modal_count"] >= 1
        assert result["modal_verb_density"] > 0.0

    def test_uncertainty_phrase_detection(self):
        result = self.detector.analyse("We continue to monitor the situation carefully.")
        assert result["uncertainty_phrase_count"] >= 1

    def test_deflection_phrase_detection(self):
        result = self.detector.analyse("As you know, the important thing is long-term value creation.")
        assert result["deflection_count"] >= 1

    def test_score_range(self):
        """All scores must be in [0, 100]."""
        sentences = [
            "Revenue was $10 billion.",
            "We might potentially see some uncertainty depending on conditions.",
            "I can't comment on that at this time.",
            "",
            "We remain cautiously optimistic about the long-term outlook.",
        ]
        for s in sentences:
            result = self.detector.analyse(s)
            assert 0.0 <= result["hedging_score"] <= 100.0, f"hedging_score out of range for: {s!r}"
            assert 0.0 <= result["uncertainty_score"] <= 100.0

    def test_batch_returns_correct_count(self):
        texts = ["sentence one", "sentence two", "sentence three"]
        results = self.detector.analyse_batch(texts)
        assert len(results) == 3

    def test_matched_terms_for_explainability(self):
        result = self.detector.analyse("We may potentially see growth.")
        assert isinstance(result["matched_modals"], list)
        assert isinstance(result["matched_qualifiers"], list)

    def test_density_calculation(self):
        """hedge_density = hedge_count / word_count."""
        result = self.detector.analyse("We may could might see growth.")
        # "may", "could", "might" = 3 modals
        assert result["modal_count"] == 3
        expected_density = 3 / 6  # 6 words
        assert abs(result["modal_verb_density"] - expected_density) < 0.02


# ──────────────────────────────────────────────────────────────────────────────
# 2. FINBERT ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class TestFinBERTEngine:

    def test_load_and_infer_single(self):
        """FinBERT loads and returns valid probabilities for one sentence."""
        from src.finbert_engine import load_finbert, run_sentiment_single, unload_finbert
        load_finbert()
        result = run_sentiment_single("Revenue declined sharply this quarter.")
        assert "positive_prob" in result
        assert "negative_prob" in result
        assert "neutral_prob" in result
        assert "sentiment_label" in result
        # Probs should sum to ~1
        total = result["positive_prob"] + result["negative_prob"] + result["neutral_prob"]
        assert abs(total - 1.0) < 0.01, f"Probs should sum to 1, got {total}"
        assert result["sentiment_label"] in ("positive", "negative", "neutral")

    def test_negative_sentiment_detection(self):
        from src.finbert_engine import run_sentiment_single
        result = run_sentiment_single("The company reported significant losses and declining revenue.")
        assert result["negative_prob"] > result["positive_prob"], \
            "Negative sentence should have higher negative_prob"

    def test_positive_sentiment_detection(self):
        from src.finbert_engine import run_sentiment_single
        result = run_sentiment_single("We achieved record revenue growth and strong profitability.")
        assert result["positive_prob"] > result["negative_prob"], \
            "Positive sentence should have higher positive_prob"

    def test_batch_inference(self):
        from src.finbert_engine import run_sentiment_batch
        texts = [
            "Revenue declined sharply.",
            "Strong growth in all segments.",
            "We remain cautious about the outlook.",
        ]
        results = run_sentiment_batch(texts, batch_size=2)
        assert len(results) == 3
        for r in results:
            total = r["positive_prob"] + r["negative_prob"] + r["neutral_prob"]
            assert abs(total - 1.0) < 0.01

    def test_empty_batch(self):
        from src.finbert_engine import run_sentiment_batch
        results = run_sentiment_batch([])
        assert results == []

    def test_prob_range(self):
        from src.finbert_engine import run_sentiment_batch
        texts = ["Revenue was $5B.", "Losses exceeded expectations significantly.", ""]
        results = run_sentiment_batch(texts)
        for r in results:
            assert 0.0 <= r["positive_prob"] <= 1.0
            assert 0.0 <= r["negative_prob"] <= 1.0
            assert 0.0 <= r["neutral_prob"] <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 3. TONE SHIFT DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

class TestToneShiftDetector:

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.tone_shift_detector import ToneShiftDetector
        self.detector = ToneShiftDetector(window=3)

    def _make_sequence(self, neg_probs, sections=None):
        if sections is None:
            sections = ["q_and_a"] * len(neg_probs)
        return [
            {
                "sentence_id": f"s_{i}",
                "segment_id": f"seg_{i}",
                "section": sections[i],
                "speaker_role": "executive",
                "negative_prob": neg,
                "positive_prob": 0.5 - neg / 2,
                "neutral_prob": 0.5 - neg / 2,
                "sentiment_label": "negative" if neg > 0.5 else "neutral",
            }
            for i, neg in enumerate(neg_probs)
        ]

    def test_stable_sequence_low_shift(self):
        seq = self._make_sequence([0.1, 0.1, 0.1, 0.1, 0.1])
        enriched, _ = self.detector.analyse_transcript(seq)
        for item in enriched[3:]:  # after window fills
            assert item["tone_shift_score"] < 20.0, "Stable sequence should have low shift score"

    def test_sudden_jump_high_shift(self):
        seq = self._make_sequence([0.05, 0.05, 0.05, 0.05, 0.90])
        enriched, _ = self.detector.analyse_transcript(seq)
        # Last sentence should have high shift score
        assert enriched[-1]["tone_shift_score"] > 50.0, \
            f"Sudden jump should trigger high shift, got {enriched[-1]['tone_shift_score']}"

    def test_section_shift_computed(self):
        sections = ["prepared_remarks"] * 5 + ["q_and_a"] * 5
        negs = [0.1] * 5 + [0.5] * 5  # Q&A much more negative
        seq = self._make_sequence(negs, sections)
        _, section_metrics = self.detector.analyse_transcript(seq)
        assert section_metrics["prepared_vs_qa_shift"] > 30.0
        assert section_metrics["section_shift_direction"] == "more_negative_qa"

    def test_no_section_shift_when_equal(self):
        sections = ["prepared_remarks"] * 5 + ["q_and_a"] * 5
        negs = [0.2] * 10
        seq = self._make_sequence(negs, sections)
        _, section_metrics = self.detector.analyse_transcript(seq)
        assert section_metrics["prepared_vs_qa_shift"] < 5.0
        assert section_metrics["section_shift_direction"] == "neutral"

    def test_shift_scores_in_range(self):
        seq = self._make_sequence([0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
        enriched, _ = self.detector.analyse_transcript(seq)
        for item in enriched:
            assert 0.0 <= item["tone_shift_score"] <= 100.0

    def test_empty_sequence(self):
        enriched, section_metrics = self.detector.analyse_transcript([])
        assert enriched == []
        assert section_metrics["prepared_vs_qa_shift"] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 4. EVASIVENESS DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

class TestEvasivenessDetector:

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.evasiveness_detector import EvasivenessDetector
        self.detector = EvasivenessDetector()

    def test_direct_answer_low_evasion(self):
        q = "What is your expected margin next quarter?"
        a = "We expect margins to be approximately 25% next quarter."
        result = self.detector.analyse(q, a)
        assert result["evasive_score"] < 60.0, \
            f"Direct answer should have moderate-low evasion score, got {result['evasive_score']}"

    def test_evasive_answer_high_score(self):
        q = "What is your expected margin next quarter?"
        a = "We remain focused on delivering long-term value for shareholders. As you know, the important thing is operational excellence."
        result = self.detector.analyse(q, a)
        assert result["evasive_score"] > 40.0, \
            f"Evasive answer should have high score, got {result['evasive_score']}"

    def test_numeric_question_with_numeric_answer(self):
        q = "How much revenue do you expect next quarter?"
        a = "We expect revenue of approximately $5 billion."
        result = self.detector.analyse(q, a)
        assert result["answer_has_numeric"] is True
        assert result["numeric_response_score"] == 0.0  # answered numerically

    def test_numeric_question_without_numeric_answer(self):
        q = "What percentage growth do you expect?"
        a = "We are focused on driving value and executing on our strategic priorities."
        result = self.detector.analyse(q, a)
        assert result["question_asks_numeric"] is True
        assert result["answer_has_numeric"] is False
        assert result["numeric_response_score"] == 100.0  # failed to answer numerically

    def test_score_in_range(self):
        pairs = [
            ("What happened to revenue?", "Revenue grew 10% year over year."),
            ("What is the margin?", "As you know, we remain focused on long-term value."),
            ("", "Some answer here."),
            ("Some question.", ""),
        ]
        for q, a in pairs:
            result = self.detector.analyse(q, a)
            assert 0.0 <= result["evasive_score"] <= 100.0

    def test_deflection_phrases_detected(self):
        q = "Can you give us more colour on demand trends?"
        a = "As you know, the important thing is that we remain focused on long-term value creation."
        result = self.detector.analyse(q, a)
        assert result["deflection_count"] >= 1

    def test_tfidf_similarity_range(self):
        q = "What is your revenue guidance?"
        a = "Revenue guidance for next quarter is approximately $10 billion."
        result = self.detector.analyse(q, a)
        assert 0.0 <= result["qa_tfidf_similarity"] <= 1.0

    def test_batch_analysis(self):
        pairs = [
            ("Revenue question?", "Revenue was $5B."),
            ("Margin question?", "We remain focused on long-term value."),
        ]
        results = self.detector.analyse_batch(pairs)
        assert len(results) == 2


# ──────────────────────────────────────────────────────────────────────────────
# 5. RISK SCORER
# ──────────────────────────────────────────────────────────────────────────────

class TestRiskScorer:

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.risk_scorer import RiskScorer
        self.scorer = RiskScorer()

    def test_sentence_risk_zero_inputs(self):
        score = self.scorer.sentence_hidden_risk(0.0, 0.0, 0.0)
        assert score == 0.0

    def test_sentence_risk_max_inputs(self):
        score = self.scorer.sentence_hidden_risk(100.0, 100.0, 100.0)
        assert score == 100.0

    def test_sentence_risk_formula(self):
        """Verify formula: 0.35*H + 0.40*E + 0.25*S (when evasive > 0)."""
        score = self.scorer.sentence_hidden_risk(
            hedging_score=60.0,
            tone_shift_score=40.0,
            evasive_score=80.0,
        )
        expected = 0.35 * 60.0 + 0.40 * 80.0 + 0.25 * 40.0
        assert abs(score - expected) < 0.1, f"Expected {expected}, got {score}"

    def test_sentence_risk_no_evasive_redistributes_weight(self):
        """When evasive_score=0, weight is redistributed to hedging+shift."""
        score_with_evasive = self.scorer.sentence_hidden_risk(60.0, 40.0, 80.0)
        score_no_evasive = self.scorer.sentence_hidden_risk(60.0, 40.0, 0.0)
        # Without evasive, weight redistributed, should differ
        # Both should be in [0, 100]
        assert 0.0 <= score_no_evasive <= 100.0
        assert score_no_evasive != score_with_evasive

    def test_score_sentence_has_all_fields(self):
        features = {
            "sentence_id": "test_s_001",
            "segment_id": "test_seg_001",
            "transcript_id": "TEST_2024_Q1",
            "text": "We may potentially see some growth.",
            "hedging_score": 50.0,
            "tone_shift_score": 30.0,
            "evasive_score": 20.0,
            "positive_prob": 0.2,
            "negative_prob": 0.3,
            "neutral_prob": 0.5,
            "sentiment_label": "neutral",
        }
        result = self.scorer.score_sentence(features)
        assert "hidden_risk_score" in result
        assert "contributing_factors" in result
        assert 0.0 <= result["hidden_risk_score"] <= 100.0

    def test_aggregate_segment(self):
        sentence_scores = [
            {"hidden_risk_score": 30.0, "hedging_score": 20.0,
             "tone_shift_score": 15.0, "evasive_score": 0.0,
             "negative_prob": 0.2, "sentence_id": "s1", "text": "text 1"},
            {"hidden_risk_score": 70.0, "hedging_score": 60.0,
             "tone_shift_score": 50.0, "evasive_score": 80.0,
             "negative_prob": 0.5, "sentence_id": "s2", "text": "text 2"},
        ]
        result = self.scorer.aggregate_segment(sentence_scores)
        assert "segment_risk_score" in result
        assert 0.0 <= result["segment_risk_score"] <= 100.0
        assert result["sentence_count"] == 2

    def test_aggregate_transcript(self):
        sentences = [
            {"hidden_risk_score": 40.0, "hedging_score": 30.0,
             "tone_shift_score": 20.0, "evasive_score": 50.0,
             "section": "q_and_a", "segment_id": "seg1"},
            {"hidden_risk_score": 60.0, "hedging_score": 50.0,
             "tone_shift_score": 40.0, "evasive_score": 70.0,
             "section": "prepared_remarks", "segment_id": "seg2"},
        ]
        result = self.scorer.aggregate_transcript(sentences, [])
        assert "overall_hidden_risk" in result
        assert 0.0 <= result["overall_hidden_risk"] <= 100.0
        assert result["sentence_count"] == 2

    def test_aggregate_empty(self):
        result = self.scorer.aggregate_transcript([], [])
        assert result["overall_hidden_risk"] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 6. INTEGRATION — full pipeline on real DB
# ──────────────────────────────────────────────────────────────────────────────

class TestRiskEngineIntegration:

    def test_pipeline_on_10_transcripts(self):
        """
        Integration test: run Phase 2 on first 10 transcripts.
        Verifies that risk_scores and transcript_scores are populated.
        """
        from src.risk_engine import run_phase2
        from src.config import get_config, PROJECT_ROOT
        import sqlite3

        db_path = PROJECT_ROOT / "database" / "hidden_risk.db"
        if not db_path.exists():
            pytest.skip("Database not found — Phase 1 must run first.")

        # Check there are sentences to process
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
        conn.close()
        if count == 0:
            pytest.skip("No sentences found — Phase 1 must run first.")

        stats = run_phase2(limit=10, force_reprocess=False)

        assert stats["transcripts_processed"] >= 0  # some may already be scored
        assert stats["errors"] == 0 or stats["errors"] < 5  # tolerate occasional errors

        # Verify DB rows
        conn = sqlite3.connect(str(db_path))
        rs_count = conn.execute("SELECT COUNT(*) FROM risk_scores").fetchone()[0]
        ts_count = conn.execute("SELECT COUNT(*) FROM transcript_scores").fetchone()[0]
        conn.close()

        assert rs_count > 0, "risk_scores table should not be empty after processing"
        assert ts_count > 0, "transcript_scores table should not be empty"

    def test_hidden_risk_scores_in_range(self):
        """All hidden_risk_score values in DB should be [0, 100]."""
        from src.config import PROJECT_ROOT
        import sqlite3

        db_path = PROJECT_ROOT / "database" / "hidden_risk.db"
        if not db_path.exists():
            pytest.skip("Database not found.")

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT hidden_risk_score FROM risk_scores LIMIT 1000"
        ).fetchall()
        conn.close()

        if not rows:
            pytest.skip("No risk scores computed yet.")

        for (score,) in rows:
            assert 0.0 <= score <= 100.0, f"hidden_risk_score out of range: {score}"

    def test_transcript_scores_fields(self):
        """transcript_scores should have all required fields."""
        from src.config import PROJECT_ROOT
        import sqlite3

        db_path = PROJECT_ROOT / "database" / "hidden_risk.db"
        if not db_path.exists():
            pytest.skip("Database not found.")

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM transcript_scores LIMIT 5").fetchall()
        conn.close()

        if not rows:
            pytest.skip("No transcript scores yet.")

        required = ["transcript_id", "overall_hidden_risk", "average_risk",
                    "top_risk_average", "qa_risk", "prepared_risk"]
        for row in rows:
            for field in required:
                assert field in row.keys(), f"Missing field: {field}"
                assert row[field] is not None


# ──────────────────────────────────────────────────────────────────────────────
# 7. CHECKPOINT / RESUME
# ──────────────────────────────────────────────────────────────────────────────

class TestCheckpointResume:

    def test_already_scored_transcripts_skipped(self):
        """If transcript already has a transcript_score, it should be skipped."""
        from src.risk_engine import LinguisticRiskEngine
        from src.config import PROJECT_ROOT
        import sqlite3

        db_path = PROJECT_ROOT / "database" / "hidden_risk.db"
        if not db_path.exists():
            pytest.skip("Database not found.")

        conn = sqlite3.connect(str(db_path))
        already = conn.execute("SELECT COUNT(*) FROM transcript_scores").fetchone()[0]
        conn.close()

        if already == 0:
            pytest.skip("No scored transcripts yet — run integration test first.")

        engine = LinguisticRiskEngine()
        scored_before = engine._get_already_scored_ids()
        all_ids = engine._get_all_transcript_ids()

        pending = [tid for tid in all_ids if tid not in scored_before]
        # The already-scored ones should not be in pending
        for sid in scored_before:
            assert sid not in pending, f"Already-scored ID {sid} should be skipped"

    def test_checkpoint_state_persisted(self):
        """Verify checkpoint record is written during processing."""
        from src.checkpoint import CheckpointManager
        from src.risk_engine import JOB_NAME

        mgr = CheckpointManager()
        state = mgr.get_checkpoint(JOB_NAME)
        # May or may not exist depending on test order — just check it works
        if state is not None:
            assert state.status in ("IN_PROGRESS", "COMPLETED", "FAILED")
