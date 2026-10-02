"""
Tone Shift Detector — Phase 2

Detects abrupt changes in financial sentiment across a transcript.

Two detection modes:
  A. LOCAL tone shift  — sentence vs. rolling window of preceding N sentences
  B. SECTION tone shift — prepared_remarks vs. q_and_a section averages

Inputs: pre-computed FinBERT probabilities per sentence.
Output per sentence:
  - local_sentiment_mean      : rolling mean of negative_prob over window
  - local_sentiment_std       : rolling std  of negative_prob over window
  - tone_shift_score          : 0-100 score for abrupt local change
  - prepared_vs_qa_shift      : absolute difference of section means (0-100)
  - section_sentiment_label   : 'prepared_remarks' or 'q_and_a' or None

Design notes:
  - Only uses negative_prob as the primary tone signal (most informative for risk)
  - Window size configurable via config.risk.local_tone_window (default 5)
  - Does NOT require or use any post-call data
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from src.config import get_config
from src.logger import get_logger

logger = get_logger("tone_shift_detector")


class ToneShiftDetector:
    """
    Compute tone-shift scores from pre-computed FinBERT sentiment sequences.
    """

    def __init__(self, window: Optional[int] = None):
        cfg = get_config()
        self.window = window or cfg.risk.local_tone_window

    # ─────────────────────────────────────────────────────────────────────
    # Helper
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _safe_std(values: List[float], mean: float) -> float:
        if len(values) < 2:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)

    # ─────────────────────────────────────────────────────────────────────
    # Local tone shift
    # ─────────────────────────────────────────────────────────────────────

    def compute_local_shifts(
        self, sentiment_sequence: List[Dict]
    ) -> List[Dict]:
        """
        Compute local tone shift for each item in sentiment_sequence.

        Parameters
        ----------
        sentiment_sequence : list of dicts, each containing:
            {
              'sentence_id': str,
              'segment_id': str,
              'section': str,           # 'prepared_remarks' | 'q_and_a' | ...
              'speaker_role': str,
              'negative_prob': float,
              'positive_prob': float,
              'neutral_prob': float,
              'sentiment_label': str,
            }

        Returns
        -------
        Same list, each item enriched with:
            local_sentiment_mean, local_sentiment_std,
            tone_shift_score, previous_sentiment
        """
        n = len(sentiment_sequence)
        results = []

        neg_probs = [s.get("negative_prob", 0.0) for s in sentiment_sequence]

        for i, sent in enumerate(sentiment_sequence):
            # Build window of previous sentences (exclusive of current)
            window_start = max(0, i - self.window)
            window_probs = neg_probs[window_start:i]

            if not window_probs:
                local_mean = 0.0
                local_std = 0.0
                prev_sentiment = 0.0
                shift_score = 0.0
            else:
                local_mean = self._safe_mean(window_probs)
                local_std = self._safe_std(window_probs, local_mean)
                prev_sentiment = neg_probs[i - 1] if i > 0 else 0.0

                current_neg = neg_probs[i]
                delta = abs(current_neg - local_mean)

                # Normalise: delta / (std + epsilon)  —  clipped at 3 sigma
                z_score = delta / (local_std + 1e-6)
                shift_score = round(min(z_score / 3.0, 1.0) * 100, 2)

            result = {
                **sent,
                "local_sentiment_mean": round(local_mean, 4),
                "local_sentiment_std": round(local_std, 4),
                "previous_sentiment": round(prev_sentiment, 4),
                "tone_shift_score": shift_score,
            }
            results.append(result)

        return results

    # ─────────────────────────────────────────────────────────────────────
    # Section (prepared vs Q&A) tone shift
    # ─────────────────────────────────────────────────────────────────────

    def compute_section_shift(
        self, sentiment_sequence: List[Dict]
    ) -> Dict:
        """
        Compare average negative_prob of prepared_remarks vs q_and_a sections.

        Returns a dict with:
          prepared_neg_mean  : mean negative prob in prepared remarks
          qa_neg_mean        : mean negative prob in Q&A
          prepared_vs_qa_shift : abs(qa_neg_mean - prepared_neg_mean) * 100  [0-100]
          section_shift_direction: 'more_negative_qa' | 'more_negative_prepared' | 'neutral'
        """
        prepared_negs: List[float] = []
        qa_negs: List[float] = []

        for s in sentiment_sequence:
            section = s.get("section", "")
            neg = s.get("negative_prob", 0.0)
            if section == "prepared_remarks":
                prepared_negs.append(neg)
            elif section == "q_and_a":
                qa_negs.append(neg)

        prepared_mean = self._safe_mean(prepared_negs)
        qa_mean = self._safe_mean(qa_negs)

        shift = abs(qa_mean - prepared_mean) * 100

        if qa_mean > prepared_mean + 0.02:
            direction = "more_negative_qa"
        elif prepared_mean > qa_mean + 0.02:
            direction = "more_negative_prepared"
        else:
            direction = "neutral"

        return {
            "prepared_neg_mean": round(prepared_mean, 4),
            "qa_neg_mean": round(qa_mean, 4),
            "prepared_vs_qa_shift": round(shift, 2),
            "section_shift_direction": direction,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Combined entry point
    # ─────────────────────────────────────────────────────────────────────

    def analyse_transcript(
        self, sentiment_sequence: List[Dict]
    ) -> Tuple[List[Dict], Dict]:
        """
        Full analysis: local shifts + section-level shift.

        Returns
        -------
        (enriched_sequence, section_metrics)
        """
        enriched = self.compute_local_shifts(sentiment_sequence)
        section_metrics = self.compute_section_shift(sentiment_sequence)

        # Back-fill prepared_vs_qa_shift into each sentence record
        shift_val = section_metrics["prepared_vs_qa_shift"]
        for item in enriched:
            item["prepared_vs_qa_shift"] = shift_val

        return enriched, section_metrics

