"""
Risk Scorer — Phase 2

Aggregates per-sentence linguistic features into:

  1. Sentence-level Hidden Risk Score
  2. Segment-level risk scores
  3. Q&A-level risk scores
  4. Transcript-level Hidden Risk Score

Formula (configurable via config.yaml):
  sentence_hidden_risk = w_hedge * hedging_score
                       + w_evasive * evasive_score  (or 0 if not in QA)
                       + w_shift * tone_shift_score

Transcript aggregation:
  overall_hidden_risk = avg_weight * avg_risk + top_weight * top_k_avg_risk

All weights are read from AppConfig → RiskConfig and AggregationConfig.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

from src.config import get_config
from src.logger import get_logger

logger = get_logger("risk_scorer")


class RiskScorer:
    """
    Computes Hidden Risk Scores at sentence, segment, Q&A, and transcript levels.
    Weights are pulled from AppConfig and therefore configurable via config.yaml.
    """

    def __init__(self):
        cfg = get_config()
        r = cfg.risk
        a = cfg.aggregation

        self.w_hedge = r.hedging_weight        # default 0.35
        self.w_evasive = r.evasiveness_weight  # default 0.40
        self.w_shift = r.tone_shift_weight     # default 0.25

        self.avg_weight = a.average_weight     # default 0.60
        self.top_weight = a.top_risk_weight    # default 0.40
        self.top_k_pct = a.top_k_percentile   # default 0.10 (top 10%)

        logger.debug(
            f"RiskScorer weights: hedge={self.w_hedge}, evasive={self.w_evasive}, "
            f"shift={self.w_shift} | agg: avg={self.avg_weight}, top={self.top_weight}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Sentence-level
    # ─────────────────────────────────────────────────────────────────────

    def sentence_hidden_risk(
        self,
        hedging_score: float,
        tone_shift_score: float,
        evasive_score: float = 0.0,
        evasive_weight_override: Optional[float] = None,
    ) -> float:
        """
        Compute Hidden Risk Score for a single sentence.

        Parameters
        ----------
        hedging_score    : 0-100
        tone_shift_score : 0-100
        evasive_score    : 0-100 (use 0 for non-QA sentences)
        evasive_weight_override : optional override for weight redistribution

        Returns
        -------
        hidden_risk_score : float [0, 100]
        """
        # When evasive_score is not applicable (not in Q&A), redistribute weight
        if evasive_score == 0.0:
            w_h = self.w_hedge / (self.w_hedge + self.w_shift)
            w_s = self.w_shift / (self.w_hedge + self.w_shift)
            w_e = 0.0
        else:
            w_h = self.w_hedge
            w_s = self.w_shift
            w_e = evasive_weight_override if evasive_weight_override is not None else self.w_evasive

        raw = w_h * hedging_score + w_e * evasive_score + w_s * tone_shift_score
        return round(min(max(raw, 0.0), 100.0), 2)

    def score_sentence(self, features: Dict) -> Dict:
        """
        Given a feature dict (output of merging hedging + finbert + tone_shift),
        compute all risk scores and return enriched dict.

        Expected keys:
          hedging_score, tone_shift_score, evasive_score (optional),
          positive_prob, negative_prob, neutral_prob, sentiment_label, ...
        """
        hedging = features.get("hedging_score", 0.0)
        shift = features.get("tone_shift_score", 0.0)
        evasive = features.get("evasive_score", 0.0)

        hidden_risk = self.sentence_hidden_risk(hedging, shift, evasive)

        contributing_factors = {
            "hedging_score": hedging,
            "tone_shift_score": shift,
            "evasive_score": evasive,
            "hedging_weight": self.w_hedge,
            "evasive_weight": self.w_evasive,
            "shift_weight": self.w_shift,
            # Explainability details
            "matched_modals": features.get("matched_modals", []),
            "matched_qualifiers": features.get("matched_qualifiers", []),
            "matched_uncertainty_phrases": features.get("matched_uncertainty_phrases", []),
            "matched_deflection_phrases": features.get("matched_deflection_phrases", []),
            "deflection_count": features.get("deflection_count", 0),
            "uncertainty_phrase_count": features.get("uncertainty_phrase_count", 0),
            "sentiment_label": features.get("sentiment_label", ""),
            "negative_prob": features.get("negative_prob", 0.0),
        }

        return {
            **features,
            "hidden_risk_score": hidden_risk,
            "contributing_factors": contributing_factors,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Segment-level aggregation
    # ─────────────────────────────────────────────────────────────────────

    def aggregate_segment(self, sentence_scores: List[Dict]) -> Dict:
        """
        Aggregate sentence-level risk scores into a segment-level summary.

        Parameters
        ----------
        sentence_scores : list of scored sentence dicts (output of score_sentence)

        Returns
        -------
        Dict with segment-level aggregated fields.
        """
        if not sentence_scores:
            return self._empty_segment()

        hidden_risks = [s.get("hidden_risk_score", 0.0) for s in sentence_scores]
        hedging_scores = [s.get("hedging_score", 0.0) for s in sentence_scores]
        shift_scores = [s.get("tone_shift_score", 0.0) for s in sentence_scores]
        evasive_scores = [s.get("evasive_score", 0.0) for s in sentence_scores]
        neg_probs = [s.get("negative_prob", 0.0) for s in sentence_scores]

        avg_hidden_risk = _mean(hidden_risks)
        top_k = _top_k_mean(hidden_risks, self.top_k_pct)
        segment_risk = round(
            self.avg_weight * avg_hidden_risk + self.top_weight * top_k, 2
        )

        # Find highest-risk sentence
        max_idx = hidden_risks.index(max(hidden_risks))
        top_sentence = sentence_scores[max_idx]

        return {
            "segment_risk_score": segment_risk,
            "avg_hidden_risk": round(avg_hidden_risk, 2),
            "top_k_risk": round(top_k, 2),
            "avg_hedging": round(_mean(hedging_scores), 2),
            "avg_tone_shift": round(_mean(shift_scores), 2),
            "avg_evasive": round(_mean(evasive_scores), 2),
            "avg_negative_prob": round(_mean(neg_probs), 4),
            "max_hidden_risk_in_segment": round(max(hidden_risks), 2),
            "sentence_count": len(sentence_scores),
            "top_risk_sentence_id": top_sentence.get("sentence_id", ""),
            "top_risk_sentence_text": top_sentence.get("text", "")[:200],
        }

    # ─────────────────────────────────────────────────────────────────────
    # Q&A-level aggregation
    # ─────────────────────────────────────────────────────────────────────

    def score_qa_pair(
        self,
        qa_features: Dict,
        answer_sentence_scores: List[Dict],
    ) -> Dict:
        """
        Compute Q&A-level risk score combining evasiveness features
        with the risk scores of the answer's sentences.

        Parameters
        ----------
        qa_features          : output of EvasivenessDetector.analyse()
        answer_sentence_scores: list of scored answer sentences

        Returns
        -------
        Dict with qa_hidden_risk_score and contributing factors.
        """
        evasive_score = qa_features.get("evasive_score", 0.0)
        hedging_density = qa_features.get("hedging_density_answer", 0.0)

        # Average sentence risk from the answer block
        if answer_sentence_scores:
            avg_sentence_risk = _mean(
                [s.get("hidden_risk_score", 0.0) for s in answer_sentence_scores]
            )
            avg_hedging = _mean([s.get("hedging_score", 0.0) for s in answer_sentence_scores])
            avg_shift = _mean([s.get("tone_shift_score", 0.0) for s in answer_sentence_scores])
        else:
            avg_sentence_risk = 0.0
            avg_hedging = 0.0
            avg_shift = 0.0

        # Q&A risk: heavy weight on evasiveness since that's QA-specific
        qa_risk = round(
            0.50 * evasive_score
            + 0.25 * avg_sentence_risk
            + 0.15 * avg_hedging
            + 0.10 * avg_shift,
            2,
        )

        return {
            "qa_hidden_risk_score": qa_risk,
            "evasive_score": evasive_score,
            "qa_tfidf_similarity": qa_features.get("qa_tfidf_similarity", 0.0),
            "topic_overlap": qa_features.get("topic_overlap", 0.0),
            "deflection_count": qa_features.get("deflection_count", 0),
            "deflection_indicator_score": qa_features.get("deflection_indicator_score", 0.0),
            "generic_language_score": qa_features.get("generic_language_score", 0.0),
            "numeric_response_score": qa_features.get("numeric_response_score", 0.0),
            "answer_length_words": qa_features.get("answer_length_words", 0),
            "avg_sentence_risk_in_answer": round(avg_sentence_risk, 2),
            "avg_hedging_in_answer": round(avg_hedging, 2),
            "avg_shift_in_answer": round(avg_shift, 2),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Transcript-level aggregation
    # ─────────────────────────────────────────────────────────────────────

    def aggregate_transcript(
        self,
        all_sentence_scores: List[Dict],
        qa_scores: List[Dict],
        section_shift_metrics: Optional[Dict] = None,
    ) -> Dict:
        """
        Compute transcript-level Hidden Risk Score.

        Parameters
        ----------
        all_sentence_scores  : scored sentences across entire transcript
        qa_scores            : scored Q&A pairs
        section_shift_metrics: prepared_vs_qa_shift info from ToneShiftDetector

        Returns
        -------
        Dict suitable for inserting into transcript_scores table.
        """
        if not all_sentence_scores:
            return self._empty_transcript()

        hidden_risks = [s.get("hidden_risk_score", 0.0) for s in all_sentence_scores]
        hedging_scores = [s.get("hedging_score", 0.0) for s in all_sentence_scores]
        shift_scores = [s.get("tone_shift_score", 0.0) for s in all_sentence_scores]

        # Separate by section
        prepared_risks = [
            s.get("hidden_risk_score", 0.0)
            for s in all_sentence_scores
            if s.get("section") in ("prepared_remarks", "opening")
        ]
        qa_risks = [
            s.get("hidden_risk_score", 0.0)
            for s in all_sentence_scores
            if s.get("section") == "q_and_a"
        ]

        avg_risk = _mean(hidden_risks)
        top_k_risk = _top_k_mean(hidden_risks, self.top_k_pct)

        overall_hidden_risk = round(
            self.avg_weight * avg_risk + self.top_weight * top_k_risk, 2
        )

        prepared_risk = round(_mean(prepared_risks), 2) if prepared_risks else 0.0
        qa_risk_avg = round(_mean(qa_risks), 2) if qa_risks else 0.0

        # Supplement with Q&A-level scores if available
        if qa_scores:
            qa_pair_risk = _mean([q.get("qa_hidden_risk_score", 0.0) for q in qa_scores])
            qa_risk_avg = round(0.5 * qa_risk_avg + 0.5 * qa_pair_risk, 2)

        # Find the highest-risk segment
        max_idx = hidden_risks.index(max(hidden_risks))
        top_sentence = all_sentence_scores[max_idx]
        highest_risk_segment = top_sentence.get("segment_id", "")

        # Section shift bonus (if prepared vs QA diverges strongly → higher overall risk)
        section_shift_bonus = 0.0
        if section_shift_metrics:
            pqs = section_shift_metrics.get("prepared_vs_qa_shift", 0.0)
            section_shift_bonus = min(pqs * 0.1, 5.0)  # up to +5 bonus

        final_risk = round(min(overall_hidden_risk + section_shift_bonus, 100.0), 2)

        return {
            "overall_hidden_risk": final_risk,
            "average_risk": round(avg_risk, 2),
            "top_risk_average": round(top_k_risk, 2),
            "qa_risk": qa_risk_avg,
            "prepared_risk": prepared_risk,
            "highest_risk_segment": highest_risk_segment,
            "hedging_avg": round(_mean(hedging_scores), 2),
            "evasiveness_avg": round(
                _mean([s.get("evasive_score", 0.0) for s in all_sentence_scores]), 2
            ),
            "tone_shift_avg": round(_mean(shift_scores), 2),
            "prepared_vs_qa_shift": (
                section_shift_metrics.get("prepared_vs_qa_shift", 0.0)
                if section_shift_metrics
                else 0.0
            ),
            "sentence_count": len(all_sentence_scores),
            "qa_pair_count": len(qa_scores),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Empty results
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_segment() -> Dict:
        return {
            "segment_risk_score": 0.0,
            "avg_hidden_risk": 0.0,
            "top_k_risk": 0.0,
            "avg_hedging": 0.0,
            "avg_tone_shift": 0.0,
            "avg_evasive": 0.0,
            "avg_negative_prob": 0.0,
            "max_hidden_risk_in_segment": 0.0,
            "sentence_count": 0,
            "top_risk_sentence_id": "",
            "top_risk_sentence_text": "",
        }

    @staticmethod
    def _empty_transcript() -> Dict:
        return {
            "overall_hidden_risk": 0.0,
            "average_risk": 0.0,
            "top_risk_average": 0.0,
            "qa_risk": 0.0,
            "prepared_risk": 0.0,
            "highest_risk_segment": "",
            "hedging_avg": 0.0,
            "evasiveness_avg": 0.0,
            "tone_shift_avg": 0.0,
            "prepared_vs_qa_shift": 0.0,
            "sentence_count": 0,
            "qa_pair_count": 0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _top_k_mean(values: List[float], k_pct: float) -> float:
    """Mean of the top-k% highest values."""
    if not values:
        return 0.0
    k = max(1, math.ceil(len(values) * k_pct))
    top_k = sorted(values, reverse=True)[:k]
    return sum(top_k) / len(top_k)

