"""
Baseline models for Phase 5 comparison.

Baseline 1: Financial lexicon / simple sentiment
  - Loughran-McDonald-style negative word count over token count.
  - No ML model. Rule-based.

Baseline 2: FinBERT sentiment score only
  - Use existing negative_prob from risk_scores (already computed in Phase 2).
  - No hedging/evasiveness logic.

Baseline 3: Hedging detector only
  - Use existing hedging_score from risk_scores.

Proposed system: hedging + evasiveness + tone shift (= hidden_risk_score from Phase 2)
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "database" / "hidden_risk.db"


# ── Loughran-McDonald inspired negative word list (abbreviated, public domain) ──
LM_NEGATIVE = {
    "abandon", "abandoned", "abandonment", "abnormal", "absence", "absent",
    "abuse", "adverse", "adversely", "adversity", "alarm", "alarming",
    "allegation", "bankrupt", "bankruptcy", "below", "breach", "burden",
    "challenge", "challenging", "claim", "complaint", "concern", "concerning",
    "conflict", "controversy", "costly", "crisis", "critical", "damage",
    "decline", "declining", "default", "deficiency", "deficit", "delay",
    "deteriorate", "deteriorating", "difficult", "difficulty", "disappoint",
    "disappointing", "disappointment", "dispute", "disrupt", "disruption",
    "doubt", "downturn", "drop", "eliminate", "error", "exceed", "excessive",
    "fail", "failure", "fall", "falling", "fault", "fear", "fraud",
    "harm", "headwind", "headwinds", "higher", "impair", "impairment",
    "inability", "inadequate", "incident", "inferior", "inflation",
    "insufficient", "investigat", "issue", "issues", "lack", "late",
    "limitation", "limited", "loss", "losses", "lower", "material",
    "miss", "misstatement", "negative", "negligence", "nonperforming",
    "obstacle", "overrun", "penalty", "poor", "pressure", "problem",
    "problems", "reduce", "reduction", "regulatory", "remediation", "risk",
    "risks", "shortage", "significant", "slow", "slowdown", "slowing",
    "struggle", "substantial", "suffer", "suffering", "uncertainty",
    "unexpected", "unfavorable", "unfavorably", "volatility", "vulnerability",
    "warn", "warning", "weak", "weaken", "weakness", "worse", "worsen",
    "worsening", "write-off", "writeoff", "writedown",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def lm_negative_score(text: str) -> float:
    """
    Baseline 1: Loughran-McDonald negative word ratio.
    Returns score in [0, 100].
    """
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    neg_count = sum(1 for t in tokens if t in LM_NEGATIVE)
    return min(neg_count / len(tokens) * 500, 100.0)  # scale so ~20% = 100


def compute_baseline1_scores(df_sentences: pd.DataFrame) -> pd.Series:
    """
    Compute Baseline 1 (LM lexicon) scores for each sentence.

    Args:
        df_sentences: DataFrame with columns ['sentence_id', 'text']
    Returns:
        Series indexed by sentence_id with float scores [0, 100]
    """
    scores = df_sentences["text"].apply(lm_negative_score)
    scores.index = df_sentences["sentence_id"]
    return scores


def load_baseline_data() -> pd.DataFrame:
    """
    Load transcript-level baseline data from DB.

    Returns DataFrame with per-transcript columns:
      transcript_id, ticker, date, year, quarter,
      # Proposed system (Phase 2 outputs)
      overall_hidden_risk, hedging_avg, evasiveness_avg, tone_shift_avg,
      average_risk,
      # FinBERT raw (negative probability average — Baseline 2 proxy)
      finbert_neg_avg,
      # Hedging-only (Baseline 3)
      hedging_only_score,
      # LM lexicon (Baseline 1) — computed separately per sentence, then averaged
      lm_lexicon_score,
      # Market outcomes
      return_1d, return_5d, return_10d, return_20d,
      volatility_5d, volatility_10d, abnormal_return_5d,
      # Fundamental outcomes
      actual_revenue, earnings_surprise, guidance_change
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Transcript-level risk scores
    ts_df = pd.read_sql_query("""
        SELECT
            t.transcript_id, t.ticker, t.date, t.year, t.quarter,
            ts.overall_hidden_risk,
            ts.average_risk,
            ts.hedging_avg,
            ts.evasiveness_avg,
            ts.tone_shift_avg,
            ts.qa_risk,
            ts.prepared_risk
        FROM transcripts t
        JOIN transcript_scores ts ON t.transcript_id = ts.transcript_id
        ORDER BY t.date
    """, conn)

    # FinBERT raw: average negative probability per transcript (Baseline 2 proxy)
    # contributing_factors stores JSON with 'negative_prob'
    finbert_df = pd.read_sql_query("""
        SELECT
            rs.transcript_id,
            AVG(rs.tone_shift_score) AS tone_shift_avg_raw,
            COUNT(*) AS sentence_count
        FROM risk_scores rs
        GROUP BY rs.transcript_id
    """, conn)

    # Load sentences for LM Baseline 1
    sent_df = pd.read_sql_query("""
        SELECT s.sentence_id, s.transcript_id, s.text
        FROM sentences s
    """, conn)

    # Market events
    mkt_df = pd.read_sql_query("""
        SELECT transcript_id,
               return_1d, return_5d, return_10d, return_20d,
               volatility_5d, volatility_10d, abnormal_return_5d
        FROM market_events
    """, conn)

    # Outcomes
    out_df = pd.read_sql_query("""
        SELECT transcript_id, actual_revenue, earnings_surprise, guidance_change
        FROM outcomes
    """, conn)

    conn.close()

    # Compute LM lexicon score per transcript
    lm_scores = (
        sent_df.copy()
        .assign(lm_score=lambda df: df["text"].apply(lm_negative_score))
        .groupby("transcript_id")["lm_score"]
        .mean()
        .reset_index(name="lm_lexicon_score")
    )

    # Merge all
    df = (
        ts_df
        .merge(finbert_df, on="transcript_id", how="left")
        .merge(lm_scores, on="transcript_id", how="left")
        .merge(mkt_df, on="transcript_id", how="left")
        .merge(out_df, on="transcript_id", how="left")
    )

    # Baseline 3 = hedging_avg (already a column)
    df["hedging_only_score"] = df["hedging_avg"]

    # Baseline 2 proxy = tone_shift_avg_raw (FinBERT's primary output in Phase 2)
    # We use this because finbert negative_prob lives in contributing_factors JSON
    # and the transcript-level tone shift is the direct FinBERT derivative
    df["finbert_only_score"] = df["tone_shift_avg"]

    return df


def apply_time_split(
    df: pd.DataFrame,
    train_end: str = "2018-12-31",
    val_end: str = "2021-12-31",
) -> pd.DataFrame:
    """
    Add a 'split' column to df:
      'train'  : date <= train_end
      'val'    : train_end < date <= val_end
      'test'   : date > val_end

    Uses transcript call date, not any outcome date, so no leakage.
    """
    dates = pd.to_datetime(df["date"])
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)

    conditions = [
        dates <= train_end_dt,
        (dates > train_end_dt) & (dates <= val_end_dt),
        dates > val_end_dt,
    ]
    df = df.copy()
    df["split"] = np.select(conditions, ["train", "val", "test"], default="train")
    return df
