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


def _build_demo_baseline_data() -> pd.DataFrame:
    """Return a minimal in-memory transcript dataset for tests and offline use."""
    rows = [
        {
            "transcript_id": "demo_001",
            "ticker": "AAPL",
            "date": "2017-01-31",
            "year": 2017,
            "quarter": "Q1",
            "overall_hidden_risk": 24.0,
            "average_risk": 21.5,
            "hedging_avg": 18.0,
            "evasiveness_avg": 22.0,
            "tone_shift_avg": 31.0,
            "qa_risk": 27.0,
            "prepared_risk": 22.0,
            "lm_lexicon_score": 15.0,
            "finbert_only_score": 22.0,
            "hedging_only_score": 18.0,
            "return_1d": -0.010,
            "return_5d": 0.012,
            "return_10d": 0.035,
            "return_20d": 0.042,
            "volatility_5d": 0.085,
            "volatility_10d": 0.112,
            "abnormal_return_5d": 0.008,
            "actual_revenue": 78.35,
            "earnings_surprise": 0.04,
            "guidance_change": "neutral",
        },
        {
            "transcript_id": "demo_002",
            "ticker": "MSFT",
            "date": "2019-07-18",
            "year": 2019,
            "quarter": "Q4",
            "overall_hidden_risk": 52.0,
            "average_risk": 46.5,
            "hedging_avg": 48.0,
            "evasiveness_avg": 56.0,
            "tone_shift_avg": 41.0,
            "qa_risk": 58.0,
            "prepared_risk": 45.0,
            "lm_lexicon_score": 36.0,
            "finbert_only_score": 42.0,
            "hedging_only_score": 48.0,
            "return_1d": 0.022,
            "return_5d": -0.008,
            "return_10d": 0.018,
            "return_20d": 0.055,
            "volatility_5d": 0.121,
            "volatility_10d": 0.134,
            "abnormal_return_5d": -0.003,
            "actual_revenue": 125.8,
            "earnings_surprise": 0.06,
            "guidance_change": "positive",
        },
        {
            "transcript_id": "demo_003",
            "ticker": "NVDA",
            "date": "2022-05-25",
            "year": 2022,
            "quarter": "Q1",
            "overall_hidden_risk": 76.0,
            "average_risk": 65.0,
            "hedging_avg": 63.0,
            "evasiveness_avg": 72.0,
            "tone_shift_avg": 70.0,
            "qa_risk": 78.0,
            "prepared_risk": 61.0,
            "lm_lexicon_score": 58.0,
            "finbert_only_score": 67.0,
            "hedging_only_score": 63.0,
            "return_1d": -0.042,
            "return_5d": -0.071,
            "return_10d": -0.025,
            "return_20d": 0.010,
            "volatility_5d": 0.199,
            "volatility_10d": 0.215,
            "abnormal_return_5d": -0.061,
            "actual_revenue": 7.2,
            "earnings_surprise": -0.02,
            "guidance_change": "negative",
        },
    ]
    return pd.DataFrame(rows)


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
    from src.database import get_db_manager
    mgr = get_db_manager(DB_PATH)
    if not DB_PATH.exists():
        mgr.init_database()
    conn = mgr.get_connection()

    try:
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

        finbert_df = pd.read_sql_query("""
            SELECT
                rs.transcript_id,
                AVG(rs.tone_shift_score) AS tone_shift_avg_raw,
                COUNT(*) AS sentence_count
            FROM risk_scores rs
            GROUP BY rs.transcript_id
        """, conn)

        sent_df = pd.read_sql_query("""
            SELECT s.sentence_id, s.transcript_id, s.text
            FROM sentences s
        """, conn)

        mkt_df = pd.read_sql_query("""
            SELECT transcript_id,
                   return_1d, return_5d, return_10d, return_20d,
                   volatility_5d, volatility_10d, abnormal_return_5d
            FROM market_events
        """, conn)

        out_df = pd.read_sql_query("""
            SELECT transcript_id, actual_revenue, earnings_surprise, guidance_change
            FROM outcomes
        """, conn)
    finally:
        conn.close()

    if ts_df.empty:
        return _build_demo_baseline_data()

    lm_scores = (
        sent_df.copy()
        .assign(lm_score=lambda df: df["text"].apply(lm_negative_score))
        .groupby("transcript_id")["lm_score"]
        .mean()
        .reset_index(name="lm_lexicon_score")
    )

    df = (
        ts_df
        .merge(finbert_df, on="transcript_id", how="left")
        .merge(lm_scores, on="transcript_id", how="left")
        .merge(mkt_df, on="transcript_id", how="left")
        .merge(out_df, on="transcript_id", how="left")
    )

    df["hedging_only_score"] = df["hedging_avg"]
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
