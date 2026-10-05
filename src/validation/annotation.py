"""
Annotation workflow for Phase 5 — Human annotation of Q&A evasiveness.

Supports:
  - Sampling Q&A pairs for annotation
  - Exporting to CSV
  - Importing human labels from CSV
  - Computing automated-vs-human label agreement

Label scheme (evasiveness):
  0 = direct answer
  1 = mildly evasive
  2 = strongly evasive

Optional labels (binary, 0/1):
  hedging, topic_avoidance, perceived_uncertainty
"""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.database import get_db_manager

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "database" / "hidden_risk.db"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_COLUMNS = [
    "qa_id",
    "transcript_id",
    "ticker",
    "date",
    "question_text",
    "answer_text",
    "auto_evasiveness_score",  # from Phase 2 risk_scores
    "auto_hedging_score",
    "auto_hidden_risk_score",
    "evaluation_type",         # 'human' or 'synthetic'
    "annotator_id",            # annotator identifier for multi-annotator agreement
    # human label columns — blank on export, filled on import
    "label_evasive",           # 0/1/2
    "label_hedging",           # 0/1 (optional)
    "label_topic_avoidance",   # 0/1 (optional)
    "label_perceived_uncertainty",  # 0/1 (optional)
    "annotator_notes",
]

# Automated score thresholds that map continuous evasiveness score → label
EVASIVE_THRESHOLDS = (20.0, 50.0)   # <20 → 0, 20-50 → 1, >50 → 2


def _get_connection() -> sqlite3.Connection:
    """Get a database connection, ensuring directory and schema exist."""
    mgr = get_db_manager(DB_PATH)
    if not DB_PATH.exists():
        mgr.init_database()
    return mgr.get_connection()



def sample_qa_for_annotation(
    n: int = 500,
    seed: int = 42,
    split: str = "all",
    time_cutoff: Optional[str] = None,
) -> List[Dict]:
    """
    Sample up to `n` substantive Q&A pairs for human annotation.

    For reproducibility uses a deterministic seed.
    Only QA pairs with question_text > 30 chars and answer_text > 50 chars.

    Args:
        n: Max pairs to sample.
        seed: Random seed.
        split: 'all', 'train', 'val', 'test'
        time_cutoff: If provided, only pairs where transcript date <= cutoff.
    Returns:
        List of dict rows ready for export.
    """
    conn = _get_connection()
    try:
        query = """
            SELECT
                qp.qa_id,
                qp.transcript_id,
                t.ticker,
                t.date,
                qp.question_text,
                qp.answer_text,
                ts_q.evasiveness_score  AS auto_evasiveness_score,
                ts_q.hedging_score      AS auto_hedging_score,
                ts_q.hidden_risk_score  AS auto_hidden_risk_score
            FROM qa_pairs qp
            JOIN transcripts t ON qp.transcript_id = t.transcript_id
            -- Get scores for the answer segment's first sentence
            LEFT JOIN (
                SELECT s.segment_id, AVG(rs.evasiveness_score) AS evasiveness_score,
                       AVG(rs.hedging_score) AS hedging_score,
                       AVG(rs.hidden_risk_score) AS hidden_risk_score
                FROM sentences s
                JOIN risk_scores rs ON s.sentence_id = rs.sentence_id
                GROUP BY s.segment_id
            ) ts_q ON ts_q.segment_id = qp.answer_segment_id
            WHERE length(qp.question_text) > 30
              AND length(qp.answer_text) > 50
        """
        params: list = []

        if split == "train":
            query += " AND t.date < '2019-01-01'"
        elif split == "val":
            query += " AND t.date >= '2019-01-01' AND t.date < '2022-01-01'"
        elif split == "test":
            query += " AND t.date >= '2022-01-01'"

        if time_cutoff:
            query += " AND t.date <= ?"
            params.append(time_cutoff)

        rows = conn.execute(query, params).fetchall()
        records = [dict(r) for r in rows]

        # Deterministic shuffle and sample
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(records))[:n]
        sampled = [records[i] for i in sorted(indices)]

        return sampled
    finally:
        conn.close()


def export_annotation_csv(
    rows: List[Dict],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Export Q&A rows to a CSV for human annotation.

    Human-fillable columns: label_evasive, label_hedging,
    label_topic_avoidance, label_perceived_uncertainty, annotator_notes.

    Returns path to written file.
    """
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = ANNOTATION_DIR / f"annotation_export_{ts}.csv"

    output_path = Path(output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {col: row.get(col, "") for col in EXPORT_COLUMNS}
            writer.writerow(out)

    print(f"[ANNOTATION] Exported {len(rows)} rows → {output_path}")
    return output_path


def import_annotation_csv(path: Path) -> List[Dict]:
    """
    Import a completed human annotation CSV.

    Returns list of dicts; only rows with non-empty label_evasive are returned.
    Validates that label_evasive is 0, 1, or 2.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    results = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            label_str = row.get("label_evasive", "").strip()
            if not label_str:
                continue  # skip unannotated rows
            try:
                label = int(label_str)
                if label not in (0, 1, 2):
                    raise ValueError(f"label_evasive must be 0/1/2, got {label}")
            except ValueError as e:
                print(f"[WARN] Row {i} ({row.get('qa_id', '?')}): {e}")
                continue

            record = {
                "qa_id": row["qa_id"],
                "transcript_id": row.get("transcript_id", ""),
                "label_evasive": label,
                "label_hedging": _safe_int(row.get("label_hedging", "")),
                "label_topic_avoidance": _safe_int(row.get("label_topic_avoidance", "")),
                "label_perceived_uncertainty": _safe_int(row.get("label_perceived_uncertainty", "")),
                "auto_evasiveness_score": _safe_float(row.get("auto_evasiveness_score", "")),
                "auto_hedging_score": _safe_float(row.get("auto_hedging_score", "")),
                "auto_hidden_risk_score": _safe_float(row.get("auto_hidden_risk_score", "")),
                "annotator_notes": row.get("annotator_notes", ""),
            }
            results.append(record)

    print(f"[ANNOTATION] Imported {len(results)} labeled rows from {path}")
    return results


def score_to_label(score: float) -> int:
    """Convert continuous evasiveness score (0-100) → 0/1/2 label."""
    lo, hi = EVASIVE_THRESHOLDS
    if score < lo:
        return 0
    elif score < hi:
        return 1
    else:
        return 2


def build_synthetic_annotations(rows: List[Dict]) -> List[Dict]:
    """
    Build synthetic human annotations by applying a noisy rule-based labeling
    to existing auto scores. Used when no real human labels exist.

    IMPORTANT: This is not a ground truth. It is a lower-bound evaluation proxy
    and clearly labeled as such in the report. It introduces noise to simulate
    realistic inter-annotator disagreement.

    Returns list of records with label_evasive populated.
    """
    rng = np.random.default_rng(123)
    labeled = []
    for row in rows:
        auto_score = row.get("auto_evasiveness_score") or 0.0
        auto_label = score_to_label(float(auto_score))

        # Add realistic annotator noise: flip label ±1 step with 15% probability
        flip = rng.random()
        if flip < 0.08 and auto_label < 2:
            noisy_label = auto_label + 1
        elif flip < 0.15 and auto_label > 0:
            noisy_label = auto_label - 1
        else:
            noisy_label = auto_label

        record = dict(row)
        record["evaluation_type"] = "synthetic"
        record["annotator_id"] = "synthetic_generator"
        record["label_evasive"] = noisy_label
        record["label_hedging"] = 1 if (row.get("auto_hedging_score") or 0) > 25 else 0
        record["label_topic_avoidance"] = 1 if noisy_label >= 1 else 0
        record["label_perceived_uncertainty"] = 1 if (row.get("auto_hidden_risk_score") or 0) > 40 else 0
        labeled.append(record)

    return labeled


def compute_cohens_kappa(
    rater1_labels: List[int],
    rater2_labels: List[int],
    n_classes: int = 3,
) -> Dict[str, float]:
    """
    Calculate Cohen's Kappa and percentage agreement between two annotators.
    """
    if len(rater1_labels) != len(rater2_labels) or len(rater1_labels) == 0:
        return {"cohens_kappa": 0.0, "observed_agreement": 0.0, "expected_agreement": 0.0}

    n = len(rater1_labels)
    # Observed agreement
    agreements = sum(1 for a, b in zip(rater1_labels, rater2_labels) if a == b)
    po = agreements / n

    # Expected agreement by chance
    pe = 0.0
    for c in range(n_classes):
        p1_c = sum(1 for a in rater1_labels if a == c) / n
        p2_c = sum(1 for b in rater2_labels if b == c) / n
        pe += p1_c * p2_c

    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) > 1e-6 else 1.0

    return {
        "cohens_kappa": round(kappa, 4),
        "observed_agreement": round(po, 4),
        "expected_agreement": round(pe, 4),
    }


def _safe_int(v: str) -> Optional[int]:
    try:
        return int(v) if v.strip() else None
    except (ValueError, AttributeError):
        return None


def _safe_float(v: str) -> Optional[float]:
    try:
        return float(v) if v.strip() else None
    except (ValueError, AttributeError):
        return None

