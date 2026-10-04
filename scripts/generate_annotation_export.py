"""
Annotation export utility.

Usage:
    python scripts/generate_annotation_export.py [--n N] [--seed S] [--split SPLIT]

Creates a CSV at data/annotations/annotation_export_<timestamp>.csv
with N Q&A pairs ready for human annotation.

Fills these columns: qa_id, transcript_id, ticker, date, question_text,
answer_text, auto_evasiveness_score, auto_hedging_score, auto_hidden_risk_score.

Leave these blank for human annotators to fill:
  label_evasive (0/1/2), label_hedging (0/1),
  label_topic_avoidance (0/1), label_perceived_uncertainty (0/1),
  annotator_notes

Import with: python scripts/run_phase5.py --annotation-csv <path>
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.annotation import export_annotation_csv, sample_qa_for_annotation


def main() -> int:
    p = argparse.ArgumentParser(description="Generate annotation export CSV")
    p.add_argument("--n", type=int, default=500, help="Number of QA pairs (default: 500)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    p.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = p.parse_args()

    print(f"Sampling {args.n} Q&A pairs (seed={args.seed}, split={args.split})...")
    rows = sample_qa_for_annotation(n=args.n, seed=args.seed, split=args.split)
    print(f"Sampled: {len(rows)} rows")

    output_path = Path(args.output) if args.output else None
    path = export_annotation_csv(rows, output_path=output_path)
    print(f"Done → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
