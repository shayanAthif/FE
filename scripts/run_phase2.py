"""
Phase 2 — Run Script

Usage:
  python scripts/run_phase2.py [--limit N] [--batch-size N] [--force]

  --limit N       Process only first N transcripts (default: all)
  --batch-size N  FinBERT batch size (default: from config)
  --force         Re-process already-scored transcripts
  --test          Run on 10 transcripts only (quick validation mode)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.database import get_db_manager
from src.logger import get_logger
from src.risk_engine import run_phase2

logger = get_logger("run_phase2")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2 — Linguistic Risk Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N transcripts.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="FinBERT inference batch size (default: from config.yaml).")
    parser.add_argument("--force", action="store_true",
                        help="Re-process transcripts that already have scores.")
    parser.add_argument("--test", action="store_true",
                        help="Quick test mode: process only 10 transcripts.")
    return parser.parse_args()


def print_sample_results(limit: int = 5) -> None:
    """Print sample high-risk sentences from the database."""
    import sqlite3
    db_path = PROJECT_ROOT / "database" / "hidden_risk.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    print("\n" + "=" * 70)
    print("TOP HIGH-RISK SENTENCES")
    print("=" * 70)
    rows = conn.execute(
        """
        SELECT rs.hidden_risk_score, rs.hedging_score, rs.evasiveness_score,
               rs.tone_shift_score, s.text, rs.transcript_id
        FROM risk_scores rs
        JOIN sentences s ON rs.sentence_id = s.sentence_id
        WHERE rs.hidden_risk_score IS NOT NULL
        ORDER BY rs.hidden_risk_score DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    for i, row in enumerate(rows, 1):
        print(f"\n{i}. Score: {row['hidden_risk_score']:.1f} | "
              f"H:{row['hedging_score']:.0f} E:{row['evasiveness_score']:.0f} "
              f"T:{row['tone_shift_score']:.0f} | {row['transcript_id']}")
        print(f"   \"{row['text'][:120]}\"")

    print("\n" + "=" * 70)
    print("TRANSCRIPT RISK SUMMARY (TOP 10)")
    print("=" * 70)
    ts_rows = conn.execute(
        """
        SELECT transcript_id, overall_hidden_risk, average_risk,
               qa_risk, prepared_risk, hedging_avg, evasiveness_avg, tone_shift_avg
        FROM transcript_scores
        ORDER BY overall_hidden_risk DESC
        LIMIT 10
        """,
    ).fetchall()
    print(f"{'Transcript ID':<35} {'Risk':>6} {'QA':>6} {'Hedge':>6} {'Evasive':>8} {'Shift':>6}")
    print("-" * 75)
    for row in ts_rows:
        print(
            f"{row['transcript_id']:<35} "
            f"{row['overall_hidden_risk']:>6.1f} "
            f"{row['qa_risk']:>6.1f} "
            f"{row['hedging_avg']:>6.1f} "
            f"{row['evasiveness_avg']:>8.1f} "
            f"{row['tone_shift_avg']:>6.1f}"
        )

    conn.close()


def main():
    args = parse_args()
    cfg = load_config()

    # Ensure schema is up to date
    db = get_db_manager()
    db.init_database()

    limit = 10 if args.test else args.limit
    batch_size = args.batch_size

    print(f"\n{'='*60}")
    print(f"Phase 2 — Linguistic Risk Engine")
    print(f"{'='*60}")
    print(f"  Mode        : {'TEST (10 transcripts)' if args.test else 'FULL'}")
    print(f"  Limit       : {limit or 'ALL'}")
    print(f"  Batch size  : {batch_size or cfg.processing.batch_size} (FinBERT)")
    print(f"  Force       : {args.force}")
    print(f"  Device      : CPU (CUDA not available)")
    print(f"{'='*60}\n")

    t0 = time.time()
    stats = run_phase2(
        limit=limit,
        batch_size=batch_size,
        force_reprocess=args.force,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print("PHASE 2 COMPLETE")
    print(f"{'='*60}")
    print(f"  Transcripts processed  : {stats['transcripts_processed']}")
    print(f"  Transcripts skipped    : {stats['transcripts_skipped']}")
    print(f"  Sentences scored       : {stats['sentences_scored']}")
    print(f"  Q&A pairs scored       : {stats['qa_pairs_scored']}")
    print(f"  Errors                 : {stats['errors']}")
    print(f"  Total time             : {elapsed:.1f}s")
    print(f"  Sentences/sec          : {stats.get('sentences_per_sec', 0):.1f}")
    print(f"  Peak RAM               : {stats['total_ram_peak_gb']:.2f} GB")

    # Show sample results
    print_sample_results(limit=10)

    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

