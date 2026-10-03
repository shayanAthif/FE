"""
Phase 3 — Run Script

Usage:
  python scripts/run_phase3.py [--limit N] [--force] [--test]

Flags:
  --test      Run on 5 transcripts only (validation mode)
  --limit N   Process only first N pending transcripts
  --force     Re-process transcripts that already have claims
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.database import get_db_manager
from src.logger import get_logger
from src.phase3_engine import run_phase3

logger = get_logger("run_phase3")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3 — Real-World Claim + Evidence Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--test", action="store_true",
                        help="Quick test mode: process 5 transcripts only.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N pending transcripts.")
    parser.add_argument("--force", action="store_true",
                        help="Re-process transcripts that already have claims.")
    return parser.parse_args()


def print_summary():
    """Print database statistics for Phase 3 tables."""
    db_path = PROJECT_ROOT / "database" / "hidden_risk.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    print("\n" + "=" * 70)
    print("PHASE 3 VERIFICATION SUMMARY")
    print("=" * 70)

    # Claim status breakdown
    statuses = conn.execute(
        """SELECT status, COUNT(*) as cnt
           FROM verification
           GROUP BY status
           ORDER BY cnt DESC"""
    ).fetchall()

    total_ver = sum(r["cnt"] for r in statuses)
    print(f"\nTotal Verified Claims: {total_ver}")
    for r in statuses:
        pct = (r["cnt"] / total_ver * 100) if total_ver else 0
        print(f"  {r['status']:<25} : {r['cnt']:>5} ({pct:>5.1f}%)")

    # Evidence count by source type
    ev_types = conn.execute(
        """SELECT source_type, COUNT(*) as cnt
           FROM evidence
           GROUP BY source_type"""
    ).fetchall()
    print("\nEvidence Items by Source:")
    for r in ev_types:
        print(f"  {r['source_type'].upper():<25} : {r['cnt']:>5}")

    # Market events summary
    me_count = conn.execute("SELECT COUNT(*) FROM market_events").fetchone()[0]
    print(f"\nTranscripts with Market Reaction Data: {me_count}")

    # Sample verified claims
    print("\n" + "=" * 70)
    print("SAMPLE VERIFIED CLAIMS")
    print("=" * 70)
    samples = conn.execute(
        """SELECT c.claim_text, c.metric, c.period, v.status, v.confidence, v.reasoning
           FROM claims c
           JOIN verification v ON c.claim_id = v.claim_id
           ORDER BY v.confidence DESC
           LIMIT 5"""
    ).fetchall()

    for i, row in enumerate(samples, 1):
        print(f"\n{i}. [{row['status']}] (Conf: {row['confidence']:.2f}) | Metric: {row['metric']} | Period: {row['period']}")
        print(f"   Claim: \"{row['claim_text'][:120]}\"")
        print(f"   Reason: {row['reasoning'][:140]}...")

    conn.close()


def main():
    args = parse_args()
    cfg = load_config()

    db = get_db_manager()
    db.init_database()

    limit = 5 if args.test else args.limit

    print("\n" + "=" * 60)
    print("Phase 3 — Real-World Claim + Evidence Verification")
    print("=" * 60)
    print(f"  Mode   : {'TEST (5 transcripts)' if args.test else 'FULL'}")
    print(f"  Limit  : {limit or 'ALL'}")
    print(f"  Force  : {args.force}")
    print("=" * 60 + "\n")

    t0 = time.time()
    stats = run_phase3(limit=limit, force_reprocess=args.force)
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("PHASE 3 EXECUTION COMPLETE")
    print("=" * 60)
    print(f"  Transcripts Processed : {stats.get('transcripts_processed', 0)}")
    print(f"  Transcripts Skipped   : {stats.get('transcripts_skipped', 0)}")
    print(f"  Claims Extracted      : {stats.get('claims_extracted', 0)}")
    print(f"  Claims Verified       : {stats.get('claims_verified', 0)}")
    print(f"  Market Coverage       : {stats.get('market_coverage', 0)}")
    print(f"  Errors                : {stats.get('errors', 0)}")
    print(f"  Total Time            : {elapsed:.1f}s")
    print(f"  Peak RAM              : {stats.get('peak_ram_gb', 0):.2f} GB")

    print_summary()
    return 0 if stats.get("errors", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
