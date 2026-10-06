"""
End-to-End Pipeline Orchestrator — scripts/run_pipeline.py

Primary entry point for running the Hidden Risk Analysis pipeline end-to-end
or stage-by-stage with full checkpoint/resume support, memory safety,
and configurable limits.

Supported stages:
  - ingest / preprocess : Ingest, preprocess, segment, and Q&A match transcripts into SQLite
  - risk                : Phase 2 linguistic risk engine (hedging, FinBERT, tone shift, evasiveness)
  - claims / evidence / verify / market : Phase 3 claim extraction, evidence retrieval, verification, and market reaction
  - all                 : Runs all stages in dependency order

Usage:
  python scripts/run_pipeline.py --stage all --limit 10
  python scripts/run_pipeline.py --stage ingest --limit 50
  python scripts/run_pipeline.py --stage risk --limit 50
  python scripts/run_pipeline.py --stage verify --limit 50
  python scripts/run_pipeline.py --stage all --force --limit 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import psutil

from src.checkpoint import CheckpointManager
from src.config import load_config
from src.data_loader import run_ingestion
from src.database import get_db_manager
from src.logger import get_logger
from src.phase3_engine import run_phase3
from src.risk_engine import run_phase2

logger = get_logger("pipeline_runner")


def _ram_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hidden Risk Analysis — End-to-End Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "ingest",
            "preprocess",
            "risk",
            "claims",
            "evidence",
            "verify",
            "market",
            "all",
        ],
        help="Pipeline stage to execute (default: all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N transcripts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already completed transcripts/stages.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from last checkpoint (default: True).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.yaml.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="huggingface",
        choices=["huggingface", "jsonl", "json"],
        help="Data source for ingestion (default: huggingface).",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Path to local dataset fixture (if source is jsonl or json).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for model inference (e.g. FinBERT).",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Filter to a specific company ticker.",
    )
    return parser.parse_args()


def run_stage_ingest(
    limit: Optional[int] = None,
    force: bool = False,
    source: str = "huggingface",
    local_path: Optional[str] = None,
) -> Dict:
    logger.info(">>> STAGE: Ingestion & Preprocessing (Phase 1)")
    path_obj = Path(local_path) if local_path else None
    return run_ingestion(limit=limit, source=source, local_path=path_obj, force=force)


def run_stage_risk(
    limit: Optional[int] = None,
    force: bool = False,
    batch_size: Optional[int] = None,
) -> Dict:
    logger.info(">>> STAGE: Linguistic Risk Engine (Phase 2)")
    return run_phase2(limit=limit, force_reprocess=force, batch_size=batch_size)


def run_stage_verification(
    limit: Optional[int] = None,
    force: bool = False,
) -> Dict:
    logger.info(">>> STAGE: Claims & Fact Verification (Phase 3)")
    return run_phase3(limit=limit, force_reprocess=force)


def main() -> int:
    args = parse_args()
    if args.config:
        load_config(args.config)
    db = get_db_manager()
    db.init_database()

    force = args.force

    print("\n" + "=" * 65)
    print("HIDDEN RISK ANALYSIS PIPELINE")
    print("=" * 65)
    print(f"  Stage      : {args.stage.upper()}")
    print(f"  Limit      : {args.limit if args.limit is not None else 'ALL'}")
    print(f"  Force      : {force}")
    print(f"  Source     : {args.source}")
    print(f"  Initial RAM: {_ram_gb():.2f} GB")
    print("=" * 65 + "\n")

    t0 = time.time()
    errors = 0

    try:
        if args.stage in ("ingest", "preprocess"):
            stats = run_stage_ingest(
                limit=args.limit,
                force=force,
                source=args.source,
                local_path=args.local_path,
            )
            print(f"\n[Ingestion Summary] Processed: {stats.get('transcripts_processed', 0)}, "
                  f"Skipped: {stats.get('transcripts_skipped', 0)}, "
                  f"Sentences: {stats.get('total_sentences', 0)}, "
                  f"QA pairs: {stats.get('total_qa_pairs', 0)}")

        elif args.stage == "risk":
            stats = run_stage_risk(
                limit=args.limit,
                force=force,
                batch_size=args.batch_size,
            )
            print(f"\n[Risk Summary] Processed: {stats.get('transcripts_processed', 0)}, "
                  f"Sentences Scored: {stats.get('sentences_scored', 0)}, "
                  f"QA Scored: {stats.get('qa_pairs_scored', 0)}")

        elif args.stage in ("claims", "evidence", "verify", "market"):
            stats = run_stage_verification(
                limit=args.limit,
                force=force,
            )
            print(f"\n[Verification Summary] Processed: {stats.get('transcripts_processed', 0)}, "
                  f"Claims: {stats.get('claims_extracted', 0)}, "
                  f"Verified: {stats.get('claims_verified', 0)}")

        elif args.stage == "all":
            # 1. Ingestion
            stats1 = run_stage_ingest(
                limit=args.limit,
                force=force,
                source=args.source,
                local_path=args.local_path,
            )
            print(f"[Stage 1/3 Complete] Transcripts in DB: {stats1.get('transcripts_processed', stats1.get('transcripts_in_db', 0))}")

            # 2. Risk Engine
            stats2 = run_stage_risk(
                limit=args.limit,
                force=force,
                batch_size=args.batch_size,
            )
            print(f"[Stage 2/3 Complete] Transcripts Scored: {stats2.get('transcripts_processed', 0)}")

            # 3. Verification
            stats3 = run_stage_verification(
                limit=args.limit,
                force=force,
            )
            print(f"[Stage 3/3 Complete] Claims Verified: {stats3.get('claims_verified', 0)}")

    except Exception as exc:
        logger.error(f"Pipeline execution failed: {exc}", exc_info=True)
        print(f"\n[ERROR] Pipeline aborted with exception: {exc}")
        return 1

    total_time = time.time() - t0
    peak_ram = _ram_gb()

    print("\n" + "=" * 65)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 65)
    print(f"  Total Duration : {total_time:.1f}s")
    print(f"  Peak RAM       : {peak_ram:.2f} GB")
    print("=" * 65 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
