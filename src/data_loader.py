"""
Data Loader — Phase 1

Responsibilities:
  1. Load transcripts from the Hugging Face dataset (Bose345/sp500_earnings_transcripts)
     using streaming mode so the full corpus never enters RAM.
  2. Normalize and validate each transcript record.
  3. Produce deterministic transcript IDs.
  4. Detect and skip duplicates.
  5. Call preprocessing → segmentation → Q&A matching.
  6. Persist results to SQLite using bounded transactions.
  7. Support full checkpoint/resume so that restarting continues from the
     last safe point.
  8. Support loading from a local JSON/JSONL fixture for testing.

Memory strategy
---------------
- One transcript is processed at a time.
- All intermediate lists (segments, sentences, qa_pairs) are released after
  each transcript's transaction commits.
- The set of already-processed IDs is fetched once at startup and stored as
  a compact Python set (just string IDs, not full records).

Public interface
----------------
  run_ingestion(limit, force, ...)   — primary pipeline entry point
  load_single_transcript(record)     — process one record dict
  iter_transcripts(source, ...)      — generator over raw records
  make_transcript_id(record)         — deterministic ID builder
"""

from __future__ import annotations

import gc
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set

import psutil

from src.checkpoint import CheckpointManager
from src.config import PROJECT_ROOT, get_config
from src.database import DatabaseManager, get_db_manager
from src.logger import get_logger
from src.preprocessing import preprocess_transcript_record
from src.qa_matcher import match_qa_pairs
from src.segmentation import segment_transcript

logger = get_logger("data_loader")

JOB_NAME = "phase1_ingestion"

# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────

_VALID_QUARTERS = frozenset({"Q1", "Q2", "Q3", "Q4"})


def _validate_record(record: Dict) -> Optional[str]:
    """
    Validate a raw transcript record.

    Returns None if the record is valid, or an error message string if invalid.
    """
    if not record.get("symbol") and not record.get("ticker"):
        return "missing symbol/ticker"
    if not record.get("date"):
        return "missing date"
    if not record.get("content") and not record.get("structured_content"):
        return "empty transcript (no content or structured_content)"

    # Validate year
    year = record.get("year")
    if year is not None:
        try:
            y = int(year)
            if not (1990 <= y <= 2030):
                return f"invalid year: {year}"
        except (TypeError, ValueError):
            return f"non-integer year: {year!r}"

    # Validate quarter
    quarter = record.get("quarter", "")
    if quarter and str(quarter).upper() not in _VALID_QUARTERS:
        # Some datasets use "1", "2", etc. — normalise
        q_norm = _normalise_quarter(quarter)
        if not q_norm:
            return f"invalid quarter: {quarter!r}"

    return None


def _normalise_quarter(raw: Any) -> Optional[str]:
    """Convert raw quarter value to Q1/Q2/Q3/Q4 or return None."""
    if not raw:
        return None
    s = str(raw).strip().upper()
    if s in _VALID_QUARTERS:
        return s
    if s in ("1", "2", "3", "4"):
        return f"Q{s}"
    if s.startswith("Q") and s[1:] in ("1", "2", "3", "4"):
        return s[:2]
    return None


def make_transcript_id(record: Dict) -> str:
    """
    Build a deterministic transcript ID from key metadata fields.

    Format: <TICKER>_<YEAR>_<QUARTER>_<DATE>

    Example: AAPL_2024_Q2_2024-05-02

    Falls back gracefully when fields are missing.
    """
    ticker = (
        record.get("symbol") or record.get("ticker") or "UNKNOWN"
    ).upper().strip()

    year = record.get("year", "")
    quarter = _normalise_quarter(record.get("quarter", "")) or record.get("quarter", "")
    date = record.get("date", "")

    parts = [ticker]
    if year:
        parts.append(str(year))
    if quarter:
        parts.append(str(quarter).upper())
    if date:
        parts.append(str(date))

    return "_".join(parts)


def _extract_ticker(record: Dict) -> str:
    return (record.get("symbol") or record.get("ticker") or "UNKNOWN").upper().strip()


# ──────────────────────────────────────────────────────────────────────────────
# Record iterators
# ──────────────────────────────────────────────────────────────────────────────

def iter_from_huggingface(
    dataset_name: str,
    split: str = "train",
    streaming: bool = True,
    limit: Optional[int] = None,
) -> Generator[Dict, None, None]:
    """
    Stream records from a HuggingFace dataset.

    Uses streaming mode to avoid loading the full corpus into RAM.
    Yields raw record dicts.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace ingestion. "
            "Install it with: pip install datasets"
        )

    logger.info(f"Loading dataset '{dataset_name}' (split={split}, streaming={streaming}) …")
    ds = load_dataset(dataset_name, split=split, streaming=streaming, trust_remote_code=True)

    count = 0
    for record in ds:
        yield dict(record)
        count += 1
        if limit is not None and count >= limit:
            break


def iter_from_jsonl(
    path: Path,
    limit: Optional[int] = None,
) -> Generator[Dict, None, None]:
    """
    Iterate over records from a local JSON Lines (.jsonl) file.

    Each line must be a JSON object matching the HF dataset schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fixture file not found: {path}")

    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                yield record
                count += 1
                if limit is not None and count >= limit:
                    break
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSONL line: {e}")


def iter_from_json(
    path: Path,
    limit: Optional[int] = None,
) -> Generator[Dict, None, None]:
    """
    Iterate over records from a local JSON file (array or single object).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON fixture not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    count = 0
    for record in data:
        yield record
        count += 1
        if limit is not None and count >= limit:
            break


def iter_transcripts(
    source: str = "huggingface",
    local_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Generator[Dict, None, None]:
    """
    Unified iterator over transcript records from any supported source.

    Parameters
    ----------
    source      : 'huggingface' | 'jsonl' | 'json'
    local_path  : required for 'jsonl' and 'json' sources
    limit       : stop after this many records
    """
    cfg = get_config()

    if source == "huggingface":
        yield from iter_from_huggingface(
            dataset_name=cfg.dataset.name,
            split=cfg.dataset.split,
            streaming=cfg.dataset.streaming,
            limit=limit,
        )
    elif source == "jsonl":
        if not local_path:
            raise ValueError("local_path required for 'jsonl' source")
        yield from iter_from_jsonl(local_path, limit=limit)
    elif source == "json":
        if not local_path:
            raise ValueError("local_path required for 'json' source")
        yield from iter_from_json(local_path, limit=limit)
    else:
        raise ValueError(f"Unknown source: {source!r}. Choose from huggingface, jsonl, json.")


# ──────────────────────────────────────────────────────────────────────────────
# DB persistence helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_already_processed_ids(db: DatabaseManager) -> Set[str]:
    """
    Return the set of transcript_ids already in the transcripts table.

    Fetches only IDs (not full rows) so memory usage is minimal even for
    large corpora.
    """
    with db.connection() as conn:
        rows = conn.execute("SELECT transcript_id FROM transcripts").fetchall()
    return {r[0] for r in rows}


def _insert_transcript(conn: sqlite3.Connection, transcript_id: str, record: Dict) -> None:
    """Insert a transcript metadata row (upsert)."""
    ticker = _extract_ticker(record)
    quarter = _normalise_quarter(record.get("quarter")) or str(record.get("quarter", ""))
    year = int(record.get("year", 0)) if record.get("year") else 0

    conn.execute(
        """
        INSERT OR REPLACE INTO transcripts
            (transcript_id, ticker, company_name, company_id, date,
             year, quarter, sector, raw_text, processing_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            transcript_id,
            ticker,
            record.get("company_name", ""),
            record.get("company_id", ""),
            record.get("date", ""),
            year,
            quarter,
            record.get("sector", ""),
            # Store a truncated version of raw text (saves space, preserves traceability)
            (record.get("content") or "")[:5000],
            "PROCESSED",
        ),
    )


def _insert_segments(conn: sqlite3.Connection, segments: List[Dict]) -> None:
    """Batch-insert segment rows."""
    conn.executemany(
        """
        INSERT OR REPLACE INTO segments
            (segment_id, transcript_id, speaker, speaker_role, section, text, sequence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                s["segment_id"],
                s["transcript_id"],
                s["speaker"],
                s["speaker_role"],
                s["section"],
                s["text"],
                s["sequence"],
            )
            for s in segments
        ],
    )


def _insert_sentences(conn: sqlite3.Connection, sentences: List[Dict]) -> None:
    """Batch-insert sentence rows."""
    conn.executemany(
        """
        INSERT OR REPLACE INTO sentences
            (sentence_id, segment_id, transcript_id, sentence_number, text)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                s["sentence_id"],
                s["segment_id"],
                s["transcript_id"],
                s["sentence_number"],
                s["text"],
            )
            for s in sentences
        ],
    )


def _insert_qa_pairs(conn: sqlite3.Connection, qa_pairs: List[Dict]) -> None:
    """Batch-insert Q&A pair rows."""
    conn.executemany(
        """
        INSERT OR REPLACE INTO qa_pairs
            (qa_id, transcript_id, question_segment_id, answer_segment_id,
             analyst_speaker, executive_speaker, question_text, answer_text, sequence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                q["qa_id"],
                q["transcript_id"],
                q.get("question_segment_id"),
                q.get("answer_segment_id"),
                q.get("analyst_speaker", ""),
                q.get("executive_speaker", ""),
                q["question_text"],
                q["answer_text"],
                q["sequence"],
            )
            for q in qa_pairs
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Single-transcript processing
# ──────────────────────────────────────────────────────────────────────────────

def load_single_transcript(
    record: Dict,
    db: DatabaseManager,
    *,
    force: bool = False,
    already_processed: Optional[Set[str]] = None,
) -> Dict:
    """
    Process one transcript record end-to-end and persist to DB.

    Parameters
    ----------
    record            : raw transcript dict
    db                : DatabaseManager instance
    force             : if True, reprocess even if already in DB
    already_processed : set of IDs already in DB (checked to skip duplicates)

    Returns
    -------
    Result dict: {transcript_id, status, segments, sentences, qa_pairs, error}
    """
    transcript_id = make_transcript_id(record)

    # Duplicate check
    if not force and already_processed is not None and transcript_id in already_processed:
        return {
            "transcript_id": transcript_id,
            "status": "skipped_duplicate",
            "segments": 0,
            "sentences": 0,
            "qa_pairs": 0,
        }

    # Validation
    err = _validate_record(record)
    if err:
        logger.warning(f"[{transcript_id}] Invalid record: {err}")
        return {
            "transcript_id": transcript_id,
            "status": f"invalid: {err}",
            "segments": 0,
            "sentences": 0,
            "qa_pairs": 0,
        }

    try:
        # Step 1: preprocess
        record = preprocess_transcript_record(record, transcript_id)

        # Step 2: segment + sentence split
        segments, sentences = segment_transcript(transcript_id, record)

        if not segments:
            logger.warning(f"[{transcript_id}] No segments produced; skipping.")
            return {
                "transcript_id": transcript_id,
                "status": "skipped_no_segments",
                "segments": 0,
                "sentences": 0,
                "qa_pairs": 0,
            }

        # Step 3: Q&A matching
        qa_pairs = match_qa_pairs(transcript_id, segments)

        # Step 4: persist in a single transaction
        with db.transaction() as conn:
            _insert_transcript(conn, transcript_id, record)
            _insert_segments(conn, segments)
            _insert_sentences(conn, sentences)
            _insert_qa_pairs(conn, qa_pairs)

        logger.info(
            f"[{transcript_id}] Persisted: "
            f"{len(segments)} segs, {len(sentences)} sents, {len(qa_pairs)} QA pairs."
        )

        # Update the in-memory processed set to avoid rescanning DB
        if already_processed is not None:
            already_processed.add(transcript_id)

        return {
            "transcript_id": transcript_id,
            "status": "ok",
            "segments": len(segments),
            "sentences": len(sentences),
            "qa_pairs": len(qa_pairs),
        }

    except Exception as exc:
        logger.error(f"[{transcript_id}] Error: {exc}", exc_info=True)
        return {
            "transcript_id": transcript_id,
            "status": f"error: {exc}",
            "segments": 0,
            "sentences": 0,
            "qa_pairs": 0,
            "error": str(exc),
        }
    finally:
        # Release memory held by this transcript's lists
        gc.collect()


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_ingestion(
    limit: Optional[int] = None,
    source: str = "huggingface",
    local_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    force: bool = False,
) -> Dict:
    """
    Run the Phase 1 data ingestion pipeline.

    Parameters
    ----------
    limit       : stop after this many transcripts (None = no limit)
    source      : 'huggingface' | 'jsonl' | 'json'
    local_path  : path to local fixture file (required if source != 'huggingface')
    db_path     : override default database path
    force       : reprocess transcripts already in DB

    Returns
    -------
    Statistics dict.
    """
    cfg = get_config()
    db = get_db_manager(db_path)
    checkpoint = CheckpointManager(db)

    # Ensure schema exists
    db.init_database()

    # Check whether stage already completed (unless force)
    if not force and checkpoint.is_job_completed(JOB_NAME):
        logger.info("Phase 1 ingestion already marked COMPLETED. Use --force to rerun.")
        # Still return counts from DB
        with db.connection() as conn:
            t_count = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
            s_count = conn.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
        return {
            "status": "already_completed",
            "transcripts_in_db": t_count,
            "sentences_in_db": s_count,
        }

    # Fetch already-processed IDs (skip on force)
    already_processed: Set[str] = set() if force else _get_already_processed_ids(db)
    logger.info(
        f"Phase 1 starting | source={source} | limit={limit} | "
        f"force={force} | already_in_db={len(already_processed)}"
    )

    # Mark job in-progress
    checkpoint.set_checkpoint(JOB_NAME, last_processed_id=None, status="IN_PROGRESS")

    stats = {
        "transcripts_processed": 0,
        "transcripts_skipped": 0,
        "transcripts_invalid": 0,
        "transcripts_errored": 0,
        "total_segments": 0,
        "total_sentences": 0,
        "total_qa_pairs": 0,
        "peak_ram_gb": 0.0,
        "start_time": time.time(),
    }

    proc = psutil.Process()
    checkpoint_interval = cfg.processing.checkpoint_interval
    last_tid: Optional[str] = None

    for idx, raw_record in enumerate(iter_transcripts(source, local_path, limit)):
        # RAM guard
        ram_gb = proc.memory_info().rss / (1024 ** 3)
        stats["peak_ram_gb"] = max(stats["peak_ram_gb"], ram_gb)
        if ram_gb > cfg.memory.warning_ram_gb:
            logger.warning(f"RAM at {ram_gb:.2f} GB — approaching limit.")

        result = load_single_transcript(
            raw_record,
            db,
            force=force,
            already_processed=already_processed,
        )

        status = result.get("status", "")
        tid = result.get("transcript_id", "")
        last_tid = tid

        if status == "ok":
            stats["transcripts_processed"] += 1
            stats["total_segments"] += result.get("segments", 0)
            stats["total_sentences"] += result.get("sentences", 0)
            stats["total_qa_pairs"] += result.get("qa_pairs", 0)
        elif status == "skipped_duplicate":
            stats["transcripts_skipped"] += 1
        elif status.startswith("invalid"):
            stats["transcripts_invalid"] += 1
        else:
            stats["transcripts_errored"] += 1

        # Periodic checkpoint
        if (idx + 1) % checkpoint_interval == 0:
            checkpoint.set_checkpoint(JOB_NAME, last_processed_id=last_tid, status="IN_PROGRESS")
            elapsed = time.time() - stats["start_time"]
            logger.info(
                f"Progress [{idx+1}]: processed={stats['transcripts_processed']} "
                f"| skipped={stats['transcripts_skipped']} "
                f"| errors={stats['transcripts_errored']} "
                f"| elapsed={elapsed:.0f}s | RAM={ram_gb:.2f}GB"
            )

    # Mark completed
    checkpoint.mark_completed(JOB_NAME, last_processed_id=last_tid)

    elapsed = time.time() - stats["start_time"]
    stats["total_elapsed_sec"] = round(elapsed, 1)
    stats.pop("start_time", None)

    logger.info(
        f"Phase 1 complete. "
        f"processed={stats['transcripts_processed']} | "
        f"skipped={stats['transcripts_skipped']} | "
        f"invalid={stats['transcripts_invalid']} | "
        f"errors={stats['transcripts_errored']} | "
        f"sentences={stats['total_sentences']} | "
        f"qa_pairs={stats['total_qa_pairs']} | "
        f"time={elapsed:.1f}s | RAM_peak={stats['peak_ram_gb']:.2f}GB"
    )
    return stats
