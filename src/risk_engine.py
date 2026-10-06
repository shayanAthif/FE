"""
Linguistic Risk Engine — Phase 2 Orchestrator

Processes transcripts already stored in the SQLite database (from Phase 1)
and computes all risk scores:

  For each transcript:
    1. Load sentences (with segment/section metadata)
    2. Run HedgingDetector on all sentences (rule-based, fast)
    3. Run FinBERT on all sentences in batches
    4. Run ToneShiftDetector using FinBERT outputs
    5. Look up Q&A pairs; run EvasivenessDetector on each pair
    6. Run RiskScorer to aggregate to sentence → segment → QA → transcript
    7. Persist results to risk_scores and transcript_scores tables
    8. Checkpoint progress

Memory strategy:
  - One transcript at a time
  - FinBERT sentences batched in chunks of batch_size
  - No full corpus in memory
  - Explicit gc.collect() between transcripts

Resume strategy:
  - Check transcript_scores table for already-scored transcripts
  - Skip them on restart (per-transcript checkpointing)
"""

from __future__ import annotations

import gc
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import psutil

from src.checkpoint import CheckpointManager
from src.config import get_config
from src.database import get_db_manager
from src.evasiveness_detector import EvasivenessDetector
from src.finbert_engine import load_finbert, run_sentiment_batch, unload_finbert
from src.hedging_detector import HedgingDetector
from src.logger import get_logger
from src.risk_scorer import RiskScorer
from src.tone_shift_detector import ToneShiftDetector

logger = get_logger("risk_engine")

JOB_NAME = "phase2_linguistic_risk"


def _ram_gb() -> float:
    """Return current process RSS in GB."""
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 ** 3)


class LinguisticRiskEngine:
    """
    Orchestrates the full Phase 2 linguistic risk pipeline for a corpus of
    transcripts stored in SQLite from Phase 1.
    """

    def __init__(self, db_path: Optional[str] = None, batch_size: Optional[int] = None):
        self.cfg = get_config()
        self.db = get_db_manager(db_path)
        self.checkpoint = CheckpointManager(self.db)
        self.batch_size = batch_size or self.cfg.processing.batch_size

        # Detectors (instantiated once, reused)
        self.hedging = HedgingDetector()
        self.tone_shift = ToneShiftDetector()
        self.evasiveness = EvasivenessDetector()
        self.scorer = RiskScorer()

        self._stats = {
            "transcripts_processed": 0,
            "transcripts_skipped": 0,
            "sentences_scored": 0,
            "qa_pairs_scored": 0,
            "errors": 0,
            "start_time": None,
            "total_ram_peak_gb": 0.0,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Database helpers
    # ─────────────────────────────────────────────────────────────────────

    def _get_all_transcript_ids(self) -> List[str]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT transcript_id FROM transcripts ORDER BY transcript_id"
            ).fetchall()
        return [r[0] for r in rows]

    def _get_already_scored_ids(self) -> Set[str]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT transcript_id FROM transcript_scores"
            ).fetchall()
        return {r[0] for r in rows}

    def _load_sentences(self, transcript_id: str) -> List[Dict]:
        """Load sentences with segment metadata for a transcript."""
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    s.sentence_id,
                    s.segment_id,
                    s.transcript_id,
                    s.sentence_number,
                    s.text,
                    seg.speaker,
                    seg.speaker_role,
                    seg.section,
                    seg.sequence as segment_sequence
                FROM sentences s
                JOIN segments seg ON s.segment_id = seg.segment_id
                WHERE s.transcript_id = ?
                ORDER BY seg.sequence, s.sentence_number
                """,
                (transcript_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def _load_qa_pairs(self, transcript_id: str) -> List[Dict]:
        """Load Q&A pairs for a transcript."""
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT qa_id, transcript_id, question_segment_id, answer_segment_id,
                       analyst_speaker, executive_speaker, question_text, answer_text, sequence
                FROM qa_pairs
                WHERE transcript_id = ?
                ORDER BY sequence
                """,
                (transcript_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def _load_answer_sentences(self, answer_segment_id: Optional[str]) -> List[Dict]:
        """Load scored sentences from a particular answer segment."""
        if not answer_segment_id:
            return []
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT rs.*, s.text
                FROM risk_scores rs
                JOIN sentences s ON rs.sentence_id = s.sentence_id
                WHERE rs.segment_id = ?
                """,
                (answer_segment_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────

    def _save_sentence_scores(
        self,
        scored_sentences: List[Dict],
        conn,
    ) -> None:
        """Insert sentence-level risk scores (batch insert for speed)."""
        rows = []
        for s in scored_sentences:
            score_id = f"{s['sentence_id']}_rs"
            factors = json.dumps(s.get("contributing_factors", {}), default=str)
            rows.append((
                score_id,
                s.get("sentence_id"),
                s.get("segment_id"),
                s.get("transcript_id"),
                s.get("hedging_score", 0.0),
                s.get("hedge_density", 0.0),
                s.get("uncertainty_score", 0.0),
                s.get("positive_prob", 0.0),
                s.get("negative_prob", 0.0),
                s.get("neutral_prob", 0.0),
                s.get("sentiment_label", "neutral"),
                s.get("tone_shift_score", 0.0),
                s.get("evasive_score", 0.0),
                s.get("hidden_risk_score", 0.0),
                factors,
            ))

        conn.executemany(
            """
            INSERT OR REPLACE INTO risk_scores (
                score_id, sentence_id, segment_id, transcript_id,
                hedging_score, hedge_density, uncertainty_score,
                positive_prob, negative_prob, neutral_prob, sentiment_label,
                tone_shift_score, evasiveness_score, hidden_risk_score,
                contributing_factors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _save_transcript_score(self, transcript_id: str, ts: Dict, conn) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO transcript_scores (
                transcript_id, overall_hidden_risk, average_risk, top_risk_average,
                qa_risk, prepared_risk, highest_risk_segment,
                hedging_avg, evasiveness_avg, tone_shift_avg, computed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transcript_id,
                ts.get("overall_hidden_risk", 0.0),
                ts.get("average_risk", 0.0),
                ts.get("top_risk_average", 0.0),
                ts.get("qa_risk", 0.0),
                ts.get("prepared_risk", 0.0),
                ts.get("highest_risk_segment", ""),
                ts.get("hedging_avg", 0.0),
                ts.get("evasiveness_avg", 0.0),
                ts.get("tone_shift_avg", 0.0),
                datetime.now().isoformat(),
            ),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Single-transcript processing
    # ─────────────────────────────────────────────────────────────────────

    def _process_transcript(self, transcript_id: str) -> Optional[Dict]:
        """
        Run the full linguistic risk pipeline for one transcript.
        Returns the transcript_score dict or None on failure.
        """
        t0 = time.time()

        # ── Step 1: Load sentences ────────────────────────────────────────
        sentences = self._load_sentences(transcript_id)
        if not sentences:
            logger.warning(f"[{transcript_id}] No sentences found, skipping.")
            return None

        texts = [s["text"] for s in sentences]
        logger.info(f"[{transcript_id}] {len(sentences)} sentences loaded.")

        # ── Step 2: Hedging detector (fast, no model) ─────────────────────
        hedging_results = self.hedging.analyse_batch(texts)

        # ── Step 3: FinBERT sentiment (batched) ───────────────────────────
        sentiment_results = run_sentiment_batch(
            texts,
            batch_size=self.batch_size,
            max_length=self.cfg.processing.max_length,
        )

        # ── Step 4: Tone shift (uses FinBERT outputs) ─────────────────────
        # Build sequence with section metadata for tone_shift
        sentiment_sequence = []
        for s, sent in zip(sentences, sentiment_results):
            sentiment_sequence.append({
                "sentence_id": s["sentence_id"],
                "segment_id": s["segment_id"],
                "section": s["section"],
                "speaker_role": s["speaker_role"],
                **sent,
            })
        enriched_sentiment, section_metrics = self.tone_shift.analyse_transcript(
            sentiment_sequence
        )

        # ── Step 5: Merge all features per sentence ───────────────────────
        merged_sentences = []
        for s, hedge, enriched in zip(sentences, hedging_results, enriched_sentiment):
            merged = {
                **s,            # sentence_id, segment_id, text, section, speaker_role, …
                **hedge,        # hedging_score, hedge_density, …
                **enriched,     # positive_prob, tone_shift_score, …
                "evasive_score": 0.0,  # filled in below for QA sentences
            }
            merged_sentences.append(merged)

        # ── Step 6: Evasiveness on Q&A pairs ─────────────────────────────
        qa_pairs = self._load_qa_pairs(transcript_id)
        qa_scored: List[Dict] = []

        # Build a segment_id → hedge_density lookup
        segment_hedge: Dict[str, float] = {}
        for s in merged_sentences:
            sid = s["segment_id"]
            if sid not in segment_hedge:
                segment_hedge[sid] = s.get("hedge_density", 0.0)

        for qa in qa_pairs:
            q_text = qa.get("question_text", "")
            a_text = qa.get("answer_text", "")
            answer_seg_id = qa.get("answer_segment_id", "")

            hedge_density = segment_hedge.get(answer_seg_id, 0.0)
            eva_result = self.evasiveness.analyse(q_text, a_text, hedge_density)

            # Propagate evasive_score to sentences in the answer segment
            for ms in merged_sentences:
                if ms["segment_id"] == answer_seg_id:
                    ms["evasive_score"] = eva_result.get("evasive_score", 0.0)

            qa_with_id = {
                "qa_id": qa["qa_id"],
                "transcript_id": transcript_id,
                **eva_result,
            }
            qa_scored.append(qa_with_id)

        # ── Step 7: Risk scoring ───────────────────────────────────────────
        final_sentences = [self.scorer.score_sentence(s) for s in merged_sentences]

        # Q&A-level scores
        qa_risk_scores: List[Dict] = []
        for qa, qa_eva in zip(qa_pairs, qa_scored):
            answer_seg_id = qa.get("answer_segment_id", "")
            answer_sents = [s for s in final_sentences if s["segment_id"] == answer_seg_id]
            qa_score = self.scorer.score_qa_pair(qa_eva, answer_sents)
            qa_score["qa_id"] = qa["qa_id"]
            qa_risk_scores.append(qa_score)

        # Transcript-level score
        transcript_score = self.scorer.aggregate_transcript(
            final_sentences, qa_risk_scores, section_metrics
        )
        transcript_score["transcript_id"] = transcript_id
        transcript_score["processing_time_sec"] = round(time.time() - t0, 2)

        # ── Step 8: Persist ───────────────────────────────────────────────
        with self.db.transaction() as conn:
            self._save_sentence_scores(final_sentences, conn)
            self._save_transcript_score(transcript_id, transcript_score, conn)

        self._stats["sentences_scored"] += len(final_sentences)
        self._stats["qa_pairs_scored"] += len(qa_risk_scores)
        elapsed = time.time() - t0
        ram = _ram_gb()
        self._stats["total_ram_peak_gb"] = max(self._stats["total_ram_peak_gb"], ram)

        logger.info(
            f"[{transcript_id}] Done. "
            f"risk={transcript_score['overall_hidden_risk']:.1f} | "
            f"sentences={len(final_sentences)} | qa={len(qa_risk_scores)} | "
            f"time={elapsed:.1f}s | RAM={ram:.2f}GB"
        )
        return transcript_score

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        limit: Optional[int] = None,
        force_reprocess: bool = False,
    ) -> Dict:
        """
        Process all (or limit) transcripts through the linguistic risk pipeline.

        Parameters
        ----------
        limit           : if set, process at most this many transcripts
        force_reprocess : if True, re-score even already-scored transcripts

        Returns
        -------
        Processing statistics dict.
        """
        self._stats["start_time"] = time.time()

        all_ids = self._get_all_transcript_ids()
        scored_ids = self._get_already_scored_ids() if not force_reprocess else set()

        pending = [tid for tid in all_ids if tid not in scored_ids]
        if limit:
            pending = pending[:limit]

        logger.info(
            f"Phase 2 starting: {len(pending)} transcripts to process "
            f"({len(scored_ids)} already scored, force={force_reprocess})."
        )

        # Load FinBERT once for the entire run
        logger.info("Loading FinBERT model …")
        load_finbert()

        self.checkpoint.set_checkpoint(JOB_NAME, last_processed_id=None, status="IN_PROGRESS")

        for i, tid in enumerate(pending):
            ram = _ram_gb()

            try:
                result = self._process_transcript(tid)
                if result is not None:
                    self._stats["transcripts_processed"] += 1
                    self.checkpoint.set_checkpoint(
                        JOB_NAME, last_processed_id=tid, status="IN_PROGRESS"
                    )
                else:
                    self._stats["transcripts_skipped"] += 1

            except Exception as exc:
                self._stats["errors"] += 1
                logger.error(f"[{tid}] Error during processing: {exc}", exc_info=True)
                self.checkpoint.set_checkpoint(
                    JOB_NAME, last_processed_id=tid,
                    status="IN_PROGRESS",
                    error_message=str(exc),
                )
                # Continue to next transcript instead of aborting
                continue

            finally:
                # Explicit GC between transcripts
                gc.collect()

            # Log progress every 10 transcripts
            if (i + 1) % 10 == 0:
                elapsed = time.time() - self._stats["start_time"]
                logger.info(
                    f"Progress: {i+1}/{len(pending)} | "
                    f"scored={self._stats['transcripts_processed']} | "
                    f"errors={self._stats['errors']} | "
                    f"elapsed={elapsed:.0f}s | RAM={_ram_gb():.2f}GB"
                )

        self.checkpoint.mark_completed(JOB_NAME, last_processed_id=pending[-1] if pending else None)

        # Final stats
        elapsed = time.time() - self._stats["start_time"]
        self._stats["total_elapsed_sec"] = round(elapsed, 1)
        self._stats["sentences_per_sec"] = round(
            self._stats["sentences_scored"] / max(elapsed, 1), 1
        )

        logger.info(
            f"Phase 2 complete. "
            f"Processed={self._stats['transcripts_processed']} | "
            f"Skipped={self._stats['transcripts_skipped']} | "
            f"Errors={self._stats['errors']} | "
            f"Sentences={self._stats['sentences_scored']} | "
            f"QA={self._stats['qa_pairs_scored']} | "
            f"Time={elapsed:.1f}s | "
            f"Peak RAM={self._stats['total_ram_peak_gb']:.2f}GB"
        )
        return self._stats


def run_phase2(
    limit: Optional[int] = None,
    db_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    force_reprocess: bool = False,
) -> Dict:
    """
    Convenience function — create engine and run Phase 2.

    Parameters
    ----------
    limit          : process at most this many transcripts (None = all)
    db_path        : override database path
    batch_size     : override FinBERT batch size
    force_reprocess: re-score already-scored transcripts

    Returns
    -------
    Processing statistics dict.
    """
    engine = LinguisticRiskEngine(db_path=db_path, batch_size=batch_size)
    return engine.run(limit=limit, force_reprocess=force_reprocess)
