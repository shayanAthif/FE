"""
Phase 3 Orchestrator — Real-World Claim + Evidence Verification

Processes transcripts one at a time through the full pipeline:
  1. Load sentences from DB
  2. Extract candidate claims
  3. Normalize claims
  4. Retrieve SEC filings evidence
  5. Retrieve news evidence
  6. Compute market reactions
  7. Verify claims against temporally-filtered evidence
  8. Persist claims, evidence, verification, market_events to SQLite
  9. Checkpoint progress per transcript

Temporal safety:
  - Evidence for verification is labelled CONTEMPORANEOUS / POST_CALL / OUTCOME.
  - POST_CALL and OUTCOME evidence only is used for claim verification.
  - No evidence is fed back into Hidden Risk Score computation.

Usage:
  from src.phase3_engine import run_phase3
  stats = run_phase3(limit=5)   # test on 5 transcripts
  stats = run_phase3()           # full 100-transcript run
"""

from __future__ import annotations

import gc
import json
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import psutil

from src.checkpoint import CheckpointManager
from src.claim_extractor import ClaimExtractor
from src.claim_normalizer import ClaimNormalizer
from src.config import PROJECT_ROOT, get_config
from src.database import get_db_manager
from src.fact_verifier import FactVerifier
from src.logger import get_logger
from src.providers.market.market_provider import MarketProvider
from src.providers.news.news_provider import NewsProvider
from src.providers.sec.sec_provider import SecProvider

logger = get_logger("phase3_engine")
JOB_NAME = "phase3_verification"

# Max claims extracted per transcript to control API rate pressure.
# Purpose: prevents unbounded memory growth and API quota exhaustion during
# large-scale runs.  Increase with caution.
# Configure via config.yaml:  claims.max_claims_per_transcript: 20
MAX_CLAIMS_PER_TRANSCRIPT = 20
# Evidence search window: ±180 days around call for SEC filings, post-call for news
SEC_WINDOW_DAYS_BEFORE = 180
SEC_WINDOW_DAYS_AFTER  = 365
NEWS_WINDOW_DAYS_AFTER = 180


def _ram_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


class Phase3Engine:
    """Orchestrates the claim extraction, evidence retrieval and fact verification pipeline."""

    def __init__(self):
        self.cfg      = get_config()
        self.db       = get_db_manager()
        self.ckpt     = CheckpointManager()
        self.extractor  = ClaimExtractor()
        self.normalizer = ClaimNormalizer()
        self.verifier   = FactVerifier()
        self.sec_provider    = SecProvider()
        self.news_provider   = NewsProvider()
        self.market_provider = MarketProvider()

        # Max claims per transcript (configurable; falls back to module constant).
        # Read from config.claims.max_claims_per_transcript when present.
        cfg_claims = getattr(self.cfg, "claims", None)
        self.max_claims = (
            getattr(cfg_claims, "max_claims_per_transcript", None)
            or MAX_CLAIMS_PER_TRANSCRIPT
        )

    # ─────────────────────────────────────────────────────────────────────
    # DB helpers
    # ─────────────────────────────────────────────────────────────────────

    def _get_pending_transcripts(
        self,
        limit: Optional[int],
        force_reprocess: bool = False,
    ) -> List[Dict]:
        """
        Return transcripts that still need Phase 3 processing.

        force_reprocess=False (default / resume mode):
            Transcripts that already have at least one claim in the claims
            table are *excluded* — they have already been processed and will
            be skipped (standard checkpoint/resume behaviour).

        force_reprocess=True (--force flag):
            ALL transcripts are returned so every transcript is reprocessed.
            Existing claims/evidence/verification data for each transcript
            will be deleted before reinserting (see _delete_existing_claims).
        """
        conn = sqlite3.connect(str(PROJECT_ROOT / "database" / "hidden_risk.db"))
        conn.row_factory = sqlite3.Row
        all_trans = conn.execute(
            "SELECT transcript_id, ticker, date, company_name FROM transcripts ORDER BY date"
        ).fetchall()

        if force_reprocess:
            # Include every transcript; existing data will be wiped and replaced
            pending = [dict(row) for row in all_trans]
        else:
            already_done = set(
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT transcript_id FROM claims"
                ).fetchall()
            )
            pending = [
                dict(row)
                for row in all_trans
                if row["transcript_id"] not in already_done
            ]

        conn.close()

        if limit:
            pending = pending[:limit]
        return pending

    def _delete_existing_claims(self, transcript_id: str) -> None:
        """
        Remove all existing Phase 3 data for a transcript.
        Called only when force_reprocess=True so that reprocessing produces
        a clean result instead of accumulating duplicate rows.
        """
        with self.db.transaction() as conn:
            conn.execute(
                "DELETE FROM market_events WHERE transcript_id = ?", (transcript_id,)
            )
            # evidence and verification cascade-delete when claims are deleted
            conn.execute(
                "DELETE FROM claims WHERE transcript_id = ?", (transcript_id,)
            )


    def _get_sentences(self, transcript_id: str) -> List[Dict]:
        conn = sqlite3.connect(str(PROJECT_ROOT / "database" / "hidden_risk.db"))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT s.sentence_id, s.segment_id, s.text,
                   seg.section, seg.speaker, seg.speaker_role
            FROM sentences s
            JOIN segments seg ON s.segment_id = seg.segment_id
            WHERE s.transcript_id = ?
            ORDER BY seg.sequence, s.sentence_number
            """,
            (transcript_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def _save_claims_and_results(
        self,
        transcript_id: str,
        normalized_claims: List,
        verifications: List[Dict],
        evidence_map: Dict,
        market_result,
    ) -> None:
        conn = sqlite3.connect(str(PROJECT_ROOT / "database" / "hidden_risk.db"))
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            # Upsert claims
            for c in normalized_claims:
                conn.execute(
                    """INSERT OR REPLACE INTO claims
                       (claim_id, transcript_id, segment_id, sentence_id, speaker,
                        claim_text, claim_type, entity, metric, value, lower_bound,
                        upper_bound, unit, period, geography, direction, confidence)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (c.claim_id, c.transcript_id, c.segment_id, c.sentence_id, c.speaker,
                     c.claim_text[:800], c.claim_type, c.entity, c.metric,
                     c.value, c.lower_bound, c.upper_bound, c.unit,
                     c.period, c.geography, c.direction, c.confidence),
                )

                # Save evidence for this claim
                for ev in evidence_map.get(c.claim_id, []):
                    ev_id = f"{c.claim_id}_{hash(ev.source_url or ev.title) & 0xFFFFFF:06x}"
                    conn.execute(
                        """INSERT OR REPLACE INTO evidence
                           (evidence_id, claim_id, source_type, source_name, source_url,
                            filing_type, publication_date, title, text, relevance_score,
                            authority_score)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (ev_id, c.claim_id, ev.source_type, ev.source_name, ev.source_url,
                         ev.filing_type, ev.publication_date,
                         (ev.title or "")[:200],
                         (ev.text or "")[:800],
                         ev.relevance_score, ev.authority_score),
                    )

            # Save verifications
            for v in verifications:
                conn.execute(
                    """INSERT OR REPLACE INTO verification
                       (verification_id, claim_id, status, confidence, reasoning, evidence_ids)
                       VALUES (?,?,?,?,?,?)""",
                    (v["verification_id"], v["claim_id"], v["status"],
                     v["confidence"], v["reasoning"],
                     json.dumps(v.get("evidence_ids", []))),
                )

            # Save market events
            if market_result:
                mr = market_result
                conn.execute(
                    """INSERT OR REPLACE INTO market_events
                       (transcript_id, ticker, call_date, price_before, price_after,
                        return_1d, return_5d, return_10d, return_20d,
                        volatility_5d, volatility_10d,
                        benchmark_return_5d, abnormal_return_5d)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (transcript_id, mr.ticker, mr.call_date,
                     mr.price_before, mr.price_after,
                     mr.returns.get("1d"), mr.returns.get("5d"),
                     mr.returns.get("10d"), mr.returns.get("20d"),
                     mr.volatilities.get("5d"), mr.volatilities.get("10d"),
                     mr.benchmark_returns.get("5d"),
                     mr.abnormal_returns.get("5d")),
                )

            conn.commit()
        finally:
            conn.close()

    # ─────────────────────────────────────────────────────────────────────
    # Single transcript processing
    # ─────────────────────────────────────────────────────────────────────

    def process_transcript(self, transcript: Dict, force_reprocess: bool = False) -> Dict:
        tid       = transcript["transcript_id"]
        ticker    = transcript.get("ticker", "")
        call_date = transcript.get("date", "")
        t0        = time.time()

        try:
            # If forced, wipe existing Phase 3 data first so we get a clean rewrite
            if force_reprocess:
                self._delete_existing_claims(tid)

            # 1. Load sentences
            sentences = self._get_sentences(tid)
            if not sentences:
                return {"transcript_id": tid, "status": "skipped_no_sentences",
                        "claims": 0, "verified": 0, "errors": 0}

            # 2. Extract claims (capped at self.max_claims for memory safety)
            raw_claims = self.extractor.extract_claims_from_sentences(tid, sentences)
            # Sort by confidence descending so the most important claims are kept
            raw_claims = sorted(raw_claims, key=lambda c: c.confidence, reverse=True)
            raw_claims = raw_claims[:self.max_claims]

            if not raw_claims:
                logger.debug(f"[{tid}] No claims extracted.")
                return {"transcript_id": tid, "status": "no_claims",
                        "claims": 0, "verified": 0, "errors": 0}

            # 3. Normalize claims
            normalized = [self.normalizer.normalize(rc, ticker=ticker) for rc in raw_claims]

            # 4. Build date windows
            call_dt   = datetime.strptime(call_date, "%Y-%m-%d")
            sec_start = (call_dt - timedelta(days=SEC_WINDOW_DAYS_BEFORE)).strftime("%Y-%m-%d")
            sec_end   = (call_dt + timedelta(days=SEC_WINDOW_DAYS_AFTER)).strftime("%Y-%m-%d")
            news_end  = (call_dt + timedelta(days=NEWS_WINDOW_DAYS_AFTER)).strftime("%Y-%m-%d")

            # 5. Retrieve SEC filings (once per transcript, cache per CIK)
            sec_evidence = self.sec_provider.search_filings(
                ticker, start_date=sec_start, end_date=sec_end,
                filing_types=["10-K", "10-Q", "8-K"],
            )

            # 6. Retrieve news + build evidence map per claim
            evidence_map: Dict[str, List] = {}
            for c in normalized:
                claim_ev = list(sec_evidence)  # all SEC filings apply to all claims

                # News evidence keyed on claim type
                news_query = f"{c.metric} {c.direction or ''}".strip()
                news_ev = self.news_provider.search_news(
                    ticker=ticker,
                    company_name=None,
                    query=news_query,
                    start_date=call_date,
                    end_date=news_end,
                    limit=3,
                )
                claim_ev.extend(news_ev)
                evidence_map[c.claim_id] = claim_ev

            # 7. Verify claims
            verifications: List[Dict] = []
            for c in normalized:
                v = self.verifier.verify(c, evidence_map.get(c.claim_id, []), call_date)
                verifications.append(v)

            # 8. Compute market reaction
            try:
                market_result = self.market_provider.calculate_market_reaction(
                    ticker=ticker, call_date=call_date
                )
            except Exception as me:
                logger.debug(f"[{tid}] Market data error: {me}")
                market_result = None

            # 9. Persist everything
            self._save_claims_and_results(tid, normalized, verifications, evidence_map, market_result)

            elapsed = round(time.time() - t0, 1)
            status_counts = {}
            for v in verifications:
                s = v["status"]
                status_counts[s] = status_counts.get(s, 0) + 1

            logger.info(
                f"[{tid}] Done | claims={len(normalized)} | verified={len(verifications)} "
                f"| {status_counts} | market={'ok' if market_result else 'N/A'} "
                f"| time={elapsed}s | RAM={_ram_gb():.2f}GB"
            )
            return {
                "transcript_id": tid,
                "status": "ok",
                "claims": len(normalized),
                "verified": len(verifications),
                "status_counts": status_counts,
                "market_ok": market_result is not None,
                "errors": 0,
                "elapsed_s": elapsed,
            }

        except Exception as exc:
            logger.error(f"[{tid}] Error: {exc}", exc_info=True)
            return {"transcript_id": tid, "status": "error",
                    "claims": 0, "verified": 0, "errors": 1, "error_msg": str(exc)}


def run_phase3(
    limit: Optional[int] = None,
    force_reprocess: bool = False,
) -> Dict:
    """
    Run the Phase 3 pipeline.

    Parameters
    ----------
    limit            : process at most this many transcripts (None = all)
    force_reprocess  : if True, delete and re-run transcripts that already have claims.
                       Without this flag, already-processed transcripts are skipped
                       (standard checkpoint/resume behaviour).
    """
    engine = Phase3Engine()

    # ── KEY FIX: pass force_reprocess so the pending list respects --force ──
    pending = engine._get_pending_transcripts(limit=limit, force_reprocess=force_reprocess)

    if not pending:
        logger.info("Phase 3: No pending transcripts to process.")
        return {"transcripts_processed": 0, "transcripts_skipped": 0, "errors": 0,
                "claims_extracted": 0, "claims_verified": 0,
                "status_counts": {}, "market_coverage": 0, "peak_ram_gb": _ram_gb()}

    total = len(pending)
    logger.info(
        f"Phase 3 starting: {total} transcripts to process "
        f"(force={force_reprocess})."
    )

    # Reset checkpoint when forcing a full rerun
    if force_reprocess:
        engine.ckpt.reset_checkpoint(JOB_NAME)
    engine.ckpt.set_checkpoint(JOB_NAME, last_processed_id=None, status="IN_PROGRESS")

    agg = {
        "transcripts_processed": 0,
        "transcripts_skipped": 0,
        "errors": 0,
        "claims_extracted": 0,
        "claims_verified": 0,
        "status_counts": {},
        "market_coverage": 0,
        "peak_ram_gb": 0.0,
    }
    t0_total = time.time()

    for i, transcript in enumerate(pending, 1):
        # ── KEY FIX: pass force_reprocess to process_transcript ──
        result = engine.process_transcript(transcript, force_reprocess=force_reprocess)

        if result["status"] == "ok":
            agg["transcripts_processed"] += 1
            agg["claims_extracted"]       += result.get("claims", 0)
            agg["claims_verified"]        += result.get("verified", 0)
            if result.get("market_ok"):
                agg["market_coverage"] += 1
            for s, cnt in result.get("status_counts", {}).items():
                agg["status_counts"][s] = agg["status_counts"].get(s, 0) + cnt
        elif result["status"] in ("skipped_no_sentences", "no_claims"):
            agg["transcripts_skipped"] += 1
        else:
            agg["errors"] += 1

        ram = _ram_gb()
        agg["peak_ram_gb"] = max(agg["peak_ram_gb"], ram)

        # Periodic checkpoint so progress is preserved on interruption
        engine.ckpt.set_checkpoint(
            JOB_NAME,
            last_processed_id=transcript["transcript_id"],
            status="IN_PROGRESS",
        )

        if i % 10 == 0:
            elapsed = round(time.time() - t0_total, 0)
            logger.info(
                f"Progress: {i}/{total} | claims={agg['claims_extracted']} "
                f"| errors={agg['errors']} | elapsed={elapsed}s | RAM={ram:.2f}GB"
            )

        gc.collect()

    # Mark phase 3 checkpoint complete
    engine.ckpt.mark_completed(JOB_NAME)

    agg["total_elapsed_s"] = round(time.time() - t0_total, 1)
    logger.info(
        f"Phase 3 complete. Processed={agg['transcripts_processed']} "
        f"| Errors={agg['errors']} | Claims={agg['claims_extracted']} "
        f"| Verified={agg['claims_verified']} | Market={agg['market_coverage']}"
    )
    return agg


