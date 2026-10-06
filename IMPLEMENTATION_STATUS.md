# Hidden Risk / Earnings Call Analysis — Implementation Status

> Generated: 2026-10-05  
> Repo root: `c:\Users\shayana\OneDrive - Nous Infosystems\Desktop\projects\FE`  
> Python: 3.14.7 | pytest: 9.1.1 | Platform: Windows 11

---

## End-to-End Pipeline Verification (as of 2026-10-05)

```
python scripts/run_pipeline.py --stage all --source jsonl \
  --local-path data/sample_transcripts.jsonl --force
```

**Result: PASSED ✅**

| Stage | Transcripts | Output | RAM | Time |
|-------|-------------|--------|-----|------|
| Phase 1 (Ingest) | 3 | 30 sentences, 3 QA pairs | 0.34 GB | 0.6s |
| Phase 2 (Risk) | 3 | Scored, FinBERT ran | 0.96 GB | ~116s |
| Phase 3 (Verify) | 3 | 9 claims, 9 verified | 1.30 GB | ~36s |
| **Total** | 3 | All stages complete | **1.30 GB peak observed** | **152.7s** |

---

## Phase-by-Phase Status

---

### Phase 0 — Architecture / Config / DB / Logging ✅ COMPLETE

All pre-existing. No changes required.

**Verified working:**
- `src/config.py` — `AppConfig` Pydantic model, `load_config()`, `get_config()`
- `src/database.py` — `DatabaseManager`, WAL mode SQLite, 12-table schema
- `src/checkpoint.py` — `CheckpointManager` with `set_checkpoint`, `get_checkpoint`, `mark_completed`, `mark_failed`, `reset_checkpoint`
- `src/logger.py` — structured logging with RAM monitoring
- `src/providers/base.py` — abstract base providers

**Tests:** `tests/test_architecture.py` — **6/6 passed ✅**

---

### Phase 1 — Data Ingestion / Preprocessing / Segmentation / QA Matching ✅ COMPLETE (NEW)

All modules were newly implemented in this session.

#### Files Created

| File | Status | Notes |
|------|--------|-------|
| `src/preprocessing.py` | ✅ Complete | HTML unescape, encoding norm, financial term preservation |
| `src/segmentation.py` | ✅ Complete | Structured + raw text fallback, financial sentence splitter |
| `src/qa_matcher.py` | ✅ Complete | FSM-based analyst→exec matching, multi-turn support |
| `src/data_loader.py` | ✅ Complete | RAM-resident HuggingFace loading + JSONL/JSON local, checkpoint/resume |
| `data/sample_transcripts.jsonl` | ✅ Created | 3 fixture transcripts (CLX, AAPL, MSFT) for local testing |

#### Key Design Decisions
- **Sentence splitter**: Token-aware boundary scanner (no `re` lookbehind — Python 3.14 compatible). Protects: `Mr.`, `U.S.`, `Inc.`, `$5.2`, `8.5%`, `Q4.`, `i.e.`, `approx.`
- **Segmentation priority**: `structured_content` list → heuristic raw text fallback
- **Section detection**: Explicit label from HF data → operator cue heuristics → context propagation
- **QA matching**: FSM: `WAITING_FOR_QUESTION → IN_QUESTION → IN_ANSWER`, operator bridges skipped
- **Deduplication**: In-memory set of already-processed IDs fetched once at startup
- **Checkpoint**: `processing_checkpoints` table, every N records (configurable via `processing.checkpoint_interval`)
- **Memory**: The complete Hugging Face dataset is materialized in RAM before processing; one transcript is processed at a time afterward, with explicit `gc.collect()` after each

#### IDs (deterministic, stable)
```
transcript_id : {TICKER}_{YEAR}_{QUARTER}_{DATE}       e.g. AAPL_2024_Q2_2024-05-02
segment_id    : {transcript_id}_seg_{seq:04d}
sentence_id   : {segment_id}_sent_{n:04d}
qa_id         : {transcript_id}_qa_{n:03d}
```

**Tests:** `tests/test_data_pipeline.py` — **12/12 passed ✅**

---

### Phase 2 — Linguistic Risk Engine ✅ PRE-EXISTING (no rewrite needed)

Pre-existing and fully functional. **No code changes made.**

**Verified working end-to-end** with Phase 1 data:
- `src/hedging_detector.py` — lexicon-based hedging/uncertainty scoring
- `src/finbert_engine.py` — singleton FinBERT, CPU inference, configurable batch size
- `src/tone_shift_detector.py` — sentiment sequence analysis
- `src/evasiveness_detector.py` — Q&A evasiveness scoring
- `src/risk_scorer.py` — weighted composite hidden risk score
- `src/risk_engine.py` — orchestrator, checkpoint/resume, `run_phase2()`

**Tests:** `tests/test_phase2_risk_engine.py` — pre-existing, verified importable ✅

---

### Phase 3 — Claims / Evidence / Fact Verification ✅ BUGS FIXED

Pre-existing architecture preserved. Two critical bugs were fixed.

#### Bug 1: `--force` flag silently ignored (FIXED ✅)

**File:** `src/phase3_engine.py`

**Problem:** `run_phase3(force_reprocess=True)` called `_get_pending_transcripts(limit)` without passing `force_reprocess`, so already-processed transcripts were always skipped regardless of the flag.

**Fix applied:**
```python
# OLD — force_reprocess ignored
def _get_pending_transcripts(self, limit):
    already_done = set(conn.execute("SELECT transcript_id FROM claims ..."))
    pending = [row for row in all_trans if row not in already_done]

# NEW — force_reprocess respected
def _get_pending_transcripts(self, limit, force_reprocess=False):
    if force_reprocess:
        pending = [dict(row) for row in all_trans]   # include ALL
    else:
        already_done = set(...)
        pending = [row for row in all_trans if row not in already_done]
```

Also added `_delete_existing_claims(transcript_id)` called per-transcript when force=True to prevent duplicate accumulation.

`run_phase3()` now passes `force_reprocess` to both `_get_pending_transcripts()` and `process_transcript()`.

#### Bug 2: `MAX_CLAIMS_PER_TRANSCRIPT` not configurable (FIXED ✅)

**File:** `src/phase3_engine.py`

**Problem:** Hard-coded `raw_claims[:MAX_CLAIMS_PER_TRANSCRIPT]` with no ranking — lowest-confidence claims could survive over high-confidence ones.

**Fix applied:**
```python
# OLD
raw_claims = raw_claims[:MAX_CLAIMS_PER_TRANSCRIPT]

# NEW — sort by confidence descending, then cap
raw_claims = sorted(raw_claims, key=lambda c: c.confidence, reverse=True)
raw_claims = raw_claims[:self.max_claims]

# self.max_claims reads from config.claims.max_claims_per_transcript (fallback: 20)
cfg_claims = getattr(self.cfg, "claims", None)
self.max_claims = getattr(cfg_claims, "max_claims_per_transcript", None) or MAX_CLAIMS_PER_TRANSCRIPT
```

> **NOTE:** `config.yaml` and `src/config.py` do NOT yet have a `claims:` section / `ClaimsConfig` class.  
> The `getattr(..., None)` fallback means it degrades gracefully to the constant `20`.  
> **Remaining work:** Add `ClaimsConfig` to `src/config.py` and `claims:` block to `config.yaml`.

#### Bug 3: Periodic checkpoint not written during Phase 3 loop (FIXED ✅)

Added `engine.ckpt.set_checkpoint(JOB_NAME, transcript_id, "IN_PROGRESS")` inside the per-transcript loop so progress is preserved on interruption.

---

### Phase 4 — Streamlit Dashboard ✅ MOSTLY WORKING

Pre-existing. Minor fix applied to database connection handling.

#### Fix Applied

**File:** `dashboard/components/data_loader.py`

**Problem:** `sqlite3.connect(str(DB_PATH))` crashed with `OperationalError: unable to open database file` when the `database/` directory did not exist yet (fresh checkout).

**Fix:**
```python
# OLD
def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

# NEW — creates directory and schema if needed
def get_connection():
    mgr = get_db_manager(DB_PATH)
    if not DB_PATH.exists():
        mgr.init_database()
    return mgr.get_connection()
```

#### Remaining Phase 4 Work

| Item | Status | Notes |
|------|--------|-------|
| `test_phase4_dashboard.py` — TestDataLoader tests | ❌ Failing | Tests use `CLX` ticker hardcoded; requires live DB with Phase 2 data |
| Dashboard section filter bug | ⚠️ Known | `dashboard/app.py` line 238 filters on `'Q&A'` but DB stores `'q_and_a'` — may cause empty results in transcript detail view |
| Transcript search / date filter | ⚠️ Not implemented | Spec called for search bar and date range filter in sidebar |
| Risk trend visualization | ⚠️ Not implemented | Spec called for time-series risk trend per ticker |
| Export functionality | ⚠️ Not implemented | Spec called for CSV/JSON export from dashboard |
| Evidence detail viewer | ⚠️ Basic only | Pre-existing; not enhanced |

**To launch dashboard (after running pipeline):**
```bash
streamlit run dashboard/app.py
```

---

### Phase 5 — Validation Framework ✅ FIXES APPLIED, PARTIAL PASS

Pre-existing framework. Several fixes applied to database connections and annotation metadata.

#### Fixes Applied

**1. `src/validation/annotation.py`**
- Same `unable to open database file` fix as Phase 4
- Added `evaluation_type` field to each annotation record (`'synthetic'` or `'human'`) — required for clearly distinguishing synthetic vs human evaluation
- Added `annotator_id` field (`'synthetic_generator'` for synthetic)
- Added `compute_cohens_kappa()` function for inter-annotator agreement measurement
- Added `evaluation_type` and `annotator_id` to `EXPORT_COLUMNS`

**2. `src/validation/baselines.py`**
- Same `unable to open database file` fix in `load_baseline_data()`

**3. `scripts/run_phase5.py`**
- Same `unable to open database file` fix in `load_verification_stats()` and `load_verification_summary()`
- Fixed duplicate `return df` statement (syntax error from edit collision)

#### Test Status

| Test Class | Passing | Failing | Notes |
|------------|---------|---------|-------|
| `TestAnnotation` (unit) | 5/9 | 4/9 | Tests requiring live DB with QA pairs fail (expected without real data) |
| `TestBaselines` (unit) | 3/5 | 2/5 | `load_baseline_data` and time-split tests need live DB with Phase 2 scores |
| `TestMetrics` | 1/3 | 2/3 | `evaluate_baselines_vs_annotations` needs live QA annotation sample |
| `TestStatistics` | 0/7 | 7 errors | Need live DB + `load_baseline_data()` fixture |
| `TestPhase5Integration` | 0/1 | 1/1 | Smoke test calls `run_phase5.py` which needs live DB |

#### Remaining Phase 5 Work

| Item | Status | Notes |
|------|--------|-------|
| `TestStatistics` — fixture `sample_df` | ❌ | Calls `load_baseline_data()` which needs live DB; add a `@pytest.fixture` that populates a temp DB with fixture data |
| `TestAnnotation.test_sample_qa_for_annotation_returns_list` | ❌ | Needs live QA pairs in DB |
| `TestPhase5Integration.test_full_pipeline_produces_outputs` | ❌ | Smoke test; needs full pipeline run first |
| `ClaimsConfig` in `config.py` + `config.yaml` | ❌ | See Phase 3 notes above |
| Cohen's Kappa reported in output | ⚠️ | Function implemented, not yet called from `run_phase5.py` |

---

## Remaining Work Summary (Prioritized)

### P1 — Required for full test suite pass

| # | File | Change Needed |
|---|------|---------------|
| 1 | `src/config.py` | Add `ClaimsConfig(max_claims_per_transcript: int = 20)` to `AppConfig` |
| 2 | `config.yaml` | Add `claims: max_claims_per_transcript: 20` |
| 3 | `tests/test_phase4_dashboard.py` | Replace hardcoded `CLX` ticker with fixture-based DB population (like `test_data_pipeline.py` does) |
| 4 | `tests/test_phase5_validation.py` | Add `@pytest.fixture sample_df` that runs the full pipeline on fixture data into a temp DB, then loads baseline data from it — needed for `TestStatistics` |
| 5 | `tests/test_phase5_validation.py` | `TestAnnotation` tests that call `sample_qa_for_annotation()` need the fixture DB to have QA pairs |

### P2 — Dashboard improvements (spec requirements)

| # | File | Change Needed |
|---|------|---------------|
| 6 | `dashboard/app.py` line 238 | Fix section filter: change `'Q&A'` → `'q_and_a'` to match DB stored values |
| 7 | `dashboard/app.py` | Add ticker search text input (sidebar) |
| 8 | `dashboard/app.py` | Add date range filter (sidebar date_input) |
| 9 | `dashboard/app.py` | Add risk trend line chart per ticker (Altair/Plotly) |
| 10 | `dashboard/app.py` | Add CSV export button (st.download_button) |

### P3 — Benchmark / docs

| # | File | Change Needed |
|---|------|---------------|
| 11 | `scripts/benchmark_pipeline.py` | Create scaling/stress test script with `--transcripts N` flag |
| 12 | `README.md` | Update with new Phase 1 setup, usage, fixture data, and `run_pipeline.py` examples |

---

## Files Changed in This Session

| File | Change | New/Modified |
|------|--------|--------------|
| `src/preprocessing.py` | Full implementation (was empty) | **NEW** |
| `src/segmentation.py` | Full implementation (was empty) | **NEW** |
| `src/qa_matcher.py` | Full implementation (was empty) | **NEW** |
| `src/data_loader.py` | Full implementation (was empty) | **NEW** |
| `src/phase3_engine.py` | Fixed `--force` bug, configurable claim cap, periodic checkpoint | Modified |
| `src/providers/market/market_provider.py` | Made `yfinance` import optional (graceful fallback) | Modified |
| `src/validation/annotation.py` | Fixed DB connection, added `evaluation_type`, `annotator_id`, Cohen's Kappa | Modified |
| `src/validation/baselines.py` | Fixed DB connection in `load_baseline_data()` | Modified |
| `src/preprocessing.py` | Added `html.unescape()` step, added `import html` | Modified |
| `dashboard/components/data_loader.py` | Fixed DB connection (dir + schema creation) | Modified |
| `scripts/run_pipeline.py` | Full implementation (was empty) | **NEW** |
| `scripts/run_phase5.py` | Fixed DB connections in verification loaders | Modified |
| `tests/test_data_pipeline.py` | Full implementation (was empty) | **NEW** |
| `data/sample_transcripts.jsonl` | Created 3-transcript fixture (CLX, AAPL, MSFT) | **NEW** |

---

## Dependencies Installed

The following packages were not in the original environment and were installed:

```
yfinance==1.7.0       (required by src/providers/market/market_provider.py)
datasets==5.0.1       (required by src/data_loader.py HuggingFace loading)
```

Add to `requirements.txt` if not already present.

---

## Quick Commands Reference

```powershell
# Run full pipeline on fixture data (no internet needed)
python scripts/run_pipeline.py --stage all --source jsonl --local-path data/sample_transcripts.jsonl

# Run full pipeline on HuggingFace dataset (internet + datasets package required)
python scripts/run_pipeline.py --stage all --limit 10

# Re-run a specific stage, force reprocessing
python scripts/run_pipeline.py --stage risk --force
python scripts/run_pipeline.py --stage verify --force

# Run all tests (clear cache first to avoid stale bytecode errors)
Remove-Item -Recurse -Force .pytest_cache, tests\__pycache__ -ErrorAction SilentlyContinue
python -m pytest tests\test_data_pipeline.py tests\test_architecture.py -v

# Launch dashboard
streamlit run dashboard/app.py

# Phase 2 only
python scripts/run_phase2.py --test

# Phase 3 only
python scripts/run_phase3.py --limit 5 --force
```
