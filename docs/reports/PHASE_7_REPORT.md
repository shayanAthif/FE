# Phase 7 Report — Full Dataset

## Purpose

Phase 7 processes every available transcript incrementally after Phase 6 has
passed. It streams the source, persists each transcript and stage result, and
uses the existing database checkpoints to resume after interruption.

## Command

```powershell
python scripts/run_phase7.py --batch-size 4
```

For a bounded rehearsal:

```powershell
python scripts/run_phase7.py --limit 100
```

## Acceptance criteria

- Phase 6 has passed first.
- Exit code is `0`.
- `outputs/phase7_full_dataset_report.json` has `returncode: 0`.
- `peak_ram_gb` records observed process-tree RSS; no application RAM budget is
  enforced.
- The resume check succeeds without increasing counts for already-completed
  transcripts.
- API failures are investigated from `logs/pipeline.log`; they are not silently
  treated as successful verification.

The final dataset size, runtime, and any provider failures must be copied from
the generated JSON report into this document after the server run.
