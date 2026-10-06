# Phase 6 Report — Scale Test

## Purpose

Phase 6 runs the complete pipeline on a bounded target of 2,000 transcripts and
records RAM, runtime, database growth, and resume behavior.

## Command

```powershell
python scripts/run_phase6.py --limit 2000 --batch-size 4
```

Use `--source jsonl --local-path <file>` for a local fixture. The default source
is the streamed Hugging Face dataset.

## Acceptance criteria

- Exit code is `0`.
- `outputs/phase6_scale_report.json` has `returncode: 0`.
- `peak_ram_gb` records observed process-tree RSS for capacity planning; no
  application RAM budget is enforced.
- The `resume_check.returncode` is `0`.
- The second run does not duplicate database rows.

The JSON report is the authoritative measurement record. Do not claim Phase 6
passed until the server run has been completed and reviewed.
