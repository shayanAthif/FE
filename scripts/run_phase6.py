"""Run the Phase 6 scale test with resource and resume reporting.

Examples:
    python scripts/run_phase6.py --limit 2000 --batch-size 4
    python scripts/run_phase6.py --limit 2000 --source huggingface
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scale_runner import run_pipeline_monitored


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 — scale test")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--source", choices=["huggingface", "jsonl", "json"], default="huggingface")
    parser.add_argument("--local-path")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-resume-check", action="store_true")
    args = parser.parse_args()

    report = run_pipeline_monitored(
        limit=args.limit,
        source=args.source,
        local_path=args.local_path,
        batch_size=args.batch_size,
        report_name="phase6_scale_report.json",
        resume_check=not args.no_resume_check,
    )
    print(f"Phase 6 return code: {report['returncode']}")
    print(f"Elapsed: {report['elapsed_seconds']}s")
    print(f"Peak RAM: {report['peak_ram_gb']:.3f} GB / {report['ram_budget_gb']:.1f} GB")
    print("Report: outputs/phase6_scale_report.json")
    return int(report["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
