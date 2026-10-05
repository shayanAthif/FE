"""Benchmark the pipeline at scale and report time / memory profile."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ram_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


def run_benchmark(transcripts: int, source: str = "jsonl", local_path: str | None = None, stage: str = "all") -> dict:
    """Execute the pipeline for N transcripts and capture timing and memory stats."""
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
        "--stage",
        stage,
        "--limit",
        str(transcripts),
        "--force",
    ]
    if source:
        command.extend(["--source", source])
    if local_path:
        command.extend(["--local-path", local_path])

    start = time.perf_counter()
    start_ram = _ram_gb()
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    end_ram = _ram_gb()

    summary = {
        "transcripts": transcripts,
        "stage": stage,
        "source": source,
        "elapsed_seconds": round(elapsed, 3),
        "start_ram_gb": round(start_ram, 3),
        "end_ram_gb": round(end_ram, 3),
        "peak_ram_gb": round(max(start_ram, end_ram), 3),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Hidden Risk pipeline.")
    parser.add_argument("--transcripts", type=int, default=10, help="Number of transcripts to include in the benchmark.")
    parser.add_argument("--stage", type=str, default="all", choices=["ingest", "risk", "verify", "all"], help="Pipeline stage to benchmark.")
    parser.add_argument("--source", type=str, default="jsonl", choices=["huggingface", "jsonl", "json"], help="Data source.")
    parser.add_argument("--local-path", type=str, default=str(PROJECT_ROOT / "data" / "sample_transcripts.jsonl"), help="Local path for json/jsonl sources.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_benchmark(args.transcripts, args.source, args.local_path, args.stage)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("Benchmark Summary")
        print("=" * 60)
        for key in ["transcripts", "stage", "source", "elapsed_seconds", "start_ram_gb", "end_ram_gb", "peak_ram_gb", "returncode"]:
            print(f"{key}: {summary[key]}")
        if summary["stdout"]:
            print("\nSTDOUT:\n" + summary["stdout"][:4000])
        if summary["stderr"]:
            print("\nSTDERR:\n" + summary["stderr"][:4000])
    return 0 if summary["returncode"] == 0 else summary["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
