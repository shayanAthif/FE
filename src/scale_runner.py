"""Resource-aware orchestration helpers for the scale and full-dataset phases."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import psutil

from src.config import PROJECT_ROOT, get_config


def _process_tree_rss_gb(process: psutil.Process) -> float:
    """Return RSS for a process and its children, in GiB."""
    total = 0
    try:
        processes = [process, *process.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        processes = [process]
    for child in processes:
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / (1024 ** 3)


def _database_size_bytes() -> int:
    db_path = get_config().database.get_absolute_path()
    total = 0
    for path in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
        if path.exists():
            total += path.stat().st_size
    return total


def _database_counts() -> dict[str, int]:
    db_path = get_config().database.get_absolute_path()
    if not db_path.exists():
        return {}
    tables = (
        "transcripts",
        "sentences",
        "risk_scores",
        "claims",
        "evidence",
        "verification",
        "market_events",
    )
    with sqlite3.connect(str(db_path)) as conn:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }


def run_pipeline_monitored(
    *,
    limit: Optional[int],
    source: str,
    local_path: Optional[str],
    batch_size: Optional[int],
    report_name: str,
    resume_check: bool = True,
) -> dict[str, Any]:
    """Run the existing end-to-end pipeline and write a resource report."""
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
        "--stage",
        "all",
        "--source",
        source,
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if local_path:
        command.extend(["--local-path", local_path])
    if batch_size is not None:
        command.extend(["--batch-size", str(batch_size)])

    before_counts = _database_counts()
    before_size = _database_size_bytes()
    started = time.perf_counter()
    child = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    process = psutil.Process(child.pid)
    peak_ram = 0.0
    output: list[str] = []
    while child.poll() is None:
        peak_ram = max(peak_ram, _process_tree_rss_gb(process))
        time.sleep(1.0)
    remaining = child.stdout.read() if child.stdout else ""
    if remaining:
        output.append(remaining)
    peak_ram = max(peak_ram, _process_tree_rss_gb(process))
    elapsed = time.perf_counter() - started

    after_counts = _database_counts()
    after_size = _database_size_bytes()
    report: dict[str, Any] = {
        "command": command,
        "returncode": child.returncode,
        "elapsed_seconds": round(elapsed, 2),
        "peak_ram_gb": round(peak_ram, 3),
        "database_size_before_bytes": before_size,
        "database_size_after_bytes": after_size,
        "database_size_delta_bytes": after_size - before_size,
        "database_counts_before": before_counts,
        "database_counts_after": after_counts,
        "stdout": "".join(output)[-12000:],
    }

    if resume_check and child.returncode == 0:
        check_started = time.perf_counter()
        check = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        report["resume_check"] = {
            "returncode": check.returncode,
            "elapsed_seconds": round(time.perf_counter() - check_started, 2),
            "database_counts_after": _database_counts(),
            "stdout": (check.stdout + check.stderr)[-4000:],
        }
    else:
        report["resume_check"] = None

    output_path = PROJECT_ROOT / "outputs" / report_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
