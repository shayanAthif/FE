"""Checkpointing and resume tracking system."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Set
from src.database import DatabaseManager, get_db_manager
from src.logger import get_logger

logger = get_logger("checkpoint")


@dataclass
class CheckpointState:
    job_name: str
    last_processed_id: Optional[str]
    last_processed_timestamp: Optional[str]
    status: str  # 'IN_PROGRESS', 'COMPLETED', 'FAILED'
    error_message: Optional[str]


class CheckpointManager:
    """Manages stage-level and item-level checkpoints in SQLite."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or get_db_manager()

    def set_checkpoint(
        self,
        job_name: str,
        last_processed_id: Optional[str],
        status: str = "IN_PROGRESS",
        error_message: Optional[str] = None,
    ) -> None:
        """Upsert a job checkpoint record."""
        now_str = datetime.now().isoformat()
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO processing_checkpoints (job_name, last_processed_id, last_processed_timestamp, status, error_message)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_name) DO UPDATE SET
                    last_processed_id = excluded.last_processed_id,
                    last_processed_timestamp = excluded.last_processed_timestamp,
                    status = excluded.status,
                    error_message = excluded.error_message
                """,
                (job_name, last_processed_id, now_str, status, error_message),
            )
        logger.debug(f"[Checkpoint] Job '{job_name}' -> status: {status}, last_id: {last_processed_id}")

    def get_checkpoint(self, job_name: str) -> Optional[CheckpointState]:
        """Fetch current checkpoint for a given job."""
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT job_name, last_processed_id, last_processed_timestamp, status, error_message FROM processing_checkpoints WHERE job_name = ?",
                (job_name,),
            ).fetchone()

        if row:
            return CheckpointState(
                job_name=row["job_name"],
                last_processed_id=row["last_processed_id"],
                last_processed_timestamp=str(row["last_processed_timestamp"]),
                status=row["status"],
                error_message=row["error_message"],
            )
        return None

    def is_job_completed(self, job_name: str) -> bool:
        """Check if a specific pipeline stage is already fully completed."""
        state = self.get_checkpoint(job_name)
        return state is not None and state.status == "COMPLETED"

    def mark_completed(self, job_name: str, last_processed_id: Optional[str] = None) -> None:
        """Mark a job as COMPLETED."""
        self.set_checkpoint(job_name, last_processed_id=last_processed_id, status="COMPLETED")
        logger.info(f"[Checkpoint] Stage '{job_name}' marked as COMPLETED.")

    def mark_failed(self, job_name: str, error_message: str, last_processed_id: Optional[str] = None) -> None:
        """Mark a job as FAILED with error context."""
        self.set_checkpoint(job_name, last_processed_id=last_processed_id, status="FAILED", error_message=error_message)
        logger.error(f"[Checkpoint] Stage '{job_name}' failed at ID '{last_processed_id}': {error_message}")

    def get_processed_transcript_ids(self, table_name: str = "transcripts") -> Set[str]:
        """Get set of all transcript IDs that have already been recorded in a given table."""
        with self.db.connection() as conn:
            cursor = conn.execute(f"SELECT transcript_id FROM {table_name}")
            return {row[0] for row in cursor.fetchall()}

    def reset_checkpoint(self, job_name: str) -> None:
        """Clear a checkpoint so stage can re-run from scratch."""
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM processing_checkpoints WHERE job_name = ?", (job_name,))
        logger.info(f"[Checkpoint] Reset checkpoint for '{job_name}'.")

