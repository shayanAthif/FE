"""Tests for Phase 0: Architecture, Configuration, Database Schema, Logging, and Checkpoints."""

import pytest
import sqlite3
from src.config import load_config, get_config, AppConfig
from src.database import DatabaseManager
from src.checkpoint import CheckpointManager
from src.logger import setup_logger, get_current_ram_gb, check_ram_headroom
from src.providers.base import BaseMarketProvider, BaseNewsProvider, BaseSecProvider


def test_config_loading(sample_config: AppConfig):
    """Verify master configuration loads and conforms to expected bounds and types."""
    assert sample_config.dataset.name == "Bose345/sp500_earnings_transcripts"
    assert sample_config.dataset.streaming is False

    # Risk weights validation
    rw = sample_config.risk
    total_weight = rw.hedging_weight + rw.evasiveness_weight + rw.tone_shift_weight
    assert abs(total_weight - 1.0) < 1e-4, f"Risk weights should sum to 1.0, got {total_weight}"

    # Aggregation weights
    agg = sample_config.aggregation
    total_agg = agg.average_weight + agg.top_risk_weight
    assert abs(total_agg - 1.0) < 1e-4

    # Memory limits
    assert sample_config.memory.max_ram_gb == 12.0
    assert sample_config.memory.warning_ram_gb <= sample_config.memory.max_ram_gb

    # Lexicon phrases present
    assert len(rw.hedging_lexicon.modals) > 0
    assert "too early to tell" in rw.hedging_lexicon.hedging_phrases


def test_database_init(db_manager: DatabaseManager):
    """Verify all 12 core tables exist and can be queried."""
    expected_tables = {
        "transcripts",
        "segments",
        "sentences",
        "qa_pairs",
        "risk_scores",
        "transcript_scores",
        "claims",
        "evidence",
        "verification",
        "market_events",
        "outcomes",
        "processing_checkpoints",
    }

    with db_manager.connection() as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

    assert expected_tables.issubset(tables), f"Missing tables: {expected_tables - tables}"


def test_foreign_key_enforcement(db_manager: DatabaseManager):
    """Verify that foreign keys are enforced by SQLite."""
    with pytest.raises(sqlite3.IntegrityError):
        with db_manager.transaction() as conn:
            # Attempt to insert a segment referencing a nonexistent transcript
            conn.execute(
                """
                INSERT INTO segments (segment_id, transcript_id, speaker, speaker_role, section, text, sequence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("seg_1", "NONEXISTENT_TRANSCRIPT", "Speaker", "executive", "q_and_a", "Hello", 1),
            )


def test_checkpoint_lifecycle(db_manager: DatabaseManager):
    """Verify checkpoint tracking, state updates, resume queries, and completions."""
    cp_mgr = CheckpointManager(db_manager=db_manager)

    # Initial state should be None
    assert cp_mgr.get_checkpoint("ingest") is None
    assert cp_mgr.is_job_completed("ingest") is False

    # Set in progress
    cp_mgr.set_checkpoint("ingest", last_processed_id="AAPL_2024_Q2", status="IN_PROGRESS")
    state = cp_mgr.get_checkpoint("ingest")
    assert state is not None
    assert state.last_processed_id == "AAPL_2024_Q2"
    assert state.status == "IN_PROGRESS"
    assert cp_mgr.is_job_completed("ingest") is False

    # Mark completed
    cp_mgr.mark_completed("ingest", last_processed_id="AAPL_2024_Q2")
    assert cp_mgr.is_job_completed("ingest") is True

    # Mark another job failed
    cp_mgr.mark_failed("risk", error_message="Simulated OOM", last_processed_id="MSFT_2024_Q1")
    state_failed = cp_mgr.get_checkpoint("risk")
    assert state_failed is not None
    assert state_failed.status == "FAILED"
    assert "Simulated OOM" in state_failed.error_message

    # Reset
    cp_mgr.reset_checkpoint("risk")
    assert cp_mgr.get_checkpoint("risk") is None


def test_logger_and_memory(tmp_path):
    """Verify structured logger writes to file and memory monitoring measures RAM."""
    logger = setup_logger(name="test_logger", log_dir=tmp_path, log_file="test.log")
    logger.info("Testing architecture logger output.")

    log_file = tmp_path / "test.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Testing architecture logger output." in content

    # Test RAM tracker
    ram_gb = get_current_ram_gb()
    assert ram_gb > 0.0, "RAM usage should be greater than 0"
    headroom = check_ram_headroom(logger=logger)
    assert headroom == ram_gb


def test_abstract_providers_cannot_be_instantiated():
    """Verify that provider base classes enforce abstract methods."""
    with pytest.raises(TypeError):
        BaseSecProvider()  # type: ignore

    with pytest.raises(TypeError):
        BaseNewsProvider()  # type: ignore

    with pytest.raises(TypeError):
        BaseMarketProvider()  # type: ignore

