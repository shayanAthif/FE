"""Pytest configuration and shared test fixtures."""

from pathlib import Path
from typing import Generator
import pytest

from src.config import AppConfig, load_config
from src.database import DatabaseManager


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Provide a temporary database file path managed by pytest's tmp_path."""
    db_file = tmp_path / "test_hidden_risk.db"
    return db_file


@pytest.fixture
def db_manager(temp_db_path: Path) -> DatabaseManager:
    """Initialize a fresh temporary database with the full schema."""
    mgr = DatabaseManager(db_path=temp_db_path)
    mgr.init_database()
    return mgr


@pytest.fixture
def sample_config() -> AppConfig:
    """Load default application configuration for testing."""
    return load_config()


@pytest.fixture
def sample_transcript_dict():
    """Return a representative single transcript dictionary matching the HF dataset format."""
    return {
        "symbol": "AAPL",
        "company_name": "Apple Inc.",
        "company_id": "0000320193",
        "year": 2024,
        "quarter": "Q2",
        "date": "2024-05-02",
        "content": "Operator: Welcome to Apple's Q2 2024 Earnings Conference Call...",
        "structured_content": [
            {
                "speaker": "Operator",
                "role": "operator",
                "section": "opening",
                "text": "Welcome to Apple's Second Quarter 2024 Earnings Conference Call."
            },
            {
                "speaker": "Tim Cook",
                "role": "executive",
                "section": "prepared_remarks",
                "text": "Good afternoon, everyone. Today Apple is reporting revenue of $90.8 billion for the March quarter. We expect revenue to grow low single digits next quarter, though we remain cautious given macroeconomic headwinds."
            },
            {
                "speaker": "Tami Zakaria",
                "role": "analyst",
                "section": "q_and_a",
                "text": "Could you provide more color on China demand and whether you expect gross margins to sustain around 46%?"
            },
            {
                "speaker": "Luca Maestri",
                "role": "executive",
                "section": "q_and_a",
                "text": "What I would say is that it is too early to tell regarding long-term trends in Greater China. We are continuing to monitor the situation, but we remain focused on delivering value."
            }
        ]
    }

