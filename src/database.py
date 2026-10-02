"""Database connection, schema management, and persistence module."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional
from src.config import PROJECT_ROOT, get_config
from src.logger import get_logger

logger = get_logger("database")

SCHEMA_SQL = """
-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Transcripts metadata and status
CREATE TABLE IF NOT EXISTS transcripts (
    transcript_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    company_name TEXT,
    company_id TEXT,
    date TEXT NOT NULL,
    year INTEGER NOT NULL,
    quarter TEXT NOT NULL,
    sector TEXT,
    raw_text TEXT,
    processing_status TEXT DEFAULT 'PENDING',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_transcripts_ticker_date ON transcripts(ticker, date);
CREATE INDEX IF NOT EXISTS idx_transcripts_year_quarter ON transcripts(year, quarter);

-- Transcript segments (Speaker turns)
CREATE TABLE IF NOT EXISTS segments (
    segment_id TEXT PRIMARY KEY,
    transcript_id TEXT NOT NULL,
    speaker TEXT NOT NULL,
    speaker_role TEXT NOT NULL,
    section TEXT NOT NULL,
    text TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    FOREIGN KEY (transcript_id) REFERENCES transcripts(transcript_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_segments_transcript ON segments(transcript_id);
CREATE INDEX IF NOT EXISTS idx_segments_section_role ON segments(section, speaker_role);

-- Sentence-level segmentation
CREATE TABLE IF NOT EXISTS sentences (
    sentence_id TEXT PRIMARY KEY,
    segment_id TEXT NOT NULL,
    transcript_id TEXT NOT NULL,
    sentence_number INTEGER NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (segment_id) REFERENCES segments(segment_id) ON DELETE CASCADE,
    FOREIGN KEY (transcript_id) REFERENCES transcripts(transcript_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sentences_segment ON sentences(segment_id);
CREATE INDEX IF NOT EXISTS idx_sentences_transcript ON sentences(transcript_id);

-- Q&A Pairs (Analyst Question -> Executive Answer)
CREATE TABLE IF NOT EXISTS qa_pairs (
    qa_id TEXT PRIMARY KEY,
    transcript_id TEXT NOT NULL,
    question_segment_id TEXT,
    answer_segment_id TEXT,
    analyst_speaker TEXT,
    executive_speaker TEXT,
    question_text TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    FOREIGN KEY (transcript_id) REFERENCES transcripts(transcript_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_qa_transcript ON qa_pairs(transcript_id);

-- Linguistic Risk Scores per sentence/segment
CREATE TABLE IF NOT EXISTS risk_scores (
    score_id TEXT PRIMARY KEY,
    sentence_id TEXT,
    segment_id TEXT NOT NULL,
    transcript_id TEXT NOT NULL,
    hedging_score REAL DEFAULT 0.0,
    hedge_density REAL DEFAULT 0.0,
    uncertainty_score REAL DEFAULT 0.0,
    positive_prob REAL DEFAULT 0.0,
    negative_prob REAL DEFAULT 0.0,
    neutral_prob REAL DEFAULT 0.0,
    sentiment_label TEXT,
    tone_shift_score REAL DEFAULT 0.0,
    evasiveness_score REAL DEFAULT 0.0,
    hidden_risk_score REAL DEFAULT 0.0,
    contributing_factors TEXT,
    FOREIGN KEY (sentence_id) REFERENCES sentences(sentence_id) ON DELETE CASCADE,
    FOREIGN KEY (segment_id) REFERENCES segments(segment_id) ON DELETE CASCADE,
    FOREIGN KEY (transcript_id) REFERENCES transcripts(transcript_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_risk_sentence ON risk_scores(sentence_id);
CREATE INDEX IF NOT EXISTS idx_risk_segment ON risk_scores(segment_id);
CREATE INDEX IF NOT EXISTS idx_risk_transcript ON risk_scores(transcript_id);

-- Transcript-level aggregated risk scores
CREATE TABLE IF NOT EXISTS transcript_scores (
    transcript_id TEXT PRIMARY KEY,
    overall_hidden_risk REAL NOT NULL,
    average_risk REAL NOT NULL,
    top_risk_average REAL NOT NULL,
    qa_risk REAL DEFAULT 0.0,
    prepared_risk REAL DEFAULT 0.0,
    highest_risk_segment TEXT,
    hedging_avg REAL DEFAULT 0.0,
    evasiveness_avg REAL DEFAULT 0.0,
    tone_shift_avg REAL DEFAULT 0.0,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transcript_id) REFERENCES transcripts(transcript_id) ON DELETE CASCADE
);

-- Extracted claims
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    transcript_id TEXT NOT NULL,
    segment_id TEXT,
    sentence_id TEXT,
    speaker TEXT,
    claim_text TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    entity TEXT,
    metric TEXT,
    value REAL,
    lower_bound REAL,
    upper_bound REAL,
    unit TEXT,
    period TEXT,
    geography TEXT,
    direction TEXT,
    confidence REAL DEFAULT 1.0,
    FOREIGN KEY (transcript_id) REFERENCES transcripts(transcript_id) ON DELETE CASCADE,
    FOREIGN KEY (sentence_id) REFERENCES sentences(sentence_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_transcript ON claims(transcript_id);
CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type);

-- Evidence items (SEC, News, Market)
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    source_type TEXT NOT NULL, -- 'sec', 'news', 'market'
    source_name TEXT,
    source_url TEXT,
    filing_type TEXT,
    publication_date TEXT,
    title TEXT,
    text TEXT,
    relevance_score REAL DEFAULT 0.0,
    authority_score REAL DEFAULT 0.0,
    retrieval_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_evidence_claim ON evidence(claim_id);
CREATE INDEX IF NOT EXISTS idx_evidence_type ON evidence(source_type);

-- Fact verification results
CREATE TABLE IF NOT EXISTS verification (
    verification_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL, -- 'SUPPORTED', 'CONTRADICTED', 'PARTIALLY_SUPPORTED', 'NOT_VERIFIABLE'
    confidence REAL NOT NULL,
    reasoning TEXT NOT NULL,
    evidence_ids TEXT, -- JSON array of evidence IDs
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_verification_status ON verification(status);

-- Stock Market Event outcomes
CREATE TABLE IF NOT EXISTS market_events (
    transcript_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    call_date TEXT NOT NULL,
    price_before REAL,
    price_after REAL,
    return_1d REAL,
    return_5d REAL,
    return_10d REAL,
    return_20d REAL,
    volatility_5d REAL,
    volatility_10d REAL,
    benchmark_return_5d REAL,
    abnormal_return_5d REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transcript_id) REFERENCES transcripts(transcript_id) ON DELETE CASCADE
);

-- Subsequent Fundamental Outcomes
CREATE TABLE IF NOT EXISTS outcomes (
    transcript_id TEXT PRIMARY KEY,
    later_guidance TEXT,
    guidance_change TEXT,
    actual_revenue REAL,
    earnings_surprise REAL,
    outcome_date TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transcript_id) REFERENCES transcripts(transcript_id) ON DELETE CASCADE
);

-- Checkpoints for resumable stages
CREATE TABLE IF NOT EXISTS processing_checkpoints (
    job_name TEXT PRIMARY KEY,
    last_processed_id TEXT,
    last_processed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL, -- 'IN_PROGRESS', 'COMPLETED', 'FAILED'
    error_message TEXT
);
"""


class DatabaseManager:
    """Manages SQLite database connections, initialization, and transactions."""

    def __init__(self, db_path: Optional[str | Path] = None, timeout: float = 30.0):
        config = get_config()
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = config.database.get_absolute_path()
        self.timeout = timeout or config.database.timeout_seconds
        self.wal_mode = config.database.wal_mode

    def get_connection(self) -> sqlite3.Connection:
        """Create and configure a new SQLite connection."""
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.timeout,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        if self.wal_mode and str(self.db_path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 30000;")
        return conn

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for automatic connection closing."""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for transactional operations with automatic commit/rollback."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_database(self) -> None:
        """Execute the initial schema creation script."""
        logger.info(f"Initializing database at {self.db_path}")
        conn = self.get_connection()
        try:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()
        logger.info("Database schema initialized successfully.")


_default_db_manager: Optional[DatabaseManager] = None


def get_db_manager(db_path: Optional[str | Path] = None) -> DatabaseManager:
    """Return singleton or configured DatabaseManager instance."""
    global _default_db_manager
    if db_path is not None:
        return DatabaseManager(db_path=db_path)
    if _default_db_manager is None:
        _default_db_manager = DatabaseManager()
    return _default_db_manager

