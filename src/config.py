"""Configuration management module."""

import os
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load optional .env file
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DatasetConfig(BaseModel):
    name: str = "Bose345/sp500_earnings_transcripts"
    split: str = "train"
    streaming: bool = False


class DatabaseConfig(BaseModel):
    path: str = "database/hidden_risk.db"
    timeout_seconds: float = 30.0
    wal_mode: bool = True

    def get_absolute_path(self, base_dir: Optional[Path] = None) -> Path:
        base = base_dir or PROJECT_ROOT
        db_path = Path(self.path)
        return db_path if db_path.is_absolute() else base / db_path


class ProcessingConfig(BaseModel):
    batch_size: int = 16
    max_length: int = 256
    checkpoint_interval: int = 100
    device: str = "cpu"


class HedgingLexiconConfig(BaseModel):
    modals: List[str] = Field(default_factory=lambda: ["may", "might", "could", "would"])
    uncertainty: List[str] = Field(default_factory=lambda: [
        "potentially", "approximately", "roughly", "uncertain", "cautious",
        "cautiously optimistic", "depending on", "subject to"
    ])
    hedging_phrases: List[str] = Field(default_factory=lambda: [
        "we believe", "we expect", "we anticipate", "it is difficult to say",
        "too early to tell", "we are evaluating", "we continue to monitor",
        "cannot comment", "we remain cautious"
    ])
    deflection_phrases: List[str] = Field(default_factory=lambda: [
        "i can't comment", "cannot comment", "it is too early", "we are monitoring",
        "we continue to evaluate", "we will provide information later",
        "we are focused on the long term", "as you know", "the important thing is",
        "i think the key point is", "what i would say is"
    ])


class RiskConfig(BaseModel):
    hedging_weight: float = 0.35
    evasiveness_weight: float = 0.40
    tone_shift_weight: float = 0.25
    local_tone_window: int = 5
    hedging_lexicon: HedgingLexiconConfig = Field(default_factory=HedgingLexiconConfig)


class AggregationConfig(BaseModel):
    average_weight: float = 0.60
    top_risk_weight: float = 0.40
    top_k_percentile: float = 0.10


class MarketConfig(BaseModel):
    provider: str = "yfinance"
    windows: List[int] = Field(default_factory=lambda: [1, 5, 10, 20])
    benchmark_symbol: str = "^GSPC"


class EvidenceConfig(BaseModel):
    sec_rate_limit_per_sec: int = 10
    minimum_evidence_score: float = 0.50
    authority_weights: Dict[str, float] = Field(default_factory=lambda: {
        "tier_1": 1.0,
        "tier_2": 0.75,
        "tier_3": 0.50,
        "tier_4": 0.25,
    })


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "logs"
    log_file: str = "pipeline.log"


class MemoryConfig(BaseModel):
    max_ram_gb: float = 12.0
    warning_ram_gb: float = 10.0


class ClaimsConfig(BaseModel):
    max_claims_per_transcript: int = 20


class AppConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    claims: ClaimsConfig = Field(default_factory=ClaimsConfig)

    # SEC / API credentials read from environment
    sec_user_agent: str = Field(default_factory=lambda: os.getenv("SEC_USER_AGENT", "FinancialRiskResearch contact@research.org"))
    market_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("MARKET_API_KEY"))
    news_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("NEWS_API_KEY"))


_cached_config: Optional[AppConfig] = None


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Load configuration from a YAML file, overlaid with environment variables."""
    global _cached_config
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
    else:
        raw_data = {}

    config = AppConfig(**raw_data)
    _cached_config = config
    return config


def get_config() -> AppConfig:
    """Return the cached configuration or load default if not cached."""
    global _cached_config
    if _cached_config is None:
        return load_config()
    return _cached_config
