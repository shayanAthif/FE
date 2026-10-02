"""Abstract base classes for external evidence and market providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class RetrievedEvidence:
    source_type: str  # 'sec', 'news', 'market'
    source_name: str
    source_url: str
    publication_date: str
    title: str
    text: str
    filing_type: Optional[str] = None
    relevance_score: float = 0.0
    authority_score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MarketReaction:
    ticker: str
    call_date: str
    price_before: Optional[float]
    price_after: Optional[float]
    returns: Dict[str, Optional[float]]       # e.g., {'1d': -0.02, '5d': 0.05, ...}
    volatilities: Dict[str, Optional[float]]  # e.g., {'5d': 0.035, '10d': 0.042, ...}
    benchmark_returns: Dict[str, Optional[float]]
    abnormal_returns: Dict[str, Optional[float]]


class BaseSecProvider(ABC):
    """Abstract interface for SEC filing retrieval."""

    @abstractmethod
    def search_filings(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        filing_types: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> List[RetrievedEvidence]:
        """Retrieve relevant SEC filings for a company within a date range."""
        pass

    @abstractmethod
    def get_reported_facts(
        self,
        ticker: str,
        metric: str,
        period_end_date: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve XBRL/reported accounting fact for a metric and period."""
        pass


class BaseNewsProvider(ABC):
    """Abstract interface for financial news retrieval."""

    @abstractmethod
    def search_news(
        self,
        ticker: str,
        company_name: Optional[str],
        query: str,
        start_date: str,
        end_date: str,
        limit: int = 10,
    ) -> List[RetrievedEvidence]:
        """Search news articles regarding a company and claim topic in a time window."""
        pass


class BaseMarketProvider(ABC):
    """Abstract interface for stock market data and post-call returns."""

    @abstractmethod
    def get_price_history(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch historical daily prices DataFrame with OHLCV columns."""
        pass

    @abstractmethod
    def calculate_market_reaction(
        self,
        ticker: str,
        call_date: str,
        windows: Optional[List[int]] = None,
        benchmark: Optional[str] = "^GSPC",
    ) -> Optional[MarketReaction]:
        """Compute post-earnings trading returns and realized volatilities."""
        pass

