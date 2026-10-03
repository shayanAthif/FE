"""
Market Data Provider — Phase 3

Fetches historical daily prices using yfinance with persistent local caching.
Calculates post-earnings trading day returns (1D, 5D, 10D, 20D) and realized volatilities,
using actual trading day calendars.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

from src.config import PROJECT_ROOT, get_config
from src.logger import get_logger
from src.providers.base import BaseMarketProvider, MarketReaction

logger = get_logger("market_provider")

_CACHE_DIR = PROJECT_ROOT / "data" / "market"


class MarketProvider(BaseMarketProvider):
    """Market data provider with trading-day calendar alignment and local caching."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cfg = get_config()
        self.cache_dir = cache_dir or _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_symbol = self.cfg.market.benchmark_symbol or "^GSPC"
        self._price_cache = {}

    def get_price_history(
        self,
        ticker: str,
        start_date: str = "2005-01-01",
        end_date: str = "2026-01-01",
    ) -> pd.DataFrame:
        """Fetch historical daily price DataFrame (OHLCV) with disk caching."""
        clean_ticker = ticker.upper().strip()
        if clean_ticker in self._price_cache:
            return self._price_cache[clean_ticker]

        cache_path = self.cache_dir / f"{clean_ticker}.parquet"
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                self._price_cache[clean_ticker] = df
                return df
            except Exception:
                pass

        logger.debug(f"Downloading historical price series for {clean_ticker}...")
        try:
            # Add small buffer to dates
            s = (pd.to_datetime(start_date) - timedelta(days=60)).strftime("%Y-%m-%d")
            e = (pd.to_datetime(end_date) + timedelta(days=60)).strftime("%Y-%m-%d")
            df = yf.download(clean_ticker, start=s, end=e, progress=False)

            if df.empty:
                logger.warning(f"No price data returned for {clean_ticker}")
                return pd.DataFrame()

            # Flatten MultiIndex columns if present in newer yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index = pd.to_datetime(df.index)
            # Ensure timezone-naive
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            df.to_parquet(cache_path)
            self._price_cache[clean_ticker] = df
            return df
        except Exception as exc:
            logger.warning(f"Failed to download price history for {clean_ticker}: {exc}")
            return pd.DataFrame()

    def calculate_market_reaction(
        self,
        ticker: str,
        call_date: str,
        windows: Optional[List[int]] = None,
        benchmark: Optional[str] = None,
    ) -> Optional[MarketReaction]:
        """
        Compute trading-day post-call returns and realized volatilities.
        """
        windows = windows or [1, 5, 10, 20]
        bench_sym = benchmark or self.benchmark_symbol

        df = self.get_price_history(ticker)
        if df.empty or "Close" not in df.columns:
            return None

        call_dt = pd.to_datetime(call_date)
        if call_dt.tz is not None:
            call_dt = call_dt.tz_localize(None)

        # Trading days strictly before call
        prior_dates = df.index[df.index < call_dt]
        # Trading days on or after call
        post_dates = df.index[df.index >= call_dt]

        if len(prior_dates) == 0 or len(post_dates) == 0:
            return None

        # Price before call: close price of last trading day before call
        dt_before = prior_dates[-1]
        price_before = float(df.loc[dt_before, "Close"])

        # Price after call: close price of first trading day on/after call
        dt_call_day = post_dates[0]
        price_after = float(df.loc[dt_call_day, "Close"])

        # Locate integer index in post_dates
        call_idx = df.index.get_loc(dt_call_day)

        returns = {}
        volatilities = {}

        for w in windows:
            target_idx = call_idx + w
            if target_idx < len(df):
                p_end = float(df.iloc[target_idx]["Close"])
                ret = (p_end - price_before) / price_before
                returns[f"{w}d"] = round(float(ret), 4)

                # Realized daily volatility over the window
                window_slice = df.iloc[call_idx : target_idx + 1]["Close"]
                daily_pct = window_slice.pct_change().dropna()
                if len(daily_pct) > 1:
                    vol = float(daily_pct.std() * math.sqrt(252))  # Annualized volatility
                    volatilities[f"{w}d"] = round(vol, 4)
                else:
                    volatilities[f"{w}d"] = None
            else:
                returns[f"{w}d"] = None
                volatilities[f"{w}d"] = None

        # Benchmark returns (S&P 500)
        benchmark_returns = {}
        abnormal_returns = {}

        bench_df = self.get_price_history(bench_sym)
        if not bench_df.empty and "Close" in bench_df.columns:
            b_prior = bench_df.index[bench_df.index < call_dt]
            b_post = bench_df.index[bench_df.index >= call_dt]
            if len(b_prior) > 0 and len(b_post) > 0:
                b_dt_before = b_prior[-1]
                b_price_before = float(bench_df.loc[b_dt_before, "Close"])
                b_call_idx = bench_df.index.get_loc(b_post[0])

                for w in [5, 10]:
                    b_target_idx = b_call_idx + w
                    if b_target_idx < len(bench_df):
                        b_end = float(bench_df.iloc[b_target_idx]["Close"])
                        b_ret = (b_end - b_price_before) / b_price_before
                        benchmark_returns[f"{w}d"] = round(float(b_ret), 4)

                        stock_ret = returns.get(f"{w}d")
                        if stock_ret is not None:
                            abnormal_returns[f"{w}d"] = round(stock_ret - b_ret, 4)

        return MarketReaction(
            ticker=ticker,
            call_date=call_date,
            price_before=round(price_before, 2),
            price_after=round(price_after, 2),
            returns=returns,
            volatilities=volatilities,
            benchmark_returns=benchmark_returns,
            abnormal_returns=abnormal_returns,
        )
