"""
SEC EDGAR Evidence Provider — Phase 3

Integrates with SEC EDGAR public APIs to retrieve official financial disclosures:
  1. Company CIK lookup: https://www.sec.gov/files/company_tickers.json
  2. Submissions API: https://data.sec.gov/submissions/CIK{cik}.json
  3. XBRL Company Facts: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json

Features:
  - Adheres to SEC Fair Access guidelines (custom User-Agent)
  - Disk-based caching of responses in data/sec_cache/ to minimize network calls
  - Robust error handling and offline fallback
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

from src.config import get_config, PROJECT_ROOT
from src.logger import get_logger
from src.providers.base import BaseSecProvider, RetrievedEvidence

logger = get_logger("sec_provider")

_CACHE_DIR = PROJECT_ROOT / "data" / "sec_cache"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Common GAAP metric concepts in SEC XBRL
_GAAP_CONCEPT_MAP = {
    "revenue": ["Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "revenue_growth": ["Revenues", "SalesRevenueNet"],
    "operating_income": ["OperatingIncomeLoss"],
    "operating_margin": ["OperatingIncomeLoss"],
    "gross_margin": ["GrossProfit"],
    "net_income": ["NetIncomeLoss"],
    "eps": ["EarningsPerShareBasic", "EarningsPerShareDiluted"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
}


class SecProvider(BaseSecProvider):
    """SEC EDGAR evidence provider with local file caching and XBRL parsing."""

    def __init__(self, cache_dir: Optional[Path] = None, timeout: float = 12.0):
        self.cfg = get_config()
        self.cache_dir = cache_dir or _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.headers = {
            "User-Agent": self.cfg.sec_user_agent or "FinancialRiskResearch admin@research.org",
            "Accept-Encoding": "gzip, deflate",
        }
        self._cik_map: Dict[str, str] = {}
        self._load_cik_map()

    def _load_cik_map(self) -> None:
        """Load ticker -> CIK mapping from cache or SEC API."""
        cache_file = self.cache_dir / "company_tickers.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    for item in raw.values():
                        self._cik_map[item["ticker"].upper()] = str(item["cik_str"]).zfill(10)
                logger.debug(f"Loaded {len(self._cik_map)} CIK mappings from cache.")
                return
            except Exception as e:
                logger.warning(f"Error reading CIK cache: {e}")

        # Fetch from SEC
        try:
            r = requests.get(_TICKERS_URL, headers=self.headers, timeout=self.timeout)
            if r.status_code == 200:
                raw = r.json()
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(raw, f)
                for item in raw.values():
                    self._cik_map[item["ticker"].upper()] = str(item["cik_str"]).zfill(10)
                logger.info(f"Fetched and cached {len(self._cik_map)} CIK mappings from SEC.")
        except Exception as exc:
            logger.warning(f"Failed to fetch SEC CIK mapping: {exc}")

    def get_cik(self, ticker: str) -> Optional[str]:
        """Look up zero-padded 10-digit CIK for a ticker."""
        return self._cik_map.get(ticker.upper())

    def search_filings(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        filing_types: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> List[RetrievedEvidence]:
        """
        Retrieve filings (10-K, 10-Q, 8-K) for a company between start_date and end_date.
        """
        cik = self.get_cik(ticker)
        if not cik:
            logger.debug(f"No CIK found for ticker {ticker}")
            return []

        allowed_types = set(filing_types or ["10-K", "10-Q", "8-K"])
        cache_file = self.cache_dir / f"submissions_{cik}.json"
        data = None

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass

        if not data:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            try:
                time.sleep(0.1)  # respect rate limit (10 req/s)
                r = requests.get(url, headers=self.headers, timeout=self.timeout)
                if r.status_code == 200:
                    data = r.json()
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(data, f)
            except Exception as e:
                logger.warning(f"Error fetching SEC submissions for {ticker} (CIK {cik}): {e}")
                return []

        if not data or "filings" not in data:
            return []

        filing_sets = []
        if "recent" in data["filings"]:
            filing_sets.append(data["filings"]["recent"])

        # Check for older archives in filings.files
        for file_meta in data["filings"].get("files", []):
            f_from = file_meta.get("filingFrom", "")
            f_to = file_meta.get("filingTo", "")
            if (start_date <= f_to) and (end_date >= f_from):
                archive_name = file_meta["name"]
                arch_cache = self.cache_dir / archive_name
                arch_data = None
                if arch_cache.exists():
                    try:
                        with open(arch_cache, "r", encoding="utf-8") as f:
                            arch_data = json.load(f)
                    except Exception:
                        pass
                if not arch_data:
                    arch_url = f"https://data.sec.gov/submissions/{archive_name}"
                    try:
                        time.sleep(0.1)
                        r = requests.get(arch_url, headers=self.headers, timeout=self.timeout)
                        if r.status_code == 200:
                            arch_data = r.json()
                            with open(arch_cache, "w", encoding="utf-8") as f:
                                json.dump(arch_data, f)
                    except Exception as e:
                        logger.warning(f"Error fetching historical archive {archive_name}: {e}")
                if arch_data:
                    filing_sets.append(arch_data)

        results: List[RetrievedEvidence] = []
        for fset in filing_sets:
            forms = fset.get("form", [])
            filing_dates = fset.get("filingDate", [])
            accessions = fset.get("accessionNumber", [])
            primary_docs = fset.get("primaryDocument", [])
            descriptions = fset.get("primaryDocDescription", [])

            for i in range(len(forms)):
                form = forms[i]
                f_date = filing_dates[i]
                acc_num = accessions[i] if i < len(accessions) else ""
                acc_cleaned = acc_num.replace("-", "")
                doc = primary_docs[i] if i < len(primary_docs) else ""
                desc = descriptions[i] if i < len(descriptions) else f"{form} Filing"

                if form not in allowed_types:
                    continue

                # Temporal filter: must be within the specified window
                if not (start_date <= f_date <= end_date):
                    continue

                sec_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_cleaned}/{doc}" if doc else f"https://www.sec.gov/edgar/browse/?CIK={cik}"

                evidence = RetrievedEvidence(
                    source_type="sec",
                    source_name=f"SEC EDGAR Form {form}",
                    source_url=sec_url,
                    publication_date=f_date,
                    title=f"{ticker} Form {form} Filed {f_date}",
                    text=f"Official SEC Form {form} ({desc}) filed by {ticker} on {f_date}.",
                    filing_type=form,
                    relevance_score=0.90,
                    authority_score=1.00,  # Tier 1 official authority
                    metadata={"accession_number": acc_num, "cik": cik},
                )
                results.append(evidence)

        logger.debug(f"Found {len(results)} SEC filings for {ticker} between {start_date} and {end_date}.")
        return results

    def get_reported_facts(
        self,
        ticker: str,
        metric: str,
        period_end_date: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Query SEC XBRL company facts for a reported quantitative metric.
        """
        cik = self.get_cik(ticker)
        if not cik:
            return None

        cache_file = self.cache_dir / f"companyfacts_{cik}.json"
        facts = None

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    facts = json.load(f)
            except Exception:
                pass

        if not facts:
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            try:
                time.sleep(0.1)
                r = requests.get(url, headers=self.headers, timeout=self.timeout)
                if r.status_code == 200:
                    facts = r.json()
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(facts, f)
            except Exception as e:
                logger.warning(f"Error fetching XBRL company facts for {ticker}: {e}")
                return None

        if not facts:
            return None

        gaap_concepts = facts.get("facts", {}).get("us-gaap", {})
        candidate_concepts = _GAAP_CONCEPT_MAP.get(metric, [metric])

        for concept_name in candidate_concepts:
            concept_data = gaap_concepts.get(concept_name)
            if not concept_data or "units" not in concept_data:
                continue

            for unit_key, unit_items in concept_data["units"].items():
                for item in reversed(unit_items):
                    # Check period match
                    end_val = item.get("end")
                    if end_val and abs((pd.to_datetime(end_val) - pd.to_datetime(period_end_date)).days) <= 45:
                        return {
                            "metric": metric,
                            "concept": concept_name,
                            "value": float(item.get("val", 0)),
                            "unit": unit_key,
                            "period_end": end_val,
                            "filing_date": item.get("filed"),
                            "form": item.get("form"),
                        }

        return None
