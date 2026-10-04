#!/usr/bin/env python3
"""
Populate the outcomes table with later fundamental outcomes.

This script uses SEC company facts data (XBRL GAAP concepts) to extract
actual revenue and earnings figures reported after earnings calls.

Usage:
    python scripts/populate_outcomes.py [--limit N] [--reset]
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sec_cache"
DB_PATH = PROJECT_ROOT / "database" / "hidden_risk.db"


def get_company_facts(cik: str) -> Optional[Dict]:
    """Load company facts JSON for a CIK."""
    cache_file = DATA_DIR / f"CIK{cik}-companyfacts.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    cache_file = DATA_DIR / f"CIK{cik}-facts.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    return None


def find_latest_before(entries: List[Dict], as_of_date: str, lookback_days: int) -> Optional[float]:
    """Find the most recent value before as_of_date within lookback window."""
    cutoff = datetime.fromisoformat(as_of_date) - timedelta(days=lookback_days)
    valid = []
    
    for entry in entries:
        if entry.get("frame") or entry.get("accrual") or entry.get("preferred"):
            continue
        
        try:
            filed = entry.get("end") or entry.get("filed")
            if not filed:
                continue
            
            if len(filed) == 10:
                filing_dt = datetime.fromisoformat(filed)
            else:
                filing_dt = datetime.fromisoformat(filed[:10])
            
            if filing_dt <= datetime.fromisoformat(as_of_date) and filing_dt >= cutoff:
                valid.append((filing_dt, entry.get("val", 0)))
        
        except (ValueError, TypeError):
            continue
    
    if valid:
        valid.sort(key=lambda x: x[0], reverse=True)
        return valid[0][1] / 1_000_000
    
    return None


def extract_recent_metrics(facts: Dict, as_of_date: str, lookback_days: int = 365) -> Tuple[Optional[float], Optional[float]]:
    """Extract revenue and earnings from company facts."""
    try:
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        
        revenue_entries = us_gaap.get("Revenues", {}).get("units", {}).get("USD", [])
        revenue = find_latest_before(revenue_entries, as_of_date, lookback_days) if revenue_entries else None
        
        earnings_entries = us_gaap.get("NetIncomeLoss", {}).get("units", {}).get("USD", [])
        earnings = find_latest_before(earnings_entries, as_of_date, lookback_days) if earnings_entries else None
        
        return revenue, earnings
    
    except Exception as exc:
        print(f"[WARN] Failed to extract metrics: {exc}")
        return None, None


def load_company_tickers() -> Dict[str, str]:
    """Load ticker -> CIK mapping from SEC cache."""
    ticker_file = DATA_DIR / "company_tickers.json"
    if not ticker_file.exists():
        return {}
    
    try:
        with open(ticker_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        mapping = {}
        for item in data.values():
            if isinstance(item, dict):
                cik = str(item.get("cik_str", "")).zfill(10)
                ticker = item.get("ticker", "").upper()
                if ticker and cik:
                    mapping[ticker] = cik
        
        return mapping
    except Exception as exc:
        print(f"[WARN] Failed to load company tickers: {exc}")
        return {}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Populate outcomes table")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many transcripts")
    parser.add_argument("--reset", action="store_true", help="Clear existing outcomes first")
    
    args = parser.parse_args()
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        if args.reset:
            print("Clearing existing outcomes...")
            conn.execute("DELETE FROM outcomes")
            conn.commit()
        
        existing = set(
            r[0] for r in conn.execute(
                "SELECT transcript_id FROM outcomes"
            ).fetchall()
        )
        
        if existing:
            placeholders = ",".join("?" * len(existing))
            rows = conn.execute(
                f"SELECT transcript_id, ticker, date, company_id FROM transcripts "
                f"WHERE transcript_id NOT IN ({placeholders})",
                tuple(existing)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT transcript_id, ticker, date, company_id FROM transcripts"
            ).fetchall()
        
        print(f"Found {len(rows)} transcripts to process")
        
        if args.limit:
            rows = rows[:args.limit]
        
        ticker_to_cik = load_company_tickers()
        processed = 0
        errors = 0
        skipped = 0
        
        for row in rows:
            tid = row["transcript_id"]
            ticker = row["ticker"]
            call_date = row["date"]
            cik = row["company_id"] or ticker_to_cik.get(ticker)
            
            if not cik:
                print(f"[SKIP] [{tid}] No CIK found for ticker {ticker}")
                skipped += 1
                continue
            
            facts = get_company_facts(cik)
            if not facts:
                print(f"[DEBUG] [{tid}] No company facts for CIK {cik}")
                conn.execute(
                    "INSERT INTO outcomes (transcript_id, outcome_date) VALUES (?, ?)",
                    (tid, call_date)
                )
                processed += 1
                continue
            
            revenue, earnings = extract_recent_metrics(facts, call_date, lookback_days=365)
            guidance_change = None
            
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO outcomes 
                       (transcript_id, later_guidance, guidance_change, 
                        actual_revenue, earnings_surprise, outcome_date)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (tid, None, guidance_change, revenue, earnings, call_date)
                )
                processed += 1
                
                if processed % 50 == 0:
                    print(f"[INFO] Progress: {processed}/{len(rows)} transcripts processed")
            
            except Exception as exc:
                print(f"[ERROR] [{tid}] Error inserting outcome: {exc}")
                errors += 1
        
        conn.commit()
        print(f"\n[COMPLETE] Outcomes population: {processed} processed, {errors} errors, {skipped} skipped")
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()
