"""Database queries for the dashboard."""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


from src.database import get_db_manager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "database" / "hidden_risk.db"


def get_connection():
    """Get a database connection, initializing schema if DB file does not exist yet."""
    mgr = get_db_manager(DB_PATH)
    if not DB_PATH.exists():
        mgr.init_database()
    return mgr.get_connection()



def get_tickers() -> List[str]:
    """Get list of available tickers."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT ticker 
            FROM transcripts 
            ORDER BY ticker
            """
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def get_transcripts_for_ticker(ticker: str) -> List[Dict]:
    """Get all transcripts for a ticker."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT t.transcript_id, t.date, t.company_name, t.year, t.quarter,
                   ts.overall_hidden_risk, ts.average_risk, ts.hedging_avg,
                   ts.evasiveness_avg, ts.tone_shift_avg
            FROM transcripts t
            LEFT JOIN transcript_scores ts ON t.transcript_id = ts.transcript_id
            WHERE t.ticker = ?
            ORDER BY t.date DESC
            """,
            (ticker,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_transcript_detail(transcript_id: str) -> Optional[Dict]:
    """Get transcript metadata."""
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT t.*, ts.*
            FROM transcripts t
            LEFT JOIN transcript_scores ts ON t.transcript_id = ts.transcript_id
            WHERE t.transcript_id = ?
            """,
            (transcript_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_sentences(transcript_id: str) -> List[Dict]:
    """Get all sentences for a transcript."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT s.*, seg.section, seg.speaker, seg.speaker_role
            FROM sentences s
            JOIN segments seg ON s.segment_id = seg.segment_id
            WHERE s.transcript_id = ?
            ORDER BY seg.sequence, s.sentence_number
            """,
            (transcript_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_risk_scores(transcript_id: str) -> List[Dict]:
    """Get risk scores for sentences in a transcript."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT rs.*, s.text
            FROM risk_scores rs
            JOIN sentences s ON rs.sentence_id = s.sentence_id
            WHERE rs.transcript_id = ?
            ORDER BY rs.score_id
            """,
            (transcript_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_claims(transcript_id: str) -> List[Dict]:
    """Get claims for a transcript."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM claims
            WHERE transcript_id = ?
            ORDER BY claim_id
            """,
            (transcript_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_verifications(claim_ids: List[str]) -> List[Dict]:
    """Get verifications for a list of claim IDs."""
    if not claim_ids:
        return []
    
    conn = get_connection()
    try:
        placeholders = ",".join("?" * len(claim_ids))
        rows = conn.execute(
            f"""
            SELECT * FROM verification
            WHERE claim_id IN ({placeholders})
            """,
            tuple(claim_ids)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_evidence_for_claims(claim_ids: List[str]) -> List[Dict]:
    """Get evidence for a list of claim IDs."""
    if not claim_ids:
        return []
    
    conn = get_connection()
    try:
        placeholders = ",".join("?" * len(claim_ids))
        rows = conn.execute(
            f"""
            SELECT * FROM evidence
            WHERE claim_id IN ({placeholders})
            ORDER BY authority_score DESC
            """,
            tuple(claim_ids)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_market_reaction(transcript_id: str) -> Optional[Dict]:
    """Get market reaction data for a transcript."""
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT * FROM market_events
            WHERE transcript_id = ?
            """,
            (transcript_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_outcomes(transcript_id: str) -> Optional[Dict]:
    """Get outcome data for a transcript."""
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT * FROM outcomes
            WHERE transcript_id = ?
            """,
            (transcript_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_companies_with_scores() -> List[Dict]:
    """Get all companies with their average risk scores."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT t.ticker, t.company_name,
                   COUNT(DISTINCT t.transcript_id) as call_count,
                   AVG(ts.overall_hidden_risk) as avg_risk,
                   AVG(ts.hedging_avg) as avg_hedging,
                   AVG(ts.evasiveness_avg) as avg_evasiveness,
                   AVG(ts.tone_shift_avg) as avg_tone_shift
            FROM transcripts t
            LEFT JOIN transcript_scores ts ON t.transcript_id = ts.transcript_id
            GROUP BY t.ticker, t.company_name
            ORDER BY avg_risk DESC NULLS LAST
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_ticker_comparison(tickers: List[str]) -> List[Dict]:
    """Get comparison data for specific tickers."""
    if not tickers:
        return []
    
    conn = get_connection()
    try:
        placeholders = ",".join("?" * len(tickers))
        rows = conn.execute(
            f"""
            SELECT t.ticker, t.company_name,
                   t.date, t.year, t.quarter,
                   ts.overall_hidden_risk, ts.average_risk,
                   ts.hedging_avg, ts.evasiveness_avg, ts.tone_shift_avg
            FROM transcripts t
            LEFT JOIN transcript_scores ts ON t.transcript_id = ts.transcript_id
            WHERE t.ticker IN ({placeholders})
            ORDER BY t.ticker, t.date DESC
            """,
            tuple(tickers)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
