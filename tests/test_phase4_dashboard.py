"""
Phase 4 Test Suite — Streamlit Analyst Dashboard & Outcomes Pipeline.

Tests:
  1. Data Loader: verifies database queries return expected schema and data.
  2. Outcome Extraction: verifies latest before date math & metrics extraction.
  3. Dashboard Components: verifies cards and viewers handle normal and edge case data.
"""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from dashboard.components import data_loader
from dashboard.components.risk_score_card import risk_score_card
from dashboard.components.market_viewer import _metric_card, event_timeline
from dashboard.components.claim_card import claim_card
from dashboard.components.sentence_viewer import sentence_viewer, explainable_sentence
from scripts.populate_outcomes import find_latest_before, extract_recent_metrics


class TestDataLoader:

    def test_get_tickers(self):
        tickers = data_loader.get_tickers()
        assert isinstance(tickers, list)
        assert len(tickers) > 0
        assert "CLX" in tickers

    def test_get_transcripts_for_ticker(self):
        transcripts = data_loader.get_transcripts_for_ticker("CLX")
        assert isinstance(transcripts, list)
        assert len(transcripts) > 0
        first = transcripts[0]
        assert "transcript_id" in first
        assert "overall_hidden_risk" in first

    def test_get_transcript_detail(self):
        transcripts = data_loader.get_transcripts_for_ticker("CLX")
        tid = transcripts[0]["transcript_id"]
        detail = data_loader.get_transcript_detail(tid)
        assert detail is not None
        assert detail["transcript_id"] == tid
        assert "ticker" in detail

    def test_get_sentences_and_scores(self):
        transcripts = data_loader.get_transcripts_for_ticker("CLX")
        tid = transcripts[0]["transcript_id"]
        sentences = data_loader.get_sentences(tid)
        assert isinstance(sentences, list)
        assert len(sentences) > 0
        assert "text" in sentences[0]

        scores = data_loader.get_risk_scores(tid)
        assert isinstance(scores, list)
        assert len(scores) > 0
        assert "hidden_risk_score" in scores[0]

    def test_get_claims_and_evidence(self):
        transcripts = data_loader.get_transcripts_for_ticker("CLX")
        tid = transcripts[0]["transcript_id"]
        claims = data_loader.get_claims(tid)
        assert isinstance(claims, list)
        
        cids = [c["claim_id"] for c in claims]
        verifications = data_loader.get_verifications(cids)
        assert isinstance(verifications, list)

        evidence = data_loader.get_evidence_for_claims(cids)
        assert isinstance(evidence, list)

    def test_empty_claims_or_evidence(self):
        assert data_loader.get_verifications([]) == []
        assert data_loader.get_evidence_for_claims([]) == []
        assert data_loader.get_ticker_comparison([]) == []

    def test_get_market_reaction_and_outcomes(self):
        transcripts = data_loader.get_transcripts_for_ticker("CLX")
        tid = transcripts[0]["transcript_id"]
        market = data_loader.get_market_reaction(tid)
        assert market is not None
        assert "return_1d" in market

        outcome = data_loader.get_outcomes(tid)
        assert outcome is not None

    def test_get_all_companies_with_scores(self):
        companies = data_loader.get_all_companies_with_scores()
        assert len(companies) > 0
        assert "avg_risk" in companies[0]


class TestPopulateOutcomesHelpers:

    def test_find_latest_before_returns_nearest(self):
        entries = [
            {"filed": "2020-01-15", "val": 100_000_000},
            {"filed": "2020-02-15", "val": 200_000_000},
            {"filed": "2020-05-15", "val": 300_000_000},
        ]
        # As of 2020-03-01, should pick 2020-02-15 entry (200.0 M)
        val = find_latest_before(entries, as_of_date="2020-03-01", lookback_days=100)
        assert val == 200.0

    def test_find_latest_before_ignores_future_and_out_of_window(self):
        entries = [
            {"filed": "2019-01-01", "val": 50_000_000},
            {"filed": "2020-06-01", "val": 500_000_000},
        ]
        # Window of 60 days before 2020-03-01 excludes both
        val = find_latest_before(entries, as_of_date="2020-03-01", lookback_days=60)
        assert val is None

    def test_extract_recent_metrics(self):
        facts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [{"filed": "2020-02-01", "val": 1_000_000_000}]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [{"filed": "2020-02-01", "val": 250_000_000}]
                        }
                    }
                }
            }
        }
        rev, earn = extract_recent_metrics(facts, as_of_date="2020-03-01", lookback_days=365)
        assert rev == 1000.0
        assert earn == 250.0


class TestDashboardComponents:

    @patch("dashboard.components.risk_score_card.st")
    def test_risk_score_card(self, mock_st):
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        score = risk_score_card(
            title="Hedging",
            score=75.0,
            max_score=100.0,
            details={"modal_count": 3}
        )
        assert score == 75.0
        mock_st.progress.assert_called_once()

    @patch("dashboard.components.claim_card.st")
    def test_claim_card(self, mock_st):
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_st.expander.return_value.__enter__.return_value = MagicMock()
        mock_st.container.return_value.__enter__.return_value = MagicMock()

        claim = {
            "claim_text": "Revenue increased 5%",
            "metric": "revenue",
            "value": 5,
            "unit": "%",
            "period": "Q1",
            "direction": "increase"
        }
        verification = {
            "status": "SUPPORTED",
            "confidence": 0.95,
            "reasoning": "Confirmed via 10-Q"
        }
        evidence = [{
            "source_type": "sec_filing",
            "source_name": "10-Q",
            "publication_date": "2020-02-01",
            "relevance_score": 0.9,
            "source_url": "https://sec.gov",
            "text": "Revenue was up 5 percent."
        }]

        c, v = claim_card(claim, verification, evidence, show_details=True)
        assert c["claim_text"] == "Revenue increased 5%"
        assert v["status"] == "SUPPORTED"

    @patch("dashboard.components.market_viewer.st")
    def test_market_viewer_components(self, mock_st):
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        events = [{
            "date": "2020-01-01",
            "title": "Call",
            "description": "Earnings call held",
            "type": "earnings"
        }]
        mock_st.container.return_value.__enter__.return_value = MagicMock()
        event_timeline(events)

        _metric_card("Return", 0.0512, format_pct=True)
        mock_st.metric.assert_called_with("Return", "5.12%")

    @patch("dashboard.components.sentence_viewer.st")
    def test_sentence_viewer(self, mock_st):
        mock_st.container.return_value.__enter__.return_value = MagicMock()
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        sentences = [{"sentence_id": "s1", "text": "This is test sentence."}]
        sentence_viewer(sentences, highlighted_ids=["s1"])
        mock_st.markdown.assert_called()

        explainable_sentence(
            sentence={"text": "Sentence", "detected_phrases": ["may"]},
            risk_factors={"hedging": 0.5, "evasiveness": 0.1, "tone_shift": 0.2}
        )
