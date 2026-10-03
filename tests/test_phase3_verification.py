"""
Phase 3 Test Suite — Real-World Claim + Evidence Verification

Tests:
  1. Claim Extraction: detection, boilerplate filtering, typing.
  2. Claim Normalization: percentage, range, currency, temporal parsing.
  3. Temporal Filter: contemporaneous vs post-call vs outcome, non-leakage check.
  4. Fact Verifier: numerical match/mismatch, directional match, status values.
  5. Market Provider: trading-day window returns & realized volatility calculation.
  6. Integration: end-to-end processing of a sample transcript.
"""

from __future__ import annotations

import pytest
import pandas as pd
from datetime import datetime

from src.claim_extractor import ClaimExtractor, RawClaim
from src.claim_normalizer import ClaimNormalizer, NormalizedClaim
from src.temporal_filter import TemporalCategory, categorise_evidence, filter_outcome_evidence, validate_no_future_leak
from src.fact_verifier import FactVerifier, VerificationStatus
from src.providers.base import RetrievedEvidence
from src.providers.market.market_provider import MarketProvider


class TestClaimExtractor:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.extractor = ClaimExtractor()

    def test_filter_safe_harbor_boilerplate(self):
        text = "This presentation contains forward-looking statements subject to risks and uncertainties."
        assert not self.extractor.is_claim_candidate(text)

    def test_extract_guidance_claim(self):
        text = "We expect revenue growth of approximately 10% next quarter."
        assert self.extractor.is_claim_candidate(text)
        assert self.extractor.determine_claim_type(text) in ("guidance", "revenue")
        assert self.extractor.extract_direction(text) == "increase"

    def test_extract_margin_claim(self):
        text = "Operating margins are projected to expand by 50 bps."
        assert self.extractor.is_claim_candidate(text)
        assert self.extractor.determine_claim_type(text) == "margin"
        assert self.extractor.extract_direction(text) == "increase"

    def test_extract_declining_demand_claim(self):
        text = "We saw order intake and demand decline across Europe."
        assert self.extractor.is_claim_candidate(text)
        assert self.extractor.determine_claim_type(text) == "demand"
        assert self.extractor.extract_direction(text) == "decrease"


class TestClaimNormalizer:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.normalizer = ClaimNormalizer()

    def test_normalize_single_percentage(self):
        raw = RawClaim(
            claim_id="c1", transcript_id="t1", segment_id="s1", sentence_id="sent1",
            speaker="CEO", claim_text="We expect revenue growth of 12% next quarter.",
            claim_type="revenue", direction="increase", has_numerical=True, confidence=0.85
        )
        norm = self.normalizer.normalize(raw, ticker="PEP")
        assert norm.value == 12.0
        assert norm.unit == "percent"
        assert norm.period == "next_quarter"
        assert norm.metric in ("revenue_growth", "revenue")

    def test_normalize_range_percentage(self):
        raw = RawClaim(
            claim_id="c2", transcript_id="t1", segment_id="s1", sentence_id="sent2",
            speaker="CFO", claim_text="Gross margin will be between 8 and 10% for the full year.",
            claim_type="margin", direction="increase", has_numerical=True, confidence=0.85
        )
        norm = self.normalizer.normalize(raw, ticker="CLX")
        assert norm.lower_bound == 8.0
        assert norm.upper_bound == 10.0
        assert norm.value == 9.0  # midpoint
        assert norm.unit == "percent"
        assert norm.period == "full_year"

    def test_normalize_dollar_magnitude(self):
        raw = RawClaim(
            claim_id="c3", transcript_id="t1", segment_id="s1", sentence_id="sent3",
            speaker="CEO", claim_text="Capex will be capped at $2 billion in North America.",
            claim_type="capex", direction=None, has_numerical=True, confidence=0.80
        )
        norm = self.normalizer.normalize(raw, ticker="RTX")
        assert norm.value == 2_000_000_000.0
        assert norm.geography == "North America"
        assert norm.metric == "capex"


class TestTemporalLogic:

    def test_categorise_evidence_windows(self):
        call_date = "2024-04-25"

        ev_before = RetrievedEvidence(
            source_type="sec", source_name="10-K", source_url="http://sec/1",
            publication_date="2024-04-20", title="10-K", text="Past year results"
        )
        assert categorise_evidence(ev_before, call_date) == TemporalCategory.CONTEMPORANEOUS

        ev_post = RetrievedEvidence(
            source_type="news", source_name="News", source_url="http://news/1",
            publication_date="2024-05-15", title="Update", text="Mid-quarter check"
        )
        assert categorise_evidence(ev_post, call_date) == TemporalCategory.POST_CALL

        ev_outcome = RetrievedEvidence(
            source_type="sec", source_name="10-Q", source_url="http://sec/2",
            publication_date="2024-07-28", title="10-Q", text="Q2 actuals reported"
        )
        assert categorise_evidence(ev_outcome, call_date) == TemporalCategory.OUTCOME

    def test_filter_outcome_evidence(self):
        call_date = "2024-04-25"
        items = [
            RetrievedEvidence("sec", "10-K", "url1", "2024-04-20", "Prior 10-K", "prior text"),
            RetrievedEvidence("sec", "10-Q", "url2", "2024-07-28", "Later 10-Q", "reported 10% growth"),
        ]
        verif, contemp = filter_outcome_evidence(items, call_date)
        assert len(verif) == 1
        assert verif[0].publication_date == "2024-07-28"
        assert len(contemp) == 1

    def test_no_future_leak_validation(self):
        call_date = "2024-04-25"
        clean_inputs = {"text": "We expect growth.", "hedging_score": 25.0}
        post_ev = [
            RetrievedEvidence("news", "News", "http://leaked-url.com", "2024-06-01", "Later event", "Subsequent event")
        ]
        is_clean, violations = validate_no_future_leak(clean_inputs, post_ev, call_date)
        assert is_clean
        assert len(violations) == 0

        leaked_inputs = {"text": "Reference to http://leaked-url.com in score"}
        is_clean, violations = validate_no_future_leak(leaked_inputs, post_ev, call_date)
        assert not is_clean
        assert len(violations) > 0


class TestFactVerifier:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.verifier = FactVerifier()

    def test_numerical_support(self):
        claim = NormalizedClaim(
            claim_id="c1", transcript_id="t1", segment_id="s1", sentence_id="sent1",
            speaker="CEO", claim_text="We expect revenue growth of 10%.",
            claim_type="revenue", entity="PEP", metric="revenue_growth",
            value=10.0, lower_bound=None, upper_bound=None, unit="percent",
            period="next_quarter", geography=None, direction="increase", confidence=0.90
        )
        evidence = [
            RetrievedEvidence(
                source_type="sec", source_name="Form 10-Q", source_url="http://sec/q2",
                publication_date="2024-07-25", title="Q2 10-Q",
                text="The company delivered 9.8% revenue growth for the quarter.",
                authority_score=1.0, relevance_score=0.95
            )
        ]
        res = self.verifier.verify(claim, evidence, call_date="2024-04-25")
        assert res["status"] == VerificationStatus.SUPPORTED.value
        assert res["confidence"] >= 0.70

    def test_numerical_contradiction(self):
        claim = NormalizedClaim(
            claim_id="c2", transcript_id="t1", segment_id="s1", sentence_id="sent2",
            speaker="CEO", claim_text="We anticipate 15% revenue expansion.",
            claim_type="revenue", entity="PEP", metric="revenue_growth",
            value=15.0, lower_bound=None, upper_bound=None, unit="percent",
            period="next_quarter", geography=None, direction="increase", confidence=0.90
        )
        evidence = [
            RetrievedEvidence(
                source_type="sec", source_name="Form 10-Q", source_url="http://sec/q2",
                publication_date="2024-07-25", title="Q2 10-Q",
                text="Revenue declined by 4.2% as orders fell significantly below expectations.",
                authority_score=1.0, relevance_score=0.95
            )
        ]
        res = self.verifier.verify(claim, evidence, call_date="2024-04-25")
        assert res["status"] == VerificationStatus.CONTRADICTED.value

    def test_not_verifiable_without_evidence(self):
        claim = NormalizedClaim(
            claim_id="c3", transcript_id="t1", segment_id="s1", sentence_id="sent3",
            speaker="CEO", claim_text="We believe our brand equity will improve.",
            claim_type="other", entity="PEP", metric="other",
            value=None, lower_bound=None, upper_bound=None, unit=None,
            period="long_term", geography=None, direction="increase", confidence=0.60
        )
        res = self.verifier.verify(claim, [], call_date="2024-04-25")
        assert res["status"] == VerificationStatus.NOT_VERIFIABLE.value


class TestMarketProvider:

    def test_trading_day_reaction(self):
        provider = MarketProvider()
        reaction = provider.calculate_market_reaction("PEP", "2008-02-08")
        assert reaction is not None
        assert reaction.ticker == "PEP"
        assert reaction.price_before is not None
        assert reaction.price_after is not None
        assert "1d" in reaction.returns
        assert "5d" in reaction.returns
        assert "5d" in reaction.volatilities
