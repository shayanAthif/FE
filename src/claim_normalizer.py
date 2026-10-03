"""
Claim Normalizer Module — Phase 3

Parses unstructured natural language claims into structured, verifiable fields:
  - metric         : normalized metric name (revenue_growth, eps, capex, etc.)
  - value          : single numeric point estimate
  - lower_bound    : lower bound if range
  - upper_bound    : upper bound if range
  - unit           : 'percent', 'usd', 'usd_millions', 'usd_billions', 'bps'
  - period         : 'next_quarter', 'full_year', 'q1', 'q2', 'q3', 'q4', etc.
  - geography      : 'Europe', 'China', 'North America', 'global', etc.
  - entity         : company or business segment
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from src.claim_extractor import RawClaim
from src.logger import get_logger

logger = get_logger("claim_normalizer")

# Patterns for percentage ranges: "between 8 and 10%", "8% to 10%", "8 - 10%"
_RANGE_PERCENT_REGEX = re.compile(
    r"(?:between\s+(\d+[\.,]?\d*)\s+(?:and|to)\s+(\d+[\.,]?\d*)\s*(?:percent|%)|"
    r"(\d+[\.,]?\d*)\s*(?:%|percent)?\s*(?:to|-)\s*(\d+[\.,]?\d*)\s*(?:percent|%))",
    re.IGNORECASE,
)

# Pattern for single percentage: "10%", "10.5 percent"
_SINGLE_PERCENT_REGEX = re.compile(
    r"(\d+[\.,]?\d*)\s*(?:percent\b|%)",
    re.IGNORECASE,
)

# Pattern for basis points: "50 bps", "100 basis points"
_BPS_REGEX = re.compile(
    r"(\d+[\.,]?\d*)\s*(?:bps|basis points)\b",
    re.IGNORECASE,
)

# Pattern for dollar currency with magnitude: "$2 billion", "$500 million", "$1.5B"
_DOLLAR_MAGNITUDE_REGEX = re.compile(
    r"\$\s*(\d+[\.,]?\d*)\s*(billion|million|thousand|b|m|k)?\b",
    re.IGNORECASE,
)

# Temporal patterns
_PERIOD_PATTERNS = [
    (r"\b(?:next quarter|q[1-4]\s*(?:outlook|next)|coming quarter)\b", "next_quarter"),
    (r"\b(?:full year|fiscal year|fy\s*\d{2,4}|annual|for the year)\b", "full_year"),
    (r"\b(?:first half|1h|first six months)\b", "first_half"),
    (r"\b(?:second half|2h|back half)\b", "second_half"),
    (r"\b(?:q1|first quarter)\b", "q1"),
    (r"\b(?:q2|second quarter)\b", "q2"),
    (r"\b(?:q3|third quarter)\b", "q3"),
    (r"\b(?:q4|fourth quarter)\b", "q4"),
    (r"\b(?:long[- ]term|three to five years)\b", "long_term"),
    (r"\b(?:current quarter|this quarter)\b", "current_quarter"),
]

# Geography patterns
_GEOGRAPHY_PATTERNS = [
    (r"\b(?:north america|us|united states|domestic)\b", "North America"),
    (r"\b(?:europe|european|emea)\b", "Europe"),
    (r"\b(?:china|chinese)\b", "China"),
    (r"\b(?:asia|asia[- ]pacific|apac)\b", "Asia"),
    (r"\b(?:latin america|latam|brazil)\b", "Latin America"),
    (r"\b(?:global|worldwide|international)\b", "Global"),
]

# Metric mapping
_METRIC_MAP = [
    (r"\b(?:revenue growth|sales growth|top[- ]line growth)\b", "revenue_growth"),
    (r"\b(?:gross margin)\b", "gross_margin"),
    (r"\b(?:operating margin|operating profit margin)\b", "operating_margin"),
    (r"\b(?:eps|earnings per share)\b", "eps"),
    (r"\b(?:net income|net profit)\b", "net_income"),
    (r"\b(?:operating income|operating profit)\b", "operating_income"),
    (r"\b(?:capex|capital expenditure|capital spending)\b", "capex"),
    (r"\b(?:free cash flow|fcf)\b", "free_cash_flow"),
    (r"\b(?:debt reduction|debt paydown|leverage)\b", "debt_reduction"),
    (r"\b(?:cost reduction|cost savings)\b", "cost_reduction"),
    (r"\b(?:organic revenue|organic growth)\b", "organic_growth"),
    (r"\b(?:revenue|sales)\b", "revenue"),
]


@dataclass
class NormalizedClaim:
    claim_id: str
    transcript_id: str
    segment_id: Optional[str]
    sentence_id: Optional[str]
    speaker: Optional[str]
    claim_text: str
    claim_type: str
    entity: Optional[str]
    metric: str
    value: Optional[float]
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    unit: Optional[str]
    period: str
    geography: Optional[str]
    direction: Optional[str]
    confidence: float

    def to_dict(self) -> Dict:
        return {
            "claim_id": self.claim_id,
            "transcript_id": self.transcript_id,
            "segment_id": self.segment_id,
            "sentence_id": self.sentence_id,
            "speaker": self.speaker,
            "claim_text": self.claim_text,
            "claim_type": self.claim_type,
            "entity": self.entity,
            "metric": self.metric,
            "value": self.value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "unit": self.unit,
            "period": self.period,
            "geography": self.geography,
            "direction": self.direction,
            "confidence": self.confidence,
        }


class ClaimNormalizer:
    """Normalizes numerical, temporal, metric, and geographical expressions in claims."""

    def normalize(self, raw_claim: RawClaim, ticker: str = "") -> NormalizedClaim:
        text = raw_claim.claim_text

        # 1. Parse Metric
        metric = self._parse_metric(text, raw_claim.claim_type)

        # 2. Parse Numbers & Ranges
        val, lower, upper, unit = self._parse_numbers(text)

        # 3. Parse Period
        period = self._parse_period(text)

        # 4. Parse Geography
        geography = self._parse_geography(text)

        # 5. Entity
        entity = ticker if ticker else None

        return NormalizedClaim(
            claim_id=raw_claim.claim_id,
            transcript_id=raw_claim.transcript_id,
            segment_id=raw_claim.segment_id,
            sentence_id=raw_claim.sentence_id,
            speaker=raw_claim.speaker,
            claim_text=raw_claim.claim_text,
            claim_type=raw_claim.claim_type,
            entity=entity,
            metric=metric,
            value=val,
            lower_bound=lower,
            upper_bound=upper,
            unit=unit,
            period=period,
            geography=geography,
            direction=raw_claim.direction,
            confidence=raw_claim.confidence,
        )

    def _parse_metric(self, text: str, default_type: str) -> str:
        for pat, met in _METRIC_MAP:
            if re.search(pat, text, re.IGNORECASE):
                return met
        return default_type

    def _parse_numbers(self, text: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
        # Check range percentage first: "between 8 and 10%" or "8 to 10%"
        m_range = _RANGE_PERCENT_REGEX.search(text)
        if m_range:
            g = [x for x in m_range.groups() if x is not None]
            if len(g) >= 2:
                try:
                    low = float(g[0].replace(",", ""))
                    high = float(g[1].replace(",", ""))
                    val = (low + high) / 2.0
                    return round(val, 2), round(low, 2), round(high, 2), "percent"
                except ValueError:
                    pass

        # Check basis points: "50 bps"
        m_bps = _BPS_REGEX.search(text)
        if m_bps:
            try:
                val = float(m_bps.group(1).replace(",", ""))
                return val, None, None, "bps"
            except ValueError:
                pass

        # Check single percentage: "10%"
        m_pct = _SINGLE_PERCENT_REGEX.search(text)
        if m_pct:
            try:
                val = float(m_pct.group(1).replace(",", ""))
                return val, None, None, "percent"
            except ValueError:
                pass

        # Check dollar amounts: "$2 billion", "$500M"
        m_dol = _DOLLAR_MAGNITUDE_REGEX.search(text)
        if m_dol:
            try:
                num = float(m_dol.group(1).replace(",", ""))
                mag = (m_dol.group(2) or "").lower()
                unit = "usd"
                if mag in ("billion", "b"):
                    num = num * 1_000_000_000
                    unit = "usd_billions"
                elif mag in ("million", "m"):
                    num = num * 1_000_000
                    unit = "usd_millions"
                elif mag in ("thousand", "k"):
                    num = num * 1_000
                return round(num, 2), None, None, unit
            except ValueError:
                pass

        return None, None, None, None

    def _parse_period(self, text: str) -> str:
        for pat, per in _PERIOD_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                return per
        return "unspecified_period"

    def _parse_geography(self, text: str) -> Optional[str]:
        for pat, geo in _GEOGRAPHY_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                return geo
        return None
