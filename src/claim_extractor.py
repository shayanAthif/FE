"""
Claim Extraction Module — Phase 3

Identifies factual, measurable, or forward-looking claims made in earnings calls.
Extracts structured candidate claims while filtering out legal safe-harbor boilerplate.

Claim Types:
  - guidance
  - revenue
  - margin
  - earnings
  - demand
  - sales
  - costs
  - debt
  - capex
  - cash_flow
  - market_conditions
  - other
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from src.logger import get_logger

logger = get_logger("claim_extractor")

# Regex indicators for claim classification (specific metrics precede generic guidance)
_CLAIM_PATTERNS = [
    # Margin / Profitability
    (r"\b(?:gross margin|operating margin|ebitda margin|profit margin|margin expansion|margins)\b", "margin"),
    # Revenue / Sales
    (r"\b(?:revenue|sales|top[- ]line|turnover|net sales)\b", "revenue"),
    # Earnings / EPS
    (r"\b(?:eps|earnings per share|net income|operating income|ebitda|profitability)\b", "earnings"),
    # Capex / Capital allocation
    (r"\b(?:capex|capital expenditure|capital spending|investments in capacity)\b", "capex"),
    # Debt / Leverage
    (r"\b(?:debt|leverage|deleveraging|debt reduction|borrowing|credit facility)\b", "debt"),
    # Cash flow
    (r"\b(?:free cash flow|operating cash flow|cash generation|cash conversion)\b", "cash_flow"),
    # Costs / Expenses
    (r"\b(?:cost reduction|cost savings|restructuring charges|operating expenses|sg&a)\b", "costs"),
    # Demand / Orders
    (r"\b(?:demand|order intake|backlog|order book|bookings|customer demand)\b", "demand"),
    # Market conditions / Geography
    (r"\b(?:market conditions|macroeconomic environment|headwinds|tailwinds|inflation)\b", "market_conditions"),
    # Generic Guidance / Forward looking
    (r"\b(?:guidance|forecast|outlook|target|projecting|projected|expect(?:s|ed|ing)? to deliver)\b", "guidance"),
]

# Numerical indicators (percentages, dollar amounts, multiples, numbers)
_NUMERICAL_REGEX = re.compile(
    r"(?:\$\s*\d+[\.,]?\d*\s*(?:billion|million|b|m)?|\b\d+[\.,]?\d*\s*(?:percent|%|bps|basis points|cents)\b|\bbetween\s+\d+\s+and\s+\d+\b)",
    re.IGNORECASE,
)

# Directional keywords
_DIRECTION_REGEX = re.compile(
    r"\b(grow|growth|increase|increasing|increased|higher|accelerat\w*|improv\w*|up|expand\w*|declin\w*|decreas\w*|lower|slow\w*|down|contract\w*|flat|stable)\b",
    re.IGNORECASE,
)

# Safe-harbor and introductory boilerplate to filter out
_BOILERPLATE_REGEX = re.compile(
    r"(?:forward[- ]looking statements|safe harbor|sec filings|press release is available|"
    r"conference call will contain|subject to risks and uncertainties|risk factors detailed|"
    r"actual results may differ materially|please refer to our form 10[- ]k)",
    re.IGNORECASE,
)


@dataclass
class RawClaim:
    claim_id: str
    transcript_id: str
    segment_id: Optional[str]
    sentence_id: Optional[str]
    speaker: Optional[str]
    claim_text: str
    claim_type: str
    direction: Optional[str]
    has_numerical: bool
    confidence: float


class ClaimExtractor:
    """Extracts factual and guidance claims from earnings call sentences."""

    def __init__(self, min_confidence: float = 0.50):
        self.min_confidence = min_confidence

    def is_claim_candidate(self, text: str, section: str = "") -> bool:
        """Check if a sentence is a potential claim and not generic boilerplate."""
        if not text or len(text.strip().split()) < 4:
            return False

        # Exclude opening disclaimers/safe harbor
        if _BOILERPLATE_REGEX.search(text):
            return False

        # Must match either a claim topic pattern OR a financial numerical expression
        has_topic = any(re.search(pat, text, re.IGNORECASE) for pat, _ in _CLAIM_PATTERNS)
        has_num = bool(_NUMERICAL_REGEX.search(text))
        has_dir = bool(_DIRECTION_REGEX.search(text))

        return (has_topic and (has_num or has_dir)) or (has_topic and section == "prepared_remarks")

    def determine_claim_type(self, text: str) -> str:
        """Categorize the claim into a specific financial type."""
        for pattern, claim_type in _CLAIM_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return claim_type
        return "other"

    def extract_direction(self, text: str) -> Optional[str]:
        """Extract directional sentiment of the claim."""
        m = _DIRECTION_REGEX.search(text)
        if not m:
            return None
        word = m.group(1).lower()
        if word in ("grow", "growth", "increase", "increasing", "increased", "higher", "accelerating", "improve", "improving", "improved", "up", "expand", "expanding", "expansion"):
            return "increase"
        elif word in ("decline", "declining", "declined", "decrease", "decreasing", "decreased", "lower", "slowing", "down", "contracting"):
            return "decrease"
        elif word in ("flat", "stable"):
            return "flat"
        return word

    def extract_claims_from_sentences(
        self,
        transcript_id: str,
        sentences: List[Dict],
    ) -> List[RawClaim]:
        """
        Scan sentences and extract candidate claims.
        
        Parameters
        ----------
        transcript_id : ID of transcript
        sentences     : list of sentence dicts from database
        """
        claims: List[RawClaim] = []
        claim_idx = 1

        for sent in sentences:
            text = sent.get("text", "")
            section = sent.get("section", "")
            speaker = sent.get("speaker", "")
            speaker_role = sent.get("speaker_role", "")

            # Primarily focus on executive statements or analysts asking specific figures
            if not self.is_claim_candidate(text, section=section):
                continue

            claim_type = self.determine_claim_type(text)
            direction = self.extract_direction(text)
            has_num = bool(_NUMERICAL_REGEX.search(text))

            # Confidence scoring
            confidence = 0.50
            if has_num:
                confidence += 0.30
            if direction:
                confidence += 0.15
            if speaker_role == "executive":
                confidence += 0.05
            confidence = min(round(confidence, 2), 1.0)

            if confidence < self.min_confidence:
                continue

            claim_id = f"{transcript_id}_claim_{claim_idx:03d}"
            claims.append(
                RawClaim(
                    claim_id=claim_id,
                    transcript_id=transcript_id,
                    segment_id=sent.get("segment_id"),
                    sentence_id=sent.get("sentence_id"),
                    speaker=speaker,
                    claim_text=text.strip(),
                    claim_type=claim_type,
                    direction=direction,
                    has_numerical=has_num,
                    confidence=confidence,
                )
            )
            claim_idx += 1

        logger.debug(f"[{transcript_id}] Extracted {len(claims)} candidate claims.")
        return claims
