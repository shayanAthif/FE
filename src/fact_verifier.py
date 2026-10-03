"""
Fact Verification Engine — Phase 3

Compares extracted claims against retrieved external evidence and determines:

  SUPPORTED            — Evidence clearly supports the claim
  CONTRADICTED         — Evidence clearly contradicts the claim
  PARTIALLY_SUPPORTED  — Evidence partially supports the claim or shows mixed results
  NOT_VERIFIABLE       — Insufficient evidence to make a determination

Principle:
  - Stock price movement alone NEVER determines factual claim status.
  - For numerical claims, official financial sources take priority.
  - Source authority (tier 1 > tier 2 > tier 3 > tier 4) is factored in.
  - Temporal validity is enforced.
  - Semantic text similarity used as supplementary signal.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.claim_normalizer import NormalizedClaim
from src.config import get_config
from src.logger import get_logger
from src.providers.base import RetrievedEvidence
from src.temporal_filter import filter_outcome_evidence

logger = get_logger("fact_verifier")


class VerificationStatus(str, Enum):
    SUPPORTED           = "SUPPORTED"
    CONTRADICTED        = "CONTRADICTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    NOT_VERIFIABLE      = "NOT_VERIFIABLE"


class FactVerifier:
    """
    Determines SUPPORTED / CONTRADICTED / PARTIALLY_SUPPORTED / NOT_VERIFIABLE
    status for each claim by comparing it against retrieved evidence.
    """

    MIN_EVIDENCE_TO_VERIFY = 1  # At least this many items required to attempt verification

    def __init__(self):
        self.cfg = get_config()
        self._authority = self.cfg.evidence.authority_weights
        self._min_score = self.cfg.evidence.minimum_evidence_score
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, sublinear_tf=True)

    # ─────────────────────────────────────────────────────────────────────
    # Semantic similarity
    # ─────────────────────────────────────────────────────────────────────

    def _semantic_similarity(self, claim_text: str, evidence_text: str) -> float:
        """TF-IDF cosine similarity between claim and evidence snippet."""
        if not evidence_text or not claim_text:
            return 0.0
        try:
            vecs = self._vectorizer.fit_transform([claim_text, evidence_text])
            return float(cosine_similarity(vecs[0:1], vecs[1:2])[0][0])
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────────────────────────────
    # Numerical comparison
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_numbers_from_text(text: str) -> List[float]:
        """Extract all plausible numerical values from text."""
        raw = re.findall(r"[-+]?\d+[\.,]?\d*", text)
        out = []
        for r in raw:
            try:
                out.append(float(r.replace(",", "")))
            except ValueError:
                pass
        return out

    def _numerical_comparison(
        self,
        claim: NormalizedClaim,
        evidence_numbers: List[float],
    ) -> Optional[float]:
        """
        Compare claimed value/range against numbers found in evidence.

        Returns a score [0, 1]:
          1.0  = strong numerical support (evidence within 10% of claim)
          0.5  = partial match (within 25%) or range overlap
          0.0  = clear contradiction
          None = no numerical claim or no evidence numbers
        """
        if claim.value is None and claim.lower_bound is None:
            return None
        if not evidence_numbers:
            return None

        claim_val = claim.value
        claim_lo  = claim.lower_bound
        claim_hi  = claim.upper_bound

        best = 0.0
        for ev_num in evidence_numbers:
            if claim_val is not None:
                if claim_val == 0:
                    continue
                pct_diff = abs(ev_num - claim_val) / abs(claim_val)
                if pct_diff <= 0.10:
                    best = max(best, 1.0)
                elif pct_diff <= 0.25:
                    best = max(best, 0.5)
                elif pct_diff <= 0.50:
                    best = max(best, 0.2)
            elif claim_lo is not None and claim_hi is not None:
                # Range claim
                if claim_lo <= ev_num <= claim_hi:
                    best = max(best, 1.0)
                elif claim_lo * 0.8 <= ev_num <= claim_hi * 1.2:
                    best = max(best, 0.5)

        return best

    # ─────────────────────────────────────────────────────────────────────
    # Direction comparison
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _direction_check(claim_direction: Optional[str], evidence_text: str) -> Optional[float]:
        """
        Check whether the evidence text's direction matches the claim direction.
        Returns 1.0 (match), 0.0 (contradiction), 0.5 (neutral/ambiguous), None (N/A)
        """
        if not claim_direction or not evidence_text:
            return None

        ev_lower = evidence_text.lower()

        positive_signals = r"\b(grew|growth|increase|increased|higher|improved|improved|strong|surge|beat)\b"
        negative_signals = r"\b(fell|declined|decreased|lower|missed|weak|below|disappoint|short|cut)\b"

        has_pos = bool(re.search(positive_signals, ev_lower))
        has_neg = bool(re.search(negative_signals, ev_lower))

        if claim_direction in ("increase", "grow"):
            if has_pos and not has_neg:
                return 1.0
            elif has_neg and not has_pos:
                return 0.0
            else:
                return 0.5
        elif claim_direction == "decrease":
            if has_neg and not has_pos:
                return 1.0
            elif has_pos and not has_neg:
                return 0.0
            else:
                return 0.5
        elif claim_direction == "flat":
            if not has_pos and not has_neg:
                return 1.0
            return 0.5
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Authority scoring
    # ─────────────────────────────────────────────────────────────────────

    def _authority_weight(self, evidence: RetrievedEvidence) -> float:
        """Map source_type and authority_score to a configurable weight."""
        if evidence.source_type == "sec":
            return self._authority.get("tier_1", 1.00)
        elif evidence.authority_score >= 0.70:
            return self._authority.get("tier_2", 0.75)
        elif evidence.authority_score >= 0.45:
            return self._authority.get("tier_3", 0.50)
        else:
            return self._authority.get("tier_4", 0.25)

    # ─────────────────────────────────────────────────────────────────────
    # Core verification
    # ─────────────────────────────────────────────────────────────────────

    def verify(
        self,
        claim: NormalizedClaim,
        evidence_list: List[RetrievedEvidence],
        call_date: str,
    ) -> Dict:
        """
        Produce a verification result for one claim against available evidence.

        Returns
        -------
        dict with:
          status, confidence, reasoning, evidence_ids,
          num_support_signals, num_contradict_signals, temporal_warning
        """
        verification_id = f"ver_{claim.claim_id}_{uuid.uuid4().hex[:6]}"

        # 1. Temporal filtering — only use post-call / outcome evidence for verification
        verif_evidence, contemp_evidence = filter_outcome_evidence(evidence_list, call_date)

        if not verif_evidence:
            return {
                "verification_id": verification_id,
                "claim_id": claim.claim_id,
                "status": VerificationStatus.NOT_VERIFIABLE.value,
                "confidence": 0.20,
                "reasoning": "No post-call or outcome evidence found to verify this claim.",
                "evidence_ids": [],
                "num_support_signals": 0,
                "num_contradict_signals": 0,
                "temporal_warning": False,
            }

        support_score  = 0.0
        contradict_score = 0.0
        total_weight   = 0.0
        evidence_ids   = []
        reasoning_parts = []

        for ev in verif_evidence:
            if not ev.text and not ev.title:
                continue

            ev_text = f"{ev.title or ''} {ev.text or ''}"
            authority_w = self._authority_weight(ev)
            ev_id = f"{claim.claim_id}_{ev.source_type}_{hash(ev.source_url or ev.title) & 0xFFFF:04x}"
            evidence_ids.append(ev_id)

            # Semantic similarity signal
            sem = self._semantic_similarity(claim.claim_text, ev_text)

            # Numerical comparison (official SEC facts take precedence)
            ev_nums = self._extract_numbers_from_text(ev_text)
            num_score = self._numerical_comparison(claim, ev_nums)

            # Directional signal
            dir_score = self._direction_check(claim.direction, ev_text)

            # Aggregate per-evidence support / contradict signals
            # SEC (Tier 1) sources dominate via authority_w
            if ev.source_type == "sec":
                # SEC is authoritative for numerical accounting claims
                if num_score is not None:
                    if num_score >= 0.8:
                        support_score    += authority_w * num_score
                        reasoning_parts.append(f"SEC ({ev.title}): numerical match {num_score:.1%}.")
                    elif num_score >= 0.4:
                        support_score    += authority_w * 0.5
                        contradict_score += authority_w * 0.3
                        reasoning_parts.append(f"SEC ({ev.title}): partial numerical match.")
                    else:
                        contradict_score += authority_w
                        reasoning_parts.append(f"SEC ({ev.title}): numerical contradiction detected.")
                elif dir_score is not None:
                    if dir_score >= 0.8:
                        support_score    += authority_w * dir_score
                    elif dir_score <= 0.2:
                        contradict_score += authority_w
                    else:
                        support_score    += authority_w * 0.4
                elif sem >= 0.30:
                    support_score    += authority_w * sem
                total_weight += authority_w
            else:
                # News / other sources: use semantic + directional signals
                if sem >= 0.20:
                    if dir_score is not None:
                        combined = (sem + dir_score) / 2.0
                    else:
                        combined = sem
                    if combined >= 0.50:
                        support_score    += authority_w * combined
                    elif combined >= 0.25:
                        support_score    += authority_w * 0.4
                        contradict_score += authority_w * 0.2
                    total_weight += authority_w

        # Normalise
        if total_weight > 0:
            sup  = support_score    / total_weight
            con  = contradict_score / total_weight
        else:
            sup  = 0.0
            con  = 0.0

        # Determine status
        if not evidence_ids:
            status = VerificationStatus.NOT_VERIFIABLE.value
            confidence = 0.20
            summary = "No usable evidence items after filtering."
        elif sup >= 0.60 and con < 0.20:
            status = VerificationStatus.SUPPORTED.value
            confidence = round(min(0.50 + sup * 0.50, 0.95), 2)
            summary = f"Evidence primarily supports the claim (support={sup:.2f}, contradict={con:.2f})."
        elif con >= 0.60 and sup < 0.20:
            status = VerificationStatus.CONTRADICTED.value
            confidence = round(min(0.50 + con * 0.50, 0.95), 2)
            summary = f"Evidence primarily contradicts the claim (contradict={con:.2f}, support={sup:.2f})."
        elif sup >= 0.30 or con >= 0.30:
            status = VerificationStatus.PARTIALLY_SUPPORTED.value
            confidence = round(0.45 + abs(sup - con) * 0.30, 2)
            summary = f"Mixed evidence signals (support={sup:.2f}, contradict={con:.2f})."
        else:
            status = VerificationStatus.NOT_VERIFIABLE.value
            confidence = 0.25
            summary = "Insufficient evidence signal strength for a determination."

        if reasoning_parts:
            full_reasoning = summary + " Details: " + " | ".join(reasoning_parts[:3])
        else:
            full_reasoning = summary

        return {
            "verification_id": verification_id,
            "claim_id": claim.claim_id,
            "status": status,
            "confidence": confidence,
            "reasoning": full_reasoning[:800],
            "evidence_ids": evidence_ids,
            "num_support_signals": len([r for r in reasoning_parts if "match" in r.lower() or "support" in r.lower()]),
            "num_contradict_signals": len([r for r in reasoning_parts if "contradict" in r.lower()]),
            "temporal_warning": len(contemp_evidence) > 0 and len(verif_evidence) == 0,
        }

