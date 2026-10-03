"""
Temporal Evidence Logic — Phase 3

CRITICAL: Every piece of evidence must be categorised by its temporal relationship
to the earnings call. This prevents data leakage into the Hidden Risk Score.

Three evidence windows:
  CONTEMPORANEOUS  — published at or before call_date (available at call time)
  POST_CALL        — published within 90 days after call_date (context window)
  OUTCOME          — published 91+ days after call_date (later outcome evidence)

IMPORTANT:
  - Contemporaneous evidence CANNOT be used to verify forward-looking claims
    (the future hasn't happened yet).
  - POST_CALL and OUTCOME evidence ARE used for claim verification/fact checking.
  - NOTHING from any temporal category is used to recalculate Hidden Risk Score.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Tuple
from src.providers.base import RetrievedEvidence
from src.logger import get_logger

logger = get_logger("temporal_filter")


class TemporalCategory(str, Enum):
    CONTEMPORANEOUS = "contemporaneous"   # At/before call date
    POST_CALL       = "post_call"         # 1-90 days after call
    OUTCOME         = "outcome"           # 90+ days after call


def categorise_evidence(
    evidence: RetrievedEvidence,
    call_date: str,
    post_call_window_days: int = 90,
) -> TemporalCategory:
    """
    Assign a temporal category to an evidence item based on call_date.

    Parameters
    ----------
    evidence                : the evidence item to categorise
    call_date               : the date of the earnings call (ISO YYYY-MM-DD)
    post_call_window_days   : days after call that define the POST_CALL window
    """
    if not evidence.publication_date:
        return TemporalCategory.POST_CALL  # conservative: treat unknown as post-call

    try:
        ev_date  = datetime.strptime(evidence.publication_date[:10], "%Y-%m-%d")
        call_dt  = datetime.strptime(call_date[:10], "%Y-%m-%d")
    except ValueError:
        return TemporalCategory.POST_CALL

    delta_days = (ev_date - call_dt).days

    if delta_days <= 0:
        return TemporalCategory.CONTEMPORANEOUS
    elif delta_days <= post_call_window_days:
        return TemporalCategory.POST_CALL
    else:
        return TemporalCategory.OUTCOME


def filter_outcome_evidence(
    evidence_list: List[RetrievedEvidence],
    call_date: str,
    post_call_window_days: int = 90,
) -> Tuple[List[RetrievedEvidence], List[RetrievedEvidence]]:
    """
    Separate evidence into:
      (verification_evidence, contemporaneous_only)

    Only POST_CALL and OUTCOME evidence is suitable for verifying forward-looking claims.
    CONTEMPORANEOUS evidence is returned separately for context but not used for claim verdicts.

    Returns
    -------
    (verification_items, contemporaneous_items)
    """
    verification_items: List[RetrievedEvidence] = []
    contemporaneous_items: List[RetrievedEvidence] = []

    for ev in evidence_list:
        cat = categorise_evidence(ev, call_date, post_call_window_days)
        if cat in (TemporalCategory.POST_CALL, TemporalCategory.OUTCOME):
            verification_items.append(ev)
        else:
            contemporaneous_items.append(ev)

    return verification_items, contemporaneous_items


def validate_no_future_leak(
    risk_inputs: dict,
    evidence_list: List[RetrievedEvidence],
    call_date: str,
) -> Tuple[bool, List[str]]:
    """
    Audit function: confirm that no evidence from after the call date has
    been mixed into Hidden Risk Score inputs.

    Parameters
    ----------
    risk_inputs     : dict of features used by risk scorer (should be text-only)
    evidence_list   : evidence items being audited
    call_date       : the earnings call date

    Returns
    -------
    (is_clean, list_of_violations)
    """
    violations: List[str] = []
    for ev in evidence_list:
        cat = categorise_evidence(ev, call_date)
        if cat in (TemporalCategory.POST_CALL, TemporalCategory.OUTCOME):
            # Check if this source_url or text fragment appears in risk_inputs
            ev_url = ev.source_url or ""
            ev_text = ev.text[:40] if ev.text else ""
            for k, v in risk_inputs.items():
                if isinstance(v, str) and (ev_url in v or (ev_text and ev_text in v)):
                    violations.append(
                        f"Future evidence leak detected: '{k}' contains evidence from {ev.publication_date}"
                    )
    return len(violations) == 0, violations

