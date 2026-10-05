"""
Q&A Matcher — Phase 1

Identifies analyst-question / executive-answer pairs within a list of
pre-segmented transcript segments.

Strategy
--------
After segmentation each segment has a speaker_role and section.  In the
Q&A section, the typical pattern is:

  Operator  → bridge / "next question from …"
  Analyst   → question
  Executive → answer (can span multiple consecutive segments from same/different execs)
  [Operator → next question bridge]
  Analyst   → question
  …

This module implements a simple finite-state matching pass:

  State: WAITING_FOR_QUESTION | IN_QUESTION | IN_ANSWER

Robustness rules:
  - Multiple analyst segments before an answer are concatenated into one question.
  - Multiple executive segments after a question (before the next analyst) are
    concatenated into one answer.
  - Operator bridge segments are ignored as question/answer content.
  - Segments outside the q_and_a section are skipped.
  - Q&A pairs missing either the question or the answer are discarded.

Output
------
Each Q&A pair dict contains:
  qa_id               : deterministic ID  (<transcript_id>_qa_<N>)
  transcript_id       : str
  question_segment_id : segment_id of the first question segment
  answer_segment_id   : segment_id of the first answer segment
  analyst_speaker     : str
  executive_speaker   : str
  question_text       : concatenated question text
  answer_text         : concatenated answer text
  sequence            : int (ordering within transcript)

The structure matches the qa_pairs table schema exactly.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.logger import get_logger

logger = get_logger("qa_matcher")

# Section labels that belong to the Q&A portion of a call
_QA_SECTIONS = frozenset({"q_and_a"})

# Roles that can ask questions
_QUESTION_ROLES = frozenset({"analyst", "unknown"})

# Roles that should answer
_ANSWER_ROLES = frozenset({"executive", "unknown"})

# Operator role (skipped as question/answer content)
_OPERATOR_ROLE = "operator"

# Minimum length thresholds to avoid empty/trivial pairs
_MIN_QUESTION_CHARS = 20
_MIN_ANSWER_CHARS = 20


def _is_question_segment(segment: Dict) -> bool:
    """Return True if this segment should be treated as an analyst question."""
    if segment.get("section") not in _QA_SECTIONS:
        return False
    role = segment.get("speaker_role", "unknown")
    if role == _OPERATOR_ROLE:
        return False
    if role in _QUESTION_ROLES:
        return True
    return False


def _is_answer_segment(segment: Dict) -> bool:
    """Return True if this segment should be treated as an executive answer."""
    if segment.get("section") not in _QA_SECTIONS:
        return False
    role = segment.get("speaker_role", "unknown")
    if role == _OPERATOR_ROLE:
        return False
    if role in _ANSWER_ROLES:
        return True
    return False


def _is_operator_bridge(segment: Dict) -> bool:
    """Return True for operator segments that act as question bridges."""
    return segment.get("speaker_role") == _OPERATOR_ROLE


def match_qa_pairs(
    transcript_id: str,
    segments: List[Dict],
) -> List[Dict]:
    """
    Match analyst questions to executive answers in a list of segments.

    Parameters
    ----------
    transcript_id : canonical transcript ID
    segments      : list of segment dicts (from segmentation.py), ordered by sequence

    Returns
    -------
    List of Q&A pair dicts matching the qa_pairs table schema.
    """
    # Sort by sequence to ensure correct ordering
    sorted_segs = sorted(segments, key=lambda s: s.get("sequence", 0))

    qa_pairs: List[Dict] = []
    qa_idx = 0

    # State machine variables
    current_question_text: List[str] = []
    current_question_seg_id: Optional[str] = None
    current_question_speaker: str = ""

    current_answer_text: List[str] = []
    current_answer_seg_id: Optional[str] = None
    current_answer_speaker: str = ""

    state = "WAITING_FOR_QUESTION"

    def _flush_pair() -> Optional[Dict]:
        """Finalize and return the current Q&A pair, or None if incomplete."""
        nonlocal qa_idx
        q_text = " ".join(current_question_text).strip()
        a_text = " ".join(current_answer_text).strip()

        if (
            len(q_text) >= _MIN_QUESTION_CHARS
            and len(a_text) >= _MIN_ANSWER_CHARS
            and current_question_seg_id
            and current_answer_seg_id
        ):
            qa_id = f"{transcript_id}_qa_{qa_idx:03d}"
            qa_idx += 1
            return {
                "qa_id": qa_id,
                "transcript_id": transcript_id,
                "question_segment_id": current_question_seg_id,
                "answer_segment_id": current_answer_seg_id,
                "analyst_speaker": current_question_speaker,
                "executive_speaker": current_answer_speaker,
                "question_text": q_text,
                "answer_text": a_text,
                "sequence": qa_idx - 1,
            }
        return None

    for seg in sorted_segs:
        section = seg.get("section", "")
        role = seg.get("speaker_role", "unknown")
        text = seg.get("text", "").strip()
        seg_id = seg.get("segment_id", "")
        speaker = seg.get("speaker", "")

        # Skip non-Q&A and operator bridge content for state changes
        if section not in _QA_SECTIONS:
            continue

        if _is_operator_bridge(seg):
            # An operator bridge after we already have a question+answer means
            # we should flush the current pair and wait for the next question.
            if state == "IN_ANSWER" and current_question_text and current_answer_text:
                pair = _flush_pair()
                if pair:
                    qa_pairs.append(pair)
                # Reset
                current_question_text = []
                current_question_seg_id = None
                current_question_speaker = ""
                current_answer_text = []
                current_answer_seg_id = None
                current_answer_speaker = ""
                state = "WAITING_FOR_QUESTION"
            continue

        if state == "WAITING_FOR_QUESTION":
            if _is_question_segment(seg):
                current_question_text = [text] if text else []
                current_question_seg_id = seg_id
                current_question_speaker = speaker
                current_answer_text = []
                current_answer_seg_id = None
                current_answer_speaker = ""
                state = "IN_QUESTION"

        elif state == "IN_QUESTION":
            if _is_question_segment(seg):
                # Another analyst segment — append to current question
                if text:
                    current_question_text.append(text)
            elif _is_answer_segment(seg):
                # Transition to answer
                if text:
                    current_answer_text = [text]
                    current_answer_seg_id = seg_id
                    current_answer_speaker = speaker
                state = "IN_ANSWER"

        elif state == "IN_ANSWER":
            if _is_answer_segment(seg):
                # Another executive segment — append to current answer
                if text:
                    current_answer_text.append(text)
            elif _is_question_segment(seg):
                # New question: flush current pair first
                pair = _flush_pair()
                if pair:
                    qa_pairs.append(pair)
                # Start new question
                current_question_text = [text] if text else []
                current_question_seg_id = seg_id
                current_question_speaker = speaker
                current_answer_text = []
                current_answer_seg_id = None
                current_answer_speaker = ""
                state = "IN_QUESTION"

    # Flush final pair if in progress
    if state == "IN_ANSWER" and current_question_text and current_answer_text:
        pair = _flush_pair()
        if pair:
            qa_pairs.append(pair)

    logger.debug(f"[{transcript_id}] Matched {len(qa_pairs)} Q&A pairs.")
    return qa_pairs


def extract_qa_pairs(transcript_id: str, segments: List[Dict]) -> List[Dict]:
    """
    Convenience alias for match_qa_pairs.
    """
    return match_qa_pairs(transcript_id, segments)
