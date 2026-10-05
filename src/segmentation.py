"""
Segmentation Module — Phase 1

Handles:
  A. Speaker / Section segmentation from structured_content (preferred path)
  B. Heuristic segmentation from raw text (fallback)
  C. Sentence segmentation within each segment

Section labels produced:
  prepared_remarks | q_and_a | opening | closing | unknown

Speaker roles produced:
  executive | analyst | operator | unknown

Each segment gets a deterministic segment_id:
  <transcript_id>_seg_<zero-padded-sequence>

Each sentence gets a deterministic sentence_id:
  <segment_id>_sent_<zero-padded-number>

Memory strategy:
  - Everything is yielded as generators / lists per transcript
  - Nothing is accumulated across transcripts
"""

from __future__ import annotations

import re
import uuid
from typing import Dict, Generator, Iterable, List, Optional, Tuple

from src.logger import get_logger
from src.preprocessing import normalize_section_label, normalize_speaker_role

logger = get_logger("segmentation")


# ──────────────────────────────────────────────────────────────────────────────
# Sentence splitter (deterministic, financial-text-aware)
# ──────────────────────────────────────────────────────────────────────────────

# Common abbreviations that should NOT end a sentence
_ABBREVS = frozenset([
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "al",
    "e.g", "i.e", "cf", "fig", "no",
    # Financial abbreviations
    "approx", "est", "inc", "corp", "co", "ltd", "llc", "plc",
    "q1", "q2", "q3", "q4", "h1", "h2",
    "fy", "ytd", "qoq", "yoy", "bps", "eps", "ebitda", "ebit",
    "u.s", "u.k", "e.u",
])

# Candidate sentence boundary: punctuation followed by whitespace and uppercase/quote
_CANDIDATE_BOUNDARY_RE = re.compile(r'([.!?]+)(\s+)(?=[A-Z\"])')


# Q-like indicators in raw text to detect section boundaries
_QA_SECTION_RE = re.compile(
    r"^(?:question[s]?\s*(?:&|and)\s*answers?|q\s*&\s*a|"
    r"analyst\s+q(?:uestion)?|questions?\s*from\s*analysts?)\s*[:\-]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_PREPARED_SECTION_RE = re.compile(
    r"^(?:prepared\s+remarks?|management\s+(?:discussion|remarks?|presentation)|"
    r"opening\s+(?:remarks?|statement))\s*[:\-]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Speaker label pattern: matches "Tim Cook:" or "OPERATOR:" at line start
_SPEAKER_LABEL_RE = re.compile(
    r"(?:^|\n)\s*([A-Z][A-Za-z0-9 ,.\-']{1,50}):\s*",
)


# Operator cues for Q&A section detection
_OPERATOR_QA_CUE = re.compile(
    r"\b(?:we will now (?:begin|open)|open (?:the )?(?:floor|call) for questions?|"
    r"our first question|next question|we have time for|"
    r"caller.*please (?:go ahead|proceed)|"
    r"please (?:state|provide) your name)\b",
    re.IGNORECASE,
)

# Common executive title patterns for speaker-role detection
_EXEC_TITLE_RE = re.compile(
    r"\b(?:chief\s+(?:executive|financial|operating|technology|marketing)|"
    r"president|chairman|founder|co-founder|director|ceo|cfo|coo|cto|cmo|evp|svp|vp)\b",
    re.IGNORECASE,
)

_ANALYST_TITLE_RE = re.compile(
    r"\b(?:analyst|research\s+analyst|equity\s+analyst|"
    r"(?:head|director)\s+of\s+research)\b",
    re.IGNORECASE,
)


def split_sentences(text: str) -> List[str]:
    """
    Split a block of text into sentences with financial-awareness.
    Preserves abbreviations (U.S., Inc., Mr., Q4., approx.) and numbers ($5.2 billion).

    Returns a list of non-empty sentence strings.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    sentences = []
    start = 0

    for match in _CANDIDATE_BOUNDARY_RE.finditer(text):
        punct = match.group(1)
        punct_end = match.start() + len(punct)
        split_point = match.end()

        # Extract preceding word to test against abbreviations
        preceding_text = text[start:match.start()].strip()
        last_word_match = re.search(r'([A-Za-z0-9\.\-]+)$', preceding_text)
        if last_word_match:
            last_word = last_word_match.group(1).lower().rstrip('.')
            # If preceding word is a known abbreviation (mr, inc, u.s, etc.)
            if last_word in _ABBREVS:
                continue
            # Single capital initial like "John D. Rockefeller"
            if len(last_word) == 1 and last_word.isalpha():
                continue
            # Digit immediately preceding period (e.g. "3.")
            if last_word and last_word[-1].isdigit():
                continue

        # Found valid boundary
        sent = text[start:punct_end].strip()
        if sent:
            sentences.append(sent)
        start = split_point

    # Append remaining trailing sentence
    trailing = text[start:].strip()
    if trailing:
        sentences.append(trailing)

    return sentences if sentences else [text]



def _infer_speaker_role(speaker: str, role: Optional[str] = None) -> str:
    """
    Infer speaker role from provided role string or from speaker name heuristics.

    Returns one of: executive | analyst | operator | unknown
    """
    if role:
        canonical = normalize_speaker_role(role)
        if canonical != "unknown":
            return canonical

    if not speaker:
        return "unknown"

    sn = speaker.strip().lower()

    if sn in ("operator", "moderator", "facilitator"):
        return "operator"

    if _EXEC_TITLE_RE.search(sn):
        return "executive"

    if _ANALYST_TITLE_RE.search(sn):
        return "analyst"

    # Heuristic: unrecognized speaker names in Q&A are often analysts
    # Caller that says "question from …" → likely analyst introduced by operator
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Structured-content segmentation (primary path)
# ──────────────────────────────────────────────────────────────────────────────

def segments_from_structured_content(
    transcript_id: str,
    structured_content: List[Dict],
    ticker: str = "",
    date: str = "",
    year: Optional[int] = None,
    quarter: str = "",
) -> List[Dict]:
    """
    Build segments from the HuggingFace structured_content list.

    Each element of structured_content is expected to have at minimum:
      speaker (str), text (str)
    Optionally:
      role (str), section (str)

    The section is tracked across the sequence: once an 'opening' is seen,
    it remains until the pattern changes; once Q&A is detected it is
    propagated forward.

    Returns
    -------
    List of segment dicts ready for DB insertion.
    """
    segments: List[Dict] = []
    current_section = "unknown"

    # Try to detect the Q&A section from the sequence order
    # (some datasets label it explicitly; others do not)
    has_explicit_sections = any(
        seg.get("section") and seg["section"] not in ("", "unknown")
        for seg in structured_content
    )

    for seq_idx, raw_seg in enumerate(structured_content):
        if not isinstance(raw_seg, dict):
            continue

        text = (raw_seg.get("text") or "").strip()
        if not text:
            continue

        speaker = (raw_seg.get("speaker") or "Unknown").strip()
        raw_role = raw_seg.get("role", "")
        raw_section = raw_seg.get("section", "")

        # Determine section
        if has_explicit_sections and raw_section:
            section = normalize_section_label(raw_section)
        else:
            # Heuristic progression
            section = _detect_section_heuristic(speaker, text, current_section)

        current_section = section

        # Determine role
        speaker_role = _infer_speaker_role(speaker, raw_role)

        # Deterministic segment ID
        segment_id = f"{transcript_id}_seg_{seq_idx:04d}"

        segments.append({
            "segment_id": segment_id,
            "transcript_id": transcript_id,
            "speaker": speaker,
            "speaker_role": speaker_role,
            "section": section,
            "text": text,
            "sequence": seq_idx,
        })

    return segments


def _detect_section_heuristic(
    speaker: str,
    text: str,
    current_section: str,
) -> str:
    """
    Heuristic section detection when no explicit label is available.

    Rules (in priority order):
      1. Operator opening → 'opening'
      2. Operator Q&A cue → 'q_and_a' (and stays q_and_a)
      3. Operator closing → 'closing'
      4. Any section already detected as q_and_a → stays q_and_a
      5. Speaker is operator and no Q&A yet → 'opening'
      6. Default for early segments → 'prepared_remarks'
    """
    sn = speaker.strip().lower()
    is_operator = sn in ("operator", "moderator")

    if is_operator:
        if _OPERATOR_QA_CUE.search(text):
            return "q_and_a"
        if "welcome" in text.lower() or "good morning" in text.lower():
            return "opening"
        if "thank you" in text.lower() and current_section in ("q_and_a",):
            return "closing"
        # If already in q_and_a, operator bridging lines stay in q_and_a
        if current_section == "q_and_a":
            return "q_and_a"
        return current_section if current_section != "unknown" else "opening"

    # Non-operator: if q_and_a phase started, stay there
    if current_section == "q_and_a":
        return "q_and_a"

    if current_section in ("opening",):
        return "prepared_remarks"

    return current_section if current_section not in ("unknown", "") else "prepared_remarks"


# ──────────────────────────────────────────────────────────────────────────────
# Raw-text segmentation (fallback)
# ──────────────────────────────────────────────────────────────────────────────

def segments_from_raw_text(
    transcript_id: str,
    raw_text: str,
) -> List[Dict]:
    """
    Extract speaker turns from an unstructured raw text blob.
    Supports "Speaker Name: text..." or "Speaker Name:\ntext...".

    If no speaker labels are found, the whole text is returned as one segment.
    """
    if not raw_text or not raw_text.strip():
        return []

    matches = list(_SPEAKER_LABEL_RE.finditer(raw_text))

    if not matches:
        seg_id = f"{transcript_id}_seg_0000"
        return [{
            "segment_id": seg_id,
            "transcript_id": transcript_id,
            "speaker": "Unknown",
            "speaker_role": "unknown",
            "section": "unknown",
            "text": raw_text.strip(),
            "sequence": 0,
        }]

    segments: List[Dict] = []
    current_section = "unknown"
    seq_idx = 0

    for idx, match in enumerate(matches):
        speaker = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if (idx + 1) < len(matches) else len(raw_text)
        text = raw_text[start:end].strip()
        if not text:
            continue

        speaker_role = _infer_speaker_role(speaker)
        section = _detect_section_heuristic(speaker, text, current_section)
        current_section = section

        seg_id = f"{transcript_id}_seg_{seq_idx:04d}"
        segments.append({
            "segment_id": seg_id,
            "transcript_id": transcript_id,
            "speaker": speaker,
            "speaker_role": speaker_role,
            "section": section,
            "text": text,
            "sequence": seq_idx,
        })
        seq_idx += 1

    return segments



# ──────────────────────────────────────────────────────────────────────────────
# Sentence segmentation
# ──────────────────────────────────────────────────────────────────────────────

def sentences_from_segment(segment: Dict) -> List[Dict]:
    """
    Split a segment into sentence-level records.

    Each sentence record contains:
      sentence_id, segment_id, transcript_id, sentence_number, text

    Returns
    -------
    List of sentence dicts.
    """
    text = segment.get("text", "")
    if not text:
        return []

    raw_sentences = split_sentences(text)
    result: List[Dict] = []

    for idx, sent_text in enumerate(raw_sentences):
        if not sent_text.strip():
            continue
        sent_id = f"{segment['segment_id']}_sent_{idx:04d}"
        result.append({
            "sentence_id": sent_id,
            "segment_id": segment["segment_id"],
            "transcript_id": segment["transcript_id"],
            "sentence_number": idx,
            "text": sent_text.strip(),
        })

    return result


def segment_transcript(
    transcript_id: str,
    record: Dict,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Primary entry point: produce (segments, sentences) for one transcript.

    Uses structured_content if available; falls back to raw text.

    Parameters
    ----------
    transcript_id : canonical transcript ID
    record        : preprocessed transcript record dict

    Returns
    -------
    (segments, sentences) — two flat lists of dicts ready for DB insertion.
    """
    structured = record.get("structured_content")
    raw_text = record.get("content", "")

    if structured and isinstance(structured, list) and len(structured) > 0:
        segments = segments_from_structured_content(
            transcript_id=transcript_id,
            structured_content=structured,
        )
        if not segments:
            logger.warning(
                f"[{transcript_id}] structured_content yielded no segments; "
                "falling back to raw text."
            )
            segments = segments_from_raw_text(transcript_id, raw_text)
    else:
        logger.debug(f"[{transcript_id}] No structured_content; using raw text fallback.")
        segments = segments_from_raw_text(transcript_id, raw_text)

    # Generate sentences for each segment
    all_sentences: List[Dict] = []
    for seg in segments:
        sents = sentences_from_segment(seg)
        all_sentences.extend(sents)

    logger.debug(
        f"[{transcript_id}] Segmented: {len(segments)} segments, "
        f"{len(all_sentences)} sentences."
    )

    return segments, all_sentences
