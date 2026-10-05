"""
Preprocessing Module — Phase 1

Normalizes raw earnings-call transcript text while preserving:
  - Financial figures: $2 billion, 50 bps, 8-10%, Q4, FY2025
  - Speaker attribution
  - Transcript ordering
  - Section labels
  - Timestamps (if present)

What IS cleaned:
  - Encoding artifacts (smart quotes, non-breaking spaces, etc.)
  - Excessive whitespace / blank lines
  - HTML/XML tags that leak in from web scraping
  - Repeated header/footer patterns
  - CR/LF normalization

What is NOT changed:
  - Sentence structure
  - Numbers and financial expressions
  - Proper nouns and tickers
  - Speaker names
"""

import html
import re
import unicodedata
from typing import Optional

from src.logger import get_logger


logger = get_logger("preprocessing")

# ──────────────────────────────────────────────────────────────────────────────
# Compiled patterns (module-level for efficiency)
# ──────────────────────────────────────────────────────────────────────────────

# HTML/XML tags that sometimes appear in scraped transcripts
_HTML_TAG = re.compile(r"<[^>]{1,200}>", re.IGNORECASE)

# Runs of 3+ blank lines → 2 blank lines
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

# Trailing whitespace on each line
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)

# Leading whitespace on each line (keep single indent; collapse deeper ones)
_DEEP_INDENT = re.compile(r"^[ \t]{4,}", re.MULTILINE)

# Repeated dashes / underscores used as decorative separators
_DECORATIVE_SEP = re.compile(r"[-_=]{5,}")

# Page-break markers that leak from PDF extraction
_PAGE_BREAK = re.compile(r"\f|\[PAGE\s*\d*\]|\(continued\)", re.IGNORECASE)

# Repeated disclaimer headers that some sources prepend on every page
_REPEATED_DISCLAIMER = re.compile(
    r"(?:This transcript (?:has been|was) (?:edited|prepared|produced)|"
    r"Operator instructions:|Questions? and Answers?:)\s*\n",
    re.IGNORECASE,
)

# Normalize typographic (smart) quotes to ASCII
_SMART_QUOTE_MAP = str.maketrans({
    "\u2018": "'",   # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",   # RIGHT SINGLE QUOTATION MARK
    "\u201c": '"',   # LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',   # RIGHT DOUBLE QUOTATION MARK
    "\u2013": "-",   # EN DASH
    "\u2014": "--",  # EM DASH
    "\u00a0": " ",   # NO-BREAK SPACE
    "\u200b": "",    # ZERO WIDTH SPACE
    "\u200c": "",    # ZERO WIDTH NON-JOINER
    "\u200d": "",    # ZERO WIDTH JOINER
    "\ufeff": "",    # BOM / ZERO WIDTH NO-BREAK SPACE
})

# Fraction of non-ASCII characters that triggers a "likely encoding garbage" warning
_GARBAGE_THRESHOLD = 0.15


def normalize_encoding(text: str) -> str:
    """
    Fix encoding artifacts while preserving all ASCII and financial text.

    Applies NFC Unicode normalization so composed characters are consistent,
    then substitutes typographic punctuation with ASCII equivalents.
    """
    if not text:
        return text
    # NFC normalization: e.g. é stored as e + combining accent → single code point
    text = unicodedata.normalize("NFC", text)
    # Smart quotes, dashes, zero-width chars
    text = text.translate(_SMART_QUOTE_MAP)
    return text


def _check_encoding_quality(text: str, transcript_id: str = "") -> None:
    """Warn if the text contains an unusual proportion of non-ASCII characters."""
    if not text:
        return
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ratio = non_ascii / len(text)
    if ratio > _GARBAGE_THRESHOLD:
        logger.warning(
            f"[{transcript_id}] High non-ASCII ratio {ratio:.2%} — "
            "possible encoding issue in source data."
        )


def clean_text(text: str, transcript_id: str = "") -> str:
    """
    Core cleaning function for a block of raw transcript text.

    Parameters
    ----------
    text          : raw text from dataset
    transcript_id : used only for log messages

    Returns
    -------
    Cleaned text string.  Never returns None.
    """
    if not text:
        return ""

    # Step 1: encoding normalization
    text = normalize_encoding(text)

    # Step 2: unescape HTML entities (&nbsp;, &amp;, etc.)
    text = html.unescape(text)

    # Optionally warn about quality
    _check_encoding_quality(text, transcript_id)

    # Step 3: strip HTML/XML tags
    text = _HTML_TAG.sub(" ", text)


    # Step 3: normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Step 4: remove page-break markers
    text = _PAGE_BREAK.sub("\n", text)

    # Step 5: remove decorative separators
    text = _DECORATIVE_SEP.sub("", text)

    # Step 6: remove trailing whitespace per line
    text = _TRAILING_WS.sub("", text)

    # Step 7: collapse deep indentation to a single space
    text = _DEEP_INDENT.sub(" ", text)

    # Step 8: collapse excess blank lines
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)

    # Step 9: strip leading/trailing whitespace from the whole block
    text = text.strip()

    return text


def normalize_section_label(label: str) -> str:
    """
    Map raw section strings to one of the canonical labels.

    Canonical labels: prepared_remarks | q_and_a | opening | closing | unknown

    Parameters
    ----------
    label : raw section label (e.g., from structured_content)

    Returns
    -------
    One of the canonical section label strings.
    """
    if not label:
        return "unknown"
    norm = label.strip().lower().replace("-", "_").replace(" ", "_")

    if norm in ("prepared_remarks", "prepared", "remarks", "management_discussion"):
        return "prepared_remarks"
    if norm in ("q_and_a", "q&a", "qa", "questions_and_answers",
                "questions_answers", "question_and_answer"):
        return "q_and_a"
    if norm in ("opening", "introduction", "intro", "welcome", "operator_intro"):
        return "opening"
    if norm in ("closing", "conclusion", "outro", "wrap_up", "wrap"):
        return "closing"
    return "unknown"


def normalize_speaker_role(role: str) -> str:
    """
    Map raw speaker-role strings to one of the canonical role labels.

    Canonical roles: executive | analyst | operator | unknown
    """
    if not role:
        return "unknown"
    norm = role.strip().lower()

    if norm in ("executive", "management", "ceo", "cfo", "coo", "chairman",
                "president", "evp", "svp", "vp"):
        return "executive"
    if norm in ("analyst", "questioner", "question"):
        return "analyst"
    if norm in ("operator", "moderator", "facilitator"):
        return "operator"
    return "unknown"


def preprocess_transcript_record(record: dict, transcript_id: str = "") -> dict:
    """
    Apply all preprocessing steps to a raw transcript record dict.

    Modifies in-place and also returns the dict.

    Expected input keys (HuggingFace dataset format):
      symbol, company_name, company_id, year, quarter, date,
      content, structured_content

    Returns the record with cleaned fields.
    """
    # Clean raw content if present
    if "content" in record and record["content"]:
        record["content"] = clean_text(record["content"], transcript_id)

    # Clean structured_content text blocks
    if "structured_content" in record and isinstance(record["structured_content"], list):
        for seg in record["structured_content"]:
            if isinstance(seg, dict):
                if "text" in seg and seg["text"]:
                    seg["text"] = clean_text(seg["text"], transcript_id)
                # Normalize section and role labels from the provider's raw values
                if "section" in seg:
                    seg["section"] = normalize_section_label(seg.get("section", ""))
                if "role" in seg:
                    seg["role"] = normalize_speaker_role(seg.get("role", ""))

    return record
