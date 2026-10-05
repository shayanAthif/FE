"""Tests for Phase 1 Data Pipeline: Ingestion, Preprocessing, Segmentation, QA Matching, and DB Persistence."""

import json
from pathlib import Path
import pytest

from src.checkpoint import CheckpointManager
from src.data_loader import (
    load_single_transcript,
    make_transcript_id,
    run_ingestion,
    _validate_record,
)
from src.database import DatabaseManager
from src.preprocessing import (
    clean_text,
    normalize_encoding,
    normalize_section_label,
    normalize_speaker_role,
    preprocess_transcript_record,
)
from src.qa_matcher import match_qa_pairs
from src.segmentation import (
    segment_transcript,
    segments_from_raw_text,
    segments_from_structured_content,
    split_sentences,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Preprocessing Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_preprocessing_preserves_financial_terms():
    """Ensure financial figures, abbreviations, and percentages remain intact."""
    raw = "Apple reported $90.8 billion in revenue, up 8-10% with margins rising 50 bps in Q4 FY2025."
    cleaned = clean_text(raw)
    assert "$90.8 billion" in cleaned
    assert "8-10%" in cleaned
    assert "50 bps" in cleaned
    assert "Q4" in cleaned
    assert "FY2025" in cleaned


def test_preprocessing_cleans_artifacts_and_html():
    """Ensure smart quotes, non-breaking spaces, HTML tags, and page breaks are cleaned."""
    raw = "<p>Welcome to the call.&nbsp;Here’s our ‘guidance’ for next quarter.</p>\f\n\n\n---"
    cleaned = clean_text(raw)
    assert "<p>" not in cleaned
    assert "</p>" not in cleaned
    assert "&nbsp;" not in cleaned
    assert "'" in cleaned or "guidance" in cleaned
    assert "\f" not in cleaned


def test_normalize_section_and_role_labels():
    """Test standard normalization mappings for sections and speaker roles."""
    assert normalize_section_label("prepared-remarks") == "prepared_remarks"
    assert normalize_section_label("Q&A") == "q_and_a"
    assert normalize_section_label("opening") == "opening"
    assert normalize_section_label("random_label") == "unknown"

    assert normalize_speaker_role("CEO") == "executive"
    assert normalize_speaker_role("Analyst") == "analyst"
    assert normalize_speaker_role("Operator") == "operator"
    assert normalize_speaker_role("Guest") == "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Segmentation Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_financial_sentence_segmentation():
    """Verify sentence splitting does not break on decimal numbers or financial abbreviations."""
    text = (
        "Good morning, Mr. Cook. We achieved revenue of $90.8 billion, i.e. 8.5% growth in the U.S. market. "
        "Our Q4 guidance is strong. We expect operating margins to expand further."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "Good morning, Mr. Cook."
    assert "$90.8 billion" in sentences[1]
    assert "8.5%" in sentences[1]
    assert "U.S. market" in sentences[1]
    assert sentences[2] == "Our Q4 guidance is strong."
    assert "operating margins" in sentences[3]



def test_segment_from_structured_content(sample_transcript_dict):
    """Test structured_content conversion into segments and sentences."""
    tid = make_transcript_id(sample_transcript_dict)
    segments, sentences = segment_transcript(tid, sample_transcript_dict)

    assert len(segments) == len(sample_transcript_dict["structured_content"])
    assert len(sentences) >= len(segments)

    # Check operator segment
    assert segments[0]["speaker_role"] == "operator"
    # Check executive segment
    assert segments[1]["speaker_role"] == "executive"
    assert segments[1]["section"] == "prepared_remarks"
    # Check analyst segment
    assert segments[2]["speaker_role"] == "analyst"
    assert segments[2]["section"] == "q_and_a"


def test_segment_from_raw_text_fallback():
    """Test heuristic segmentation when only raw text with speaker labels is present."""
    tid = "AAPL_2024_Q2_2024-05-02"
    record = {
        "content": (
            "Operator: Welcome everyone to today's call.\n\n"
            "Tim Cook: Thank you. Revenue reached $90 billion.\n\n"
            "Analyst: How is demand holding up?\n\n"
            "Tim Cook: Demand remains robust across geographies."
        )
    }
    segments, sentences = segment_transcript(tid, record)
    assert len(segments) >= 3
    speakers = [s["speaker"] for s in segments]
    assert "Tim Cook" in speakers
    assert len(sentences) >= 3


# ──────────────────────────────────────────────────────────────────────────────
# 3. Q&A Matching Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_qa_matching_flow(sample_transcript_dict):
    """Test question-answer matching between analysts and executives."""
    tid = make_transcript_id(sample_transcript_dict)
    segments, _ = segment_transcript(tid, sample_transcript_dict)
    qa_pairs = match_qa_pairs(tid, segments)

    assert len(qa_pairs) == 1
    pair = qa_pairs[0]
    assert pair["transcript_id"] == tid
    assert "China demand" in pair["question_text"]
    assert "too early to tell" in pair["answer_text"]
    assert pair["analyst_speaker"] == "Tami Zakaria"
    assert pair["executive_speaker"] == "Luca Maestri"


def test_qa_matching_multi_turn_and_operator_skipping():
    """Test Q&A matcher ignores operator bridges and correctly pairs questions and answers."""
    tid = "TEST_2024_Q1"
    segments = [
        {"segment_id": "seg_0", "section": "prepared_remarks", "speaker_role": "executive", "speaker": "CEO", "text": "Remarks.", "sequence": 0},
        {"segment_id": "seg_1", "section": "q_and_a", "speaker_role": "operator", "speaker": "Operator", "text": "First question from John.", "sequence": 1},
        {"segment_id": "seg_2", "section": "q_and_a", "speaker_role": "analyst", "speaker": "John", "text": "What is the margin outlook?", "sequence": 2},
        {"segment_id": "seg_3", "section": "q_and_a", "speaker_role": "executive", "speaker": "CFO", "text": "We anticipate 45% margins.", "sequence": 3},
        {"segment_id": "seg_4", "section": "q_and_a", "speaker_role": "executive", "speaker": "CEO", "text": "And we continue to invest.", "sequence": 4},
    ]
    pairs = match_qa_pairs(tid, segments)
    assert len(pairs) == 1
    assert pairs[0]["analyst_speaker"] == "John"
    assert pairs[0]["executive_speaker"] == "CFO"
    assert "45% margins" in pairs[0]["answer_text"]
    assert "continue to invest" in pairs[0]["answer_text"]


# ──────────────────────────────────────────────────────────────────────────────
# 4. Database Persistence and Checkpointing
# ──────────────────────────────────────────────────────────────────────────────

def test_load_single_transcript_persistence(db_manager: DatabaseManager, sample_transcript_dict):
    """Verify single transcript processes end-to-end into fresh SQLite tables."""
    result = load_single_transcript(sample_transcript_dict, db_manager)
    assert result["status"] == "ok"
    assert result["segments"] > 0
    assert result["sentences"] > 0
    assert result["qa_pairs"] > 0

    with db_manager.connection() as conn:
        t_row = conn.execute("SELECT * FROM transcripts WHERE transcript_id = ?", (result["transcript_id"],)).fetchone()
        assert t_row is not None
        assert t_row["ticker"] == "AAPL"

        seg_count = conn.execute("SELECT COUNT(*) FROM segments WHERE transcript_id = ?", (result["transcript_id"],)).fetchone()[0]
        assert seg_count == result["segments"]

        sent_count = conn.execute("SELECT COUNT(*) FROM sentences WHERE transcript_id = ?", (result["transcript_id"],)).fetchone()[0]
        assert sent_count == result["sentences"]

        qa_count = conn.execute("SELECT COUNT(*) FROM qa_pairs WHERE transcript_id = ?", (result["transcript_id"],)).fetchone()[0]
        assert qa_count == result["qa_pairs"]


def test_duplicate_transcript_skipping(db_manager: DatabaseManager, sample_transcript_dict):
    """Verify loading the same transcript without force skips processing."""
    res1 = load_single_transcript(sample_transcript_dict, db_manager)
    assert res1["status"] == "ok"

    processed = {res1["transcript_id"]}
    res2 = load_single_transcript(sample_transcript_dict, db_manager, already_processed=processed, force=False)
    assert res2["status"] == "skipped_duplicate"


def test_malformed_transcript_handling(db_manager: DatabaseManager):
    """Verify validation catches invalid records gracefully without raising crashes."""
    # Missing ticker/symbol
    res = load_single_transcript({"date": "2024-01-01", "content": "Hello"}, db_manager)
    assert "invalid" in res["status"]

    # Missing date
    res = load_single_transcript({"symbol": "MSFT", "content": "Hello"}, db_manager)
    assert "invalid" in res["status"]

    # Empty content
    res = load_single_transcript({"symbol": "MSFT", "date": "2024-01-01"}, db_manager)
    assert "invalid" in res["status"]


def test_checkpoint_resume_flow(db_manager: DatabaseManager, tmp_path: Path, sample_transcript_dict):
    """Verify ingestion pipeline records checkpoints and can resume."""
    fixture_file = tmp_path / "fixtures.jsonl"
    with open(fixture_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_transcript_dict) + "\n")

    stats = run_ingestion(source="jsonl", local_path=fixture_file, db_path=db_manager.db_path)
    assert stats["transcripts_processed"] == 1

    ckpt = CheckpointManager(db_manager)
    assert ckpt.is_job_completed("phase1_ingestion")

    # Second run without force should detect completed status
    stats2 = run_ingestion(source="jsonl", local_path=fixture_file, db_path=db_manager.db_path, force=False)
    assert stats2.get("status") == "already_completed"
