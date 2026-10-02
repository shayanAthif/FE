"""Sample data inspection for Phase 2 design."""
import sqlite3

conn = sqlite3.connect("database/hidden_risk.db")
conn.row_factory = sqlite3.Row

# Sample transcripts
rows = conn.execute("SELECT transcript_id, ticker, company_name, date, year, quarter, processing_status FROM transcripts LIMIT 5").fetchall()
print("=== Sample Transcripts ===")
for r in rows:
    print(dict(r))

# Sample segments
rows = conn.execute("SELECT segment_id, transcript_id, speaker, speaker_role, section, length(text) as textlen, sequence FROM segments LIMIT 5").fetchall()
print("\n=== Sample Segments ===")
for r in rows:
    print(dict(r))

# Sample sentences
rows = conn.execute("SELECT sentence_id, segment_id, transcript_id, sentence_number, text FROM sentences LIMIT 3").fetchall()
print("\n=== Sample Sentences ===")
for r in rows:
    d = dict(r)
    d['text'] = d['text'][:100]
    print(d)

# Sample QA pairs
rows = conn.execute("SELECT qa_id, transcript_id, analyst_speaker, executive_speaker, length(question_text) as qlen, length(answer_text) as alen FROM qa_pairs LIMIT 5").fetchall()
print("\n=== Sample QA Pairs ===")
for r in rows:
    print(dict(r))

# Section distribution
rows = conn.execute("SELECT section, speaker_role, COUNT(*) as cnt FROM segments GROUP BY section, speaker_role ORDER BY cnt DESC").fetchall()
print("\n=== Section/Role Distribution ===")
for r in rows:
    print(dict(r))

# Processing status
rows = conn.execute("SELECT processing_status, COUNT(*) FROM transcripts GROUP BY processing_status").fetchall()
print("\n=== Processing Status ===")
for r in rows:
    print(dict(r))

conn.close()

