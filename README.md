# FE

## AI-Powered "Hidden Risk" Earnings Call Analyzer
### with Real-World Financial and News Fact Verification

Production-quality research prototype analyzing S&P 500 earnings-call transcripts to detect potentially concealed financial risk signals in executive language.

### Core Signals
1. **Hedging / Uncertainty**: Lexical patterns, modal verbs, qualifier density, and forward-looking caveats.
2. **Evasiveness / Topic Avoidance**: Semantic similarity between questions and answers, deflection indicators, and non-numeric responses to quantitative inquiries.
3. **Tone Shifts**: Abrupt local sentiment shifts (using ProsusAI/FinBERT) and prepared remarks vs. Q&A divergence.

### Implementation Phases
- **Phase 0**: Architecture, Configuration, Database Schema, and Testing Infrastructure.
- **Phase 1**: Ingestion & Preprocessing Pipeline (Transcripts parsing, sentence segmentation, speaker role detection, and Q&A pairing).
- **Phase 2**: Linguistic Risk Engine (FinBERT inference, tone shift, evasiveness, hedging detectors, and multi-level Hidden Risk Score aggregation).
- **Phase 3**: Real-World Verification (SEC EDGAR, market outcome correlation, and news retrieval).
- **Phase 4**: Interactive Streamlit Dashboard.
- **Phase 5**: Validation & Empirical Analytics.
- **Phase 6**: Scale Testing (2,000 Transcripts).
- **Phase 7**: Full S&P 500 Dataset Execution.

## Quickstart

```powershell
# Local fixture-based pipeline run
python scripts/run_pipeline.py --stage all --source jsonl --local-path data/sample_transcripts.jsonl --force

# Phase 5 validation
python scripts/run_phase5.py --limit 20 --skip-annotation

# Launch the dashboard
streamlit run dashboard/app.py

# Benchmark the pipeline at scale
python scripts/benchmark_pipeline.py --transcripts 10 --source jsonl --local-path data/sample_transcripts.jsonl
```

## Current Status

The project is validated through Phase 5 and is ready for Phase 6 scale testing once the user confirms the next checkpoint. The codebase includes:
- deterministic fixture ingest for local validation
- full DB-backed pipeline stages
- dashboard explorer/filter/export flows
- validation/report generation
- benchmark harness for timing and memory capture

## Notes

- The sample JSONL fixture in `data/sample_transcripts.jsonl` is intended for offline validation and smoke tests.
- Real dataset execution should continue with checkpoints and memory monitoring to stay under the configured RAM budget.

## Project documents

- Handoff checklist: [TODO.md](TODO.md)
- Phase reports: [docs/reports/](docs/reports/)

## End-to-end server handoff

From the repository root, install dependencies and initialize the database:

```powershell
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --stage all --limit 10
```

Run the scale gate before processing the full source:

```powershell
python scripts/run_phase6.py --limit 2000 --batch-size 4
```

Review `outputs/phase6_scale_report.json`. Continue only if the return code,
resume check, and RAM budget checks pass. Then run the full available dataset:

```powershell
python scripts/run_phase7.py --batch-size 4
```

Phase 7 streams the configured Hugging Face dataset and resumes from the
existing SQLite checkpoints after interruption. The generated report is
`outputs/phase7_full_dataset_report.json`. Start the dashboard after processing:

```powershell
streamlit run dashboard\app.py
```

Use `--source jsonl --local-path <file>` for a local JSONL dataset. Keep API
credentials in `.env`; never commit that file or provider keys.
