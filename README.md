# AI-Powered Hidden Risk Earnings Call Analyzer

An interpretable research tool for analyzing earnings-call transcripts and
identifying language patterns that may indicate elevated financial risk.

The system combines linguistic analysis with independent evidence and market
outcome data. It does not label executives as deceptive, provide trading
recommendations, or treat a stock-price movement as proof that a statement was
false.

## What it analyzes

- **Hedging and uncertainty** — modal verbs, qualifiers, and forward-looking
  caveats.
- **Evasiveness and topic avoidance** — question-answer relevance, deflection
  patterns, and non-numeric responses to quantitative questions.
- **Tone shifts** — local sentiment changes and differences between prepared
  remarks and Q&A responses.
- **Claim verification** — comparisons with SEC filings, news, and market data.
- **Market outcomes** — post-call returns, volatility, and benchmark-relative
  performance.

Every result is designed to remain traceable to the source transcript,
sentence, extracted claim, evidence item, and contributing risk features.

## Project structure

- `src/` — ingestion, preprocessing, scoring, verification, and validation
  logic
- `dashboard/` — Streamlit analyst interface
- `scripts/` — command-line tools for processing, validation, and benchmarking
- `database/` — SQLite database containing processed results
- `outputs/` — generated analysis and validation outputs
- `docs/reports/` — project reports and technical documentation
- `tests/` — automated tests

## Setup

From the repository root:

```powershell
python -m pip install -r requirements.txt
```

For external evidence providers, copy `.env.example` to `.env` and configure
the relevant credentials. Keep `.env` private and do not commit provider keys.

## Running the dashboard

The dashboard reads processed results from the SQLite database:

```powershell
streamlit run dashboard\app.py
```

Open `http://localhost:8501` in a browser. The interface includes:

- Overview metrics
- Transcript exploration
- Sentence-level risk explanations
- Claim verification and evidence
- Market reactions and outcomes
- Company comparison

## Running the pipeline

The default data source is the configured Hugging Face dataset. The complete
dataset is loaded into RAM before records are processed, then results are
persisted incrementally so that work can resume after an interruption:

```powershell
python scripts/run_pipeline.py --stage all
```

For an offline smoke test using the included sample data:

```powershell
python scripts/run_pipeline.py --stage all `
  --source jsonl `
  --local-path data/sample_transcripts.jsonl `
  --limit 10
```

Useful commands:

```powershell
python scripts/run_phase5.py --skip-annotation
python scripts/benchmark_pipeline.py --transcripts 10 `
  --source jsonl `
  --local-path data/sample_transcripts.jsonl
python -m pytest -q
```

## Data and research notes

- The system is a research prototype, not financial advice.
- Linguistic risk, factual verification, and market outcomes are separate
  signals and should not be conflated.
- Statistical relationships are observational and do not establish causation.
- Validation results based on synthetic labels should not be interpreted as
  human-annotated ground truth.
- External data availability, API limits, and cache state can affect results.

## Documentation

- Project checklist: [TODO.md](TODO.md)
- Reports: [docs/reports/](docs/reports/)
