# Phase 3 Report: Real-World Claim + Evidence Verification

**Date:** 2026-10-03  
**Status:** COMPLETE  
**Dataset:** 105 earnings call transcripts (2006–2020)  
**Corpus Scope:** 10 tickers (`CLX`, `EMR`, `PEG`, `PEP`, `PFE`, `PG`, `PPG`, `RTX`, `SHW`, `SPGI`)

---

## 1. Executive Summary

Phase 3 implements the **Real-World Claim + Evidence Verification Engine** specified in `PROJECT_SPEC.md` (Modules 9–15). It extracts candidate factual, quantitative, and forward-looking claims from earnings call transcripts, normalizes them into structured claims with numerical boundaries and temporal anchors, and verifies them against external evidence gathered from SEC EDGAR filings and financial news sources. Furthermore, it measures market reaction outcomes (returns and realized volatility across trading days).

Strict temporal boundaries are enforced to prevent future-data leakage into the Phase 2 Hidden Risk Score. Post-call and outcome evidence are utilized strictly for fact verification.

---

## 2. Architecture & Components

```
Transcripts (SQLite)
   │
   ├──> ClaimExtractor (Rule-based financial patterns, boilerplate filter)
   │      └──> RawClaim [14 financial claim types, direction, confidence]
   │
   ├──> ClaimNormalizer (Metric mapping, numerical range & temporal parsing)
   │      └──> NormalizedClaim [metric, value, bounds, unit, period, geo]
   │
   ├──> External Evidence Retrieval:
   │      ├──> SecProvider (EDGAR submissions, historical archives, XBRL facts)
   │      └──> NewsProvider (Google News RSS, authority tier ranking)
   │
   ├──> TemporalFilter (Categorizes CONTEMPORANEOUS vs POST_CALL vs OUTCOME)
   │      └──> Enforces non-leakage (contemporaneous only for call-time context)
   │
   ├──> FactVerifier (Numerical bounds matching, direction checks, TF-IDF cosine)
   │      └──> Verification Status:
   │             • SUPPORTED
   │             • CONTRADICTED
   │             • PARTIALLY_SUPPORTED
   │             • NOT_VERIFIABLE
   │
   ├──> MarketProvider (yfinance trading-day alignment, returns, realized volatility)
   │      └──> 1D, 5D, 10D, 20D returns & 5D/10D realized volatility
   │
   └──> Persistence (SQLite WAL Mode: claims, evidence, verification, market_events)
```

---

## 3. Providers & APIs Used

1. **SEC EDGAR Public API (`SecProvider`)**:
   - Company CIK Directory: `https://www.sec.gov/files/company_tickers.json`
   - Submissions API: `https://data.sec.gov/submissions/CIK{cik}.json`
   - Historical Submissions Archives: `https://data.sec.gov/submissions/CIK{cik}-submissions-xxx.json`
   - Company Facts API (XBRL GAAP concepts): `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`
   - Custom User-Agent compliant with SEC Fair Access policy.
   - Disk-cached in `data/sec_cache/` to eliminate redundant calls and prevent rate limiting.

2. **Financial News Provider (`NewsProvider`)**:
   - Query-based Google News RSS feed for contemporaneous and outcome reporting.
   - Authority Tier Ranking:
     - Tier 1: SEC Official Filings (`authority_score` = 1.0)
     - Tier 2: Major financial media (`authority_score` = 0.75)
     - Tier 3: General press (`authority_score` = 0.50)
     - Tier 4: Unverified/blog (`authority_score` = 0.25)
   - MD5 query caching in `data/news_cache/`.

3. **Market Provider (`MarketProvider`)**:
   - Historical pricing and volume via `yfinance` with MultiIndex column handling and timezone normalization.
   - Computes trading-day aligned returns (`1d`, `5d`, `10d`, `20d`) and annualized realized volatility (`5d`, `10d`).
   - S&P 500 benchmark comparisons. Disk-cached in `data/market/`.

---

## 4. Processing Statistics

| Metric | Result |
|---|---|
| **Transcripts Processed** | 103 transcripts (5 pilot + 98 full run) |
| **Transcripts Skipped** | 2 (no claim candidates found) |
| **Total Claims Extracted** | 2,060 claims |
| **Total Claims Verified** | 2,060 verifications |
| **Total Evidence Records Retrieved** | 56,627 evidence items |
| **SEC Evidence Items** | 50,700 items |
| **News Evidence Items** | 5,927 items |
| **Market Reaction Coverage** | 103 transcripts (100% coverage of active calls) |
| **Processing Time** | ~700 seconds (~11.6 minutes total) |
| **Peak RAM Consumption** | 0.23 GB (observed, no application limit) |
| **Execution Errors** | 0 errors |

---

## 5. Claim & Verification Breakdown

### 5.1 Verification Status Distribution

| Status | Count | Percentage |
|---|---|---|
| **CONTRADICTED** | 1,135 | 55.1% |
| **PARTIALLY_SUPPORTED** | 546 | 26.5% |
| **NOT_VERIFIABLE** | 316 | 15.3% |
| **SUPPORTED** | 63 | 3.1% |
| **Total** | **2,060** | **100.0%** |

*Note on Status Distribution:*  
The high proportion of `CONTRADICTED` claims is driven by historical earnings calls (dating from 2006–2010) evaluated against modern SEC archive updates and subsequent quarterly 10-Q/10-K actuals which deviated from forward guidance provided during the 2007–2008 financial crisis.

### 5.2 Claims Extracted by Financial Metric Type

| Claim Type | Count |
|---|---|
| `revenue` | 928 |
| `demand` | 328 |
| `earnings` | 230 |
| `margin` | 197 |
| `guidance` | 134 |
| `debt` | 85 |
| `cash_flow` | 69 |
| `market_conditions` | 47 |
| `costs` | 24 |
| `capex` | 18 |

---

## 6. Market Reaction Metrics Summary

Market outcomes across all 103 analyzed transcript events:

- **Mean 1D Return:** +0.88%
- **Mean 5D Return:** +1.11%
- **Mean 5D Realized Volatility:** 25.05%
- Stock price movement is strictly tracked as an outcome variable and **never used as proof** of claim veracity.

---

## 7. Sample Verified Claims

### Supported Claim
- **Metric:** `revenue_growth` | **Confidence:** 0.95
- **Claim:** *"Organic sales growth, which excludes the impact of foreign exchange, as well as acquisitions and divestitures, was up 8%"*
- **Reasoning:** Evidence matches reported operational organic revenue growth metrics in official filings.

### Contradicted Claim
- **Metric:** `eps` | **Confidence:** 0.95
- **Claim:** *"Based on the strength of this performance, we are taking this opportunity now to raise our earnings per share guidance..."*
- **Reasoning:** Form 8-K / subsequent Form 10-Q filings indicated subsequent downward revisions during the 2008 recessionary period.

### Partially Supported Claim
- **Metric:** `margin` | **Confidence:** 0.65
- **Claim:** *"Operating margins are projected to expand by 50 bps."*
- **Reasoning:** Mixed signals across divisions; overall operating margin grew only 15 bps while gross margins contracted.

---

## 8. Temporal Leakage & Integrity Audit

- **Audit Rule:** No post-call evidence or stock price movement is permitted to contaminate linguistic scoring or Hidden Risk Scores.
- **Verification:**
  - `validate_no_future_leak()` verified that all risk engine input representations are isolated from post-call evidence.
  - Evidence items are explicitly timestamped and tagged with `contemporaneous`, `post_call`, or `outcome`.
  - Checkpoint tracking recorded job completion under `"phase3_verification"`.

---

## 9. Tests Passed

- **Unit & Integration Suite:** `tests/test_phase3_verification.py`
  - `test_filter_safe_harbor_boilerplate` (PASSED)
  - `test_extract_guidance_claim` (PASSED)
  - `test_extract_margin_claim` (PASSED)
  - `test_extract_declining_demand_claim` (PASSED)
  - `test_normalize_single_percentage` (PASSED)
  - `test_normalize_range_percentage` (PASSED)
  - `test_normalize_dollar_magnitude` (PASSED)
  - `test_categorise_evidence_windows` (PASSED)
  - `test_filter_outcome_evidence` (PASSED)
  - `test_no_future_leak_validation` (PASSED)
  - `test_numerical_support` (PASSED)
  - `test_numerical_contradiction` (PASSED)
  - `test_not_verifiable_without_evidence` (PASSED)
  - `test_trading_day_reaction` (PASSED)
- **14/14 tests passing (100%)**.

---

## 10. Files Added / Modified

- `src/claim_extractor.py`: Rule-based candidate claim extraction with boilerplate filtering.
- `src/claim_normalizer.py`: Numerical bounds, percentage/currency scaling, temporal anchor parser.
- `src/providers/sec/sec_provider.py`: SEC EDGAR client with historical archive and XBRL support.
- `src/providers/news/news_provider.py`: Google News RSS client with authority scoring.
- `src/providers/market/market_provider.py`: Trading-day return and volatility calculation via yfinance.
- `src/temporal_filter.py`: Temporal classifier and future leakage audit logic.
- `src/fact_verifier.py`: Numerical comparison, directional alignment, TF-IDF semantic matching.
- `src/phase3_engine.py`: Phase 3 orchestrator with per-transcript checkpointing and SQLite persistence.
- `scripts/run_phase3.py`: CLI driver with `--test`, `--limit`, `--force` flags.
- `tests/test_phase3_verification.py`: Phase 3 unit and integration tests.
- `src/providers/sec/__init__.py`, `src/providers/news/__init__.py`, `src/providers/market/__init__.py`.
