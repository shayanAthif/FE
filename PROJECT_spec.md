PROJECT TITLE

AI-Powered “Hidden Risk” Earnings Call Analyzer
with Real-World Financial and News Fact Verification


============================================================
1. PROJECT OVERVIEW
============================================================

Build a complete production-quality research prototype called:

“AI-Powered Hidden Risk Earnings Call Analyzer”

The system analyzes S&P 500 earnings-call transcripts to detect potentially concealed financial-risk signals hidden in executive language.

The system must NOT simply perform ordinary positive/negative sentiment analysis.

Instead, it must identify three major linguistic signals:

1. Hedging / uncertainty
2. Evasiveness / topic avoidance
3. Tone shifts

These signals must be analyzed at sentence and segment level and aggregated into an interpretable:

HIDDEN RISK SCORE

The system must also contain a second major layer:

REAL-WORLD FACT AND OUTCOME VERIFICATION

For important claims made during earnings calls, the system should retrieve and compare evidence from:

1. SEC filings / official financial disclosures
2. Real stock-market data
3. News articles / news databases

The purpose is to determine whether claims made by management were later:

- Supported
- Contradicted
- Partially supported
- Not verifiable

The system should also analyze what happened in the stock market after the earnings call and determine whether high linguistic risk scores are associated with subsequent volatility or other measurable market outcomes.

This is a financial NLP + information retrieval + fact verification + market analytics + dashboard project.

The final system must be interpretable.

Every risk score and every verification result should be traceable back to:

- the original transcript sentence
- the extracted claim
- the evidence source
- the reasoning/features that produced the result

============================================================
IMPLEMENTATION ROADMAP — FOLLOW THIS ORDER
============================================================

IMPORTANT:

This project MUST be implemented incrementally.

Do NOT attempt to build the entire system in one operation.

The implementation consists of the following phases:

PHASE 0 — ARCHITECTURE & SETUP
- Analyze requirements
- Create project structure
- Create configuration system
- Create SQLite schema
- Create logging
- Create checkpoint/resume system
- Create provider interfaces
- Create tests
- Do NOT process the full dataset

PHASE 1 — DATA PIPELINE
Target: 100 transcripts

Implement:
- Hugging Face dataset ingestion
- streaming/chunked loading
- preprocessing
- speaker identification
- speaker-role mapping
- section detection
- sentence segmentation
- Q&A pairing
- SQLite persistence
- checkpointing

Acceptance:
100 transcripts can be processed and resumed successfully.

PHASE 2 — LINGUISTIC RISK ENGINE
Target: same 100 transcripts

Implement:
- hedging detector
- FinBERT sentiment
- tone-shift detector
- evasiveness detector
- sentence-level scoring
- segment-level scoring
- Q&A scoring
- transcript-level Hidden Risk Score

Acceptance:
Every processed transcript has explainable risk scores.

PHASE 3 — REAL-WORLD VERIFICATION
Target: same 100 transcripts

Implement:
- claim extraction
- claim normalization
- SEC evidence retrieval
- market-data retrieval
- news retrieval
- evidence ranking
- temporal filtering
- fact verification

Verification statuses:
SUPPORTED
CONTRADICTED
PARTIALLY_SUPPORTED
NOT_VERIFIABLE

Acceptance:
Claims can be traced from transcript -> evidence -> verification result.

PHASE 4 — DASHBOARD
Target: same 100 transcripts

Implement:
- Streamlit dashboard
- transcript explorer
- Hidden Risk Score visualization
- flagged statements
- claim verification
- evidence viewer
- market reaction
- event timeline

Acceptance:
A user can select a company/call and inspect the entire analysis.

PHASE 5 — VALIDATION
Target: 500 transcripts

Implement:
- human annotation subset
- precision/recall/F1
- correlation analysis
- volatility analysis
- guidance/outcome analysis
- baseline comparisons

Acceptance:
Generate a research evaluation report.

PHASE 6 — SCALE TEST
Target: 2,000 transcripts

Measure:
- RAM usage
- CPU usage
- GPU usage
- processing time
- database size
- API usage
- cache effectiveness
- failure/recovery behavior

Acceptance:
Remain within the 12 GB RAM target and resume correctly after interruption.

PHASE 7 — FULL DATASET
Target:
All available transcripts

Only begin this phase after Phases 0–6 have passed.

Process incrementally with checkpointing.

NEVER restart completed stages unnecessarily.


IMPLEMENTATION RULE:

Complete ONE phase at a time.

At the end of every phase:
1. Run tests
2. Verify outputs
3. Report files changed
4. Report resource usage
5. Report known problems
6. Wait for approval before beginning the next phase

Do not silently skip phases.
Do not implement future phases early unless explicitly required for the current phase.


============================================================
2. IMPORTANT RESEARCH PRINCIPLE
============================================================

Do NOT build this as a black-box system that says:

“This executive is lying.”

That is NOT the goal.

The system is intended to identify:

“linguistic patterns that may indicate elevated uncertainty, avoidance, or concealed risk”

and separately determine:

“whether a claim is supported, contradicted, partially supported, or unverifiable based on independently retrieved evidence.”

Never infer deception purely from sentiment.

Never label a statement as false solely because the stock price declined.

Never label an executive deceptive based only on model output.

The system must distinguish:

A. Linguistic risk
B. Factual verification
C. Subsequent financial outcome

These are related but separate signals.


============================================================
3. DATASET
============================================================

Primary dataset:

Hugging Face:
Bose345/sp500_earnings_transcripts

The dataset described for this project contains approximately:

- 33,362 transcripts
- 685 companies
- 2005–2025
- 11 GICS sectors

Relevant fields include:

- symbol
- company_name
- company_id
- year
- quarter
- date
- content
- structured_content

Use:

structured_content

whenever possible because it contains speaker-segmented dialogue.

The raw content should be retained for traceability.

DO NOT load the entire dataset into RAM.

The target environment can have only approximately 12 GB system RAM.

Use streaming/chunked processing and persistent storage.

Preferred approach:

Dataset stream
    ->
process one transcript
    ->
save result
    ->
release memory
    ->
process next transcript

The system MUST support checkpointing so that crashes or interruptions do not require restarting from the beginning.


============================================================
4. MEMORY / PERFORMANCE REQUIREMENTS
============================================================

Critical requirement:

The application must be designed to run within a 12 GB system RAM budget.

Do not create giant pandas DataFrames containing the full dataset.

Do not keep the complete transcript corpus in memory.

Do not store unnecessary embeddings.

Do not process all transcript sentences simultaneously.

Use:

- streaming
- generators
- chunk processing
- SQLite
- Parquet where useful
- incremental writes
- batch inference
- explicit memory cleanup
- checkpoints
- resume support

Long-running stages must save progress.

Every processing stage must be resumable.

For example:

processed_transcripts.db

should record which transcript IDs have already been completed.

If the application crashes at transcript 14,382, rerunning it should continue from there rather than start from transcript 1.


============================================================
5. HIGH-LEVEL ARCHITECTURE
============================================================

Implement the following architecture:

                    EARNINGS CALL TRANSCRIPT
                              |
                              v
                    DATA INGESTION
                              |
                              v
                    PREPROCESSING
                              |
                              v
                  SENTENCE / SEGMENTATION
                              |
                +-------------+--------------+
                |             |              |
                v             v              v
             HEDGING      EVASIVENESS    TONE SHIFT
             DETECTOR       DETECTOR       DETECTOR
                |             |              |
                +-------------+--------------+
                              |
                              v
                     HIDDEN RISK SCORE
                              |
                +-------------+--------------+
                |                            |
                v                            v
          CLAIM EXTRACTION             MARKET ANALYSIS
                |
                v
        EVIDENCE RETRIEVAL
                |
       +--------+--------+
       |        |        |
       v        v        v
      SEC      NEWS     MARKET
    SOURCES   SOURCES   SOURCES
       |        |        |
       +--------+--------+
                |
                v
        FACT VERIFICATION
                |
       +--------+--------+--------+
       |        |        |        |
       v        v        v        v
   SUPPORTED CONTRADICTED PARTIAL UNKNOWN
                |
                v
             DATABASE
                |
                v
          STREAMLIT DASHBOARD


============================================================
6. MODULE 1 — DATA INGESTION
============================================================

Create:

src/data_loader.py

Responsibilities:

- Load Hugging Face dataset
- Stream records
- Normalize metadata
- Detect missing fields
- Assign unique transcript IDs
- Save metadata
- Resume from checkpoints

Transcript ID should be deterministic.

Example:

AAPL_2024_Q2_2024-05-02

Avoid duplicate processing.

Add validation for:

- missing company symbol
- missing date
- malformed structured_content
- empty transcript
- duplicate transcript
- invalid quarter/year


============================================================
7. MODULE 2 — TRANSCRIPT PARSING
============================================================

Create:

src/preprocessing.py
src/segmentation.py

Convert transcripts into normalized records.

Each segment should contain:

- transcript_id
- segment_id
- company
- ticker
- date
- year
- quarter
- speaker
- speaker_role
- section
- text

Possible section labels:

- prepared_remarks
- q_and_a
- opening
- closing
- unknown

Speaker roles should preferably be mapped to:

- executive
- analyst
- operator
- unknown

The most important distinction is:

EXECUTIVE ANSWERS VS ANALYST QUESTIONS

For Q&A, associate each analyst question with the corresponding executive answer whenever structurally possible.

Store:

question_id
question_text
answer_text
speaker
date
transcript_id


============================================================
8. MODULE 3 — SENTENCE SEGMENTATION
============================================================

Split executive statements into sentences.

Each sentence must retain:

- transcript_id
- segment_id
- sentence_id
- sentence_number
- speaker
- section
- text

Do NOT lose the original context.

For every sentence, preserve links to:

original transcript
original segment
speaker
question-answer pair if applicable

This is required for explainability.


============================================================
9. MODULE 4 — HEDGING DETECTOR
============================================================

Implement a financial-language hedging/uncertainty detector.

Start with a rule-based + lexical approach.

Examples of potential hedging/uncertainty language:

- may
- might
- could
- potentially
- approximately
- roughly
- we believe
- we expect
- we anticipate
- it is difficult to say
- too early to tell
- we are evaluating
- we continue to monitor
- uncertain
- depending on
- subject to
- cannot comment
- we remain cautious
- we are cautiously optimistic

Do NOT hard-code these as automatically suspicious.

They are features.

The detector should calculate metrics such as:

- hedge_count
- hedge_density
- uncertainty_phrase_count
- qualifier_density
- modal_verb_density
- uncertainty_score

Normalize scores between 0 and 100.

The final output per sentence should resemble:

{
    "hedging_score": 74.2,
    "hedge_density": 0.083,
    "uncertainty_score": 69.8
}

Also implement a baseline using financial-domain lexical resources where appropriate.

The system must allow the hedging dictionary to be extended through configuration rather than hard-coded only inside Python.


============================================================
10. MODULE 5 — FINANCIAL SENTIMENT / TONE
============================================================

Use FinBERT or an equivalent financial-domain transformer model.

Preferred model:

ProsusAI/finbert

Do not classify only the entire transcript.

Run sentiment at sentence / segment level.

For each sentence store:

- positive_probability
- negative_probability
- neutral_probability
- sentiment_label

Example:

{
    "positive": 0.21,
    "negative": 0.67,
    "neutral": 0.12,
    "label": "negative"
}

Use max_length conservatively, for example 256 tokens for normal sentence inference.

Batch inference to improve performance.

If GPU exists, support GPU automatically.

If GPU does not exist, use CPU.

The system must not require a GPU to function.


============================================================
11. MODULE 6 — TONE SHIFT DETECTOR
============================================================

Tone shift is NOT simply sentiment.

The goal is to determine whether sentiment changes unusually within a call.

Implement at least:

A. Local tone shift

Compare a sentence's sentiment to the previous N sentences.

Example:

Previous local average:
negative = 0.10

Current sentence:
negative = 0.72

Tone shift = high

B. Section tone shift

Compare:

prepared remarks
vs
Q&A

C. Sequential sentiment movement

Detect abrupt changes in sentiment probabilities.

Store:

- previous_sentiment
- local_sentiment_mean
- local_sentiment_std
- tone_shift_score
- prepared_vs_qa_shift

Normalize tone shift to 0–100.


============================================================
12. MODULE 7 — EVASIVENESS DETECTOR
============================================================

This is a core component.

The system should detect linguistic evidence of answering indirectly or avoiding the substance of an analyst question.

For each Q&A pair:

QUESTION
    ->
EXECUTIVE ANSWER

Calculate:

1. Question-answer semantic similarity

2. Topic overlap

3. Keyword overlap

4. Direct-answer indicators

5. Deflection indicators

6. Hedging density

7. Generic language density

8. Answer length

9. Question-specific entity overlap

10. Numeric-response behavior


QUESTION-ANSWER SIMILARITY

Start with a lightweight TF-IDF + cosine similarity baseline.

Then provide an optional sentence embedding implementation.

Possible embedding models:

sentence-transformers or another suitable compact semantic encoder.

Do not load millions of embeddings into memory.

Process and store only what is necessary.


DEFLECTION FEATURES

Identify language such as:

- I can't comment
- it is too early
- we are monitoring
- we continue to evaluate
- we will provide information later
- we are focused on the long term
- as you know
- the important thing is
- I think the key point is

Again:

These are linguistic features, NOT proof of deception.


DIRECT RESPONSE FEATURES

Detect whether the answer directly addresses:

- requested metric
- requested timeframe
- requested geography
- requested product
- requested guidance

For example:

Question:
“What is the expected margin next quarter?”

Answer:
“We expect margins to be around 25%.”

This is direct.

Question:
“What is the expected margin next quarter?”

Answer:
“We remain focused on delivering long-term value for shareholders.”

This should produce a higher evasiveness score.

Calculate:

evasive_score

between 0 and 100.


============================================================
13. MODULE 8 — HIDDEN RISK SCORE
============================================================

Combine:

- hedging score
- evasiveness score
- tone shift score

into:

Hidden Risk Score

Initial formulation:

HiddenRisk =
    0.35 * Hedging
  + 0.40 * Evasiveness
  + 0.25 * ToneShift

These weights are INITIAL DESIGN VALUES ONLY.

Make the weighting configurable.

Do not claim these weights are scientifically optimal.

Later provide a model-based version where weights can be learned from labeled data / validation outcomes.

The system should produce scores at:

1. sentence level
2. segment level
3. Q&A level
4. transcript level

For transcript-level aggregation, do not use only a simple mean.

Use a combination of:

- average segment risk
- high-risk segment concentration
- top-k risk average

Suggested initial approach:

60% average risk
40% top-risk segment risk

Again, make configurable.


============================================================
14. MODULE 9 — CLAIM EXTRACTION
============================================================

This is a NEW and IMPORTANT component.

The system must identify factual or measurable claims made by executives.

Examples:

“We expect revenue growth of 10% next quarter.”

“Demand in Europe remains strong.”

“We expect margins to improve.”

“Capex will remain below $2 billion.”

“We have reduced our debt by 15%.”

“We expect the Chinese market to recover.”

For each claim extract structured fields where possible:

- claim_id
- transcript_id
- segment_id
- sentence_id
- speaker
- claim_text
- claim_type
- entity
- metric
- value
- unit
- period
- geography
- product
- direction
- confidence

Potential claim types:

- revenue
- earnings
- margin
- guidance
- demand
- sales
- costs
- debt
- capex
- cash flow
- market conditions
- customer growth
- geographic performance
- product performance
- hiring
- layoffs
- strategic investment
- regulatory impact
- future outlook
- other


============================================================
15. MODULE 10 — CLAIM NORMALIZATION
============================================================

Convert natural language claims into structured representations.

Example:

Text:

“We expect approximately 10% revenue growth next quarter.”

Structured form:

{
    "metric": "revenue_growth",
    "value": 10,
    "unit": "percent",
    "period": "next_quarter",
    "claim_type": "guidance"
}

The normalization layer should be robust to:

- percentages
- currencies
- millions/billions
- ranges
- approximate values
- negative values
- increases/decreases
- quarter references
- year references

Example:

“Revenue should grow between 8 and 10%.”

Store:

lower_bound = 8
upper_bound = 10

Do not force every claim into a number.


============================================================
16. MODULE 11 — EVIDENCE RETRIEVAL
============================================================

For each important claim, retrieve external evidence.

Use three evidence categories:

A. OFFICIAL FINANCIAL EVIDENCE
B. NEWS EVIDENCE
C. MARKET EVIDENCE


------------------------------------------------------------
16A. SEC / OFFICIAL FILINGS
------------------------------------------------------------

Integrate SEC EDGAR data where applicable.

Relevant sources can include:

- 10-K
- 10-Q
- 8-K
- earnings releases
- company filing metadata
- XBRL financial data

The SEC integration must be modular.

Create:

src/evidence/sec_provider.py

Do not hard-code credentials or user-agent values.

Use environment variables/configuration where required.

For every retrieved source store:

- source_type
- source_url
- filing_type
- filing_date
- title
- accession_number if available
- relevant_text
- relevance_score


------------------------------------------------------------
16B. NEWS EVIDENCE
------------------------------------------------------------

Integrate a configurable news provider.

The system should be able to search for:

company
+
claim topic
+
relevant time period

Examples:

ticker
company_name
“revenue”
“demand”
“guidance”
“margin”
“China”
etc.

The architecture must support at least one working news provider and make it possible to add another provider later.

Possible provider implementations can include:

- financial news API
- GDELT
- another reliable news search provider

Do not scrape websites illegally or in ways that violate their terms.

For every news result store:

- headline
- source
- url
- publication_time
- article snippet/text where permitted
- relevance_score
- retrieval_timestamp

The system must preserve the original source URL.


------------------------------------------------------------
16C. STOCK MARKET DATA
------------------------------------------------------------

Integrate historical stock price data.

For each company and earnings-call date, calculate:

- price before call
- closing price on call date where appropriate
- next trading day return
- 1-day return
- 5-day return
- 10-day return
- 20-day return
- 5-day realized volatility
- 10-day realized volatility
- abnormal-return proxy if benchmark data is available

The system must understand trading-day gaps.

Do NOT assume every calendar day is a trading day.

Use appropriate market timestamps.

Make market data provider configurable.


============================================================
17. MODULE 12 — TEMPORAL EVIDENCE LOGIC
============================================================

This is CRITICAL.

Do not mix evidence from the wrong time period.

Every evidence item must have:

- publication_date
- source_date
- event_date
- retrieval_date where applicable

Distinguish:

1. CONTEMPORANEOUS EVIDENCE
Evidence available at or before the earnings call.

2. POST-CALL EVIDENCE
Evidence published after the call.

3. OUTCOME EVIDENCE
Later evidence showing what actually happened.

Example:

Earnings call:
2025-04-25

Claim:
“We expect 10% growth next quarter.”

Later actual result:
2025-07-25

That later result is valid for outcome verification.

However, it must NEVER be used as input when calculating the original Hidden Risk Score.

This prevents data leakage.


============================================================
18. MODULE 13 — FACT VERIFICATION
============================================================

For every extracted claim, compare it with retrieved evidence.

Final statuses:

SUPPORTED
CONTRADICTED
PARTIALLY_SUPPORTED
NOT_VERIFIABLE

NEVER simply output TRUE/FALSE.

Each verification should contain:

{
    "claim_id": "...",
    "status": "PARTIALLY_SUPPORTED",
    "confidence": 0.82,
    "reason": "...",
    "evidence_ids": [...]
}

Example:

CLAIM:

“We expect revenue growth of approximately 10%.”

Later result:

Revenue growth = 5.8%

Output:

CONTRADICTED

with evidence:

- official filing
- earnings release
- relevant news if available


IMPORTANT:

Stock-price movement should NEVER by itself determine whether a factual claim is true.

For example:

Claim:
“Demand remained strong.”

Stock falls 8%.

This does NOT mean the claim was false.

The stock reaction is only market context.

For numerical accounting claims, official financial disclosures should generally receive greater evidentiary weight.


============================================================
19. MODULE 14 — EVIDENCE RANKING
============================================================

Evidence should receive a source-quality score.

Example conceptual hierarchy:

Tier 1:
Official SEC/company filing
Official financial statement
Official earnings release

Tier 2:
High-quality financial news

Tier 3:
General news / secondary sources

Tier 4:
Low-authority or unverified sources

The exact weights must be configurable.

For each claim show:

Evidence strength
Source quality
Date relevance
Semantic relevance

Example:

Claim:
“Revenue will increase 10%.”

Evidence:

SEC filing
Relevance: 0.96
Authority: High

News article
Relevance: 0.87
Authority: Medium

Market movement
Relevance: 0.52
Authority: Outcome/context only


============================================================
20. MODULE 15 — EVIDENCE-BASED REASONING
============================================================

Use a lightweight Natural Language Inference / reasoning layer where useful.

Compare:

CLAIM

against

EVIDENCE

and determine:

- entailment/support
- contradiction
- neutral/insufficient evidence

Do not make the entire project dependent on an expensive external LLM.

The system should function with deterministic/reproducible features plus optional LLM reasoning.

If an LLM API is used:

- keep API provider configurable
- store prompts/version
- store response
- store model name
- never expose API keys
- implement retry logic
- implement rate limiting
- cache repeated requests

The LLM must not be blindly trusted.

The final verification engine should combine:

- structured numerical comparison
- semantic similarity
- evidence source quality
- temporal validity
- optional NLI/LLM reasoning


============================================================
21. MODULE 16 — MARKET VALIDATION
============================================================

The project must test whether Hidden Risk Score has any relationship with later market behavior.

For each transcript calculate:

- Hidden Risk Score
- 1-day return
- 5-day return
- 10-day return
- 20-day return
- 5-day volatility
- 10-day volatility

Perform statistical analysis.

At minimum calculate:

- Pearson correlation
- Spearman correlation
- regression analysis
- high-risk vs low-risk descriptive comparison

Do NOT make unsupported causal claims.

The analysis should use language such as:

“associated with”
“correlated with”
“observed relationship”

not:

“causes stock price decline”


============================================================
22. MODULE 17 — OUTCOME VALIDATION
============================================================

Also test whether high-risk calls are associated with later:

- guidance reductions
- negative earnings surprises
- increased volatility
- deteriorating reported metrics

Where the necessary data can be reliably obtained.

Create a structured outcomes table.

Example:

transcript_id
hidden_risk
guidance_at_call
later_guidance
guidance_change
actual_revenue
expected_revenue
earnings_surprise
market_return
market_volatility


============================================================
23. MODULE 18 — HUMAN ANNOTATION DATASET
============================================================

Create an optional manual annotation workflow.

Select approximately 500 Q&A responses.

Allow human annotators to label:

0 = direct
1 = mildly evasive
2 = strongly evasive

Also optionally annotate:

- hedging
- topic avoidance
- directness
- perceived uncertainty

Store labels separately.

Use these labels to evaluate the automated detector with:

- precision
- recall
- F1
- confusion matrix

Do not train on the test annotations.

Provide an annotation export/import format such as CSV.


============================================================
24. MACHINE LEARNING MODEL
============================================================

Start with interpretable feature-based models.

Candidate models:

- Logistic Regression
- Random Forest
- LightGBM
- XGBoost

Preferred main tabular model:

LightGBM

Possible features:

hedge_density
uncertainty_score
evasive_score
qa_similarity
topic_overlap
answer_length
generic_language_score
tone_shift_score
negative_probability
positive_probability
neutral_probability
question_numeric_indicator
answer_numeric_indicator
section_type
speaker_role
etc.

Train an aggregation model for Hidden Risk where appropriate.

Do not train an enormous deep model unless justified by experimental results.


============================================================
25. BASELINE VS PROPOSED SYSTEM
============================================================

The project must include baseline comparisons.

Baseline 1:

Simple financial sentiment / lexicon approach

Baseline 2:

FinBERT sentiment only

Baseline 3:

Hedging detector only

Proposed system:

Hedging
+
Evasiveness
+
Tone Shift
+
Claim verification
+
Market outcome validation

The goal is to demonstrate what additional information the combined framework provides.

Do not invent performance improvements.


============================================================
26. DATABASE DESIGN
============================================================

Use SQLite for the main persistent local database.

Recommended tables:


transcripts

- transcript_id
- ticker
- company_name
- company_id
- date
- year
- quarter
- sector
- raw_text
- processing_status


segments

- segment_id
- transcript_id
- speaker
- speaker_role
- section
- text
- sequence


sentences

- sentence_id
- segment_id
- transcript_id
- sentence_number
- text


risk_scores

- sentence_id
- hedging_score
- evasiveness_score
- tone_shift_score
- hidden_risk_score


transcript_scores

- transcript_id
- overall_hidden_risk
- average_risk
- top_risk_average
- qa_risk
- prepared_risk
- highest_risk_segment


claims

- claim_id
- transcript_id
- segment_id
- sentence_id
- claim_text
- claim_type
- entity
- metric
- value
- lower_bound
- upper_bound
- unit
- period
- geography


evidence

- evidence_id
- claim_id
- source_type
- source_name
- source_url
- publication_date
- title
- text
- relevance_score
- authority_score


verification

- claim_id
- status
- confidence
- reasoning
- created_at


market_events

- transcript_id
- ticker
- call_date
- price_before
- price_after
- return_1d
- return_5d
- return_10d
- return_20d
- volatility_5d
- volatility_10d


outcomes

- transcript_id
- later_guidance
- guidance_change
- actual_revenue
- earnings_surprise
- outcome_date


processing_checkpoints

- job_name
- last_processed_id
- last_processed_timestamp
- status
- error_message


============================================================
27. DASHBOARD
============================================================

Build the final interface using Streamlit.

Application:

dashboard/app.py

The dashboard must be analyst-friendly.

Main screen:

SEARCH / SELECT COMPANY
SEARCH / SELECT TRANSCRIPT
DATE
QUARTER


Then display:

-----------------------------------
HIDDEN RISK SCORE
-----------------------------------

Overall score: 74 / 100

Hedging:      81
Evasiveness: 77
Tone Shift:  62


-----------------------------------
RISK DISTRIBUTION
-----------------------------------

Show:

- prepared remarks score
- Q&A score
- timeline across transcript
- high-risk segments


-----------------------------------
FLAGGED STATEMENTS
-----------------------------------

For every flagged statement show:

Original sentence

Risk score

Hedging score

Evasiveness score

Tone-shift score

Reason / contributing features


Example:

“It's too early to comment on the impact...”

Risk = 89

Hedging = 84
Evasiveness = 91
Tone shift = 72


-----------------------------------
FACT VERIFICATION
-----------------------------------

For each extracted claim:

CLAIM:
“We expect margin to improve next quarter.”

STATUS:
PARTIALLY SUPPORTED

Confidence:
82%

Evidence:

[SEC filing]
[News article]
[Later result]


Each evidence item should be clickable and show:

- title
- source
- date
- URL
- relevant excerpt if permitted


-----------------------------------
MARKET REACTION
-----------------------------------

Display:

Price before call
Price after call
1D return
5D return
10D return
20D return
5D volatility
10D volatility


-----------------------------------
EVENT TIMELINE
-----------------------------------

Example:

APR 25
Earnings Call
Risk = 78

APR 26
Stock -3.4%

APR 29
News: demand concerns

MAY 08
Guidance revision

JUL 25
Reported results


============================================================
28. COMPANY COMPARISON
============================================================

Allow comparison across companies.

Filters:

- sector
- year
- quarter
- company
- risk range

Display:

Company
Transcript Date
Hidden Risk Score
Q&A Risk
Fact Contradiction Count
5D Volatility
Guidance Change


Do not create arbitrary “best/worst company” rankings.

This is an analytical research tool, not a trading recommendation system.


============================================================
29. TRANSCRIPT EXPLORER
============================================================

Allow the user to open the entire transcript.

Highlight:

- high-risk sentences
- hedging
- evasive answers
- tone shifts
- extracted factual claims

Use different visual indicators for:

Hedging
Evasiveness
Tone Shift
Claims requiring verification


============================================================
30. FACT-CHECK VIEW
============================================================

Create a dedicated page:

FACT CHECKER

Input:

Company
Transcript
Claim

Output:

Claim:
“Revenue will grow by 10%.”

Evidence retrieved:

1. SEC filing
2. Earnings release
3. Financial news
4. Later financial result

Final result:

CONTRADICTED

Confidence:
0.91

Reason:

“The subsequent reported revenue growth was 5.8%, below the stated expected range.”

The user must be able to inspect the original evidence.


============================================================
31. API / PROVIDER ARCHITECTURE
============================================================

Do NOT tightly couple the project to one API.

Implement interfaces/providers:

MarketDataProvider
NewsProvider
SecProvider

Example:

src/providers/market/base.py
src/providers/market/provider_x.py

src/providers/news/base.py
src/providers/news/provider_x.py

src/providers/sec/sec_edgar.py


Provider selection should be controlled by configuration.

Environment variables should be used for secrets.

Example:

.env

SEC_USER_AGENT=
MARKET_API_KEY=
NEWS_API_KEY=


Never commit secrets.

Provide:

.env.example


============================================================
32. CACHING
============================================================

External API calls can become expensive and slow.

Cache:

- SEC documents
- news results
- market data
- claim verification results
- model results

Use SQLite / disk cache.

Never repeatedly request the same evidence.


============================================================
33. ERROR HANDLING
============================================================

The system must gracefully handle:

- API timeout
- API rate limits
- missing company data
- missing market data
- missing news
- SEC unavailable
- malformed transcript
- malformed structured_content
- model inference errors
- network errors

A missing external source must NOT crash the entire pipeline.

Example:

News unavailable

should result in:

news_status = unavailable

while SEC and market evidence continue processing.


============================================================
34. CHECKPOINTING
============================================================

Every long-running pipeline must support resume.

Example:

python run_pipeline.py --stage transcript_analysis

If interrupted:

python run_pipeline.py --stage transcript_analysis --resume

It should continue from the last successful transcript.

Store checkpoints in SQLite.

Stages should be independently runnable:

1. ingest
2. preprocess
3. segment
4. risk
5. claims
6. evidence
7. verification
8. market
9. aggregation
10. dashboard


============================================================
35. COMMAND-LINE INTERFACE
============================================================

Create a simple CLI.

Examples:

python run_pipeline.py --stage ingest

python run_pipeline.py --stage preprocess

python run_pipeline.py --stage risk

python run_pipeline.py --stage claims

python run_pipeline.py --stage evidence

python run_pipeline.py --stage verify

python run_pipeline.py --stage market

python run_pipeline.py --stage all

Support:

--limit 100
--resume
--company AAPL
--year 2024
--quarter Q2


============================================================
36. SMALL-SCALE DEVELOPMENT MODE
============================================================

Before processing the full dataset, create:

DEVELOPMENT MODE

Example:

python run_pipeline.py --stage all --limit 100

This should process only 100 transcripts.

Then:

--limit 500

Then:

--limit 2000

Only after successful validation should the full dataset be processed.


============================================================
37. OUTPUT FILES
============================================================

Generate useful export files.

Examples:

outputs/transcript_scores.csv

outputs/sentence_scores.csv

outputs/claims.csv

outputs/evidence.csv

outputs/verification.csv

outputs/market_events.csv

outputs/company_summary.csv

outputs/research_results.csv


============================================================
38. RESEARCH REPORT OUTPUT
============================================================

Create scripts that generate statistical analysis.

Generate:

- descriptive statistics
- score distributions
- sector distributions
- Q&A vs prepared comparison
- correlation tables
- volatility comparison
- guidance-change analysis
- verification statistics

Example:

Percentage of claims:

Supported
Contradicted
Partially supported
Not verifiable


Also calculate:

Average Hidden Risk Score

Average Q&A score

Average prepared-remarks score

Average post-call volatility for high-risk calls

Average post-call volatility for lower-risk calls


Do not make unsupported causal claims.


============================================================
39. VISUALIZATIONS
============================================================

Generate charts such as:

1. Hidden Risk distribution

2. Risk score by sector

3. Q&A vs prepared remarks

4. Hidden Risk vs 5D volatility

5. Hidden Risk vs subsequent return

6. Tone shift timeline

7. Number of verified/contradicted claims

8. Risk over time

9. Company-level timeline

10. High-risk sentence concentration


============================================================
40. EVALUATION METRICS
============================================================

For linguistic detection:

- Precision
- Recall
- F1

For classification:

- ROC-AUC where appropriate
- PR-AUC where appropriate

For fact verification:

- Accuracy
- Precision
- Recall
- F1

For market relationship:

- Pearson correlation
- Spearman correlation
- regression coefficients
- confidence intervals where appropriate

Do not present correlation as causation.


============================================================
41. TRAIN / VALIDATION / TEST SPLIT
============================================================

Avoid random sentence-level splitting.

Use time-aware splits.

Example:

TRAIN:
2005–2018

VALIDATION:
2019–2021

TEST:
2022–2025


The exact split should be configurable.

Never allow future outcomes to become model features for an earlier transcript.

The market outcome after a call may be used for:

VALIDATION

but must not be used to calculate:

ORIGINAL HIDDEN RISK SCORE


============================================================
42. AVOID DATA LEAKAGE
============================================================

This must be explicitly enforced.

Example:

Call:
2024-04-25

Hidden Risk Score may use:

transcript text
speaker information
language
question-answer structure
information available at call time

It must NOT use:

future stock returns
future SEC filings
future news
later reported earnings
future guidance

Those can be used only as:

ground truth / outcome validation


============================================================
43. INTERPRETABILITY
============================================================

Every risk score must be explainable.

For a high-risk sentence, show:

Risk score

Hedging contribution

Evasiveness contribution

Tone shift contribution

Relevant linguistic phrases

Question-answer similarity

Comparison with surrounding context


For every fact verification:

Show:

Original claim
Evidence
Source
Evidence date
Verification status
Confidence
Reasoning


============================================================
44. CONFIDENCE HANDLING
============================================================

Never show:

TRUE

with 100% confidence simply because the model predicts it.

Use probabilities/confidence scores where meaningful.

Example:

SUPPORTED
Confidence: 0.86


If evidence is insufficient:

NOT_VERIFIABLE

Confidence: 0.42


The dashboard should clearly distinguish:

model confidence

from

evidence strength


============================================================
45. IMPORTANT DISTINCTION BETWEEN RISK AND FACT-CHECKING
============================================================

The system must treat these as independent dimensions.

Example:

Statement:
“We remain cautiously optimistic about demand.”

This may have:

Hedging = high
Evasiveness = medium
Tone shift = high

Therefore:

Hidden Risk = high

But factual verification might be:

NOT VERIFIABLE

because “optimistic” is subjective.

Another example:

“We expect revenue of $10 billion.”

Risk could be low.

Later actual revenue could be:

$7 billion.

Fact verification:

CONTRADICTED

Therefore:

HIGH FACTUAL DISCREPANCY
LOW LINGUISTIC RISK

This kind of separation is a major feature of the system.


============================================================
46. OPTIONAL ADVANCED ANALYSIS
============================================================

After the basic system works, optionally implement:

A. Abnormal return relative to S&P 500

B. Sector-adjusted return

C. Market-volatility-adjusted risk

D. Earnings surprise detection

E. Guidance revision detection

F. Longitudinal company risk trends

G. Repeated high-risk language patterns

H. Executive-level linguistic patterns

I. Q&A topic clusters

J. Retrieval-augmented evidence summaries


============================================================
47. RAG ARCHITECTURE
============================================================

The evidence verification system can use RAG.

Pipeline:

Claim
  ->
retrieve relevant evidence
  ->
rank evidence
  ->
construct evidence context
  ->
NLI / reasoning
  ->
verification


Do NOT embed the entire internet.

Index only retrieved/cached evidence or a manageable evidence corpus.

Use FAISS or another vector index only where beneficial.

Prefer structured filtering first:

company
date
claim type
metric
topic

Then use semantic retrieval.

This will reduce memory and computational requirements.


============================================================
48. PROJECT FOLDER STRUCTURE
============================================================

Use something close to:

hidden-risk-analyzer/

├── data/
│   ├── raw/
│   ├── processed/
│   ├── market/
│   └── news/
│
├── database/
│   └── hidden_risk.db
│
├── models/
│   ├── finbert/
│   └── risk_model/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── hedging.py
│   ├── sentiment.py
│   ├── tone_shift.py
│   ├── evasiveness.py
│   ├── risk_score.py
│   ├── claims.py
│   ├── verification.py
│   ├── market.py
│   ├── aggregation.py
│   └── utils.py
│
├── src/providers/
│   ├── sec/
│   ├── news/
│   └── market/
│
├── dashboard/
│   ├── app.py
│   ├── pages/
│   └── components/
│
├── scripts/
│   ├── run_pipeline.py
│   ├── evaluate.py
│   └── build_reports.py
│
├── tests/
│
├── outputs/
│
├── checkpoints/
│
├── .env.example
├── requirements.txt
├── README.md
└── config.yaml


============================================================
49. CONFIGURATION
============================================================

All important settings must be configurable.

Example config.yaml:

dataset:
    name: Bose345/sp500_earnings_transcripts

processing:
    batch_size: 16
    max_length: 256
    checkpoint_interval: 100

risk:
    hedging_weight: 0.35
    evasiveness_weight: 0.40
    tone_shift_weight: 0.25

aggregation:
    average_weight: 0.60
    top_risk_weight: 0.40

market:
    windows:
        - 1
        - 5
        - 10
        - 20

verification:
    minimum_evidence_score: 0.50


============================================================
50. TESTING
============================================================

Create unit tests for:

- transcript parsing
- sentence segmentation
- speaker detection
- question-answer pairing
- hedging detection
- tone shift calculation
- question-answer similarity
- claim extraction
- numerical claim normalization
- temporal filtering
- fact verification
- market calculations
- checkpointing

Create integration tests using a tiny sample dataset.

The entire pipeline must be able to run on a small test set before running on 33,000+ transcripts.


============================================================
51. LOGGING
============================================================

Implement proper logging.

Logs should include:

- current stage
- transcript count
- processing speed
- errors
- API calls
- API failures
- cache hits
- cache misses
- memory-sensitive operations
- checkpoint saves

Example:

[Risk] Processed 10,000 / 33,362
[Risk] Current speed: 3.2 transcripts/sec
[Risk] Checkpoint saved


============================================================
52. FINAL USER EXPERIENCE
============================================================

A user should be able to:

1. Select company
2. Select earnings call
3. View transcript
4. See Hidden Risk Score
5. See risk at sentence level
6. See Q&A risk
7. Inspect flagged statements
8. Inspect extracted claims
9. See evidence from SEC/news
10. See verification status
11. See stock-market reaction
12. View timeline
13. Compare multiple calls
14. Export results


============================================================
53. EXAMPLE FINAL ANALYSIS
============================================================

For a hypothetical company:

Company:
ABC Corp

Quarter:
Q2 2025

Hidden Risk Score:
76 / 100

Components:

Hedging:
82

Evasiveness:
79

Tone Shift:
61


High-risk statement:

“We are continuing to evaluate the impact, and it is too early to provide specific guidance.”

Risk:
91


Claim:

“We expect approximately 12% revenue growth next quarter.”

Verification:

PARTIALLY_SUPPORTED

Confidence:
0.83


Evidence:

SEC:
Later reported revenue growth = 8.4%

News:
Several financial reports mention weaker demand.

Market:

1-day return:
-3.2%

5-day return:
-7.1%

10-day volatility:
6.5%


The system should NOT say:

“The executive lied.”

Instead it should say:

“The claim was partially supported by subsequent reported results. Linguistic analysis identified elevated hedging and evasiveness in the surrounding Q&A. The stock also experienced negative post-call returns and elevated volatility. These signals are presented as separate pieces of evidence rather than proof of deception.”


============================================================
54. DEVELOPMENT STRATEGY
============================================================

DO NOT immediately process all 33,362 transcripts.

Build in stages.

STAGE 1:

100 transcripts

Implement:

- ingestion
- preprocessing
- segmentation
- Q&A pairing

STAGE 2:

100 transcripts

Implement:

- hedging
- FinBERT
- tone shift
- evasiveness
- Hidden Risk Score


STAGE 3:

100 transcripts

Implement:

- claim extraction
- SEC retrieval
- market data
- news retrieval
- verification


STAGE 4:

100 transcripts

Build:

- Streamlit dashboard
- transcript explorer
- fact-check page
- market timeline


STAGE 5:

500 transcripts

Evaluate accuracy and performance.


STAGE 6:

2,000 transcripts

Stress-test memory, API limits, processing time.


STAGE 7:

Full dataset

Process all available transcripts incrementally.


============================================================
55. PERFORMANCE TARGET
============================================================

The system should prioritize correctness and resumability before maximum speed.

Optimize bottlenecks only after profiling.

Do not sacrifice correctness for speed.

Use:

- batching
- caching
- SQLite indexing
- parallel CPU processing where safe
- GPU inference when available
- streaming


============================================================
56. DOCUMENTATION REQUIREMENTS
============================================================

Create a README containing:

1. Project overview
2. Architecture
3. Installation
4. Environment setup
5. Dataset setup
6. API setup
7. Database schema
8. Pipeline commands
9. Dashboard commands
10. Model descriptions
11. Fact verification methodology
12. Evaluation methodology
13. Data leakage safeguards
14. Memory optimization
15. Troubleshooting
16. Example output


============================================================
57. FINAL DELIVERABLES
============================================================

The finished implementation must provide:

1. Complete source code

2. Reproducible environment

3. Dataset ingestion

4. Memory-safe processing

5. Persistent SQLite database

6. Checkpoint/resume support

7. Hedging detection

8. Evasiveness detection

9. FinBERT sentiment

10. Tone shift detection

11. Hidden Risk Score

12. Claim extraction

13. SEC evidence retrieval

14. News evidence retrieval

15. Market-data retrieval

16. Fact verification

17. Market validation

18. Statistical evaluation

19. Streamlit dashboard

20. CSV exports

21. Research visualizations

22. Tests

23. Documentation

24. Configuration system

25. Logging


============================================================
58. ACCEPTANCE CRITERIA
============================================================

The implementation is considered successful only when all of the following work:

A. A transcript can be loaded.

B. It can be converted into structured speaker segments.

C. Q&A questions and answers can be associated.

D. Individual sentences receive:
   - hedging score
   - evasiveness score
   - tone shift score
   - Hidden Risk Score

E. High-risk statements can be displayed with explanations.

F. Important factual claims can be extracted.

G. Evidence can be retrieved from official financial sources.

H. News evidence can be retrieved.

I. Stock-market data can be retrieved.

J. Each claim receives:
   - supported
   - contradicted
   - partially supported
   - not verifiable

K. Evidence and dates are displayed.

L. Market reaction is displayed.

M. Hidden Risk Score can be statistically compared with subsequent market outcomes.

N. No future information leaks into the original risk score.

O. The pipeline can resume after interruption.

P. The first 100 transcripts can be processed within the 12 GB RAM environment.

Q. Full processing can be performed incrementally without requiring the entire corpus in memory.

R. Streamlit dashboard works locally.

S. Results can be exported.

T. All important assumptions and limitations are documented.


============================================================
59. IMPORTANT IMPLEMENTATION RULES
============================================================

1. Do not fabricate data.

2. Do not fabricate news.

3. Do not fabricate SEC filings.

4. Do not fabricate stock-market values.

5. Do not claim an API returned information unless it actually did.

6. If an API is unavailable, mark the evidence as unavailable.

7. Never mark an unsupported claim as verified.

8. Preserve source URLs.

9. Preserve evidence dates.

10. Prevent future information leakage.

11. Keep raw transcript text traceable.

12. Make every long-running operation resumable.

13. Keep memory usage low.

14. Keep API providers replaceable.

15. Keep model parameters configurable.

16. Prefer reproducible deterministic calculations over opaque LLM reasoning whenever possible.

17. Do not describe linguistic risk as proof of deception.

18. Do not interpret stock price movement alone as proof that a statement was false.

19. Clearly distinguish:
    linguistic signal,
    factual evidence,
    market outcome.

20. Build and verify each stage before scaling.


============================================================
60. STARTING TASK
============================================================

Do not immediately implement the entire 33,362-transcript system.

First create the project structure, configuration, database schema, logging system, checkpoint system, and a minimal end-to-end prototype.

The FIRST executable milestone should be:

100 transcripts
    ->
preprocessing
    ->
speaker segmentation
    ->
Q&A pairing
    ->
sentence segmentation
    ->
hedging score
    ->
FinBERT sentiment
    ->
tone shift
    ->
evasive score
    ->
Hidden Risk Score
    ->
claim extraction
    ->
market evidence
    ->
SEC/news evidence where available
    ->
verification
    ->
SQLite
    ->
Streamlit dashboard

After this end-to-end version works, produce a performance report and memory report.

Only then scale the pipeline.

When implementing, make reasonable engineering decisions without asking me to redesign the architecture unless a required external API, dataset field, or critical assumption is genuinely unavailable.

Before using any external API, verify its current documentation and implement against the actual current interface.

The primary priorities are:

CORRECTNESS
TRACEABILITY
NO DATA LEAKAGE
RESUMABILITY
MEMORY EFFICIENCY
INTERPRETABILITY

in that order.