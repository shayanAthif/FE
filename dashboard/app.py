import json
import streamlit as st
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from components import data_loader


def main():
    st.set_page_config(
        page_title="Hidden Risk Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Hidden Risk Analyzer")
    st.markdown("*Analyzing earnings call transcripts for concealed risk signals*")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Transcript Explorer", "Company Comparison", "About"]
    )
    
    # Load tickers
    tickers = data_loader.get_tickers()
    
    if page == "Home":
        show_home(tickers)
    elif page == "Transcript Explorer":
        show_transcript_explorer(tickers)
    elif page == "Company Comparison":
        show_company_comparison(tickers)
    elif page == "About":
        show_about()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Data sources: SEC EDGAR, Google News, Yahoo Finance. "
        "Hidden Risk Score: Hedging + Evasiveness + Tone Shift signals. "
        "Fact verification is separate from risk scoring."
    )


def show_home(tickers):
    st.markdown("## Welcome to Hidden Risk Analyzer")
    st.markdown("This dashboard helps analysts identify concealed risk signals in S&P 500 earnings call transcripts.")
    
    total_calls = sum(len(data_loader.get_transcripts_for_ticker(t)) for t in tickers)
    avg_risk = get_avg_risk()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Companies", len(tickers))
    with col2:
        st.metric("Total Transcripts", total_calls)
    with col3:
        st.metric("Average Risk Score", f"{avg_risk:.1f}")
    
    st.markdown("### Quick Stats")
    st.markdown("---")
    st.markdown("**Linguistic Risk**")
    st.markdown("- Hedging: Uncertainty language in executive responses")
    st.markdown("- Evasiveness: Topic avoidance in Q&A sessions")
    st.markdown("- Tone Shift: Negative sentiment accumulation in prepared remarks")
    st.markdown("---")
    st.markdown("**Fact Verification**")
    st.markdown("- Claims are verified against SEC filings and news articles")
    st.markdown("- Status: SUPPORTED, CONTRADICTED, PARTIALLY_SUPPORTED, or NOT_VERIFIABLE")
    st.markdown("---")
    st.markdown("**Market Outcome**")
    st.markdown("- Post-call stock price movements (1D, 5D, 10D, 20D)")
    st.markdown("- Volatility metrics relative to S&P 500 benchmark")
    
    # Quick search
    st.markdown("### Quick Search")
    search_ticker = st.selectbox(
        "Select a company to explore:",
        tickers,
        format_func=lambda x: f"{x} - {get_company_name(x)}"
    )
    if st.button("Explore"):
        st.session_state.selected_ticker = search_ticker
        st.rerun()


def show_transcript_explorer(tickers):
    st.markdown("## Transcript Explorer")
    
    selected_ticker = st.selectbox(
        "Select Company",
        tickers,
        format_func=lambda x: f"{x} - {get_company_name(x)}"
    )
    
    transcripts = data_loader.get_transcripts_for_ticker(selected_ticker)
    
    if not transcripts:
        st.warning("No transcripts found for this company.")
        return
    
    transcript_options = [
        f"{t['date']} - Q{t['quarter']} {t['year']} (Risk: {t['overall_hidden_risk']:.1f})"
        for t in transcripts
    ]
    selected_idx = st.selectbox(
        "Select Transcript",
        range(len(transcript_options)),
        format_func=lambda x: transcript_options[x]
    )
    
    selected_transcript = transcripts[selected_idx]
    show_transcript_detail(selected_transcript['transcript_id'])


def show_company_comparison(tickers):
    st.markdown("## Company Comparison")
    
    selected_tickers = st.multiselect(
        "Select companies to compare",
        tickers,
        default=tickers[:3] if len(tickers) >= 3 else tickers,
        format_func=lambda x: f"{x} - {get_company_name(x)}"
    )
    
    if not selected_tickers:
        st.info("Select at least one company to compare.")
        return
    
    comparison = data_loader.get_ticker_comparison(selected_tickers)
    
    st.markdown("### Risk Comparison")
    for t in selected_tickers:
        t_data = [r for r in comparison if r['ticker'] == t]
        if t_data:
            latest = t_data[0]
            st.metric(
                f"{t} - Latest Call (Risk: {latest['overall_hidden_risk']:.1f})",
                f"Q{latest['quarter']} {latest['year']}",
                delta=f"Risk: {latest['overall_hidden_risk']:.1f}"
            )
    
    st.markdown("### Detailed Comparison")
    import pandas as pd
    df = pd.DataFrame(comparison)
    st.dataframe(
        df[['ticker', 'date', 'overall_hidden_risk', 'hedging_avg', 'evasiveness_avg', 'tone_shift_avg']]
    )


def show_about():
    st.markdown("## About Hidden Risk Analyzer")
    st.markdown("""
    This dashboard is part of a research prototype that analyzes earnings call transcripts to identify
    concealed financial risk signals hidden in executive language.
    
    ### What We Measure
    1. **Linguistic Risk**: Hedging, evasiveness, and tone shifts in executive language
    2. **Fact Verification**: Whether claims are supported by SEC filings and news articles
    3. **Market Outcome**: Stock price reaction after earnings calls
    
    ### Important Notes
    - This is a research tool, not financial advice
    - Linguistic risk ≠ factual inaccuracy
    - Market outcomes are correlations, not causations
    - No trading recommendations are implied
    
    ### Data Sources
    - Earnings transcripts: HuggingFace Bose345/sp500_earnings_transcripts
    - SEC filings: EDGAR database
    - News articles: Google News RSS
    - Market data: Yahoo Finance
    """)


def get_company_name(ticker: str) -> str:
    conn = data_loader.get_connection()
    try:
        row = conn.execute(
            "SELECT company_name FROM transcripts WHERE ticker = ? LIMIT 1",
            (ticker,)
        ).fetchone()
        return row[0] if row else ticker
    finally:
        conn.close()


def get_avg_risk() -> float:
    conn = data_loader.get_connection()
    try:
        row = conn.execute("SELECT AVG(overall_hidden_risk) FROM transcript_scores").fetchone()
        return row[0] if row and row[0] else 0.0
    finally:
        conn.close()


def show_transcript_detail(transcript_id: str):
    transcript = data_loader.get_transcript_detail(transcript_id)
    sentences = data_loader.get_sentences(transcript_id)
    risk_scores = data_loader.get_risk_scores(transcript_id)
    claims = data_loader.get_claims(transcript_id)
    verifications = data_loader.get_verifications([c['claim_id'] for c in claims])
    evidence = data_loader.get_evidence_for_claims([c['claim_id'] for c in claims])
    market_data = data_loader.get_market_reaction(transcript_id)
    outcome = data_loader.get_outcomes(transcript_id)
    
    verification_map = {v['claim_id']: v for v in verifications}
    evidence_map = {}
    for ev in evidence:
        cid = ev['claim_id']
        if cid not in evidence_map:
            evidence_map[cid] = []
        evidence_map[cid].append(ev)
    
    # Risk score summary
    st.markdown("### Overall Hidden Risk Score")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Overall Risk", f"{transcript.get('overall_hidden_risk', 0):.1f}")
    with cols[1]:
        st.metric("Average Risk", f"{transcript.get('average_risk', 0):.1f}")
    with cols[2]:
        st.metric("Hedging", f"{transcript.get('hedging_avg', 0):.1f}")
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("Evasiveness", f"{transcript.get('evasiveness_avg', 0):.1f}")
    with cols[1]:
        st.metric("Tone Shift", f"{transcript.get('tone_shift_avg', 0):.1f}")
    
    st.markdown("---")
    
    # Q&A vs Prepared comparison
    st.markdown("### Q&A vs Prepared Remarks")
    qa_sentences = [s for s in sentences if s.get('section') == 'Q&A']
    prepared_sentences = [s for s in sentences if s.get('section') == 'Prepared Remarks']
    
    cols = st.columns(2)
    with cols[0]:
        st.metric("Q&A Sentences", len(qa_sentences))
    with cols[1]:
        st.metric("Prepared Remarks Sentences", len(prepared_sentences))
    
    qa_scores = [r for r in risk_scores if any(s['sentence_id'] == r['sentence_id'] for s in qa_sentences)]
    prepared_scores = [r for r in risk_scores if any(s['sentence_id'] == r['sentence_id'] for s in prepared_sentences)]
    
    cols = st.columns(2)
    with cols[0]:
        if qa_scores:
            avg_qa_risk = sum(s['hidden_risk_score'] for s in qa_scores) / len(qa_scores)
            st.metric("Avg Q&A Risk Score", f"{avg_qa_risk:.2f}")
    with cols[1]:
        if prepared_scores:
            avg_prepared_risk = sum(s['hidden_risk_score'] for s in prepared_scores) / len(prepared_scores)
            st.metric("Avg Prepared Risk Score", f"{avg_prepared_risk:.2f}")
    
    st.markdown("---")
    
    # Flagged statements
    st.markdown("### Flagged High-Risk Statements")
    high_risk_sentences = [s for s in sentences if s.get('hidden_risk_score', 0) > 50]
    
    if high_risk_sentences:
        for sentence in high_risk_sentences:
            risk_score = next((r for r in risk_scores if r['sentence_id'] == sentence['sentence_id']), None)
            if risk_score:
                with st.expander(f"Risk Score: {risk_score['hidden_risk_score']:.1f} | {sentence.get('text', '')[:100]}..."):
                    st.markdown(f"**Sentence:** {sentence.get('text', '')}")
                    st.markdown(f"**Section:** {sentence.get('section', 'N/A')} | **Speaker:** {sentence.get('speaker', 'N/A')}")
                    st.markdown(f"**Hedging:** {risk_score.get('hedging_score', 0):.2f}")
                    st.markdown(f"**Evasiveness:** {risk_score.get('evasiveness_score', 0):.2f}")
                    st.markdown(f"**Tone Shift:** {risk_score.get('tone_shift_score', 0):.2f}")
                    if risk_score.get('contributing_factors'):
                        raw_factors = risk_score['contributing_factors']
                        factors = json.loads(raw_factors) if isinstance(raw_factors, str) else raw_factors
                        if factors and isinstance(factors, dict):
                            st.markdown("**Contributing factors:**")
                            for k, v in factors.items():
                                st.write(f"- {k}: {v}")
    else:
        st.info("No high-risk statements flagged (threshold: 50)")
    
    st.markdown("---")
    
    # Extracted claims
    st.markdown("### Extracted Claims & Verification")
    
    if claims:
        for claim in claims:
            verification = verification_map.get(claim['claim_id'])
            evidence_list = evidence_map.get(claim['claim_id'], [])
            
            status = verification.get('status', 'UNKNOWN') if verification else 'UNKNOWN'
            status_colors = {
                'SUPPORTED': '#32CD32',
                'CONTRADICTED': '#FF4B4B',
                'PARTIALLY_SUPPORTED': '#FFD700',
                'NOT_VERIFIABLE': '#808080',
            }
            
            st.markdown("---")
            st.markdown(f"**{claim.get('claim_text', '')}**")
            
            cols = st.columns(3)
            with cols[0]:
                st.write(f"Metric: {claim.get('metric', 'N/A')}")
            with cols[1]:
                st.write(f"Value: {claim.get('value', 'N/A')} {claim.get('unit', '')}")
            with cols[2]:
                st.write(f"Period: {claim.get('period', 'N/A')}")
            
            if verification:
                st.markdown(
                    f'<div style="background-color: {status_colors[status]}20; border-left: 4px solid {status_colors[status]}; padding: 10px;">'
                    f'<strong style="color: {status_colors[status]};">{status}</strong> '
                    f'(Confidence: {verification.get("confidence", 0):.1%})</div>',
                    unsafe_allow_html=True
                )
                st.markdown(f"**Reasoning:** {verification.get('reasoning', '')}")
            
            if evidence_list:
                with st.expander(f"Show evidence ({len(evidence_list)} sources)", expanded=False):
                    for i, ev in enumerate(evidence_list, 1):
                        st.markdown(f"**Evidence {i}:** {ev.get('source_name', 'Unknown')}")
                        st.caption(f"Date: {ev.get('publication_date', 'N/A')}")
                        url = ev.get('source_url', '#')
                        if url:
                            st.markdown(f"[View source]({url})")
                        if ev.get('text'):
                            excerpt = ev['text'][:300] + "..." if len(ev['text']) > 300 else ev['text']
                            st.markdown(f"*{excerpt}*")
    else:
        st.info("No claims extracted for this transcript.")
    
    st.markdown("---")
    
    # Market reaction
    st.markdown("### Market Reaction")
    if market_data:
        cols = st.columns(4)
        with cols[0]:
            st.metric("1-Day Return", f"{market_data.get('return_1d', 0):.2%}")
        with cols[1]:
            st.metric("5-Day Return", f"{market_data.get('return_5d', 0):.2%}")
        with cols[2]:
            st.metric("10-Day Return", f"{market_data.get('return_10d', 0):.2%}")
        with cols[3]:
            st.metric("20-Day Return", f"{market_data.get('return_20d', 0):.2%}")
        
        cols = st.columns(2)
        with cols[0]:
            st.metric("5-Day Volatility", f"{market_data.get('volatility_5d', 0):.2%}")
        with cols[1]:
            st.metric("10-Day Volatility", f"{market_data.get('volatility_10d', 0):.2%}")
        
        if market_data.get('abnormal_return_5d') is not None:
            st.metric("Abnormal 5-Day Return", f"{market_data.get('abnormal_return_5d', 0):.2%}")
    else:
        st.info("No market data available.")
    
    # Outcome
    if outcome:
        st.markdown("### Later Fundamental Outcomes")
        cols = st.columns(2)
        with cols[0]:
            st.metric("Actual Revenue (M)", f"{outcome.get('actual_revenue', 'N/A')}")
        with cols[1]:
            st.metric("Earnings Surprise", f"{outcome.get('earnings_surprise', 'N/A')}")
    else:
        st.info("No outcome data available.")
    
    # Export button
    if st.button("Export Transcript Data to CSV"):
        export_transcript(transcript_id)


def export_transcript(transcript_id: str):
    import pandas as pd
    from io import StringIO
    
    sentences = data_loader.get_sentences(transcript_id)
    risk_scores = data_loader.get_risk_scores(transcript_id)
    
    df_sentences = pd.DataFrame(sentences)
    df_risk = pd.DataFrame(risk_scores)
    
    output = StringIO()
    
    if not df_sentences.empty and not df_risk.empty:
        df_combined = df_sentences.merge(df_risk, on='sentence_id', how='left')
        df_combined.to_csv(output, index=False)
    
    st.download_button(
        label="Download CSV",
        data=output.getvalue(),
        file_name=f"transcript_{transcript_id}.csv",
        mime="text/csv"
    )
    st.success("CSV export ready! Click the download button above.")


if __name__ == "__main__":
    main()
