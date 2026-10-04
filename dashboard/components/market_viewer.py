import streamlit as st
from typing import Dict, List, Optional
import pandas as pd


def market_reaction_card(
    market_data: Dict,
    ticker: str,
):
    """
    Display market reaction data for an earnings call.
    
    Args:
        market_data: Dict with 'return_1d', 'return_5d', 'return_10d', 'return_20d', etc.
        ticker: Company ticker symbol
    """
    st.markdown(f"### Market Reaction: {ticker}")
    
    if not market_data:
        st.warning("No market data available for this transcript.")
        return
    
    # Key metrics
    cols = st.columns(4)
    with cols[0]:
        _metric_card("1-Day Return", market_data.get('return_1d'), format_pct=True)
    with cols[1]:
        _metric_card("5-Day Return", market_data.get('return_5d'), format_pct=True)
    with cols[2]:
        _metric_card("10-Day Return", market_data.get('return_10d'), format_pct=True)
    with cols[3]:
        _metric_card("20-Day Return", market_data.get('return_20d'), format_pct=True)
    
    # Volatility metrics
    st.markdown("**Volatility (Annualized)**")
    vol_cols = st.columns(2)
    with vol_cols[0]:
        _metric_card("5-Day Volatility", market_data.get('volatility_5d'), format_pct=True)
    with vol_cols[1]:
        _metric_card("10-Day Volatility", market_data.get('volatility_10d'), format_pct=True)
    
    # Benchmark comparison
    if market_data.get('benchmark_return_5d') is not None:
        abnormal = market_data.get('abnormal_return_5d', 0)
        st.metric(
            "Abnormal 5-Day Return",
            f"{abnormal:.2%}",
            help="Return relative to S&P 500 benchmark"
        )
    
    # Price movement
    price_before = market_data.get('price_before')
    price_after = market_data.get('price_after')
    if price_before and price_after:
        price_change = ((price_after - price_before) / price_before) * 100
        st.metric(
            "Price Change",
            f"${price_after:.2f}",
            delta=f"{price_change:.2f}%",
            delta_color="inverse" if price_change < 0 else "normal"
        )


def _metric_card(label: str, value: Optional[float], format_pct: bool = False):
    """Display a metric in a card."""
    if value is None:
        st.write(f"{label}: N/A")
        return
    
    display_value = f"{value:.2%}" if format_pct else f"{value:.4f}"
    st.metric(label, display_value)


def event_timeline(events: List[Dict]):
    """
    Display an event timeline for a company.
    
    Args:
        events: List of event dicts with 'date', 'title', 'description', 'type'
    """
    if not events:
        st.info("No events to display.")
        return
    
    # Sort by date
    events_sorted = sorted(events, key=lambda x: x.get('date', ''))
    
    st.markdown("### Event Timeline")
    
    for event in events_sorted:
        _event_item(event)


def _event_item(event: Dict):
    """Display a single event item."""
    date = event.get('date', 'Unknown')
    title = event.get('title', 'Untitled')
    description = event.get('description', '')
    event_type = event.get('type', 'info')
    
    # Color by type
    type_colors = {
        'earnings': '#32CD32',
        'sec_filing': '#0066CC',
        'news': '#800080',
        'other': '#666666',
    }
    color = type_colors.get(event_type, '#666666')
    
    with st.container():
        st.markdown(
            f'<div style="border-left: 3px solid {color}; padding-left: 10px; margin-bottom: 15px;">'
            f'<strong style="color: {color};">{date}</strong><br>'
            f'<strong>{title}</strong><br>'
            f'{description}</div>',
            unsafe_allow_html=True
        )
