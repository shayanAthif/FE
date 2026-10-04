import streamlit as st
from typing import Dict, Optional


def risk_score_card(
    title: str,
    score: float,
    max_score: float = 100.0,
    color_theme: str = "orange",
    description: Optional[str] = None,
    details: Optional[Dict] = None,
):
    """
    Display a risk score card with visual indicator.
    
    Args:
        title: Card title (e.g., 'Hedging Score', 'Evasiveness Score')
        score: Numeric risk score (0-100)
        max_score: Maximum possible score (default 100)
        color_theme: 'red', 'orange', 'yellow', 'green' for progress bar color
        description: Optional description text below title
        details: Optional dict of detail items to show below score
    """
    # Normalize score to 0-100 range
    normalized_score = min(max(score, 0), max_score)
    percentage = (normalized_score / max_score) * 100
    
    # Color mapping
    color_map = {
        "red": "#FF4B4B",
        "orange": "#FFA500",
        "yellow": "#FFD700",
        "green": "#32CD32",
    }
    bar_color = color_map.get(color_theme, "orange")
    
    # Determine risk level
    if percentage >= 70:
        risk_level = "HIGH RISK"
        risk_color = "#FF4B4B"
    elif percentage >= 40:
        risk_level = "ELEVATED"
        risk_color = "#FFA500"
    elif percentage >= 20:
        risk_level = "MODERATE"
        risk_color = "#FFD700"
    else:
        risk_level = "LOW"
        risk_color = "#32CD32"
    
    # Card header
    st.markdown(f"### {title}")
    if description:
        st.caption(description)
    
    # Score display
    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric(label="Score", value=f"{normalized_score:.1f}", delta=f"{percentage:.0f}%")
    with col2:
        st.metric(label="Risk Level", value=risk_level, delta=risk_level)
    
    # Progress bar
    st.progress(percentage / 100, text=f"Risk: {normalized_score:.1f} / {max_score:.0f}")
    
    # Details (if provided)
    if details:
        st.markdown("**Contributing factors:**")
        for key, value in details.items():
            st.write(f"- {key}: {value}")
    
    return normalized_score
