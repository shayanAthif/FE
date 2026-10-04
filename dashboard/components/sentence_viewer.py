import streamlit as st
from typing import Dict, List, Optional


def sentence_viewer(
    sentences: List[Dict],
    highlighted_ids: Optional[List[str]] = None,
    max_display: int = 50,
):
    """
    Display sentences in transcript order with optional highlighting.
    
    Args:
        sentences: List of sentence dicts with 'sentence_id', 'text', 'segment_id', etc.
        highlighted_ids: List of sentence_ids to highlight in red
        max_display: Maximum sentences to show (for performance)
    """
    if not sentences:
        st.info("No sentences found for this transcript.")
        return
    
    display_count = 0
    for sentence in sentences[:max_display]:
        sentence_id = sentence.get('sentence_id', 'unknown')
        text = sentence.get('text', '')
        
        # Highlight if in highlighted list
        is_highlighted = highlighted_ids and sentence_id in highlighted_ids
        
        with st.container():
            if is_highlighted:
                st.markdown(
                    f'<div style="background-color: #fff3f3; border-left: 4px solid #FF4B4B; padding: 10px;">'
                    f'<strong style="color: #FF4B4B;">Sentence {display_count + 1}</strong><br>'
                    f'{text}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="padding: 8px;">'
                    f'<span style="color: #666;">Sentence {display_count + 1}</span><br>'
                    f'{text}</div>',
                    unsafe_allow_html=True
                )
            display_count += 1
    
    if len(sentences) > max_display:
        st.caption(f"... and {len(sentences) - max_display} more sentences (truncated for performance)")


def explainable_sentence(
    sentence: Dict,
    risk_factors: Dict,
    confidence: float = 1.0,
):
    """
    Display a single sentence with risk factor explainability.
    
    Args:
        sentence: Sentence dict with 'text' and metadata
        risk_factors: Dict with 'hedging', 'evasiveness', 'tone_shift' scores
        confidence: Confidence in risk assessment (0-1)
    """
    st.markdown("---")
    
    # Original sentence
    st.markdown(f"**{sentence.get('text', '')}**")
    
    # Risk scores
    cols = st.columns(3)
    with cols[0]:
        st.metric("Hedging", f"{risk_factors.get('hedging', 0):.2f}")
    with cols[1]:
        st.metric("Evasiveness", f"{risk_factors.get('evasiveness', 0):.2f}")
    with cols[2]:
        st.metric("Tone Shift", f"{risk_factors.get('tone_shift', 0):.2f}")
    
    # Detected phrases (if available)
    if 'detected_phrases' in sentence:
        st.markdown("**Detected phrases:**")
        for phrase in sentence['detected_phrases']:
            st.markdown(f"- {phrase}")
