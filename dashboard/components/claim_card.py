import streamlit as st
from typing import Dict, List, Optional


def claim_card(
    claim: Dict,
    verification: Optional[Dict] = None,
    evidence: Optional[List[Dict]] = None,
    show_details: bool = True,
):
    """
    Display a claim with its verification status and evidence.
    
    Args:
        claim: Claim dict with 'claim_text', 'metric', 'value', etc.
        verification: Optional verification dict with 'status', 'confidence', 'reasoning'
        evidence: Optional list of evidence dicts
        show_details: Whether to show expanded evidence
    """
    st.markdown("### " + claim.get('claim_text', 'Claim'))
    
    # Basic claim info
    cols = st.columns(4)
    with cols[0]:
        st.write(f"**Metric:** {claim.get('metric', 'N/A')}")
    with cols[1]:
        st.write(f"**Value:** {claim.get('value', 'N/A')} {claim.get('unit', '')}")
    with cols[2]:
        st.write(f"**Period:** {claim.get('period', 'N/A')}")
    with cols[3]:
        st.write(f"**Direction:** {claim.get('direction', 'N/A')}")
    
    # Verification status
    if verification:
        status = verification.get('status', 'UNKNOWN')
        confidence = verification.get('confidence', 0)
        
        # Status badge with color
        status_colors = {
            'SUPPORTED': '#32CD32',
            'CONTRADICTED': '#FF4B4B',
            'PARTIALLY_SUPPORTED': '#FFD700',
            'NOT_VERIFIABLE': '#808080',
        }
        status_color = status_colors.get(status, '#808080')
        
        st.markdown(
            f'<div style="background-color: {status_color}20; border-left: 4px solid {status_color}; padding: 10px;">'
            f'<strong style="color: {status_color}; font-size: 1.2em;">{status}</strong> '
            f'(Confidence: {confidence:.1%})</div>',
            unsafe_allow_html=True
        )
        
        # Reasoning
        if show_details:
            st.markdown("**Reasoning:**")
            st.write(verification.get('reasoning', 'No reasoning provided'))
    
    # Evidence
    if evidence and show_details:
        with st.expander(f"Show evidence ({len(evidence)} sources)", expanded=False):
            for i, ev in enumerate(evidence, 1):
                _evidence_source(ev, prefix=f"Evidence {i}")
    
    return claim, verification


def _evidence_source(evidence: Dict, prefix: str = "Evidence"):
    """Display a single evidence source."""
    source_type = evidence.get('source_type', 'unknown')
    source_name = evidence.get('source_name', evidence.get('source_url', 'Unknown'))
    publication_date = evidence.get('publication_date', 'Unknown')
    relevance = evidence.get('relevance_score', 0)
    
    with st.container():
        st.markdown(f"**{prefix}: {source_name}** ({source_type})")
        st.caption(f"Date: {publication_date} | Relevance: {relevance:.2f}")
        
        # URL link
        url = evidence.get('source_url')
        if url:
            st.markdown(f"[View source]({url})")
        
        # Excerpt (if available)
        if evidence.get('text'):
            excerpt = evidence['text'][:500] + "..." if len(evidence['text']) > 500 else evidence['text']
            st.markdown(f"*\"{excerpt}\"*")
