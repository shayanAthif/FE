"""
Hedging / Uncertainty Detector — Phase 2

Rule-based + lexical hedging detector that computes:
  - hedge_count          : raw count of hedging matches in sentence
  - hedge_density        : hedge_count / word_count
  - uncertainty_score    : 0-100 normalised score based on hedge density + phrase signals
  - modal_verb_density   : modal verbs / word_count
  - qualifier_density    : qualifier words / word_count
  - hedging_score        : composite 0-100 final score

Dictionary is fully configurable via config.yaml (HedgingLexiconConfig).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
from src.config import get_config, HedgingLexiconConfig
from src.logger import get_logger

logger = get_logger("hedging_detector")

# ──────────────────────────────────────────────────────────────────────────────
# Static / supplemental lexicons (added to config-driven ones)
# ──────────────────────────────────────────────────────────────────────────────

_EXTRA_MODALS: List[str] = [
    "may", "might", "could", "would", "should",
    "can", "will",  # weaker hedges but captured
]

_QUALIFIER_WORDS: List[str] = [
    "approximately", "roughly", "about", "around", "nearly", "somewhat",
    "fairly", "rather", "generally", "typically", "usually", "often",
    "occasionally", "sometimes", "largely", "mostly", "primarily",
    "potentially", "possibly", "likely", "unlikely", "probably",
    "perhaps", "maybe", "conceivably",
]

_UNCERTAINTY_PHRASES: List[str] = [
    "we believe", "we expect", "we anticipate", "we estimate",
    "we think", "we feel", "we hope", "we project",
    "it is difficult", "it is hard to say", "it is too early",
    "too early to tell", "too early to say",
    "we are evaluating", "we continue to monitor", "we continue to evaluate",
    "we are monitoring", "we are assessing", "we are reviewing",
    "cannot comment", "can't comment", "no comment", "not in a position",
    "we remain cautious", "cautiously optimistic", "proceed with caution",
    "depending on", "subject to", "contingent on", "assuming that",
    "if conditions", "if the market", "if demand", "if macroeconomic",
    "uncertain", "uncertainty", "unpredictable", "volatility",
    "range of outcomes", "remains to be seen", "yet to be determined",
    "we will see", "hard to predict", "difficult to predict",
]

_DEFLECTION_PHRASES: List[str] = [
    "i can't comment", "cannot comment on that",
    "it is too early", "it's too early",
    "we are monitoring", "we continue to evaluate",
    "we will provide information later", "we will update you",
    "we are focused on the long term", "long-term value",
    "as you know", "the important thing is", "i think the key point is",
    "what i would say is", "let me step back", "the bigger picture",
    "we are working through", "we are working on", "we are navigating",
    "we will cross that bridge", "we will take it one quarter at a time",
    "we are not providing guidance", "we are not in a position to provide",
    "we are not prepared to", "we prefer not to",
]


class HedgingDetector:
    """
    Detects hedging and uncertainty language in financial earnings-call text.

    All dictionaries are merged from the AppConfig at construction time,
    so changes to config.yaml are automatically picked up.
    """

    def __init__(self, lexicon: Optional[HedgingLexiconConfig] = None):
        cfg = get_config()
        lex = lexicon or cfg.risk.hedging_lexicon

        # Build merged sets / lists
        self._modals: List[str] = sorted(
            set(_EXTRA_MODALS + lex.modals), key=len, reverse=True
        )
        self._qualifiers: List[str] = sorted(
            set(_QUALIFIER_WORDS + lex.uncertainty), key=len, reverse=True
        )
        self._uncertainty_phrases: List[str] = sorted(
            set(_UNCERTAINTY_PHRASES + lex.hedging_phrases), key=len, reverse=True
        )
        self._deflection_phrases: List[str] = sorted(
            set(_DEFLECTION_PHRASES + lex.deflection_phrases), key=len, reverse=True
        )

        # Pre-compile regex patterns (case-insensitive, word-boundary aware)
        self._modal_patterns = [
            re.compile(r"\b" + re.escape(m) + r"\b", re.IGNORECASE)
            for m in self._modals
        ]
        self._qualifier_patterns = [
            re.compile(r"\b" + re.escape(q) + r"\b", re.IGNORECASE)
            for q in self._qualifiers
        ]
        self._uncertainty_patterns = [
            re.compile(re.escape(p), re.IGNORECASE)
            for p in self._uncertainty_phrases
        ]
        self._deflection_patterns = [
            re.compile(re.escape(p), re.IGNORECASE)
            for p in self._deflection_phrases
        ]

        logger.debug(
            f"HedgingDetector initialised: {len(self._modals)} modals, "
            f"{len(self._qualifiers)} qualifiers, "
            f"{len(self._uncertainty_phrases)} uncertainty phrases, "
            f"{len(self._deflection_phrases)} deflection phrases"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _word_count(text: str) -> int:
        return max(1, len(text.split()))

    def _count_matches(self, text: str, patterns: list) -> Tuple[int, List[str]]:
        """Return (total_match_count, list_of_matched_terms)."""
        count = 0
        matched: List[str] = []
        for pat in patterns:
            hits = pat.findall(text)
            if hits:
                count += len(hits)
                matched.extend(hits)
        return count, matched

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def analyse(self, text: str) -> Dict:
        """
        Analyse a sentence / segment for hedging and uncertainty signals.

        Returns a dict with all features needed for downstream scoring.
        """
        if not text or not text.strip():
            return self._empty_result()

        words = self._word_count(text)

        modal_count, modal_matches = self._count_matches(text, self._modal_patterns)
        qual_count, qual_matches = self._count_matches(text, self._qualifier_patterns)
        unc_count, unc_matches = self._count_matches(text, self._uncertainty_patterns)
        defl_count, defl_matches = self._count_matches(text, self._deflection_patterns)

        # Raw feature counts
        hedge_count = modal_count + qual_count + unc_count
        total_match_count = hedge_count + defl_count

        # Densities (per word)
        hedge_density = hedge_count / words
        modal_verb_density = modal_count / words
        qualifier_density = qual_count / words
        deflection_density = defl_count / words

        # Uncertainty score: weighted combination, clipped at 1.0 before scaling
        uncertainty_raw = (
            0.35 * min(modal_verb_density * 10, 1.0)
            + 0.35 * min(qualifier_density * 10, 1.0)
            + 0.30 * min(unc_count / 3, 1.0)  # phrase-level signal
        )
        uncertainty_score = round(min(uncertainty_raw, 1.0) * 100, 2)

        # Hedging score: composite of density + deflection bonus
        hedging_raw = (
            0.50 * min(hedge_density * 8, 1.0)
            + 0.25 * min(unc_count / 2, 1.0)
            + 0.25 * min(defl_count / 2, 1.0)
        )
        hedging_score = round(min(hedging_raw, 1.0) * 100, 2)

        return {
            # Counts
            "hedge_count": hedge_count,
            "modal_count": modal_count,
            "qualifier_count": qual_count,
            "uncertainty_phrase_count": unc_count,
            "deflection_count": defl_count,
            # Densities
            "hedge_density": round(hedge_density, 4),
            "modal_verb_density": round(modal_verb_density, 4),
            "qualifier_density": round(qualifier_density, 4),
            "deflection_density": round(deflection_density, 4),
            # Normalised scores [0, 100]
            "uncertainty_score": uncertainty_score,
            "hedging_score": hedging_score,
            # Matched terms (for explainability)
            "matched_modals": modal_matches[:10],
            "matched_qualifiers": qual_matches[:10],
            "matched_uncertainty_phrases": unc_matches[:5],
            "matched_deflection_phrases": defl_matches[:5],
        }

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "hedge_count": 0,
            "modal_count": 0,
            "qualifier_count": 0,
            "uncertainty_phrase_count": 0,
            "deflection_count": 0,
            "hedge_density": 0.0,
            "modal_verb_density": 0.0,
            "qualifier_density": 0.0,
            "deflection_density": 0.0,
            "uncertainty_score": 0.0,
            "hedging_score": 0.0,
            "matched_modals": [],
            "matched_qualifiers": [],
            "matched_uncertainty_phrases": [],
            "matched_deflection_phrases": [],
        }

    def analyse_batch(self, texts: List[str]) -> List[Dict]:
        """Analyse a list of texts; returns one result dict per text."""
        return [self.analyse(t) for t in texts]

