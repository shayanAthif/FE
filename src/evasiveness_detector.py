"""
Evasiveness Detector — Phase 2

Analyses Q&A pairs (analyst question → executive answer) for signs of
indirect, deflecting, or topic-avoiding responses.

Features computed:
  1. qa_tfidf_similarity      — TF-IDF cosine similarity (baseline)
  2. topic_overlap            — fraction of question nouns/entities found in answer
  3. keyword_overlap          — simple token-level Jaccard
  4. direct_answer_indicators — count of direct-answer linguistic signals
  5. deflection_indicators    — count of deflection phrases
  6. hedging_density          — passed in from HedgingDetector output
  7. generic_language_score   — density of boilerplate phrases
  8. answer_length_ratio      — answer tokens / question tokens
  9. numeric_in_question      — boolean: does the question ask for a number?
  10. numeric_in_answer        — boolean: does the answer contain a number?
  11. numeric_response_score   — 0-100 score rewarding numeric answers to numeric Qs

Final output:
  evasive_score               — composite 0-100 score (higher = more evasive)

All sub-scores are stored for explainability.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import get_config
from src.logger import get_logger

logger = get_logger("evasiveness_detector")

# ──────────────────────────────────────────────────────────────────────────────
# Phrase lists
# ──────────────────────────────────────────────────────────────────────────────

_DIRECT_ANSWER_INDICATORS = [
    r"\b\d+[\.,]?\d*\s*(?:percent|%|million|billion|billion dollars|dollars|bps|basis points)\b",
    r"\b(?:specifically|exactly|precisely|the answer is|yes|no|we did|we will|we have)\b",
    r"\b(?:our guidance is|our target is|we expect)\s+\d",
    r"\byes\b",
    r"\bno\b",
    r"\bwe achieved\b",
    r"\bwe reported\b",
    r"\bwe delivered\b",
    r"\bwe completed\b",
    r"\bwe reduced\b",
    r"\bwe increased\b",
    r"\bwe grew\b",
]

_DEFLECTION_INDICATORS = [
    r"\bas you know\b",
    r"\bthe important thing is\b",
    r"\bi think the key (?:point|thing|message) is\b",
    r"\bwhat i would say is\b",
    r"\blet me step back\b",
    r"\bmore broadly\b",
    r"\bthe bigger picture\b",
    r"\bwe are focused on the long[- ]term\b",
    r"\blong[- ]term value\b",
    r"\bwe are not (?:in a position|prepared|able) to\b",
    r"\bwe (?:are not |don't |cannot )(?:comment|provide guidance|discuss)\b",
    r"\bwe will (?:update|provide) (?:you|more information|further details) later\b",
    r"\bwe (?:continue to|are still) (?:evaluate|monitor|assess|review|work through)\b",
    r"\bit (?:is|'s) too early\b",
    r"\bremains to be seen\b",
    r"\bwe are navigating\b",
    r"\bwe are working through\b",
    r"\bwe prefer not to\b",
    r"\bnot in a position to\b",
    r"\btake it one quarter at a time\b",
]

_GENERIC_LANGUAGE = [
    r"\blong[- ]term (?:value|growth|shareholder value|strategy)\b",
    r"\bdriving (?:growth|value|profitability|efficiency)\b",
    r"\bremain(?:s)? committed\b",
    r"\bremain(?:s)? focused\b",
    r"\bdeliver(?:ing)? (?:value|results|growth)\b",
    r"\bcreating value\b",
    r"\boptimizing (?:for |our )?(?:performance|growth|efficiency|returns)\b",
    r"\bbest position(?:ed)? to\b",
    r"\bstrongest (?:portfolio|team|position|pipeline)\b",
    r"\bworld[- ]class\b",
    r"\bstrategic (?:priorities|initiatives|investments|transformation)\b",
    r"\bdigital (?:transformation|journey|strategy)\b",
    r"\boperational excellence\b",
    r"\bmomentum\b",
    r"\bexecuting (?:well|on our|against our)\b",
]

_NUMERIC_PATTERN = re.compile(
    r"\b\d+[\.,]?\d*\s*"
    r"(?:percent|%|million|billion|thousand|bps|basis points|dollars|"
    r"cents|x|times|quarters|years)?\b",
    re.IGNORECASE,
)

_QUESTION_NUMERIC_KEYWORDS = re.compile(
    r"\b(?:how much|how many|what (?:is|was|are|were) the|"
    r"what (?:percent|percentage|growth|margin|revenue|earnings|guidance)|"
    r"what (?:number|level|amount|range|target)|"
    r"quantify|specific(?:ally)?|exact(?:ly)?|\d+)\b",
    re.IGNORECASE,
)


def _compile_patterns(phrase_list: List[str]) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in phrase_list]


_DIRECT_PATTERNS = _compile_patterns(_DIRECT_ANSWER_INDICATORS)
_DEFLECTION_PATTERNS = _compile_patterns(_DEFLECTION_INDICATORS)
_GENERIC_PATTERNS = _compile_patterns(_GENERIC_LANGUAGE)


class EvasivenessDetector:
    """
    Computes evasiveness features and composite score for Q&A pairs.
    """

    def __init__(self):
        # TF-IDF vectoriser (fitted per-call or lazily)
        self._tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )
        logger.debug("EvasivenessDetector initialised.")

    # ─────────────────────────────────────────────────────────────────────
    # Feature extractors
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _count_patterns(text: str, patterns: List[re.Pattern]) -> int:
        count = 0
        for p in patterns:
            count += len(p.findall(text))
        return count

    @staticmethod
    def _jaccard_overlap(text_a: str, text_b: str) -> float:
        tokens_a = set(re.findall(r"\b\w{3,}\b", text_a.lower()))
        tokens_b = set(re.findall(r"\b\w{3,}\b", text_b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        inter = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(inter) / len(union)

    @staticmethod
    def _noun_overlap(question: str, answer: str) -> float:
        """Simple proxy for topic overlap: content words in Q found in A."""
        q_tokens = set(re.findall(r"\b[A-Za-z]{4,}\b", question.lower()))
        a_tokens = set(re.findall(r"\b[A-Za-z]{4,}\b", answer.lower()))
        # Remove stopwords approximately
        stopwords = {
            "that", "this", "with", "from", "have", "been", "will", "your",
            "what", "when", "where", "which", "would", "could", "should",
            "about", "their", "there", "they", "were", "then", "than",
            "into", "also", "just", "because", "more", "some", "like",
        }
        q_content = q_tokens - stopwords
        if not q_content:
            return 0.5  # neutral if no content tokens
        found = q_content & a_tokens
        return len(found) / len(q_content)

    @staticmethod
    def _has_numeric(text: str) -> bool:
        return bool(_NUMERIC_PATTERN.search(text))

    @staticmethod
    def _question_asks_for_number(question: str) -> bool:
        return bool(_QUESTION_NUMERIC_KEYWORDS.search(question))

    def _tfidf_similarity(self, question: str, answer: str) -> float:
        """TF-IDF cosine similarity between question and answer."""
        try:
            vecs = self._tfidf.fit_transform([question, answer])
            sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
            return float(sim)
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────────────────────────────
    # Main analysis method
    # ─────────────────────────────────────────────────────────────────────

    def analyse(
        self,
        question: str,
        answer: str,
        hedging_density: float = 0.0,
    ) -> Dict:
        """
        Analyse a single Q&A pair for evasiveness.

        Parameters
        ----------
        question       : analyst question text
        answer         : executive answer text
        hedging_density: hedge_density from HedgingDetector on the answer

        Returns
        -------
        Dict with all evasiveness features and evasive_score (0-100).
        """
        if not question or not answer:
            return self._empty_result()

        # Feature 1: TF-IDF similarity
        qa_tfidf_similarity = self._tfidf_similarity(question, answer)

        # Feature 2: Topic overlap (noun/content words)
        topic_overlap = self._noun_overlap(question, answer)

        # Feature 3: Keyword overlap (Jaccard)
        keyword_overlap = self._jaccard_overlap(question, answer)

        # Feature 4: Direct answer indicators in answer
        direct_count = self._count_patterns(answer, _DIRECT_PATTERNS)
        direct_indicator_score = min(direct_count / 3.0, 1.0)  # 0-1

        # Feature 5: Deflection indicators in answer
        deflection_count = self._count_patterns(answer, _DEFLECTION_PATTERNS)
        deflection_indicator_score = min(deflection_count / 3.0, 1.0)  # 0-1

        # Feature 7: Generic language density
        generic_count = self._count_patterns(answer, _GENERIC_PATTERNS)
        answer_words = max(1, len(answer.split()))
        generic_language_score = round(min(generic_count / answer_words * 10, 1.0) * 100, 2)

        # Feature 8: Answer length ratio
        q_words = max(1, len(question.split()))
        answer_length_ratio = round(answer_words / q_words, 2)

        # Feature 9-10: Numeric signals
        question_asks_numeric = self._question_asks_for_number(question)
        answer_has_numeric = self._has_numeric(answer)

        # Feature 11: Numeric response score (0-100)
        #   If question asks for a number and answer provides one → low evasion contribution
        #   If question asks for a number and answer doesn't → high evasion contribution
        if question_asks_numeric:
            numeric_response_score = 0.0 if answer_has_numeric else 100.0
        else:
            numeric_response_score = 0.0  # No numeric expectation

        # ── Composite evasive_score ──────────────────────────────────────
        # Low similarity → high evasion; high overlap → low evasion
        similarity_evasion = (1.0 - qa_tfidf_similarity) * 100
        topic_evasion = (1.0 - topic_overlap) * 100
        keyword_evasion = (1.0 - keyword_overlap) * 100
        direct_bonus = (1.0 - direct_indicator_score) * 100  # less direct = more evasive
        hedging_contribution = min(hedging_density * 200, 100)  # hedging boosts evasion

        evasive_raw = (
            0.20 * similarity_evasion
            + 0.20 * topic_evasion
            + 0.10 * keyword_evasion
            + 0.10 * direct_bonus
            + 0.15 * (deflection_indicator_score * 100)
            + 0.10 * hedging_contribution
            + 0.05 * generic_language_score
            + 0.10 * numeric_response_score
        )
        evasive_score = round(min(max(evasive_raw, 0.0), 100.0), 2)

        return {
            # Similarity features
            "qa_tfidf_similarity": round(qa_tfidf_similarity, 4),
            "topic_overlap": round(topic_overlap, 4),
            "keyword_overlap": round(keyword_overlap, 4),
            # Direct vs deflection
            "direct_answer_count": direct_count,
            "direct_indicator_score": round(direct_indicator_score * 100, 2),
            "deflection_count": deflection_count,
            "deflection_indicator_score": round(deflection_indicator_score * 100, 2),
            # Generic language
            "generic_language_count": generic_count,
            "generic_language_score": generic_language_score,
            # Length
            "answer_length_words": answer_words,
            "question_length_words": q_words,
            "answer_length_ratio": answer_length_ratio,
            # Numeric signals
            "question_asks_numeric": question_asks_numeric,
            "answer_has_numeric": answer_has_numeric,
            "numeric_response_score": numeric_response_score,
            # Passed-through hedging
            "hedging_density_answer": round(hedging_density, 4),
            # Final score
            "evasive_score": evasive_score,
        }

    def analyse_batch(
        self,
        qa_pairs: List[Tuple[str, str]],
        hedging_densities: Optional[List[float]] = None,
    ) -> List[Dict]:
        """
        Analyse a batch of (question, answer) tuples.

        Parameters
        ----------
        qa_pairs         : list of (question_text, answer_text)
        hedging_densities: optional per-pair hedging density values

        Returns
        -------
        List of result dicts.
        """
        if hedging_densities is None:
            hedging_densities = [0.0] * len(qa_pairs)
        return [
            self.analyse(q, a, hd)
            for (q, a), hd in zip(qa_pairs, hedging_densities)
        ]

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "qa_tfidf_similarity": 0.0,
            "topic_overlap": 0.5,
            "keyword_overlap": 0.0,
            "direct_answer_count": 0,
            "direct_indicator_score": 0.0,
            "deflection_count": 0,
            "deflection_indicator_score": 0.0,
            "generic_language_count": 0,
            "generic_language_score": 0.0,
            "answer_length_words": 0,
            "question_length_words": 0,
            "answer_length_ratio": 1.0,
            "question_asks_numeric": False,
            "answer_has_numeric": False,
            "numeric_response_score": 0.0,
            "hedging_density_answer": 0.0,
            "evasive_score": 0.0,
        }

