"""
Financial News Evidence Provider — Phase 3

Retrieves external financial news articles related to companies and claims.
Uses news aggregation feeds (Google News RSS) with local disk caching to prevent
repeated network requests and handle rate limits gracefully.
"""

from __future__ import annotations

import email.utils
import hashlib
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import requests

from src.config import PROJECT_ROOT, get_config
from src.logger import get_logger
from src.providers.base import BaseNewsProvider, RetrievedEvidence

logger = get_logger("news_provider")

_CACHE_DIR = PROJECT_ROOT / "data" / "news_cache"

# High authority financial news domains/sources
_TIER2_SOURCES = {
    "reuters", "bloomberg", "wall street journal", "wsj", "cnbc",
    "financial times", "barron's", "marketwatch", "forbes", "yahoo finance",
    "associated press", "ap news", "investor's business daily",
}


class NewsProvider(BaseNewsProvider):
    """News provider with query building, RSS parsing, and disk caching."""

    def __init__(self, cache_dir: Optional[Path] = None, timeout: float = 8.0):
        self.cfg = get_config()
        self.cache_dir = cache_dir or _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        }

    def _cache_key(self, query: str, start: str, end: str) -> str:
        h = hashlib.md5(f"{query}_{start}_{end}".encode("utf-8")).hexdigest()
        return f"news_{h}.json"

    def search_news(
        self,
        ticker: str,
        company_name: Optional[str],
        query: str,
        start_date: str,
        end_date: str,
        limit: int = 5,
    ) -> List[RetrievedEvidence]:
        """
        Search for news articles related to ticker and claim topic in a date window.
        """
        search_term = f"{ticker} {query}".strip()
        cache_file = self.cache_dir / self._cache_key(search_term, start_date, end_date)

        # Check disk cache
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_items = json.load(f)
                    return [RetrievedEvidence(**item) for item in cached_items]
            except Exception:
                pass

        encoded_q = urllib.parse.quote(search_term)
        rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"

        results: List[RetrievedEvidence] = []
        try:
            r = requests.get(rss_url, headers=self.headers, timeout=self.timeout)
            if r.status_code == 200:
                root = ET.fromstring(r.content)
                items = root.findall(".//item")

                for it in items[:limit]:
                    title_elem = it.find("title")
                    link_elem = it.find("link")
                    pub_elem = it.find("pubDate")
                    desc_elem = it.find("description")

                    title = title_elem.text if title_elem is not None and title_elem.text else "News Article"
                    link = link_elem.text if link_elem is not None and link_elem.text else ""
                    desc = desc_elem.text if desc_elem is not None and desc_elem.text else ""

                    # Strip HTML from description snippet
                    snippet = re.sub(r"<[^>]+>", "", desc).strip()
                    if not snippet:
                        snippet = title

                    # Parse publication date to YYYY-MM-DD
                    pub_date = start_date
                    if pub_elem is not None and pub_elem.text:
                        try:
                            parsed_tuple = email.utils.parsedate(pub_elem.text)
                            if parsed_tuple:
                                pub_date = datetime(*parsed_tuple[:3]).strftime("%Y-%m-%d")
                        except Exception:
                            pass

                    # Extract source name from title (Google News formats as: "Title - Source")
                    source_name = "Financial Press"
                    if " - " in title:
                        parts = title.rsplit(" - ", 1)
                        source_name = parts[1].strip()

                    # Determine authority score
                    source_lower = source_name.lower()
                    if any(s in source_lower for s in _TIER2_SOURCES):
                        authority = 0.75  # Tier 2: High quality financial media
                    else:
                        authority = 0.50  # Tier 3: General media

                    # Simple keyword relevance score
                    q_words = set(query.lower().split())
                    t_words = set(title.lower().split())
                    overlap = len(q_words & t_words) / max(1, len(q_words))
                    relevance = round(min(0.60 + 0.40 * overlap, 1.0), 2)

                    evidence = RetrievedEvidence(
                        source_type="news",
                        source_name=source_name,
                        source_url=link,
                        publication_date=pub_date,
                        title=title,
                        text=snippet,
                        relevance_score=relevance,
                        authority_score=authority,
                    )
                    results.append(evidence)

                # Save cache
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump([vars(e) for e in results], f)

        except Exception as exc:
            logger.debug(f"News retrieval error for {ticker} '{query}': {exc}")

        return results
