"""Self-learning ticker lookup for entity extraction.

Loads a comprehensive mapping from ``nasdaq_tickers.json`` (generated from the
official NASDAQ-listed symbols feed plus manual supplements).  On cache miss the
tool falls back to Google Search, extracts the ticker from the search summary,
and appends the new entry directly to ``nasdaq_tickers.json`` so future runs are
instant.

Keys are lowercased for case-insensitive matching.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from google.adk.tools.function_tool import FunctionTool

logger = logging.getLogger(__name__)

_NASDAQ_JSON_PATH = Path(__file__).resolve().parent / "nasdaq_tickers.json"


def _load_ticker_map() -> dict[str, str]:
    """Load the ticker map from ``nasdaq_tickers.json``."""
    if _NASDAQ_JSON_PATH.exists():
        try:
            data = json.loads(_NASDAQ_JSON_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k).lower(): str(v).upper() for k, v in data.items()}
        except Exception:
            logger.warning(
                "Could not load %s; falling back to empty map.",
                _NASDAQ_JSON_PATH,
            )
    return {}


def _save_ticker_map(ticker_map: dict[str, str]) -> None:
    """Persist the full ticker map back to ``nasdaq_tickers.json``."""
    try:
        _NASDAQ_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        _NASDAQ_JSON_PATH.write_text(
            json.dumps(ticker_map, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        logger.warning(
            "Could not write ticker map to %s", _NASDAQ_JSON_PATH, exc_info=True,
        )


TICKER_MAP: dict[str, str] = _load_ticker_map()

_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")


def _extract_ticker_from_summary(summary: str, company_name: str) -> str | None:
    """Try to extract a ticker symbol from a Google Search summary string."""
    if not summary:
        return None
    patterns = [
        re.compile(
            r"\b(?i:NYSE|NASDAQ|AMEX|TSX|LSE)[:\s]+([A-Z]{1,5})\b",
        ),
        re.compile(r"\(([A-Z]{1,5})\)"),
        re.compile(
            r"(?i:ticker|symbol|stock|trades?(?:\s+(?:as|under)))\s*[:\s]?\s*\(?([A-Z]{1,5})\b\)?",
        ),
    ]
    for pat in patterns:
        m = pat.search(summary)
        if m:
            return m.group(1).upper()
    candidates = _TICKER_RE.findall(summary)
    noise = {
        "THE", "AND", "FOR", "INC", "LTD", "LLC", "CO", "CORP",
        "NYSE", "NASDAQ", "AMEX", "USA", "USD", "OR", "TWO", "A",
        "IS", "IT", "AS", "AT", "BY", "OF", "ON", "TO", "IN", "NO",
        "TICKE", "STOCK", "SYMBO", "PRICE", "SHARE", "TRADE",
        "E", "I", "S",
    }
    filtered = [c for c in candidates if c not in noise and len(c) >= 1]
    if len(filtered) == 1:
        return filtered[0]
    return None


def _build_norm_index(cache: dict[str, str]) -> list[tuple[str, str, str]]:
    """Pre-compute normalized keys for fuzzy contains matching.

    Returns a list of ``(norm_key, original_key, ticker)`` sorted longest-first
    so the first match is always the best (longest) match.
    """
    items: list[tuple[str, str, str]] = []
    for k, v in cache.items():
        nk = _norm_for_match(k)
        if len(nk) >= MIN_CONTAINS_KEY_LEN:
            items.append((nk, k, v))
    items.sort(key=lambda t: len(t[0]), reverse=True)
    return items


def build_ticker_cache() -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """Return a mutable copy of the ticker map and a precomputed fuzzy index."""
    cache = dict(TICKER_MAP)
    return cache, _build_norm_index(cache)


_QUERY_NOISE_RE = re.compile(
    r"\s+(?:stock|ticker|symbol|price|share|nyse|nasdaq|quote)[\s\w]*$",
    re.IGNORECASE,
)
_STRIP_SPECIAL_RE = re.compile(r"[^a-z0-9&\s]+")
_COLLAPSE_SPACE_RE = re.compile(r"\s+")

MIN_CONTAINS_KEY_LEN = 3


def _sanitize_company_name(raw: str) -> str:
    """Strip search-query noise the LLM sometimes appends to company names."""
    clean = raw.strip().strip('"').strip("'").strip()
    clean = _QUERY_NOISE_RE.sub("", clean).strip()
    return clean


def _norm_for_match(name: str) -> str:
    """Normalize for fuzzy matching: lowercase, strip special chars, collapse spaces."""
    n = name.lower()
    n = _STRIP_SPECIAL_RE.sub(" ", n)
    n = _COLLAPSE_SPACE_RE.sub(" ", n).strip()
    return n


def _lookup_in_map(
    company_name: str,
    cache: dict[str, str],
    norm_index: list[tuple[str, str, str]],
) -> str | None:
    """Case-insensitive lookup: exact match first, then longest-key-contains.

    Normalizes both the query and cache keys by stripping special characters
    so that e.g. "Yahoo! Inc" matches the cache key "yahoo" -> YHOO.
    """
    key = _sanitize_company_name(company_name).lower().rstrip(".")
    if not key:
        return None

    # 1) Direct exact match
    ticker = cache.get(key)
    if ticker:
        return ticker

    # 2) Try with/without common corporate suffixes
    for suffix in (" inc", " corp", " co", " ltd", " group", " holdings"):
        ticker = cache.get(key + suffix) or cache.get(key.removesuffix(suffix))
        if ticker:
            return ticker

    # 3) Fuzzy contains: normalize the query, scan the pre-sorted index
    #    (longest-first) so the first hit is the best match.
    query_norm = _norm_for_match(key)
    if len(query_norm) < MIN_CONTAINS_KEY_LEN:
        return None

    for nk, _orig_key, ticker in norm_index:
        if nk in query_norm:
            return ticker

    return None


def create_ticker_lookup_tool() -> FunctionTool:
    """Create an ADK FunctionTool that checks the ticker map then falls back to
    Google Search.  New discoveries are appended to ``nasdaq_tickers.json``.
    """
    cache, initial_index = build_ticker_cache()
    state = {"norm_index": initial_index}

    async def lookup_ticker(company_name: str) -> dict[str, Any]:
        """Look up the stock ticker symbol for a company.

        Checks a local cache first. If not found, searches Google to resolve
        the ticker and saves it for future lookups.

        Parameters
        ----------
        company_name : str
            The company name to look up (e.g. "Apple", "Goldman Sachs").

        Returns
        -------
        dict
            Result with keys:

            - **status** (str): ``"found"`` or ``"not_found"``
            - **ticker** (str): The ticker symbol (only present when found)
            - **source** (str): ``"cache"`` or ``"search"``
            - **company_name** (str): The queried company name
        """
        ticker = _lookup_in_map(company_name, cache, state["norm_index"])
        if ticker:
            return {
                "status": "found",
                "ticker": ticker,
                "source": "cache",
                "company_name": company_name,
            }

        clean_name = _sanitize_company_name(company_name)
        logger.info("Ticker cache miss for '%s'; falling back to Google Search.", clean_name)
        try:
            from aieng.agent_evals.tools.search import google_search as _google_search

            result = await _google_search(f'"{clean_name}" stock ticker symbol')
            summary = result.get("summary", "")
            found_ticker = _extract_ticker_from_summary(summary, clean_name)

            if found_ticker:
                key = clean_name.lower()
                cache[key] = found_ticker
                TICKER_MAP[key] = found_ticker
                state["norm_index"] = _build_norm_index(cache)
                _save_ticker_map(cache)
                logger.info(
                    "Resolved '%s' -> %s via search; saved to nasdaq_tickers.json.",
                    company_name,
                    found_ticker,
                )
                return {
                    "status": "found",
                    "ticker": found_ticker,
                    "source": "search",
                    "company_name": company_name,
                }
        except Exception:
            logger.warning("Google Search fallback failed for '%s'.", company_name, exc_info=True)

        return {"status": "not_found", "company_name": company_name}

    return FunctionTool(func=lookup_ticker)


__all__ = [
    "TICKER_MAP",
    "build_ticker_cache",
    "create_ticker_lookup_tool",
]
