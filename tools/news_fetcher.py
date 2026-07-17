import asyncio
import time
from datetime import datetime, timezone

import httpx

from config.settings import NEWSAPI_KEY
from tools.canary_tokens import tag_article

NEWSDATA_URL = "https://newsdata.io/api/1/news"
GDELT_URL    = "https://api.gdeltproject.org/api/v2/doc/doc"

GDELT_PARAMS = {
    "mode":       "artlist",
    "format":     "json",
    "maxrecords": "25",
    "sort":       "datedesc",
}

# ── Search-query builder ────────────────────────────────────────────────────────
# The user's question ROUTES; it must never FETCH. A conversational sentence
# ("what is the status of corridors and supplies to india") passed as the search
# string returns zero relevant articles — NewsData full-text-matches the words and
# GDELT ANDs them — so GRI scores every corridor at baseline and the board reports
# "routine" no matter what is happening in the world. Searches are therefore built
# from a fixed corridor vocabulary; the user's phrasing only narrows WHICH
# corridors to search, never supplies the search terms itself.

# corridor_id → (detection aliases matched against the lowercased user query,
#                quoted search phrases sent to the news APIs)
_CORRIDOR_SEARCH: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "strait_of_hormuz":  (("hormuz", "persian gulf", "iran"),
                          ('"Hormuz"',)),
    "suez_canal":        (("suez", "egypt"),
                          ('"Suez"',)),
    "bab_el_mandeb":     (("bab el-mandeb", "bab_el_mandeb", "bab al-mandab",
                           "red sea", "yemen", "houthi"),
                          ('"Bab el-Mandeb"', '"Red Sea"')),
    "malacca_strait":    (("malacca",),
                          ('"Malacca"',)),
    "turkish_straits":   (("turkish strait", "turkish_straits", "bosphorus",
                           "bosporus"),
                          ('"Bosphorus"', '"Turkish Straits"')),
    "danish_straits":    (("danish strait", "danish_straits"),
                          ('"Danish Straits"',)),
    "cape_of_good_hope": (("cape of good hope", "cape_of_good_hope"),
                          ('"Cape of Good Hope"',)),
    "panama_canal":      (("panama",),
                          ('"Panama Canal"',)),
}

# Broad monitoring sweep when the query names no corridor (status questions, the
# background twin loop). Kept under ~100 chars so NewsData never rejects it; the
# OR group is parenthesized because GDELT requires OR'd statements in parens.
_DEFAULT_SEARCH_QUERY = ('("Strait of Hormuz" OR "Suez Canal" OR "Red Sea" '
                         'OR "oil tanker" OR "crude oil")')


def build_search_query(user_query: str) -> str:
    """Turn a conversational query into a corridor-keyword news search.

    Corridors named (or hinted at — 'iran', 'houthi', …) in the query are searched
    specifically; a query naming none gets the broad energy-chokepoint sweep. The
    raw user text is never used as a search term.
    """
    q = (user_query or "").lower()
    phrases: list[str] = []
    for aliases, search in _CORRIDOR_SEARCH.values():
        if any(a in q for a in aliases):
            phrases.extend(p for p in search if p not in phrases)
    if not phrases:
        return _DEFAULT_SEARCH_QUERY
    if len(phrases) == 1:
        return phrases[0]
    return "(" + " OR ".join(phrases) + ")"


async def _fetch_newsapi(client: httpx.AsyncClient, query: str, api_key: str) -> list[dict]:
    params = {
        "q":        query,
        "apikey":   api_key,
        "language": "en",
        "size":     10,
    }
    r = await client.get(NEWSDATA_URL, params=params, timeout=10)
    r.raise_for_status()
    articles = r.json().get("results", [])
    return [
        {
            "title":        a.get("title", ""),
            "url":          a.get("link", ""),
            "source":       a.get("source_id", "unknown"),
            "published_at": a.get("pubDate", ""),
            "description":  a.get("description", ""),
            "origin":       "newsdata",
        }
        for a in articles
    ]


async def _fetch_gdelt(client: httpx.AsyncClient, query: str) -> list[dict]:
    params = {**GDELT_PARAMS, "query": query}
    r = await client.get(GDELT_URL, params=params, timeout=10)
    r.raise_for_status()
    articles = r.json().get("articles", [])
    return [
        {
            "title":        a.get("title", ""),
            "url":          a.get("url", ""),
            "source":       a.get("domain", "unknown"),
            "published_at": a.get("seendate", ""),
            "description":  "",
            "origin":       "gdelt",
        }
        for a in articles
    ]


async def _fetch_all(query: str, api_key: str) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            _fetch_newsapi(client, query, api_key),
            _fetch_gdelt(client, query),
            return_exceptions=True,
        )

    articles: list[dict] = []
    for source_name, result in zip(("newsapi", "gdelt"), results):
        if isinstance(result, Exception):
            errors.append(f"{source_name}: {type(result).__name__}: {result}")
        else:
            articles.extend(result)

    return articles, errors


def fetch_news(query: str, api_key: str = NEWSAPI_KEY, staleness_limit: int = 3600) -> dict:
    fetched_at = datetime.now(timezone.utc)
    t0 = time.monotonic()

    articles, errors = asyncio.run(_fetch_all(query, api_key))

    tagged = [tag_article(a) for a in articles]

    trust_scores = [a["trust_score"] for a in tagged] if tagged else [0.0]
    trust_avg    = round(sum(trust_scores) / len(trust_scores), 4)
    low_trust    = sum(1 for a in tagged if not a["trusted"])

    elapsed = int(time.monotonic() - t0)

    if not errors:
        status = "ok"
    elif len(errors) == 2:
        status = "failed"
    else:
        status = "degraded"

    return {
        "tool":                     "news_fetcher",
        "status":                   status,
        "data":                     {"articles": tagged, "errors": errors},
        "source_trust_avg":         trust_avg,
        "low_trust_sources_flagged": low_trust,
        "retrieved_at":             fetched_at.isoformat(),
        "staleness_seconds":        elapsed,
    }
