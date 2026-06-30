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
