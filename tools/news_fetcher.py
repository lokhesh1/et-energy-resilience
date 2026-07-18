import asyncio
import copy
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx

from config.settings import NEWSAPI_KEY, NEWS_CACHE_TTL
from tools.canary_tokens import extract_domain, tag_article

NEWSDATA_URL = "https://newsdata.io/api/1/news"
GDELT_URL    = "https://api.gdeltproject.org/api/v2/doc/doc"
GNEWS_URL    = "https://news.google.com/rss/search"

GDELT_PARAMS = {
    "mode":       "artlist",
    "format":     "json",
    "maxrecords": "25",
    # hybridrel = GDELT's relevance+recency ranking. datedesc sampled the newest
    # 25 mentions — a volume-weighted long-tail firehose where tier-1 wires
    # almost never surface. Recency is kept via timespan.
    "sort":       "hybridrel",
    "timespan":   "7d",
}

_GNEWS_MAX_ARTICLES = 25

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

# Energy/shipping context ANDed onto every per-corridor GDELT search so
# '"Panama Canal"' doesn't return tourism pieces. GDELT: space = AND, ORs in parens.
_GDELT_CONTEXT = "(oil OR tanker OR crude OR shipping OR blockade)"

# GDELT throttles by IP (~1 request / 5 s courtesy limit — a parallel 8-corridor
# burst gets 429'd across the board), so the fan-out runs SEQUENTIALLY with this
# spacing. The result cache absorbs the latency: the background twin loop keeps
# it warm, so a user query normally pays zero GDELT round-trips.
_GDELT_REQUEST_SPACING = 5.0  # seconds between consecutive GDELT requests
_GDELT_TIMEOUT = 20           # GDELT can take >10 s to answer at all

# Circuit breaker: once GDELT answers 429 its per-IP block lingers for minutes,
# and every further request EXTENDS it — so on the first 429 the remaining
# fan-out is skipped and GDELT is left alone for a cooldown. Without this the
# twin loop's tick would keep the block alive forever.
_GDELT_COOLDOWN = 300.0       # seconds to back off after a 429
_GDELT_BLOCKED_UNTIL = 0.0    # monotonic deadline; module state


def _is_throttle(e: Exception) -> bool:
    return (isinstance(e, httpx.HTTPStatusError)
            and e.response is not None and e.response.status_code == 429)


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


def route_corridors(user_query: str) -> list[str]:
    """Corridor ids the query names (or hints at). Empty list = none named —
    callers should then fan out to all corridors (broad monitoring)."""
    q = (user_query or "").lower()
    return [cid for cid, (aliases, _) in _CORRIDOR_SEARCH.items()
            if any(a in q for a in aliases)]


def _gdelt_corridor_query(corridor_id: str) -> str:
    phrases = _CORRIDOR_SEARCH[corridor_id][1]
    subject = phrases[0] if len(phrases) == 1 else "(" + " OR ".join(phrases) + ")"
    return f"{subject} {_GDELT_CONTEXT} sourcelang:english"


def _gnews_corridor_query(corridor_id: str) -> str:
    phrases = _CORRIDOR_SEARCH[corridor_id][1]
    return f"{' OR '.join(phrases)} oil when:7d"


# ── Per-article scoring metrics ─────────────────────────────────────────────────
# Deterministic inputs for GRI's evidence judgment: recency (a disruption signal
# is a NOW question — weight the last 72 h, taper to 14 d, drop older), and an
# attribution hint (attributed reporting beats analysis/market commentary).
# The LLM makes the severity call; these numbers keep it honest.

_MAX_AGE_DAYS = 14.0       # older news is ignored outright (dropped)
_FRESH_DAYS = 3.0          # full weight inside this window
_STALE_FLOOR = 0.3         # weight at the 14-day edge
_UNKNOWN_AGE_WEIGHT = 0.5  # unparseable/missing date: mild penalty, not a drop

_DATE_FORMATS = (
    "%a, %d %b %Y %H:%M:%S %Z",   # RFC-822 (Google News / NewsData pubDate)
    "%a, %d %b %Y %H:%M:%S %z",
    "%Y-%m-%d %H:%M:%S",          # NewsData
    "%Y%m%dT%H%M%SZ",             # GDELT seendate
    "%Y%m%d%H%M%S",               # GDELT compact
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
)

_ATTRIBUTED_MARKERS = (
    "says", "said", "reports", "reported", "confirms", "confirmed", "announces",
    "announced", "warns", "warned", "official", "navy", "military", "coast guard",
    "iea", "opec", "minister", "authority", "according to",
)
_ANALYSIS_MARKERS = (
    "opinion", "analysis", "explainer", "explained", "editorial", "comment",
    "what it means", "outlook", "prediction", "preview", "review", "vs.",
)


def _parse_published(raw: str) -> datetime | None:
    if not raw:
        return None
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _recency_weight(age_days: float | None) -> float:
    if age_days is None:
        return _UNKNOWN_AGE_WEIGHT
    if age_days <= _FRESH_DAYS:
        return 1.0
    if age_days >= _MAX_AGE_DAYS:
        return 0.0
    span = _MAX_AGE_DAYS - _FRESH_DAYS
    return round(1.0 - (1.0 - _STALE_FLOOR) * ((age_days - _FRESH_DAYS) / span), 3)


def _attribution_hint(article: dict) -> str:
    text = f"{article.get('title', '')} {article.get('description', '')}".lower()
    if any(m in text for m in _ATTRIBUTED_MARKERS):
        return "attributed"
    if any(m in text for m in _ANALYSIS_MARKERS):
        return "analysis"
    return "unknown"


def _infer_corridors(article: dict) -> set[str]:
    """Tag an article with every corridor its title/description mentions."""
    text = f"{article.get('title', '')} {article.get('description', '')}".lower()
    found: set[str] = set()
    for cid, (_aliases, phrases) in _CORRIDOR_SEARCH.items():
        if any(p.strip('"').lower() in text for p in phrases):
            found.add(cid)
    return found


# ── Per-request result cache ────────────────────────────────────────────────────
# Keyed per sub-request (the NewsData sweep by query; each GDELT search by
# corridor) so a routed Hormuz run reuses articles the broad twin-loop sweep
# already fetched. Quota protection: without it, an 8-corridor fan-out on every
# twin tick would exhaust free-tier limits in hours. Failures are never cached.

_CACHE: dict[tuple, tuple[float, list[dict]]] = {}
_CACHE_LOCK = threading.Lock()


def _cache_get(key: tuple) -> list[dict] | None:
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if hit is not None and (time.monotonic() - hit[0]) < NEWS_CACHE_TTL:
        return copy.deepcopy(hit[1])
    return None


def _cache_put(key: tuple, articles: list[dict]) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (time.monotonic(), copy.deepcopy(articles))


# ── Fetchers ────────────────────────────────────────────────────────────────────

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
    r = await client.get(GDELT_URL, params=params, timeout=_GDELT_TIMEOUT)
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


async def _fetch_gnews(client: httpx.AsyncClient, query: str) -> list[dict]:
    """Google News RSS search — free, keyless, and AUTHORITY-ranked (the same
    ranking a manual Google search gets), which the API feeds can't do. The item
    link is a news.google.com redirect; the real outlet comes from the <source>
    element and is carried as `source_domain` for the trust lookup."""
    params = {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    r = await client.get(GNEWS_URL, params=params, timeout=15,
                         follow_redirects=True)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    out: list[dict] = []
    for item in root.iter("item"):
        src = item.find("source")
        source_url = src.get("url", "") if src is not None else ""
        source_domain = extract_domain(source_url) if source_url else "unknown"
        out.append({
            "title":         item.findtext("title") or "",
            "url":           item.findtext("link") or "",
            "source":        source_domain if source_domain != "unknown"
                             else ((src.text if src is not None else None) or "unknown"),
            "source_domain": source_domain,
            "published_at":  item.findtext("pubDate") or "",
            "description":   "",
            "origin":        "gnews",
        })
        if len(out) >= _GNEWS_MAX_ARTICLES:
            break
    return out


async def _fetch_all(query: str, api_key: str,
                     corridor_ids: list[str]) -> tuple[list[dict], list[str], str]:
    """One NewsData sweep (quota-limited: 1 credit) + one free GDELT search and
    one free Google News RSS search per corridor, cache-first. Returns (deduped
    corridor-tagged articles, errors, status)."""
    plan: list[dict] = [{"source": "newsapi", "cid": None, "key": ("newsdata", query)}]
    plan += [{"source": "gdelt", "cid": cid, "key": ("gdelt", cid)}
             for cid in corridor_ids]
    plan += [{"source": "gnews", "cid": cid, "key": ("gnews", cid)}
             for cid in corridor_ids]

    results: list = [None] * len(plan)
    to_fetch: list[int] = []
    for i, p in enumerate(plan):
        hit = _cache_get(p["key"])
        if hit is not None:
            results[i] = hit
        else:
            to_fetch.append(i)

    if to_fetch:
        newsapi_misses = [i for i in to_fetch if plan[i]["source"] == "newsapi"]
        gdelt_misses   = [i for i in to_fetch if plan[i]["source"] == "gdelt"]
        gnews_misses   = [i for i in to_fetch if plan[i]["source"] == "gnews"]
        async with httpx.AsyncClient() as client:
            # NewsData + Google News run concurrently; GDELT misses run
            # SEQUENTIALLY with spacing — a parallel burst trips GDELT's per-IP
            # 429 throttle and every corridor comes back empty.
            newsapi_task = (asyncio.create_task(_fetch_newsapi(client, query, api_key))
                            if newsapi_misses else None)
            gnews_tasks = {
                i: asyncio.create_task(
                    _fetch_gnews(client, _gnews_corridor_query(plan[i]["cid"])))
                for i in gnews_misses
            }
            global _GDELT_BLOCKED_UNTIL
            sent = 0
            for i in gdelt_misses:
                if time.monotonic() < _GDELT_BLOCKED_UNTIL:
                    results[i] = RuntimeError(
                        "skipped — GDELT throttled (429), backing off")
                    continue
                if sent:
                    await asyncio.sleep(_GDELT_REQUEST_SPACING)
                sent += 1
                try:
                    r = await _fetch_gdelt(client, _gdelt_corridor_query(plan[i]["cid"]))
                except Exception as e:
                    r = e
                    if _is_throttle(e):
                        _GDELT_BLOCKED_UNTIL = time.monotonic() + _GDELT_COOLDOWN
                results[i] = r
                if not isinstance(r, Exception):
                    _cache_put(plan[i]["key"], r)
            for i, task in gnews_tasks.items():
                try:
                    r = await task
                except Exception as e:
                    r = e
                results[i] = r
                if not isinstance(r, Exception):
                    _cache_put(plan[i]["key"], r)
            if newsapi_task is not None:
                try:
                    r = await newsapi_task
                except Exception as e:
                    r = e
                results[newsapi_misses[0]] = r
                if not isinstance(r, Exception):
                    _cache_put(plan[newsapi_misses[0]]["key"], r)

    # Merge: dedupe by URL — plus by normalized title, because the same story
    # arrives under a different URL per source (Google links are redirects) —
    # and union corridor tags across duplicate hits.
    errors: list[str] = []
    newsapi_failed = False
    failed_by_source: dict[str, list[str]] = {"gdelt": [], "gnews": []}
    first_error: dict[str, Exception] = {}
    merged: dict[str, dict] = {}
    order: list[str] = []
    title_index: dict[str, str] = {}   # normalized title → merge key

    for p, result in zip(plan, results):
        if isinstance(result, Exception):
            if p["source"] == "newsapi":
                newsapi_failed = True
                errors.append(f"newsapi: {type(result).__name__}: {result}")
            else:
                failed_by_source[p["source"]].append(p["cid"])
                first_error.setdefault(p["source"], result)
            continue
        for a in result:
            art = dict(a)
            tags = _infer_corridors(art)
            if p["cid"]:
                tags.add(p["cid"])
            key = art.get("url") or f"{p['source']}:{art.get('title', '')}"
            title_key = " ".join((art.get("title") or "").lower().split())
            dup_key = key if key in merged else title_index.get(title_key)
            if dup_key:
                existing = merged[dup_key]
                existing["corridors"] = sorted(set(existing.get("corridors", [])) | tags)
            else:
                art["corridors"] = sorted(tags)
                merged[key] = art
                order.append(key)
                if title_key:
                    title_index[title_key] = key

    # Aggregate per-source fan-out failures into one error entry each.
    for source, failed in failed_by_source.items():
        if failed:
            e = first_error[source]
            errors.append(
                f"{source}: {len(failed)}/{len(corridor_ids)} corridor searches failed "
                f"({type(e).__name__}: {e}) — corridors: {', '.join(failed)}")

    all_gdelt_failed = len(failed_by_source["gdelt"]) == len(corridor_ids)
    all_gnews_failed = len(failed_by_source["gnews"]) == len(corridor_ids)
    if not errors:
        status = "ok"
    elif newsapi_failed and all_gdelt_failed and all_gnews_failed:
        status = "failed"
    else:
        status = "degraded"

    return [merged[k] for k in order], errors, status


def fetch_news(query: str, api_key: str = NEWSAPI_KEY, staleness_limit: int = 3600,
               corridors: list[str] | None = None) -> dict:
    """Fetch corridor news: a single NewsData sweep with `query` plus a per-
    corridor GDELT fan-out (`corridors`; None/empty = all 8 — broad monitoring),
    so every corridor gets its own evidence slots instead of competing for one
    10-article page. Articles carry a `corridors` tag; `data.evidence_by_corridor`
    reports the per-corridor count, zeros included — a zero means UNVERIFIED this
    run, never confirmed calm."""
    fetched_at = datetime.now(timezone.utc)
    t0 = time.monotonic()

    corridor_ids = list(corridors) if corridors else list(_CORRIDOR_SEARCH)
    articles, errors, status = asyncio.run(_fetch_all(query, api_key, corridor_ids))

    # Enrich with scoring metrics; ignore old news outright (> _MAX_AGE_DAYS —
    # persistence of KNOWN events is the memory/decay layer's job, the news
    # window's job is spotting changes).
    tagged: list[dict] = []
    for a in (tag_article(a) for a in articles):
        dt = _parse_published(a.get("published_at", ""))
        age = None if dt is None else max(0.0, (fetched_at - dt).total_seconds() / 86400.0)
        if age is not None and age > _MAX_AGE_DAYS:
            continue
        a["age_days"] = None if age is None else round(age, 2)
        a["recency_weight"] = _recency_weight(age)
        a["attribution"] = _attribution_hint(a)
        tagged.append(a)

    # Per-corridor evidence aggregates. `evidence_by_corridor` stays a plain
    # count (zeros = UNVERIFIED); `corridor_evidence` carries the scoring
    # metrics — independent domains (5 syndicated copies ≠ 5 sources),
    # freshness, top trust, and a trust×recency weight.
    corridor_evidence = {cid: {"articles": 0, "independent_domains": 0,
                               "fresh_72h": 0, "top_trust": 0.0,
                               "evidence_weight": 0.0}
                         for cid in corridor_ids}
    _domains: dict[str, set] = {cid: set() for cid in corridor_ids}
    for a in tagged:
        for cid in a.get("corridors", []):
            ce = corridor_evidence.setdefault(
                cid, {"articles": 0, "independent_domains": 0, "fresh_72h": 0,
                      "top_trust": 0.0, "evidence_weight": 0.0})
            dset = _domains.setdefault(cid, set())
            ce["articles"] += 1
            dset.add(a.get("source", "unknown"))
            if a.get("age_days") is not None and a["age_days"] <= _FRESH_DAYS:
                ce["fresh_72h"] += 1
            ce["top_trust"] = round(max(ce["top_trust"], float(a.get("trust_score", 0))), 2)
            ce["evidence_weight"] = round(
                ce["evidence_weight"]
                + float(a.get("trust_score", 0)) * float(a.get("recency_weight", 0)), 3)
    for cid, dset in _domains.items():
        corridor_evidence[cid]["independent_domains"] = len(dset)
    evidence_by_corridor = {cid: ce["articles"] for cid, ce in corridor_evidence.items()}

    trust_scores = [a["trust_score"] for a in tagged] if tagged else [0.0]
    trust_avg    = round(sum(trust_scores) / len(trust_scores), 4)
    low_trust    = sum(1 for a in tagged if not a["trusted"])

    elapsed = int(time.monotonic() - t0)

    return {
        "tool":                     "news_fetcher",
        "status":                   status,
        "data":                     {"articles": tagged, "errors": errors,
                                     "evidence_by_corridor": evidence_by_corridor,
                                     "corridor_evidence": corridor_evidence,
                                     "corridors_searched": corridor_ids},
        "source_trust_avg":         trust_avg,
        "low_trust_sources_flagged": low_trust,
        "retrieved_at":             fetched_at.isoformat(),
        "staleness_seconds":        elapsed,
    }
