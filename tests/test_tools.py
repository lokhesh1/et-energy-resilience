import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

# ─────────────────────────────────────────────
# canary_tokens
# ─────────────────────────────────────────────
from tools.canary_tokens import (
    extract_domain,
    tag_article,
    embed_canary,
    check_canary_leak,
    SOURCE_TRUST_MAP,
)


def test_extract_domain_standard():
    assert extract_domain("https://www.reuters.com/article/oil") == "reuters.com"


def test_extract_domain_no_www():
    assert extract_domain("https://eia.gov/data") == "eia.gov"


def test_extract_domain_invalid():
    assert extract_domain("not-a-url") == "unknown"


def test_tag_article_trusted_domain():
    article = {"url": "https://reuters.com/article/oil", "title": "Oil rises"}
    tagged = tag_article(article)
    assert tagged["trust_score"] == SOURCE_TRUST_MAP["reuters.com"]
    assert tagged["trusted"] is True


def test_tag_article_unknown_domain():
    article = {"url": "https://randomsite.xyz/news", "title": "Oil news"}
    tagged = tag_article(article)
    assert tagged["trust_score"] == SOURCE_TRUST_MAP["unknown"]
    assert tagged["trusted"] is False


def test_tag_article_preserves_original_fields():
    article = {"url": "https://bloomberg.com/news", "title": "Crude", "extra": 42}
    tagged = tag_article(article)
    assert tagged["title"] == "Crude"
    assert tagged["extra"] == 42


def test_embed_canary_adds_field():
    data = {"corridor": "hormuz", "risk": 0.8}
    result = embed_canary(data, "spr-001")
    assert "_canary" in result
    assert result["_canary"].startswith("INDIA-EIB-spr-001-")


def test_check_canary_leak_detected():
    data = {"value": 123}
    embedded = embed_canary(data, "leak-test")
    canary_str = embedded["_canary"]
    alert = check_canary_leak(f"Agent output contains {canary_str} here", "leak-test")
    assert alert is not None
    assert alert["alert"] == "canary_leak_detected"
    assert alert["token_id"] == "leak-test"


def test_check_canary_no_leak():
    embed_canary({"x": 1}, "safe-test")
    result = check_canary_leak("normal agent output with no canary", "safe-test")
    assert result is None


def test_check_canary_unknown_token():
    result = check_canary_leak("some text", "nonexistent-token")
    assert result is None


# ─────────────────────────────────────────────
# news_fetcher
# ─────────────────────────────────────────────
from tools.news_fetcher import fetch_news, _fetch_newsapi, _fetch_gdelt

MOCK_NEWSAPI_ARTICLES = [
    {"title": "Hormuz tension", "url": "https://reuters.com/hormuz", "source": {"name": "Reuters"},
     "publishedAt": "2026-06-30T10:00:00Z", "description": "Tensions rise"},
]

MOCK_GDELT_ARTICLES = [
    {"title": "Oil corridor risk", "url": "https://bloomberg.com/oil", "domain": "bloomberg.com",
     "seendate": "2026-06-30T09:00:00Z"},
]


def _make_mock_response(json_data):
    mock = AsyncMock()
    mock.raise_for_status = AsyncMock()
    mock.json = lambda: json_data
    return mock


@pytest.fixture(autouse=True)
def _stub_gnews(monkeypatch):
    """Google News RSS is the third live source inside fetch_news; stub it
    empty by default so the older two-source tests stay deterministic.
    gnews-specific tests configure this mock directly."""
    mock = AsyncMock(return_value=[])
    monkeypatch.setattr("tools.news_fetcher._fetch_gnews", mock)
    return mock


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_fetch_news_ok_schema(mock_gdelt, mock_newsapi):
    mock_newsapi.return_value = [
        {"title": "T1", "url": "https://reuters.com/a", "source": "Reuters",
         "published_at": "2026-06-30T10:00:00Z", "description": "", "origin": "newsapi"}
    ]
    mock_gdelt.return_value = [
        {"title": "T2", "url": "https://bloomberg.com/b", "source": "bloomberg.com",
         "published_at": "2026-06-30T09:00:00Z", "description": "", "origin": "gdelt"}
    ]
    result = fetch_news("hormuz crude oil", api_key="test-key")

    assert result["tool"] == "news_fetcher"
    assert result["status"] == "ok"
    assert "articles" in result["data"]
    assert isinstance(result["source_trust_avg"], float)
    assert isinstance(result["low_trust_sources_flagged"], int)
    assert "retrieved_at" in result
    assert isinstance(result["staleness_seconds"], int)


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_fetch_news_trust_scoring(mock_gdelt, mock_newsapi):
    mock_newsapi.return_value = [
        {"title": "T", "url": "https://reuters.com/x", "source": "Reuters",
         "published_at": "", "description": "", "origin": "newsapi"}
    ]
    mock_gdelt.return_value = [
        {"title": "T2", "url": "https://randomsite.xyz/y", "source": "unknown",
         "published_at": "", "description": "", "origin": "gdelt"}
    ]
    result = fetch_news("oil", api_key="test-key")
    assert result["low_trust_sources_flagged"] >= 1
    assert 0.0 <= result["source_trust_avg"] <= 1.0


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_fetch_news_degraded_one_source_fails(mock_gdelt, mock_newsapi):
    mock_newsapi.side_effect = Exception("NewsAPI timeout")
    mock_gdelt.return_value = [
        {"title": "T", "url": "https://bloomberg.com/z", "source": "bloomberg.com",
         "published_at": "", "description": "", "origin": "gdelt"}
    ]
    result = fetch_news("oil", api_key="test-key")
    assert result["status"] == "degraded"
    assert len(result["data"]["errors"]) == 1


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_fetch_news_failed_all_sources_fail(mock_gdelt, mock_newsapi, _stub_gnews):
    mock_newsapi.side_effect = Exception("NewsAPI down")
    mock_gdelt.side_effect = Exception("GDELT down")
    _stub_gnews.side_effect = Exception("gnews down")
    result = fetch_news("oil", api_key="test-key")
    assert result["status"] == "failed"
    assert len(result["data"]["errors"]) == 3   # newsapi + gdelt agg + gnews agg
    assert result["data"]["articles"] == []


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_fetch_news_merges_both_sources(mock_gdelt, mock_newsapi):
    mock_newsapi.return_value = [
        {"title": "N", "url": "https://reuters.com/n", "source": "Reuters",
         "published_at": "", "description": "", "origin": "newsapi"}
    ]
    mock_gdelt.return_value = [
        {"title": "G", "url": "https://ft.com/g", "source": "ft.com",
         "published_at": "", "description": "", "origin": "gdelt"}
    ]
    result = fetch_news("oil", api_key="test-key")
    origins = [a["origin"] for a in result["data"]["articles"]]
    assert "newsapi" in origins
    assert "gdelt" in origins


# ─────────────────────────────────────────────
# price_feed
# ─────────────────────────────────────────────
from tools.price_feed import fetch_price


def _mock_history(days=5):
    now = datetime.now(timezone.utc)
    dates = pd.date_range(end=now, periods=days, freq="D", tz="UTC")
    return pd.DataFrame({
        "Close":  [70.0, 71.5, 72.0, 73.0, 74.0],
        "High":   [71.0, 72.5, 73.0, 74.0, 75.0],
        "Low":    [69.0, 70.5, 71.0, 72.0, 73.0],
        "Volume": [10000, 11000, 12000, 13000, 14000],
    }, index=dates)


@patch("tools.price_feed.yf.Ticker")
def test_price_feed_ok_schema(mock_ticker):
    mock_ticker.return_value.history.return_value = _mock_history()
    result = fetch_price()
    assert result["tool"] == "price_feed"
    assert result["status"] == "ok"
    assert result["source_trust_avg"] == 1.0
    assert result["low_trust_sources_flagged"] == 0
    assert "retrieved_at" in result
    assert isinstance(result["staleness_seconds"], int)


@patch("tools.price_feed.yf.Ticker")
def test_price_feed_data_fields(mock_ticker):
    mock_ticker.return_value.history.return_value = _mock_history()
    data = fetch_price()["data"]
    assert data["ticker"] == "BZ=F"
    assert data["current_price"] == 74.0
    assert data["currency"] == "USD"
    assert isinstance(data["change_pct"], float)
    assert isinstance(data["history"], list)
    assert len(data["history"]) == 5


@patch("tools.price_feed.yf.Ticker")
def test_price_feed_change_pct(mock_ticker):
    mock_ticker.return_value.history.return_value = _mock_history()
    data = fetch_price()["data"]
    expected = round((74.0 - 73.0) / 73.0 * 100, 4)
    assert data["change_pct"] == expected


@patch("tools.price_feed.yf.Ticker")
def test_price_feed_history_shape(mock_ticker):
    mock_ticker.return_value.history.return_value = _mock_history()
    history = fetch_price()["data"]["history"]
    for row in history:
        assert "date" in row
        assert "close" in row
        assert "high" in row
        assert "low" in row
        assert "volume" in row


@patch("tools.price_feed.yf.Ticker")
def test_price_feed_failed_empty_data(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame()
    result = fetch_price()
    assert result["status"] == "failed"
    assert "error" in result["data"]


@patch("tools.price_feed.yf.Ticker")
def test_price_feed_failed_exception(mock_ticker):
    mock_ticker.side_effect = Exception("network error")
    result = fetch_price()
    assert result["status"] == "failed"
    assert result["staleness_seconds"] == -1


# ─────────────────────────────────────────────
# corridor_status
# ─────────────────────────────────────────────
from tools.corridor_status import get_corridor_status, apply_incident, clear_incident

CORRIDOR_IDS = [
    "strait_of_hormuz", "suez_canal", "malacca_strait", "bab_el_mandeb",
    "turkish_straits", "danish_straits", "cape_of_good_hope", "panama_canal",
]


def test_corridor_status_ok_schema():
    result = get_corridor_status()
    assert result["tool"] == "corridor_status"
    assert result["status"] == "ok"
    assert result["source_trust_avg"] == 1.0
    assert result["low_trust_sources_flagged"] == 0
    assert "retrieved_at" in result
    assert "staleness_seconds" in result


def test_corridor_status_all_8_corridors():
    corridors = get_corridor_status()["data"]["corridors"]
    assert len(corridors) == 8
    ids = [c["id"] for c in corridors]
    for cid in CORRIDOR_IDS:
        assert cid in ids


def test_corridor_status_baseline_no_disruption():
    result = get_corridor_status({})
    for c in result["data"]["corridors"]:
        assert c["disruption_pct"] == 0.0
        assert c["current_flow_mbd"] == c["baseline_flow_mbd"]
        assert c["status"] == "open"
    assert result["data"]["total_disrupted_flow_mbd"] == 0.0


def test_corridor_status_corridor_fields():
    corridors = get_corridor_status()["data"]["corridors"]
    for c in corridors:
        assert "id" in c
        assert "baseline_flow_mbd" in c
        assert "current_flow_mbd" in c
        assert "disruption_pct" in c
        assert "risk_score" in c
        assert "status" in c
        assert "alternative_routes" in c
        assert isinstance(c["risk_factors"], list)


def test_corridor_status_with_override():
    override = {"strait_of_hormuz": {"disruption_pct": 60.0, "last_incident": "blockade"}}
    result = get_corridor_status(override)
    hormuz = next(c for c in result["data"]["corridors"] if c["id"] == "strait_of_hormuz")
    assert hormuz["disruption_pct"] == 60.0
    assert hormuz["current_flow_mbd"] == round(21.0 * 0.4, 4)
    assert hormuz["status"] == "restricted"
    assert hormuz["last_incident"] == "blockade"


def test_corridor_status_risk_score_range():
    override = {"suez_canal": {"disruption_pct": 80.0}}
    corridors = get_corridor_status(override)["data"]["corridors"]
    for c in corridors:
        assert 0.0 <= c["risk_score"] <= 1.0


def test_corridor_status_highest_risk():
    override = {"panama_canal": {"disruption_pct": 100.0}}
    result = get_corridor_status(override)
    assert result["data"]["highest_risk_corridor"] == "panama_canal"


def test_corridor_status_total_disrupted_flow():
    override = {"strait_of_hormuz": {"disruption_pct": 50.0}}
    result = get_corridor_status(override)
    expected = round(21.0 * 0.5, 4)
    assert result["data"]["total_disrupted_flow_mbd"] == expected


def test_apply_and_clear_incident():
    apply_incident("malacca_strait", 30.0, "piracy surge")
    r = get_corridor_status()
    malacca = next(c for c in r["data"]["corridors"] if c["id"] == "malacca_strait")
    assert malacca["disruption_pct"] == 30.0
    clear_incident("malacca_strait")
    r2 = get_corridor_status()
    malacca2 = next(c for c in r2["data"]["corridors"] if c["id"] == "malacca_strait")
    assert malacca2["disruption_pct"] == 0.0


# ── Fixture: isolate apply_incident side-effects between tests ────────────────

@pytest.fixture(autouse=True)
def _clear_all_incidents():
    for cid in CORRIDOR_IDS:
        clear_incident(cid)
    yield
    for cid in CORRIDOR_IDS:
        clear_incident(cid)


# ── corridor_status: file-missing fallback ────────────────────────────────────

def test_corridor_status_file_missing_returns_failed():
    with patch("tools.corridor_status._load_baselines", side_effect=FileNotFoundError("no file")):
        result = get_corridor_status()
    assert result["status"] == "failed"
    assert "error" in result["data"]


# ═══════════════════════════════════════════════════════════════════════════════
# fetch_news — httpx-level tests
# These test _fetch_newsapi and _fetch_gdelt with a mocked httpx response,
# verifying that each source's JSON schema is parsed into the correct article dict.
# ═══════════════════════════════════════════════════════════════════════════════
import asyncio
import httpx
from tools.news_fetcher import _fetch_newsapi, _fetch_gdelt

NEWSDATA_JSON = {
    "results": [
        {
            "title":       "Hormuz shipping lanes under threat",
            "link":        "https://reuters.com/hormuz",
            "source_id":   "reuters",
            "pubDate":     "2026-07-01T10:00:00",
            "description": "Rising tensions in the Persian Gulf.",
        },
        {
            "title":       "Iran warns of retaliation",
            "link":        "https://ft.com/iran",
            "source_id":   "ft",
            "pubDate":     "2026-07-01T09:00:00",
            "description": "Statement from Iran foreign ministry.",
        },
    ]
}

GDELT_JSON = {
    "articles": [
        {
            "title":    "Gulf crisis deepens",
            "url":      "https://bloomberg.com/gulf",
            "domain":   "bloomberg.com",
            "seendate": "2026-07-01T08:00:00",
        }
    ]
}


def _mock_response(json_data):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=json_data)
    return resp


def _make_client(newsdata_json=None, gdelt_json=None,
                 newsdata_error=None, gdelt_error=None):
    """AsyncMock client whose .get() dispatches by URL."""
    client = AsyncMock()

    async def mock_get(url, **kwargs):
        if "newsdata" in url:
            if newsdata_error:
                raise newsdata_error("newsdata failure")
            return _mock_response(newsdata_json or NEWSDATA_JSON)
        else:
            if gdelt_error:
                raise gdelt_error("gdelt failure")
            return _mock_response(gdelt_json or GDELT_JSON)

    client.get = mock_get
    return client


# ── _fetch_newsapi ────────────────────────────────────────────────────────────

def test_fetch_newsapi_returns_correct_article_count():
    client = _make_client()
    articles = asyncio.run(_fetch_newsapi(client, "hormuz", "test_key"))
    assert len(articles) == len(NEWSDATA_JSON["results"])


def test_fetch_newsapi_maps_title_field():
    client = _make_client()
    articles = asyncio.run(_fetch_newsapi(client, "hormuz", "test_key"))
    assert articles[0]["title"] == "Hormuz shipping lanes under threat"


def test_fetch_newsapi_maps_url_from_link():
    client = _make_client()
    articles = asyncio.run(_fetch_newsapi(client, "hormuz", "test_key"))
    assert articles[0]["url"] == "https://reuters.com/hormuz"


def test_fetch_newsapi_maps_source_from_source_id():
    client = _make_client()
    articles = asyncio.run(_fetch_newsapi(client, "hormuz", "test_key"))
    assert articles[0]["source"] == "reuters"


def test_fetch_newsapi_sets_origin_to_newsdata():
    client = _make_client()
    articles = asyncio.run(_fetch_newsapi(client, "hormuz", "test_key"))
    assert all(a["origin"] == "newsdata" for a in articles)


def test_fetch_newsapi_raises_on_http_error():
    client = _make_client(newsdata_error=httpx.ConnectError)
    with pytest.raises(httpx.ConnectError):
        asyncio.run(_fetch_newsapi(client, "hormuz", "test_key"))


def test_fetch_newsapi_empty_results_returns_empty_list():
    client = _make_client(newsdata_json={"results": []})
    articles = asyncio.run(_fetch_newsapi(client, "hormuz", "test_key"))
    assert articles == []


# ── _fetch_gdelt ──────────────────────────────────────────────────────────────

def test_fetch_gdelt_returns_correct_article_count():
    client = _make_client()
    articles = asyncio.run(_fetch_gdelt(client, "hormuz"))
    assert len(articles) == len(GDELT_JSON["articles"])


def test_fetch_gdelt_maps_title_field():
    client = _make_client()
    articles = asyncio.run(_fetch_gdelt(client, "hormuz"))
    assert articles[0]["title"] == "Gulf crisis deepens"


def test_fetch_gdelt_maps_url_field():
    client = _make_client()
    articles = asyncio.run(_fetch_gdelt(client, "hormuz"))
    assert articles[0]["url"] == "https://bloomberg.com/gulf"


def test_fetch_gdelt_maps_source_from_domain():
    client = _make_client()
    articles = asyncio.run(_fetch_gdelt(client, "hormuz"))
    assert articles[0]["source"] == "bloomberg.com"


def test_fetch_gdelt_sets_origin_to_gdelt():
    client = _make_client()
    articles = asyncio.run(_fetch_gdelt(client, "hormuz"))
    assert all(a["origin"] == "gdelt" for a in articles)


def test_fetch_gdelt_raises_on_http_error():
    client = _make_client(gdelt_error=httpx.ConnectError)
    with pytest.raises(httpx.ConnectError):
        asyncio.run(_fetch_gdelt(client, "hormuz"))


def test_fetch_gdelt_empty_articles_returns_empty_list():
    client = _make_client(gdelt_json={"articles": []})
    articles = asyncio.run(_fetch_gdelt(client, "hormuz"))
    assert articles == []


# ── build_search_query ────────────────────────────────────────────────────────
# The user's question ROUTES; it must never FETCH — the search string is always
# built from the fixed corridor vocabulary, never from the raw user text.

from tools.news_fetcher import build_search_query, _DEFAULT_SEARCH_QUERY


def test_search_query_named_corridor():
    q = build_search_query("Iran closes the Strait of Hormuz")
    assert '"Hormuz"' in q
    # raw user words never leak into the search string
    assert "closes" not in q


def test_search_query_actor_hint_maps_to_corridor():
    assert '"Red Sea"' in build_search_query("houthi attacks on shipping")
    assert '"Hormuz"' in build_search_query("iran military escalation")


def test_search_query_status_question_gets_default_sweep():
    q = build_search_query("what is the status of corridors and supplies to india")
    assert q == _DEFAULT_SEARCH_QUERY
    assert '"Strait of Hormuz"' in q  # the broad sweep covers key chokepoints


def test_search_query_twin_loop_sentence_gets_default_sweep():
    q = build_search_query(
        "continuous background twin refresh — current corridor monitoring")
    assert q == _DEFAULT_SEARCH_QUERY


def test_search_query_multiple_corridors_parenthesized_or():
    q = build_search_query("Hormuz is closed AND Suez Canal is disrupted")
    assert q.startswith("(") and q.endswith(")")
    assert '"Hormuz"' in q and '"Suez"' in q and " OR " in q


def test_search_query_empty_query_never_blank():
    assert build_search_query("") == _DEFAULT_SEARCH_QUERY
    assert build_search_query(None) == _DEFAULT_SEARCH_QUERY


# ── route_corridors + per-corridor GDELT fan-out ──────────────────────────────
# Every corridor gets its own evidence slots (free GDELT search per corridor);
# NewsData stays at ONE request per fetch (quota). Articles carry corridor tags,
# duplicates dedupe by URL, and evidence_by_corridor reports zeros honestly.

from tools.news_fetcher import route_corridors
import tools.news_fetcher as nf


def test_route_corridors_named_and_hinted():
    assert route_corridors("Iran closes the Strait of Hormuz") == ["strait_of_hormuz"]
    assert "bab_el_mandeb" in route_corridors("houthi attacks on shipping")


def test_route_corridors_none_named_returns_empty():
    assert route_corridors("what is the status of all corridors") == []
    assert route_corridors("") == []


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_fanout_one_gdelt_search_per_corridor(mock_gdelt, mock_newsapi):
    mock_newsapi.return_value = []
    mock_gdelt.return_value = []
    fetch_news("oil", api_key="k", corridors=["strait_of_hormuz", "suez_canal"])
    assert mock_gdelt.call_count == 2
    assert mock_newsapi.call_count == 1   # NewsData never fans out (quota)


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_fanout_defaults_to_all_corridors(mock_gdelt, mock_newsapi):
    mock_newsapi.return_value = []
    mock_gdelt.return_value = []
    fetch_news("oil", api_key="k")   # no corridors routed → broad monitoring
    assert mock_gdelt.call_count == len(nf._CORRIDOR_SEARCH)


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_fanout_articles_tagged_and_deduped(mock_gdelt, mock_newsapi):
    mock_newsapi.return_value = []
    # The SAME url returned by both corridor searches → one article, both tags.
    mock_gdelt.return_value = [
        {"title": "Tanker attacked", "url": "https://reuters.com/t", "source": "reuters.com",
         "published_at": "", "description": "", "origin": "gdelt"}]
    result = fetch_news("oil", api_key="k",
                        corridors=["strait_of_hormuz", "suez_canal"])
    arts = result["data"]["articles"]
    assert len(arts) == 1
    assert arts[0]["corridors"] == ["strait_of_hormuz", "suez_canal"]


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_newsdata_sweep_articles_get_inferred_tags(mock_gdelt, mock_newsapi):
    mock_gdelt.return_value = []
    mock_newsapi.return_value = [
        {"title": "Suez Canal convoy halted", "url": "https://ft.com/s", "source": "ft.com",
         "published_at": "", "description": "", "origin": "newsdata"}]
    result = fetch_news("oil", api_key="k", corridors=["suez_canal"])
    assert result["data"]["articles"][0]["corridors"] == ["suez_canal"]


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_evidence_by_corridor_reports_zeros(mock_gdelt, mock_newsapi):
    # A searched corridor with no hits must appear with 0 — "unverified this
    # run" must be distinguishable from "not searched".
    mock_gdelt.return_value = []
    mock_newsapi.return_value = [
        {"title": "Hormuz tension rises", "url": "https://reuters.com/h",
         "source": "reuters.com", "published_at": "", "description": "",
         "origin": "newsdata"}]
    result = fetch_news("oil", api_key="k",
                        corridors=["strait_of_hormuz", "suez_canal"])
    evidence = result["data"]["evidence_by_corridor"]
    assert evidence["strait_of_hormuz"] == 1
    assert evidence["suez_canal"] == 0


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_gdelt_failures_aggregate_to_one_error(mock_gdelt, mock_newsapi):
    mock_newsapi.return_value = []
    mock_gdelt.side_effect = Exception("GDELT down")
    result = fetch_news("oil", api_key="k",
                        corridors=["strait_of_hormuz", "suez_canal"])
    assert result["status"] == "degraded"        # NewsData still delivered
    assert len(result["data"]["errors"]) == 1    # one aggregated gdelt entry
    assert "2/2" in result["data"]["errors"][0]


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_news_cache_prevents_refetch_within_ttl(mock_gdelt, mock_newsapi):
    mock_newsapi.return_value = []
    mock_gdelt.return_value = [
        {"title": "T", "url": "https://reuters.com/c", "source": "reuters.com",
         "published_at": "", "description": "", "origin": "gdelt"}]
    r1 = fetch_news("oil", api_key="k", corridors=["strait_of_hormuz"])
    r2 = fetch_news("oil", api_key="k", corridors=["strait_of_hormuz"])
    assert mock_gdelt.call_count == 1            # second call served from cache
    assert mock_newsapi.call_count == 1
    assert r1["data"]["articles"] == r2["data"]["articles"]


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_news_cache_ttl_zero_disables_cache(mock_gdelt, mock_newsapi, monkeypatch):
    monkeypatch.setattr(nf, "NEWS_CACHE_TTL", 0)
    mock_newsapi.return_value = []
    mock_gdelt.return_value = []
    fetch_news("oil", api_key="k", corridors=["strait_of_hormuz"])
    fetch_news("oil", api_key="k", corridors=["strait_of_hormuz"])
    assert mock_gdelt.call_count == 2


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_gdelt_429_opens_circuit_breaker(mock_gdelt, mock_newsapi):
    # First 429 → remaining fan-out skipped AND later fetches back off; every
    # extra request during a GDELT block extends the block.
    from unittest.mock import MagicMock
    resp = MagicMock()
    resp.status_code = 429
    mock_newsapi.return_value = []
    mock_gdelt.side_effect = httpx.HTTPStatusError(
        "429", request=MagicMock(), response=resp)
    r = fetch_news("oil", api_key="k",
                   corridors=["strait_of_hormuz", "suez_canal", "panama_canal"])
    assert mock_gdelt.call_count == 1        # circuit opened on the first 429
    assert r["status"] == "degraded"
    fetch_news("oil", api_key="k", corridors=["strait_of_hormuz"])
    assert mock_gdelt.call_count == 1        # still backing off — no new request


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_news_cache_never_stores_failures(mock_gdelt, mock_newsapi):
    mock_newsapi.return_value = []
    mock_gdelt.side_effect = [Exception("boom"), [   # fails once, then recovers
        {"title": "T", "url": "https://reuters.com/r", "source": "reuters.com",
         "published_at": "", "description": "", "origin": "gdelt"}]]
    r1 = fetch_news("oil", api_key="k", corridors=["strait_of_hormuz"])
    r2 = fetch_news("oil", api_key="k", corridors=["strait_of_hormuz"])
    assert r1["data"]["articles"] == []
    assert len(r2["data"]["articles"]) == 1      # retried, not a cached failure


# ── Google News RSS (third source) + trust honesty ────────────────────────────

from tools.news_fetcher import _fetch_gnews   # real fn, bound before the stub
from tools.canary_tokens import tag_article

_GNEWS_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>q - Google News</title>
<item>
  <title>Hormuz tanker halted after strike - Reuters</title>
  <link>https://news.google.com/rss/articles/abc123</link>
  <pubDate>Fri, 17 Jul 2026 08:00:00 GMT</pubDate>
  <source url="https://www.reuters.com">Reuters</source>
</item>
</channel></rss>"""


def test_fetch_gnews_parses_rss_and_carries_source_domain():
    from unittest.mock import MagicMock
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.text = _GNEWS_RSS
    client = AsyncMock()
    client.get = AsyncMock(return_value=resp)
    arts = asyncio.run(_fetch_gnews(client, "q"))
    assert len(arts) == 1
    a = arts[0]
    assert a["origin"] == "gnews"
    assert a["url"] == "https://news.google.com/rss/articles/abc123"
    assert a["source_domain"] == "reuters.com"   # outlet, not the redirect host
    assert "Hormuz tanker halted" in a["title"]


def test_tag_article_trust_rated_flag_and_source_domain_override():
    rated = tag_article({"url": "https://reuters.com/x"})
    assert rated["trust_rated"] is True and rated["trust_score"] == 0.95
    unrated = tag_article({"url": "https://smallblog.example/x"})
    assert unrated["trust_rated"] is False and unrated["trust_score"] == 0.20
    # Google News redirect link + source_domain → trust of the real outlet.
    gn = tag_article({"url": "https://news.google.com/rss/articles/abc",
                      "source_domain": "bbc.com"})
    assert gn["trust_rated"] is True and gn["trust_score"] == 0.88


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_gnews_fans_out_per_corridor(mock_gdelt, mock_newsapi, _stub_gnews):
    mock_newsapi.return_value = []
    mock_gdelt.return_value = []
    fetch_news("oil", api_key="k", corridors=["strait_of_hormuz", "suez_canal"])
    assert _stub_gnews.call_count == 2


# ── Per-article scoring metrics (recency / attribution / corridor aggregates) ──

from tools.news_fetcher import _parse_published, _recency_weight, _attribution_hint


def test_parse_published_handles_all_three_feed_formats():
    assert _parse_published("Fri, 17 Jul 2026 08:00:00 GMT") is not None   # gnews RFC-822
    assert _parse_published("2026-07-17 08:00:00") is not None             # NewsData
    assert _parse_published("20260717T080000Z") is not None                # GDELT
    assert _parse_published("not a date") is None
    assert _parse_published("") is None


def test_recency_weight_bands():
    assert _recency_weight(0.5) == 1.0        # fresh: full weight
    assert _recency_weight(3.0) == 1.0
    assert 0.3 < _recency_weight(8.0) < 1.0   # tapering
    assert _recency_weight(20.0) == 0.0       # beyond the window
    assert _recency_weight(None) == 0.5       # unknown date: penalty, not a drop


def test_attribution_hint():
    assert _attribution_hint({"title": "Navy says tanker seized near strait"}) == "attributed"
    assert _attribution_hint({"title": "Analysis: what the blockade could mean"}) == "analysis"
    assert _attribution_hint({"title": "Tanker fire in gulf"}) == "unknown"


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_old_news_dropped_and_metrics_attached(mock_gdelt, mock_newsapi):
    from datetime import datetime, timedelta, timezone as tz
    fresh = (datetime.now(tz.utc) - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
    stale = (datetime.now(tz.utc) - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    mock_gdelt.return_value = []
    mock_newsapi.return_value = [
        {"title": "Hormuz strike reported by navy", "url": "https://reuters.com/f",
         "source": "reuters.com", "published_at": fresh, "description": "",
         "origin": "newsdata"},
        {"title": "Old Hormuz story", "url": "https://reuters.com/o",
         "source": "reuters.com", "published_at": stale, "description": "",
         "origin": "newsdata"},
    ]
    result = fetch_news("oil", api_key="k", corridors=["strait_of_hormuz"])
    arts = result["data"]["articles"]
    assert len(arts) == 1                       # >14d news ignored outright
    a = arts[0]
    assert a["age_days"] < 1 and a["recency_weight"] == 1.0
    assert a["attribution"] == "attributed"     # "navy", "reported"


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_corridor_evidence_aggregates(mock_gdelt, mock_newsapi):
    from datetime import datetime, timedelta, timezone as tz
    fresh = (datetime.now(tz.utc) - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
    mock_gdelt.return_value = []
    mock_newsapi.return_value = [
        {"title": "Hormuz blockade confirmed", "url": "https://reuters.com/1",
         "source": "reuters.com", "published_at": fresh, "description": "",
         "origin": "newsdata"},
        {"title": "Hormuz shipping halted, officials say", "url": "https://reuters.com/2",
         "source": "reuters.com", "published_at": fresh, "description": "",
         "origin": "newsdata"},
    ]
    result = fetch_news("oil", api_key="k", corridors=["strait_of_hormuz", "suez_canal"])
    ce = result["data"]["corridor_evidence"]
    hormuz = ce["strait_of_hormuz"]
    assert hormuz["articles"] == 2
    assert hormuz["independent_domains"] == 1   # syndication ≠ independent sources
    assert hormuz["fresh_72h"] == 2
    assert hormuz["top_trust"] == 0.95
    assert hormuz["evidence_weight"] == 1.9     # 2 × (0.95 trust × 1.0 recency)
    assert ce["suez_canal"]["articles"] == 0    # zero-aggregate present (unverified)


@patch("tools.news_fetcher._fetch_newsapi", new_callable=AsyncMock)
@patch("tools.news_fetcher._fetch_gdelt", new_callable=AsyncMock)
def test_same_story_across_sources_deduped_by_title(mock_gdelt, mock_newsapi,
                                                    _stub_gnews):
    # Same story, different URLs (Google links are redirects) → one article.
    mock_newsapi.return_value = []
    mock_gdelt.return_value = [
        {"title": "Hormuz tanker halted", "url": "https://reuters.com/a",
         "source": "reuters.com", "published_at": "", "description": "",
         "origin": "gdelt"}]
    _stub_gnews.return_value = [
        {"title": "  Hormuz  Tanker Halted ", "url": "https://news.google.com/x",
         "source": "reuters.com", "source_domain": "reuters.com",
         "published_at": "", "description": "", "origin": "gnews"}]
    result = fetch_news("oil", api_key="k", corridors=["strait_of_hormuz"])
    assert len(result["data"]["articles"]) == 1
    assert result["data"]["evidence_by_corridor"]["strait_of_hormuz"] == 1
