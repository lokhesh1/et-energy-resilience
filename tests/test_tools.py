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
def test_fetch_news_failed_both_sources_fail(mock_gdelt, mock_newsapi):
    mock_newsapi.side_effect = Exception("NewsAPI down")
    mock_gdelt.side_effect = Exception("GDELT down")
    result = fetch_news("oil", api_key="test-key")
    assert result["status"] == "failed"
    assert len(result["data"]["errors"]) == 2
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
