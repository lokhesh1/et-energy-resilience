import hashlib
from urllib.parse import urlparse
from datetime import datetime, timezone

SOURCE_TRUST_MAP = {
    # ── Official / institutional ──
    "eia.gov":                 1.00,
    "iea.org":                 0.95,
    "opec.org":                0.92,
    "un.org":                  0.90,
    "worldbank.org":           0.90,

    # ── Tier-1 wire services ──
    "reuters.com":             0.95,
    "apnews.com":              0.93,
    "afp.com":                 0.92,

    # ── Tier-1 financial / energy press ──
    "bloomberg.com":           0.90,
    "ft.com":                  0.90,
    "wsj.com":                 0.88,
    "economist.com":           0.88,
    "platts.com":              0.90,
    "argusmedia.com":          0.88,
    "icis.com":                0.87,

    # ── Geopolitics / security ──
    "foreignpolicy.com":       0.85,
    "cfr.org":                 0.87,
    "iiss.org":                0.87,
    "stratfor.com":            0.83,
    "janes.com":               0.85,

    # ── Energy trade press ──
    "maritimeexecutive.com":   0.75,
    "tankeroperator.com":      0.72,
    "rigzone.com":             0.70,
    "offshore-technology.com": 0.68,
    "oilprice.com":            0.65,
    "energymonitor.ai":        0.70,
    "naturalgasworld.com":     0.68,

    # ── Indian / regional sources ──
    "thehindubusinessline.com":       0.80,
    "thehindu.com":                   0.80,
    "economictimes.indiatimes.com":   0.78,
    "business-standard.com":          0.78,
    "hindustantimes.com":             0.78,
    "ndtvprofit.com":                 0.75,
    "ndtv.com":                       0.75,
    "livemint.com":                   0.75,
    "timesofindia.com":               0.72,
    "timesofindia.indiatimes.com":    0.72,
    "indiatimes.com":                 0.68,
    "zeenews.india.com":              0.65,
    "zeebiz.com":                     0.65,
    "dnaindia.com":                   0.65,
    "moneycontrol.com":               0.72,
    "financialexpress.com":           0.75,

    # ── Gulf / Middle East ──
    "arabnews.com":            0.72,
    "gulfnews.com":            0.70,
    "aljazeera.com":           0.75,
    "middleeasteye.net":       0.68,
    "thenationalnews.com":     0.72,
    "alarabiya.net":           0.65,
    "khaleejitimes.com":       0.68,
    "zawya.com":               0.70,

    # ── European ──
    "bbc.com":                 0.88,
    "bbc.co.uk":               0.88,
    "dw.com":                  0.85,
    "euronews.com":            0.78,
    "theguardian.com":         0.85,
    "lemonde.fr":              0.82,
    "spiegel.de":              0.80,
    "corriere.it":             0.75,
    "elpais.com":              0.78,

    # ── Asia-Pacific ──
    "nikkei.com":              0.88,
    "scmp.com":                0.82,
    "straitstimes.com":        0.83,
    "bangkokpost.com":         0.72,
    "theaustralian.com.au":    0.78,
    "smh.com.au":              0.78,
    "channelnewsasia.com":     0.80,
    "koreatimes.co.kr":        0.72,

    # ── Americas ──
    "latimes.com":             0.82,
    "washingtonpost.com":      0.85,
    "nytimes.com":             0.85,
    "globo.com":               0.72,
    "buenosairesherald.com":   0.68,
    "elfinanciero.com.mx":     0.68,

    # ── Africa ──
    "dailymaverick.co.za":     0.75,
    "businessday.ng":          0.68,
    "theafricareport.com":     0.70,
    "allafrica.com":           0.60,

    # ── Russia / CIS (state-influenced — low trust) ──
    "tass.com":                0.40,
    "rt.com":                  0.30,
    "interfax.com":            0.45,
    "kommersant.ru":           0.50,

    # ── Financial wire / general ──
    "rttnews.com":             0.65,
    "econotimes.com":          0.60,
    "marketwatch.com":         0.78,
    "investing.com":           0.70,
    "tradingeconomics.com":    0.72,

    # ── Aggregators / data sources ──
    "newsapi.org":             0.50,
    "gdelt.com":               0.55,

    # ── Fallback ──
    "unknown":                 0.20,
}

TRUST_THRESHOLD = 0.65

_ACTIVE_CANARIES: dict[str, str] = {}


def extract_domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.replace("www.", "")
        return host if host else "unknown"
    except Exception:
        return "unknown"


def tag_article(article: dict) -> dict:
    domain = extract_domain(article.get("url", ""))
    trust = SOURCE_TRUST_MAP.get(domain, SOURCE_TRUST_MAP["unknown"])
    return {**article, "trust_score": trust, "trusted": trust >= TRUST_THRESHOLD}


def embed_canary(data: dict, token_id: str) -> dict:
    payload_str = str(sorted(data.items()))
    digest = hashlib.sha256(payload_str.encode()).hexdigest()[:8]
    canary = f"INDIA-EIB-{token_id}-{digest}"
    _ACTIVE_CANARIES[token_id] = canary
    return {**data, "_canary": canary}


def check_canary_leak(text: str, token_id: str) -> dict | None:
    canary = _ACTIVE_CANARIES.get(token_id)
    if canary and canary in text:
        return {
            "alert": "canary_leak_detected",
            "token_id": token_id,
            "canary": canary,
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }
    return None
