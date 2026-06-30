from datetime import datetime, timezone

import yfinance as yf

from config.settings import BRENT_TICKER


def fetch_price(ticker: str = BRENT_TICKER) -> dict:
    retrieved_at = datetime.now(timezone.utc)

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")

        if hist.empty:
            raise ValueError(f"No data returned for ticker {ticker!r}")

        hist.index = hist.index.tz_convert("UTC")

        rows = [
            {
                "date":   str(idx.date()),
                "close":  round(float(row["Close"]), 4),
                "high":   round(float(row["High"]), 4),
                "low":    round(float(row["Low"]), 4),
                "volume": int(row["Volume"]),
            }
            for idx, row in hist.iterrows()
        ]

        latest   = rows[-1]
        previous = rows[-2] if len(rows) >= 2 else latest

        change_pct = round(
            (latest["close"] - previous["close"]) / previous["close"] * 100, 4
        )

        last_ts = hist.index[-1]
        staleness = int((retrieved_at - last_ts.to_pydatetime()).total_seconds())

        return {
            "tool":                      "price_feed",
            "status":                    "ok",
            "data": {
                "ticker":          ticker,
                "current_price":   latest["close"],
                "change_pct":      change_pct,
                "high":            latest["high"],
                "low":             latest["low"],
                "volume":          latest["volume"],
                "currency":        "USD",
                "history":         rows,
            },
            "source_trust_avg":          1.0,
            "low_trust_sources_flagged": 0,
            "retrieved_at":              retrieved_at.isoformat(),
            "staleness_seconds":         staleness,
        }

    except Exception as e:
        return {
            "tool":                      "price_feed",
            "status":                    "failed",
            "data":                      {"error": str(e), "ticker": ticker},
            "source_trust_avg":          1.0,
            "low_trust_sources_flagged": 0,
            "retrieved_at":              retrieved_at.isoformat(),
            "staleness_seconds":         -1,
        }
