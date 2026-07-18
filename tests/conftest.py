import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that call real external APIs (skipped by default)",
    )


@pytest.fixture(autouse=True)
def _clear_news_cache(monkeypatch):
    """fetch_news caches per-source results (quota protection); tests patch the
    fetchers with different mocks per test, so a warm cache = cross-test bleed.
    The GDELT inter-request spacing is real-API politeness — zero it in tests."""
    import tools.news_fetcher as nf
    nf._CACHE.clear()
    monkeypatch.setattr(nf, "_GDELT_REQUEST_SPACING", 0.0)
    monkeypatch.setattr(nf, "_GDELT_BLOCKED_UNTIL", 0.0)
    yield


@pytest.fixture(autouse=True)
def _audit_db_to_tmp(tmp_path, monkeypatch):
    """Keep every test's default audit DB out of the real data/audit_log.db —
    any end-to-end board run would otherwise append rows to the repo's log."""
    import eib_guardrails.audit_logger as al
    monkeypatch.setattr(al, "AUDIT_DB_PATH", str(tmp_path / "audit_log.db"))
