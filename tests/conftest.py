import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that call real external APIs (skipped by default)",
    )
