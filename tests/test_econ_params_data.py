"""Data-integrity tests for data/econ_params.json.

Same discipline as test_refineries_data.py — a CI guard that catches bad edits
to the seeded params before they reach the agent. These are *our own* data,
so a broken value is always a regression, not a runtime surprise.
"""
import json
from pathlib import Path

import pytest

_PARAMS_PATH = Path(__file__).parent.parent / "data" / "econ_params.json"


@pytest.fixture(scope="module")
def params():
    with open(_PARAMS_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_file_loads(params):
    assert isinstance(params, dict)


def test_india_imports_positive(params):
    assert params["india_crude_imports_mbd"] > 0


def test_grm_positive(params):
    assert params["avg_refinery_grm_usd_per_bbl"] > 0


def test_elasticity_in_range(params):
    e = params["short_run_demand_elasticity"]
    assert 0 < e < 1, f"Elasticity {e} outside (0, 1)"


def test_spike_cap_positive(params):
    assert params["brent_spike_cap_usd"] > 0


def test_global_demand_positive(params):
    assert params["global_crude_demand_mbd"] > 0


def test_cpi_coefficient_positive(params):
    assert params["cpi_bps_per_10usd_brent"] > 0


def test_restraint_ceiling_sane(params):
    pct = params["max_feasible_restraint_pct"]
    assert 0 < pct <= 0.20, f"Restraint ceiling {pct} outside (0, 0.20]"


def test_illustrative_note_present(params):
    assert "ILLUSTRATIVE" in params.get("_note", "").upper()
