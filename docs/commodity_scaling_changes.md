# Commodity Generalization — Future Change Map

> **What this file is:** the complete, file-by-file change map for scaling this system
> beyond crude oil to ANY import/export commodity that affects the economy (grain,
> LNG, semiconductors, fertilizer, pharma, …). Written while the crude system is the
> only implementation, to be executed AFTER the crude system is fully tested — so
> nothing is forgotten when scaling starts.
>
> **Core finding:** the architecture is already commodity-agnostic. The math is
> `capacity × dependency_share × disruption_fraction` and `min(max_volume, gap)` —
> dimension-free. What is crude-specific is the **data, the units, the vocabulary,
> and the parameter tables**. Generalization = swap tables, not rewrite logic.
>
> Status: 📋 planned (not started). Last updated: 2026-07-11.

---

## 1. Target architecture — the "commodity pack"

Everything commodity-specific moves into one self-contained data bundle per commodity.
The engine loads the active pack; crude becomes just the first pack (and must stay
**bit-identical** in behaviour after the refactor — that is the regression bar).

```
data/commodities/<name>/
  network.json          ← generalizes corridors.json  (nodes, chokepoints, baselines, transport mode)
  consumers.json        ← generalizes refineries.json (ports/factories/fabs/plants + dependency shares)
  substitutability.json ← generalizes grade_matrix.json (what can replace what; MAY be empty)
  suppliers.json        ← per-commodity supplier catalog (+ sanctioned traps)
  params.json           ← merges dsm_params + procurement_params (durations, focus-economy
                           import shares, reroute deltas, scoring weights, coverage bands,
                           constitution numeric bounds)
  meta.json             ← unit ("mbd" → "kt_day"/"teu_day"/"mmbtu_day"), price ticker
                           (BZ=F → ZW=F …), price fallback value, focus_economy ("india"),
                           commodity nouns for prompts/UI ("crude oil", "cargo", "refinery"),
                           supplier region list (["west_africa","americas","spot"] for crude)
```

One new loader module — `config/commodity.py` (or extend `config/settings.py`) —
exposes the active pack: `PACK.network`, `PACK.consumers`, `PACK.unit`,
`PACK.known_nodes` (replaces `KNOWN_CORRIDORS`), etc. Selected by a single
`COMMODITY=<name>` env knob. **Every `json.load` of a `data/*.json` in the codebase
is replaced by a pack read** — the sections below list each one.

### Invariants that must NOT break (this is the efficiency being preserved)

- Deterministic cores stay deterministic: DSM tables, SCTD projection, bidder sizing,
  evaluator scoring, coordinator plan. LLM stays decoration-with-fallback.
- Constitution **recompute cross-checks** (DSM-07/08, PROC, COORD-03) keep working —
  they are arithmetic, so only their bounds/field names parametrize.
- Stigmergy markers stay plain `target → intensity` strings (targets become pack node
  ids — no code change).
- xMemory / decay / distiller untouched in logic; only trajectory field names follow
  the state renames.
- Append-only episodic invariant, twin-loop stale-beats-blank, best-effort
  never-raise — all unchanged.
- The crude pack must reproduce today's numbers exactly (golden-run test, §8).

---

## 2. State & naming decisions (decide ONCE, before touching code)

Hard-coded crude/India nouns appear in **state keys** — decide rename vs alias first,
because every file below follows this decision.

| Today | Generalized | Where it lives |
|---|---|---|
| `affected_refineries` | `affected_facilities` | `graph/eib_state.py` + every reader |
| `capacity_mbd`, `max_volume_mbd`, `covered_mbd`, `residual_gap_mbd`, `total_disrupted_flow_mbd`, `total_india_shortfall_mbd` | drop the unit suffix (`capacity`, `max_volume`, …) + a single `unit` field from `meta.json` carried in `twin_state` / bids / mix | state, agents, tools, API, tests |
| `india_import_share`, `india_exposure`, "Indian refineries" | `focus_import_share`, `focus_exposure`, focus economy name from `meta.json` | dsm_params, DSM, SCTD, GRI prompt |
| `corridor_risk`, `corridor_events`, `KNOWN_CORRIDORS` | `node_risk` / `node_events` / `PACK.known_nodes` — or keep "corridor" as the generic term for any transport chokepoint (cheaper; corridors exist for air/rail/pipeline too) | everywhere |
| Region names `west_africa` / `americas` / `spot` | pack-defined region list (see §5 procurement) | procurement pod, workflow fan-out |

**Recommendation:** keep the word "corridor" (it generalizes fine), rename the
refinery/india/mbd-suffixed keys in ONE sweep with the pack refactor. Doing it in one
sweep is why this file exists.

---

## 3. `data/` — replaced by packs

| File | Fate |
|---|---|
| `data/corridors.json` | → `commodities/crude_oil/network.json`. **Add `mode: "maritime"` per node** — other commodities need `air` / `pipeline` / `rail` / `land_border` (new modeling, §7). Keep `coord_source` caveat. |
| `data/refineries.json` | → `commodities/crude_oil/consumers.json`. Schema: `id, name, capacity, unit, lat/lon, corridor_dependency` — already generic, only nouns change. |
| `data/grade_matrix.json` | → `commodities/crude_oil/substitutability.json`. **Schema must support "no substitute exists"** (empty compatibility set) — semiconductors/pharma often have none; the procurement pod must degrade gracefully to "unfillable gap, model duration" instead of erroring (§7). |
| `data/suppliers.json` | → `commodities/crude_oil/suppliers.json`. `region` values become pack-defined. Keep the 3 sanctioned traps per pack — the SDN gate test pattern is generic. |
| `data/sdn_seed.json` | **Stays global** (`data/sdn_seed.json`) — sanctions screening is commodity-independent. Entities list grows per commodity's supplier geography. |
| `data/dsm_params.json` | → merged into `commodities/crude_oil/params.json` (`duration_days`, `focus_import_share`, `reroute_deltas`). Same "illustrative until calibrated" caveat propagates per pack. |
| `data/procurement_params.json` | → merged into `params.json` (scoring weights, cost-of-delay, urgency, coverage band 0.8–1.3×). |
| `data/historical_disruptions/` | → per-pack `commodities/<name>/historical/` (still unseeded). |

---

## 4. `config/` — settings + constitutions

### `config/settings.py`
- `BRENT_TICKER = "BZ=F"` → `PACK.meta.price_ticker` (yfinance covers wheat ZW=F,
  natgas NG=F, copper HG=F …; commodities with no liquid ticker set
  `price_ticker: null` and the feed falls back to `meta.price_fallback`).
- Add `COMMODITY = os.getenv("COMMODITY", "crude_oil")` + the pack loader.
- `RISK_THRESHOLD`, `DSM_MODEL_THRESHOLD` — review whether per-pack (probably yes →
  `params.json` with these as defaults).
- `NEWS_PAGE_SIZE`, decay half-lives, twin-loop knobs — stay global (geopolitics is
  commodity-independent).

### `config/constitutions/*.xml` — 4 files
The **rule logic is generic** (arithmetic recompute, sanctions re-screen, evidence
citation, escalation vocab). What is crude-specific and must parametrize from
`params.json` / `meta.json`:
- `gri_constitution.xml` — corridor closed-world list (GRI-05-style "only the 8
  known") → read from `PACK.known_nodes`; count is per-pack, not 8.
- `dsm_constitution.xml` — numeric bounds (max plausible volume, duration caps) are
  crude-scaled → per-pack bounds; DSM-07/08 recompute formulas unchanged.
- `procurement_constitution.xml` — "price = brent + premium" (PROC recompute) →
  "price = reference_price + premium"; grade-compatibility rule reads the pack
  substitutability (and must pass vacuously when the pack has no substitution matrix).
- `coordinator_constitution.xml` — COORD-03 plan↔twin arithmetic is unit-free
  (unchanged); COORD-05 escalation vocab unchanged; any "cargo/mbd" wording in
  rule text → pack nouns.

---

## 5. `agents/` — file by file

| File | Crude-specific today | Change |
|---|---|---|
| `gri_agent.py` | `KNOWN_CORRIDORS` set (8, duplicated ×3); `_SYSTEM_PROMPT` says "Indian energy supply chains"; user prompt says "the 8 known corridors"; news query is the raw user query (crude-flavored by context) | Corridor set → `PACK.known_nodes` (fixes the CLAUDE.md centralization backlog item at the same time). Prompt takes commodity + focus economy + node count from `meta.json`. Add pack-supplied search keywords appended to the news query (e.g. "wheat export ban Black Sea" vs "oil tanker Hormuz") — `tools/news_fetcher.py` itself is already generic (query passed in). Event taxonomy (war/sanctions/weather + future reversal events) stays global. |
| `dsm_agent.py` | `KNOWN_CORRIDORS` dup; loads `dsm_params.json`; `india_import_share`; mbd everywhere; narrative prompt oil-flavored | Params from `PACK.params`; field renames per §2; narrative prompt takes pack nouns. **Deterministic banded fraction mapping unchanged.** |
| `sctd_agent.py` | Loads `refineries.json`; `capacity_mbd`/`feed_at_risk`; status bands (critical ≥30% / stressed ≥10%); "refinery" nouns; reroute overload vs corridor baseline | Consumers from `PACK.consumers`; bands → `params.json` (defaults keep 30/10); projection math unchanged. `SCTD-CONTRACT` / `SCTD-LIVENESS` guardrails unchanged (structural, not crude). |
| `crisis_coordinator.py` | Plan/action strings say "cargo", "mbd", corridor nouns; `_TOL` on gap arithmetic | Nouns + unit label from `meta.json`; `_TOL` → per-pack (0.02 mbd ≠ 0.02 TEU — make it relative or pack-scaled). Escalation dial, block-flag reconstruction from audit_trail, template fallback — all unchanged. |
| `procurement/_sourcing_base.py` | Reads `suppliers.json`; grade screen via `grade_lookup`; Brent fetch + `80.0` fallback; `max_volume_mbd` | Suppliers + substitutability from pack; price via generalized `price_feed` (§6) with `meta.price_fallback`; **when the pack has no substitutability matrix, the grade screen passes all bids with a `substitutability: "n/a"` stamp** (never blocks on missing data). Sizing `min(max_volume, gap)` unchanged. |
| `procurement/west_africa_agent.py`, `americas_agent.py`, `spot_market_agent.py` | The three regions ARE crude sourcing geography | **Structural decision:** regions come from `meta.supplier_regions`. Cheapest path: keep exactly 3 graph nodes (`region_a`/`region_b`/`spot-like`) mapped to the pack's region list — no graph rewiring. Full path: dynamic fan-out (build bidder nodes from the region list at graph-compile time) — LangGraph supports it, but touches `graph/workflow.py` topology + the fan-in join edge + tests. Do the cheap path first. The spot agent's scarcity-premium behaviour is a *role*, not a region — mark one region `"type": "spot"` in meta to keep it. |
| `procurement/bid_evaluator.py` | Loads `procurement_params.json`; grade/route penalties; cost-of-delay; urgency from SCTD bands; PROC-07 trim | Weights from `PACK.params`; grade penalty reads pack substitutability (0 when n/a); everything else (greedy fill, 0.8–1.3× band, trim, bid pheromones) unchanged. |
| `distiller/experience_distiller.py` | Trajectory digest keys carry mbd/refinery field names; `_run_outcome` reads `covered_mbd`/`residual_gap_mbd` | Follows the §2 renames only — outcome labelling logic (facts, not LLM opinion) unchanged. |
| `distiller/consolidation_agent.py` | Nothing crude-specific | No change. |
| `distiller/pod.py` | Nothing crude-specific | No change. |

---

## 6. `tools/` — file by file

| File | Change |
|---|---|
| `price_feed.py` | `BZ=F` → `PACK.meta.price_ticker`; handle `null` ticker (no liquid market) → return `meta.price_fallback` with `source: "static_fallback"`. Envelope unchanged. |
| `corridor_status.py` | Reads `corridors.json` → `PACK.network`. "8 baselines" becomes "N baselines". Incident-override mechanism unchanged. |
| `grade_lookup.py` | Reads `grade_matrix.json` + `refineries.json` → `PACK.substitutability` + `PACK.consumers`. Rename to `substitutability_lookup.py` (keep a thin `grade_lookup` alias until tests migrate). Must return a well-formed "no data" answer for packs without a matrix. |
| `sanctions_check.py` | **No logic change** (matcher is generic). Seed data stays global; grows with each pack's supplier geography. Live-OFAC refresh TODO unchanged. |
| `geospatial_mapper.py` | Refinery/corridor tooltip strings + `marker_color` semantics → pack nouns + unit label. GeoJSON structure unchanged. |
| `news_fetcher.py` | **No change** (query passed in; dual-source + trust map are commodity-independent). |
| `canary_tokens.py` | No change. |
| `spr_calculator.py` (⬜ unbuilt) | **Design generic from day one:** "strategic reserve coverage" = `reserve_stock / daily_shortfall`, with per-pack `reserve_stock`, `unit`, and **`shelf_life`** (grain spoils, oil doesn't — §7). Do NOT build it crude-only and migrate later. |
| `route_ranker.py` (⬜ unbuilt) | Same: build against `PACK.network` + per-mode reroute deltas from `params.json`. |
| `mcp_energy_server.py` / `mcp_memory_server.py` (⬜ unbuilt) | Wrap pack-parametrized tools; nothing crude baked in. |
| future `cargo_availability.py` (backlog) | Per-pack live feed adapter (Kpler for crude, AgFlow for grain, …) + fallback to pack `suppliers.json`. The seed-now-source-later pattern already anticipates this. |

---

## 7. Genuinely NEW modeling (not renames — budget real design time)

1. **Substitutability spectrum.** Crude grades are partially fungible; wheat mostly
   fungible; advanced chips often have ZERO substitutes. The substitutability schema
   and the procurement pod must support the empty case end-to-end: bidders return no
   compatible cargo → evaluator reports `covers_gap: false` with reason
   `no_substitute` → coordinator escalates on an *unfillable* gap (COORD-01 already
   forces "critical" here — verify the wording works for "no substitute exists").
2. **Transport modes.** `network.json` nodes get `mode`; reroute logic needs per-mode
   penalty tables (air freight reroutes in days not weeks; pipelines don't reroute at
   all → disruption = full loss for the duration). SCTD's overload check
   (rerouted volume > alt-node baseline) generalizes as-is once baselines exist per mode.
3. **Perishability / storage.** New per-pack fields: `shelf_life_days`,
   `storage_capacity`. Affects only `spr_calculator` (coverage window shrinks with
   spoilage) and DSM duration framing. Oil: `shelf_life: null`.
4. **Reference price without a liquid market.** Many goods have no yfinance ticker —
   the static-fallback path must be first-class, not an error path.
5. **Demand elasticity** (optional, likely never): today's system models supply-side
   only. Fine — state it as a scope line in each pack's caveats.

---

## 8. `graph/`, `eib_guardrails/`, `api/`, `protocols/`, `scripts/`

| File | Change |
|---|---|
| `graph/eib_state.py` | §2 renames (`affected_refineries` → `affected_facilities`; comment nouns). Reducers/stigmergy untouched. |
| `graph/nodes.py` | No logic change (pheromone rebuild is generic). |
| `graph/workflow.py` | Only if dynamic bidder fan-out is chosen (§5); otherwise rename-only. `run_board_with_learning` unchanged. |
| `eib_guardrails/constitution_checker.py` | Remove its `KNOWN_CORRIDORS` dup → `PACK.known_nodes`; `_check_dsm` / `_check_procurement` / `_check_coordinator` recompute field names follow §2; numeric bounds from `PACK.params`. Checker mechanism, severity levels, audit embedding — unchanged. |
| `eib_guardrails/principal_hierarchy.py`, `audit_logger.py` (⬜ unbuilt) | Commodity-independent — build once, no pack awareness. |
| `api/main.py` | `_AGENTS` role strings → pack nouns; `_summarize` twin fields follow §2 renames + expose `unit` and `commodity` in every response. Consider `GET /commodity` (active pack meta) and — only if multi-commodity-per-server is ever wanted — a `?commodity=` selector + one twin loop per pack (until then: one process = one pack, run N processes). |
| `api/twin_loop.py` | No logic change; snapshot gains `commodity` + `unit` fields. |
| `protocols/agent_cards.py` | Skill descriptions say crude/corridors/Indian refineries → generate from pack meta (cards become f-strings over `PACK.meta`). Card structure unchanged. |
| `protocols/a2a_server.py` | `_artifact_from_board` field names follow §2; add `commodity` + `unit` to the data part. |
| `scripts/demo_distiller.py` | Scenarios are crude (Hormuz/Suez) — keep as the crude-pack demo; add one non-crude pack demo as the generalization acceptance test. |
| `memory/*` | **No changes.** Decay event types, stores, xMemory facade are commodity-independent. |
| `ui/*` (⬜ unbuilt) | Build pack-aware from day one: map layer reads pack GeoJSON, labels/units from meta. Do not hard-code "refinery"/"mbd" in the dashboard. |

---

## 9. `tests/` — parametrize, then add the golden run

- `test_refineries_data.py`, `test_suppliers_data.py` → parametrize over every pack
  in `data/commodities/*` (pytest `params=discovered_packs`): shares ≤ 1, positive
  capacities, known node ids, SDN-trap wiring. Every new pack gets the CI guard free.
- `test_dsm.py`, `test_sctd.py`, `test_procurement.py`, `test_coordinator.py`,
  `test_workflow.py`, `test_api.py`, `test_a2a.py`, `test_distiller.py` → follow the
  §2 field renames; scenario fixtures (Hormuz etc.) stay crude-pack-pinned
  (`COMMODITY=crude_oil` in fixture) so they keep testing real numbers.
- **NEW `test_golden_crude.py`** — the regression bar: run the full offline board on
  the crude pack before and after the refactor; every load-bearing number
  (corridor risk mapping, scenario volumes, twin shortfall, mix coverage, escalation)
  must be identical. Write it BEFORE starting the refactor.
- **NEW `test_pack_schema.py`** — validate every pack against a JSON schema
  (`docs/pack_schema.json`, to be written): required keys, unit present, region list
  non-empty, substitutability-may-be-empty honored.
- **NEW second pack** (suggest `wheat` — simple, real chokepoints: Bosphorus/Black
  Sea, Suez; fungible grades; real ticker ZW=F) as the living proof + template.

---

## 10. Migration order (each phase independently shippable, crude behaviour identical throughout)

1. **Phase 0 — freeze the bar:** write `test_golden_crude.py` against today's system.
2. **Phase 1 — pack skeleton:** create `data/commodities/crude_oil/` by MOVING the
   existing JSONs (+ `meta.json`, merged `params.json`); add the pack loader;
   repoint every `json.load`; centralize `KNOWN_CORRIDORS`. *No renames yet.* Golden
   run green.
3. **Phase 2 — the rename sweep (§2):** state keys, unit suffixes, focus-economy
   fields, in one commit with all tests updated. Golden run green (values identical,
   keys renamed → update golden accordingly, once).
4. **Phase 3 — parametrize prompts + constitutions + cards** from `meta.json`/`params.json`.
5. **Phase 4 — new modeling (§7):** substitutability-may-be-empty, transport modes,
   price-fallback-first-class. Crude unaffected (its pack has a matrix, maritime mode,
   a ticker).
6. **Phase 5 — second pack (`wheat`):** author the data, run the full board on it,
   fix what breaks, add its demo scenario. This is the acceptance test of the whole map.
7. **Phase 6 — later:** dynamic bidder fan-out, per-pack twin loops, pack-aware UI,
   pack-specific live feeds (cargo availability equivalents).

---

## 11. Out-of-scope confirmations (checked, genuinely no change needed)

- `memory/` (all 6 modules), `tools/news_fetcher.py`, `tools/canary_tokens.py`,
  `agents/distiller/consolidation_agent.py`, `agents/distiller/pod.py`,
  `graph/nodes.py`, `api/twin_loop.py` (logic), A2A task envelope, MemorySaver
  checkpointing, audit-trail append-only mechanics, escalation vocabulary, event-type
  decay taxonomy, trust-scored news aggregation.

> **Keep this file updated** the same way CLAUDE.md is: when any listed file changes
> for other reasons, re-verify its row here still holds.
