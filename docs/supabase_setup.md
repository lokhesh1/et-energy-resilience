# Supabase Setup — EIB Memory Tables

Run these in the Supabase **SQL Editor** to (re)create the memory backend used by
`memory/episodic_store.py` and `memory/procedural_store.py`.

When prompted about **Row Level Security (RLS)**, enable it. The Python backend
uses the **service_role** key (see `config/.env` → `SUPABASE_KEY`), which bypasses
RLS — so no policies are needed, and a leaked anon key can't touch these tables.

Required `.env` values:

```
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_KEY=<service_role secret key>
```

---

## 1. Episodic memory — `episodic_events`

Durable, append-only diary of everything agents do (one row per occurrence).
Used by `EpisodicStore` (`store`, `query`, `recent`).

```sql
CREATE TABLE episodic_events (
    id         UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    event_type TEXT NOT NULL,
    agent      TEXT NOT NULL,
    payload    JSONB NOT NULL,
    outcome    TEXT,                 -- 'success' | 'failure' | NULL
    timestamp  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_episodic_agent   ON episodic_events (agent);
CREATE INDEX idx_episodic_outcome ON episodic_events (outcome);
CREATE INDEX idx_episodic_type    ON episodic_events (event_type);
```

Column notes:
- `outcome` — nullable; set only on attempt-type events, enabling fast
  `query(outcome='failure')` lookups for failure-avoidance.
- `payload` — free-form event body; put `reason` and other context here.

---

## 2. Procedural memory — `procedural_skills`

Reusable "cookbook" of skill templates (one row per named recipe, upserted).
Used by `ProceduralStore` (`store_skill`, `get_skill`, `increment_use`, `list_skills`).

```sql
CREATE TABLE procedural_skills (
    id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    skill_name    TEXT NOT NULL UNIQUE,   -- upsert key: one row per recipe
    agent         TEXT NOT NULL,
    template      JSONB NOT NULL,         -- {trigger, steps, notes, source, ...}
    use_count     INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_procedural_agent ON procedural_skills (agent);
```

Column notes:
- `skill_name` — `UNIQUE`; `store_skill` upserts on it, so re-storing updates the
  template instead of duplicating (guarantees this stays a cookbook, not a log).
- `use_count` / `success_count` — efficacy counters bumped by `increment_use`;
  agents prefer skills with the best success ratio. `list_skills` orders by
  `use_count DESC` (most battle-tested first).
- Optional future: seed human-authored skills with `template.source = "human"`
  (see CLAUDE.md → Future / backlog).

---

## 3. Semantic memory — Pinecone (not Supabase)

For completeness: semantic memory (`memory/semantic_store.py`) does **not** live in
Supabase. It uses a **Pinecone** serverless index:

- Index name: value of `PINECONE_INDEX` (default `eib-semantic`)
- Dimensions: **384** (matches `all-MiniLM-L6-v2`)
- Metric: **cosine**
- `.env`: `PINECONE_API_KEY`, `PINECONE_INDEX=eib-semantic`

---

## Verifying setup

Live smoke-tests (throwaway, self-cleaning) live in the session scratchpad; the
permanent mocked tests are in `tests/test_memory.py`:

```bash
pytest tests/test_memory.py -q      # mocked, no network
```
