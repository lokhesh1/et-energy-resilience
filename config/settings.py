import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── LLM ──
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GRI_MODEL           = "google/gemini-2.5-flash"        # foundation agent — accuracy over cost
# Distillation is a judgment task (extract reusable lessons) — kept on its own
# knob so it can be upgraded independently of GRI. Prod model TBD.
DISTILLER_MODEL     = "google/gemini-2.5-flash"
# DSM narrative is decoration only (numbers are deterministic) — cheapest Flash.
DSM_MODEL           = "google/gemini-2.5-flash-lite"
# Coordinator writes the final board-level recommendation (synthesis over the whole
# run) — the most judgment-heavy narrative, so its own knob at the Flash tier. The
# response_plan itself is deterministic; the LLM only phrases it, template fallback.
COORDINATOR_MODEL   = "google/gemini-2.5-flash"
# Chat layer: intent-gating router (run_board vs answer-from-last-run) + standalone
# query rewriter + grounded follow-up answers. Classification/phrasing over a compact
# digest — cheapest Flash tier, own knob.
CHAT_MODEL          = "google/gemini-2.5-flash-lite"
CHAT_HISTORY_TURNS  = 8   # max turns kept in any chat LLM prompt (working-memory budget)

# ── News / data sources ──
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")   # NewsData.io key (pub_...)
EIA_API_KEY = os.getenv("EIA_API_KEY")
# Each news sub-request (the NewsData sweep, or one per-corridor GDELT search) is
# cached this long, so the twin loop's tick and back-to-back board runs reuse
# articles instead of burning quota (NewsData free tier ≈ 200 credits/day).
NEWS_CACHE_TTL = int(os.getenv("NEWS_CACHE_TTL", "900"))  # seconds

# ── Observability ──
LANGSMITH_API_KEY  = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT  = os.getenv("LANGSMITH_PROJECT", "et-energy-resilience")

# ── Infrastructure ──
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ── Memory (cloud) ──
SUPABASE_URL     = os.getenv("SUPABASE_URL")
SUPABASE_KEY     = os.getenv("SUPABASE_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "eib-semantic")

# ── Audit log (durable, tamper-evident) ──
# Every board run's audit_trail is flushed to an append-only, hash-chained SQLite
# file after the answer is returned (off the hot path, best-effort). Local file so
# the demo works offline; point AUDIT_DB_PATH elsewhere (or disable) as needed.
AUDIT_LOG_ENABLED = os.getenv("AUDIT_LOG_ENABLED", "true").lower() == "true"
AUDIT_DB_PATH     = os.getenv(
    "AUDIT_DB_PATH", str(Path(__file__).parent.parent / "data" / "audit_log.db"))

# ── Twin loop (continuous SCTD) ──
# The digital twin refreshes on its OWN clock, independent of user queries: a
# background task re-runs GRI→DSM→SCTD every interval and stores the latest snapshot
# the API serves. Enabled by default (it's the "24/7 crisis team" feature); each
# refresh is a live GRI news+LLM read, so tune the interval / disable for dev.
TWIN_LOOP_ENABLED      = os.getenv("TWIN_LOOP_ENABLED", "true").lower() == "true"
TWIN_REFRESH_INTERVAL  = int(os.getenv("TWIN_REFRESH_INTERVAL", "180"))  # seconds

# ── Tunables ──
RISK_THRESHOLD         = 0.7
DSM_MODEL_THRESHOLD    = 0.5      # min corridor_risk score for DSM to model a scenario
MEMORY_DECAY_HALFLIFE  = 30       # days
TRUST_THRESHOLD        = 0.65
BRENT_TICKER           = "BZ=F"
NEWS_PAGE_SIZE         = 25
CORRIDOR_CACHE_TTL     = 300      # seconds
