import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── LLM ──
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GRI_MODEL           = "google/gemini-2.0-flash-001"

# ── News / data sources ──
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")   # NewsData.io key (pub_...)
EIA_API_KEY = os.getenv("EIA_API_KEY")

# ── Observability ──
LANGSMITH_API_KEY  = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT  = os.getenv("LANGSMITH_PROJECT", "et-energy-resilience")

# ── Infrastructure ──
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ── Tunables ──
RISK_THRESHOLD         = 0.7
MEMORY_DECAY_HALFLIFE  = 30       # days
TRUST_THRESHOLD        = 0.65
BRENT_TICKER           = "BZ=F"
NEWS_PAGE_SIZE         = 25
CORRIDOR_CACHE_TTL     = 300      # seconds
