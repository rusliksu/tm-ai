"""Runtime configuration for the LLM AI server (OpenRouter-only).

All values come from environment variables with sensible defaults. See README /
CLAUDE.md for the full table. Load secrets via `source .env` — never read .env here.
"""
from __future__ import annotations
import os
from pathlib import Path

# --- server ---
PORT = int(os.getenv("PORT", "8000"))
LLM_DEBUG = os.getenv("LLM_DEBUG", "false").lower() == "true"

# --- OpenRouter ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-opus-4-7")
# Global thinking/reasoning override: "auto" honours the per-model capability table,
# "off" forces reasoning off for every model (fast — no hidden reasoning tokens that
# dominate latency), "on" forces it on. Anything else is treated as "auto".
OPENROUTER_THINKING = os.getenv("OPENROUTER_THINKING", "auto").strip().lower()
if OPENROUTER_THINKING not in ("auto", "on", "off"):
    OPENROUTER_THINKING = "auto"
# Thinking budget for setup / per-generation reflection (deeper reasoning).
OPENROUTER_THINKING_BUDGET = int(os.getenv("OPENROUTER_THINKING_BUDGET", "1024"))
# Thinking budget for tactical action turns (smaller = faster for reasoning models).
OPENROUTER_ACTION_THINKING_BUDGET = int(os.getenv("OPENROUTER_ACTION_THINKING_BUDGET", "512"))
# Must comfortably exceed the thinking budget or reasoning models truncate before CHOICE.
OPENROUTER_MAX_OUTPUT_TOKENS = int(os.getenv("OPENROUTER_MAX_OUTPUT_TOKENS", "4096"))
# Ceiling when the truncation-retry loop widens max_tokens.
OPENROUTER_TRUNCATION_MAX = int(os.getenv("OPENROUTER_TRUNCATION_MAX", "16384"))
OPENROUTER_TRUNCATION_RETRIES = int(os.getenv("OPENROUTER_TRUNCATION_RETRIES", "2"))

# --- state persistence (see specs/LLM-state-persistence.md) ---
_REPO_ROOT = Path(__file__).resolve().parents[3]
LLM_STATE_DIR = Path(os.getenv("LLM_STATE_DIR", str(_REPO_ROOT / "logs" / "llm-state")))
LLM_STATE_MAX_AGE_DAYS = int(os.getenv("LLM_STATE_MAX_AGE_DAYS", "7"))
LLM_STATE_PERSIST = os.getenv("LLM_STATE_PERSIST", "true").lower() == "true"
LLM_STATE_SCHEMA_VERSION = 1

# --- card DB ---
CARD_DB_PATH = _REPO_ROOT / "data" / "card_db.json"

# Action-phase retry budget within a single /move (validation feedback loop).
MAX_ACTION_RETRIES = 2

# Decision-tree node types that constitute the opening "setup" phase.
SETUP_TYPES = {"initialCards", "prelude"}
