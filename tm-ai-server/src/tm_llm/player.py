"""LLMPlayer — per-player state, token accounting, and the LLM call wrapper.

One instance per AI player (keyed by player_id). Holds the two-part memory (coarse
inter-generation `strategy` + intra-generation `tactical`), the cached per-game action
system prompt, and cumulative token usage. The action phase is stateless: every call is a
fresh [system, user] via openrouter.single_shot — no chat history is kept.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone

from . import config, openrouter

logger = logging.getLogger(__name__)


class LLMPlayer:
    def __init__(self, player_id: str, game_id: str, model: str) -> None:
        self.player_id = player_id
        self.game_id = game_id
        self.model = model

        self.strategy: str = ""          # Part 2: coarse, inter-generation (+ backup/switch)
        self.tactical: str = ""          # Part 1: intra-generation next-steps
        self.action_system: str = ""     # stable per-game system prompt (rebuilt, not persisted)
        self.last_generation: int = -1

        self.token_usage: dict = {
            "calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0,
        }
        self.summary_logged: bool = False

    # ------------------------------------------------------------------
    # LLM call wrapper (accumulates usage + logs)
    # ------------------------------------------------------------------

    def single_shot(self, system: str, user: str, *, think: bool = True,
                    thinking_budget: int | None = None,
                    max_output_tokens: int | None = None) -> str:
        if config.LLM_DEBUG:
            _log("> SHOT user (player=%s)" % self.player_id, user)
        text, usage = openrouter.single_shot(
            self.model, system, user, think=think,
            thinking_budget=thinking_budget, max_output_tokens=max_output_tokens,
        )
        self._accumulate(usage)
        if config.LLM_DEBUG:
            _log("< SHOT response (player=%s)" % self.player_id, text)
        return text

    def _accumulate(self, usage: dict) -> None:
        self.token_usage["calls"] += 1
        for k in ("input", "output", "cache_read", "cache_write"):
            self.token_usage[k] += usage.get(k, 0)
        logger.info(
            "tokens player=%s model=%s in=%d out=%d cache_read=%d | totals calls=%d in=%d out=%d est_cost=$%.4f",
            self.player_id, self.model, usage.get("input", 0), usage.get("output", 0),
            usage.get("cache_read", 0), self.token_usage["calls"],
            self.token_usage["input"], self.token_usage["output"],
            openrouter.cost(self.model, self.token_usage),
        )

    def log_token_summary(self) -> None:
        if self.summary_logged:
            return
        self.summary_logged = True
        u = self.token_usage
        in_p, out_p = openrouter.get_pricing(self.model)
        billed_in = u["input"] - u["cache_read"]
        total = billed_in * in_p + u["cache_read"] * in_p * 0.1 + u["output"] * out_p
        logger.info(
            "TOKEN SUMMARY player=%s | game=%s | model=%s | calls=%d | input=%d (billed=%d cached=%d) | output=%d | TOTAL=$%.4f",
            self.player_id, self.game_id, self.model, u["calls"], u["input"], billed_in,
            u["cache_read"], u["output"], total,
        )

    # ------------------------------------------------------------------
    # Persistence (see specs/LLM-state-persistence.md)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema_version": config.LLM_STATE_SCHEMA_VERSION,
            "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "player_id": self.player_id,
            "game_id": self.game_id,
            "model": self.model,
            "strategy": self.strategy,
            "tactical": self.tactical,
            "last_generation": self.last_generation,
            "token_usage": dict(self.token_usage),
            "summary_logged": self.summary_logged,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LLMPlayer":
        p = cls(player_id=data["player_id"], game_id=data["game_id"], model=data["model"])
        p.strategy = data.get("strategy", "") or ""
        p.tactical = data.get("tactical", "") or ""
        p.last_generation = int(data.get("last_generation", -1))
        usage = data.get("token_usage") or {}
        for k in ("calls", "input", "output", "cache_read", "cache_write"):
            if k in usage:
                p.token_usage[k] = int(usage[k])
        p.summary_logged = bool(data.get("summary_logged", False))
        return p


def _log(header: str, text: str | None) -> None:
    if text is None:
        logger.info("%s\n(empty)", header)
        return
    logger.info("%s\n%s", header, text)
