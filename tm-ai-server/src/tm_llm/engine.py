"""Action-selection orchestration: setup phase, action phase, per-generation reflection.

`select_action_llm` is the single entry point used by the FastAPI `/move` handler. The
action phase is stateless — each /move is one self-contained call carrying the full state
snapshot plus the player's own memory — with an internal validation-retry loop that resends
the prompt with an error banner if CHOICE is missing or the payment is underfunded.
"""
from __future__ import annotations
import logging
import re

from . import config, prompts, registry
from .options import flatten_options, index_to_response, _default_response
from .payment import check_payment_valid

logger = logging.getLogger(__name__)


def select_action_llm(state: dict, waiting_for: dict, game_id: str, player_id: str,
                      last_error: str | None = None) -> tuple[dict, dict]:
    player = registry.get_or_create_player(player_id, game_id)
    wf_type = waiting_for.get("type", "")
    try:
        if wf_type in config.SETUP_TYPES:
            return _select_setup(state, waiting_for, player, last_error)
        return _select_action(state, waiting_for, player, last_error)
    except Exception as exc:
        logger.error("LLM selection failed (player=%s type=%s): %s — using default",
                     player_id, wf_type, exc, exc_info=True)
        return _default_response(waiting_for), {"llm_error": str(exc)}


# ---------------------------------------------------------------------------
# Setup phase
# ---------------------------------------------------------------------------

def _select_setup(state: dict, waiting_for: dict, player, last_error: str | None) -> tuple[dict, dict]:
    g = state.get("game", {})
    logger.info("LLM setup (player=%s type=%s board=%s exps=%s)",
                player.player_id, waiting_for.get("type"), g.get("boardName"), g.get("expansions"))

    user = prompts.build_setup_prompt(state, waiting_for)
    if last_error:
        user = (
            f'⚠ THE GAME SERVER REJECTED YOUR PREVIOUS RESPONSE:\n  Error: "{last_error}"\n'
            "The server is always correct. Adapt your answer.\n\n"
        ) + user

    system = prompts.build_setup_system(state)
    text = player.single_shot(system, user, think=True, thinking_budget=config.OPENROUTER_THINKING_BUDGET)
    logger.info("Setup response (player=%s):\n%s", player.player_id, text[:1000])

    input_response, strategy = prompts.parse_setup_response(text, waiting_for, prior_strategy=player.strategy)
    player.strategy = strategy
    logger.info("Player %s strategy stored:\n%s", player.player_id, strategy)
    return input_response, {"llm_phase": "setup", "strategy": strategy[:300]}


# ---------------------------------------------------------------------------
# Per-generation reflection
# ---------------------------------------------------------------------------

def _maybe_per_generation_update(player, generation: int, state: dict) -> None:
    if player.last_generation >= 0 and generation > player.last_generation:
        logger.info("Generation bump %d→%d (player=%s) — strategy update",
                    player.last_generation, generation, player.player_id)
        prompt = prompts.build_pergen_prompt(state, player.strategy, generation)
        try:
            strategy = player.single_shot(player.action_system, prompt,
                                          thinking_budget=config.OPENROUTER_THINKING_BUDGET)
            player.strategy = strategy.strip()
            logger.info("Per-gen strategy (player=%s gen=%d):\n%s", player.player_id, generation, player.strategy)
        except Exception as exc:
            logger.warning("Per-gen update failed (player=%s gen=%d): %s", player.player_id, generation, exc)
    player.last_generation = generation


# ---------------------------------------------------------------------------
# Action phase
# ---------------------------------------------------------------------------

def _select_action(state: dict, waiting_for: dict, player, last_error: str | None) -> tuple[dict, dict]:
    options = flatten_options(waiting_for)
    if not options:
        return _default_response(waiting_for), {}

    if not player.action_system:
        player.action_system = prompts.build_action_system(state)

    generation = state.get("game", {}).get("generation", 1)
    _maybe_per_generation_update(player, generation, state)

    p = state.get("player", {})
    base_user = prompts.build_action_prompt(
        state, waiting_for, options,
        strategy=player.strategy, tactical=player.tactical, last_error=last_error,
    )

    max_out = config.OPENROUTER_MAX_OUTPUT_TOKENS
    budget = config.OPENROUTER_ACTION_THINKING_BUDGET
    text = player.single_shot(player.action_system, base_user, max_output_tokens=max_out, thinking_budget=budget)

    response: dict = {}
    debug: dict = {}
    for attempt in range(config.MAX_ACTION_RETRIES + 1):
        response, debug = prompts.parse_action_response(text, options, waiting_for, player.player_id, player=p)

        errors: list[str] = []
        if not re.search(r"CHOICE:\s*(\d+)", text):
            opts_str = "  ".join(f"{i+1}. {o['title'][:40]}" for i, o in enumerate(options))
            errors.append(
                "Your response did not include a CHOICE: N line. End with CHOICE: N on its own "
                f"line where N is the option number. Options: {opts_str}"
            )
        pay_err = check_payment_valid(response, options, waiting_for, p)
        if pay_err:
            errors.append(pay_err)

        if not errors:
            break

        combined = "\n".join(f"⚠ {e}" for e in errors)
        if attempt < config.MAX_ACTION_RETRIES:
            logger.warning("Action retry %d/%d (player=%s):\n%s",
                           attempt + 1, config.MAX_ACTION_RETRIES, player.player_id, combined)
            retry_user = "⚠ YOUR PREVIOUS RESPONSE WAS INVALID — fix it and answer again:\n" + combined + "\n\n" + base_user
            text = player.single_shot(player.action_system, retry_user, max_output_tokens=max_out, thinking_budget=budget)
        else:
            only_choice_error = all("CHOICE" in e for e in errors)
            pass_opt = next((o for o in options if "pass" in o.get("title", "").lower()), None) if only_choice_error else None
            if pass_opt is not None:
                response = index_to_response(waiting_for, pass_opt["path"])
                debug = {"fallback": "pass"}
                logger.warning("Action invalid after retries (player=%s) — best-effort Pass (%r)",
                               player.player_id, pass_opt["title"])
            else:
                logger.error("Action invalid after retries (player=%s):\n%s — sending best-effort",
                             player.player_id, combined)

    tactical = prompts.capture_tactical(text)
    if tactical:
        player.tactical = tactical

    g = state.get("game", {})
    if g.get("temperature", -30) >= 8 and g.get("oxygen", 0) >= 14 and g.get("oceanCount", 0) >= 9:
        registry.log_game_token_summary(player.game_id)

    return response, debug
