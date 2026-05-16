# Gemini LLM Performance Analysis — Terraforming Mars

**Date:** 2026-05-16  
**Board:** Tharsis  
**Expansions:** Prelude  
**Model:** `gemini-2.5-flash`  
**Session duration:** 09:10–10:57 (1h 47m)

---

## Games Observed

| Game ID | Type | Gens logged | Notes |
|---------|------|-------------|-------|
| `g26eefd6f414d` | LLM self-play (2 AI players) | 1–20 | Primary data source; started from previous session |
| `g2410f6d7b846` | Claude (manual) vs Gemini | 1–4 | Interactive game, cut short by crashes |

---

## Quantitative Summary

| Metric | Value |
|--------|-------|
| Total AI turns (requestAiMove) | 377 |
| Total move rejections (HTTP 400) | 60 (16% rejection rate) |
| Rejections — player p32dda05feb38 | 39 |
| Rejections — player p8737ee778d23 | 21 |
| Rejection cause | Always "You do not have that many resources to spend" |
| 429 RESOURCE_EXHAUSTED hits | 817 |
| Max retry delay observed | ~59 s |
| Session recoveries | 1 (server restart between sessions) |
| Gemini context cache refreshes | 2 (at ~50 min TTL as designed) |
| Per-gen strategy updates reached | Gen 20 (game g26eefd6f414d still running at log end) |

---

## Issue 1 — Resource Arithmetic Failures (Root Cause)

**Severity: Critical.** Every single rejection was "You do not have that many resources to spend." Gemini's fundamental problem is that it cannot maintain accurate bookkeeping of its own MC between turns.

### How it manifests

Gemini receives the current MC balance in each prompt. It performs mental arithmetic to plan a sequence of actions, but the arithmetic is wrong — it thinks it can afford a card it can't. When rejected, it doubles down with long self-justifying reasoning rather than simply picking a cheaper card.

**Example (Gen 1, game g26eefd6f414d):**
```
Turn N:   prompt shows MC:33, Gemini plans "Play Building Industries (6 MC)"
Turn N+1: prompt shows MC:8
Gemini:   "My previous action was intended to be 'Play Building Industries (6 MC)'. 
           If I played that, I should have 27 MC left (from 33 MC previously). 
           The fact I have only 8 MC suggests either I made a different action, 
           or MC was significantly reduced, or the '33 MC' was incorrect."
```

Gemini assumes the prompt is wrong. It played more than Building Industries (the log shows it played multiple cards that gen), but it cannot reconcile the discrepancy because it doesn't track what it has already spent within a generation.

### Why the error feedback loop fails

The `last_error` mechanism sends Gemini the rejection message and asks it to try again. But Gemini re-reasons from stale internal accounting. It tries the same or another similarly over-budget move, loops 3 times, then falls back to Pass. The error does not fix the root cause.

**Gemini's own diagnosis (line 7633):**
> "The game state is persistently inconsistent across turns, and my actions are frequently misattributed or ignored, resulting in a crippled economy despite my resource-generating cards."

This is wrong — the game state is correct. Gemini is blaming the server for its own arithmetic errors.

---

## Issue 2 — Rate Limiting (429 RESOURCE_EXHAUSTED)

**Severity: High.** 817 quota hits over ~1h47m. The quota being exhausted is `GenerateContentPaidTierInputTokensPerModelPerMinute` at 1M tokens/min for gemini-2.5-flash.

This is a **prompt size** problem, not a request frequency problem. The prompts are large because:
- Full hand descriptions (10 cards × ~50 tokens each) every turn
- Long session history (Gemini 1M-token context keeps full history)
- Per-gen strategy updates add 200–400 token responses that stay in context forever

When rate limited, `_gemini_with_retry` backs off 5s/10s/3 attempts, then falls back to a default response. This means several turns per generation are decided by a random fallback rather than actual Gemini reasoning.

**Effect on gameplay:** Rate limit hits cluster toward later generations (context is larger). By gen 10+, a significant fraction of Gemini's "decisions" are actually fallback defaults.

---

## Issue 3 — Strategic Confusion from Tag Attribution

**Severity: Medium.** Gemini confuses its own tags with its opponent's when both players have played cards with the same tags in the same generation.

**Example (Gen 1, game g26eefd6f414d):**
```
Opponent log says: "AI-Red played Phobos Space Haven, gained 1 titanium production"
Gemini's own state shows: titanium production: 2, tags: {'space': 1, 'city': 1}
Gemini concludes: "The presence of 'space' and 'city' tags suggests I played 
                   Phobos Space Haven, despite the log for AI-Red."
```

The recentLog correctly attributes "AI-Red played..." but Gemini second-guesses it when its own state doesn't match its expectations.

---

## Issue 4 — Verbose Reasoning, No Output Discipline

**Severity: Medium.** Gemini's responses are extremely long (hundreds of tokens of chain-of-thought) even when `think=False` was intended. Many responses recalculate the same numbers 2–3 times, then give `CHOICE: N` at the end.

This wastes input tokens in subsequent turns (the reasoning stays in context) and contributes directly to the 429 rate limit problem. The model is also prone to appending additional commentary after the `CHOICE: N` line, which the parser has to handle.

---

## Issue 5 — Per-Generation Strategy Updates Don't Fix Arithmetic

**Severity: Low (design observation).** The per-gen strategy prompt asks Gemini to restate its standing, engine, milestone target, and next-gen priority. It produces good strategic text (mentions specific milestones, correct production values). But this doesn't help with turn-to-turn MC tracking because the strategy is stored as context text, not as a live resource counter.

The per-gen strategy update is working as intended for long-term reasoning. It fails to address the within-generation arithmetic problem, which is structural.

---

## Strategic Quality Observations

Despite the errors, Gemini shows genuine strategic understanding:

**Good:**
- Correctly identifies synergies (e.g., titanium for space cards, science tags for requirements)
- Correctly analyzes card playability given current resources when arithmetic is accurate
- Milestones and awards show up in per-gen strategy with concrete plans
- Gemini reached gen 20 in self-play, demonstrating the game can complete

**Bad:**
- Cannot track spending within a single generation (multiple actions per turn)
- Occasionally tries to use resources for actions that triggered the "payment" prompt when it's already at 0 MC
- In late-game rate-limit collapse, decisions become random (fallback defaults)

---

## Token Usage — Measured via countTokens API

Token counts verified using `POST /v1beta/models/gemini-2.5-flash:countTokens` against actual log prompts.

> **Note:** Google Cloud Monitoring API (which tracks actual quota consumption) requires OAuth2 / service account credentials, not available with the AI Studio API key. Counts below are from `countTokens` + log-derived estimates.

### Per-turn prompt sizes (new user message only)

| Turn position | Prompt tokens | Response tokens |
|---------------|--------------|-----------------|
| Turn 0 (gen 1) | 831 | 7 (CHOICE only) |
| Turn 38 (~gen 5) | 644 | 508 |
| Turn 76 (~gen 10) | 778 | 1,156 |
| Turn 115 (~gen 15) | 797 | 2,065 |
| Turn 153 (~gen 18) | 789 | 902 |
| Turn 192 (~gen 20) | 683 | 1,289 |
| **Average** | **754** | **1,184** |

**Observed response size distribution (219 turns):** min=10 chars, p25=1987, median=3678, p75=4622, max=8686 chars — averaging ~833 tokens/response. The long tail is Gemini re-reasoning from scratch after a 429 fallback.

### Cumulative context growth (critical — causes 429s)

Because it's a chat session, turn N sends **all N-1 previous messages** as input. At ~1938 tokens/turn of context accumulation (754 prompt + 1184 response), each request grows:

| Turn | Cumulative input tokens per API call |
|------|--------------------------------------|
| 10   | ~20,000 |
| 20   | ~40,000 |
| 50   | ~98,000 |
| 100  | ~195,000 |
| 150  | ~291,000 |
| 193  | ~375,000 |

### Total session token consumption (g26eefd6f414d, 193 turns × 2 players)

| Metric | Value |
|--------|-------|
| Estimated total input tokens (both players) | ~72 million |
| Estimated API cost (input only, $0.075/1M) | ~$5.41 |
| Quota metric hit | `GenerateContentPaidTierInputTokensPerModelPerMinute` |
| Quota limit | 1,000,000 tokens/min |
| First 429 hit | ~turn 50 (single player ~98K/request × 2 players × 2 turns/min ≈ 390K/min) |
| 429 rate at turn 193 | ~375K/request × 4 requests/min (2 players, retries) > 1M/min |

### Retry delay analysis

| Metric | Value |
|--------|-------|
| 429 retry delays in API responses | n=564, min=1s, median=40s, max=59s |
| Previous retry waits (5s + 10s) | 15s total |
| Gap to actual quota reset | 25s — all 3 attempts within the same quota window |
| Fallback-to-default events | 184 |
| Wasted API calls (3 per fallback vs 1 needed) | 368 extra calls |

**Key insight:** every failed turn burned 3× quota because the retries happened before the 40s quota window reset. Fixing the retry delay alone eliminates ~73% of wasted quota calls.

---

## Root Cause Summary

| Problem | Root cause | Fix status |
|---------|-----------|------------|
| Resource overspend (60 rejections) | Gemini can't track within-generation MC spending | Open (prompt engineering needed) |
| Rate limiting (817 hits) | Cumulative chat context grows unboundedly; retry waits too short | **Fixed (see below)** |
| Verbose responses inflating context | No output length cap on action turns | **Fixed** |
| All retries inside quota window | Fixed 5s/10s backoff shorter than 40s median quota reset | **Fixed** |
| Tag attribution errors | Opponent log not trusted when own state contradicts it | Open |

---

## Fixes Implemented

### Fix A — Respect API retryDelay on 429 errors (`llm_player.py`)

**Before:** fixed 5s/10s backoff, all 3 attempts within the same quota window.  
**After:** parse `retryDelay` from the error response (`re.search(r"retryDelay.*?(\d+\.?\d*)s", str(exc))`), wait that long + 2s buffer. Default 62s if not parseable.

**Impact:** eliminates 73% of wasted API calls (368 extra calls → ~1 retry per failure). Per-minute quota footprint drops from 3× to 1× on quota-hit turns.

New behavior: 503/overloaded still uses 5s/10s exponential backoff (these don't exhaust quota). Only 429 respects the API-specified delay.

### Fix B — Cap action-turn output to 350 tokens (`llm_player.py`)

**Before:** no limit; median response was ~920 tokens, max 2,172 tokens.  
**After:** `chat.send_message(user, config=GenerateContentConfig(max_output_tokens=350))` for every action turn. Per-gen strategy updates and setup remain uncapped.

**Impact on context growth:** avg context accumulation/turn drops from 1,938 tok to ~1,100 tok (754 prompt + 350 capped response). Turn-193 cumulative input drops from 375K → **213K tokens** — a 43% reduction.

Also added prompt suffix: *"Brief reasoning (1-2 sentences), then CHOICE: N on its own line. No text after CHOICE."* — encourages concise outputs even before the cap applies.

**Env override:** `GEMINI_ACTION_MAX_OUTPUT_TOKENS=350` (default).

### Fix C — Gemini session history trimming after 80 turns (`llm_player.py`)

**Before:** full chat history kept forever (comment said "no trim needed for 15-gen game", but self-play runs 20+ gens × 2 players).  
**After:** `_trim_gemini_session()` — when `_gemini_turn_count[game_id] >= _MAX_GEMINI_TURNS` (default 80), keep last 40 message pairs + a synthetic strategy-summary pair at the front. Rebuilds chat with same cache/system.

**Impact:** worst-case per-request input tokens after trim = 40 × 1,100 + 754 = **44,754 tokens** — well within the 1M/min budget for 4 concurrent players.

**Env override:** `GEMINI_MAX_TURNS=80` (default).

---

## Remaining Open Issues

### Priority 1 — Show spent-MC balance during action phase
The only reliable fix for resource overspend: track cumulative MC spent this turn and add "MC spent this turn: X. Remaining: Y" to each action prompt. This requires the TM server to expose an "actions taken this turn" counter, or the AI server to reconstruct it from the payment decision history.

### Priority 2 — `_fallback_pass()` picks wrong option on failure
`_fallback_pass()` uses the last `or` option as "Pass", but this can be "Sell Patents" which also fails if empty. Fix: search options by title substring for "pass" before falling back to last index.

### Priority 3 — Elide full card descriptions on repeated turns
Cards shown during research phase are re-sent with full descriptions every subsequent action turn. After the research phase, they should appear as name-only with `"(see research phase above)"`. Already partly done for `space`/`payment`/`amount` decision types; extend to all action turns when card was already described this gen.

---

## Session Infrastructure Notes

- **Context caching** worked correctly: 1 cache created per game, refreshed twice at ~50 min intervals, no rebuild needed.
- **Session recovery** triggered once at start (server restarted between sessions); strategy was preserved and reinjected — recovered correctly.
- **Per-gen strategy updates** reached gen 20 without drift issues.
- **Cloud Monitoring API** (`monitoring.googleapis.com`) returns 401 with an AI Studio key — requires OAuth2 service account. No GCP project is configured. Actual quota consumption can only be checked at `console.cloud.google.com` or `ai.dev/rate-limit`.
