# TODO — LLM-player improvements

Derived from analysis of game log `logs/llm-test/2026-06-07T14-01-03Z/`
(UI game: Kai = `deepseek/deepseek-v4-flash` vs humans Peter & Sandra, gens 3→11).
**Final score: Kai 33 VP — Sandra 90 — Peter 99 (blowout loss).**

The losses come from one critical mechanical bug, several prompt/feedback-loop bugs,
and a weak model. Items are ordered by impact. File refs are
`tm-ai-server/src/tm_llm/`.

---

## P0 — mechanical correctness (highest impact; helps every model)

- [x] **Standard projects always plays Power Plant** — `options.py:flatten_options`
  - TM's "Standard projects" is a single `projectCard` node listing
    `[Power Plant, Asteroid, Aquifer, Greenery, City]`. `flatten_options` only expands
    nested `or` nodes whose children are all bare `option` types, so the LLM sees only
    `"3. Standard projects"` and `index_to_response` → `_default_response(projectCard)`
    returns the FIRST card = Power Plant **every time**.
  - Evidence: fired 8× (88 MC burned on useless +1 energy). Gen-10 (ai-server.log:6900)
    LLM wrote "Place ocean via SP … CHOICE: 3" → engine logged
    `Auto-payment for 'Power Plant:SP'`.
  - Fix: expand a `projectCard` child of the top OR into one option per card
    (`"Standard projects: Aquifer (ocean)"`, path `[i, j]`). `index_to_response` already
    handles the `or → projectCard` path; `parse_action_response` resolves the named card.
  - Consider also expanding the hand "Play project card" `projectCard` node so card choice
    + payment are explicit.
  - Add regression test in `tests/test_options.py` ("Standard projects expands to N options").

- [x] **Board adjacency drops opponent greeneries + special tiles** — `board.py:render_space_choices`
  - Adjacency loop only counts own cities, own greeneries, opp cities, oceans. Opponent
    greeneries and ALL special tiles are silently dropped → every candidate hex reports
    "no adjacent tiles" even with 12+ opponent greeneries on the board.
  - Evidence: user's notes.txt; `render_live_board` (correct) shows the greeneries the
    space-choice render omits. Blinds the model on city/greenery placement = where board VP
    is won.
  - Fix: count + describe opponent greeneries and special tiles
    ("opponent greenery ×N", "your nuclear zone").
  - Add regression test in `tests/test_board.py` ("opponent greenery shown adjacent").

---

## P1 — feedback-loop / memory hygiene

- [x] **Per-gen strategy update returns action-format, polluting memory** —
  `engine._maybe_per_generation_update` + `prompts.build_pergen_prompt`
  - The model answered the 7-point strategy prompt in action format
    (`**Reasoning** / **TACTICAL** / **CHOICE:** 15 / **PAYMENT:** MC=8`); engine saved it
    verbatim as `player.strategy`, so a stale meaningless `CHOICE: 15` echoed into EVERY
    later turn. Model also kept copying the `----- PRIOR STRATEGY -----` header
    (nested duplication, token bloat).
  - Evidence: ai-server.log:6478.
  - Fix: instruct per-gen prompt to emit prose strategy ONLY; strip
    `CHOICE:`/`PAYMENT:`/stray option-number lines before saving; de-dupe the
    prior-strategy header.

- [x] **Free-city hallucination loop** — `prompts.build_pergen_prompt` DEFERRAL CHECK + corp note
  - Identical "I must execute my pending free city placement immediately…" repeated gens 5–8
    (ai-server.log:1590,1598,1640,1706,1771,1831,1901,1975,2064,2149…). Tharsis Republic has
    NO deferred free-city action (+1 MC per city; the one free city happens only at game
    start). The DEFERRAL CHECK ("execute the undone goal as your FIRST action") reinforced
    the impossible goal → infinite re-deferral until gen 9.
  - Fix: soften DEFERRAL CHECK to "if it's still legal AND offered in the options, do it;
    if not offered, drop it." Add a one-line corp-ability clarification
    (Tharsis = +1 MC/city, no recurring free city). Reconcile strategy against actual
    options / recent log.

- [x] **No game-end-proximity awareness** — `prompts.build_action_prompt` /
  `build_pergen_prompt`
  - At final gen 11 (temp maxed, O₂ maxed, oceans 7/9) the model wrote "save MC for next gen"
    (ai-server.log:7217,7280). TM has no fixed generation count; nothing signals the game is
    about to end when the last oceans drop (likely placed by an opponent that gen).
  - Fix: add a banner when ≥2 of 3 global params are maxed:
    "Game may END this generation — spend ALL resources on VP now, save nothing."
    Push end-game greenery / Venus VP.

---

## P2 — strategic depth

- [x] **Award realism + opponent-engine digest** — `prompts.compute_award_standings`,
  `build_action_prompt`
  - Kai funded Banker for 20 MC on a 9-vs-8 MC-prod lead (gen 5); by gen 11 Peter's MC prod
    was 27 vs Kai's 14 → Kai LOST the award it paid 20 MC for. Standings show only the
    current snapshot, no trajectory/defensibility, and the model can't read opponent engines.
  - Fix: (a) state awards are scored at game END and current leads erode — don't fund thin or
    early (<~gen 8) leads unless structural; (b) add a compact opponent-engine digest
    (production deltas + key board tiles, derived from `state.opponents` — cheap, no token
    explosion) so the model judges whether a lead holds and reads opponents
    (cf. "lokaler Schatten" every-other-gen income).

- [x] **Stronger early milestone targeting** — `prompts.build_setup_prompt` +
  earlier advisory
  - By gen 8 all 3 milestones were claimed by opponents; Kai got 0. Setup asks for
    "milestone/award targets" but doesn't force a concrete reachable pick + path.
  - Fix: setup names ONE reachable milestone + the tag/tile count and gen to claim it by;
    add an earlier "milestones closing — N left" advisory.

- [ ] **Validate harness fixes with a strong model** (not yet done — requires running a game)
  - This game used `deepseek/deepseek-v4-flash` (cheap `start.sh` default), not the
    `config.py` default `anthropic/claude-opus-4-7`. After P0/P1, re-run the SAME board with a
    strong model (opus-4-7 or sonnet-4-6) to separate "model quality" from "harness bugs".

---

## Notes / minor

- Junk-move floundering: 4 consecutive "Sell patents" (gens ~9, ai-server.log:4888–5593) +
  8 Power-Plant misfires = large fraction of 66 action turns wasted. Mostly downstream of the
  P0 option bug + weak model.
- Restore reflection gap: restored at gen 3 with state at gen 5 → single `3→5` bump, one
  generation's strategy reflection skipped. Minor; consider reflecting per missed generation.
