# TM-AI Model Design Rationale

> Source material for a future blog post. Extracted verbatim from the implementation plan (`~/.claude/plans/fluffy-yawning-creek.md`).

---

## Design rationale (extract to ai-model-design.md)

### Why the current architecture is undersized

`PolicyValueNet` is a vanilla MLP: state vector (492 floats) → 3 dense layers × 512 units → 128 action logits + 1 value scalar. State is packed as a flat list of numbers; the model sees no structure. Two specific blind spots:

**1. Per-card resources are wasted slots.** For each player, 199 floats hold "how many resource cubes are on card *i*", normalized by 20. Two problems:
- **Sparsity:** a player owns ~10 of 199 cards, so ~95% of slots are zero every turn. Very sparse gradient signal.
- **No similarity prior:** *Tardigrades* (microbe-storer) and *Ants* (microbe-storer) are in totally separate dimensions; the network must discover their similarity from data alone.

**2. The board is not encoded at all.** The hex grid with tile positions is in the request but `encode_state()` ignores it. The model can't see where greeneries are placed, which ocean rows are filled, or who controls which spaces.

### Card embeddings — what they are

**Important sizing distinction.** Two different card counts exist:
- `CARD_RESOURCE_VOCAB` = **199** cards — only those that *store resources on themselves* (microbe-, animal-, science-, floater-storers). This is what the current 199-dim per-card resource block addresses.
- `CARD_DB` = **970** cards — every card in the game across base + all expansions.

The embedding table must cover **all 970 cards** plus a reserved index 0 for "unknown / padding" → `nn.Embedding(971, 32)`. The 199 resource-bearing cards remain a *subset* used only for the resource-count aggregation; the embedding table itself is over the full 970.

Replace the 199-dim one-hot resource block (and the lone played-card count) with the learned embedding table. Each card has a 32-float learnable row. At encode time:

- **Played cards:** gather rows for the player's played cards by global card ID (1..970) → mean-pool → one 32-dim "tableau summary" vector
- **Hand cards:** gather rows for cards in hand → mean-pool → one 32-dim "hand summary" vector
- **Resource-bearing cards:** for each card with resources (subset of 199), look up its row from the same 971-table and weight by resource count → sum → one 32-dim "resource-card summary" vector

**Padding bounds** (chosen to comfortably exceed realistic game state):
- `MAX_HAND_CARDS = 40` — large enough for any normal hand including Inventrix/research-heavy/deferred-pickup edge cases
- `MAX_PLAYED_CARDS = 80` — covers very long games and tableau-heavy strategies

Padded slots use the reserved index 0; `nn.Embedding(padding_idx=0)` ensures pad rows contribute zero to mean-pool automatically.

What this buys:
- Embeddings for *Tardigrades* and *Ants* can naturally become similar during training — both contribute to similar reward outcomes, gradient descent pushes their rows together. The "these cards are similar" prior emerges for free.
- Generalization to unseen tableaux composed of seen cards.
- Fewer parameters in the input layer; denser gradient signal.

### Tensor layout

PyTorch tensors are typed N-dim arrays. The pipeline today:

```
state          tensor[batch, 492]  float32   ← what model.forward() takes
mask           tensor[batch, 128]  bool      ← which options are legal
policy logits  tensor[batch, 128]  float32   ← output, one per action slot
value          tensor[batch]       float32   ← output, scalar per state
```

With card embeddings + spatial board + per-option scoring (Option C):
```
state_other          tensor[batch, ~120]      float32   ← resources, prod, tags, milestones
played_ids_self      tensor[batch, 80]        long      ← padded
hand_ids_self        tensor[batch, 40]        long
played_ids_opp       tensor[batch, 80]        long
resource_card_ids    tensor[batch, 199]       long
resource_card_counts tensor[batch, 199]       float32
board_grid           tensor[batch, 6, 9, 9]   float32   ← 6 channels × 9×9 hex grid
option_features      tensor[batch, 128, F]    float32   ← F≈64, per-option features
mask                 tensor[batch, 128]       bool
embedding_table      nn.Embedding(971, 32)              ← learnable
```

Inside the model:
```
played_emb_self = embedding(played_ids_self).mean(dim=1)   # → [batch, 32]
hand_emb_self   = embedding(hand_ids_self).mean(dim=1)     # → [batch, 32]
played_emb_opp  = embedding(played_ids_opp).mean(dim=1)    # → [batch, 32]
resource_emb    = (embedding(resource_card_ids) *
                   resource_card_counts.unsqueeze(-1)).sum(dim=1)  # → [batch, 32]
board_feat      = board_conv(board_grid).flatten(1)        # → [batch, 128]
combined        = cat([state_other, played_emb_self, hand_emb_self,
                       played_emb_opp, resource_emb, board_feat], dim=1)
                                                            # → [batch, ~310]
state_embed     = backbone(combined)                       # → [batch, 768]
# Per-option scoring (Option C):
state_expanded  = state_embed.unsqueeze(1).expand(-1, 128, -1)
                                                            # → [batch, 128, 768]
slot_input      = cat([state_expanded, option_features], -1)
                                                            # → [batch, 128, 832]
slot_hidden     = scoring_mlp_layer1(slot_input)           # Linear(832, 256), ReLU
logits          = scoring_mlp_layer2(slot_hidden).squeeze(-1)
                                                            # → [batch, 128]
logits          = logits.masked_fill(~mask, -inf)
value           = value_head(state_embed).squeeze(-1)      # → [batch]
```

The first axis is always batch — that's how parallelism works on GPU/CPU. Typically batch=64 or 256.

### Why "slot N" of the policy output changes meaning every turn

`flatten_options()` walks the `waitingFor` decision tree depth-first and emits a flat list of leaf choices. That list is **freshly constructed each turn** from the current decision context. The model's `tensor[128]` output assigns one logit per slot, but the **meaning of each slot is set externally** by the order `flatten_options` happens to emit options.

Concrete example. Turn 23 — "which standard project?":
```
waitingFor = OrOptions([
  SelectOption("Sell Patents"),         # → slot 0
  SelectOption("Power Plant: pay 11"),  # → slot 1
  ...
  SelectOption("Greenery: pay 23"),     # → slot 7   ← slot 7 = Greenery
])
```

Turn 24 — "which project card to play from hand?":
```
waitingFor = OrOptions([
  SelectProjectCardToPlay("Nuclear Power"),  # → slot 0
  ...
  SelectProjectCardToPlay("Search for Life"), # → slot 7   ← slot 7 = Search for Life
])
```

Same slot, totally different action, totally different context. The model has no way of being told "slot 7 means Greenery this time" — all it sees is "produce 128 floats; the system picks the highest legal one". So it has to learn "given this state pattern, prefer-whatever-is-at-slot-7" — which only works if `flatten_options` enumerates in a consistent order. Reorder the option list and the model has to relearn. The model can't generalize "playing a Greenery is good at oxygen=10" because there's no notion of "Greenery" in the action representation — only positions in an arbitrary list.

### Option C — per-option scoring (the chosen architecture)

Per-option scoring replaces "one head that emits 128 unrelated logits" with "one head that scores each option given the state and the option's own feature vector".

**Step 1: per-option features.** `flatten_options` is extended to emit a feature vector for each leaf:

| Option type | Feature vector contents |
|---|---|
| `SelectOption("Greenery: pay 23")` | type=[1,0,0,0,0], parsed_cost=23, card_emb=zeros |
| `SelectProjectCardToPlay("Tardigrades")` | type=[0,1,0,0,0], cost=4, card_emb=embedding["Tardigrades"] |
| `SelectSpace("H05")` | type=[0,0,1,0,0], x=3, y=2, tile_type_onehot |
| `SelectAmount(n)` | type=[0,0,0,1,0], normalized_amount=n/cap |
| `SelectPlayer("blue")` | type=[0,0,0,0,1], player_id_onehot |

Stacked into `tensor[batch, 128, F]` where `F` ≈ 64 features. Empty slots zeroed.

**Step 2: state encoder.** Produces `tensor[batch, 768]` state embedding from cards, board, resources, tags.

**Step 3: scoring head.** A 2-layer MLP `Linear(832 → 256) → ReLU → Linear(256 → 1)` scores `[state_embed; option_feat[i]]` for each slot, weights **shared across all 128 slots** — same function applied 128 times.

**Why it's the right endpoint:**
- Slot meaning lives in features, not index. Reordering options doesn't change `score(state, "Greenery")` whether Greenery is at slot 3 or 11. Order-invariance.
- True generalization: model can transfer "playing a microbe card with 4+ microbes is good in late-game" to any card option, because every card option carries its embedding and microbe-storer embeddings cluster.
- Scales naturally to action sets the model has never seen exactly, as long as the features are familiar.
- Subsumes Option A: card embeddings + spatial board are still needed (they're the state encoder); per-option scoring sits on top.

### Does Option C still need 128 output slots?

**Yes — PyTorch shape constraint — but the cap is decoupled from model capacity.**

The reason we need *some* fixed slot count is that batched ops require `forward()` to return `tensor[batch, MAX_ACTIONS]` with `MAX_ACTIONS` as a compile-time constant. Avoiding this requires either single-example processing (slow) or ragged tensors (complex).

The crucial difference between Option A and C is what changing `MAX_ACTIONS` costs:

- **Option A:** policy head is `Linear(D_hidden, MAX_ACTIONS)`. Doubling 128 → 256 doubles those parameters, and each slot has its own dedicated weights that must learn its rotating meaning from scratch.
- **Option C:** the scoring MLP is shared. Going 128 → 256 → 512 adds zero parameters; it just costs more FLOPs (the MLP runs N times instead of 128).

Empirical max from `flatten_options`:
- Card-play (`projectCard`): hand size × 1 entry per card → typically 5–15 options
- Standard project: ~5 options
- Action-phase `OrOptions`: card actions + play card + standard project + pass → typically 20–40
- Pathological worst case (many simultaneous card actions, deep nested Or-And): could approach 60–80

128 is a comfortable safety margin. **First implementation step: instrument** `len(flatten_options(...))` over the existing 1540 training logs to find empirical max. If <100, keep 128. If approaching 128, bump to 256 (essentially free in Option C).

### Cost terminology — what "implementation days" means

The day estimates throughout this plan are **implementation effort** (Sonnet writing code, running tests, fixing bugs), not training time. Training wall time is a separate budget covered in Part B.

### Architectures considered and rejected

- **Option A (card embeddings + spatial board, fixed 128 policy head):** addresses both encoding blind spots cheaply but doesn't fix the rotating-slot-meaning learning problem. The user opted for Option C as the proper endpoint.
- **Option B (card embeddings only, no board):** insufficient — leaves the board blind spot, unlikely to fix the gen-70 stall.
