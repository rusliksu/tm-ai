# LLM Death Match — 5-Player Cost Estimate

## Game setup

```bash
source /home/pmunk/workspace/tm-ai/.env
cd tm-ai-server && USE_LLM=true OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000 &

uv run python ../scripts/play_game.py --players 4 \
  --models "anthropic/claude-sonnet-4-6,openai/gpt-4o-mini,google/gemini-flash-latest,deepseek/deepseek-v3"
```

## Model mapping

| User request | OpenRouter ID | Note |
|---|---|---|
| claude opus 4.6 | `anthropic/claude-sonnet-4-6` | downgraded from opus; ~40% cheaper per token |
| gpt-5.5 pro | `openai/gpt-4o-mini` | replaced earlier: gpt-5.5-pro $63 → gpt-4o $8 → mini $0.40 |
| gemini-3.1-pro | `google/gemini-flash-latest` | downgraded from pro; router → latest flash |
| deepseek v4 pro | `deepseek/deepseek-v3` | downgraded from v4-pro; no thinking → fast responses |
| grok 4.2 | _(removed)_ | dropped to stay within budget |

## Cost estimate per game

Assumptions: ~150 API calls per player, average prompt ~18K tokens (4 players → smaller
opponent section than 5-player), ~200 output tokens per call, ~2.5M total input tokens
per player.

| Player | Model | $/M in | $/M out | Caching | Est. $/game |
|---|---|---|---|---|---|
| Claude Sonnet 4-6 | `anthropic/claude-sonnet-4-6` | $3 | $15 | yes (10% on cached) | ~$4 |
| GPT-4o-mini | `openai/gpt-4o-mini` | $0.15 | $0.60 | auto | ~$0.40 |
| Gemini Flash Latest | `google/gemini-flash-latest` | ~$0.30 | ~$1.20 | no | ~$1 |
| DeepSeek V3 | `deepseek/deepseek-v3` | $0.27 | $1.10 | auto | ~$0.70 |
| **Total** | | | | | **~$6.10/game** |

Within the $9 budget with room to spare.

## Pricing source

Fetched from `https://openrouter.ai/api/v1/models` on 2026-05-17.
