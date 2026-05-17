# LLM Death Match — 5-Player Cost Estimate

## Game setup

```bash
source /home/pmunk/workspace/tm-ai/.env
cd tm-ai-server && USE_LLM=true OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000 &

uv run python ../scripts/play_game.py --players 5 \
  --models "anthropic/claude-opus-4.7,openai/gpt-4o,google/gemini-pro-latest,deepseek/deepseek-v4-pro,x-ai/grok-4.3"
```

## Model mapping

| User request | OpenRouter ID | Note |
|---|---|---|
| claude opus 4.6 | `anthropic/claude-opus-4.7` | 4.6 non-fast doesn't exist; 4.7 is latest |
| gpt-5.5 pro | `openai/gpt-4o` | replaced: gpt-5.5-pro costs ~$63/game; gpt-4o ~$8 |
| gemini-3.1-pro | `google/gemini-pro-latest` | 3.1-pro not available; router → latest pro |
| deepseek v4 pro | `deepseek/deepseek-v4-pro` | exact match |
| grok 4.2 | `x-ai/grok-4.3` | 4.2 doesn't exist; 4.3 is newest |

## Cost estimate per game

Assumptions: ~150 API calls per player, average prompt ~20K tokens (growing context,
trimmed at 62 messages), ~200 output tokens per call, ~3M total input tokens per player.

| Player | Model | $/M in | $/M out | Caching | Est. $/game |
|---|---|---|---|---|---|
| Claude Opus 4.7 | `anthropic/claude-opus-4.7` | $5 | $25 | yes (10% on cached) | ~$8 |
| GPT-4o | `openai/gpt-4o` | $2.50 | $10 | auto | ~$8 |
| Gemini Pro Latest | `google/gemini-pro-latest` | $2 | $12 | yes | ~$6 |
| DeepSeek V4 Pro | `deepseek/deepseek-v4-pro` | $0.44 | $0.87 | auto | ~$1.35 |
| Grok 4.3 | `x-ai/grok-4.3` | $1.25 | $2.50 | yes | ~$4 |
| **Total** | | | | | **~$27/game** |

All five players are now in the same cost bracket ($1–8/game each).
GPT-5.5-pro ($63/game alone) was swapped out for GPT-4o ($8/game).

## Pricing source

Fetched from `https://openrouter.ai/api/v1/models` on 2026-05-17.
