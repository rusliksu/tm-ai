# LLM Death Match — 5-Player Cost Estimate

## Game setup

```bash
source /home/pmunk/workspace/tm-ai/.env
cd tm-ai-server && USE_LLM=true OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000 &

uv run python ../scripts/play_game.py --players 5 \
  --models "anthropic/claude-opus-4.7,openai/gpt-5.5-pro,google/gemini-pro-latest,deepseek/deepseek-v4-pro,x-ai/grok-4.3"
```

## Model mapping

| User request | OpenRouter ID | Note |
|---|---|---|
| claude opus 4.6 | `anthropic/claude-opus-4.7` | 4.6 non-fast doesn't exist; 4.7 is latest |
| gpt-5.5 pro | `openai/gpt-5.5-pro` | exact match |
| gemini-3.1-pro | `google/gemini-pro-latest` | 3.1-pro not available; router → latest pro |
| deepseek v4 pro | `deepseek/deepseek-v4-pro` | exact match |
| grok 4.2 | `x-ai/grok-4.3` | 4.2 doesn't exist; 4.3 is newest |

## Cost estimate per game

Assumptions: ~150 API calls per player, average prompt ~20K tokens (growing context,
trimmed at 62 messages), ~200 output tokens per call, ~3M total input tokens per player.

| Player | Model | $/M in | $/M out | Caching | Est. $/game |
|---|---|---|---|---|---|
| Claude Opus 4.7 | `anthropic/claude-opus-4.7` | $5 | $25 | yes (10% on cached) | ~$8 |
| GPT-5.5 Pro | `openai/gpt-5.5-pro` | $30 | $180 | auto | ~$63 |
| Gemini Pro Latest | `google/gemini-pro-latest` | $2 | $12 | yes | ~$6 |
| DeepSeek V4 Pro | `deepseek/deepseek-v4-pro` | $0.44 | $0.87 | auto | ~$1.35 |
| Grok 4.3 | `x-ai/grok-4.3` | $1.25 | $2.50 | yes | ~$4 |
| **Total** | | | | | **~$82/game** |

**GPT-5.5-pro dominates at $30/M input** — roughly 6× Claude Opus and 70× DeepSeek.
Without it, the other four players cost ~$19/game combined.

Cheaper alternative for the GPT slot: `openai/gpt-5.4` or `openai/gpt-4o`.

## Pricing source

Fetched from `https://openrouter.ai/api/v1/models` on 2026-05-17.
