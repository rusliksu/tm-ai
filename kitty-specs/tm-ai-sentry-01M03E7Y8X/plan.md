# План подключения Sentry к tm-ai-server

## Technical Context

**Language/Version**: Python 3.11+; FastAPI/Uvicorn; `sentry-sdk` 2.x
**Primary Dependencies**: FastAPI, Starlette, Pydantic, pytest, ruff, uv
**Storage**: существующий файловый `logs/llm-state`; изменений нет
**Testing**: pytest и fake Sentry transport без сети
**Target Platform**: локальный/VPS Linux process, loopback-first
**Project Type**: web service / FastAPI API
**Performance Goals**: error reporting best-effort, bounded flush не блокирует обычный ответ
**Constraints**: OpenRouter payload и state должны оставаться локальными; секреты не читаются и не логируются
**Scale/Scope**: один FastAPI process, четыре публичные route-группы

## Архитектура

1. `src/tm_llm/sentry.py` содержит typed env gate, no-op reporter, sanitizer и fake
   transport seam для тестов.
2. `app.py` устанавливает единый FastAPI exception handler: ожидаемые `HTTPException`
   4xx проходят без capture, неожиданные ошибки capture-ятся после локального логирования.
3. lifespan получает startup boundary с единым reporter и bounded flush; ошибки до
   импорта приложения не маскируются как Sentry-captured.
4. `sentry-sdk` используется без `send_default_pii`, tracing, profiling и logs; request
   integration не включается автоматически.
5. `docs/integrations/sentry.md` фиксирует env contract и отдельные delivery gates.

## Privacy

Sanitizer строит новый event из allowlisted полей. Error type/function/value/message,
request/user/headers/cookies/query/body и произвольные contexts не проходят в transport.
DSN валидируется строго: lowercase HTTPS, raw public key `[A-Za-z0-9_]+`, host и numeric
project id, без query/fragment/credentials.

## Проверки

- `uv run --project tm-ai-server pytest -q tm-ai-server/tests`
- `uvx ruff check tm-ai-server/src/tm_llm/sentry.py tm-ai-server/tests/test_sentry.py`
- `uv run --project tm-ai-server python -m compileall -q tm-ai-server/src`
- privacy fake-transport negative oracle
- `git diff --check`

Общий live/deploy gate не выполняется этим планом.
