# Sentry для tm-ai-server

Интеграция включается только переменной `TM_AI_SENTRY_DSN`. Пустой или malformed DSN
оставляет reporter no-op и не создаёт сетевых вызовов.

```text
TM_AI_SENTRY_DSN=<DSN проекта>
TM_AI_SENTRY_ENVIRONMENT=staging|production
TM_AI_SENTRY_RELEASE=<git SHA или безопасный идентификатор релиза>
```

Захватываются только неожиданные 500 и ошибки startup/lifespan. Ошибки 400/422, 404,
`/health` и `/version` не создают события. Перед фактическим envelope удаляются message,
request, user, headers, cookies, query, body, contexts, extra, breadcrumbs, Error value и
function names. Tracing, profiling, logs и default PII отключены.

Проверка использует реальный `sentry-sdk` client с fake transport и не требует сети:

```text
uv run --project tm-ai-server pytest -q tm-ai-server/tests/test_sentry.py
uvx ruff check tm-ai-server/src/tm_llm/sentry.py tm-ai-server/tests/test_sentry.py
```

Создание Sentry-проекта, ввод live DSN, push/merge и deployment/restart остаются отдельными
воротами и этим commit не выполняются.
