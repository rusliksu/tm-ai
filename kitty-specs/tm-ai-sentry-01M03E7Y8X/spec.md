# Подключение Sentry к tm-ai-server

## Цель

Подключить Sentry к FastAPI-серверу Terraforming Mars AI так, чтобы неожиданные ошибки
из `/move`, ошибки запуска и необработанные runtime-исключения были видны в Sentry,
но без передачи OpenRouter-ключа, запросов игроков, состояния игры, cookies, headers,
IP, query/body и произвольных текстов ошибок.

## Границы

В scope входят только `tm-ai-server`, env-конфигурация, error boundary, privacy sanitizer,
тесты и операторская документация. Существующие OpenRouter, state persistence, game
logic и `/health`/`/version` контракты не меняются. Live DSN, создание Sentry-проекта,
push/merge и deployment остаются отдельными воротами.

## Требования

### FR-001 — Env-gated SDK

Без `TM_AI_SENTRY_DSN` или с malformed DSN приложение работает как no-op и не выполняет
сетевых вызовов Sentry. `TM_AI_SENTRY_ENVIRONMENT` и `TM_AI_SENTRY_RELEASE` принимаются
только в безопасной однострочной форме.

### FR-002 — Ошибки API

Неожиданный 500 из FastAPI создаёт одно событие с service/runtime/error_kind. Ожидаемые
400/422 и `/health`/`/version` не создают события.

### FR-003 — Ошибки запуска и lifecycle

Исключение в startup/lifespan-контуре логируется локально и best-effort передаётся
в Sentry с bounded flush; Sentry не задерживает завершение процесса. Ошибки до
импорта приложения остаются вне этого boundary и обрабатываются операторским логом.

### FR-004 — Privacy fail-closed

Перед отправкой envelope удаляются `message`, `request`, `user`, `extra`, `contexts`,
`breadcrumbs`, headers, cookies, query, body и произвольные stack values. В событии
остаются только фиксированные service/runtime/error_kind, безопасные environment/release
и ограниченные source frame coordinates без function/value.

### FR-005 — Независимый oracle

Тест с fake transport проверяет фактический envelope, минимум с sentinels для API key,
Authorization, Cookie, IP, query, body, nested unknown и произвольного Error message.

## Не в scope

- performance tracing, profiling, logs и send-default-Pii;
- автоматическая отправка prompts, game state, OpenRouter responses или user identity;
- live deploy, restart, push, PR, merge и создание Sentry project;
- изменение публичного HTTP-формата tm-ai.

## Критерии приёмки

- целевые `pytest`/`ruff` проверки затронутого boundary проходят; полный repo-wide
  ruff остаётся отдельным baseline из существующих нарушений;
- malformed/no DSN подтверждён no-op тестом;
- 500/startup capture и 4xx no-capture подтверждены fake transport;
- итоговый envelope не содержит запрещённых sentinels;
- в diff нет DSN/token; рабочее дерево и ветка изолированы.
