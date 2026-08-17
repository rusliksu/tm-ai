# Пакеты реализации: Sentry для tm-ai-server

**Миссия:** `tm-ai-sentry-01M03E7Y8X`
**Ветка:** `codex/tm-ai-sentry`
**Bead:** `tm-ai-0or`
**Режим:** env-gated observability; live delivery отдельно.

## Порядок

1. WP01 — reporter, DSN gate и FastAPI boundary.
2. WP02 — privacy oracle и регрессионные тесты.
3. WP03 — документация, lockfile и quality gates.

## Подзадачи

| ID | Описание | WP |
|---|---|---|
| T001 | Добавить `sentry-sdk` и typed env/no-op reporter | WP01 — готово |
| T002 | Подключить capture к FastAPI 500 и startup boundary | WP01 — готово |
| T003 | Добавить fake transport и privacy negative oracle | WP02 — готово |
| T004 | Проверить 4xx/health/version no-capture и regression | WP02 — готово |
| T005 | Обновить env-документацию, lockfile и прогнать quality gates | WP03 — готово |

## Acceptance

- no/malformed DSN не вызывает транспорт;
- неожиданный 500/startup дают одно privacy-safe событие;
- 400/422/health/version не создают событие;
- targeted tests, ruff, compileall и diff-check зелёные;
- нет live DSN/token и нет deploy/restart в этой ветке.
