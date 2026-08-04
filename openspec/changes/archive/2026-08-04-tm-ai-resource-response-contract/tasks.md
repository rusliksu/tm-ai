## Общие ограничения

- Изменяются только resource builders и их offline tests; TM server wire contract остаётся `{type: "resource", resource: <unit>}`.
- Допустимые значения берутся только из `include` либо совместимого `resources`; пустой список завершается `ActionContractError("no_choices")`.
- Model routing, prompts, budgets, retry, gameplay policy и dependencies не меняются; provider inference, commit, push и pin update запрещены этим change.

## 1. Контракт и тесты

- [x] 1.1 Получить одобрение offline delta для canonical resource response и server-authoritative arena checkpoint.
  - Evidence 2026-08-04: Руслан подтвердил `одобряю delta resource/save checkpoint offline`; разрешение не включает provider inference, paid resume, commit, push или deploy.
- [x] 1.2 Добавить red-first regressions для root/nested action-contract и legacy resource paths.
  - Интерфейсы: server-shaped resource node → exact expected TM `InputResponse`.
  - Проверка: `uv run pytest tests/test_action_contract.py tests/test_options.py -q` до исправления должен показать целевые падения.
  - Evidence: первый focused run дал ожидаемые `8 failed, 42 passed`, включая несовместимый v2 output, отсутствие legacy `include` и выдуманный default.
- [x] 1.3 Исправить все resource response builders на exact поле `resource`, сохранив input allowlist `include`/`resources`.
  - Интерфейсы: `action_contract._build_node`, `flatten_options`, `index_to_response`, `_default_response` → canonical resource response либо `ActionContractError("no_choices")`.
  - Проверка: focused suite завершается без `resourceType` в production source.
  - Evidence: focused suite после исправления — `50 passed`; `resourceType` отсутствует в production source.
- [x] 1.4 Подтвердить новый shape независимым canonical server-shaped validator.
  - Проверка: compiled `isSelectResourceResponse` принимает новый exact shape и отклоняет legacy shape.
  - Evidence: фактический compiled `isSelectResourceResponse` принял `{type:'resource',resource:'floaters'}` и отклонил legacy `{type:'resource',resourceType:'floaters'}`.

## 2. Офлайн-приёмка

- [x] 2.1 Выполнить focused resource tests и полный pytest без provider credentials.
  - Проверка: `uv run pytest tests/ -q` и `uv run python -m compileall -q src tests`.
  - Evidence: focused `50 passed`, полный suite `106 passed`, compileall exit `0`; provider credentials очищены в process environment, provider invocations=`0`.
- [x] 2.2 Выполнить `git diff --check`, языковую проверку и строгую OpenSpec validation.
  - Evidence: `git diff --check`, `Test-OpenSpecRussian.ps1` (`4` файла) и `openspec validate tm-ai-resource-response-contract --strict --no-interactive` завершились exit code `0`.
- [x] 2.3 Зафиксировать evidence и остановиться перед commit, push, pin update и любым provider call.
  - Evidence: task-owned branch остаётся unpublished и uncommitted; изменены только два source-файла, два test-файла и отдельный OpenSpec change, provider invocations=`0`.
