## Почему

Canonical Terraforming Mars server принимает ответ выбора ресурса только в форме `{type: "resource", resource: <unit>}`. Текущий tm-ai в action-contract/v2 и legacy mapper возвращает поле `resourceType`, поэтому сервер отклоняет структурно понятный модели ход с HTTP 400. Ошибка уже воспроизведена на сохранённой партии и подтверждена исходниками сервера.

## Что изменится

- Исправить только wire-поле ответа `resourceType` на canonical `resource` во всех mapper paths tm-ai.
- Сохранить server-owned allowlist ресурса: canonical `include` и совместимый legacy fallback `resources`.
- Добавить офлайн-регрессии для корневого и вложенного resource input, а также legacy index/default paths.
- Подтвердить canonical server-shaped replay без provider inference.

## Возможности

### Новые возможности

- `resource-response-contract`: точное построение и офлайн-проверка canonical TM `resource` response.

### Изменённые возможности

- Нет: внешний контракт TM server не меняется, tm-ai лишь начинает ему соответствовать.

## Влияние

- Код: `tm-ai-server/src/tm_llm/action_contract.py`, `tm-ai-server/src/tm_llm/options.py` и focused tests.
- Не меняются model routing, prompts, budgets, retries, gameplay policy и выбор действий.
- Credentials и provider calls не требуются; commit, push, pin update и платный resume остаются отдельными gates.
