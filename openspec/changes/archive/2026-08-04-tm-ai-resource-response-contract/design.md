## Контекст

TM server сериализует ресурсный input с allowlist в `include` и проверяет ответ точным shape `{type, resource}`. tm-ai уже ограничивает значение известными серверу ресурсами, но записывает его под несовместимым ключом `resourceType`. В legacy code часть путей также читает только старое поле `resources`.

## Цели / Вне целей

**Цели:**

- выдавать canonical `{type: "resource", resource: value}` во всех resource paths;
- выбирать значение только из `include` либо совместимого `resources`;
- доказать поведение на чистых fixtures и canonical server validator;
- сохранить fail-closed поведение для отсутствующего или недопустимого ресурса.

**Вне целей:**

- изменение prompts, выбора модели, reasoning, retry или timeout;
- изменение игровой стратегии или серверных правил;
- provider inference, новая партия, deploy, commit, push или pin update.

## Решения

### 1. Один canonical wire shape

Action-contract builder и legacy mapper возвращают только поле `resource`. Поле `resourceType` не сохраняется как fallback: точный серверный validator отвергает лишние и неверно названные поля, поэтому двойной payload не обеспечит совместимость.

### 2. Сервер задаёт список допустимых значений

Источник допустимых значений остаётся в полученном `PlayerInputModel`. Сначала используется canonical `include`; `resources` допускается только как совместимый вход старого fixture/runtime. tm-ai не добавляет ресурс вне списка и не угадывает новый default при пустом allowlist.

### 3. Независимые офлайн-oracles

Focused Python tests проверяют exact response для корневого и nested action-contract/v2, а также legacy enumerate/index/default paths. Отдельный canonical server-shaped oracle принимает новый ответ и отклоняет старый `resourceType`. Полный pytest и OpenSpec gates выполняются без secret environment и provider calls.

## Риски / Компромиссы

- [Скрытый legacy input использует `resources`] → сохранить input fallback, но всегда нормализовать output в `resource`.
- [Исправлен только один mapper path] → тестировать action-contract builder и все legacy response builders отдельно.
- [Тест повторяет ошибочную реализацию] → exact expected object и canonical server validator служат независимыми oracle.

## План отката

До публикации откат состоит в удалении task-owned diff. После отдельного commit/pin gate private arena сможет вернуться на предыдущий exact tm-ai SHA; сервер и сохранённая партия не изменяются этой offline-дельтой.
