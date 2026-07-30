## Почему

Текущий tm-ai умеет показывать модели плоский список действий, но затем восстанавливает сложные `PlayerInputModel` через скрытые значения по умолчанию. Из-за этого корректная модель может получить недоступную карту либо выбрать составное действие, которое mapper превращает в заведомо невалидный ответ, как `Stormcraft` с оплатой `0/0`.

Проект небольшой и активный, поэтому полный rewrite создаст больше новых ошибок, чем устранит. Корневой `LICENSE.txt` и GitHub metadata указывают Apache-2.0, но README одновременно называет `tm-ai-server` GPLv3. Это расхождение нужно явно сохранить в документации fork, но оно не требует разрешения мейнтейнера на fork, локальные изменения или публикацию исходников: до уточнения изменения `tm-ai-server` публикуются консервативно как GPLv3 source с сохранением корневого Apache-2.0 text, README notice, истории и attribution без заявления о перелицензировании.

## Что изменится

- Создать после exact baseline approval публичный fork `rusliksu/tm-ai`, сохранить `mfgpcm/tm-ai` как `upstream`, root Apache-2.0 text и README GPLv3 notice; приватный `tm-advisor` продолжит хранить arena/config/evidence и использовать exact commit pin.
- Зафиксировать upstream license ambiguity и применить консервативную publication policy: сохранить оба исходных notices и attribution, считать изменения `tm-ai-server` GPLv3 source до уточнения и не выдавать GitHub SPDX detection за окончательную лицензионную интерпретацию. Ответ мейнтейнера может уточнить документацию позднее, но не блокирует source-only PR.
- Ввести чистый `ActionCandidate`/`ActionPlan` contract: модель выбирает candidate и явно задаёт все значения, которые требуют решения игрока; mapper больше не заполняет такие поля скрытым `min` или первым элементом.
- Добавить рекурсивный compiler и structural validator для `or`, `and`, `amount`, `card`, `projectCard`, `space`, `player`, `colony`, `delegate`, `party` и `resource`, включая bounds, disabled values и исходные индексы.
- Сохранить внешний TM-server `InputResponse` wire format; преобразование structured model output в wire response выполняется локально и fail-closed до `player.process()`.
- Ввести версионированный внутренний action contract и compatibility path для простых существующих решений; legacy `_default_response` остаётся только для узлов без пользовательского выбора и удаляется из decision-bearing paths после parity gate.
- Добавить публичный conformance corpus из canonical server-shaped fixtures без рук, game ids, credentials и model content; первая обязательная regression fixture — связанная оплата Stormcraft.
- Не менять в первом slice prompts стратегии, memory, model routing, pricing, retries, arena seats, SmartBot или server gameplay. Real provider inference и новая партия остаются отдельными платными gates.

## Возможности

### Новые возможности

- `structured-action-contract`: компиляция, model-facing schema, structural validation и построение полного `InputResponse` без скрытых gameplay defaults.
- `tm-ai-fork-governance`: публичный fork, upstream sync, credential/artifact boundary, exact-pin consumption и offline-only publication gates.

### Изменённые возможности

- Нет: в репозитории пока отсутствуют canonical OpenSpec capabilities; внешний `/move` response format сохраняется.

## Влияние

- Основной код: `tm-ai-server/src/tm_llm/options.py`, новый чистый action-contract модуль, `engine.py`, `prompts.py` и соответствующие tests/specs.
- Потребитель: private `rusliksu/tm-advisor` обновит только exact `tmAiCommit` pin после отдельного conformance gate.
- GitHub: публичный `rusliksu/tm-ai`; изменённый source разрешено публиковать только после pre-push audit с сохранением обоих upstream notices, истории и attribution.
- Безопасность: `.env`, secret-store, raw game logs, player hands, run artifacts и provider responses не публикуются; CI и offline gates не используют credentials.
- Совместимость: TM-server wire format не меняется; внутренний model-response contract версионируется и сначала включается через явный compatibility gate.
