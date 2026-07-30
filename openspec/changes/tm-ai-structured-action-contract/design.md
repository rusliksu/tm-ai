## Контекст

Upstream `mfgpcm/tm-ai` — публичный проект примерно на `3.5K` строк Python с одним основным contributor, `FastAPI`, prompt/memory/provider слоями и чистым модулем `options.py`. GitHub metadata и корневой `LICENSE.txt` указывают Apache-2.0, но README говорит, что `tm-ai-server` имеет ту же GPLv3-лицензию, что и TM server. Это противоречие требует сохранения обоих notices и осторожной маркировки derivative source, но не разрешения мейнтейнера на fork, локальные изменения или публикацию исходников.

Текущий action mapper сводит `PlayerInputModel` к плоскому списку с `path`, а при восстановлении составного ответа использует `_default_response`: первый элемент, `min` amount или другое скрытое значение. Это уже породило два разных server-side `input_rejected`: disabled `projectCard` и связанный `Stormcraft AndOptions`.

TM server wire contract остаётся авторитетным: только `player.process()` знает полную gameplay-семантику callback, а сериализованный `PlayerInputModel` обычно содержит типы, bounds и titles, но не произвольные межполевые ограничения. Поэтому клиент может полностью гарантировать структурную корректность, но не должен выдумывать семантический solver для всех карт.

Публичный fork GitHub публичного репозитория тоже будет публичным и не может отдельно сменить visibility. Приватный вариант потребовал бы отдельного несвязанного mirror repository. Код action contract не содержит конкурентных или персональных данных; private arena, credentials и live artifacts уже принадлежат `rusliksu/tm-advisor`.

## Цели / Вне целей

**Цели:**

- Сделать `rusliksu/tm-ai` поддерживаемым публичным fork с явным `upstream`, сохранёнными исходными license notices и чистой границей публикации.
- Отделить чистую компиляцию действий от prompt rendering, LLM provider и orchestration.
- Требовать от модели все decision-bearing значения и локально проверять структуру до TM server.
- Сохранить существующий `InputResponse` wire format и exact-pin потребление из private arena.
- Ввести staged migration с parity evidence, regression corpus и быстрым откатом.

**Вне целей:**

- Полный rewrite tm-ai, собственная игровая модель или обучение нейросети.
- Изменение gameplay rules либо универсальный вывод скрытых callback constraints из текста.
- Перенос private arena/config/logs в публичный fork.
- Одновременный redesign prompts стратегии, memory, provider routing, retries или pricing.
- Real provider inference, новая paid партия, staging/prod deploy либо live restart.

## Решения

### 1. Публичный fork вместо private mirror или rewrite

После явного одобрения baseline создаётся публичный `rusliksu/tm-ai`. В локальном clone `origin` указывает на fork, `upstream` — на `mfgpcm/tm-ai`; `main` отслеживает fork main, а upstream updates принимаются отдельными reviewable sync commits либо fast-forward, когда история совместима.

Сохраняются `LICENSE.txt`, README license statement и upstream history. До возможного уточнения upstream изменения внутри `tm-ai-server` публикуются по консервативному GPLv3 предположению: полный source доступен в публичном fork, оба исходных notices и attribution остаются на месте, а fork не заявляет перелицензирование upstream. Письменный ответ мейнтейнера может уточнить будущую документацию, но не является permission gate. Codex не делает окончательный юридический вывод только из GitHub SPDX detection.

До первого push также выполняется credential/artifact audit: `.env`, secret-store, `logs/`, player state, game artifacts, model content и local tool state не попадают в Git. Private `tm-advisor` хранит operational integration и закрепляет только принятый commit SHA.

Private mirror отклонён как default: он скрывает несекретный reusable код, теряет fork-network workflow и усложняет upstream sync. Rewrite отклонён: полезные provider, prompt, board, persistence и engine части пришлось бы повторно реализовать без доказанной выгоды. Если divergence позднее станет системным, публичный fork можно мигрировать в самостоятельный derivative repo отдельным change, не переписывая action core.

### 2. Чистый typed action core

Новый pure module владеет тремя моделями:

- `ActionCandidate`: стабильный id из исходного tree path, понятный title, response skeleton и список обязательных value slots;
- `ActionValueSlot`: тип, bounds/cardinality и allowlisted values для `amount`, `card`, `space`, `player`, `colony`, `delegate`, `party` или `resource`;
- `ActionPlan`: выбранный candidate id и явные values модели.

Compiler рекурсивно проходит `PlayerInputModel`. `or` создаёт альтернативные candidates; `and` композиционно объединяет обязательные дочерние slots без Cartesian enumeration; leaf с фиксированным ответом не создаёт slot. `isDisabled=true` исключается до prompt, а исходные indices/path сохраняются.

Builder принимает только validated `ActionPlan` и создаёт полный существующий `InputResponse`. Неизвестный candidate, отсутствующий или лишний slot, неверный тип, выход за bounds, недопустимое значение либо disabled choice завершаются локальным `invalid_response`. `_default_response` остаётся только для действительно статических узлов без решения игрока.

### 3. Версионированный model-facing contract

Для `action-contract/v2` prompt выводит candidates и typed slots. Модель отвечает одной строкой `ACTION: {"candidate":"<id>","values":{...}}`. JSON ограничивается локальным schema parser; произвольный full `InputResponse` от модели не принимается.

Compatibility path сохраняет `CHOICE` только для candidates без slots и setup contract. Для decision-bearing composite node отсутствие `ACTION` не превращается в min/first: engine выполняет существующий bounded validation retry, затем fail-closed. Public debug/evidence сохраняет только allowlisted stage/reason/type tokens, а не model content или values.

Contract version публикуется в `/version` и public manifest. Private arena принимает новый exact pin только при ожидаемом version; mismatch закрывается до provider call.

### 4. Семантика остаётся у сервера

Structural validator гарантирует соответствие сериализованному tree, но не угадывает callback constraints, которых нет в wire model. Для Stormcraft v2 впервые передаёт модели оба amount slots, target и публичные titles, поэтому tm-ai больше не подставляет `0/0` самостоятельно. Server-side semantic rejection остаётся authoritative и возвращается как sanitized `last_error` в существующий bounded retry.

Conformance corpus различает два oracle:

- structural oracle проверяет candidate/slots/bounds/disabled/path и builder output;
- server-equivalent oracle для canonical fixtures подтверждает gameplay acceptance конкретного полного ответа, не выдавая его за общий solver.

Первый corpus включает простые leaf actions, nested `or`, `and` со статическими детьми, multi-select cards, disabled project cards и Stormcraft `and/amount/amount`. Никакие fixtures не содержат реальную руку, game id или raw model response.

### 5. Поэтапное включение

Первый implementation slice создаёт pure types/compiler/validator и compare-mode tests, не меняя production selection. Второй переводит только composite `and` на v2 под explicit config и закрывает legacy defaults для decision-bearing children. Третий расширяет типы по corpus и только затем делает v2 default в fork. Каждый slice имеет собственный exact pin и может быть откатан возвратом private arena на предыдущий SHA.

Provider abstraction и prompts стратегии остаются за пределами первого change. Их можно рефакторить позднее поверх стабильного action core без смешивания корректности действий с качеством LLM.

## Риски / Компромиссы

- [Публичный fork случайно включает operational данные] → pre-push tracked-file audit, denylist для `.env`/logs/artifacts/local state и отдельная проверка фактического Git diff.
- [V2 parser увеличивает число validation retries] → compare mode, zero-provider fixtures, bounded retry и fail-closed без legacy decision defaults.
- [Большой `and` создаёт combinatorial explosion] → slots компонуются без перечисления Cartesian product; лимиты применяются к candidates и serialized schema size.
- [Структурно допустимый ответ нарушает скрытый gameplay callback] → server остаётся semantic authority, exact rejection возвращается модели в sanitized form, а повтор bounded; известные случаи пополняют public conformance corpus.
- [Upstream меняет `PlayerInputModel`] → upstream remote, versioned contract, corpus parity и exact-pin upgrade вместо плавающей зависимости.
- [Fork слишком расходится с upstream] → action core добавляется отдельными модулями и узкими adapters; upstream sync проверяется до merge, а standalone migration остаётся отдельным будущим решением.
- [Противоречивые Apache-2.0/GPLv3 notices приводят к неверной маркировке] → fork сохраняет оба источника, историю и attribution; изменения `tm-ai-server` публикуются как доступный GPLv3 source без заявления о перелицензировании, а maintainer clarification остаётся необязательным уточнением документации.

## План применения

1. После explicit approval создать неизменённый публичный fork и настроить `origin/upstream`.
2. Зафиксировать license ambiguity и консервативную GPLv3 policy для изменений `tm-ai-server`; сохранить оба notices, историю и attribution, не блокируя source-only PR ожиданием ответа мейнтейнера.
3. Выполнить metadata/content hygiene audit без чтения credentials.
4. Зафиксировать current legacy behavior corpus и red-first Stormcraft test, доказывающий потерю двух decision-bearing amounts.
5. Добавить pure action models/compiler/validator/builder и compare mode; прогнать полный offline pytest без provider invocation.
6. Включить `action-contract/v2` только для composite `and`, проверить legacy parity для простых actions и server-shaped Stormcraft conformance.
7. После credential/artifact audit и independent diff/test review создать task-owned commit/PR в fork. Private arena pin не менять до отдельного integration slice.
8. В private arena обновить exact pin, выполнить offline bridge/arena gates и только после отдельного paid approval запускать новую fixed-seat партию.

Откат: вернуть private arena на предыдущий exact SHA; в fork выключить v2 compatibility flag либо revert узкого task commit. Server/database/provider state не мигрируется.

## Открытые вопросы

- Имя fork рекомендуется оставить `rusliksu/tm-ai`, чтобы сохранить очевидную связь с upstream. Другое имя нужно выбрать до GitHub creation gate.
- Ответ upstream maintainer на issue `#2` может позднее уточнить формулировки notices; до этого действует консервативная GPLv3 publication policy без блокировки source-only PR.
- После первого accepted v2 game отдельно решить, оставлять ли `CHOICE` compatibility постоянно или удалить её из action turns новым breaking change.
