## Общие ограничения

- Целевой repository после approval: публичный GitHub fork `rusliksu/tm-ai`; `origin=rusliksu/tm-ai`, `upstream=mfgpcm/tm-ai`, root Apache-2.0 text, README GPLv3 notice и upstream history сохраняются.
- Task-owned branch: `codex/tm-ai-structured-action-contract`; base содержит accepted disabled-card commit `aabdb031b3fc57e65ab394469ef76f34d8772c92`.
- Внешний TM `InputResponse` wire format не меняется; новый внутренний protocol называется `action-contract/v2` и использует `ACTION: {"candidate":"<id>","values":{...}}`.
- `_default_response` запрещён для любого child, который требует решения игрока; он допустим только для статического узла без value slots.
- Compiler не перечисляет Cartesian product составных amounts; `and` объединяет typed slots композиционно.
- Structural validation не подменяет server gameplay callback; semantic rejection остаётся authoritative и допускает только существующий bounded sanitized retry.
- Первый scope не меняет strategy prompts, memory, provider routing, pricing, retries, SmartBot, TM server gameplay или private arena behavior.
- `.env`, credentials, secret-store, raw logs, player state, private hands, game ids, model content и run artifacts запрещены в публичном Git history и fixtures.
- Все implementation и publication checks выполняются без provider credentials и real inference. Paid smoke/game, deploy и restart требуют отдельных явных разрешений.

## 1. Согласование ownership и публикации

- [x] 1.1 Получить явное одобрение exact baseline: публичный fork `rusliksu/tm-ai`, staged `action-contract/v2`, без rewrite и без paid inference.
  - Проверка: Руслан подтверждает формулировку `одобряю public fork baseline`; до этого GitHub repository, remotes и source code не меняются.
  - Approval evidence 2026-07-30: Руслан дословно подтвердил `одобряю public fork baseline`; approval включает public fork и offline implementation, но не разрешает paid inference, merge либо deploy.
- [x] 1.2 После одобрения создать неизменённый public fork и настроить локальные remotes без изменения protected/base branch или push task commits.
  - Интерфейсы: `mfgpcm/tm-ai` public history → неизменённый public `rusliksu/tm-ai`; local `origin/upstream`.
  - Проверка: `gh repo view rusliksu/tm-ai` подтверждает `isFork=true`, `isPrivate=false`, parent `mfgpcm/tm-ai`; `git remote -v` показывает exact URLs; fork head совпадает с upstream head.
  - Evidence 2026-07-30: `https://github.com/rusliksu/tm-ai` создан как public fork parent `mfgpcm/tm-ai`; `origin=https://github.com/rusliksu/tm-ai.git`, `upstream=https://github.com/mfgpcm/tm-ai.git`, а `origin/main` и `upstream/main` совпадают на `bcbdfaf3d69159b0b8a033f2d57ad1436c4fec8d`. Task branch не push.
- [x] 1.3 Зафиксировать upstream license ambiguity, принять консервативную publication policy и выполнить credential/artifact-safe tracked inventory.
  - Интерфейсы: root `LICENSE.txt` Apache-2.0 + README GPLv3 statement → сохранение обоих notices, истории и attribution; изменения `tm-ai-server` считаются GPLv3 source до возможного уточнения.
  - Проверка: evidence явно цитирует оба upstream источника и принятое решение; staged/tracked scan не содержит запрещённых operational файлов или secret patterns. Ответ мейнтейнера не является permission gate для fork или source-only PR.
  - Evidence 2026-07-30: root/GitHub показывают Apache-2.0, README lines `202–203` говорит GPLv3 для `tm-ai-server`; Apache Foundation документирует одностороннюю совместимость Apache-2.0 с GPLv3, а GNU GPL FAQ разрешает private modifications и публикацию modified source при соблюдении GPL. Поэтому fork сохраняет оба notices и полный source, не заявляет перелицензирование и консервативно применяет GPLv3 к изменениям `tm-ai-server`. Issue `https://github.com/mfgpcm/tm-ai/issues/2` остаётся необязательным запросом на документальное уточнение. Локальный tracked/untracked audit показывает forbidden paths=`0`, secret-pattern files=`0` и absolute user path files=`0`.

## 2. Чистое ядро action contract

- [x] 2.1 Добавить red-first public conformance corpus для simple leaf, nested `or`, composite `and`, multi-select card, disabled `projectCard` и Stormcraft `and/amount/amount`.
  - Интерфейсы: canonical public `PlayerInputModel` shapes → sanitized fixtures и независимые structural/server-equivalent assertions.
  - Проверка: focused tests сначала доказывают legacy loss decision-bearing values и прежний hidden `0/0`, при этом fixtures не содержат private/live данных.
  - Evidence 2026-07-30: первый focused run завершился collection error `ModuleNotFoundError: tm_llm.action_contract`, а независимый Stormcraft oracle отдельно доказал, что legacy response равен `0/0` и не проходит canonical достаточность/переплату. Corpus содержит только вручную заданные public server-shaped dictionaries и literal expected wire responses.
- [x] 2.2 Добавить pure `ActionCandidate`, `ActionValueSlot`, `ActionPlan` и recursive compiler без изменения engine behavior.
  - Интерфейсы: `dict PlayerInputModel` → candidates с path-based id, response skeleton и typed slots.
  - Проверка: focused compiler tests покрывают все scope node types, сохраняют original paths/indices, исключают exact boolean disabled values и не создают Cartesian combinations.
  - Evidence 2026-07-30: `action_contract.py` компилирует path-based candidates, explicit amount/card/projectCard/space/player/colony/delegate/party/resource/globalEvent slots и conditional nested `or` branches; composite `and` объединяет slots без Cartesian enumeration. Post-review correction использует canonical TM `resource.include` с legacy-compatible fallback `resources`; server-shaped tests проверяют оба direct и nested `AndOptions` пути.
- [x] 2.3 Добавить structural validator и deterministic builder полного существующего `InputResponse`.
  - Интерфейсы: `PlayerInputModel + ActionCandidate + ActionPlan` → validated existing wire response либо allowlisted local `invalid_response`.
  - Проверка: tests отклоняют unknown/missing/extra slots, неверные types/bounds/cardinality и disabled values; accepted nested responses совпадают с independent expected wire objects; decision-bearing paths не вызывают `_default_response`.
  - Evidence 2026-07-30: validator/builder возвращает существующий wire format и allowlisted local reasons. Узкая mutation `amount := node.min` воспроизвела исторический `4/2 → 0/0`, и exact Stormcraft regression ожидаемо упал; после возврата explicit plan values suite снова зелёный.

## 3. Версионированный model-facing protocol

- [x] 3.1 Добавить ограниченный parser `ACTION` и prompt rendering typed candidates, сохранив legacy `CHOICE` только для setup и candidates без slots.
  - Интерфейсы: candidates → prompt fragment; model text → schema-limited `ActionPlan`.
  - Проверка: parser tests принимают единственный полный JSON plan, отклоняют arbitrary wire payload, malformed JSON и incomplete composite values; raw model content не входит в public diagnostics.
  - Evidence 2026-07-30: parser принимает plain/markdown `ACTION` line только с exact top-level keys `candidate/values`; retry передаёт модели только allowlisted reason и не echo предыдущий response. `ACTION` удаляется из tactical/strategy memory.
- [x] 3.2 Подключить v2 к `engine.py` под explicit compatibility config и сохранить bounded validation retry/fail-closed behavior.
  - Интерфейсы: parsed `ActionPlan` → validator/builder → существующий `/move` response; validation failure → sanitized retry stage/reason.
  - Проверка: fake-player/provider tests доказывают zero server call для structural invalid response, отсутствие min/first fallback и неизменный legacy path для simple actions.
  - Evidence 2026-07-30: `TM_AI_ACTION_CONTRACT=legacy|compare|v2`; v2 активен только для root composite `and`. Valid plan строит explicit response; missing slot после bounded retries даёт sanitized HTTP `422 action_contract` вместо hidden default; provider exception не маскируется как input failure и не вызывает fallback.
- [x] 3.3 Публиковать `action-contract/v2` в `/version`/manifest и добавить compare mode без provider invocation.
  - Интерфейсы: runtime config + compiler version → public version capability и parity report.
  - Проверка: legacy-compatible corpus даёт эквивалентные wire responses; composite fixtures показывают явные slots вместо defaults. External-consumer version mismatch завершается до provider call в отдельном private integration task `4.3`.
  - Evidence 2026-07-30: fork-side `/version` публикует exact version/mode, compare mode доказывает `equivalent` для zero-slot legacy response и `requires_values` для Stormcraft без изменения submitted move. Endpoint, compare summary и отсутствие provider invocation покрыты offline tests; consumer mismatch gate не смешан с public fork scope.

## 4. Offline acceptance и доставка fork

- [x] 4.1 Выполнить focused/full pytest, OpenSpec language/strict gates, diff check и credential/artifact audit.
  - Проверка: `uv run pytest tests/ -v`, focused action-contract suite, `git diff --check`, `Test-OpenSpecRussian.ps1` и `openspec validate tm-ai-structured-action-contract --strict --no-interactive` завершаются с exit code `0`; provider invocations равны `0`.
  - Evidence 2026-07-30: test environment принудительно очищал `OPENROUTER_API_KEY` и `DEEPSEEK_API_KEY`; focused suite `33 passed`, full suite `95 passed`, включая canonical `resource.include` regression. Pre-publication audit: forbidden paths=`0`, secret-pattern files=`0`, absolute user path files=`0`; real provider invocations=`0`.
- [ ] 4.2 Провести независимый review фактического diff и verification commands, затем создать task-owned commit и PR в `rusliksu/tm-ai`.
  - Интерфейсы: accepted offline diff → public fork PR с source/specs/tests и без operational artifacts.
  - Проверка: reviewer verdict `accepted`; PR head SHA совпадает с local task head, `gh pr diff` содержит только approved scope, CI/mergeability не имеют blockers. Merge остаётся отдельным gate текущего PR lifecycle.
  - Review evidence 2026-07-30: отдельный structural/test-quality pass сначала отклонил неверное поле `resource.resources`, сверил canonical `SelectResourceModel.include`, потребовал минимальную correction и повторил focused/full tests. После correction verdict=`accepted`: новых source-файлов больше `1000` строк нет, generated/private artifacts отсутствуют, typed core изолирует domain mapping, failure/retry/compare paths покрыты observable assertions. Commit/PR evidence пока ожидается.
- [ ] 4.3 После принятого fork commit сформировать exact-pin handoff для отдельного private `tm-advisor` integration change.
  - Интерфейсы: accepted fork SHA + `action-contract/v2` → proposed `tmAiCommit` и offline bridge/arena verification packet.
  - Проверка: handoff не меняет private pin автоматически и явно фиксирует previous SHA для rollback.
- [ ] 4.4 После private offline acceptance отдельно согласовать любой real smoke или fixed-seat game.
  - Проверка: разрешение обязано назвать exact models/seats, request caps, deadlines и общий cost envelope; без него provider calls равны `0`.
