## ADDED Requirements

### Requirement: Публичный fork сохраняет происхождение
После explicit baseline approval repository SHALL быть создан как публичный GitHub fork `rusliksu/tm-ai` от `mfgpcm/tm-ai` с сохранёнными root Apache-2.0 text, README GPLv3 notice, историей и attribution.

#### Scenario: Создание fork
- **WHEN** Руслан одобряет exact public-fork baseline
- **THEN** GitHub fork MUST оставаться публичным, указывать upstream relation и MUST NOT маскироваться как независимый original work

#### Scenario: Локальные remotes
- **WHEN** task-owned clone переключается на fork ownership
- **THEN** `origin` SHALL указывать на `rusliksu/tm-ai`, `upstream` SHALL указывать на `mfgpcm/tm-ai`, а protected/base branch MUST NOT получать прямые task code commits

### Requirement: Противоречие лицензий обрабатывается консервативно
Система MUST зафиксировать расхождение между root/GitHub Apache-2.0 и README GPLv3 statement, сохранить оба исходных notices, историю и attribution и MUST считать изменения `tm-ai-server` GPLv3 source до возможного уточнения upstream. Разрешение мейнтейнера на fork, локальные изменения или source-only PR MUST NOT требоваться.

#### Scenario: Публичный fork
- **WHEN** exact baseline одобрен, но license ambiguity ещё не уточнена upstream
- **THEN** GitHub MAY создать публичный fork и публиковать reviewed source changes с сохранёнными notices, историей и attribution без заявления о перелицензировании

#### Scenario: Консервативная публикация изменений
- **WHEN** fork публикует изменения внутри `tm-ai-server` до ответа upstream maintainer
- **THEN** полный changed source MUST быть доступен публично, GPLv3 statement и root Apache-2.0 text MUST сохраняться, а fork MUST NOT заявлять новую исключительную лицензию на upstream code

#### Scenario: Позднее уточнение
- **WHEN** upstream maintainer уточняет применимую лицензию
- **THEN** fork MAY отдельным reviewed documentation change привести notices в соответствие с уточнением без переписывания уже опубликованной истории

### Requirement: Граница публичных и приватных данных
Публичный fork MUST содержать только reusable source, specs, tests и sanitized fixtures; operational secrets и private arena evidence MUST оставаться вне repository history.

#### Scenario: Проверка перед публикацией
- **WHEN** готовится первый либо последующий public push
- **THEN** tracked diff MUST быть проверен на `.env`, keys/tokens, secret-store data, raw logs, player state, private hands, game ids, model content и run artifacts

#### Scenario: Тесты без credentials
- **WHEN** CI или локальный publication gate запускает tests
- **THEN** test environment MUST не требовать реальные provider credentials и MUST не выполнять paid inference

### Requirement: Private consumer использует точный pin
Private `rusliksu/tm-advisor` SHALL подключать fork только по принятому commit SHA и ожидаемой action contract version.

#### Scenario: Принятый новый pin
- **WHEN** fork commit прошёл focused/full offline tests и independent diff review
- **THEN** private consumer MAY обновить exact pin отдельным integration slice и MUST повторить bridge/arena offline gates

#### Scenario: Непринятый или несовместимый pin
- **WHEN** SHA, version либо source audit не совпадает с approved evidence
- **THEN** consumer MUST завершиться fail-closed до credential preflight или provider call

### Requirement: Upstream sync остаётся reviewable
Fork SHALL сохранять отдельный `upstream` remote и MUST принимать upstream changes только через проверяемый sync path без перезаписи task history.

#### Scenario: Совместимый upstream update
- **WHEN** новый upstream commit не конфликтует с fork action contract
- **THEN** maintainer MAY применить fast-forward или узкий reviewed sync и MUST повторить full offline conformance gate перед новым pin

#### Scenario: Конфликт с action core
- **WHEN** upstream меняет `options.py`, engine contract либо wire-related schema несовместимо
- **THEN** sync MUST остановиться до отдельной delta, diff review и обновления conformance fixtures

### Requirement: Публикация и paid inference разделены
Создание fork и offline implementation SHALL не давать разрешение на real provider inference, paid game, deploy или live restart.

#### Scenario: Offline change принят
- **WHEN** public PR и offline tests accepted
- **THEN** любой smoke либо fixed-seat game MUST ждать отдельного разрешения с exact models, request caps и cost envelope
