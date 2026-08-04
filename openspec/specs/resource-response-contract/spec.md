# resource-response-contract Specification

## Purpose
Зафиксировать точный wire response выбора ресурса для Terraforming Mars server, server-owned список допустимых значений и воспроизводимую офлайн-проверку всех tm-ai mapper paths.
## Requirements
### Requirement: Канонический ответ выбора ресурса

tm-ai SHALL преобразовывать выбранный ресурс в точный TM wire response `{type: "resource", resource: <unit>}` и MUST NOT возвращать устаревшее поле `resourceType`.

#### Scenario: Корневой resource input

- **WHEN** сервер передаёт корневой resource input с непустым `include`
- **THEN** mapper SHALL вернуть выбранное allowlisted значение в поле `resource`

#### Scenario: Вложенный resource input

- **WHEN** resource node находится внутри составного action-contract/v2 candidate
- **THEN** builder SHALL сохранить canonical resource response во вложенной структуре без изменения path или соседних responses

#### Scenario: Старое поле отклоняется

- **WHEN** response содержит `resourceType` вместо `resource`
- **THEN** canonical server-shaped validator SHALL отклонить его, а regression test MUST отличать старый response от принятого нового

### Requirement: Допустимый ресурс задаётся сервером

tm-ai SHALL выбирать resource value только из canonical `include` либо совместимого legacy `resources` входного узла.

#### Scenario: Канонический список допустимых значений

- **WHEN** resource node содержит `include`
- **THEN** compiler, enumeration и response builder SHALL использовать только значения из `include`

#### Scenario: Совместимый список допустимых значений

- **WHEN** старый fixture содержит `resources` и не содержит `include`
- **THEN** legacy mapper MAY использовать `resources`, но output MUST оставаться canonical с полем `resource`

#### Scenario: Пустой allowlist

- **WHEN** resource node не содержит допустимых значений
- **THEN** tm-ai MUST завершить mapping fail-closed и MUST NOT придумывать произвольный ресурс

### Requirement: Офлайн-регрессия без provider inference

Ресурсный контракт SHALL проверяться воспроизводимо без credentials и real provider calls.

#### Scenario: Focused и полный наборы

- **WHEN** запускаются focused resource tests и полный pytest
- **THEN** проверки SHALL завершиться без provider invocation и подтвердить exact output для v2 и legacy paths
