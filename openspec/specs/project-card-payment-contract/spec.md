# project-card-payment-contract Specification

## Purpose
Зафиксировать точную привязку nested `projectCard` к выбранному path и server-authoritative правила нормализации оплаты до отправки ответа в Terraforming Mars server.
## Requirements
### Requirement: Exact path определяет выбранную карточку
tm-ai SHALL формировать nested `projectCard.card` только из exact path выбранного flattened option и MUST NOT заменять card name упоминанием в свободном model prose.

#### Scenario: Рассуждение упоминает другую карточку
- **WHEN** `CHOICE` указывает на enabled card, а model prose упоминает другую либо disabled card
- **THEN** response сохраняет card из exact path и не выбирает упомянутую card

### Requirement: Server-owned правила ограничивают оплату
tm-ai SHALL ограничивать payment полями `paymentOptions`, `standardProjectCanPayWith`, card tags и фактическими resource counters до расчёта покрытия cost. Неизвестная card MUST NOT автоматически разрешать steel либо titanium.

#### Scenario: Power Plant запрещает steel
- **WHEN** `Power Plant:SP` имеет пустой `standardProjectCanPayWith`, а model предлагает `STEEL=6`
- **THEN** tm-ai обнуляет steel и дополняет legal MC либо возвращает local validation error до server POST

#### Scenario: Excavate разрешает steel
- **WHEN** `Excavate:SP` содержит `standardProjectCanPayWith.steel=true` и player имеет достаточно steel
- **THEN** tm-ai сохраняет steel payment в пределах доступного количества

#### Scenario: Parent запрещает heat и plants
- **WHEN** `paymentOptions.heat=false` и `paymentOptions.plants=false`
- **THEN** tm-ai возвращает ноль для heat и plants независимо от model text

#### Scenario: Разрешённые ресурсы не покрывают cost
- **WHEN** после обнуления и clamp разрешённые ресурсы вместе с доступными MC не покрывают calculated cost
- **THEN** payment validator возвращает локальную ошибку и response MUST NOT считаться валидным для server POST

### Requirement: Offline acceptance предшествует интеграции
Изменение SHALL пройти focused tests и полный offline suite без credentials либо provider inference до публикации tm-ai commit и обновления arena pin.

#### Scenario: Offline gate завершён
- **WHEN** focused payment/parser tests, полный test suite, `git diff --check`, языковая проверка и strict OpenSpec validation успешны
- **THEN** task-owned tm-ai commit MAY быть подготовлен, но credential preflight, paid rerun, merge и deploy остаются запрещены без отдельных gates
