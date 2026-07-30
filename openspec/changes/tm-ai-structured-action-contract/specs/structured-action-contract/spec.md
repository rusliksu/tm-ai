## ADDED Requirements

### Requirement: Компиляция дерева в типизированные действия
Система SHALL рекурсивно компилировать `PlayerInputModel` в `ActionCandidate` и обязательные typed value slots без скрытого выбора decision-bearing значений.

#### Scenario: Простое фиксированное действие
- **WHEN** leaf node имеет единственный фиксированный `InputResponse` и не требует значения игрока
- **THEN** compiler SHALL создать candidate без value slots и с полным статическим response skeleton

#### Scenario: Составное AndOptions
- **WHEN** `and` содержит два или больше дочерних узла, требующих значения игрока
- **THEN** compiler MUST сохранить каждый обязательный slot, исходный порядок и tree path без подстановки `min`, первого элемента или Cartesian enumeration

#### Scenario: Альтернативы OrOptions
- **WHEN** `or` содержит несколько branches
- **THEN** compiler SHALL создать различимые candidates с устойчивыми path-based ids и MUST сохранить исходный branch index для wire response

### Requirement: Явный план модели
Для `action-contract/v2` модель SHALL выбирать candidate и передавать все обязательные values через ограниченный JSON `ActionPlan`; произвольный wire response MUST NOT приниматься напрямую.

#### Scenario: Полный корректный план
- **WHEN** model response содержит известный candidate id и ровно все требуемые slots допустимых типов
- **THEN** parser SHALL передать plan structural validator без применения legacy defaults

#### Scenario: Неполный составной план
- **WHEN** отсутствует хотя бы один decision-bearing slot
- **THEN** engine MUST классифицировать ответ как локальный `invalid_response`, MAY выполнить только существующий bounded validation retry и MUST NOT отправлять частичный ход серверу

#### Scenario: Лишнее или неизвестное поле
- **WHEN** `ActionPlan` содержит неизвестный candidate, slot или произвольный nested wire payload
- **THEN** parser MUST отклонить ответ fail-closed и MUST NOT интерпретировать неизвестные данные как gameplay input

### Requirement: Структурная валидация до игрового сервера
Validator SHALL проверять типы, bounds, cardinality, allowlisted values, disabled markers и связь с исходным tree до построения `InputResponse`.

#### Scenario: Значение amount в границах
- **WHEN** amount slot получает целое значение между исходными `min` и `max`
- **THEN** validator SHALL принять значение и builder SHALL сохранить его без изменения

#### Scenario: Значение вне границ
- **WHEN** amount, index, cardinality либо allowlisted value нарушает исходный node contract
- **THEN** validator MUST завершиться локальным `invalid_response` до HTTP player input

#### Scenario: Недоступная карта
- **WHEN** card имеет точный `isDisabled=true`
- **THEN** compiler MUST не включать её в selectable values, а validator MUST отклонить попытку сослаться на неё

### Requirement: Полный wire response строится локально
Builder SHALL детерминированно преобразовывать validated `ActionPlan` в существующий TM `InputResponse` с исходными branch indices и порядком composite responses.

#### Scenario: Построение nested ответа
- **WHEN** accepted plan выбирает nested `or` внутри composite tree
- **THEN** builder MUST создать полный вложенный `InputResponse`, соответствующий исходным path и типам

#### Scenario: Отсутствие gameplay default
- **WHEN** composite child требует решения игрока
- **THEN** builder MUST использовать явное validated value и MUST NOT вызывать `_default_response` для этого child

### Requirement: Stormcraft получает два явных значения
V2 contract SHALL представлять canonical Stormcraft spend-heat как один candidate с отдельными slots обычного heat и Stormcraft floaters.

#### Scenario: Валидная оплата Stormcraft
- **WHEN** модель возвращает amounts, которые canonical server-equivalent oracle принимает для заданного target и bounds
- **THEN** builder SHALL сохранить оба значения и создать `and.responses=[amount, amount]` без подстановки `0/0`

#### Scenario: Структурно корректная, но семантически отклонённая пара
- **WHEN** values находятся в индивидуальных bounds, но server callback отклоняет их общей gameplay-проверкой
- **THEN** server SHALL остаться semantic authority, rejection MAY попасть в существующий sanitized bounded retry и система MUST NOT придумывать другой платёж локально

### Requirement: Версионирование и совместимость
Система SHALL публиковать action contract version и SHALL включать v2 поэтапно без неявного изменения простых legacy actions.

#### Scenario: Режим сравнения
- **WHEN** v2 compiler работает в compare mode на legacy-compatible fixture
- **THEN** система MUST доказать эквивалентный wire response без provider call и без изменения выбранного действия

#### Scenario: Несовместимая версия потребителя
- **WHEN** external consumer требует неизвестную action contract version
- **THEN** запуск MUST завершиться fail-closed до provider inference

#### Scenario: Legacy composite default запрещён
- **WHEN** v2 включён для composite node с decision-bearing slots
- **THEN** отсутствие полного `ActionPlan` MUST завершиться validation failure, а `CHOICE` MUST NOT активировать min/first fallback

### Requirement: Безопасный conformance corpus
Репозиторий SHALL содержать воспроизводимые server-shaped fixtures и независимые assertions без credentials, private hands, game ids, raw model content или live artifacts.

#### Scenario: Офлайн-набор тестов
- **WHEN** запускаются action-contract tests и полный pytest
- **THEN** все provider invocations MUST равняться `0`, а результаты MUST быть воспроизводимы без secret environment

#### Scenario: Добавление новой regression fixture
- **WHEN** live arena выявляет новый input mapping defect
- **THEN** public corpus MAY получить только минимальную sanitized canonical fixture и MUST отделять доказанные поля от исторических предположений
