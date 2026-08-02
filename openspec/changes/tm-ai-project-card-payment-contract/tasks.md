## Общие ограничения

- Exact card path остаётся единственным источником `projectCard.card`; model prose не меняет выбор.
- Payment policy использует `paymentOptions`, `standardProjectCanPayWith`, известные tags и resource counters; неизвестная card не получает permissive steel/titanium.
- Provider/model routes, prompts, retry/deadline, reasoning/output budgets и внешний TM wire format не меняются.
- Все проверки выполняются offline без credentials и provider inference.

## 1. Parser и payment policy

- [x] 1.1 Добавить red-first focused tests для exact-choice binding и server-authoritative standard-project payment.
  - Интерфейсы: server-shaped `projectCard` fixture с `Power Plant:SP`, `Excavate:SP`, disabled cards и player resources → assertions parser/payment validator.
  - Проверка: `uv run --project tm-ai-server pytest tm-ai-server/tests/test_prompts.py -q` воспроизвёл три ожидаемых failure: prose заменил `Power Plant:SP` на disabled `Asteroid:SP`, а steel остался допустимым для `Power Plant:SP` в normal и underfunded cases.
- [x] 1.2 Удалить переопределение card name из model prose и реализовать общую нормализацию payment по server-owned правилам.
  - Интерфейсы: `parse_action_response()` + selected parent/card model → `correct_payment()` и `check_payment_valid()` с одинаковой allowlist ресурсов.
  - Проверка: `uv run --project tm-ai-server pytest tm-ai-server/tests/test_prompts.py tm-ai-server/tests/test_options.py -q` — `41 passed`; отдельный regression сохраняет прежнюю оплату `SelectPayment` через heat.
  - Evidence 2026-08-02: parser больше не сканирует model prose для замены exact card; shared payment context использует `standardProjectCanPayWith`, `paymentOptions`, known tags, player values и server resource counters. `Power Plant:SP + STEEL=6` нормализуется в `MC=11, STEEL=0`, `Excavate:SP + STEEL=4` сохраняется.

## 2. Offline acceptance и интеграционный handoff

- [x] 2.1 Выполнить полный tm-ai offline suite и статические проверки.
  - Проверка: `uv run --project tm-ai-server pytest tm-ai-server/tests -q`, `uv run --project tm-ai-server python -m compileall -q tm-ai-server/src`, `git diff --check`, языковая проверка и `openspec validate tm-ai-project-card-payment-contract --strict --no-interactive` завершаются с exit code `0`; provider invocations=`0`.
  - Evidence 2026-08-02: полный suite завершился `99 passed`, compileall и `git diff --check` — exit code `0`. Read-only canonical replay сохранённого synthetic state принял оба преобразованных response: `Power Plant:SP` с `MC=11, STEEL=0` и `Excavate:SP` с `MC=0, STEEL=4`; credentials/provider transport не использовались.
- [ ] 2.2 Подготовить task-owned tm-ai commit и передать exact SHA для обновления arena pin; не выполнять paid rerun либо merge.
  - Интерфейсы: проверенный tm-ai tree → exact commit SHA для `config/tm-ai-openrouter-arena.json` и arena pin tests.
  - Проверка: `git status --short --branch`, `git show --stat --oneline HEAD` и clean diff scope после commit.
