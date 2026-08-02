## Почему

Live arena после устранения обрезки DeepSeek дошла до server-side HTTP `400` на `or → projectCard`. Offline replay доказал, что tm-ai принимает запрещённую оплату `Power Plant:SP` сталью, поскольку игнорирует server-owned payment rules и может заменить exact выбранную карточку упоминанием в свободном тексте модели.

## Что изменится

- Nested `projectCard` будет навсегда связан с exact path, выбранным через `CHOICE`; свободный текст больше не сможет заменить card name.
- Нормализация и проверка оплаты будут учитывать `paymentOptions`, `standardProjectCanPayWith`, card tags и фактически доступные ресурсы.
- Запрещённые ресурсы будут обнуляться до расчёта покрытия; если разрешённых ресурсов недостаточно, ответ завершится локальной validation error до server POST.
- Focused fixtures закрепят различие между `Excavate:SP`, где steel разрешён, и `Power Plant:SP`, где steel запрещён.

## Возможности

### Новые возможности

- `project-card-payment-contract`: точная привязка выбранной project card и server-authoritative нормализация её оплаты.

### Изменённые возможности

Нет.

## Влияние

Изменяются `tm-ai-server/src/tm_llm/prompts.py`, `tm-ai-server/src/tm_llm/payment.py`, `specs/TM-AI.md` и focused tests. Внешний TM `InputResponse`, prompts, provider/model routes, retries, reasoning/output budgets и server rules не меняются. После offline acceptance потребуется отдельное обновление exact tm-ai pin в arena; credential preflight и paid rerun остаются отдельными gates.
