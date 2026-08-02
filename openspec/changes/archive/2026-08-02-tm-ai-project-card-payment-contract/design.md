## Контекст

`flatten_options()` уже разворачивает nested `projectCard` в exact paths и исключает cards с `isDisabled=true`. Однако `parse_action_response()` после выбора path повторно ищет названия всех cards в model prose и может заменить выбранную card. `correct_payment()` и `check_payment_valid()` определяют steel/titanium преимущественно по локальному `CARD_DB`; для неизвестной card они permissive и не используют `standardProjectCanPayWith` либо parent `paymentOptions`.

Сохранённое synthetic state даёт точный regression case: B имеет `13 MC` и `7 steel`; `Excavate:SP` содержит `standardProjectCanPayWith.steel=true`, а `Power Plant:SP` — пустой объект. Canonical server принимает `Excavate:SP + steel=4`, но отклоняет `Power Plant:SP + steel=6`; текущий tm-ai validator принимает оба.

## Цели / Вне целей

**Цели:**

- Сохранить exact path как единственный источник card name.
- Ограничить payment server-owned правилами до проверки покрытия cost.
- Fail-closed остановить не покрываемую legal resources оплату до server POST.
- Закрепить поведение focused tests без provider inference.

**Вне целей:**

- Изменение prompt, модели, reasoning/output budget, retry/deadline либо provider route.
- Повторный provider-call после server rejection.
- Изменение canonical TM server rules или wire format.
- Paid rerun, push, merge и deploy в рамках offline implementation.

## Решения

1. Для nested `or → projectCard` card name берётся из `index_to_response()` и не пересматривается по model prose. PAYMENT parsing влияет только на payment.
2. `payment.py` получает выбранный card model из parent `projectCard.cards`. Если присутствует `standardProjectCanPayWith`, это authoritative standard-project policy: steel/titanium/seeds/kuiper разрешаются только явным `true`; heat и Luna Federation titanium дополнительно ограничиваются parent `paymentOptions`; aurorai/spire доступны только при положительном server counter.
3. Для обычной project card steel/titanium определяются по известным tags; неизвестная card больше не получает permissive steel/titanium. Heat/plants и специальные ресурсы ограничиваются parent `paymentOptions`, tags и parent counters.
4. Нормализация сначала обнуляет запрещённые поля, затем clamp-ит каждое разрешённое поле фактической доступностью и только затем дополняет недостающее `megacredits`. Проверка покрытия использует ту же payment policy, чтобы parser и validator не расходились.
5. Focused tests проверяют exact choice binding, standard-project steel boundary и локальное отклонение underfunded legal payment. Полный test suite выполняется до публикации commit.

## Риски / Компромиссы

- [У неизвестной обычной card нет tags в `CARD_DB`] → steel/titanium fail-closed обнуляются вместо permissive расширения; MC остаётся безопасным fallback.
- [Server добавит новый payment resource] → неизвестное поле не становится платёжным автоматически; потребуется явное расширение allowlist.
- [Локальная стоимость ресурса отличается от canonical player value] → текущие значения payment сохраняются вне этой delta; focused server replay остаётся обязательным acceptance oracle.
- [Parser ранее угадывал card по prose] → exact path уже содержит выбранную card, поэтому удаление эвристики снижает неоднозначность без потери контракта.

## План применения

1. Добавить red-first focused tests на текущее неверное поведение.
2. Реализовать exact-choice binding и общую server-authoritative payment policy.
3. Выполнить focused/full offline tests, языковую проверку и strict OpenSpec validation.
4. Создать task-owned tm-ai commit; отдельным шагом обновить exact pin арены и повторить её offline gates.
5. При откате вернуть предыдущий tm-ai pin `369bf4e3b868503fc0d8dbdff2d9f2094b5c3ff3`; смешанная версия pin и tests запрещена.

## Открытые вопросы

Нет.
