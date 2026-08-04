# Итоги staged-обучения

Дата серии: 2026-08-04  
Каноническая ветка результатов: `experiment/staged-training`
Supervisor: `scripts/staged_resume_experiments.py`  
Статус supervisor: `logs/staged_resume/2026-08-04_11-09-27/status.json`

## Протокол

- Isaac task: `Template-Cbriisaaclab-Direct-v0`.
- `num_envs=2048`, `rollouts=32`, `64000` environment timesteps на стадию.
- Максимум два обучения одновременно.
- Четыре curriculum-варианта, по три стадии.
- Три Stage 1 были распознаны как уже выполненные эквивалентные запуски и не повторялись.
- Один неполный Stage 1 (`easy-task-to-robust`) был продолжен с `agent_40000.pt` без reset optimizer/scheduler.
- Все девять фактически запущенных стадий завершились с `returncode=0`; ошибок и failed-вариантов нет.

Сами TensorBoard event-файлы и checkpoint-файлы остаются в локальных worktree и не коммитятся в Git: это примерно сотни мегабайт бинарных данных, а `.gitignore` намеренно исключает `logs/`. Единый индекс путей и команда TensorBoard находятся в [EXPERIMENT_LOG_INDEX.md](EXPERIMENT_LOG_INDEX.md).

## Финальные результаты

Значения — медиана последних 20 scalar-точек TensorBoard. `lifetime` — больше лучше; `termination`, `speed error` и `sit error` — меньше лучше. Высота стопы интерпретируется вместе с clearance и не должна максимизироваться без ограничений.

| Цепочка | Lifetime | Termination | Speed error | Foot height | Sit error | Решение |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `reward-curriculum` | 315 | 0.00321 | 0.465 | 0.0162 | 0.351 | скорость и sit лучше, survival потерян |
| `easy-to-robust` | 113 | 0.00943 | 0.503 | 0.0241 | **0.133** | хорошая sit/clearance, плохая ходьба |
| `easy-task-to-robust` | 244 | 0.00395 | **0.453** | 0.0162 | 0.237 | средний компромисс, но слабый survival |
| `staged-control` | **821** | **0.00065** | 0.537 | 0.0166 | 0.450 | лучший кандидат по устойчивости |
| `delta-task-repeat` reference | 691 | 0.00106 | 0.576 | 0.0175 | 0.457 | исходный полный Stage 1 для control |

### По стадиям

- `staged-control`: Stage 1 был переиспользован из полного `delta-task-repeat`; Stage 2 поднял lifetime примерно до `747`, Stage 3 — до `821`. Этот вариант сохранял `task_balanced` на всех стадиях и не показал деградации при переходе между ними.
- `reward-curriculum`: переход с Stage 2 (`lifetime≈633`) на финальный `baseline` Stage 3 (`≈315`) примерно вдвое снизил lifetime; termination вырос с `≈0.00126` до `≈0.00321`.
- `easy-to-robust`: переход с Stage 2 (`lifetime≈576`) к `baseline` Stage 3 (`≈113`) оказался самым плохим; termination вырос примерно в 7.6 раза. Предобучение на easy-survival улучшило sit/clearance, но не перенесло устойчивую ходьбу.
- `easy-task-to-robust`: финальная стадия улучшила speed и sit относительно промежуточной, но lifetime снизился примерно с `289` до `244`.

## Что считать рабочим

1. Последовательное обучение через checkpoints работает: policy/value и normalization переносятся, а переходы между стадиями запускаются supervisor автоматически.
2. Deduplication работает: эквивалентный `staged-control/task-1` был заменён полным `delta-task-repeat/agent_60000.pt`, а новые дубликаты не запускались.
3. Наиболее рабочая curriculum-гипотеза сейчас — сохранять `task_balanced` на всех стадиях. `staged-control` дал примерно `+19%` lifetime и `−38%` termination относительно своего полного `delta-task-repeat` reference.
4. Физические метрики полезнее reward: некоторые easy/baseline-переходы улучшали sit или speed error, одновременно разрушая survival.

## Что не сработало

- Резкая замена `task_balanced` на `baseline` на последней стадии. Она приводила к забыванию устойчивой ходьбы.
- Easy-pretraining как самостоятельная стратегия для финальной policy: `easy-to-robust` дал очень хорошую sit error, но худший lifetime.
- По одной 64k-серии нельзя считать результат статистически подтверждённым: seed у большинства вариантов один.

## Следующие эксперименты

### 1. Длинный baseline

Запустить текущий baseline без staged-переходов на `800000` environment timesteps (`max_iterations=25000`, seed 42). Это проверит, сохраняется ли результат `long-baseline` после длинного обучения и отделит эффект времени обучения от эффекта curriculum.

### 2. Длинный task-balanced control

Отдельно продлить наиболее устойчивый `staged-control`/`task_balanced` кандидат до длинного протокола. После этого подтвердить результат минимум двумя дополнительными seed.

### 3. Сначала ходьба, потом регуляризация действия

Идея пользователя выглядит перспективной, но штраф нужно усиливать постепенно и не заменять им весь locomotion reward:

1. **Locomotion stage:** `delta` + `task_balanced`, текущий слабый штраф величины действия, пока policy учится ходить и переходить sit/stand.
2. **Regularized stage:** загрузить checkpoint предыдущей стадии, оставить survival/speed/sit/clearance reward и увеличить штраф `-lambda_action * ||action||²`; отдельно логировать `mean_abs`, `mean_abs_rate`, saturation и `target/mean_abs_step`.
3. **Transition stage:** проверить плавный ramp `lambda_action` за первые `16k–32k` шагов, чтобы не вызвать catastrophic forgetting.

В текущем коде штраф величины действия уже есть, но он фиксирован как `-0.00001`; в `task_balanced` дополнительно действует штраф изменения действия с коэффициентом `0.0015`. Поэтому следующий эксперимент должен параметризовать эти коэффициенты в `EnvCfg`, а не добавлять ещё один неуправляемый hardcode.

Минимальный screening: `lambda_action ∈ {5e-5, 1e-4}` при одинаковых seed и checkpoint. Успешным считать вариант, который уменьшает action magnitude/rate минимум на 20%, не снижая lifetime более чем на 10% и не повышая speed error более чем на 10%.

Отдельно нужно сравнить deterministic replay: если replay уже гладкий, проблема в основном в stochastic exploration и следует проверять entropy/log-std; если replay остаётся дёрганым, action regularization и target-rate penalty действительно адресуют физическую проблему.

Эти эксперименты продолжают обучение PPO с checkpoint и не используют imitation learning или обучающие датасеты.
