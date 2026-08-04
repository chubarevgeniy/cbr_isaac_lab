# Контекст экспериментов — читать первым

Это каноническая точка входа для новых чатов и новых запусков:

```text
Ветка:   experiment/results-full-trajectory
Worktree: /home/evgenii/ws/isaac/cbr_i_results_full_trajectory
```

В этой ветке находятся одновременно:

- код среды и PPO;
- `scripts/staged_experiments.py` — supervisor новых staged-серий;
- `scripts/staged_resume_experiments.py` — resume прерванных стадий;
- все отчёты, план и правила проверки дублей;
- индекс локальных run-директорий, TensorBoard events и checkpoint-файлов.

## Текущее состояние на 2026-08-04

- Staged-когорта завершена, активных обучений нет.
- Лучший результат — `staged-control/task-3`: lifetime около `821`,
  termination около `0.00065` на последних точках 64k-прогона.
- Лучший checkpoint:

  ```text
  /home/evgenii/ws/isaac/cbr_i_staged_staged_control/logs/skrl/cbr_i_ppo/2026-08-04_15-30-43_experiment_staged-control_db7f54c_clean_ppo_torch_resume-staged-control-task-3/checkpoints/agent_60000.pt
  ```

- Неудачные направления: резкая замена `task_balanced` на `baseline` на
  последней стадии и easy-pretraining как самостоятельная curriculum-стратегия.
- Следующий приоритет: длинный baseline на `800000` environment timesteps,
  затем `task_balanced/staged-control` на том же протоколе и проверка curriculum
  штрафа за action magnitude/rate. Imitation learning и обучающие датасеты не
  используются.

## Документы

1. [TRAINING_PLAN.md](TRAINING_PLAN.md) — текущий план и критерии принятия решений.
2. [STAGED_EXPERIMENT_RESULTS.md](STAGED_EXPERIMENT_RESULTS.md) — полный отчёт staged-когорты.
3. [OVERNIGHT_EXPERIMENT_RESULTS.md](OVERNIGHT_EXPERIMENT_RESULTS.md) — overnight bundles.
4. [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) — ранний screening и полные траектории.
5. [EXPERIMENT_LOG_INDEX.md](EXPERIMENT_LOG_INDEX.md) — ручной индекс ключевых run-директорий.
6. [EXPERIMENT_INVENTORY.md](EXPERIMENT_INVENTORY.md) — автоматически обновляемый полный
   список worktree, run-директорий, events и checkpoint’ов.

## Как начать новый чат без восстановления контекста

Открыть чат из этого worktree и попросить модель сначала прочитать:

```text
EXPERIMENT_CONTEXT.md
EXPERIMENT_INVENTORY.md
TRAINING_PLAN.md
STAGED_EXPERIMENT_RESULTS.md
```

Перед новым запуском обновить инвентарь:

```bash
cd /home/evgenii/ws/isaac/cbr_i_results_full_trajectory
python3 scripts/experiment_inventory.py
```

Скрипт проверяет все worktree, зарегистрированные Git, поэтому старые логи из
отдельных experiment-worktree тоже остаются видимыми из канонической ветки.

## Правила запуска

- Запускать новые эксперименты из этого worktree, если нет причины использовать
  отдельную ветку. Тогда новые `logs/` сразу находятся рядом с документацией.
- Если нужен отдельный worktree, явно выставлять `PYTHONPATH` на его
  `source/CBRIIsaacLab`; branch name в `params/git.yaml` сам по себе недостаточен.
- Перед запуском supervisor проверяет эквивалентные локальные runs и пропускает
  дубли. `--allow-duplicate` использовать только для осознанной репликации.
- Незавершённую стадию возобновлять `staged_resume_experiments.py`; при resume
  сохраняются optimizer/scheduler, на границе curriculum они сбрасываются.
- После запуска записать в отчёт branch, commit, worktree, run directory, seed,
  timesteps, изменённые параметры, физические метрики и следующий эксперимент.

Чекпоинты и TensorBoard events намеренно не коммитятся в Git: это локальные
бинарные данные. Их точные пути и наличие перечисляются в
`EXPERIMENT_INVENTORY.md`; сами файлы остаются на общей машине и доступны новому
чату через этот индекс.
