# Индекс экспериментов

Каноническая ветка с документацией результатов:

```text
experiment/results-full-trajectory
```

Переключение:

```bash
git switch experiment/results-full-trajectory
```

## Отчёты

- [Предыдущий screening на 32k environment timesteps](EXPERIMENT_RESULTS.md)
- [Ночная серия hypothesis bundles на 64k environment timesteps](OVERNIGHT_EXPERIMENT_RESULTS.md)
- [Общий план и правила запуска](TRAINING_PLAN.md)

## Локальные артефакты

Git-ветки содержат код и markdown, а TensorBoard event-файлы намеренно остаются
локальными в worktree, чтобы не раздувать репозиторий. Реестр ночной серии:

```text
logs/overnight/2026-08-03_22-43-20/status.json
```

TensorBoard был настроен на `localhost:6006` с `--reload_multifile=true` и
агрегирует базовые логи и логи всех overnight-worktree.

## Общие проверки

- Ветка, commit, worktree и dirty-файлы записаны в `params/git.yaml` каждого run.
- Фактические `action_mode`, `reward_profile`, policy clipping и launch arguments
  проверены в `params/env.yaml`, `params/agent.yaml` и `params/launch.yaml`.
- Полные overnight-run используют `num_envs=2048`, `rollouts=32` и `64000`
  environment timesteps (`max_iterations=2000`).
- Сравнение сделано по физическим scalar-метрикам, а не только по reward:
  lifetime/termination, clearance, speed tracking, sit error и action smoothness.
- Для абсолютных targets проверены soft joint limits и safety limiter
  `absolute_target_step_limit=0.08`.
- Для start/sit/stand проверены reset-команды, распределение standing-start и
  переходы sitting ↔ standing в environment.
- Ранний stop применён только к `delta-bounded` после многокритериального
  ухудшения survival; checkpoint и run directory сохранены.
- Очередь supervisor завершена, все восемь bundle-веток зарегистрированы.
