# Индекс экспериментов

Каноническая ветка с кодом, планом, отчётами и индексом результатов:

```text
experiment/results-full-trajectory
```

Её постоянный worktree:

```bash
cd /home/evgenii/ws/isaac/cbr_i_results_full_trajectory
git branch --show-current
```

Если ветка уже используется этим worktree, не нужно делать `git switch` из
другого worktree: Git специально запрещает одновременно переключать одну ветку
в двух worktree.

Для нового чата сначала читать [EXPERIMENT_CONTEXT.md](EXPERIMENT_CONTEXT.md), а
затем обновлять [EXPERIMENT_INVENTORY.md](EXPERIMENT_INVENTORY.md):

```bash
python3 scripts/experiment_inventory.py
```

## Отчёты

- [Предыдущий screening на 32k environment timesteps](EXPERIMENT_RESULTS.md)
- [Ночная серия hypothesis bundles на 64k environment timesteps](OVERNIGHT_EXPERIMENT_RESULTS.md)
- [Общий план и правила запуска](TRAINING_PLAN.md)
- [Единый контекст для нового чата](EXPERIMENT_CONTEXT.md)
- [Полный локальный инвентарь checkpoint’ов и TensorBoard](EXPERIMENT_INVENTORY.md)

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
