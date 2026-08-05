# Отчёт: action regularization, 2026-08-04—2026-08-05

## Резюме

Серия проверяла, можно ли уменьшить дребезг policy двумя штрафами по **сырым
действиям**, до environment-side delta clamp:

- `action_magnitude_scale`: штраф величины raw action;
- `action_rate_scale`: штраф изменения raw action между соседними шагами.

Использованы два checkpoint-anchor:

1. unbounded noisy baseline (`baseline`, seed 42);
2. bounded `delta-task-repeat` (`task_balanced`, seed 43).

Запущено 8 screening jobs по 64k timesteps, затем по одному выбранному
regularized-варианту на anchor продолжено обучение на 128k timesteps и запущен
финальный dose-response stage на 64k с удвоенными ненулевыми коэффициентами.
Всего завершено 12/12 jobs, все с `returncode=0`.

Значения ниже — медиана последних 20 scalar-точек TensorBoard. Для `lifetime`,
`termination` и ошибок лучше больше/меньше соответственно; для `action_rate`,
`action_magnitude` и `saturation` меньше означает более спокойную policy.

## Screening: 64k

| Anchor / variant | Lifetime | Termination | Action rate | Action magnitude | Saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline / control | **713.2** | **0.00090** | 15.41 | 12.89 | 0.954 |
| baseline / magnitude | 306.6 | 0.00313 | 11.43 | 8.59 | 0.927 |
| baseline / rate | 13.8 | 0.07195 | **8.14** | 12.34 | 0.959 |
| baseline / combined | 11.5 | 0.08590 | 13.21 | 22.70 | 0.987 |
| task-balanced / control | 747.3 | 0.00075 | 0.813 | 0.722 | 0.467 |
| task-balanced / magnitude | **779.9** | **0.00070** | 0.835 | 0.728 | 0.477 |
| task-balanced / rate | 731.9 | 0.00084 | 0.804 | 0.728 | 0.477 |
| task-balanced / combined | 660.0 | 0.00106 | **0.803** | 0.725 | 0.472 |

## Продолжения

Супервайзер выбрал `rate`-вариант для обоих anchor, потому что при отсутствии
viable baseline-кандидата сортировка отдавала приоритет минимальному
`action_rate`. Для task-balanced это был разумный, хотя и небольшой, выигрыш;
для unbounded baseline rate penalty уменьшил шум ценой потери locomotion.

| Run | Timesteps | Lifetime | Termination | Action rate | Saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline / rate-continue | 128k | 13.7 | 0.07292 | 8.34 | 0.965 |
| baseline / rate-continue-strengthen | 64k | 15.6 | 0.06299 | 9.40 | 0.974 |
| task-balanced / rate-continue | 128k | **786.5** | **0.00075** | 0.814 | 0.490 |
| task-balanced / rate-continue-strengthen | 64k | 728.8 | 0.00098 | **0.773** | 0.493 |

## Выводы

1. **Raw-action regularization не спасает unbounded baseline.** Штрафы делают
   raw action тише, но baseline теряет устойчивую ходьбу почти сразу. Этот
   anchor не стоит использовать как основу для следующего обучения.

2. **Основной рабочий режим — bounded `task_balanced`.** Он сохраняет lifetime
   порядка 730–780 и termination около `1e-3`. Один magnitude penalty не дал
   ожидаемого снижения `action_rate`; combined-вариант ухудшил survival.

3. **Усиление rate penalty работает как dose-response.** У
   `rate-continue-strengthen` action rate снизился примерно на 5% относительно
   continuation, но lifetime снизился примерно на 7%. Это полезная настройка
   smoothness, но не лучший production/evaluation checkpoint.

4. **Рекомендованный checkpoint этой серии:**
   `task-balanced-rate-continue/agent_125000.pt`. Он лучше укреплённого варианта
   по survival и является более безопасным компромиссом для просмотра.

5. **Лучший lifetime во всём предыдущем наборе по-прежнему у
   `staged-control` (~821),** но это другой кандидат: без дополнительного
   усиленного raw-action rate penalty. Для чистого сравнения regularization
   следует использовать checkpoint из текущей серии.

## Артефакты

- Supervisor state:
  `/home/evgenii/ws/isaac/cbr_i_action_regularization/logs/action_regularization/2026-08-04_22-29-38/status.json`
- Рекомендованный checkpoint:
  `/home/evgenii/ws/isaac/cbr_i_action_regularization/logs/skrl/cbr_i_ppo/2026-08-05_04-20-51_experiment_action-regularization_efdee5d_clean_ppo_torch_action-reg-task-balanced-rate-continue/checkpoints/agent_125000.pt`
- Smoothness-first checkpoint:
  `/home/evgenii/ws/isaac/cbr_i_action_regularization/logs/skrl/cbr_i_ppo/2026-08-05_07-13-34_experiment_action-regularization_efdee5d_clean_ppo_torch_action-reg-task-balanced-rate-continue-strengthen/checkpoints/agent_60000.pt`

## TensorBoard

Текущий TensorBoard должен быть запущен с двумя log roots: каноническим и
action-regularization worktree. Нужен `--reload_multifile=true`, потому что
один skrl run содержит несколько event-файлов.

```bash
/home/evgenii/ws/isaac/env_isaaclab/bin/tensorboard \
  --logdir_spec="base:/home/evgenii/ws/isaac/cbr_isaac_lab/logs/skrl/cbr_i_ppo,action-regularization:/home/evgenii/ws/isaac/cbr_i_action_regularization/logs/skrl/cbr_i_ppo" \
  --port=6006 --reload_interval=5 --reload_multifile=true
```

## Просмотр рекомендованного checkpoint в Newton

`--viz newton` здесь означает Newton visualizer; policy и среда остаются теми
же, что в обучении. Для bounded `task-balanced` checkpoint важно явно включить
policy-side clipping, иначе `play.py` по умолчанию использует историческую
unbounded-конфигурацию baseline.

```bash
cd /home/evgenii/ws/isaac/cbr_i_action_regularization
PYTHONPATH=/home/evgenii/ws/isaac/cbr_i_action_regularization/source/CBRIIsaacLab \
  /home/evgenii/ws/isaac/IsaacLab/isaaclab.sh -p scripts/skrl/play.py \
  --task=Template-Cbriisaaclab-Direct-v0 \
  --checkpoint=/home/evgenii/ws/isaac/cbr_i_action_regularization/logs/skrl/cbr_i_ppo/2026-08-05_04-20-51_experiment_action-regularization_efdee5d_clean_ppo_torch_action-reg-task-balanced-rate-continue/checkpoints/agent_125000.pt \
  --num_envs=16 --viz=newton --max_visible_envs=16 \
  --policy_clip_actions --real-time
```
