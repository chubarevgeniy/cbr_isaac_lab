# Индекс локальных логов экспериментов

Каноническая ветка `experiment/staged-training` содержит этот индекс и отчёты.
Сами TensorBoard events и `.pt` checkpoint-файлы хранятся локально в
`logs/archive/`, потому что `logs/` исключён из Git.

## Основные staged run-директории

### `reward-curriculum`

- Stage 1, reused duplicate: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_staged_reward_curriculum/skrl/cbr_i_ppo/2026-08-04_08-41-25_experiment_staged-reward-curriculum_f03dc4a_clean_ppo_torch_staged-reward-curriculum-survival`
- Stage 2: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_staged_reward_curriculum/skrl/cbr_i_ppo/2026-08-04_11-09-32_experiment_staged-reward-curriculum_db7f54c_clean_ppo_torch_resume-reward-curriculum-task`
- Stage 3: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_staged_reward_curriculum/skrl/cbr_i_ppo/2026-08-04_14-01-58_experiment_staged-reward-curriculum_db7f54c_clean_ppo_torch_resume-reward-curriculum-tracking`

### `easy-to-robust`

- Stage 1, reused duplicate: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_staged_easy_to_robust/skrl/cbr_i_ppo/2026-08-04_08-41-25_experiment_staged-easy-to-robust_f03dc4a_clean_ppo_torch_staged-easy-to-robust-easy-survival`
- Stage 2: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_staged_easy_to_robust/skrl/cbr_i_ppo/2026-08-04_11-43-06_experiment_staged-easy-to-robust_db7f54c_clean_ppo_torch_resume-easy-to-robust-robust-task`
- Stage 3: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_staged_easy_to_robust/skrl/cbr_i_ppo/2026-08-04_14-38-31_experiment_staged-easy-to-robust_db7f54c_clean_ppo_torch_resume-easy-to-robust-robust-tracking`

### `easy-task-to-robust`

- Stage 1, resumed from `agent_40000.pt`: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_staged_easy_task_to_robust/skrl/cbr_i_ppo/2026-08-04_11-09-32_experiment_staged-easy-task-to-robust_db7f54c_clean_ppo_torch_resume-easy-task-to-robust-easy-task`
- Stage 2: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_staged_easy_task_to_robust/skrl/cbr_i_ppo/2026-08-04_13-09-47_experiment_staged-easy-task-to-robust_db7f54c_clean_ppo_torch_resume-easy-task-to-robust-robust-task`
- Stage 3: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_staged_easy_task_to_robust/skrl/cbr_i_ppo/2026-08-04_16-08-20_experiment_staged-easy-task-to-robust_db7f54c_clean_ppo_torch_resume-easy-task-to-robust-robust-tracking`

### `staged-control`

- Stage 1, reused duplicate: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_overnight_delta_task_repeat/skrl/cbr_i_ppo/2026-08-04_03-11-42_experiment_overnight-delta-task-repeat_7a882e0_clean_ppo_torch_overnight-delta-task-repeat`
- Stage 2: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_staged_staged_control/skrl/cbr_i_ppo/2026-08-04_12-36-22_experiment_staged-control_db7f54c_clean_ppo_torch_resume-staged-control-task-2`
- Stage 3: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_staged_staged_control/skrl/cbr_i_ppo/2026-08-04_15-30-43_experiment_staged-control_db7f54c_clean_ppo_torch_resume-staged-control-task-3`

Итоговый checkpoint лучшего по lifetime Stage 3:

```text
/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_staged_staged_control/skrl/cbr_i_ppo/2026-08-04_15-30-43_experiment_staged-control_db7f54c_clean_ppo_torch_resume-staged-control-task-3/checkpoints/agent_60000.pt
```

## Overnight reference roots

- `long-baseline`: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_overnight_long_baseline/skrl/cbr_i_ppo`
- `delta-bounded`: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_overnight_delta_bounded/skrl/cbr_i_ppo`
- `delta-survival`: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_overnight_delta_survival/skrl/cbr_i_ppo`
- `delta-smooth-clearance`: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_overnight_delta_smooth_clearance/skrl/cbr_i_ppo`
- `absolute-target`: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_overnight_absolute_target/skrl/cbr_i_ppo`
- `absolute-safe-task`: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_overnight_absolute_safe_task/skrl/cbr_i_ppo`
- `absolute-clearance`: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_overnight_absolute_clearance/skrl/cbr_i_ppo`
- `delta-task-repeat`: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_overnight_delta_task_repeat/skrl/cbr_i_ppo`

## Action-regularization cohort

Полный отчёт: [ACTION_REGULARIZATION_EXPERIMENT_REPORT.md](ACTION_REGULARIZATION_EXPERIMENT_REPORT.md).

- Supervisor state: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_action_regularization/action_regularization/2026-08-04_22-29-38/status.json`
- Все новые event-файлы и checkpoint’ы: `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_action_regularization/skrl/cbr_i_ppo`
- Рекомендованный checkpoint: `action-reg-task-balanced-rate-continue/checkpoints/agent_125000.pt`

## TensorBoard для всей серии

```bash
/home/evgenii/ws/isaac/env_isaaclab/bin/tensorboard \
  --logdir_spec="base:/home/evgenii/ws/isaac/cbr_isaac_lab/logs/skrl/cbr_i_ppo,\
long-baseline:/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_overnight_long_baseline/skrl/cbr_i_ppo,\
delta-bounded:/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_overnight_delta_bounded/skrl/cbr_i_ppo,\
delta-survival:/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_overnight_delta_survival/skrl/cbr_i_ppo,\
delta-smooth-clearance:/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_overnight_delta_smooth_clearance/skrl/cbr_i_ppo,\
absolute-target:/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_overnight_absolute_target/skrl/cbr_i_ppo,\
absolute-safe-task:/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_overnight_absolute_safe_task/skrl/cbr_i_ppo,\
absolute-clearance:/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_overnight_absolute_clearance/skrl/cbr_i_ppo,\
delta-task-repeat:/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_overnight_delta_task_repeat/skrl/cbr_i_ppo,\
staged-reward:/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_staged_reward_curriculum/skrl/cbr_i_ppo,\
staged-easy-robust:/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_staged_easy_to_robust/skrl/cbr_i_ppo,\
staged-easy-task:/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_staged_easy_task_to_robust/skrl/cbr_i_ppo,\
staged-control:/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_staged_staged_control/skrl/cbr_i_ppo,\
action-regularization:/home/evgenii/ws/isaac/cbr_isaac_lab/logs/archive/cbr_i_action_regularization/skrl/cbr_i_ppo" \
  --port=6006 --reload_interval=5 --reload_multifile=true
```

Supervisor status and stdout:

- `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/staged_resume/2026-08-04_11-09-27/status.json`
- `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/staged_resume/supervisor.stdout.log`
