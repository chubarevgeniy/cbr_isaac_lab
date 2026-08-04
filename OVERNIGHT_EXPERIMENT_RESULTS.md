# Результаты overnight hypothesis bundles

Дата серии: `2026-08-03/04`  
Каноническая ветка документации: `experiment/results-full-trajectory`  
Реестр: [`logs/overnight/2026-08-03_22-43-20/status.json`](logs/overnight/2026-08-03_22-43-20/status.json)  
Supervisor log: [`logs/overnight/supervisor-daemon.log`](logs/overnight/supervisor-daemon.log)

## Протокол

- `num_envs=2048`, `rollouts=32`, `max_iterations=2000`.
- Это `64000` environment timesteps — вдвое длиннее исходного screening на
  `32000` timesteps.
- Два запуска выполнялись параллельно.
- Семь bundle-веток завершились с `returncode=0`; `delta-bounded` был остановлен
  примерно на `20k` timesteps после 30 минут из-за существенно худшего survival.
- В таблице ниже приведено среднее последних пяти записанных scalar-точек.
  `lifetime` — `Episode / Total timesteps (mean)`, `speed` и `sit` — ошибки,
  поэтому для них меньше лучше. `foot` — средняя высота стопы, больше лучше.

## Сводка

| Bundle | Action / reward | Lifetime | Termination | Foot | Speed error | Sit error | Action rate / saturation | Решение |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `delta-survival` | delta / survival+clearance+speed | **752.7** | **0.00067** | **0.0198** | 0.634 | 0.505 | 0.85 / 0.47 | Лучший survival и clearance |
| `delta-task-repeat` | delta / task-balanced | **722.7** | **0.00086** | 0.0172 | 0.568 | 0.444 | **0.82 / 0.46** | Лучший компромисс |
| `long-baseline` | delta / baseline, unbounded | 599.6 | 0.00133 | 0.0155 | **0.371** | **0.316** | 14.94 / 0.95 | Reference по speed/sit, но сильный jitter |
| `absolute-clearance` | absolute + limiter / clearance+speed | 525.8 | 0.00169 | 0.0164 | 0.594 | 0.613 | 0.81 / 0.48 | Интересный survival, плохой sit/speed |
| `delta-smooth-clearance` | delta / clearance+rate | 276.9 | 0.00327 | 0.0181 | 0.504 | 0.353 | 0.83 / 0.44 | Clearance есть, survival слабый |
| `absolute-target` | absolute + limiter / baseline | 254.7 | 0.00397 | 0.0173 | **0.422** | 0.374 | **0.76 / 0.43** | Самый гладкий absolute-вариант |
| `absolute-safe-task` | absolute + limiter / task-balanced | 255.6 | 0.00415 | 0.0175 | 0.528 | 0.362 | 0.78 / 0.45 | Survival пока слабый |
| `delta-bounded` | delta / baseline, policy clip | 242.9* | 0.00443* | 0.0176* | 0.470* | 0.241* | 0.85 / 0.42 | Остановлен рано |

`*` — значения остановленного run, не полный 64k-прогон.

## Интерпретация

1. Если главным критерием считать «не падать», текущий кандидат —
   `delta-survival`: lifetime примерно `753` против `600` у baseline, а
   clearance также лучший.
2. Для более сбалансированной policy предпочтительнее `delta-task-repeat`:
   survival почти такой же, но ошибка скорости и sit заметно меньше, чем у
   `delta-survival`.
3. `long-baseline` хорошо отслеживает скорость и садится, но сырые delta-action
   дают `mean_abs_rate≈14.9` и saturation около `95%`. Это подтверждает исходную
   претензию к дребезгу.
4. Policy-side clipping снизил action rate примерно до `0.76–0.85`, а saturation
   до `42–48%`. Гипотеза о более равномерном распределении действий подтверждена,
   но одного clipping недостаточно для хорошего lifetime.
5. Absolute-target с safety limiter работает технически и даёт гладкие действия,
   но пока проигрывает delta-кандидатам по survival. `absolute-clearance` требует
   отдельной настройки reward для speed/sit.

## Проверки реализации и воспроизводимости

- Каждый run содержит `params/env.yaml`, `params/agent.yaml`, `params/launch.yaml`
  и `params/git.yaml`.
- В `params` подтверждены соответствующие `action_mode`, `reward_profile`,
  `policy_clip_actions`, `initial_log_std` и `max_log_std`.
- Для absolute bundles подтверждены `action_mode: absolute` и
  `absolute_target_step_limit: 0.08`.
- Reset/environment подтверждают standing-start для 70% окружений, начальную
  sitting-команду для остальных и переходы между sitting и standing.
- Полные run имеют 100 scalar-точек до шага 64000 и 64 histogram-точки от 1000
  до 64000. У `delta-bounded` данные заканчиваются на раннем stop.
- TensorBoard должен запускаться с `--reload_multifile=true`, потому что skrl и
  histogram writer создают два event-файла в одном run directory.

## Run directories

- `long-baseline`: `/home/evgenii/ws/isaac/cbr_i_overnight_long_baseline/logs/skrl/cbr_i_ppo/`
- `delta-bounded`: `/home/evgenii/ws/isaac/cbr_i_overnight_delta_bounded/logs/skrl/cbr_i_ppo/`
- `delta-survival`: `/home/evgenii/ws/isaac/cbr_i_overnight_delta_survival/logs/skrl/cbr_i_ppo/`
- `delta-smooth-clearance`: `/home/evgenii/ws/isaac/cbr_i_overnight_delta_smooth_clearance/logs/skrl/cbr_i_ppo/`
- `absolute-target`: `/home/evgenii/ws/isaac/cbr_i_overnight_absolute_target/logs/skrl/cbr_i_ppo/`
- `absolute-safe-task`: `/home/evgenii/ws/isaac/cbr_i_overnight_absolute_safe_task/logs/skrl/cbr_i_ppo/`
- `absolute-clearance`: `/home/evgenii/ws/isaac/cbr_i_overnight_absolute_clearance/logs/skrl/cbr_i_ppo/`
- `delta-task-repeat`: `/home/evgenii/ws/isaac/cbr_i_overnight_delta_task_repeat/logs/skrl/cbr_i_ppo/`
