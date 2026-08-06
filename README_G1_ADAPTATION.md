# Unitree G1 → CBR-I: reward и объём обучения

Этот файл фиксирует адаптацию подхода Unitree G1 к текущей задаче CBR-I:
сравнение reward-компонент, объёма PPO-обучения и принятого контракта
координат/action. Raw-знаки USD сохраняются только на границе симулятора.

## Источник сравнения

В качестве открытого источника используется официальный репозиторий Unitree:

- [unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab);
- задача `Unitree-G1-29dof-Velocity`;
- зафиксированный upstream-коммит
  [`4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`](https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3);
- [конфигурация reward и среды G1](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py);
- [конфигурация PPO Unitree](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py).

Текущая CBR-I реализация проверена по файлам:

- [`cbriisaaclab_env_cfg.py`](source/CBRIIsaacLab/CBRIIsaacLab/tasks/direct/cbriisaaclab/cbriisaaclab_env_cfg.py);
- [`cbriisaaclab_env.py`](source/CBRIIsaacLab/CBRIIsaacLab/tasks/direct/cbriisaaclab/cbriisaaclab_env.py);
- [`skrl_ppo_cfg.yaml`](source/CBRIIsaacLab/CBRIIsaacLab/tasks/direct/cbriisaaclab/agents/skrl_ppo_cfg.yaml);
- [`TRAINING_PLAN.md`](TRAINING_PLAN.md).

Численные веса нельзя переносить напрямую: у Unitree G1 manager-based-среда,
29 степеней свободы, root velocity, contact sensors и terrain curriculum. У
CBR-I direct-среда с четырьмя управляемыми суставами, другой геометрией и
режимами walking/sitting. Ниже сопоставляется физический смысл величин, а не
только одинаковые имена параметров.

## Как считается текущая reward CBR-I

В [`compute_rewards()`](source/CBRIIsaacLab/CBRIIsaacLab/tasks/direct/cbriisaaclab/cbriisaaclab_env.py#L889)
сначала выбирается walking- или sitting-часть, затем добавляются общие
слагаемые:

```text
R = R_mode + R_alive + R_stability + R_pose
```

Для walking `R_mode` и применимые Unitree-термы имеют следующий вид:

```text
R_walk = +1.0 * exp(-((body_vel - target_speed) / 0.5)^2)
          -2.0 * body_vertical_velocity^2
          -0.05 * body_angular_velocity^2
          -0.001 * sum(joint_velocity^2)
          -0.05 * sum((action - previous_action)^2)
          -5.0 * joint_position_limit_violation
          -0.5 * |body_angle|
          -0.5 * sum(|canonical_joint_position|)
          -2.5 * body_angle^2
          -5.0 * (body_height - 0.1309)^2
          -0.5 * sum(target_limit_violation^2)
          -0.01 * sum((target - current_joint_position)^2)
```

Здесь policy/reward-контракт теперь такой:

- `body_vel = 1 m * qdot(Rock_Revolute_1)` — продольная скорость-прокси;
- `body_height = -1 m * q(bottom_rotor_Revolute_2)` — высота-прокси;
- `q_hip` и `q_knee` в reward/observation переводятся в канонические
  bilateral-знаки;
- все angular velocities в observation остаются raw в `rad/s`;
- `qdot(bottom_rotor_Revolute_2)` остаётся raw в observation и является
  вертикальной скоростью-прокси `-1 m * qdot` для аналога Unitree
  `base_linear_velocity`; в reward это явно умножается на
  `height_velocity_proxy_lever_arm = 1 m`;
- `qdot(rod_1_Revolute_3)` используется как угловая скорость корпуса-прокси;
- `action_rate` сравнивает текущий и предыдущий action, как в Unitree.
- `joint_acc`, `applied_torque`, gait, foot clearance, foot slide и contacts
  намеренно не участвуют в reward: для implicit actuator и текущей геометрии
  CBR-I мы не вводим эти дополнительные сигналы и сенсоры.

Для walking теперь используется Unitree-подобный exp-трекинг одной
продольной скорости:

```text
R_track = exp(-((body_vel - command_speed) / 0.5)^2)
```

У Unitree исходный `track_lin_vel_xy` двумерный; здесь оставлена только
доступная продольная компонента. Standing target для height-прокси вычислен
CPU forward-kinematics поиском по текущему USD: при canonical
`[hip_R, hip_L, knee_R, knee_L] = [0, 0, 0, 0] deg` и `rod_body = 0 deg`
первый ground-safe шаг сетки 0.5° — это `q(bottom_rotor) = -7.5 deg`.
Поэтому в reward используется
`body_height_target = -1 m * (-7.5 deg) = +0.1309 m` и term
`-10 * (body_height - 0.1309)^2`. Это target высоты-прокси, а не абсолютная
высота root; при этой проверочной позе физические ориентиры составили
примерно `torso z = 0.414 m` и `head z = 0.572 m`. Точка `-7.0 deg` ещё давала
проникновение одной голени в пол, поэтому выбрана `-7.5 deg`, а не только
геометрическая точка касания визуальной стопы.

Это метрические прокси с выбранным плечом 1 m, а не замена точной кинематики:
точная вертикальная скорость конца балки должна вычисляться через Jacobian.
Foot/contact terms Unitree пока не переносятся: отдельного ankle/foot link и
контактного сенсора в этой задаче нет.

### Канонические joint coordinates и action

Положим `q_down = 130 deg = 2.268928 rad`. Для порядка
`[right hip, left hip, right knee, left knee]`:

```text
hip_R  = q_down - raw_hip_R
hip_L  = raw_hip_L + q_down
knee_R = -raw_knee_R
knee_L = raw_knee_L
```

Таким образом, обе ноги используют один знак: `hip = 0` означает бедро вниз,
положительный hip — сгибание к корпусу; `knee = 0` — полностью выпрямленное
колено, `knee = 124 deg` — максимальное сгибание. Сидячий target в канонических
координатах симметричен: `[130, 130, 124, 124] deg`; reference
`[0, 0, 0, 0]` — бедра вниз и прямые колени.

Action больше не накапливается в `targets`. Используется Unitree-подобное
affine-преобразование:

```text
target_canonical = action_default_target + action_scale * action
```

где action передаётся без клипирования, `action_default_target = [0, 0, 0, 0]`,
а scale уменьшен до `65°` для hip и `62°` для knee на единицу action. Поэтому
action `1` больше не покрывает весь диапазон сустава. Target переводится
обратно в raw USD-знаки перед `set_joint_position_target`; превышение лимитов
штрафуется отдельно.
Четыре target-поля старого delta-контракта заменены в observation на четыре
компоненты текущего `last_action`, как в конфигурации Unitree G1. Reference
Unitree scale `0.25 rad` сохранён в конфиге для отдельного сравнительного
эксперимента, но не используется в основном CBR-I sit/walk baseline.

Для sitting используется та же стабилизирующая структура, что и для
stand-still: скорость target равна нулю, применяются vertical/angular velocity,
joint velocity, action-rate, limits и alive. Отличаются только pose targets:
height-прокси `-0.0908 m`, `rod_body=-80°`, hips `130°`, knees `124°`. Все
угловые deviation terms умножены на `2.0`, чтобы робот не получал почти такую
же награду за грубо похожую позу.

Общие слагаемые имеют такие настройки:

| Слагаемое | Настройка | Фактический смысл |
| --- | ---: | --- |
| termination | нет | termination только завершает эпизод, отдельного death penalty нет |
| alive | `+0.15` | `+0.15` на живом шаге, как у Unitree |
| action rate | `-0.05` | `-0.05 * sum((action - previous_action)²)` |
| sitting angular multiplier | `2.0` | усиливает соответствие body/hip/knee targets |
| skrl reward shaper | `0.1` | масштабирует весь reward перед PPO |

## Соответствие reward-компонент Unitree G1

Знак `+` в конфигурации Unitree означает полезную reward-функцию, знак `-` —
штраф. У CBR-I большинство ошибок сразу умножается на отрицательный scale.
Коэффициенты в таблице не являются взаимозаменяемыми без нормализации и
проверки диапазонов.

| Unitree G1 term | Вес Unitree | Что измеряет | Ближайший аналог CBR-I сейчас | Оценка соответствия |
| --- | ---: | --- | --- | --- |
| `track_lin_vel_xy` | `+1.0` | отслеживание линейной скорости по `x/y`, exp-kernel | `exp(-((body_vel - command[:, 1]) / 0.5)^2)` в walking | Адаптировано: одна продольная скорость вместо 2D |
| `track_ang_vel_z` | `+0.5` | отслеживание yaw rate | Нет | Добавлять только если у CBR-I появится физически осмысленная угловая команда |
| `alive` | `+0.15` | продолжение эпизода без падения | `R_alive = +0.15` | Перенесено |
| `base_linear_velocity` | `-2.0` | вертикальная скорость корпуса `root_lin_vel_z` | `-2.0 * (-qdot(bottom_rotor))²` | Адаптировано через height-rate proxy |
| `base_angular_velocity` | `-0.05` | roll/pitch angular velocity корпуса | `-0.05 * qdot(rod_body)²` | Адаптировано через body-joint velocity |
| `joint_vel` | `-0.001` | L2 penalty скоростей суставов | `-0.001 * sum(qvel²)` для 4 hip/knee | Перенесено |
| `joint_acc` | `-2.5e-7` | плавность/ускорения суставов | Нет: `joint_acc` намеренно не читается | Не переносится |
| `action_rate` | `-0.05` | изменение action между шагами | `-0.05 * sum((a_t-a_{t-1})²)` | Перенесено |
| `dof_pos_limits` | `-5.0` | приближение к пределам суставов | текущая физическая позиция вне soft limits получает штраф; raw action target получает отдельный quadratic limit penalty `-0.5` | Адаптировано |
| `energy` | `-2e-5` | `|joint velocity| * |applied torque|` | Нет: `applied_torque` намеренно не читается | Не переносится |
| `joint_deviation_arms` | `-0.1` | удержание рук около default pose | Нет рук в текущей модели | Не переносить |
| `joint_deviation_waists` | `-1.0` | отклонение waist joints от default | `-0.5*|body_angle-target|` для walking; sitting multiplier `2.0` | Адаптировано |
| `joint_deviation_legs` | `-1.0` | отклонение leg joints от default | `-0.5*sum(|canonical joint-target|)` для walking; sitting multiplier `2.0` | Адаптировано для 4 hip/knee |
| `flat_orientation_l2` | `-5.0` | upright orientation корпуса | `-2.5*(body_angle-target)²` для walking; sitting multiplier `2.0` | Угловой proxy; sitting target заменяет upright |
| `base_height` | `-10.0`, target `0.78 m` | высота root корпуса | `-5.0 * (body_height - 0.1309)^2` для walking, где `body_height = -1 m * q_beam` | Адаптировано: target вычислен FK, это height-прокси, не root height |
| `gait` | `+0.5` | фазовое соответствие контактов стоп | Нет: контактный сенсор не создаётся | Не переносится |
| `feet_slide` | `-0.2` | горизонтальное скольжение контактирующей стопы | Нет отдельного contact-gated терма | Не переносится |
| `feet_clearance` | `+1.0`, target `0.1 m` | подъём swing foot до целевой высоты | Нет foot/contact reward | Не переносится |
| `undesired_contacts` | `-1.0` | контакт корпуса/не-стоп с землёй | Нет контактного сенсора | Не переносится |

### Termination и режимы

Это не reward-термы, но они влияют на ту же статистику обучения:

| Подход Unitree G1 | Текущий CBR-I |
| --- | --- |
| `time_out` при episode length `20 s` | `episode_length_s = 25 s` |
| падение при root height `< 0.2 m` | падение при `head height < 0.1 m` |
| bad orientation при угле `> 0.8 rad` | `rotor_rod > 8.9°` |
| push каждые `5 s`, mass/friction randomization, terrain curriculum | свои randomization friction/gain/gravity и randomized начальные позы |

## Что имеет смысл адаптировать первым

1. Для upright и валидации высоты логировать физические `torso_height`,
   `head_height` и ориентацию корпуса вместе с `body_height`-прокси. Не путать
   прокси с точной высотой root: в CBR-I он построен из угла
   `bottom_rotor_Revolute_2`.
2. Не переносить термы рук, G1-specific gait и terrain-specific contact terms
   без соответствующего сенсора/геометрии.
3. `joint_acc` и `energy` не включать в текущий baseline: для implicit actuator
   отсутствуют нужные сигналы в согласованном reward-контракте.

## Сравнение PPO и объёма обучения

### Исходные настройки

Текущий адаптированный baseline берётся из
[`skrl_ppo_cfg.yaml`](source/CBRIIsaacLab/CBRIIsaacLab/tasks/direct/cbriisaaclab/agents/skrl_ppo_cfg.yaml)
и длинного запуска из [`TRAINING_PLAN.md`](TRAINING_PLAN.md):

- `num_envs = 2048`;
- `rollouts = 24`;
- `trainer.timesteps = 400000`;
- это примерно `400000 / 24 = 16667` PPO updates;
- `learning_epochs = 5`, `mini_batches = 4`;
- `learning_rate = 1e-3`;
- `gamma = 0.99`, `lambda = 0.95`;
- сеть policy/value: `[128, 64, 32]`.

В скрипте [`scripts/skrl/train.py`](scripts/skrl/train.py) параметр
`--max_iterations=N` переопределяет timesteps как `N * rollouts`. Поэтому
`--max_iterations=16667` задаёт примерно `400008` timesteps; без этого флага
используется `trainer.timesteps = 400000` из YAML.

Для справки, до адаптации в рабочем конфиге были `rollouts=32`,
`learning_epochs=8`, `mini_batches=8`, `gamma=0.995`, `learning_rate=5e-4` и
`800000` timesteps. Это историческая точка сравнения, а не активный baseline.

У Unitree G1:

- `num_envs = 4096`;
- `num_steps_per_env = 24`;
- `max_iterations = 50000`;
- `num_learning_epochs = 5`, `num_mini_batches = 4`;
- `learning_rate = 1e-3`;
- `gamma = 0.99`, `lam = 0.95`;
- сеть actor/critic: `[512, 256, 128]`.

### Расчёт

| Показатель | CBR-I адаптированный | До адаптации CBR-I | Unitree G1 |
| --- | ---: | ---: | ---: |
| PPO updates | `≈16667` | `25000` | `50000` |
| envs | `2048` | `2048` | `4096` |
| rollout steps/env/update | `24` | `32` | `24` |
| transitions/update | `2048 * 24 = 49152` | `2048 * 32 = 65536` | `4096 * 24 = 98304` |
| steps на одно env за весь run | `400000` | `800000` | `24 * 50000 = 1200000` |
| всего transitions за run | `8.192e8` | `1.6384e9` | `4.9152e9` |
| PPO learning epochs/update | `5` | `8` | `5` |
| optimizer minibatch size | `49152 / 4 = 12288` | `65536 / 8 = 8192` | `98304 / 4 = 24576` |
| minibatch optimizer passes/update | `5 * 4 = 20` | `8 * 8 = 64` | `5 * 4 = 20` |
| sample presentations/update | `49152 * 5 = 245760` | `65536 * 8 = 524288` | `98304 * 5 = 491520` |
| learning rate | `1e-3` | `5e-4` | `1e-3` |

Главный вывод по текущему адаптированному baseline: Unitree использует в три
раза больше PPO updates, вдвое больше окружений и в три раза больше шагов на
одно окружение. Поэтому полный Unitree run содержит примерно **в 6 раз больше
environment transitions** (`4.9152e9` против `8.192e8`). Active PPO-параметры
уже совпадают по `rollouts`, `learning_epochs`, `mini_batches`, learning rate и
discounting; отличается главным образом общий объём.

Это не означает, что нужно сразу запускать CBR-I на `4096` окружениях: CBR-I и
G1 имеют разную размерность состояния/действий, а требуемая VRAM неизвестна.
Сначала нужно повторить пропорции на доступной конфигурации и сравнивать
`transitions`, `samples/sec`, wall-clock time, падения и физические метрики.

### Быстрый smoke-test

Текущий быстрый gate из `README.md` использует `--max_iterations=1000`:

```text
1000 * 24 = 24000 steps на одно env
24000 * 2048 = 49152000 transitions
```

Это sanity-check, а не аналог полного Unitree run. Для сравнения reward-изменений
нужно сохранять одинаковые `seed`, `num_envs`, `rollouts` и число environment
steps для всех вариантов.

## Критерии следующего этапа

Reward не следует оценивать только по среднему числу в TensorBoard. Для каждого
изменения нужно сохранить:

- `Physical/walk/speed_error_abs` и signed error по направлениям;
- высоты torso/head/knee/foot и ошибки углов sitting;
- `Physical/termination/terminated_rate` и `timeout_rate`;
- диапазоны action, target и фактических joint states;
- transitions, samples/sec, PPO updates и wall-clock time.

Порядок экспериментов: baseline → одна новая физическая величина → повтор на
нескольких seed → решение о переносе следующей величины. Это позволит отличить
рост численной reward от реального улучшения поведения CBR-I.
