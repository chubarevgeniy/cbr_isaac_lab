# Результаты полных тренировочных траекторий

Сравнение выполнено по всей записанной траектории, без фильтрации последних 10%.

## Протокол и валидация

- IsaacLab task: `Template-Cbriisaaclab-Direct-v0`.
- `num_envs=2048`, `max_iterations=1000`, `rollouts=32`, то есть `32 000` trainer/environment steps.
- Все одиночные варианты используют seed 42; tri-factor gate записан двумя seed (42 и 43).
- Видео не включалось ни в одном valid run.
- Scalar summary: простое среднее 100 равномерно записанных точек `320..32000` и отдельная последняя точка.
- Histogram summary: простое среднее 32 уникальных точек `1000..32000`; у baseline отброшены дублированные служебные записи step 1000 с `num=16` в пользу записей с `num=2048`.

Проверка полноты: у каждого valid run `66` scalar tags × `100` точек и `24` histogram tags × `32` уникальных шага.

## Реестр запусков

| ID | Изменение | Branch / commit | Каталог лога |
| --- | --- | --- | --- |
| `baseline` | Baseline | `master` / `b18a78a` | `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/skrl/cbr_i_ppo/2026-08-03_18-32-48_master_b18a78a_clean_ppo_torch` |
| `A` | A: no observation noise | `experiment/no-observation-noise` / `28a019d` | `/home/evgenii/ws/isaac/cbr_i_no_observation_noise/logs/skrl/cbr_i_ppo/2026-08-03_21-23-35_experiment_no-observation-noise_28a019d_clean_ppo_torch` |
| `B` | B: no initial tilt | `experiment/no-initial-tilt` / `c583af0` | `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/skrl/cbr_i_ppo/2026-08-03_20-51-24_experiment_no-initial-tilt_c583af0_clean_ppo_torch` |
| `C` | C: four PPO mini-batches | `experiment/ppo-four-mini-batches` / `d527b59` | `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/skrl/cbr_i_ppo/2026-08-03_20-05-31_experiment_ppo-four-mini-batches_d527b59_clean_ppo_torch` |
| `tri42` | A+B+C, seed 42 | `experiment/tri-factor-gate` / `da68462` | `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/skrl/cbr_i_ppo/2026-08-03_19-20-44_experiment_tri-factor-gate_da68462_clean_ppo_torch` |
| `tri43` | A+B+C, seed 43 | `experiment/tri-factor-gate` / `da68462` | `/home/evgenii/ws/isaac/cbr_isaac_lab/logs/skrl/cbr_i_ppo/2026-08-03_19-21-20_experiment_tri-factor-gate_da68462_clean_ppo_torch` |

### Исключённый запуск

Первый запуск A в `...20-05-31_experiment_no-observation-noise_28a019d...` не используется: его `git.yaml` указывал branch A, но `params/env.yaml` и `params/agent.yaml` показали `add_noise: true` и `mini_batches: 4`. Причина — editable install `CBRIIsaacLab` указывал на основной worktree. Корректный A перезапущен с явным `PYTHONPATH` на source A и находится в реестре выше.

## Вывод по факторам

- **A — убрать observation noise:** полный mean `Physical/walk/rod_body_angle_abs` изменился с `0.4434` до `0.3029` (−31.7%), termination rate снизился на 8.3%, но `speed_error_abs` вырос на 9.5%, а sitting `mean_joint_angle_error_abs` — на 17.0%. Это не безусловное улучшение.
- **B — убрать initial tilt:** `rod_body_angle_abs` снизился на 21.8%, `rod_body_angular_velocity_abs` на 16.9%, `body_velocity_abs` на 19.0%; moving fraction почти не изменилась (−1.1%), а termination rate выросла на 6.3%. Фактор выглядит наиболее ровным кандидатом на стабильность, но выигрыш не универсален.
- **C — 4 mini-batches:** PPO update time снизился на 34.9%, но environment stepping time вырос на 60.2%; moving fraction снизилась на 13.4%, mean episode timesteps — на 53.8%, termination rate выросла в 2.17 раза. Это скорее trade-off throughput/качество, а не готовое улучшение.
- **A+B+C gate:** оба seed показали общий сдвиг по ключевым физическим метрикам: `moving_fraction` +6.4/+6.8%, `rod_body_angle_abs` −24.1/−56.2%, `body_velocity_abs` +12.4/+11.6%, `algorithm update time` −36.6/−36.8%. При этом выросли positive-command speed error (+8.7/+20.2%) и sitting joint error (+15.5/+8.7%). Gate имеет эффект, поэтому три фактора не вычёркиваются автоматически; заметна интеракция, особенно для rod stability.
- Окончательный выбор требует более длинного протокола/повторов: текущие 32k steps — быстрый screening, а не статистически надёжное доказательство качества на 800k steps.

## Scalar metrics — среднее по всей траектории

| Metric | baseline | A | B | C | tri42 | tri43 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Episode / Total timesteps (max) | 1227.43 | 1228.99 | 1214.75 | 476.63 | 1222.88 | 1213.34 |
| Episode / Total timesteps (mean) | 378.2039 | 422.4503 | 357.8551 | 174.8675 | 478.8773 | 373.7453 |
| Episode / Total timesteps (min) | 51.75 | 50.87 | 49.09 | 21.18 | 55.35 | 8.86 |
| Learning / Learning rate | 0.001612063 | 0.001653781 | 0.001614781 | 0.002915089 | 0.002494273 | 0.002562629 |
| Loss / Entropy loss | -0.1234117 | -0.1208902 | -0.1232247 | -0.1245521 | -0.1232905 | -0.121028 |
| Loss / Policy loss | -0.007997795 | -0.006373295 | -0.007778459 | -0.00727349 | -0.007102276 | -0.00589097 |
| Loss / Value loss | 0.2753668 | 0.2571429 | 0.2955161 | 0.2539217 | 0.2664307 | 0.2298367 |
| Physical/command/moving_fraction | 0.4592139 | 0.4761043 | 0.4541243 | 0.3977437 | 0.4885868 | 0.4906026 |
| Physical/command/negative_speed_fraction | 0.2492989 | 0.2583386 | 0.2463957 | 0.2007475 | 0.2663289 | 0.2705245 |
| Physical/command/positive_speed_fraction | 0.209915 | 0.2177657 | 0.2077287 | 0.1969963 | 0.2222579 | 0.2200781 |
| Physical/command/sitting_fraction | 0.08195882 | 0.08518677 | 0.08187093 | 0.07413574 | 0.09008586 | 0.1042281 |
| Physical/command/walking_fraction | 0.9180412 | 0.9148132 | 0.9181291 | 0.9258643 | 0.9099141 | 0.8957719 |
| Physical/sit/body_velocity_abs | 0.1082917 | 0.119013 | 0.103346 | 0.1019326 | 0.1022856 | 0.1104281 |
| Physical/sit/head_height | 0.2788694 | 0.2878203 | 0.277892 | 0.221324 | 0.3059984 | 0.290302 |
| Physical/sit/left_hip_angle_error_abs | 0.1904183 | 0.2111278 | 0.2041153 | 0.1782602 | 0.2116267 | 0.1934714 |
| Physical/sit/left_knee_angle_error_abs | 0.3387865 | 0.4825629 | 0.3277039 | 0.1935761 | 0.442218 | 0.3390036 |
| Physical/sit/mean_joint_angle_error_abs | 0.2987546 | 0.3494348 | 0.2996781 | 0.1560664 | 0.3451772 | 0.3247804 |
| Physical/sit/mean_joint_velocity_abs | 0.8790481 | 0.9439909 | 0.9050399 | 0.6646835 | 0.9217722 | 0.9621254 |
| Physical/sit/right_hip_angle_error_abs | 0.5171552 | 0.5771071 | 0.5213124 | 0.1608083 | 0.497098 | 0.6098099 |
| Physical/sit/right_knee_angle_error_abs | 0.2689079 | 0.3304097 | 0.2716996 | 0.1906497 | 0.2972021 | 0.2730847 |
| Physical/sit/rod_body_angle_error_abs | 0.4477149 | 0.4588076 | 0.4432118 | 0.1952245 | 0.5875372 | 0.5012858 |
| Physical/sit/rod_body_angle_error_signed | 0.4354298 | 0.4469017 | 0.4285224 | 0.1485792 | 0.5818173 | 0.494611 |
| Physical/sit/rod_body_angular_velocity_abs | 1.062273 | 1.179318 | 1.096802 | 0.9060677 | 0.9584666 | 1.137073 |
| Physical/sit/rotor_rod_angle_error_abs | 0.0295446 | 0.03659389 | 0.03002589 | 0.01787974 | 0.03538089 | 0.03202688 |
| Physical/sit/rotor_rod_angle_error_signed | -0.0279878 | -0.03549358 | -0.0283285 | -0.0141424 | -0.03418469 | -0.03100355 |
| Physical/sit/rotor_rod_angular_velocity_abs | 0.1314507 | 0.1337559 | 0.130997 | 0.09901864 | 0.1325805 | 0.1341913 |
| Physical/sit/torso_height | 0.2253784 | 0.2326865 | 0.2257119 | 0.2119077 | 0.2314077 | 0.2283129 |
| Physical/termination/terminated_rate | 0.002618815 | 0.002402344 | 0.002784017 | 0.005678711 | 0.001959635 | 0.002692871 |
| Physical/termination/timeout_rate | 0.000102946 | 0.0001436361 | 7.446289e-05 | 0 | 0.0002075195 | 0.0002233887 |
| Physical/walk/body_velocity | -0.05244149 | -0.06249436 | -0.05995062 | -0.02576998 | -0.07485332 | -0.08323844 |
| Physical/walk/body_velocity_abs | 0.1497473 | 0.1404522 | 0.121352 | 0.10396 | 0.1683674 | 0.1670619 |
| Physical/walk/head_height | 0.4462362 | 0.4603287 | 0.4557871 | 0.4343941 | 0.4586481 | 0.4693369 |
| Physical/walk/left_foot_height | 0.01543378 | 0.01682013 | 0.0154749 | 0.01604584 | 0.01710376 | 0.01591037 |
| Physical/walk/left_foot_horizontal_speed | 0.2220852 | 0.2136064 | 0.1961493 | 0.1689428 | 0.2506524 | 0.2202301 |
| Physical/walk/left_foot_vertical_velocity | 0.01499962 | 0.02267401 | 0.01641717 | 0.01571629 | 0.02404975 | 0.008429593 |
| Physical/walk/left_knee_height | 0.155988 | 0.1571235 | 0.1581814 | 0.1417871 | 0.1591992 | 0.1580397 |
| Physical/walk/mean_foot_height | 0.01668334 | 0.01650674 | 0.01616945 | 0.01728879 | 0.01656395 | 0.02717388 |
| Physical/walk/mean_foot_horizontal_speed | 0.2304561 | 0.2268386 | 0.204197 | 0.1779638 | 0.2591927 | 0.2843016 |
| Physical/walk/right_foot_height | 0.01793291 | 0.01619335 | 0.016864 | 0.01853174 | 0.01602414 | 0.03843738 |
| Physical/walk/right_foot_horizontal_speed | 0.2388269 | 0.2400708 | 0.2122447 | 0.1869848 | 0.2677331 | 0.348373 |
| Physical/walk/right_foot_vertical_velocity | 0.02474986 | 0.01227041 | 0.01970468 | 0.03291238 | 0.009742439 | 0.01771951 |
| Physical/walk/right_knee_height | 0.1407218 | 0.1416489 | 0.1404893 | 0.1477427 | 0.1421104 | 0.1403034 |
| Physical/walk/rod_body_angle | -0.4294826 | -0.2837409 | -0.325232 | -0.4442022 | -0.3178453 | -0.1508457 |
| Physical/walk/rod_body_angle_abs | 0.4433783 | 0.3028829 | 0.3469521 | 0.4602787 | 0.3364144 | 0.1941572 |
| Physical/walk/rod_body_angular_velocity_abs | 1.175303 | 0.9938512 | 0.9762711 | 1.072897 | 1.085393 | 1.027674 |
| Physical/walk/rotor_rod_angle | -0.03783067 | -0.03789594 | -0.03694741 | -0.02995447 | -0.03837185 | -0.03635652 |
| Physical/walk/rotor_rod_angle_abs | 0.04743596 | 0.04736634 | 0.0477966 | 0.04174224 | 0.04736518 | 0.05137227 |
| Physical/walk/rotor_rod_angular_velocity_abs | 0.1205703 | 0.1081073 | 0.1080934 | 0.1096109 | 0.1222073 | 0.1190011 |
| Physical/walk/speed_error_abs | 0.4292612 | 0.4699544 | 0.4689209 | 0.4427834 | 0.4407198 | 0.4677862 |
| Physical/walk/speed_error_negative_command_abs | 0.4045311 | 0.4312506 | 0.4256311 | 0.4221951 | 0.3926913 | 0.4002688 |
| Physical/walk/speed_error_negative_command_signed | 0.3937373 | 0.4231947 | 0.4156767 | 0.4084622 | 0.3838633 | 0.3904067 |
| Physical/walk/speed_error_positive_command_abs | 0.459139 | 0.5165642 | 0.5216816 | 0.4634495 | 0.4992464 | 0.5517014 |
| Physical/walk/speed_error_positive_command_signed | -0.4569844 | -0.5153785 | -0.5208628 | -0.461615 | -0.4981264 | -0.5499482 |
| Physical/walk/speed_error_signed | 0.002774417 | -0.007556728 | -0.01416442 | -0.02234269 | -0.01954231 | -0.03155652 |
| Physical/walk/target_speed | -0.09311343 | -0.09941734 | -0.08866612 | -0.0046075 | -0.1057887 | -0.1224623 |
| Physical/walk/torso_height | 0.3233669 | 0.323428 | 0.3225063 | 0.315692 | 0.3238943 | 0.3219308 |
| Policy / Standard deviation | 6.077204 | 5.818228 | 6.048956 | 6.169051 | 6.056968 | 5.838134 |
| Reward / Instantaneous reward (max) | 0.07718213 | 0.07863812 | 0.07891513 | 0.07728339 | 0.07771275 | 0.08054182 |
| Reward / Instantaneous reward (mean) | -0.03971517 | -0.03446161 | -0.03847647 | -0.06782172 | -0.03185931 | -0.03710809 |
| Reward / Instantaneous reward (min) | -10.69062 | -10.65897 | -10.70577 | -10.68446 | -10.66013 | -10.65053 |
| Reward / Total reward (max) | 7.889141 | 11.3066 | 10.44006 | -1.948608 | 14.69669 | 18.08277 |
| Reward / Total reward (mean) | -14.38344 | -13.66075 | -13.21224 | -11.80442 | -13.95136 | -12.7991 |
| Reward / Total reward (min) | -51.04638 | -53.14346 | -50.04349 | -33.37269 | -51.70087 | -51.82185 |
| Stats / Algorithm update time (ms) | 247.8443 | 236.2647 | 236.5986 | 161.4787 | 157.1184 | 156.6845 |
| Stats / Env stepping time (ms) | 48.1755 | 44.67104 | 46.09987 | 77.19738 | 67.48548 | 68.01566 |
| Stats / Inference time (ms) | 0.9721326 | 0.9496362 | 0.9542549 | 1.053744 | 1.02103 | 1.040862 |

## Scalar metrics — последняя записанная точка

| Metric | baseline | A | B | C | tri42 | tri43 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Episode / Total timesteps (max) | 1249 | 1249 | 1249 | 457 | 1249 | 1249 |
| Episode / Total timesteps (mean) | 513.6754 | 544.4138 | 378.1208 | 173.2284 | 573.309 | 449.2074 |
| Episode / Total timesteps (min) | 35 | 62 | 38 | 17 | 59 | 8 |
| Learning / Learning rate | 0.00125625 | 0.0010875 | 0.0010875 | 0.002304527 | 0.00170625 | 0.00174375 |
| Loss / Entropy loss | -0.1367575 | -0.1367575 | -0.1367575 | -0.1367575 | -0.1367575 | -0.1367575 |
| Loss / Policy loss | -0.009360508 | -0.007782877 | -0.009286781 | -0.008334106 | -0.008219503 | -0.007152515 |
| Loss / Value loss | 0.2946904 | 0.309852 | 0.3007556 | 0.2448872 | 0.2922483 | 0.2547205 |
| Physical/command/moving_fraction | 0.5140381 | 0.5205078 | 0.4703369 | 0.3840332 | 0.5147705 | 0.5273438 |
| Physical/command/negative_speed_fraction | 0.2772217 | 0.2880859 | 0.2561035 | 0.2020264 | 0.2720947 | 0.2958984 |
| Physical/command/positive_speed_fraction | 0.2368164 | 0.2324219 | 0.2142334 | 0.1820068 | 0.2426758 | 0.2314453 |
| Physical/command/sitting_fraction | 0.09118652 | 0.0892334 | 0.08496094 | 0.08105469 | 0.1025391 | 0.1053467 |
| Physical/command/walking_fraction | 0.9088135 | 0.9107666 | 0.9150391 | 0.9189453 | 0.8974609 | 0.8946533 |
| Physical/sit/body_velocity_abs | 0.132504 | 0.1397367 | 0.09770286 | 0.09850385 | 0.09759252 | 0.1302637 |
| Physical/sit/head_height | 0.2975102 | 0.2957746 | 0.284958 | 0.2249148 | 0.316607 | 0.2875396 |
| Physical/sit/left_hip_angle_error_abs | 0.2298113 | 0.2045122 | 0.2122909 | 0.1419082 | 0.1999248 | 0.1953811 |
| Physical/sit/left_knee_angle_error_abs | 0.3828463 | 0.3702135 | 0.356279 | 0.1980664 | 0.447916 | 0.3165167 |
| Physical/sit/mean_joint_angle_error_abs | 0.3543064 | 0.343287 | 0.3110424 | 0.1489755 | 0.3443531 | 0.3145679 |
| Physical/sit/mean_joint_velocity_abs | 1.073874 | 1.112881 | 0.9811611 | 0.6814227 | 0.9481403 | 1.02206 |
| Physical/sit/right_hip_angle_error_abs | 0.6146131 | 0.6319572 | 0.5076895 | 0.1388589 | 0.4428536 | 0.5940692 |
| Physical/sit/right_knee_angle_error_abs | 0.3336734 | 0.3099993 | 0.298251 | 0.2009189 | 0.295772 | 0.2585761 |
| Physical/sit/rod_body_angle_error_abs | 0.5281295 | 0.5076285 | 0.4572795 | 0.1939811 | 0.6431805 | 0.4916901 |
| Physical/sit/rod_body_angle_error_signed | 0.514332 | 0.4972748 | 0.4476807 | 0.1565547 | 0.6412474 | 0.4806985 |
| Physical/sit/rod_body_angular_velocity_abs | 1.397224 | 1.503847 | 1.181095 | 0.8188145 | 0.9646055 | 1.298386 |
| Physical/sit/rotor_rod_angle_error_abs | 0.03676468 | 0.03541119 | 0.03446452 | 0.02011959 | 0.03647188 | 0.03117434 |
| Physical/sit/rotor_rod_angle_error_signed | -0.03565433 | -0.03520247 | -0.03244448 | -0.0162614 | -0.03560225 | -0.03045388 |
| Physical/sit/rotor_rod_angular_velocity_abs | 0.1594696 | 0.1443439 | 0.1565321 | 0.1126816 | 0.1417469 | 0.1368541 |
| Physical/sit/torso_height | 0.2328419 | 0.232401 | 0.2297209 | 0.2139719 | 0.2327854 | 0.227776 |
| Physical/termination/terminated_rate | 0.002197266 | 0.001098633 | 0.002807617 | 0.007080078 | 0.001708984 | 0.003295898 |
| Physical/termination/timeout_rate | 0 | 0.0002441406 | 0.0001220703 | 0 | 0.0002441406 | 0.0002441406 |
| Physical/walk/body_velocity | -0.04027269 | -0.07074683 | -0.08496298 | -0.03219653 | -0.08316583 | -0.0865641 |
| Physical/walk/body_velocity_abs | 0.2066728 | 0.1861782 | 0.1386176 | 0.1027981 | 0.2148184 | 0.2179293 |
| Physical/walk/head_height | 0.4496551 | 0.4813823 | 0.4552971 | 0.4049192 | 0.4701354 | 0.4749533 |
| Physical/walk/left_foot_height | 0.01482812 | 0.01546456 | 0.01467077 | 0.01531502 | 0.01513088 | 0.01567219 |
| Physical/walk/left_foot_horizontal_speed | 0.2822668 | 0.2487883 | 0.2052092 | 0.1581377 | 0.290964 | 0.2578032 |
| Physical/walk/left_foot_vertical_velocity | 0.01487626 | 0.01496573 | 0.01144401 | 0.00889842 | 0.01316382 | 0.007587707 |
| Physical/walk/left_knee_height | 0.1613676 | 0.163572 | 0.1574312 | 0.1440332 | 0.1620782 | 0.1587352 |
| Physical/walk/mean_foot_height | 0.0162576 | 0.01632042 | 0.01578191 | 0.01751441 | 0.01557591 | 0.04440365 |
| Physical/walk/mean_foot_horizontal_speed | 0.2889769 | 0.2696667 | 0.2246095 | 0.1743379 | 0.309954 | 0.3320628 |
| Physical/walk/right_foot_height | 0.01768709 | 0.01717628 | 0.01689306 | 0.01971379 | 0.01602094 | 0.07313509 |
| Physical/walk/right_foot_horizontal_speed | 0.295687 | 0.2905452 | 0.2440098 | 0.1905381 | 0.3289439 | 0.4063223 |
| Physical/walk/right_foot_vertical_velocity | 0.02197833 | 0.01643972 | 0.01734891 | 0.03881586 | 0.01066984 | 0.01590944 |
| Physical/walk/right_knee_height | 0.1407775 | 0.1465586 | 0.1421475 | 0.1531143 | 0.1494974 | 0.1489673 |
| Physical/walk/rod_body_angle | -0.4583836 | -0.1673996 | -0.3471551 | -0.8202152 | -0.312775 | -0.09414007 |
| Physical/walk/rod_body_angle_abs | 0.4724646 | 0.1898994 | 0.3613909 | 0.8262725 | 0.3227081 | 0.1385806 |
| Physical/walk/rod_body_angular_velocity_abs | 1.409904 | 1.089495 | 0.9448427 | 1.119541 | 1.233522 | 1.072285 |
| Physical/walk/rotor_rod_angle | -0.0463383 | -0.0479496 | -0.03764432 | -0.04572624 | -0.04836927 | -0.03732968 |
| Physical/walk/rotor_rod_angle_abs | 0.05310333 | 0.05484421 | 0.04653579 | 0.05521159 | 0.05472641 | 0.04881372 |
| Physical/walk/rotor_rod_angular_velocity_abs | 0.1367328 | 0.1157804 | 0.1091273 | 0.1021613 | 0.129034 | 0.1302934 |
| Physical/walk/speed_error_abs | 0.392429 | 0.4393337 | 0.4317701 | 0.3994552 | 0.3915186 | 0.429579 |
| Physical/walk/speed_error_negative_command_abs | 0.388892 | 0.4002607 | 0.3569942 | 0.3747742 | 0.3447107 | 0.3605882 |
| Physical/walk/speed_error_negative_command_signed | 0.379519 | 0.3901195 | 0.3427614 | 0.3605209 | 0.3347431 | 0.3486378 |
| Physical/walk/speed_error_positive_command_abs | 0.3968349 | 0.4877799 | 0.5210476 | 0.4267139 | 0.4440151 | 0.517774 |
| Physical/walk/speed_error_positive_command_signed | -0.3888952 | -0.4865631 | -0.5206622 | -0.4242905 | -0.4408615 | -0.510945 |
| Physical/walk/speed_error_signed | 0.02563893 | -0.001080059 | -0.05053692 | -0.01129656 | -0.03069923 | -0.028667 |
| Physical/walk/target_speed | -0.09165226 | -0.1249077 | -0.1000463 | -0.0287045 | -0.09763545 | -0.1476877 |
| Physical/walk/torso_height | 0.3316545 | 0.3332216 | 0.3231861 | 0.3310558 | 0.333629 | 0.3228818 |
| Policy / Standard deviation | 7.389055 | 7.389055 | 7.389055 | 7.389054 | 7.389054 | 7.389054 |
| Reward / Instantaneous reward (max) | 0.0751365 | 0.08242644 | 0.07742967 | 0.07426473 | 0.07901549 | 0.08148284 |
| Reward / Instantaneous reward (mean) | -0.03201417 | -0.02100124 | -0.03622917 | -0.07499146 | -0.02192705 | -0.02999638 |
| Reward / Instantaneous reward (min) | -10.63373 | -10.63265 | -10.65052 | -10.62627 | -10.71377 | -10.77658 |
| Reward / Total reward (max) | 10.98437 | 19.98696 | 13.0667 | -3.467421 | 22.70431 | 24.21885 |
| Reward / Total reward (mean) | -16.29679 | -11.8317 | -13.51376 | -12.62026 | -13.29484 | -13.1461 |
| Reward / Total reward (min) | -51.28516 | -56.63689 | -43.00338 | -23.08059 | -47.16275 | -56.39758 |
| Stats / Algorithm update time (ms) | 261.5387 | 235.681 | 238.9646 | 150.7534 | 154.3087 | 124.2973 |
| Stats / Env stepping time (ms) | 49.55767 | 42.76515 | 46.97508 | 71.86167 | 67.8424 | 43.03685 |
| Stats / Inference time (ms) | 0.9793669 | 0.9222761 | 0.9242915 | 1.013836 | 0.9826295 | 0.937295 |

## Histogram metrics — среднее по 32 histogram-точкам и последняя histogram-точка

| Metric | baseline mean | A mean | B mean | C mean | tri42 mean | tri43 mean | baseline last | A last | B last | C last | tri42 last | tri43 last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PhysicalHistogram/action/clipped/left_hip | 0.05556882 | 0.04681089 | 0.04682909 | 0.08002718 | 0.05402383 | 0.05176202 | 0.07382897 | 0.01712513 | 0.06840643 | 0.1143809 | 0.07918869 | 0.06096639 |
| PhysicalHistogram/action/clipped/left_knee | -0.0871802 | -0.1204492 | -0.07298413 | -0.1882565 | -0.088573 | -0.04780397 | -0.05035489 | -0.09266407 | -0.03651517 | -0.1776265 | -0.1030532 | -0.07995263 |
| PhysicalHistogram/action/clipped/right_hip | -0.0135591 | -0.02096638 | -0.02448742 | -0.06197515 | -0.02564006 | -0.01663422 | 0.02875011 | -0.02989626 | -0.02515002 | -0.08271193 | -0.0631687 | -0.006141247 |
| PhysicalHistogram/action/clipped/right_knee | 0.1255181 | 0.1707258 | 0.1706512 | 0.1675904 | 0.1392278 | 0.09460851 | 0.1702949 | 0.2527108 | 0.1706564 | 0.06200112 | 0.1776792 | 0.05466305 |
| PhysicalHistogram/action/raw/left_hip | 0.6056154 | 0.6995945 | 0.6145133 | 1.031086 | 0.6391425 | 0.6891952 | 0.6964669 | 0.5040388 | 0.6200605 | 1.04955 | 1.310872 | 1.054934 |
| PhysicalHistogram/action/raw/left_knee | -1.238426 | -1.265728 | -0.9627644 | -2.324585 | -0.9407853 | -0.6433621 | -1.63094 | -1.723506 | -1.016602 | -2.600697 | -1.613575 | -1.086509 |
| PhysicalHistogram/action/raw/right_hip | 0.007774236 | -0.2403866 | -0.4282807 | -0.51667 | -0.284606 | -0.08442944 | 0.732236 | -0.2728891 | -0.4518485 | -0.6734216 | -0.672327 | 0.02960827 |
| PhysicalHistogram/action/raw/right_knee | 1.468014 | 2.180835 | 2.193739 | 2.14963 | 1.819486 | 0.9710783 | 2.584387 | 4.03667 | 2.67584 | 1.10062 | 2.626328 | 1.057892 |
| PhysicalHistogram/action/scaled_delta/left_hip | 0.005556882 | 0.004681089 | 0.004682909 | 0.008002719 | 0.005402383 | 0.005176202 | 0.007382897 | 0.001712513 | 0.006840643 | 0.01143809 | 0.007918869 | 0.006096639 |
| PhysicalHistogram/action/scaled_delta/left_knee | -0.00871802 | -0.01204492 | -0.007298413 | -0.01882565 | -0.0088573 | -0.004780397 | -0.005035489 | -0.009266407 | -0.003651518 | -0.01776265 | -0.01030532 | -0.007995263 |
| PhysicalHistogram/action/scaled_delta/right_hip | -0.00135591 | -0.002096638 | -0.002448742 | -0.006197515 | -0.002564006 | -0.001663422 | 0.002875011 | -0.002989626 | -0.002515003 | -0.008271194 | -0.00631687 | -0.0006141247 |
| PhysicalHistogram/action/scaled_delta/right_knee | 0.01255181 | 0.01707258 | 0.01706512 | 0.01675904 | 0.01392278 | 0.009460851 | 0.01702949 | 0.02527108 | 0.01706564 | 0.006200112 | 0.01776793 | 0.005466305 |
| PhysicalHistogram/state/unnoisy_joint/left_hip | -1.144209 | -1.304713 | -1.238115 | -1.408906 | -1.204955 | -1.261159 | -1.032522 | -1.362468 | -1.188663 | -1.093931 | -1.201579 | -1.308731 |
| PhysicalHistogram/state/unnoisy_joint/left_knee | 0.704994 | 0.7246654 | 0.7225317 | 0.6474997 | 0.7476129 | 0.9033674 | 0.6496812 | 0.7259908 | 0.7668832 | 0.5799703 | 0.6964394 | 0.9356723 |
| PhysicalHistogram/state/unnoisy_joint/right_hip | 1.490208 | 1.651619 | 1.627262 | 1.371209 | 1.609895 | 1.906791 | 1.557336 | 1.899983 | 1.532829 | 0.9698602 | 1.599381 | 2.009121 |
| PhysicalHistogram/state/unnoisy_joint/right_knee | -0.7172307 | -0.6818203 | -0.6601265 | -0.6494676 | -0.7156014 | -0.8074359 | -0.6449926 | -0.5270443 | -0.6718108 | -0.6980213 | -0.6823 | -0.9577314 |
| PhysicalHistogram/target/absolute/left_hip | -1.190897 | -1.339973 | -1.281076 | -1.427122 | -1.2477 | -1.297485 | -1.095922 | -1.403492 | -1.22447 | -1.11683 | -1.241589 | -1.340256 |
| PhysicalHistogram/target/absolute/left_knee | 0.6027715 | 0.5847768 | 0.6196951 | 0.4955101 | 0.6168797 | 0.639752 | 0.5939943 | 0.5735804 | 0.6433527 | 0.4801949 | 0.59401 | 0.657031 |
| PhysicalHistogram/target/absolute/right_hip | 1.486321 | 1.643984 | 1.619412 | 1.387391 | 1.598612 | 1.889234 | 1.548774 | 1.881271 | 1.533304 | 0.9972117 | 1.583886 | 1.995695 |
| PhysicalHistogram/target/absolute/right_knee | -0.5040495 | -0.4796924 | -0.4508466 | -0.5006179 | -0.4897565 | -0.6445345 | -0.4232882 | -0.3421215 | -0.4870271 | -0.5785787 | -0.4730851 | -0.790482 |
| PhysicalHistogram/target/error_to_unnoisy_joint/left_hip | -0.04668828 | -0.0352603 | -0.04296072 | -0.01821592 | -0.0427445 | -0.03632625 | -0.06340008 | -0.04102372 | -0.03580708 | -0.02289865 | -0.04000966 | -0.03152489 |
| PhysicalHistogram/target/error_to_unnoisy_joint/left_knee | -0.1022224 | -0.1398886 | -0.1028366 | -0.1519896 | -0.1307332 | -0.2636154 | -0.05568686 | -0.1524104 | -0.1235304 | -0.09977538 | -0.1024294 | -0.2786413 |
| PhysicalHistogram/target/error_to_unnoisy_joint/right_hip | -0.0038875 | -0.007634908 | -0.007849732 | 0.01618123 | -0.01128214 | -0.01755766 | -0.008562186 | -0.0187123 | 0.0004749201 | 0.02735148 | -0.01549412 | -0.01342662 |
| PhysicalHistogram/target/error_to_unnoisy_joint/right_knee | 0.2131811 | 0.2021278 | 0.20928 | 0.1488497 | 0.2258449 | 0.1629014 | 0.2217044 | 0.1849227 | 0.1847837 | 0.1194426 | 0.2092149 | 0.1672494 |

## Ограничения интерпретации

- Средние в таблицах равновесно усредняют точки логгера, а не отдельные environment samples; это одинаковое правило для всех run.
- Сравнение A/B/C с baseline использует один общий seed 42; tri-factor имеет дополнительную проверку seed 43, но отдельные A/B/C повторным seed пока не прогонялись.
- Параллельные запуски A/C дали рабочий VRAM режим на 8 GB, но увеличили wall-clock; это не следует трактовать как изменение качества политики.
