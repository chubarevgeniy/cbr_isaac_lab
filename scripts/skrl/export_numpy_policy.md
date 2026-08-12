# NumPy policy export and ROS contract

This export is intended for a Python ROS node that must run inference without
importing PyTorch.

The exporter needs PyTorch only while reading the training checkpoint. The
generated runtime, `numpy_policy.py`, depends only on NumPy and the Python
standard library.

## Export

Run this from the Isaac Lab Python environment:

```bash
python scripts/skrl/export_numpy_policy.py \
    --checkpoint=/absolute/path/to/agent_650000.pt \
    --output_file=policy.npz
```

The exporter creates two files next to the checkpoint unless an absolute output
path is supplied:

- `policy.npz` — MLP weights, observation scaler, action offset/scales and the
  canonical hip reference;
- `policy.json` — readable inference contract, including observation layout,
  signs, units, noise settings and both configured `default_standing_state`
  poses in raw and canonical coordinates.

Copy these files and `numpy_policy.py` into the ROS Python package.

## Three coordinate levels

The policy has three different representations. Do not treat the output of
`predict()` as an angle.

| Representation | Meaning | Units |
| --- | --- | --- |
| `action = policy.predict(observation)` | Raw action produced by the neural network | Unitless |
| `command = action_filter.update(action)` | Smoothed, rate-limited action sent to the controller | Unitless |
| `canonical_target = policy.action_to_canonical_target(command)` | Target in the model's canonical joint convention | Radians |
| `raw_target = policy.action_to_raw_target(command)` | Target in the robot/USD/ROS raw joint convention | Radians |

The current action contract is:

```text
action_offset = [0, 0, 0, 0]
action_scale  = [1.134464, 1.134464, 1.082099, 1.082099] rad
                [65 deg,    65 deg,    62 deg,    62 deg]
```

The canonical target is calculated as:

```text
canonical_target = action_offset + filtered_action * action_scale
```

The canonical hip reference is:

```text
hip_down_angle = 2.268928 rad = 130 deg
```

Conversion from canonical targets to raw robot targets is:

```text
raw_right_hip  = hip_down_angle - canonical_right_hip
raw_left_hip   = canonical_left_hip - hip_down_angle
raw_right_knee = -canonical_right_knee
raw_left_knee  = canonical_left_knee
```

Therefore, the zero action means:

```text
canonical target = [0, 0, 0, 0]
raw target       = [130 deg, -130 deg, 0 deg, 0 deg]
```

It is the canonical down/straight pose, not the sitting pose.

## Observation input

The ROS node should provide raw encoder values to `build_observation()`.
Positions are in radians and velocities are in radians per second. Do not
manually subtract `130 deg` or change the hip/knee signs before calling this
function.

Raw joint order must be:

```text
[base_rotor, rotor_rod, rod_body,
 right_hip, left_hip, right_knee, left_knee]
```

The helper converts these values into the 23-element observation used during
training:

```text
[6 canonical positions,
 7 raw angular velocities,
 2 commands,
 4 previous filtered actions,
 4 normalized action-filter velocities]
```

The six position values are:

```text
obs[0] = -raw_rotor_rod
obs[1] =  raw_rod_body
obs[2] =  hip_down_angle - raw_right_hip
obs[3] =  raw_left_hip + hip_down_angle
obs[4] = -raw_right_knee
obs[5] =  raw_left_knee
```

The seven velocity values remain raw and keep the same joint order. The two
command values are:

```text
command = [is_sitting, target_speed]
```

`is_sitting` is `1` for sitting and `0` for standing/walking. Observation
indices `15:19` are the previous filtered commanded action, not target angles;
indices `19:23` are the action-filter velocity divided by its exported
per-joint maximum velocity. On startup or reset, initialize the filter position
from the measured actuated pose when possible and always reset its velocity to
zero. If the current pose is not available, both vectors may start at zero.

## Preventing command jitter

For a real robot, put a causal filter between the policy and the position
controller. `ActionFilter` is the simple EMA version. For the desired behavior
of a small first step followed by faster motion, use
`SecondOrderActionFilter`: it keeps both command position and command velocity,
then limits acceleration and sustained velocity.

The second-order filter provides:

1. a small first step because the command starts with zero velocity;
2. acceleration-limited buildup of command velocity;
3. a maximum sustained command velocity independent of the first step size.

The filter parameters are exported with the model, so the runtime can create
the same trajectory generator without hardcoding a second set of values:

```python
from numpy_policy import NumpyPolicy

policy = NumpyPolicy("policy.npz")
action_filter = policy.make_second_order_action_filter()
# q[3:7] is [right_hip, left_hip, right_knee, left_knee] in raw coordinates.
last_action = action_filter.reset(policy.raw_actuated_to_action(q[3:7]))

observation = policy.build_observation(
    q, qd, [is_sitting, target_speed], last_action,
    filter_velocity=action_filter.velocity,
)
policy_action = policy.predict(observation)
last_action = action_filter.update(policy_action)
raw_target = policy.action_to_raw_target(last_action)
send_to_position_controller(raw_target)
```

At 50 Hz, `max_velocity=3.0` permits up to `0.06` action units per cycle,
which is about 3.9° for a hip and 3.7° for a knee. The filter reaches that
speed only while the target is far enough away; it decelerates near the target.
For a custom filter, per-joint velocity or acceleration limits in radians must
be divided by `abs(policy.action_scale)` before passing them in action units.
Do not feed the unfiltered policy action as `last_action`: the observation must
describe the command and filter velocity that the robot actually received.

The first-order EMA remains available through `make_action_filter()` for cases
where latency matters more than acceleration shaping. Its optional predictor
is deliberately off by default because a derivative term can amplify sensor or
policy noise. The second-order filter also cannot remove mechanical resonance,
backlash, or vibration caused inside the robot's low-level controller.

### Meaning of the canonical angles

- Canonical hip `0` is the thigh-down/standing reference.
- Moving the hip from down toward the belly increases the canonical hip angle.
- If the raw encoder reports hip `0` when the thigh is at the belly, the
  corresponding canonical hip value is approximately `+130 deg`, not zero.
- A straight canonical knee is `0`.
- Knee flexion is positive in canonical coordinates for both legs. In raw
  coordinates the right knee has the opposite sign, while the left knee keeps
  its sign.

## ROS inference example

```python
from numpy_policy import NumpyPolicy

policy = NumpyPolicy("policy.npz")
action_filter = policy.make_second_order_action_filter()

# q and qd use the raw joint order documented above.
last_action = action_filter.reset(policy.raw_actuated_to_action(q[3:7]))
observation = policy.build_observation(
    raw_joint_positions=q,
    raw_joint_velocities=qd,
    command=[is_sitting, target_speed],
    last_action=last_action,
    filter_velocity=action_filter.velocity,
)

policy_action = policy.predict(observation)
last_action = action_filter.update(policy_action)
raw_target = policy.action_to_raw_target(last_action)

# Send raw_target to the four robot position controllers in this order:
# [right_hip, left_hip, right_knee, left_knee]
```

`predict()` also applies the saved `RunningStandardScaler`. Pass it the
un-normalized 23-element environment observation returned by
`build_observation()`, not an already standardized vector.

The simulator was trained with observation noise enabled, but the ROS runtime
should normally use the actual sensor readings and should not add artificial
noise. The exact training noise parameters are recorded in `policy.json`.
