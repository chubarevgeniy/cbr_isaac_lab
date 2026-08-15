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
| `action = policy.predict(observation)` | Policy action produced by the neural network | Unitless |
| `canonical_target = policy.action_to_canonical_target(action)` | Target in the model's canonical joint convention | Radians |
| `raw_target = policy.action_to_raw_target(action)` | Target in the robot/USD/ROS raw joint convention | Radians |

The current action contract is:

```text
action_offset = [0, 0, 0, 0]
action_scale  = [1.134464, 1.134464, 1.082099, 1.082099] rad
                [65 deg,    65 deg,    62 deg,    62 deg]
```

The canonical target is calculated as:

```text
canonical_target = action_offset + action * action_scale
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
 4 previous actions,
 4 two-steps-previous actions]
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

`is_sitting` is `1` for sitting and `0` for standing/walking. The last eight
observation values are the exact previous and two-steps-previous policy
actions, not target angles. On reset, set both history slots to the action that
would produce the commanded reset pose:

```text
reset_action = (reset_canonical_target - action_offset) / action_scale
```

This keeps the first action difference relative to the pose that was actually
commanded. Use zero history only when the reset target itself is the zero
canonical target.

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

# q_reset is the commanded raw seven-joint reset pose in the documented order.
reset_action = policy.reset_action_from_raw_joint_positions(q_reset)
last_action = reset_action.copy()
second_last_action = reset_action.copy()

# q and qd use the raw joint order documented above.
observation = policy.build_observation(
    raw_joint_positions=q,
    raw_joint_velocities=qd,
    command=[is_sitting, target_speed],
    last_action=last_action,
    second_last_action=second_last_action,
)

action = policy.predict(observation)
raw_target = policy.action_to_raw_target(action)

# Send raw_target to the four robot position controllers in this order:
# [right_hip, left_hip, right_knee, left_knee]
second_last_action = last_action
last_action = action
```

`predict()` also applies the saved `RunningStandardScaler`. Pass it the
un-normalized 23-element environment observation returned by
`build_observation()`, not an already standardized vector.

The simulator was trained with observation noise enabled, but the ROS runtime
should normally use the actual sensor readings and should not add artificial
noise. The exact training noise parameters are recorded in `policy.json`.
