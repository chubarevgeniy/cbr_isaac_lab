"""NumPy-only inference runtime for an exported CBR-I policy.

The exporter writes a ``.npz`` file that contains the MLP weights and the
observation/action parameters used by the Isaac Lab environment.  This module
intentionally has no PyTorch dependency, so it can be copied into a ROS 2
Python package and used by a node running on the robot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _validate_last_dimension(value: Any, expected: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != expected:
        raise ValueError(
            f"{name} must have shape (..., {expected}), got {array.shape}"
        )
    return array


def _scalar_from_archive(archive: Any, name: str, default: float) -> float:
    if name not in archive.files:
        return float(default)
    return float(np.asarray(archive[name]).reshape(()))


def _parameter_vector(value: Any, size: int, name: str) -> np.ndarray:
    """Convert a scalar or a per-action parameter to a float32 vector."""

    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        result = np.full((size,), float(array), dtype=np.float32)
    elif array.shape == (size,):
        result = array.copy()
    else:
        raise ValueError(f"{name} must be a scalar or shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


class ActionFilter:
    """Causal smoothing and rate limiting for actions sent to a robot.

    The filter is intentionally kept outside :class:`NumpyPolicy`: the policy
    still produces its raw action, while the filter owns the command that is
    actually sent to the position controller and fed back as ``last_action``.

    The normal update is an exponential moving average (EMA):

    ``filtered[t] = filtered[t-1] + alpha * (target[t] - filtered[t-1])``

    An optional one-step extrapolation can be enabled with ``prediction_gain``.
    A hard ``max_delta`` is applied after smoothing, so the outgoing command
    cannot change by more than the configured amount in one control cycle.
    ``max_delta`` is expressed in policy-action units, not radians.
    """

    def __init__(
        self,
        action_size: int,
        *,
        alpha: float = 0.15,
        prediction_gain: float = 0.0,
        max_delta: Any | None = None,
        output_min: Any | None = None,
        output_max: Any | None = None,
    ) -> None:
        if int(action_size) <= 0:
            raise ValueError(f"action_size must be positive, got {action_size}")
        self.action_size = int(action_size)
        self.alpha = float(alpha)
        self.prediction_gain = float(prediction_gain)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if self.prediction_gain < 0.0:
            raise ValueError(f"prediction_gain must be non-negative, got {prediction_gain}")

        self.max_delta = (
            None
            if max_delta is None
            else _parameter_vector(max_delta, self.action_size, "max_delta")
        )
        if self.max_delta is not None and np.any(self.max_delta < 0.0):
            raise ValueError("max_delta must be non-negative")

        self.output_min = (
            None
            if output_min is None
            else _parameter_vector(output_min, self.action_size, "output_min")
        )
        self.output_max = (
            None
            if output_max is None
            else _parameter_vector(output_max, self.action_size, "output_max")
        )
        if self.output_min is not None and self.output_max is not None:
            if np.any(self.output_min > self.output_max):
                raise ValueError("output_min must not be greater than output_max")

        self._filtered_action = np.zeros((self.action_size,), dtype=np.float32)
        self._previous_input = np.zeros((self.action_size,), dtype=np.float32)
        self._has_previous_input = False

    @property
    def filtered_action(self) -> np.ndarray:
        """Return a copy of the command currently held by the filter."""

        return self._filtered_action.copy()

    def reset(self, value: Any | None = None) -> np.ndarray:
        """Reset the filter and return the command it now holds.

        With no value, the filter starts at the all-zero action and ramps to
        the first policy command. Passing the currently commanded action is
        useful when restarting a control loop without introducing a jump.
        """

        if value is None:
            state = np.zeros((self.action_size,), dtype=np.float32)
            self._has_previous_input = False
        else:
            state = _validate_last_dimension(value, self.action_size, "filter reset value")
            if state.ndim != 1:
                raise ValueError(
                    "filter reset value must describe one action, "
                    f"got shape {state.shape}"
                )
            state = state.astype(np.float32, copy=True)
            self._has_previous_input = True
        if not np.all(np.isfinite(state)):
            raise ValueError("filter reset value must contain only finite values")

        if self.output_min is not None:
            state = np.maximum(state, self.output_min)
        if self.output_max is not None:
            state = np.minimum(state, self.output_max)
        self._filtered_action = state.copy()
        self._previous_input = state.copy()
        return self.filtered_action

    def update(self, action: Any) -> np.ndarray:
        """Filter one policy action and return the command to send."""

        current = _validate_last_dimension(action, self.action_size, "action")
        if current.ndim != 1:
            raise ValueError(
                "ActionFilter.update expects one action, "
                f"got shape {current.shape}"
            )
        current = current.astype(np.float32, copy=True)
        if not np.all(np.isfinite(current)):
            raise ValueError("action must contain only finite values")

        if self.output_min is not None:
            current = np.maximum(current, self.output_min)
        if self.output_max is not None:
            current = np.minimum(current, self.output_max)

        if self._has_previous_input:
            input_delta = current - self._previous_input
        else:
            # Do not amplify the first command when prediction is enabled.
            input_delta = np.zeros_like(current)
        predicted = current + self.prediction_gain * input_delta

        candidate = self._filtered_action + self.alpha * (
            predicted - self._filtered_action
        )
        if self.output_min is not None:
            candidate = np.maximum(candidate, self.output_min)
        if self.output_max is not None:
            candidate = np.minimum(candidate, self.output_max)

        if self.max_delta is not None:
            change = np.clip(
                candidate - self._filtered_action,
                -self.max_delta,
                self.max_delta,
            )
            candidate = self._filtered_action + change

        self._previous_input = current
        self._filtered_action = candidate.astype(np.float32, copy=False)
        self._has_previous_input = True
        return self.filtered_action

    step = update


class SecondOrderActionFilter:
    """Acceleration-limited command filter with a finite target velocity.

    This is useful when the first command step must be small, but a persistent
    target change should eventually move faster than a fixed per-cycle slew
    limit. The filter keeps position and velocity state, accelerates toward a
    desired velocity, and integrates the result with a trapezoidal step.

    ``max_velocity`` and ``max_acceleration`` are expressed in policy-action
    units per second and per second squared. The output is therefore smooth at
    startup while its sustained speed is controlled independently from the
    initial movement.
    """

    def __init__(
        self,
        action_size: int,
        *,
        dt: float = 0.02,
        response_time: float = 0.12,
        max_velocity: Any = 3.0,
        max_acceleration: Any = 20.0,
        output_min: Any | None = None,
        output_max: Any | None = None,
    ) -> None:
        if int(action_size) <= 0:
            raise ValueError(f"action_size must be positive, got {action_size}")
        self.action_size = int(action_size)
        self.dt = float(dt)
        self.response_time = float(response_time)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(f"dt must be positive and finite, got {dt}")
        if not np.isfinite(self.response_time) or self.response_time <= 0.0:
            raise ValueError(
                f"response_time must be positive and finite, got {response_time}"
            )

        self.max_velocity = _parameter_vector(
            max_velocity, self.action_size, "max_velocity"
        )
        self.max_acceleration = _parameter_vector(
            max_acceleration, self.action_size, "max_acceleration"
        )
        if np.any(self.max_velocity <= 0.0):
            raise ValueError("max_velocity must be positive")
        if np.any(self.max_acceleration <= 0.0):
            raise ValueError("max_acceleration must be positive")

        self.output_min = (
            None
            if output_min is None
            else _parameter_vector(output_min, self.action_size, "output_min")
        )
        self.output_max = (
            None
            if output_max is None
            else _parameter_vector(output_max, self.action_size, "output_max")
        )
        if self.output_min is not None and self.output_max is not None:
            if np.any(self.output_min > self.output_max):
                raise ValueError("output_min must not be greater than output_max")

        self._position = np.zeros((self.action_size,), dtype=np.float32)
        self._velocity = np.zeros((self.action_size,), dtype=np.float32)

    @property
    def filtered_action(self) -> np.ndarray:
        """Return a copy of the command currently held by the filter."""

        return self._position.copy()

    @property
    def velocity(self) -> np.ndarray:
        """Return the current filtered-action velocity in action units/s."""

        return self._velocity.copy()

    def reset(self, value: Any | None = None, velocity: Any | None = None) -> np.ndarray:
        """Reset position and velocity, optionally holding an existing command."""

        if value is None:
            position = np.zeros((self.action_size,), dtype=np.float32)
        else:
            position = _validate_last_dimension(value, self.action_size, "filter reset value")
            if position.ndim != 1:
                raise ValueError(
                    "filter reset value must describe one action, "
                    f"got shape {position.shape}"
                )
            position = position.astype(np.float32, copy=True)
        if not np.all(np.isfinite(position)):
            raise ValueError("filter reset value must contain only finite values")

        if velocity is None:
            current_velocity = np.zeros((self.action_size,), dtype=np.float32)
        else:
            current_velocity = _validate_last_dimension(
                velocity, self.action_size, "filter reset velocity"
            )
            if current_velocity.ndim != 1:
                raise ValueError(
                    "filter reset velocity must describe one action, "
                    f"got shape {current_velocity.shape}"
                )
            current_velocity = current_velocity.astype(np.float32, copy=True)
        if not np.all(np.isfinite(current_velocity)):
            raise ValueError("filter reset velocity must contain only finite values")

        if self.output_min is not None:
            position = np.maximum(position, self.output_min)
        if self.output_max is not None:
            position = np.minimum(position, self.output_max)
        self._position = position
        self._velocity = np.clip(
            current_velocity, -self.max_velocity, self.max_velocity
        )
        return self.filtered_action

    def update(self, action: Any) -> np.ndarray:
        """Move toward one policy action with bounded acceleration and speed."""

        target = _validate_last_dimension(action, self.action_size, "action")
        if target.ndim != 1:
            raise ValueError(
                "SecondOrderActionFilter.update expects one action, "
                f"got shape {target.shape}"
            )
        target = target.astype(np.float32, copy=True)
        if not np.all(np.isfinite(target)):
            raise ValueError("action must contain only finite values")
        if self.output_min is not None:
            target = np.maximum(target, self.output_min)
        if self.output_max is not None:
            target = np.minimum(target, self.output_max)

        error = target - self._position
        desired_velocity = np.clip(
            error / self.response_time,
            -self.max_velocity,
            self.max_velocity,
        )
        max_velocity_step = self.max_acceleration * self.dt
        new_velocity = self._velocity + np.clip(
            desired_velocity - self._velocity,
            -max_velocity_step,
            max_velocity_step,
        )
        new_position = self._position + 0.5 * (
            self._velocity + new_velocity
        ) * self.dt

        # Stop exactly at a target instead of allowing a discrete integration
        # step to cross it and create a small oscillation around the target.
        crossed_target = (
            ((error > 0.0) & (new_position > target))
            | ((error < 0.0) & (new_position < target))
        )
        new_position = np.where(crossed_target, target, new_position)
        new_velocity = np.where(crossed_target, 0.0, new_velocity)

        if self.output_min is not None:
            at_min = new_position < self.output_min
            new_position = np.maximum(new_position, self.output_min)
            new_velocity = np.where(at_min & (new_velocity < 0.0), 0.0, new_velocity)
        if self.output_max is not None:
            at_max = new_position > self.output_max
            new_position = np.minimum(new_position, self.output_max)
            new_velocity = np.where(at_max & (new_velocity > 0.0), 0.0, new_velocity)

        self._position = new_position.astype(np.float32, copy=False)
        self._velocity = new_velocity.astype(np.float32, copy=False)
        return self.filtered_action

    step = update


def raw_actuated_to_canonical(raw: Any, hip_down_angle: float) -> np.ndarray:
    """Convert raw [right hip, left hip, right knee, left knee] positions.

    The signs and reference angle match
    ``coordinate_conventions.raw_actuated_to_canonical`` in the simulator.
    """

    raw_array = _validate_last_dimension(raw, 4, "raw actuated positions")
    canonical = raw_array.copy()
    canonical[..., 0] = float(hip_down_angle) - raw_array[..., 0]
    canonical[..., 1] = raw_array[..., 1] + float(hip_down_angle)
    canonical[..., 2] = -raw_array[..., 2]
    canonical[..., 3] = raw_array[..., 3]
    return canonical


def canonical_actuated_to_raw(canonical: Any, hip_down_angle: float) -> np.ndarray:
    """Convert canonical [right hip, left hip, right knee, left knee] positions."""

    canonical_array = _validate_last_dimension(canonical, 4, "canonical actuated positions")
    raw = canonical_array.copy()
    raw[..., 0] = float(hip_down_angle) - canonical_array[..., 0]
    raw[..., 1] = canonical_array[..., 1] - float(hip_down_angle)
    raw[..., 2] = -canonical_array[..., 2]
    raw[..., 3] = canonical_array[..., 3]
    return raw


def build_observation(
    raw_joint_positions: Any,
    raw_joint_velocities: Any,
    command: Any,
    last_action: Any,
    hip_down_angle: float,
    filter_velocity: Any,
    filter_velocity_scale: Any = 3.0,
) -> np.ndarray:
    """Build the 23-element policy observation from raw robot state.

    Raw joint order must be::

        [base_rotor, rotor_rod, rod_body, right_hip, left_hip,
         right_knee, left_knee]

    The returned observation follows the environment order:

        [6 canonical positions, 7 raw velocities, 2 commands,
         4 filtered actions, 4 normalized filter velocities]

    ``command`` is ``[is_sitting, target_speed]`` and ``last_action`` is the
    previous four-dimensional filtered commanded action. ``filter_velocity``
    is the raw internal action-filter velocity in action units per second; it
    is normalized by ``filter_velocity_scale`` before being appended to the
    observation. All angles are radians and all angular velocities are radians
    per second.
    """

    raw_positions = _validate_last_dimension(raw_joint_positions, 7, "raw joint positions")
    raw_velocities = _validate_last_dimension(raw_joint_velocities, 7, "raw joint velocities")
    command_array = _validate_last_dimension(command, 2, "command")
    last_action_array = _validate_last_dimension(last_action, 4, "last_action")
    filter_velocity_array = _validate_last_dimension(
        filter_velocity, 4, "filter_velocity"
    )
    velocity_scale = _parameter_vector(filter_velocity_scale, 4, "filter_velocity_scale")
    if np.any(velocity_scale <= 0.0):
        raise ValueError("filter_velocity_scale must be positive")
    normalized_filter_velocity = filter_velocity_array / velocity_scale

    # The simulator excludes the fixed base rotor from positions, negates the
    # rotor-rod position, and converts the four actuated positions to the
    # canonical bilateral convention.
    canonical_positions = np.concatenate(
        (
            -raw_positions[..., 1:2],
            raw_positions[..., 2:3],
            raw_actuated_to_canonical(raw_positions[..., 3:7], hip_down_angle),
        ),
        axis=-1,
    )
    return np.concatenate(
        (
            canonical_positions,
            raw_velocities,
            command_array,
            last_action_array,
            normalized_filter_velocity,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


class NumpyPolicy:
    """Deterministic policy inference without importing PyTorch.

    The exported Gaussian policy is evaluated at its mean action.  The
    learned ``log_std`` is loaded and exposed for inspection, but stochastic
    sampling is deliberately not performed by this runtime.
    """

    def __init__(self, model_file: str | Path, metadata_file: str | Path | None = None):
        self.model_file = Path(model_file)
        with np.load(self.model_file, allow_pickle=False) as archive:
            format_version = int(_scalar_from_archive(archive, "format_version", 1))
            if format_version != 2:
                raise ValueError(
                    "Unsupported exported policy format version: "
                    f"{format_version}; this runtime requires version 2"
                )

            weight_names = sorted(
                (name for name in archive.files if name.startswith("weight_")),
                key=lambda name: int(name.removeprefix("weight_")),
            )
            if not weight_names:
                raise ValueError(f"No weight_# arrays found in {self.model_file}")

            self.weights = [np.asarray(archive[name], dtype=np.float32) for name in weight_names]
            self.biases = []
            for weight_name in weight_names:
                bias_name = weight_name.replace("weight_", "bias_", 1)
                if bias_name not in archive.files:
                    raise ValueError(f"Missing {bias_name} in {self.model_file}")
                self.biases.append(np.asarray(archive[bias_name], dtype=np.float32))

            self.observation_mean = np.asarray(archive["obs_mean"], dtype=np.float32)
            self.observation_variance = np.asarray(archive["obs_variance"], dtype=np.float32)
            self.observation_epsilon = _scalar_from_archive(archive, "obs_epsilon", 1.0e-8)
            self.observation_clip = _scalar_from_archive(archive, "obs_clip_threshold", 5.0)
            self.action_offset = np.asarray(archive["action_offset"], dtype=np.float32)
            self.action_scale = np.asarray(archive["action_scale"], dtype=np.float32)
            self.action_filter_max_velocity = _parameter_vector(
                archive["action_filter_max_velocity"]
                if "action_filter_max_velocity" in archive.files
                else 3.0,
                self.action_offset.size,
                "action_filter_max_velocity",
            )
            self.action_filter_max_acceleration = _parameter_vector(
                archive["action_filter_max_acceleration"]
                if "action_filter_max_acceleration" in archive.files
                else 20.0,
                self.action_offset.size,
                "action_filter_max_acceleration",
            )
            self.action_filter_output_min = _parameter_vector(
                archive["action_filter_output_min"]
                if "action_filter_output_min" in archive.files
                else -1.0,
                self.action_offset.size,
                "action_filter_output_min",
            )
            self.action_filter_output_max = _parameter_vector(
                archive["action_filter_output_max"]
                if "action_filter_output_max" in archive.files
                else 1.0,
                self.action_offset.size,
                "action_filter_output_max",
            )
            self.action_filter_dt = _scalar_from_archive(archive, "action_filter_dt", 0.02)
            self.action_filter_response_time = _scalar_from_archive(
                archive, "action_filter_response_time", 0.12
            )
            self.canonical_hip_down_angle = _scalar_from_archive(
                archive, "canonical_hip_down_angle", 0.0
            )
            self.log_std = np.asarray(
                archive["log_std"], dtype=np.float32
            ) if "log_std" in archive.files else np.empty((0,), dtype=np.float32)
            self.activation = (
                str(np.asarray(archive["activation"]).reshape(()))
                if "activation" in archive.files
                else "elu"
            )

        if self.activation != "elu":
            raise ValueError(f"Unsupported exported activation: {self.activation!r}")
        if self.observation_mean.ndim != 1 or self.observation_variance.shape != self.observation_mean.shape:
            raise ValueError("Invalid observation scaler arrays in exported policy")
        if self.observation_mean.size != 23:
            raise ValueError(
                "This runtime expects the CBR-I 23-element observation contract; "
                f"got {self.observation_mean.size}"
            )
        if self.weights[0].shape[1] != self.observation_mean.size:
            raise ValueError(
                "The first policy layer does not match the observation scaler: "
                f"{self.weights[0].shape[1]} != {self.observation_mean.size}"
            )
        if self.action_offset.ndim != 1 or self.action_scale.shape != self.action_offset.shape:
            raise ValueError("Invalid action offset/scale arrays in exported policy")
        if self.weights[-1].shape[0] != self.action_offset.size:
            raise ValueError(
                "The policy output does not match the action contract: "
                f"{self.weights[-1].shape[0]} != {self.action_offset.size}"
            )
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != bias.size:
                raise ValueError(f"Invalid layer {index} in {self.model_file}")
            if index and weight.shape[1] != self.weights[index - 1].shape[0]:
                raise ValueError(f"Layer {index} is not connected to the previous layer")

        self.metadata: dict[str, Any] = {}
        metadata_path = Path(metadata_file) if metadata_file is not None else self.model_file.with_suffix(".json")
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as stream:
                self.metadata = json.load(stream)
        self.metadata_file = metadata_path

    @property
    def observation_size(self) -> int:
        return int(self.observation_mean.size)

    @property
    def action_size(self) -> int:
        return int(self.action_offset.size)

    def predict(self, observation: Any) -> np.ndarray:
        """Return the deterministic (mean) action for one or more observations."""

        x = _validate_last_dimension(observation, self.observation_size, "observation")
        single = x.ndim == 1
        if single:
            x = x[None, :]

        # This is the same order as skrl's RunningStandardScaler: cast the
        # stored statistics to float32, add epsilon after sqrt, then clip.
        x = (x - self.observation_mean) / (np.sqrt(self.observation_variance) + self.observation_epsilon)
        x = np.clip(x, -self.observation_clip, self.observation_clip)

        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight.T + bias
            if index < len(self.weights) - 1:
                # ELU(alpha=1), written this way to avoid exp() on positive
                # activations when NumPy evaluates both np.where branches.
                x = np.where(x < 0.0, np.expm1(np.minimum(x, 0.0)), x)

        result = x.astype(np.float32, copy=False)
        return result[0] if single else result

    deterministic_action = predict

    def build_observation(
        self,
        raw_joint_positions: Any,
        raw_joint_velocities: Any,
        command: Any,
        last_action: Any,
        filter_velocity: Any,
    ) -> np.ndarray:
        """Build a 23-value observation using exported filter parameters."""

        return build_observation(
            raw_joint_positions,
            raw_joint_velocities,
            command,
            last_action,
            self.canonical_hip_down_angle,
            filter_velocity,
            self.action_filter_max_velocity,
        )

    def action_to_canonical_target(self, action: Any) -> np.ndarray:
        """Convert a policy action to canonical joint-position targets."""

        action_array = _validate_last_dimension(action, self.action_size, "action")
        return self.action_offset + action_array * self.action_scale

    def raw_actuated_to_action(self, raw_actuated_positions: Any) -> np.ndarray:
        """Convert measured raw actuated positions to normalized action state."""

        raw_array = _validate_last_dimension(
            raw_actuated_positions, self.action_size, "raw actuated positions"
        )
        canonical = raw_actuated_to_canonical(
            raw_array, self.canonical_hip_down_angle
        )
        return (canonical - self.action_offset) / self.action_scale

    def action_to_raw_target(self, action: Any) -> np.ndarray:
        """Convert a policy action to raw USD/robot joint-position targets."""

        return canonical_actuated_to_raw(
            self.action_to_canonical_target(action),
            self.canonical_hip_down_angle,
        )

    def make_action_filter(
        self,
        *,
        alpha: float = 0.15,
        prediction_gain: float = 0.0,
        max_delta: Any | None = None,
        output_min: Any | None = None,
        output_max: Any | None = None,
    ) -> ActionFilter:
        """Create a stateful outgoing-command filter for this policy.

        ``max_delta`` is in action units. To specify a per-joint limit in
        radians, divide it by ``abs(policy.action_scale)`` before passing it.
        """

        return ActionFilter(
            self.action_size,
            alpha=alpha,
            prediction_gain=prediction_gain,
            max_delta=max_delta,
            output_min=output_min,
            output_max=output_max,
        )

    def make_second_order_action_filter(
        self,
        *,
        dt: float | None = None,
        response_time: float | None = None,
        max_velocity: Any | None = None,
        max_acceleration: Any | None = None,
        output_min: Any | None = None,
        output_max: Any | None = None,
    ) -> SecondOrderActionFilter:
        """Create the acceleration-limited filter matching the exported env."""

        return SecondOrderActionFilter(
            self.action_size,
            dt=self.action_filter_dt if dt is None else dt,
            response_time=(
                self.action_filter_response_time
                if response_time is None
                else response_time
            ),
            max_velocity=(
                self.action_filter_max_velocity
                if max_velocity is None
                else max_velocity
            ),
            max_acceleration=(
                self.action_filter_max_acceleration
                if max_acceleration is None
                else max_acceleration
            ),
            output_min=(
                self.action_filter_output_min if output_min is None else output_min
            ),
            output_max=(
                self.action_filter_output_max if output_max is None else output_max
            ),
        )

    def zero_action(self, *, batch_size: int | None = None) -> np.ndarray:
        """Return the zero fallback when no measured reset pose is available."""

        if batch_size is None:
            return np.zeros((self.action_size,), dtype=np.float32)
        return np.zeros((int(batch_size), self.action_size), dtype=np.float32)


__all__ = [
    "ActionFilter",
    "NumpyPolicy",
    "SecondOrderActionFilter",
    "build_observation",
    "canonical_actuated_to_raw",
    "raw_actuated_to_canonical",
]
