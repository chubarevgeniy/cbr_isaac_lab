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
    second_last_action: Any,
    hip_down_angle: float,
) -> np.ndarray:
    """Build the 23-element policy observation from raw robot state.

    Raw joint order must be::

        [base_rotor, rotor_rod, rod_body, right_hip, left_hip,
         right_knee, left_knee]

    The returned observation follows the environment order:

        [6 canonical positions, 7 raw velocities, 2 commands,
         4 last actions, 4 second-last actions]

    ``command`` is ``[is_sitting, target_speed]``. ``last_action`` and
    ``second_last_action`` are the two most recent four-dimensional policy
    actions. All angles are radians and all angular velocities are radians per
    second.
    """

    raw_positions = _validate_last_dimension(raw_joint_positions, 7, "raw joint positions")
    raw_velocities = _validate_last_dimension(raw_joint_velocities, 7, "raw joint velocities")
    command_array = _validate_last_dimension(command, 2, "command")
    last_action_array = _validate_last_dimension(last_action, 4, "last_action")
    second_last_action_array = _validate_last_dimension(
        second_last_action, 4, "second_last_action"
    )

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
            second_last_action_array,
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
        second_last_action: Any,
    ) -> np.ndarray:
        """Build an observation using the hip reference stored in the model."""

        return build_observation(
            raw_joint_positions,
            raw_joint_velocities,
            command,
            last_action,
            second_last_action,
            self.canonical_hip_down_angle,
        )

    def action_to_canonical_target(self, action: Any) -> np.ndarray:
        """Convert a policy action to canonical joint-position targets."""

        action_array = _validate_last_dimension(action, self.action_size, "action")
        return self.action_offset + action_array * self.action_scale

    def action_to_raw_target(self, action: Any) -> np.ndarray:
        """Convert a policy action to raw USD/robot joint-position targets."""

        return canonical_actuated_to_raw(
            self.action_to_canonical_target(action),
            self.canonical_hip_down_angle,
        )

    def zero_action(self, *, batch_size: int | None = None) -> np.ndarray:
        """Return the reset value for either action-history slot."""

        if batch_size is None:
            return np.zeros((self.action_size,), dtype=np.float32)
        return np.zeros((int(batch_size), self.action_size), dtype=np.float32)


__all__ = [
    "NumpyPolicy",
    "build_observation",
    "canonical_actuated_to_raw",
    "raw_actuated_to_canonical",
]
