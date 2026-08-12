"""Export a skrl policy to a PyTorch-free NumPy model.

Unlike ``export_policy.py``, this script does not start Isaac Sim or create an
Isaac Lab environment.  It reads the policy and observation preprocessor
state directly from a skrl checkpoint and writes:

* ``policy.npz`` -- MLP weights, running observation statistics and action
  conversion parameters;
* ``policy.json`` -- the human-readable ROS inference contract, including
  observation order/signs and the configured default standing poses.

The script still needs PyTorch because it reads a PyTorch checkpoint.  The
generated NumPy runtime in ``numpy_policy.py`` has no PyTorch dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


DEFAULT_ENV_VALUES: dict[str, Any] = {
    "observation_space": 23,
    "action_space": 4,
    "base_rotor_dof_name": "Rock_Revolute_1",
    "rotor_rod_dof_name": "bottom_rotor_Revolute_2",
    "rod_body_dof_name": "rod_1_Revolute_3",
    "body_right_hip_dof_name": "body_Revolute_4",
    "body_left_hip_dof_name": "body_Revolute_5",
    "right_hip_shin_dof_name": "right_hip_Revolute_6",
    "left_hip_shin_dof_name": "left_hip_Revolute_7",
    "canonical_hip_down_angle": 130.0 * math.pi / 180.0,
    "canonical_hip_min": -(196.0 - 130.0) * math.pi / 180.0,
    "canonical_hip_max": 130.0 * math.pi / 180.0,
    "canonical_knee_min": 0.0,
    "canonical_knee_max": 124.0 * math.pi / 180.0 - 1.0e-5,
    "action_default_target": (0.0, 0.0, 0.0, 0.0),
    "action_hip_scale": 0.5 * 130.0 * math.pi / 180.0,
    "action_knee_scale": 0.5 * (124.0 * math.pi / 180.0 - 1.0e-5),
    "action_filter_dt": 0.02,
    "action_filter_response_time": 0.12,
    "action_filter_max_velocity": (3.0, 3.0, 3.0, 3.0),
    "action_filter_max_acceleration": (20.0, 20.0, 20.0, 20.0),
    "action_filter_output_min": (
        (-(196.0 - 130.0) * math.pi / 180.0) / (0.5 * 130.0 * math.pi / 180.0),
        (-(196.0 - 130.0) * math.pi / 180.0) / (0.5 * 130.0 * math.pi / 180.0),
        0.0,
        0.0,
    ),
    "action_filter_output_max": (
        2.0,
        2.0,
        2.0,
        2.0,
    ),
    "add_noise": True,
    "noise_pos_hip_knee": 0.02,
    "noise_vel_hip_knee": 0.03,
    "noise_height_pos": 0.005,
    "noise_height_vel": 0.03,
    "noise_angle_pos": 0.02,
    "noise_angle_vel": 0.03,
    "noise_vel": 0.05,
    "default_standing_state_a": {
        "rotor_rod": 1.0 * math.pi / 180.0,
        "rod_body": -17.0 * math.pi / 180.0,
        "body_right_hip": 22.0 * math.pi / 180.0,
        "body_left_hip": -138.0 * math.pi / 180.0,
        "right_hip_shin": -80.0 * math.pi / 180.0,
        "left_hip_shin": 45.0 * math.pi / 180.0,
    },
    "default_standing_state_b": {
        "rotor_rod": 1.0 * math.pi / 180.0,
        "rod_body": -17.0 * math.pi / 180.0,
        "body_right_hip": 138.0 * math.pi / 180.0,
        "body_left_hip": -22.0 * math.pi / 180.0,
        "right_hip_shin": -45.0 * math.pi / 180.0,
        "left_hip_shin": 80.0 * math.pi / 180.0,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a skrl checkpoint to a PyTorch-free NumPy policy."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a skrl .pt checkpoint.")
    parser.add_argument(
        "--output_file",
        default="policy.npz",
        help="Output NumPy archive name; relative paths are placed next to the checkpoint.",
    )
    parser.add_argument(
        "--metadata_file",
        default=None,
        help="Output JSON metadata name (default: same path as --output_file with .json suffix).",
    )
    parser.add_argument(
        "--env_config",
        default=None,
        help="Path to params/env.yaml (default: infer it from the checkpoint run directory).",
    )
    parser.add_argument(
        "--agent_config",
        default=None,
        help="Path to params/agent.yaml (default: infer it from the checkpoint run directory).",
    )
    parser.add_argument(
        "--obs_epsilon",
        type=float,
        default=None,
        help="RunningStandardScaler epsilon (default: agent.yaml or 1e-8).",
    )
    parser.add_argument(
        "--obs_clip_threshold",
        type=float,
        default=None,
        help="RunningStandardScaler clip threshold (default: agent.yaml or 5.0).",
    )
    return parser.parse_args()


def _resolve_relative(path: str | Path, base: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _infer_params_file(checkpoint: Path, filename: str) -> Path | None:
    run_dir = checkpoint.parent.parent
    candidate = run_dir / "params" / filename
    return candidate if candidate.exists() else None


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    # BaseLoader parses Isaac Lab's Python-specific YAML tags as plain scalar
    # and sequence values.  It is sufficient for the selected configuration
    # fields and does not construct arbitrary Python objects from the file.
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=yaml.BaseLoader)
    return value if isinstance(value, dict) else {}


def _config_value(config: dict[str, Any], name: str) -> Any:
    return config.get(name, DEFAULT_ENV_VALUES.get(name))


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_float_list(value: Any, default: tuple[float, ...], expected: int) -> list[float]:
    if value is None:
        values = list(default)
    else:
        values = list(value) if isinstance(value, (list, tuple)) else [value]
    if len(values) != expected:
        raise ValueError(f"Expected {expected} values, got {len(values)}")
    return [float(item) for item in values]


def _as_float_mapping(value: Any, default: dict[str, float]) -> dict[str, float]:
    source = value if isinstance(value, dict) else default
    return {str(key): float(item) for key, item in source.items()}


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions that do not expose the
        # weights_only keyword.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a module dictionary: {path}")
    return checkpoint


def _tensor_to_numpy(value: Any, *, dtype: np.dtype = np.float32) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype).copy()


def _extract_linear_layers(policy_state: dict[str, Any]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    matches: list[tuple[int, str]] = []
    for key in policy_state:
        match = re.search(r"net_container\.(\d+)\.weight$", key)
        if match:
            matches.append((int(match.group(1)), key))
    if not matches:
        raise ValueError(
            "The checkpoint policy does not contain net_container.#.weight MLP layers"
        )

    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for _, weight_key in sorted(matches):
        bias_key = f"{weight_key[:-len('weight')]}bias"
        if bias_key not in policy_state:
            raise ValueError(f"Missing policy bias for {weight_key}: {bias_key}")
        weight = _tensor_to_numpy(policy_state[weight_key])
        bias = _tensor_to_numpy(policy_state[bias_key])
        if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != bias.size:
            raise ValueError(f"Invalid linear layer shapes for {weight_key}")
        if weights and weight.shape[1] != weights[-1].shape[0]:
            raise ValueError(f"Policy layer {weight_key} is not connected to the previous layer")
        weights.append(weight)
        biases.append(bias)
    return weights, biases


def _extract_observation_scaler(
    preprocessor_state: Any,
    observation_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not isinstance(preprocessor_state, dict):
        return (
            np.zeros((observation_size,), dtype=np.float32),
            np.ones((observation_size,), dtype=np.float32),
            1.0,
        )
    if "running_mean" not in preprocessor_state or "running_variance" not in preprocessor_state:
        raise ValueError("Observation preprocessor is missing running_mean/running_variance")
    mean = _tensor_to_numpy(preprocessor_state["running_mean"])
    variance = _tensor_to_numpy(preprocessor_state["running_variance"])
    if mean.shape != (observation_size,) or variance.shape != (observation_size,):
        raise ValueError(
            "Observation scaler shape does not match the policy input: "
            f"mean={mean.shape}, variance={variance.shape}, input={observation_size}"
        )
    current_count = _tensor_to_numpy(
        preprocessor_state.get("current_count", np.asarray(1.0, dtype=np.float32))
    )
    return mean, variance, float(current_count.reshape(()))


def _scaler_hyperparameters(
    agent_config: dict[str, Any],
    epsilon_override: float | None,
    clip_override: float | None,
) -> tuple[float, float]:
    agent_section = agent_config.get("agent", {})
    if not isinstance(agent_section, dict):
        agent_section = {}
    kwargs = agent_section.get("observation_preprocessor_kwargs")
    kwargs = kwargs if isinstance(kwargs, dict) else {}
    epsilon = float(epsilon_override) if epsilon_override is not None else _as_float(kwargs.get("epsilon"), 1.0e-8)
    clip_threshold = (
        float(clip_override)
        if clip_override is not None
        else _as_float(kwargs.get("clip_threshold"), 5.0)
    )
    if epsilon < 0.0:
        raise ValueError("--obs_epsilon must be non-negative")
    if clip_threshold <= 0.0:
        raise ValueError("--obs_clip_threshold must be positive")
    return epsilon, clip_threshold


def _canonical_standing_state(raw_state: dict[str, float], hip_down_angle: float) -> dict[str, float]:
    return {
        "rotor_rod": -raw_state["rotor_rod"],
        "rod_body": raw_state["rod_body"],
        "right_hip": hip_down_angle - raw_state["body_right_hip"],
        "left_hip": raw_state["body_left_hip"] + hip_down_angle,
        "right_knee": -raw_state["right_hip_shin"],
        "left_knee": raw_state["left_hip_shin"],
    }


def _build_metadata(
    *,
    checkpoint_path: Path,
    output_path: Path,
    env_config_path: Path | None,
    agent_config_path: Path | None,
    env_config: dict[str, Any],
    agent_config: dict[str, Any],
    weights: list[np.ndarray],
    mean: np.ndarray,
    variance: np.ndarray,
    current_count: float,
    epsilon: float,
    clip_threshold: float,
    log_std: np.ndarray,
) -> dict[str, Any]:
    observation_size = int(weights[0].shape[1])
    action_size = int(weights[-1].shape[0])
    if observation_size != 23 or action_size != 4:
        raise ValueError(
            "The ROS contract currently describes the CBR-I 23->4 policy; "
            f"got {observation_size}->{action_size}"
        )

    hip_down_angle = _as_float(
        _config_value(env_config, "canonical_hip_down_angle"),
        DEFAULT_ENV_VALUES["canonical_hip_down_angle"],
    )
    action_offset = _as_float_list(
        _config_value(env_config, "action_default_target"),
        DEFAULT_ENV_VALUES["action_default_target"],
        4,
    )
    hip_scale = _as_float(
        _config_value(env_config, "action_hip_scale"),
        DEFAULT_ENV_VALUES["action_hip_scale"],
    )
    knee_scale = _as_float(
        _config_value(env_config, "action_knee_scale"),
        DEFAULT_ENV_VALUES["action_knee_scale"],
    )
    action_scale = [hip_scale, hip_scale, knee_scale, knee_scale]
    action_filter_max_velocity = _as_float_list(
        _config_value(env_config, "action_filter_max_velocity"),
        DEFAULT_ENV_VALUES["action_filter_max_velocity"],
        4,
    )
    action_filter_max_acceleration = _as_float_list(
        _config_value(env_config, "action_filter_max_acceleration"),
        DEFAULT_ENV_VALUES["action_filter_max_acceleration"],
        4,
    )
    action_filter_response_time = _as_float(
        _config_value(env_config, "action_filter_response_time"),
        DEFAULT_ENV_VALUES["action_filter_response_time"],
    )
    action_filter_dt = _as_float(
        _config_value(env_config, "action_filter_dt"),
        DEFAULT_ENV_VALUES["action_filter_dt"],
    )
    action_filter_output_min = _as_float_list(
        _config_value(env_config, "action_filter_output_min"),
        DEFAULT_ENV_VALUES["action_filter_output_min"],
        4,
    )
    action_filter_output_max = _as_float_list(
        _config_value(env_config, "action_filter_output_max"),
        DEFAULT_ENV_VALUES["action_filter_output_max"],
        4,
    )
    canonical_limits = {
        "min": [
            _as_float(_config_value(env_config, "canonical_hip_min"), DEFAULT_ENV_VALUES["canonical_hip_min"]),
            _as_float(_config_value(env_config, "canonical_hip_min"), DEFAULT_ENV_VALUES["canonical_hip_min"]),
            _as_float(_config_value(env_config, "canonical_knee_min"), DEFAULT_ENV_VALUES["canonical_knee_min"]),
            _as_float(_config_value(env_config, "canonical_knee_min"), DEFAULT_ENV_VALUES["canonical_knee_min"]),
        ],
        "max": [
            _as_float(_config_value(env_config, "canonical_hip_max"), DEFAULT_ENV_VALUES["canonical_hip_max"]),
            _as_float(_config_value(env_config, "canonical_hip_max"), DEFAULT_ENV_VALUES["canonical_hip_max"]),
            _as_float(_config_value(env_config, "canonical_knee_max"), DEFAULT_ENV_VALUES["canonical_knee_max"]),
            _as_float(_config_value(env_config, "canonical_knee_max"), DEFAULT_ENV_VALUES["canonical_knee_max"]),
        ],
    }

    joint_names = {
        "base_rotor": str(_config_value(env_config, "base_rotor_dof_name")),
        "rotor_rod": str(_config_value(env_config, "rotor_rod_dof_name")),
        "rod_body": str(_config_value(env_config, "rod_body_dof_name")),
        "right_hip": str(_config_value(env_config, "body_right_hip_dof_name")),
        "left_hip": str(_config_value(env_config, "body_left_hip_dof_name")),
        "right_knee": str(_config_value(env_config, "right_hip_shin_dof_name")),
        "left_knee": str(_config_value(env_config, "left_hip_shin_dof_name")),
    }
    raw_joint_order = [
        joint_names["base_rotor"],
        joint_names["rotor_rod"],
        joint_names["rod_body"],
        joint_names["right_hip"],
        joint_names["left_hip"],
        joint_names["right_knee"],
        joint_names["left_knee"],
    ]

    default_a = _as_float_mapping(
        _config_value(env_config, "default_standing_state_a"),
        DEFAULT_ENV_VALUES["default_standing_state_a"],
    )
    default_b = _as_float_mapping(
        _config_value(env_config, "default_standing_state_b"),
        DEFAULT_ENV_VALUES["default_standing_state_b"],
    )

    field_layout = [
        {
            "name": "position.rotor_rod",
            "indices": [0],
            "source_joint": joint_names["rotor_rod"],
            "coordinate_system": "canonical",
            "transform": "-raw",
        },
        {
            "name": "position.rod_body",
            "indices": [1],
            "source_joint": joint_names["rod_body"],
            "coordinate_system": "raw",
            "transform": "raw",
        },
        {
            "name": "position.right_hip",
            "indices": [2],
            "source_joint": joint_names["right_hip"],
            "coordinate_system": "canonical",
            "transform": "hip_down_angle - raw",
        },
        {
            "name": "position.left_hip",
            "indices": [3],
            "source_joint": joint_names["left_hip"],
            "coordinate_system": "canonical",
            "transform": "raw + hip_down_angle",
        },
        {
            "name": "position.right_knee",
            "indices": [4],
            "source_joint": joint_names["right_knee"],
            "coordinate_system": "canonical",
            "transform": "-raw",
        },
        {
            "name": "position.left_knee",
            "indices": [5],
            "source_joint": joint_names["left_knee"],
            "coordinate_system": "canonical",
            "transform": "raw",
        },
    ]
    field_layout.extend(
        {
            "name": f"velocity.{name}",
            "indices": [6 + index],
            "source_joint": joint_names[name],
            "coordinate_system": "raw",
            "transform": "raw",
        }
        for index, name in enumerate(
            ["base_rotor", "rotor_rod", "rod_body", "right_hip", "left_hip", "right_knee", "left_knee"]
        )
    )
    field_layout.extend(
        [
            {
                "name": "command.is_sitting",
                "indices": [13],
                "source": "command[:, 0]",
                "meaning": "1=sitting, 0=standing/walking",
            },
            {
                "name": "command.target_speed",
                "indices": [14],
                "source": "command[:, 4]",
                "units": "m/s proxy",
            },
            {
                "name": "last_action.right_hip",
                "indices": [15],
                "source": "previous commanded action[0]",
            },
            {
                "name": "last_action.left_hip",
                "indices": [16],
                "source": "previous commanded action[1]",
            },
            {
                "name": "last_action.right_knee",
                "indices": [17],
                "source": "previous commanded action[2]",
            },
            {
                "name": "last_action.left_knee",
                "indices": [18],
                "source": "previous commanded action[3]",
            },
            {
                "name": "filter_velocity.right_hip",
                "indices": [19],
                "source": "action filter velocity[0] / max_velocity[0]",
                "units": "normalized action velocity",
            },
            {
                "name": "filter_velocity.left_hip",
                "indices": [20],
                "source": "action filter velocity[1] / max_velocity[1]",
                "units": "normalized action velocity",
            },
            {
                "name": "filter_velocity.right_knee",
                "indices": [21],
                "source": "action filter velocity[2] / max_velocity[2]",
                "units": "normalized action velocity",
            },
            {
                "name": "filter_velocity.left_knee",
                "indices": [22],
                "source": "action filter velocity[3] / max_velocity[3]",
                "units": "normalized action velocity",
            },
        ]
    )

    noise_keys = [
        "noise_pos_hip_knee",
        "noise_vel_hip_knee",
        "noise_height_pos",
        "noise_height_vel",
        "noise_angle_pos",
        "noise_angle_vel",
        "noise_vel",
    ]
    noise_parameters = {
        key: _as_float(_config_value(env_config, key), DEFAULT_ENV_VALUES[key]) for key in noise_keys
    }

    hidden_layers = [int(weight.shape[0]) for weight in weights[:-1]]
    agent_section = agent_config.get("agent", {})
    agent_section = agent_section if isinstance(agent_section, dict) else {}
    metadata = {
        "format": "cbri_numpy_policy",
        "format_version": 2,
        "model_file": output_path.name,
        "source_checkpoint": str(checkpoint_path),
        "source_env_config": str(env_config_path) if env_config_path else None,
        "source_agent_config": str(agent_config_path) if agent_config_path else None,
        "policy": {
            "type": "gaussian_mlp",
            "inference": "deterministic_mean_action",
            "activation": "elu",
            "hidden_layers": hidden_layers,
            "output_size": action_size,
            "log_std": log_std.tolist(),
            "action_clip_during_training": False,
            "filtered_action_output_range": {
                "min": action_filter_output_min,
                "max": action_filter_output_max,
            },
            "learning_rate_at_export": agent_section.get("learning_rate"),
        },
        "preprocessing": {
            "type": "running_standard_scaler",
            "mean": mean.tolist(),
            "variance": variance.tolist(),
            "current_count": current_count,
            "epsilon": epsilon,
            "clip_threshold": clip_threshold,
            "formula": "clip((observation - mean) / (sqrt(variance) + epsilon), -clip_threshold, clip_threshold)",
        },
        "observation": {
            "size": observation_size,
            "dtype": "float32",
            "angle_units": "radians",
            "angular_velocity_units": "radians_per_second",
            "raw_joint_order": raw_joint_order,
            "field_layout": field_layout,
            "last_action_reset": "normalized action corresponding to measured raw actuated pose; zeros only if pose is unavailable",
            "filter_velocity_reset": "zeros(4), normalized by action_filter_max_velocity",
            "noise": {
                "enabled_during_training": _as_bool(
                    _config_value(env_config, "add_noise"), DEFAULT_ENV_VALUES["add_noise"]
                ),
                "recommended_for_ros": False,
                "parameters": noise_parameters,
            },
        },
        "action_contract": {
            "size": action_size,
            "order": ["right_hip", "left_hip", "right_knee", "left_knee"],
            "units": "normalized_policy_action",
            "bounds_during_training": "unbounded; clip_actions=False",
            "canonical_hip_down_angle_rad": hip_down_angle,
            "canonical_target_offset_rad": action_offset,
            "canonical_target_scale_rad_per_action": action_scale,
            "canonical_limits_rad": canonical_limits,
            "canonical_target_formula": "canonical_target = offset + filtered_action * scale",
            "raw_target_formula": [
                "raw_right_hip = hip_down_angle - canonical_right_hip",
                "raw_left_hip = canonical_left_hip - hip_down_angle",
                "raw_right_knee = -canonical_right_knee",
                "raw_left_knee = canonical_left_knee",
            ],
            "previous_action_for_observation": "feed the exact previous four-dimensional filtered commanded action; reset to zeros",
            "deployment_action_filter": {
                "implementation": "SecondOrderActionFilter in numpy_policy.py",
                "dt_s": action_filter_dt,
                "response_time_s": action_filter_response_time,
                "max_velocity_action_units_per_s": action_filter_max_velocity,
                "max_acceleration_action_units_per_s2": action_filter_max_acceleration,
                "output_range": [action_filter_output_min, action_filter_output_max],
                "feedback_value": "filtered command sent to the position controller",
            },
        },
        "reset_poses": {
            "joint_values_are_raw_radians": True,
            "source_keys": [
                "rotor_rod",
                "rod_body",
                "body_right_hip",
                "body_left_hip",
                "right_hip_shin",
                "left_hip_shin",
            ],
            "default_standing_state_a_raw": default_a,
            "default_standing_state_b_raw": default_b,
            "default_standing_state_a_canonical": _canonical_standing_state(default_a, hip_down_angle),
            "default_standing_state_b_canonical": _canonical_standing_state(default_b, hip_down_angle),
            "note": "These are the configured standing reference poses; the active reset sampler may randomize around them.",
        },
        "configuration_snapshot": {
            "canonical_hip_down_angle": hip_down_angle,
            "action_default_target": action_offset,
            "action_hip_scale": hip_scale,
            "action_knee_scale": knee_scale,
            "add_noise": _as_bool(
                _config_value(env_config, "add_noise"), DEFAULT_ENV_VALUES["add_noise"]
            ),
        },
    }
    return metadata


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    output_path = _resolve_relative(args.output_file, checkpoint_path.parent)
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")
    metadata_path = (
        _resolve_relative(args.metadata_file, checkpoint_path.parent)
        if args.metadata_file
        else output_path.with_suffix(".json")
    )
    env_config_path = (
        _resolve_relative(args.env_config, Path.cwd())
        if args.env_config
        else _infer_params_file(checkpoint_path, "env.yaml")
    )
    agent_config_path = (
        _resolve_relative(args.agent_config, Path.cwd())
        if args.agent_config
        else _infer_params_file(checkpoint_path, "agent.yaml")
    )
    if args.env_config and not env_config_path.exists():
        raise FileNotFoundError(f"Environment config does not exist: {env_config_path}")
    if args.agent_config and not agent_config_path.exists():
        raise FileNotFoundError(f"Agent config does not exist: {agent_config_path}")

    checkpoint = _load_checkpoint(checkpoint_path)
    policy_state = checkpoint.get("policy")
    if not isinstance(policy_state, dict):
        raise KeyError("Checkpoint does not contain a 'policy' state dictionary")
    weights, biases = _extract_linear_layers(policy_state)
    observation_size = int(weights[0].shape[1])
    mean, variance, current_count = _extract_observation_scaler(
        checkpoint.get("observation_preprocessor"), observation_size
    )
    agent_config = _load_yaml(agent_config_path)
    env_config = _load_yaml(env_config_path)
    epsilon, clip_threshold = _scaler_hyperparameters(
        agent_config, args.obs_epsilon, args.obs_clip_threshold
    )
    log_std = (
        _tensor_to_numpy(policy_state["log_std_parameter"])
        if "log_std_parameter" in policy_state
        else np.empty((0,), dtype=np.float32)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "format_version": np.asarray(2, dtype=np.int64),
        "activation": np.asarray("elu"),
        "obs_mean": mean,
        "obs_variance": variance,
        "obs_current_count": np.asarray(current_count, dtype=np.float64),
        "obs_epsilon": np.asarray(epsilon, dtype=np.float64),
        "obs_clip_threshold": np.asarray(clip_threshold, dtype=np.float64),
        "action_offset": np.asarray(
            _as_float_list(
                _config_value(env_config, "action_default_target"),
                DEFAULT_ENV_VALUES["action_default_target"],
                4,
            ),
            dtype=np.float32,
        ),
        "action_scale": np.asarray(
            [
                _as_float(_config_value(env_config, "action_hip_scale"), DEFAULT_ENV_VALUES["action_hip_scale"]),
                _as_float(_config_value(env_config, "action_hip_scale"), DEFAULT_ENV_VALUES["action_hip_scale"]),
                _as_float(_config_value(env_config, "action_knee_scale"), DEFAULT_ENV_VALUES["action_knee_scale"]),
                _as_float(_config_value(env_config, "action_knee_scale"), DEFAULT_ENV_VALUES["action_knee_scale"]),
            ],
            dtype=np.float32,
        ),
        "action_filter_dt": np.asarray(
            _as_float(
                _config_value(env_config, "action_filter_dt"),
                DEFAULT_ENV_VALUES["action_filter_dt"],
            ),
            dtype=np.float32,
        ),
        "action_filter_response_time": np.asarray(
            _as_float(
                _config_value(env_config, "action_filter_response_time"),
                DEFAULT_ENV_VALUES["action_filter_response_time"],
            ),
            dtype=np.float32,
        ),
        "action_filter_max_velocity": np.asarray(
            _as_float_list(
                _config_value(env_config, "action_filter_max_velocity"),
                DEFAULT_ENV_VALUES["action_filter_max_velocity"],
                4,
            ),
            dtype=np.float32,
        ),
        "action_filter_max_acceleration": np.asarray(
            _as_float_list(
                _config_value(env_config, "action_filter_max_acceleration"),
                DEFAULT_ENV_VALUES["action_filter_max_acceleration"],
                4,
            ),
            dtype=np.float32,
        ),
        "action_filter_output_min": np.asarray(
            _as_float_list(
                _config_value(env_config, "action_filter_output_min"),
                DEFAULT_ENV_VALUES["action_filter_output_min"],
                4,
            ),
            dtype=np.float32,
        ),
        "action_filter_output_max": np.asarray(
            _as_float_list(
                _config_value(env_config, "action_filter_output_max"),
                DEFAULT_ENV_VALUES["action_filter_output_max"],
                4,
            ),
            dtype=np.float32,
        ),
        "canonical_hip_down_angle": np.asarray(
            _as_float(
                _config_value(env_config, "canonical_hip_down_angle"),
                DEFAULT_ENV_VALUES["canonical_hip_down_angle"],
            ),
            dtype=np.float32,
        ),
        "log_std": log_std,
    }
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        arrays[f"weight_{index}"] = weight
        arrays[f"bias_{index}"] = bias
    np.savez_compressed(output_path, **arrays)

    metadata = _build_metadata(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        env_config_path=env_config_path,
        agent_config_path=agent_config_path,
        env_config=env_config,
        agent_config=agent_config,
        weights=weights,
        mean=mean,
        variance=variance,
        current_count=current_count,
        epsilon=epsilon,
        clip_threshold=clip_threshold,
        log_std=log_std,
    )
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(f"Exported NumPy policy: {output_path}")
    print(f"Exported ROS metadata: {metadata_path}")
    print(
        f"Architecture: {weights[0].shape[1]} -> "
        f"{' -> '.join(str(weight.shape[0]) for weight in weights)}; "
        f"observation scaler={'yes' if isinstance(checkpoint.get('observation_preprocessor'), dict) else 'no'}"
    )


if __name__ == "__main__":
    main()
