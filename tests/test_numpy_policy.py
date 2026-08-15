from __future__ import annotations

import json

import numpy as np

from scripts.skrl.numpy_policy import NumpyPolicy, build_observation


def _write_toy_policy(path) -> None:
    rng = np.random.default_rng(7)
    weights = [
        rng.normal(size=(8, 23)).astype(np.float32),
        rng.normal(size=(4, 8)).astype(np.float32),
    ]
    biases = [
        rng.normal(size=(8,)).astype(np.float32),
        rng.normal(size=(4,)).astype(np.float32),
    ]
    np.savez(
        path,
        format_version=np.asarray(2, dtype=np.int64),
        activation=np.asarray("elu"),
        weight_0=weights[0],
        bias_0=biases[0],
        weight_1=weights[1],
        bias_1=biases[1],
        obs_mean=np.linspace(-1.0, 1.0, 23, dtype=np.float32),
        obs_variance=np.linspace(0.5, 1.5, 23, dtype=np.float32),
        obs_epsilon=np.asarray(1.0e-8),
        obs_clip_threshold=np.asarray(5.0),
        action_offset=np.zeros(4, dtype=np.float32),
        action_scale=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        canonical_hip_down_angle=np.asarray(2.0, dtype=np.float32),
        log_std=np.zeros(4, dtype=np.float32),
    )


def test_numpy_policy_matches_manual_mlp(tmp_path) -> None:
    model_path = tmp_path / "policy.npz"
    _write_toy_policy(model_path)
    policy = NumpyPolicy(model_path)

    observations = np.random.default_rng(8).normal(size=(5, 23)).astype(np.float32)
    expected = (observations - policy.observation_mean) / (
        np.sqrt(policy.observation_variance) + policy.observation_epsilon
    )
    expected = np.clip(expected, -5.0, 5.0)
    expected = expected @ policy.weights[0].T + policy.biases[0]
    expected = np.where(expected < 0.0, np.expm1(np.minimum(expected, 0.0)), expected)
    expected = expected @ policy.weights[1].T + policy.biases[1]

    np.testing.assert_allclose(policy.predict(observations), expected, rtol=1.0e-6, atol=1.0e-6)
    assert policy.predict(observations[0]).shape == (4,)


def test_observation_and_action_contract_uses_canonical_signs(tmp_path) -> None:
    model_path = tmp_path / "policy.npz"
    _write_toy_policy(model_path)
    policy = NumpyPolicy(model_path)

    raw_positions = np.asarray([10.0, 0.2, 0.3, 0.4, -0.5, 0.6, -0.7], dtype=np.float32)
    raw_velocities = np.arange(7, dtype=np.float32)
    observation = policy.build_observation(
        raw_positions,
        raw_velocities,
        [1.0, -0.4],
        [0.1] * 4,
        [0.2] * 4,
    )
    expected_positions = [-0.2, 0.3, 1.6, 1.5, -0.6, -0.7]
    np.testing.assert_allclose(observation[:6], expected_positions)
    np.testing.assert_allclose(observation[6:13], raw_velocities)
    np.testing.assert_allclose(observation[13:15], [1.0, -0.4])
    np.testing.assert_allclose(observation[15:19], [0.1] * 4)
    np.testing.assert_allclose(observation[19:], [0.2] * 4)

    action = np.asarray([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    np.testing.assert_allclose(policy.action_to_canonical_target(action), [1.0, -2.0, 1.5, -2.0])
    np.testing.assert_allclose(policy.action_to_raw_target(action), [1.0, -4.0, -1.5, -2.0])


def test_sidecar_metadata_is_loaded(tmp_path) -> None:
    model_path = tmp_path / "policy.npz"
    _write_toy_policy(model_path)
    metadata_path = tmp_path / "policy.json"
    metadata_path.write_text(json.dumps({"format": "test"}), encoding="utf-8")

    assert NumpyPolicy(model_path).metadata == {"format": "test"}
    assert build_observation(
        np.zeros(7), np.zeros(7), np.zeros(2), np.zeros(4), np.zeros(4), 2.0
    ).shape == (23,)
