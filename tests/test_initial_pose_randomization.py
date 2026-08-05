# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


_SAMPLER_PATH = (
    Path(__file__).parents[1]
    / "source"
    / "CBRIIsaacLab"
    / "CBRIIsaacLab"
    / "tasks"
    / "direct"
    / "cbriisaaclab"
    / "initial_pose_randomization.py"
)
_SAMPLER_SPEC = importlib.util.spec_from_file_location("cbr_i_initial_pose_randomization", _SAMPLER_PATH)
assert _SAMPLER_SPEC is not None and _SAMPLER_SPEC.loader is not None
_SAMPLER = importlib.util.module_from_spec(_SAMPLER_SPEC)
sys.modules[_SAMPLER_SPEC.name] = _SAMPLER
_SAMPLER_SPEC.loader.exec_module(_SAMPLER)

InitialPoseIndices = _SAMPLER.InitialPoseIndices
_compute_collision_lower_z = _SAMPLER._compute_collision_lower_z
sample_initial_commands = _SAMPLER.sample_initial_commands
sample_randomized_joint_positions = _SAMPLER.sample_randomized_joint_positions


def _make_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        initial_sitting_fraction=0.30,
        initial_walking_fraction=0.50,
        initial_walking_speed_range=(0.25, 1.5),
        initial_body_tilt_range=(-math.pi / 3, math.pi / 3),
        initial_hip_delta=math.pi / 4,
        initial_knee_delta=math.radians(35.0),
        initial_rotor_rod_search_start=math.radians(1.0),
        command_info_cfg={"sit_min": 50.0},
        default_standing_state_a={
            "rotor_rod": math.radians(1.0),
            "rod_body": math.radians(-17.0),
            "body_right_hip": math.radians(22.0),
            "body_left_hip": math.radians(-138.0),
            "right_hip_shin": math.radians(-80.0),
            "left_hip_shin": math.radians(45.0),
        },
        default_standing_state_b={
            "rotor_rod": math.radians(1.0),
            "rod_body": math.radians(-17.0),
            "body_right_hip": math.radians(138.0),
            "body_left_hip": math.radians(-22.0),
            "right_hip_shin": math.radians(-45.0),
            "left_hip_shin": math.radians(80.0),
        },
    )


INDICES = InitialPoseIndices(
    rotor_rod=0,
    rod_body=1,
    body_right_hip=2,
    body_left_hip=3,
    right_hip_shin=4,
    left_hip_shin=5,
    left_shin_body=6,
    right_shin_body=7,
)


def test_command_distribution_and_stand_mode() -> None:
    cfg = _make_cfg()
    generator = torch.Generator().manual_seed(123)
    commands = sample_initial_commands(20_000, cfg, "cpu", generator=generator)

    sitting = commands[:, 0] == 1
    walking = (~sitting) & (commands[:, 4] != 0)
    standing = (~sitting) & (commands[:, 4] == 0)
    assert 0.27 < float(sitting.float().mean()) < 0.33
    assert 0.45 < float((walking.float().mean()) / (~sitting).float().mean()) < 0.55
    assert standing.any()
    assert (commands[walking, 4] > 0).any()
    assert (commands[walking, 4] < 0).any()
    assert torch.all(commands[sitting, 1] == cfg.command_info_cfg["sit_min"] * 0.5)

    stand_commands = sample_initial_commands(16, cfg, "cpu", mode="stand")
    assert torch.equal(stand_commands, torch.zeros_like(stand_commands))


def test_pose_sampling_is_seeded_broad_and_within_soft_limits() -> None:
    cfg = _make_cfg()
    num_envs = 4096
    default_pos = torch.zeros((num_envs, 8))
    limits = torch.tensor([[-4.0, 4.0]] * 8).unsqueeze(0).expand(num_envs, -1, -1).clone()

    first = sample_randomized_joint_positions(
        default_pos, limits, cfg, INDICES, generator=torch.Generator().manual_seed(7)
    )
    second = sample_randomized_joint_positions(
        default_pos, limits, cfg, INDICES, generator=torch.Generator().manual_seed(7)
    )
    different = sample_randomized_joint_positions(
        default_pos, limits, cfg, INDICES, generator=torch.Generator().manual_seed(8)
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert torch.all(first >= limits[..., 0])
    assert torch.all(first <= limits[..., 1])
    assert torch.all(first[:, INDICES.rod_body] >= -math.pi / 3)
    assert torch.all(first[:, INDICES.rod_body] <= math.pi / 3)
    assert torch.unique(first, dim=0).shape[0] > num_envs * 0.99


def test_leg_pair_modes_include_same_opposite_and_independent() -> None:
    cfg = _make_cfg()
    num_envs = 12_000
    default_pos = torch.zeros((num_envs, 8))
    limits = torch.tensor([[-4.0, 4.0]] * 8).unsqueeze(0).expand(num_envs, -1, -1).clone()
    sampled = sample_randomized_joint_positions(
        default_pos, limits, cfg, INDICES, generator=torch.Generator().manual_seed(99)
    )

    # Template A/B ranges are disjoint for both hip joints, so the template
    # can be recovered from the right-hip value before checking deltas.
    a_template = cfg.default_standing_state_a
    b_template = cfg.default_standing_state_b
    a_mask = sampled[:, INDICES.body_right_hip] < math.radians(80.0)
    right_hip_delta = sampled[:, INDICES.body_right_hip] - torch.where(
        a_mask,
        torch.tensor(a_template["body_right_hip"]),
        torch.tensor(b_template["body_right_hip"]),
    )
    left_hip_delta = sampled[:, INDICES.body_left_hip] - torch.where(
        a_mask,
        torch.tensor(a_template["body_left_hip"]),
        torch.tensor(b_template["body_left_hip"]),
    )
    right_knee_delta = sampled[:, INDICES.right_hip_shin] - torch.where(
        a_mask,
        torch.tensor(a_template["right_hip_shin"]),
        torch.tensor(b_template["right_hip_shin"]),
    )
    left_knee_delta = sampled[:, INDICES.left_hip_shin] - torch.where(
        a_mask,
        torch.tensor(a_template["left_hip_shin"]),
        torch.tensor(b_template["left_hip_shin"]),
    )

    hip_same = torch.isclose(right_hip_delta, left_hip_delta, atol=1.0e-6)
    hip_opposite = torch.isclose(right_hip_delta, -left_hip_delta, atol=1.0e-6)
    knee_same = torch.isclose(right_knee_delta, left_knee_delta, atol=1.0e-6)
    knee_opposite = torch.isclose(right_knee_delta, -left_knee_delta, atol=1.0e-6)
    assert hip_same.any() and hip_opposite.any() and (~(hip_same | hip_opposite)).any()
    assert knee_same.any() and knee_opposite.any() and (~(knee_same | knee_opposite)).any()


def test_collision_envelope_supports_batched_fk_poses() -> None:
    poses = torch.zeros((16, 8, 7))
    poses[..., 6] = 1.0
    poses[:, :, :3] = torch.rand((16, 8, 3))
    lower_z = _compute_collision_lower_z(poses)
    assert lower_z.shape == (16,)
    assert torch.isfinite(lower_z).all()
