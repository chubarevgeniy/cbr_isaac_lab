# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Sampling and ground checks for randomized CBR-I initial poses.

The functions in this module deliberately do not move the articulation root.  The
height correction is applied through ``bottom_rotor_Revolute_2`` instead.  Keeping
the sampler here makes the training environment, the policy runner and the pose
inspection script use the same initial-state distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class InitialPoseIndices:
    """Joint and body indices needed by the initial-pose sampler."""

    rotor_rod: int
    rod_body: int
    body_right_hip: int
    body_left_hip: int
    right_hip_shin: int
    left_hip_shin: int
    left_shin_body: int
    right_shin_body: int


@dataclass
class GroundPoseDiagnostics:
    """Geometry diagnostics for a batch of articulation poses."""

    left_foot: torch.Tensor
    right_foot: torch.Tensor
    collision_lower_z: torch.Tensor

    @property
    def minimum_foot_z(self) -> torch.Tensor:
        return torch.minimum(self.left_foot[:, 2], self.right_foot[:, 2])


@dataclass
class GroundSafePoseResult:
    """Result returned after sampling and aligning a batch of poses."""

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    diagnostics: GroundPoseDiagnostics
    safe: torch.Tensor


# The collision meshes in CBR-I.usda are authored in millimeters.  These are
# their link-local AABBs after the asset scale and each ``/colliders`` child
# transform (translation + orientation) are applied.  The bounds are kept
# separate from the foot reference offsets: a foot point is useful for
# reporting how close a pose is to the floor, while the full link envelopes
# protect against initial ground penetration.
_COLLISION_BOUNDS_MIN = (
    (-0.2250, -0.0575, 0.0000),
    (-0.0525, -0.0400, -0.21628427),
    (-0.03750001, -0.03600003, -0.00100000),
    (-0.01747685, -0.11205525, -0.15050000),
    (-0.06701610, 0.00300000, -0.06700465),
    (-0.03942231, -0.04300000, -0.12495395),
    (-0.06701610, 0.00900000, -0.16574659),
    (-0.03942231, 0.00300001, -0.03666601),
)
_COLLISION_BOUNDS_MAX = (
    (0.2250, 0.1925, 0.1100),
    (0.0395, 0.0400, 0.01000000),
    (0.03750008, 0.97499991, 0.09500001),
    (0.11747142, 0.17500346, 0.06550000),
    (0.13875632, 0.04900001, 0.16574659),
    (0.15296226, -0.00299998, 0.03666601),
    (0.13875632, 0.05500001, 0.06700465),
    (0.15296226, 0.04300002, 0.12495395),
)
_GROUND_NUMERICAL_TOLERANCE = 1.0e-5


def _uniform(
    low: float,
    high: float,
    shape: tuple[int, ...],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a uniform tensor without depending on Isaac Lab utilities."""

    values = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    return low + (high - low) * values


def _quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply an ``xyzw`` quaternion to a vector."""

    if quat.ndim < vector.ndim:
        quat = quat.reshape((1,) * (vector.ndim - quat.ndim) + quat.shape)
    elif vector.ndim < quat.ndim:
        vector = vector.reshape((1,) * (quat.ndim - vector.ndim) + vector.shape)
    q_xyz = quat[..., :3]
    q_w = quat[..., 3:4]
    q_xyz, vector = torch.broadcast_tensors(q_xyz, vector)
    twice_cross = 2.0 * torch.cross(q_xyz, vector, dim=-1)
    return vector + q_w * twice_cross + torch.cross(q_xyz, twice_cross, dim=-1)


def sample_initial_commands(
    num_envs: int,
    cfg: Any,
    device: torch.device | str,
    *,
    mode: str = "mixed",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample reset commands in the task's five-value command format.

    ``mixed`` keeps 30% sitting environments and splits the remaining 70%
    equally between standing and walking.  ``stand`` is used by the pose
    inspector and produces only zero-speed standing commands.
    """

    if mode not in {"mixed", "stand"}:
        raise ValueError(f"Unsupported initial command mode: {mode}")

    commands = torch.zeros((num_envs, 5), device=device, dtype=torch.float32)
    if num_envs == 0:
        return commands

    if mode == "stand":
        return commands

    sitting = torch.rand((num_envs,), device=device, generator=generator) < float(
        cfg.initial_sitting_fraction
    )
    walking = (~sitting) & (
        torch.rand((num_envs,), device=device, generator=generator)
        < float(cfg.initial_walking_fraction)
    )

    commands[:, 0] = sitting.to(dtype=commands.dtype)
    commands[sitting, 1] = float(cfg.command_info_cfg["sit_min"]) * 0.5

    min_speed, max_speed = cfg.initial_walking_speed_range
    magnitude = _uniform(
        float(min_speed),
        float(max_speed),
        (num_envs,),
        device,
        generator=generator,
    )
    direction = torch.where(
        torch.rand((num_envs,), device=device, generator=generator) < 0.5,
        -torch.ones((num_envs,), device=device),
        torch.ones((num_envs,), device=device),
    )
    commands[walking, 4] = magnitude[walking] * direction[walking]
    return commands


def apply_sitting_reset_variation(
    default_joint_pos: torch.Tensor,
    commands: torch.Tensor,
    cfg: Any,
    rod_body_index: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Keep the legacy small sitting-pose variation unchanged."""

    joint_pos = default_joint_pos.clone()
    sitting = commands[:, 0] == 1
    if bool(sitting.any().item()):
        variation = _uniform(
            -float(cfg.initial_tilt_angle_variation),
            float(cfg.initial_tilt_angle_variation),
            (len(commands),),
            joint_pos.device,
            dtype=joint_pos.dtype,
            generator=generator,
        )
        joint_pos[sitting, rod_body_index] += variation[sitting]
    return joint_pos


def _set_template_pose(
    joint_pos: torch.Tensor,
    cfg: Any,
    indices: InitialPoseIndices,
    template_b: torch.Tensor,
) -> None:
    """Write one of the two existing standing templates into ``joint_pos``."""

    template_a = cfg.default_standing_state_a
    template_b_values = cfg.default_standing_state_b

    for joint_index, key in (
        (indices.rotor_rod, "rotor_rod"),
        (indices.rod_body, "rod_body"),
        (indices.body_right_hip, "body_right_hip"),
        (indices.body_left_hip, "body_left_hip"),
        (indices.right_hip_shin, "right_hip_shin"),
        (indices.left_hip_shin, "left_hip_shin"),
    ):
        value_a = float(template_a[key])
        value_b = float(template_b_values[key])
        values = torch.where(
            template_b,
            torch.full_like(template_b, value_b, dtype=joint_pos.dtype),
            torch.full_like(template_b, value_a, dtype=joint_pos.dtype),
        )
        joint_pos[:, joint_index] = values


def _clamp_joint_positions(
    joint_pos: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    indices: InitialPoseIndices,
) -> None:
    """Clamp all joints touched by the sampler to their articulation limits."""

    joint_indices = torch.tensor(
        [
            indices.rotor_rod,
            indices.rod_body,
            indices.body_right_hip,
            indices.body_left_hip,
            indices.right_hip_shin,
            indices.left_hip_shin,
        ],
        device=joint_pos.device,
        dtype=torch.long,
    )
    limits = soft_joint_pos_limits[:, joint_indices]
    joint_pos[:, joint_indices] = torch.maximum(
        torch.minimum(joint_pos[:, joint_indices], limits[..., 1]), limits[..., 0]
    )


def sample_randomized_joint_positions(
    default_joint_pos: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    cfg: Any,
    indices: InitialPoseIndices,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate broad standing/step-like joint positions for active resets."""

    joint_pos = default_joint_pos.clone()
    num_envs = joint_pos.shape[0]
    if num_envs == 0:
        return joint_pos

    template_b = torch.rand((num_envs,), device=joint_pos.device, generator=generator) < 0.5
    _set_template_pose(joint_pos, cfg, indices, template_b)

    # The beam-to-body joint is the body tilt.  It is absolute by design:
    # sitting remains near -80 degrees, while active initial poses stay inside
    # the requested -60..+60 degree range.
    body_tilt_min, body_tilt_max = cfg.initial_body_tilt_range
    joint_pos[:, indices.rod_body] = _uniform(
        float(body_tilt_min),
        float(body_tilt_max),
        (num_envs,),
        joint_pos.device,
        dtype=joint_pos.dtype,
        generator=generator,
    )

    # Start from the configured standing/reference beam angle.  The
    # ground-alignment pass changes this joint only when the sampled pose would
    # otherwise penetrate the floor.
    joint_pos[:, indices.rotor_rod] = float(cfg.initial_rotor_rod_search_start)

    hip_delta = _uniform(
        -float(cfg.initial_hip_delta),
        float(cfg.initial_hip_delta),
        (num_envs,),
        joint_pos.device,
        dtype=joint_pos.dtype,
        generator=generator,
    )
    knee_delta = _uniform(
        -float(cfg.initial_knee_delta),
        float(cfg.initial_knee_delta),
        (num_envs,),
        joint_pos.device,
        dtype=joint_pos.dtype,
        generator=generator,
    )

    # 0 = same-direction deltas, 1 = opposite-direction deltas,
    # 2 = independent deltas.  Hip and knee pairs choose independently.
    hip_modes = torch.randint(0, 3, (num_envs,), device=joint_pos.device, generator=generator)
    knee_modes = torch.randint(0, 3, (num_envs,), device=joint_pos.device, generator=generator)
    hip_left_delta = torch.where(
        hip_modes == 0,
        hip_delta,
        torch.where(
            hip_modes == 1,
            -hip_delta,
            _uniform(
                -float(cfg.initial_hip_delta),
                float(cfg.initial_hip_delta),
                (num_envs,),
                joint_pos.device,
                dtype=joint_pos.dtype,
                generator=generator,
            ),
        ),
    )
    knee_left_delta = torch.where(
        knee_modes == 0,
        knee_delta,
        torch.where(
            knee_modes == 1,
            -knee_delta,
            _uniform(
                -float(cfg.initial_knee_delta),
                float(cfg.initial_knee_delta),
                (num_envs,),
                joint_pos.device,
                dtype=joint_pos.dtype,
                generator=generator,
            ),
        ),
    )

    joint_pos[:, indices.body_right_hip] += hip_delta
    joint_pos[:, indices.body_left_hip] += hip_left_delta
    joint_pos[:, indices.right_hip_shin] += knee_delta
    joint_pos[:, indices.left_hip_shin] += knee_left_delta
    _clamp_joint_positions(joint_pos, soft_joint_pos_limits, indices)
    return joint_pos


def _bounds_on_device(
    device: torch.device | str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    bounds_min = torch.tensor(_COLLISION_BOUNDS_MIN, device=device, dtype=dtype)
    bounds_max = torch.tensor(_COLLISION_BOUNDS_MAX, device=device, dtype=dtype)
    return bounds_min, bounds_max


def _compute_collision_lower_z_per_body(body_poses: torch.Tensor) -> torch.Tensor:
    """Return the lowest point of each conservative link envelope."""

    bounds_min, bounds_max = _bounds_on_device(body_poses.device, body_poses.dtype)
    center = (bounds_min + bounds_max) * 0.5
    half_extent = (bounds_max - bounds_min) * 0.5

    center_world = body_poses[..., :3] + _quat_apply(body_poses[..., 3:7], center)
    local_axes = torch.eye(3, device=body_poses.device, dtype=body_poses.dtype).view(1, 1, 3, 3)
    local_axes = local_axes.expand(body_poses.shape[0], body_poses.shape[1], -1, -1)
    quaternions = body_poses[..., 3:7].unsqueeze(-2).expand(-1, -1, 3, -1)
    world_axes = _quat_apply(quaternions, local_axes)
    vertical_extent = (world_axes[..., 2].abs() * half_extent.unsqueeze(0)).sum(dim=-1)
    return center_world[..., 2] - vertical_extent


def _compute_collision_lower_z(body_poses: torch.Tensor) -> torch.Tensor:
    """Return the lowest point of each pose's conservative link envelope."""

    return _compute_collision_lower_z_per_body(body_poses).amin(dim=-1)


def compute_pose_diagnostics(
    robot: Any,
    env_ids: torch.Tensor,
    collision_body_indices: torch.Tensor,
    left_shin_body_index: int,
    right_shin_body_index: int,
    left_foot_offset: torch.Tensor,
    right_foot_offset: torch.Tensor,
) -> GroundPoseDiagnostics:
    """Read feet and collision envelope from the articulation's current FK."""

    body_poses_all = robot.data.body_link_pose_w.torch
    selected_body_poses = body_poses_all[env_ids][:, collision_body_indices]
    left_shin_pose = body_poses_all[env_ids, left_shin_body_index]
    right_shin_pose = body_poses_all[env_ids, right_shin_body_index]

    left_offset = left_foot_offset.to(device=body_poses_all.device, dtype=body_poses_all.dtype)
    right_offset = right_foot_offset.to(device=body_poses_all.device, dtype=body_poses_all.dtype)
    left_foot = left_shin_pose[:, :3] + _quat_apply(left_shin_pose[:, 3:7], left_offset)
    right_foot = right_shin_pose[:, :3] + _quat_apply(right_shin_pose[:, 3:7], right_offset)
    lower_z = _compute_collision_lower_z(selected_body_poses)
    # The fixed Rock envelope is authored exactly on the ground plane.  FK
    # can report a few nanometres below zero for that contact because of float
    # rounding; normalize only that numerical band and keep real penetration
    # values visible to the safety check.
    lower_z = torch.where(
        (lower_z < 0.0) & (lower_z >= -_GROUND_NUMERICAL_TOLERANCE),
        torch.zeros_like(lower_z),
        lower_z,
    )
    return GroundPoseDiagnostics(left_foot=left_foot, right_foot=right_foot, collision_lower_z=lower_z)


def _write_probe_state(
    robot: Any,
    env_ids: torch.Tensor,
    root_pose: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    forward_fn: Callable[[], None] | None = None,
) -> None:
    """Write a candidate without changing the root's requested position."""

    root_vel = torch.zeros(
        (len(env_ids), 6), device=joint_pos.device, dtype=joint_pos.dtype
    )
    robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
    if forward_fn is not None:
        # FK data is otherwise refreshed only by the normal reset/step path.
        # The sampler needs the candidate FK before returning from reset.
        forward_fn()


def align_bottom_rotor_to_ground(
    robot: Any,
    env_ids: torch.Tensor,
    root_pose: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    cfg: Any,
    indices: InitialPoseIndices,
    collision_body_indices: torch.Tensor,
    left_foot_offset: torch.Tensor,
    right_foot_offset: torch.Tensor,
    forward_fn: Callable[[], None] | None = None,
) -> GroundSafePoseResult:
    """Adjust only ``bottom_rotor`` until the candidate clears the ground.

    The search starts from ``initial_rotor_rod_search_start`` and checks the
    lower side in small increments before trying the upper side. For this
    asset the lower side raises the feet, so this ordering avoids spending FK
    passes on the direction that lowers them while retaining a sign-agnostic
    fallback. Root position and orientation are never modified by this
    function.
    """

    if len(env_ids) == 0:
        empty = joint_pos.new_empty((0, 3))
        diagnostics = GroundPoseDiagnostics(empty, empty, joint_pos.new_empty((0,)))
        return GroundSafePoseResult(
            joint_pos,
            joint_vel,
            diagnostics,
            torch.empty(0, dtype=torch.bool, device=joint_pos.device),
        )

    work_pos = joint_pos.clone()
    work_vel = joint_vel.clone()
    min_angle = float(cfg.initial_rotor_rod_search_min)
    max_angle = min(
        float(cfg.initial_rotor_rod_search_max),
        float(cfg.termination_rod_angle) - float(cfg.initial_ground_safety_margin),
    )
    step = float(cfg.initial_rotor_rod_search_step)
    safety_margin = float(cfg.initial_ground_safety_margin)
    safety_limit = safety_margin - _GROUND_NUMERICAL_TOLERANCE
    start_angle = work_pos[:, indices.rotor_rod].clone()
    unresolved = torch.ones((len(env_ids),), device=joint_pos.device, dtype=torch.bool)
    diagnostics = None

    negative_steps = max(0, int(math.ceil((start_angle[0].item() - min_angle) / step)))
    positive_steps = max(0, int(math.ceil((max_angle - start_angle[0].item()) / step)))
    search_steps = min(
        int(cfg.initial_rotor_rod_search_steps),
        1 + negative_steps + positive_steps,
    )

    for search_index in range(search_steps):
        _write_probe_state(robot, env_ids, root_pose, work_pos, work_vel, forward_fn)
        diagnostics = compute_pose_diagnostics(
            robot,
            env_ids,
            collision_body_indices,
            indices.left_shin_body,
            indices.right_shin_body,
            left_foot_offset,
            right_foot_offset,
        )
        unresolved = diagnostics.collision_lower_z < safety_limit
        if not bool(unresolved.any().item()):
            break

        if search_index == search_steps - 1:
            break

        if search_index < negative_steps:
            distance = step * float(search_index + 1)
            direction = -1.0
        else:
            distance = step * float(search_index - negative_steps + 1)
            direction = 1.0
        next_angle = torch.clamp(start_angle + direction * distance, min=min_angle, max=max_angle)
        can_try = (next_angle - work_pos[:, indices.rotor_rod]).abs() > 1.0e-7
        try_mask = unresolved & can_try
        work_pos[try_mask, indices.rotor_rod] = next_angle[try_mask]
        if not bool(try_mask.any().item()):
            break

    # Always evaluate once at the final candidate, including the case where
    # the search reached its maximum angle on the previous iteration.
    _write_probe_state(robot, env_ids, root_pose, work_pos, work_vel, forward_fn)
    diagnostics = compute_pose_diagnostics(
        robot,
        env_ids,
        collision_body_indices,
        indices.left_shin_body,
        indices.right_shin_body,
        left_foot_offset,
        right_foot_offset,
    )
    safe = diagnostics.collision_lower_z >= safety_limit
    return GroundSafePoseResult(work_pos, work_vel, diagnostics, safe)


def sample_ground_safe_initial_pose(
    robot: Any,
    env_ids: torch.Tensor,
    default_joint_pos: torch.Tensor,
    default_joint_vel: torch.Tensor,
    root_pose: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    cfg: Any,
    indices: InitialPoseIndices,
    collision_body_indices: torch.Tensor,
    left_foot_offset: torch.Tensor,
    right_foot_offset: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    forward_fn: Callable[[], None] | None = None,
) -> GroundSafePoseResult:
    """Sample active poses, retry invalid candidates, and return safe states."""

    num_envs = len(env_ids)
    if num_envs == 0:
        empty_diag = GroundPoseDiagnostics(
            default_joint_pos.new_empty((0, 3)),
            default_joint_pos.new_empty((0, 3)),
            default_joint_pos.new_empty((0,)),
        )
        return GroundSafePoseResult(
            default_joint_pos,
            default_joint_vel,
            empty_diag,
            torch.empty(0, dtype=torch.bool, device=default_joint_pos.device),
        )

    result_pos = default_joint_pos.clone()
    result_vel = torch.zeros_like(default_joint_vel)
    result_lower_z = torch.full(
        (num_envs,), float("nan"), device=default_joint_pos.device, dtype=default_joint_pos.dtype
    )
    result_left_foot = torch.full(
        (num_envs, 3), float("nan"), device=default_joint_pos.device, dtype=default_joint_pos.dtype
    )
    result_right_foot = torch.full(
        (num_envs, 3), float("nan"), device=default_joint_pos.device, dtype=default_joint_pos.dtype
    )
    accepted = torch.zeros((num_envs,), device=default_joint_pos.device, dtype=torch.bool)
    pending = torch.arange(num_envs, device=default_joint_pos.device, dtype=torch.long)

    for _ in range(int(cfg.initial_pose_resample_attempts)):
        if len(pending) == 0:
            break
        candidate_pos = sample_randomized_joint_positions(
            default_joint_pos[pending],
            soft_joint_pos_limits[pending],
            cfg,
            indices,
            generator=generator,
        )
        candidate_vel = torch.zeros_like(default_joint_vel[pending])
        candidate_result = align_bottom_rotor_to_ground(
            robot,
            env_ids[pending],
            root_pose[pending],
            candidate_pos,
            candidate_vel,
            cfg,
            indices,
            collision_body_indices,
            left_foot_offset,
            right_foot_offset,
            forward_fn,
        )

        accepted_local = candidate_result.safe
        if bool(accepted_local.any().item()):
            accepted_indices = pending[accepted_local]
            result_pos[accepted_indices] = candidate_result.joint_pos[accepted_local]
            result_vel[accepted_indices] = candidate_result.joint_vel[accepted_local]
            result_lower_z[accepted_indices] = candidate_result.diagnostics.collision_lower_z[accepted_local]
            result_left_foot[accepted_indices] = candidate_result.diagnostics.left_foot[accepted_local]
            result_right_foot[accepted_indices] = candidate_result.diagnostics.right_foot[accepted_local]
            accepted[accepted_indices] = True
        pending = pending[~accepted_local]

    if len(pending) > 0:
        # A deterministic upright fallback prevents a rare unlucky batch of
        # broad poses from ever being written below the ground.  If this fails,
        # raising is safer than silently creating a penetrating reset state.
        fallback = default_joint_pos[pending].clone()
        _set_template_pose(
            fallback,
            cfg,
            indices,
            torch.zeros(len(pending), device=fallback.device, dtype=torch.bool),
        )
        fallback[:, indices.rod_body] = 0.0
        fallback[:, indices.rotor_rod] = float(cfg.initial_rotor_rod_search_start)
        _clamp_joint_positions(fallback, soft_joint_pos_limits[pending], indices)
        fallback_result = align_bottom_rotor_to_ground(
            robot,
            env_ids[pending],
            root_pose[pending],
            fallback,
            torch.zeros_like(default_joint_vel[pending]),
            cfg,
            indices,
            collision_body_indices,
            left_foot_offset,
            right_foot_offset,
            forward_fn,
        )
        if not bool(fallback_result.safe.all().item()):
            failed = int((~fallback_result.safe).sum().item())
            lower_min = float(fallback_result.diagnostics.collision_lower_z.min().item())
            lower_max = float(fallback_result.diagnostics.collision_lower_z.max().item())
            fallback_body_poses = robot.data.body_link_pose_w.torch[env_ids[pending]][:, collision_body_indices]
            fallback_body_lower_z = _compute_collision_lower_z_per_body(fallback_body_poses).amin(dim=0)
            raise RuntimeError(
                f"Unable to find {failed} ground-safe initial CBR-I pose(s) "
                f"within rotor range {float(cfg.initial_rotor_rod_search_min):.4f}.."
                f"{float(cfg.initial_rotor_rod_search_max):.4f} rad "
                f"(collision lower z range {lower_min:.5f}..{lower_max:.5f} m; "
                f"per-body minima {fallback_body_lower_z.detach().cpu().tolist()})"
            )
        result_pos[pending] = fallback_result.joint_pos
        result_vel[pending] = fallback_result.joint_vel
        result_lower_z[pending] = fallback_result.diagnostics.collision_lower_z
        result_left_foot[pending] = fallback_result.diagnostics.left_foot
        result_right_foot[pending] = fallback_result.diagnostics.right_foot
        accepted[pending] = True

    diagnostics = GroundPoseDiagnostics(
        left_foot=result_left_foot,
        right_foot=result_right_foot,
        collision_lower_z=result_lower_z,
    )
    return GroundSafePoseResult(result_pos, result_vel, diagnostics, accepted)
