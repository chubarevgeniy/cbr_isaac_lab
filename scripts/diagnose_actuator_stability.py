# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark CBR-I explicit-actuator stability with constant position targets."""

from __future__ import annotations

import argparse
import json
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Cbriisaaclab-Direct-v0")
parser.add_argument("--scenario", choices=("air", "contact"), default="air")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--measure_steps", type=int, default=400)
parser.add_argument("--physics_hz", type=int, default=250)
parser.add_argument("--policy_hz", type=int, default=50)
parser.add_argument("--position_iters", type=int, default=4)
parser.add_argument("--velocity_iters", type=int, default=0)
parser.add_argument("--hip_armature", type=float, default=0.0)
parser.add_argument("--knee_armature", type=float, default=0.0)
parser.add_argument("--stiffness", type=float, default=73.3)
parser.add_argument("--damping", type=float, default=3.67)
parser.add_argument("--static_friction", type=float, default=0.0)
parser.add_argument("--dynamic_friction", type=float, default=0.0)
parser.add_argument("--viscous_friction", type=float, default=0.0)
parser.add_argument("--target_step_rad", type=float, default=0.02)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--enable_events", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import CBRIIsaacLab.tasks  # noqa: F401, E402
from CBRIIsaacLab.tasks.direct.cbriisaaclab.initial_pose_randomization import (  # noqa: E402
    sample_ground_safe_initial_pose,
)


def _synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _configure_environment():
    if args_cli.physics_hz % args_cli.policy_hz != 0:
        raise ValueError("physics_hz must be an integer multiple of policy_hz")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.dt = 1.0 / args_cli.physics_hz
    env_cfg.decimation = args_cli.physics_hz // args_cli.policy_hz
    env_cfg.sim.render_interval = env_cfg.decimation
    if not args_cli.enable_events:
        env_cfg.events = None
    env_cfg.add_noise = False
    env_cfg.uneven_ground_enabled = False
    env_cfg.episode_length_s = 1.0e9
    env_cfg.termination_rod_angle = 1.0e9
    env_cfg.termination_head_height = -1.0e9
    env_cfg.initial_sitting_fraction = 1.0
    env_cfg.initial_tilt_angle_variation = 0.0
    if args_cli.scenario == "air":
        env_cfg.sim.gravity = (0.0, 0.0, 0.0)

    articulation_props = env_cfg.robot_cfg.spawn.articulation_props
    articulation_props.solver_position_iteration_count = args_cli.position_iters
    articulation_props.solver_velocity_iteration_count = args_cli.velocity_iters

    leg_cfg = env_cfg.robot_cfg.actuators["coupled_leg_actuator"]
    leg_cfg.stiffness = args_cli.stiffness
    leg_cfg.damping = args_cli.damping
    leg_cfg.friction = args_cli.static_friction
    leg_cfg.dynamic_friction = args_cli.dynamic_friction
    leg_cfg.viscous_friction = args_cli.viscous_friction
    leg_cfg.armature = {
        "body_Revolute_4": args_cli.hip_armature,
        "body_Revolute_5": args_cli.hip_armature,
        "right_hip_Revolute_6": args_cli.knee_armature,
        "left_hip_Revolute_7": args_cli.knee_armature,
    }

    root_height = 1.0 if args_cli.scenario == "air" else 0.0
    env_cfg.robot_cfg.init_state.pos = (0.0, 0.0, root_height)
    return env_cfg


def _set_diagnostic_pose(env) -> None:
    unwrapped = env.unwrapped
    robot = unwrapped.robot
    env_ids = torch.arange(args_cli.num_envs, device=unwrapped.device, dtype=torch.long)
    joint_position = robot.data.default_joint_pos.torch.clone()
    joint_velocity = torch.zeros_like(robot.data.default_joint_vel.torch)

    if args_cli.scenario == "air":
        canonical_position = torch.zeros((args_cli.num_envs, 4), device=unwrapped.device)
        canonical_position[:, 2:] = 0.5 * unwrapped.cfg.canonical_knee_max
        raw_position = unwrapped._canonical_to_raw_actuated(canonical_position)
        joint_position[:, unwrapped._actuated_dof_indices_tensor] = raw_position
        robot.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)
    else:
        root_pose = robot.data.default_root_pose.torch.clone()
        root_pose[:, :3] += unwrapped.scene.env_origins
        generator = torch.Generator(device=unwrapped.device)
        generator.manual_seed(args_cli.seed)
        result = sample_ground_safe_initial_pose(
            robot=robot,
            env_ids=env_ids,
            default_joint_pos=joint_position,
            default_joint_vel=joint_velocity,
            root_pose=root_pose,
            soft_joint_pos_limits=robot.data.soft_joint_pos_limits.torch,
            cfg=unwrapped.cfg,
            indices=unwrapped.initial_pose_indices,
            collision_body_indices=unwrapped.collision_body_indices,
            left_foot_offset=unwrapped.left_foot_offset,
            right_foot_offset=unwrapped.right_foot_offset,
            generator=generator,
            forward_fn=unwrapped.sim.forward,
        )
        if not bool(result.safe.all().item()):
            raise RuntimeError("Failed to generate ground-safe contact test poses")
        robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros_like(robot.data.default_root_vel.torch), env_ids=env_ids
        )
        robot.write_joint_position_to_sim_index(position=result.joint_pos, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)

    unwrapped.sim.forward()
    unwrapped.scene.update(dt=0.0)
    unwrapped.joint_pos = robot.data.joint_pos.torch
    unwrapped.joint_vel = robot.data.joint_vel.torch


def _constant_hold_action(env) -> torch.Tensor:
    unwrapped = env.unwrapped
    raw_position = unwrapped.robot.data.joint_pos.torch.index_select(
        1, unwrapped._actuated_dof_indices_tensor
    )
    canonical_position = unwrapped._raw_to_canonical_actuated(raw_position)
    if args_cli.scenario == "air":
        canonical_position = canonical_position + args_cli.target_step_rad
    return (canonical_position - unwrapped._canonical_action_offset) / unwrapped._canonical_action_scale


def main() -> None:
    env = gym.make(args_cli.task, cfg=_configure_environment())
    try:
        env.reset(seed=args_cli.seed)
        unwrapped = env.unwrapped
        _set_diagnostic_pose(env)
        action = _constant_hold_action(env)
        full_mass_matrix = unwrapped.robot.data.mass_matrix.torch[0]
        actuated_indices = unwrapped._actuated_dof_indices_tensor
        actuated_mass_matrix = full_mass_matrix.index_select(0, actuated_indices).index_select(
            1, actuated_indices
        )
        actuated_mass_eigenvalues = torch.linalg.eigvalsh(actuated_mass_matrix)

        with torch.inference_mode():
            for _ in range(args_cli.warmup_steps):
                env.step(action)

            scalar_zero = torch.zeros((), device=unwrapped.device)
            joint_velocity_square_sum = scalar_zero.clone()
            foot_velocity_square_sum = scalar_zero.clone()
            tracking_error_square_sum = scalar_zero.clone()
            motor_effort_square_sum = scalar_zero.clone()
            saturation_count = scalar_zero.clone()
            joint_velocity_abs_max = scalar_zero.clone()
            foot_velocity_abs_max = scalar_zero.clone()

            _synchronize(str(unwrapped.device))
            start_time = time.perf_counter()
            for _ in range(args_cli.measure_steps):
                env.step(action)

                raw_position = unwrapped.robot.data.joint_pos.torch.index_select(
                    1, unwrapped._actuated_dof_indices_tensor
                )
                canonical_position = unwrapped._raw_to_canonical_actuated(raw_position)
                joint_velocity = unwrapped.robot.data.joint_vel.torch.index_select(
                    1, unwrapped._actuated_dof_indices_tensor
                )
                left_foot_velocity = unwrapped._get_left_foot_velocity()
                right_foot_velocity = unwrapped._get_right_foot_velocity()
                foot_velocity = torch.cat((left_foot_velocity, right_foot_velocity), dim=-1)
                normalized_motor_effort = (
                    unwrapped.leg_actuator.applied_motor_effort
                    / unwrapped.leg_actuator.effort_limit.clamp_min(1.0e-6)
                )

                tracking_error = unwrapped.targets - canonical_position
                joint_velocity_square_sum += torch.square(joint_velocity).sum()
                foot_velocity_square_sum += torch.square(foot_velocity).sum()
                tracking_error_square_sum += torch.square(tracking_error).sum()
                motor_effort_square_sum += torch.square(normalized_motor_effort).sum()
                saturation_count += (normalized_motor_effort.abs() >= 0.999).sum()
                joint_velocity_abs_max = torch.maximum(joint_velocity_abs_max, joint_velocity.abs().max())
                foot_velocity_abs_max = torch.maximum(foot_velocity_abs_max, foot_velocity.abs().max())

            _synchronize(str(unwrapped.device))
            elapsed = time.perf_counter() - start_time

        joint_samples = args_cli.measure_steps * args_cli.num_envs * 4
        foot_samples = args_cli.measure_steps * args_cli.num_envs * 6
        armature = unwrapped.robot.data.joint_armature.torch.index_select(
            1, unwrapped._actuated_dof_indices_tensor
        )[0]
        runtime_static_friction = unwrapped.robot.data.joint_friction_coeff.torch.index_select(
            1, actuated_indices
        )
        runtime_dynamic_friction = unwrapped.robot.data.joint_dynamic_friction_coeff.torch.index_select(
            1, actuated_indices
        )
        runtime_viscous_friction = unwrapped.robot.data.joint_viscous_friction_coeff.torch.index_select(
            1, actuated_indices
        )
        result = {
            "scenario": args_cli.scenario,
            "num_envs": args_cli.num_envs,
            "physics_hz": args_cli.physics_hz,
            "policy_hz": args_cli.policy_hz,
            "position_iters": args_cli.position_iters,
            "velocity_iters": args_cli.velocity_iters,
            "configured_hip_armature": args_cli.hip_armature,
            "configured_knee_armature": args_cli.knee_armature,
            "configured_stiffness": args_cli.stiffness,
            "configured_damping": args_cli.damping,
            "configured_static_friction": args_cli.static_friction,
            "configured_dynamic_friction": args_cli.dynamic_friction,
            "configured_viscous_friction": args_cli.viscous_friction,
            "target_step_rad": args_cli.target_step_rad,
            "runtime_armature": armature.detach().cpu().tolist(),
            "runtime_static_friction": runtime_static_friction[0].detach().cpu().tolist(),
            "runtime_static_friction_range": [
                float(runtime_static_friction.min()),
                float(runtime_static_friction.max()),
            ],
            "runtime_dynamic_friction": runtime_dynamic_friction[0].detach().cpu().tolist(),
            "runtime_dynamic_friction_range": [
                float(runtime_dynamic_friction.min()),
                float(runtime_dynamic_friction.max()),
            ],
            "runtime_viscous_friction": runtime_viscous_friction[0].detach().cpu().tolist(),
            "runtime_viscous_friction_range": [
                float(runtime_viscous_friction.min()),
                float(runtime_viscous_friction.max()),
            ],
            "actuated_mass_diagonal_kg_m2": torch.diagonal(actuated_mass_matrix).detach().cpu().tolist(),
            "actuated_mass_eigenvalues_kg_m2": actuated_mass_eigenvalues.detach().cpu().tolist(),
            "elapsed_s": elapsed,
            "env_steps_per_s": args_cli.measure_steps / elapsed,
            "transitions_per_s": args_cli.measure_steps * args_cli.num_envs / elapsed,
            "realtime_factor": args_cli.measure_steps / elapsed / args_cli.policy_hz,
            "joint_velocity_rms_rad_s": float(torch.sqrt(joint_velocity_square_sum / joint_samples)),
            "joint_velocity_abs_max_rad_s": float(joint_velocity_abs_max),
            "foot_velocity_rms_m_s": float(torch.sqrt(foot_velocity_square_sum / foot_samples)),
            "foot_velocity_abs_max_m_s": float(foot_velocity_abs_max),
            "tracking_error_rms_rad": float(torch.sqrt(tracking_error_square_sum / joint_samples)),
            "motor_effort_rms_normalized": float(torch.sqrt(motor_effort_square_sum / joint_samples)),
            "motor_saturation_fraction": float(saturation_count / joint_samples),
        }
        print(f"ACTUATOR_DIAGNOSTIC_RESULT={json.dumps(result, sort_keys=True)}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
