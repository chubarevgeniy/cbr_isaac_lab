# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect and freeze 16 randomized standing-like CBR-I reset poses.

The script intentionally never calls ``env.step()``.  After the initial reset
and the explicit pose write, Isaac Sim is only kept alive for visualization.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect frozen randomized CBR-I initial poses.")
parser.add_argument(
    "--task",
    type=str,
    default="Template-Cbriisaaclab-Direct-v0",
    help="Registered Isaac Lab task.",
)
parser.add_argument("--seed", type=int, default=0, help="Random seed. Use -1 for a generated seed.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable Fabric and use USD I/O."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import CBRIIsaacLab.tasks  # noqa: F401, E402
from CBRIIsaacLab.tasks.direct.cbriisaaclab.coordinate_conventions import (  # noqa: E402
    raw_actuated_to_canonical,
)
from CBRIIsaacLab.tasks.direct.cbriisaaclab.initial_pose_randomization import (  # noqa: E402
    compute_pose_diagnostics,
    sample_ground_safe_initial_pose,
    sample_initial_commands,
)


NUM_ENVS = 16


def _requested_visualizers() -> list[str]:
    """Return visualizers requested through AppLauncher."""

    visualizers = getattr(args_cli, "visualizer", None) or []
    if isinstance(visualizers, str):
        visualizers = visualizers.split(",")
    return [str(visualizer).strip().lower() for visualizer in visualizers if str(visualizer).strip()]


def _configure_newton_visualizer(env_cfg) -> None:
    """Make all 16 frozen environments visible in a useful Newton camera view."""

    if "newton" not in _requested_visualizers():
        return

    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    visualizer_cfg = NewtonVisualizerCfg()
    visualizer_cfg.eye = (14.0, -14.0, 14.0)
    visualizer_cfg.lookat = (0.0, 0.0, 0.0)
    visualizer_cfg.max_visible_envs = NUM_ENVS
    visualizer_cfg.visible_env_indices = list(range(NUM_ENVS))
    visualizer_cfg.randomly_sample_visible_envs = False
    visualizer_cfg.show_joints = True
    visualizer_cfg.show_collision = True
    env_cfg.sim.visualizer_cfgs = [visualizer_cfg]


def _live_visualizers(sim) -> list[object]:
    """Return currently open standalone/Kit visualizer instances."""

    return [visualizer for visualizer in sim.visualizers if visualizer.is_running() and not visualizer.is_closed]


def _format_vector(values: torch.Tensor, precision: int = 6) -> str:
    array = values.detach().cpu().numpy()
    return np.array2string(array, precision=precision, separator=", ", suppress_small=False)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_frozen_standing_poses(env) -> tuple[torch.Tensor, object]:
    """Overwrite reset states with deterministic stand-only sampled poses."""

    unwrapped = env.unwrapped
    robot = unwrapped.robot
    device = unwrapped.device
    env_ids = torch.arange(NUM_ENVS, device=device, dtype=torch.long)

    default_joint_pos = robot.data.default_joint_pos.torch.clone()
    default_joint_vel = robot.data.default_joint_vel.torch.clone()
    root_pose = robot.data.default_root_pose.torch.clone()
    root_pose[:, :3] += unwrapped.scene.env_origins
    root_z_reference = root_pose[:, 2].clone()

    # Use a private generator so the printed set is exactly reproducible even
    # though env.reset() itself also initializes the task once.
    generator = torch.Generator(device=device)
    generator.manual_seed(args_cli.seed)
    result = sample_ground_safe_initial_pose(
        robot=robot,
        env_ids=env_ids,
        default_joint_pos=default_joint_pos,
        default_joint_vel=default_joint_vel,
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
        raise RuntimeError("The inspector received at least one unsafe pose from the shared sampler.")
    unique_count = torch.unique(result.joint_pos, dim=0).shape[0]
    if unique_count != NUM_ENVS:
        raise RuntimeError(f"The inspector expected {NUM_ENVS} different poses, got {unique_count}.")

    joint_vel = torch.zeros_like(default_joint_vel)
    root_vel = torch.zeros_like(robot.data.default_root_vel.torch)
    robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
    robot.write_joint_position_to_sim_index(position=result.joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
    unwrapped.sim.forward()

    # Force the command and all controller state to stand at zero speed.
    unwrapped.command.copy_(sample_initial_commands(NUM_ENVS, unwrapped.cfg, device, mode="stand"))
    unwrapped.targets[env_ids] = raw_actuated_to_canonical(
        result.joint_pos[:, unwrapped._actuated_dof_indices_tensor],
        unwrapped.cfg.canonical_hip_down_angle,
    )
    unwrapped.actions[env_ids] = 0.0
    unwrapped.joint_pos = robot.data.joint_pos.torch
    unwrapped.joint_vel = robot.data.joint_vel.torch

    diagnostics = compute_pose_diagnostics(
        robot,
        env_ids,
        unwrapped.collision_body_indices,
        unwrapped.initial_pose_indices.left_shin_body,
        unwrapped.initial_pose_indices.right_shin_body,
        unwrapped.left_foot_offset,
        unwrapped.right_foot_offset,
    )
    return root_z_reference, diagnostics


def _print_report(env, root_z_reference: torch.Tensor, diagnostics) -> None:
    unwrapped = env.unwrapped
    robot = unwrapped.robot
    joint_pos = robot.data.joint_pos.torch
    joint_vel = robot.data.joint_vel.torch
    joint_limits = robot.data.soft_joint_pos_limits.torch
    root_pose = robot.data.root_link_pose_w.torch
    root_vel = robot.data.root_link_vel_w.torch

    joint_names = getattr(robot, "joint_names", None)
    if joint_names is None:
        joint_names = [f"joint_{index}" for index in range(robot.num_joints)]

    joint_limits_ok = (
        (joint_pos >= joint_limits[..., 0] - 1.0e-5)
        & (joint_pos <= joint_limits[..., 1] + 1.0e-5)
    ).all(dim=-1)
    collision_safe = diagnostics.collision_lower_z >= float(unwrapped.cfg.initial_ground_safety_margin) - 1.0e-5
    zero_velocity = torch.cat((joint_vel, root_vel), dim=-1).abs().amax(dim=-1) <= 1.0e-7
    command_zero = unwrapped.command.abs().amax(dim=-1) <= 1.0e-7
    finite = torch.isfinite(joint_pos).all(dim=-1) & torch.isfinite(root_pose).all(dim=-1)
    root_z_unchanged = (root_pose[:, 2] - root_z_reference).abs() <= 1.0e-6
    all_valid = joint_limits_ok & collision_safe & zero_velocity & command_zero & finite & root_z_unchanged

    print(f"[INSPECT] seed={args_cli.seed}")
    print(f"[INSPECT] environments={len(joint_pos)} command=stand sit=0 speed=0")
    print(f"[INSPECT] unique_joint_poses={torch.unique(joint_pos, dim=0).shape[0]}/{NUM_ENVS}")

    for env_id in range(NUM_ENVS):
        print(f"\n[ENV {env_id:02d}]")
        print("joint positions [rad]:")
        for joint_index, joint_name in enumerate(joint_names):
            print(f"  {joint_name}: {float(joint_pos[env_id, joint_index]): .9f}")
        print("joint positions [deg]:")
        for joint_index, joint_name in enumerate(joint_names):
            degrees = float(torch.rad2deg(joint_pos[env_id, joint_index]))
            print(f"  {joint_name}: {degrees: .5f}")

        rotor_index = unwrapped.initial_pose_indices.rotor_rod
        body_index = unwrapped.initial_pose_indices.rod_body
        print(f"root pose [x y z qx qy qz qw]: {_format_vector(root_pose[env_id])}")
        print(
            f"bottom_rotor: {float(joint_pos[env_id, rotor_index]): .9f} rad / "
            f"{float(torch.rad2deg(joint_pos[env_id, rotor_index])): .5f} deg"
        )
        print(
            f"rod_body: {float(joint_pos[env_id, body_index]): .9f} rad / "
            f"{float(torch.rad2deg(joint_pos[env_id, body_index])): .5f} deg"
        )
        print(f"left foot [x y z]: {_format_vector(diagnostics.left_foot[env_id])}")
        print(f"right foot [x y z]: {_format_vector(diagnostics.right_foot[env_id])}")
        print(f"minimum collision z: {float(diagnostics.collision_lower_z[env_id]): .8f}")
        print(f"minimum foot z: {float(diagnostics.minimum_foot_z[env_id]): .8f}")
        print(
            "checks: "
            f"joint_limits={bool(joint_limits_ok[env_id])} "
            f"collision_safe={bool(collision_safe[env_id])} "
            f"zero_velocities={bool(zero_velocity[env_id])} "
            f"stand_command={bool(command_zero[env_id])} "
            f"root_z_unchanged={bool(root_z_unchanged[env_id])} "
            f"finite={bool(finite[env_id])} "
            f"all_valid={bool(all_valid[env_id])}"
        )

    print(f"\n[INSPECT] all_valid={bool(all_valid.all())}")


def main() -> None:
    seed = args_cli.seed
    if seed < 0:
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        args_cli.seed = seed
    _seed_everything(seed)
    print(f"[INSPECT] Using seed {seed}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=NUM_ENVS,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.num_envs = NUM_ENVS
    _configure_newton_visualizer(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        # This is the only environment reset. No env.step() follows it.
        env.reset(seed=seed)
        root_z_reference, diagnostics = _write_frozen_standing_poses(env)
        _print_report(env, root_z_reference, diagnostics)

        # Keep the selected visualizer alive while preventing the timeline from
        # advancing the physics state. ``sim.render`` updates Newton without a
        # physics step; this is required because Newton has its own viewer loop
        # and may run without a Kit application window.
        sim = env.unwrapped.sim
        sim.pause()
        visualizer_names = [type(visualizer).__name__ for visualizer in sim.visualizers]
        print(f"[INSPECT] visualizers={visualizer_names or ['none']}")
        sim.render(skip_app_pumping=not any(visualizer.pumps_app_update() for visualizer in sim.visualizers))
        print("[INSPECT] No physics steps will be executed; close the visualizer window to finish.")

        while True:
            live_visualizers = _live_visualizers(sim)
            if live_visualizers:
                sim.render(skip_app_pumping=not any(visualizer.pumps_app_update() for visualizer in live_visualizers))
            elif simulation_app.is_running():
                # This fallback keeps the old Kit-app behavior when no
                # standalone visualizer was requested.
                simulation_app.update()
            else:
                break
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
