"""Tests for the motor-space effort reward."""

from __future__ import annotations

import torch

from CBRIIsaacLab.tasks.direct.cbriisaaclab.cbriisaaclab_env import (
    compute_action_acceleration_scale,
    compute_policy_action_abs_limit,
    compute_rewards,
    compute_target_second_difference,
)
from CBRIIsaacLab.tasks.direct.cbriisaaclab.cbriisaaclab_env_cfg import RewardCfg


def test_configured_motor_effort_weight() -> None:
    assert RewardCfg().motor_effort_scale == -0.05


def test_reward_penalizes_normalized_applied_motor_effort() -> None:
    num_envs = 2
    zeros_scalar = torch.zeros((num_envs, 1))
    zeros_joints = torch.zeros((num_envs, 4))
    normalized_motor_effort = torch.tensor(
        [
            [1.0, 0.5, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    reward = compute_rewards(
        body_vel=zeros_scalar,
        body_height=zeros_scalar,
        body_vertical_vel=zeros_scalar,
        body_angular_vel=zeros_scalar,
        body_angle=zeros_scalar,
        actuated_joint_pos=zeros_joints,
        actuated_joint_vel=zeros_joints,
        joint_pos_limits=torch.zeros(num_envs),
        target_joint_limit_violation=zeros_joints,
        normalized_motor_effort=normalized_motor_effort,
        foot_height=torch.zeros((num_envs, 2)),
        foot_horizontal_speed=torch.zeros((num_envs, 2)),
        reset_terminated=torch.zeros(num_envs, dtype=torch.bool),
        command=torch.zeros((num_envs, 2)),
        actions=zeros_joints,
        previous_actions=zeros_joints,
        previous_previous_actions=zeros_joints,
        action_target_scale=torch.ones(4),
        action_acceleration_scale=0.0,
        alive_reward_scale=0.0,
        death_reward_scale=0.0,
        walk_velocity_tracking_scale=0.0,
        walk_velocity_tracking_std=1.0,
        base_vertical_velocity_scale=0.0,
        base_angular_velocity_scale=0.0,
        joint_velocity_scale=0.0,
        joint_position_limits_scale=0.0,
        action_target_limits_scale=0.0,
        motor_effort_scale=-0.01,
        foot_slip_scale=0.0,
        foot_slip_height_scale=1.0,
        joint_deviation_waist_scale=0.0,
        joint_deviation_legs_scale=0.0,
        flat_orientation_scale=0.0,
        walk_base_height_target=0.0,
        walk_base_height_scale=0.0,
        walk_body_angle_target=0.0,
        sit_body_height_target=0.0,
        sit_body_height_scale=0.0,
        sit_body_angle_target=0.0,
        sit_right_hip_angle_target=0.0,
        sit_left_hip_angle_target=0.0,
        sit_right_knee_angle_target=0.0,
        sit_left_knee_angle_target=0.0,
        sit_pose_angle_multiplier=1.0,
    )

    torch.testing.assert_close(reward, torch.tensor([-0.0225, 0.0]))


def test_target_second_difference_is_zero_for_a_constant_ramp() -> None:
    actions = torch.tensor([[2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    previous_actions = torch.tensor([[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
    previous_previous_actions = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    action_target_scale = torch.tensor([2.0, 1.0, 1.0, 1.0])

    difference = compute_target_second_difference(
        actions,
        previous_actions,
        previous_previous_actions,
        action_target_scale,
    )

    torch.testing.assert_close(
        difference,
        torch.tensor([[0.0, 0.0, 0.0, 0.0], [-6.0, 0.0, 0.0, 0.0]]),
    )


def test_reward_uses_target_second_difference_for_action_acceleration() -> None:
    zeros_scalar = torch.zeros((1, 1))
    zeros_joints = torch.zeros((1, 4))

    reward = compute_rewards(
        body_vel=zeros_scalar,
        body_height=zeros_scalar,
        body_vertical_vel=zeros_scalar,
        body_angular_vel=zeros_scalar,
        body_angle=zeros_scalar,
        actuated_joint_pos=zeros_joints,
        actuated_joint_vel=zeros_joints,
        joint_pos_limits=torch.zeros(1),
        target_joint_limit_violation=zeros_joints,
        normalized_motor_effort=zeros_joints,
        foot_height=torch.zeros((1, 2)),
        foot_horizontal_speed=torch.zeros((1, 2)),
        reset_terminated=torch.zeros(1, dtype=torch.bool),
        command=torch.zeros((1, 2)),
        actions=torch.tensor([[2.0, 0.0, 0.0, 0.0]]),
        previous_actions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        previous_previous_actions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        action_target_scale=torch.tensor([2.0, 1.0, 1.0, 1.0]),
        action_acceleration_scale=-0.5,
        alive_reward_scale=0.0,
        death_reward_scale=0.0,
        walk_velocity_tracking_scale=0.0,
        walk_velocity_tracking_std=1.0,
        base_vertical_velocity_scale=0.0,
        base_angular_velocity_scale=0.0,
        joint_velocity_scale=0.0,
        joint_position_limits_scale=0.0,
        action_target_limits_scale=0.0,
        motor_effort_scale=0.0,
        foot_slip_scale=0.0,
        foot_slip_height_scale=1.0,
        joint_deviation_waist_scale=0.0,
        joint_deviation_legs_scale=0.0,
        flat_orientation_scale=0.0,
        walk_base_height_target=0.0,
        walk_base_height_scale=0.0,
        walk_body_angle_target=0.0,
        sit_body_height_target=0.0,
        sit_body_height_scale=0.0,
        sit_body_angle_target=0.0,
        sit_right_hip_angle_target=0.0,
        sit_left_hip_angle_target=0.0,
        sit_right_knee_angle_target=0.0,
        sit_left_knee_angle_target=0.0,
        sit_pose_angle_multiplier=1.0,
    )

    # Second difference is 1 normalized action * 2 rad/action, hence
    # -0.5 * 2^2 = -2.
    torch.testing.assert_close(reward, torch.tensor([-2.0]))


def test_action_acceleration_scale_is_clamped_linear_timestep_schedule() -> None:
    assert compute_action_acceleration_scale(0.0, -0.05, -1.0, 200_000.0, 700_000.0) == -0.05
    assert compute_action_acceleration_scale(200_000.0, -0.05, -1.0, 200_000.0, 700_000.0) == -0.05
    assert compute_action_acceleration_scale(450_000.0, -0.05, -1.0, 200_000.0, 700_000.0) == -0.525
    assert compute_action_acceleration_scale(700_000.0, -0.05, -1.0, 200_000.0, 700_000.0) == -1.0
    assert compute_action_acceleration_scale(800_000.0, -0.05, -1.0, 200_000.0, 700_000.0) == -1.0


def test_policy_action_limit_uses_farther_joint_limit_plus_twenty_percent_range() -> None:
    degrees = torch.tensor([-66.0, 130.0, 0.0, 124.0]) * torch.pi / 180.0
    target_min = torch.tensor([degrees[0], degrees[0], degrees[2], degrees[2]])
    target_max = torch.tensor([degrees[1], degrees[1], degrees[3], degrees[3]])
    action_scale = torch.tensor([65.0, 65.0, 62.0, 62.0]) * torch.pi / 180.0

    action_limit = compute_policy_action_abs_limit(
        target_min,
        target_max,
        torch.zeros(4),
        action_scale,
        0.20,
    )

    expected_target_degrees = torch.tensor([169.2, 169.2, 148.8, 148.8])
    expected_action_limit = expected_target_degrees / torch.tensor([65.0, 65.0, 62.0, 62.0])
    torch.testing.assert_close(action_limit, expected_action_limit)
