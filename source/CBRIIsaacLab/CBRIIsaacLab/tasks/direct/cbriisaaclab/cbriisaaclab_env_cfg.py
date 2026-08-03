# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import gymnasium as gym
import numpy as np

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from CBRIIsaacLab.robots.CBRI import CBR_I_CONFIG

joint_names = [
    "body_Revolute_4",  # body_right_hip
    "body_Revolute_5",  # body_left_hip
    "right_hip_Revolute_6",  # right_hip_shin
    "left_hip_Revolute_7",  # left_hip_shin
]

@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.8, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=joint_names),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- scene
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, -0.1], [0.0, 0.0, 0.1]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

@configclass
class CbriisaaclabEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 5
    episode_length_s = 25.0
    # - spaces definition
    # Actions are normalized absolute joint-target commands. The environment
    # maps them into the soft joint limits; smoothness is learned through the
    # quadratic target-change penalty.
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    observation_space = 19
    state_space = 0

    phys_sps = 250

    # domain randomization config
    events: EventCfg = EventCfg()

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / phys_sps, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = CBR_I_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    feet_contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_shin",
        history_length=3,
        update_period=1 / phys_sps,
    )


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    base_rotor_dof_name = "Rock_Revolute_1"
    rotor_rod_dof_name = "bottom_rotor_Revolute_2"
    rod_body_dof_name = "rod_1_Revolute_3"
    body_right_hip_dof_name = "body_Revolute_4"
    body_left_hip_dof_name = "body_Revolute_5"
    right_hip_shin_dof_name = "right_hip_Revolute_6"
    left_hip_shin_dof_name = "left_hip_Revolute_7"

    # initial tilt angle variation
    initial_tilt_angle_variation = 1.0 / 180.0 * math.pi  # 1 degree in radians
    head_offset_from_torso_loc = [0.04,0.16,0]
    left_foot_offset_from_shin_loc = [0.14,0,0.08]
    right_foot_offset_from_shin_loc = [0.14,0,-0.08]

    # Reward terms
    # Positive velocity tracking reward:
    #   scale * exp(-(speed_error / std)^2)
    # With these values: 0.0 error -> +1.0, 0.5 error -> +0.37,
    # and 1.0 error -> +0.018 reward per walking step.
    velocity_tracking_reward_scale = 1.0
    velocity_tracking_error_std = 0.5

    # Walking posture regularization. These values are penalties per radian
    # (or per radian/second for joint velocity). For example, 0.1 rad of
    # height-joint error gives -0.05 reward with the default scale below.
    walking_height_joint_target = 0.0
    walking_height_joint_penalty_scale = 0.5
    walking_body_angle_target = 0.0
    walking_body_angle_penalty_scale = 0.05
    walking_joint_velocity_penalty_scale = 1.0e-5

    # Penalize low knees only while a walking command is active. Each low knee
    # contributes -0.05 with the default configuration.
    low_knee_height_threshold = 0.1
    low_knee_penalty_scale = 0.05

    # Sitting target and regularization. The target is kept just inside the
    # nominal +/-124 degree knee pose (1% margin), matching the previous code.
    sitting_target_state = {
        "rotor_rod": 5.2 * math.pi / 180.0,
        "rod_body": -80.0 * math.pi / 180.0,
        "body_right_hip": 0.0,
        "body_left_hip": 0.0,
        "right_hip_shin": -124.0 * math.pi / 180.0 * 0.99,
        "left_hip_shin": 124.0 * math.pi / 180.0 * 0.99,
    }
    sitting_joint_position_penalty_scale = 0.1  # 0.1 rad error in one joint -> -0.01 before sitting_reward_scale
    sitting_velocity_penalty_scale = 0.1  # 0.1 rad/s -> -0.01 before sitting_reward_scale
    sitting_reward_scale = 0.5

    # Common rewards and penalties.
    alive_reward_scale = 0.05  # reward per non-terminal step
    termination_penalty_scale = 20.0  # terminal transition contributes -20
    target_change_penalty_scale = 0.01  # per rad^2; four 0.5 rad changes -> -0.01
    feet_contact_force_threshold = 1.0
    feet_slide_penalty_scale = 0.03  # 1 m/s sliding speed on one contact -> -0.03
    feet_clearance_reward_scale = 0.03  # raw clearance score is bounded to roughly [0, 1]
    feet_clearance_height_scale = 0.07  # height scale for the tanh clearance score [m]
    feet_clearance_speed_scale = 3.0  # scale for upward swing-foot velocity [1/(m/s)]
    moving_command_threshold = 0.15
    reward_log_interval = 100
    # - reset states/conditions
    termination_rod_angle = 8.9 * math.pi / 180.0
    termination_head_height = 0.1

    # observation noise
    add_noise = True
    noise_pos_hip_knee = 0.05
    noise_vel_hip_knee = 0.05
    noise_height_pos = 0.01
    noise_height_vel = 0.05
    noise_angle_pos = 0.05
    noise_angle_vel = 0.05
    noise_vel = 0.1

    #commands settings
    command_info_cfg = {
        'sit_min':phys_sps/decimation * 1,
        'sit_max':phys_sps/decimation * 2,
        'walk_min':phys_sps/decimation * 7,
        'walk_max':phys_sps/decimation * 13,
        'speed_min':phys_sps/decimation * 2,
        'speed_max':phys_sps/decimation * 9,
    }

    default_standing_state_a = {
        'rotor_rod': 1.0 * math.pi / 180.0,
        'rod_body': -17.0 * math.pi / 180,
        'body_right_hip': 22.0 * math.pi / 180,
        'body_left_hip': -138.0 * math.pi / 180,
        'right_hip_shin': -80.0 * math.pi / 180,
        'left_hip_shin': 45.0 * math.pi / 180,
    }

    default_standing_state_b = {
        'rotor_rod': 1.0 * math.pi / 180.0,
        'rod_body': -17.0 * math.pi / 180,
        'body_right_hip': 138.0 * math.pi / 180,
        'body_left_hip': -22.0 * math.pi / 180,
        'right_hip_shin': -45.0 * math.pi / 180,
        'left_hip_shin': 80.0 * math.pi / 180,
    }
