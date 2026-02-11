# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, mdp
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
import math
from CBRIIsaacLab.robots.CBRI import CBR_I_CONFIG
import isaaclab.terrains as terrain_gen

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
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.0),
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
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
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
            "gravity_distribution_params": ([0.0, 0.0, 0.-2], [0.0, 0.0, 0.2]),
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
    action_space = 4
    observation_space = 19
    state_space = 0

    phys_sps = 250

    # domain randomization config
    events: EventCfg = EventCfg()

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / phys_sps, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = CBR_I_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")


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
    initial_tilt_angle_variation = 1.0/180 * math.pi  # 20 degrees in radians
    head_offset_from_torso_loc = [0.04,0.16,0]
    left_foot_offset_from_shin_loc = [0.14,0,0.08]
    right_foot_offset_from_shin_loc = [0.14,0,-0.08]

    # - action scale
    action_hip_scale = 196.0 / 2 /180 * math.pi  #
    action_knee_scale = 124.0 / 2 / 180 * math.pi  #
    # - reward scales
    rew_scale_alive = 1.0
    # - reset states/conditions
    termination_rod_angle = 8.9 * math.pi / 180.0
    termination_head_height = 0.1

    #commands settings
    command_info_cfg = {
        'sit_min':phys_sps/decimation * 1,
        'sit_max':phys_sps/decimation * 3,
        'walk_min':phys_sps/decimation * 7,
        'walk_max':phys_sps/decimation * 13,
        'speed_min':phys_sps/decimation * 2,
        'speed_max':phys_sps/decimation * 9,
    }

    default_standing_state_a = {
        'rotor_rod': 1.5 * math.pi / 180.0,
        'rod_body': -17.0 * math.pi / 180,
        'body_right_hip': 22.0 * math.pi / 180,
        'body_left_hip': -138.0 * math.pi / 180,
        'right_hip_shin': -80.0 * math.pi / 180,
        'left_hip_shin': 45.0 * math.pi / 180,
    }

    default_standing_state_b = {
        'rotor_rod': 1.5 * math.pi / 180.0,
        'rod_body': -17.0 * math.pi / 180,
        'body_right_hip': 138.0 * math.pi / 180,
        'body_left_hip': -22.0 * math.pi / 180,
        'right_hip_shin': -45.0 * math.pi / 180,
        'left_hip_shin': 80.0 * math.pi / 180,
    }