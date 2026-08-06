# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from CBRIIsaacLab.robots.CBRI import CBR_I_CONFIG

joint_names = [
    "body_Revolute_4",  # body_right_hip
    "body_Revolute_5",  # body_left_hip
    "right_hip_Revolute_6",  # right_hip_shin
    "left_hip_Revolute_7",  # left_hip_shin
]

# CPU FK/ground-clearance probe for the canonical standing pose:
# canonical hips/knees = [0, 0, 0, 0], body tilt = 0 deg.  At -7.0 deg one
# shin still enters the floor; -7.5 deg is the first safe 0.5-deg grid point.
STANDING_BASE_ROTOR_ANGLE_TARGET = -7.5 * math.pi / 180.0

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
            # Small, randomized contact elasticity; 1.0 caused nearly
            # perfectly elastic and overly bouncy impacts.
            "restitution_range": (0.0, 0.2),
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
class RewardCfg:
    """Weights, targets, and thresholds used to compute the task reward."""

    # Living/death terms. The explicit death term is a CBR-I adaptation.
    alive_reward_scale = 0.15
    death_reward_scale = -200.0

    # Unitree G1 terms adapted to the available CBR-I signals.
    # The one-dimensional analogue of Unitree's track_lin_vel_xy uses the
    # same exp-kernel and std=sqrt(0.25)=0.5, with the beam-rate proxy as v.
    walk_velocity_tracking_scale = 1.0
    walk_velocity_tracking_std = math.sqrt(0.25)
    base_vertical_velocity_scale = -2.0
    base_angular_velocity_scale = -0.05
    joint_velocity_scale = -0.001
    action_rate_scale = -0.05
    joint_position_limits_scale = -5.0
    # The action-generated target is intentionally not clipped. These are
    # soft quadratic penalties for asking beyond a joint limit and for
    # separating the target from the measured joint position.
    action_target_limits_scale = -0.5
    action_target_error_scale = -0.01
    # Penalize horizontal foot motion near the ground. The exponential uses
    # the actual foot height in meters and is active outside sitting mode.
    foot_slip_scale = -0.2
    foot_slip_height_scale = 0.05  # m
    # Walking/standing pose terms. Sitting keeps the previous absolute
    # strength through its angular multiplier of 2.0 and separate height scale.
    joint_deviation_waist_scale = -0.5
    joint_deviation_legs_scale = -0.5
    flat_orientation_scale = -2.5

    # Root/base terms use one-metre beam proxies agreed for CBR-I. Positions
    # and their rates are separate parameters, although both are 1 m here.
    height_proxy_lever_arm = 1.0  # m
    walk_base_height_target = -STANDING_BASE_ROTOR_ANGLE_TARGET * height_proxy_lever_arm
    walk_base_height_scale = -5.0
    walk_body_angle_target = 0.0

    # Sitting terms.  Height is represented by the negative beam angle and a
    # one-metre lever-arm proxy, so the current raw +5.2 deg reset is -0.0908 m.
    sit_body_height_target = -5.2 * math.pi / 180.0 * height_proxy_lever_arm
    sit_body_height_scale = -10.0
    sit_body_angle_target = -80.0 * math.pi / 180.0
    # Canonical hip coordinates: 0 deg is the thigh-down reference; raw hip
    # angles 0/0 in the sitting reset therefore become +130/+130 deg.
    sit_right_hip_angle_target = 130.0 * math.pi / 180.0
    sit_left_hip_angle_target = 130.0 * math.pi / 180.0
    sit_right_knee_angle_target = 124.0 * math.pi / 180.0
    sit_left_knee_angle_target = 124.0 * math.pi / 180.0
    # Sitting is intentionally a sharper pose-matching task than walking:
    # all angular deviation terms are doubled around the sitting target.
    sit_pose_angle_multiplier = 2.0


@configclass
class CbriisaaclabEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 5
    episode_length_s = 25.0
    # - spaces definition
    action_space = 4
    # Joint state, command, and current joint targets.
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    base_rotor_dof_name = "Rock_Revolute_1"
    rotor_rod_dof_name = "bottom_rotor_Revolute_2"
    rod_body_dof_name = "rod_1_Revolute_3"
    body_right_hip_dof_name = "body_Revolute_4"
    body_left_hip_dof_name = "body_Revolute_5"
    right_hip_shin_dof_name = "right_hip_Revolute_6"
    left_hip_shin_dof_name = "left_hip_Revolute_7"

    # Initial command distribution. The 70% non-sitting environments are
    # split evenly between standing and walking commands.
    initial_sitting_fraction = 0.30
    initial_walking_fraction = 0.50
    initial_walking_speed_range = (0.25, 1.5)

    # Initial active-pose distribution.
    initial_body_tilt_range = (-20.0 * math.pi / 180.0, 20.0 * math.pi / 180.0)
    initial_hip_delta = 45.0 * math.pi / 180.0
    initial_knee_delta = 35.0 * math.pi / 180.0

    # The root stays at its configured pose. Search the beam angle around the
    # current standing value; the positive direction lowers this particular
    # USD asset, so the negative side is tried first when more clearance is
    # needed. The upper bound remains below the termination threshold.
    initial_rotor_rod_search_start = 1.0 * math.pi / 180.0
    initial_rotor_rod_search_min = -8.0 * math.pi / 180.0
    initial_rotor_rod_search_max = 8.0 * math.pi / 180.0
    initial_rotor_rod_search_step = 0.5 * math.pi / 180.0
    initial_rotor_rod_search_steps = 36
    # Contact at z=0 is allowed: Rock is the fixed root body and its authored
    # collision envelope touches the ground when root z is unchanged.
    initial_ground_safety_margin = 0.0
    initial_pose_resample_attempts = 8

    # Kept for old experiment configuration files; active reset randomization
    # uses ``initial_body_tilt_range`` instead.
    initial_tilt_angle_variation = 1.0 / 180.0 * math.pi
    head_offset_from_torso_loc = [0.04,0.16,0]
    left_foot_offset_from_shin_loc = [0.14,0,0.08]
    right_foot_offset_from_shin_loc = [0.14,0,-0.08]

    # Canonical bilateral joint coordinates.
    canonical_hip_down_angle = 130.0 * math.pi / 180.0
    canonical_hip_min = -(196.0 - 130.0) * math.pi / 180.0
    canonical_hip_max = 130.0 * math.pi / 180.0
    canonical_knee_min = 0.0
    # The USD authored limit is a few floating-point ulps inside 124 deg.
    # Keep targets effectively at 124 deg while avoiding a boundary command.
    canonical_knee_max = 124.0 * math.pi / 180.0 - 1.0e-5

    # Reward/command dimensional proxies.  Runtime observations remain raw
    # angular positions and angular velocities in radians and rad/s.
    height_proxy_lever_arm = 1.0  # m
    height_velocity_proxy_lever_arm = 1.0  # m
    longitudinal_velocity_proxy_lever_arm = 1.0  # m
    standing_base_rotor_angle_target = STANDING_BASE_ROTOR_ANGLE_TARGET

    # Unitree-style direct position action:
    #     q_target = action_default_target + action_scale * action
    # The raw policy action and resulting target are not clipped. Joint-limit
    # violations are handled as soft reward penalties instead.
    # The 0.25-rad Unitree G1 scale is kept as a reference, but is too small
    # for CBR-I's 130/124-deg sitting range, so the active scales are adapted.
    unitree_reference_action_scale = 0.25  # rad
    action_default_target = (0.0, 0.0, 0.0, 0.0)  # canonical down/straight pose
    action_hip_scale = 0.5 * canonical_hip_down_angle  # rad, 65 deg per action unit
    action_knee_scale = 0.5 * canonical_knee_max  # rad, 62 deg per action unit

    # 6 canonical joint positions + 7 raw angular velocities + 2 commands
    # + 4 last actions.
    # - reward configuration
    rewards: RewardCfg = RewardCfg()
    # - reset states/conditions
    termination_rod_angle = 8.9 * math.pi / 180.0
    termination_head_height = 0.1

    # observation noise
    add_noise = True
    noise_pos_hip_knee = 0.02   # 1.15 deg
    noise_vel_hip_knee = 0.03   # 1.72 deg/s
    noise_height_pos = 0.005    # 0.29 deg
    noise_height_vel = 0.03     # 1.72 deg/s
    noise_angle_pos = 0.02      # 1.15 deg
    noise_angle_vel = 0.03      # 1.72 deg/s
    noise_vel = 0.05            # 2.86 deg/s

    # diagnostics
    # Metrics are emitted sparsely because skrl converts logged CUDA scalars to
    # Python values, which synchronizes the CUDA stream.
    metrics_log_interval = 100
    histogram_log_interval = 1000
    metrics_speed_command_threshold = 0.05

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
