# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run a trained policy for the CBRI robot with adjustable frequency and visualization.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Run a trained policy for CBRI robot.")
parser.add_argument("--policy", type=str, default=None, help="Path to the TorchScript policy file.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to the SKRL agent checkpoint.")
parser.add_argument("--task", type=str, default=None, help="Name of the task (required for SKRL checkpoint).")
parser.add_argument("--algorithm", type=str, default="PPO", help="RL algorithm (default: PPO).")
parser.add_argument("--sim_dt", type=float, default=0.004, help="Physics simulation time step.")
parser.add_argument("--decimation", type=int, default=5, help="Control decimation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Set default device to cpu
for action in parser._actions:
    if action.dest == "device":
        action.default = "cpu"

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
import gymnasium as gym
import os
import datetime
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils
from isaaclab.sim.spawners.from_files import GroundPlaneCfg

# Import robot config and env config
from CBRIIsaacLab.robots.CBRI import CBR_I_CONFIG
from CBRIIsaacLab.tasks.direct.cbriisaaclab.cbriisaaclab_env_cfg import CbriisaaclabEnvCfg

# Imports for SKRL agent loading
import CBRIIsaacLab.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry
from skrl.utils.runner.torch import Runner


class SkrlDummyEnv:
    """Dummy environment to satisfy SKRL Runner requirements."""
    def __init__(self, num_envs, obs_space, act_space, device):
        self.num_envs = num_envs
        self.device = device
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(obs_space,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_space,), dtype=np.float32)
        self.state_space = None

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "speed": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "command": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "knee": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "low_knee": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "foot_vel_ok": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.2, 0.2, 0.2),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "foot_vel_bad": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.2, 0.2, 0.2),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


def get_command(sit: float = 1, sit_time: float = 0, walk_time: float = 0, speed_time: float = 0, speed: float = 0, device="cpu"):
    return torch.tensor([sit, sit_time, walk_time, speed_time, speed], dtype=torch.float32, device=device)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=GroundPlaneCfg())
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)))
    robot = CBR_I_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class CBRIPolicyRunner:
    def __init__(self, scene: InteractiveScene, cfg: CbriisaaclabEnvCfg):
        self.scene = scene
        self.robot = scene["robot"]
        self.cfg = cfg
        self.device = self.robot.device
        self.num_envs = self.scene.num_envs

        # Indices lookup
        self.base_rotor_dof_name_idx, _ = self.robot.find_joints(self.cfg.base_rotor_dof_name)
        self.rotor_rod_dof_name_idx, _ = self.robot.find_joints(self.cfg.rotor_rod_dof_name)
        self.rod_body_dof_name_idx, _ = self.robot.find_joints(self.cfg.rod_body_dof_name)
        self.body_right_hip_dof_name_idx, _ = self.robot.find_joints(self.cfg.body_right_hip_dof_name)
        self.body_left_hip_dof_name_idx, _ = self.robot.find_joints(self.cfg.body_left_hip_dof_name)
        self.right_hip_shin_dof_name_idx, _ = self.robot.find_joints(self.cfg.right_hip_shin_dof_name)
        self.left_hip_shin_dof_name_idx, _ = self.robot.find_joints(self.cfg.left_hip_shin_dof_name)

        self.body_idx, _ = self.robot.find_bodies('body')
        self.left_hip_idx, _ = self.robot.find_bodies('left_hip')
        self.right_hip_idx, _ = self.robot.find_bodies('right_hip')
        self.left_knee_idx, _ = self.robot.find_bodies('left_shin')
        self.right_knee_idx, _ = self.robot.find_bodies('right_shin')

        self.noise_hip_knee_indices = [
            self.body_right_hip_dof_name_idx[0],
            self.body_left_hip_dof_name_idx[0],
            self.right_hip_shin_dof_name_idx[0],
            self.left_hip_shin_dof_name_idx[0]
        ]

        self.actuated_dof_indices = [
            self.body_right_hip_dof_name_idx[0],
            self.body_left_hip_dof_name_idx[0],
            self.right_hip_shin_dof_name_idx[0],
            self.left_hip_shin_dof_name_idx[0]
        ]

        print(f"Indecies:\n\
            base_rotor_dof_name_idx: {self.base_rotor_dof_name_idx}\n\
            rotor_rod_dof_name_idx: {self.rotor_rod_dof_name_idx}\n\
            rod_body_dof_name_idx: {self.rod_body_dof_name_idx}\n\
            body_right_hip_dof_name_idx: {self.body_right_hip_dof_name_idx}\n\
            body_left_hip_dof_name_idx: {self.body_left_hip_dof_name_idx}\n\
            right_hip_shin_dof_name_idx: {self.right_hip_shin_dof_name_idx}\n\
            left_hip_shin_dof_name_idx: {self.left_hip_shin_dof_name_idx}\n")

        # Buffers
        self.command = torch.zeros((self.num_envs, 5), device=self.device)
        self.actions = torch.zeros((self.num_envs, 4), device=self.device)
        self.targets = torch.zeros((self.num_envs, 4), device=self.device)

        # Markers
        self.visualization_markers = define_markers()
        self.marker_offset = torch.zeros((self.num_envs, 3), device=self.device)
        self.marker_offset[:, -1] = 0.5

        # Obs indices
        self.obs_joint_pos_indices = torch.tensor(
            [i for i in range(self.robot.num_joints) if i != self.base_rotor_dof_name_idx[0]],
            device=self.device
        )

        # References
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def reset(self):
        # Initialize command
        self.command[:, [0, 1, 2, 3, 4]] = get_command(device=self.device, sit_time=self.cfg.command_info_cfg['sit_min'] // 2)

        # Reset robot to default state
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()

        # -- Standing initial state for 70% of environments
        # Determine which envs will be standing
        stand_mask = torch.rand(self.num_envs, device=self.device) < 0.7
        stand_indices = torch.nonzero(stand_mask).flatten()
        num_standing = len(stand_indices)

        if num_standing > 0:
            # Set standing command
            self.command[stand_indices, :] = get_command(sit=0, device=self.device)

            standing_joint_pos = joint_pos[stand_mask]

            # -- Split standing envs into two groups for different poses
            pose_a_mask = torch.rand(num_standing, device=self.device) < 0.5
            num_pose_a = pose_a_mask.sum()
            num_pose_b = num_standing - num_pose_a

            # Set standing joint positions for group A
            if num_pose_a > 0:
                standing_joint_pos[pose_a_mask, self.rotor_rod_dof_name_idx] = self.cfg.default_standing_state_a['rotor_rod']
                standing_joint_pos[pose_a_mask, self.rod_body_dof_name_idx] = self.cfg.default_standing_state_a['rod_body']
                standing_joint_pos[pose_a_mask, self.body_right_hip_dof_name_idx] = self.cfg.default_standing_state_a['body_right_hip']
                standing_joint_pos[pose_a_mask, self.body_left_hip_dof_name_idx] = self.cfg.default_standing_state_a['body_left_hip']
                standing_joint_pos[pose_a_mask, self.right_hip_shin_dof_name_idx] = self.cfg.default_standing_state_a['right_hip_shin']
                standing_joint_pos[pose_a_mask, self.left_hip_shin_dof_name_idx] = self.cfg.default_standing_state_a['left_hip_shin']

            # Set standing joint positions for group B
            if num_pose_b > 0:
                standing_joint_pos[~pose_a_mask, self.rotor_rod_dof_name_idx] = self.cfg.default_standing_state_b['rotor_rod']
                standing_joint_pos[~pose_a_mask, self.rod_body_dof_name_idx] = self.cfg.default_standing_state_b['rod_body']
                standing_joint_pos[~pose_a_mask, self.body_right_hip_dof_name_idx] = self.cfg.default_standing_state_b['body_right_hip']
                standing_joint_pos[~pose_a_mask, self.body_left_hip_dof_name_idx] = self.cfg.default_standing_state_b['body_left_hip']
                standing_joint_pos[~pose_a_mask, self.right_hip_shin_dof_name_idx] = self.cfg.default_standing_state_b['right_hip_shin']
                standing_joint_pos[~pose_a_mask, self.left_hip_shin_dof_name_idx] = self.cfg.default_standing_state_b['left_hip_shin']

            joint_pos[stand_mask] = standing_joint_pos

        # Apply initial tilt variation
        joint_pos[:, self.rod_body_dof_name_idx] += math_utils.sample_uniform(
            -self.cfg.initial_tilt_angle_variation,
            self.cfg.initial_tilt_angle_variation,
            joint_pos[:, self.rod_body_dof_name_idx].shape,
            joint_pos.device,
        )

        default_root_state = self.robot.data.default_root_state.clone()
        default_root_state[:, :3] += self.scene.env_origins

        self.robot.write_root_pose_to_sim(default_root_state[:, :7])
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

        self.targets = joint_pos[:, self.actuated_dof_indices].clone()

    def update_and_sample_commands(self):
        # update timers
        self.command[:, [1, 2, 3]] += 1

        # from sit to standing
        sit_long_idx = (self.command[:, 1] >= self.cfg.command_info_cfg['sit_min']) & (self.command[:, 0] == 1)
        prob_to_stand = (self.command[:, 1] - self.cfg.command_info_cfg['sit_min']) / (self.cfg.command_info_cfg['sit_max'] - self.cfg.command_info_cfg['sit_min'])
        commands_to_change = (torch.rand(self.num_envs, device=self.device) < prob_to_stand) & sit_long_idx
        self.command[commands_to_change, 0] = 0
        self.command[commands_to_change, 1] = 0
        self.command[commands_to_change, 2] = 0
        self.command[commands_to_change, 3] = 0
        self.command[commands_to_change, 4] = 0

        # from standing to sit
        walk_long_idx = (self.command[:, 2] >= self.cfg.command_info_cfg['walk_min']) & (self.command[:, 0] == 0)
        prob_to_sit = (self.command[:, 2] - self.cfg.command_info_cfg['walk_min']) / (self.cfg.command_info_cfg['walk_max'] - self.cfg.command_info_cfg['walk_min'])
        commands_to_change = (torch.rand(self.num_envs, device=self.device) < prob_to_sit) & walk_long_idx
        self.command[commands_to_change, 0] = 1
        self.command[commands_to_change, 1] = 0
        self.command[commands_to_change, 2] = 0
        self.command[commands_to_change, 3] = 0
        self.command[commands_to_change, 4] = 0

        # set speed for long walking
        speed_long_idx = (self.command[:, 3] >= self.cfg.command_info_cfg['speed_min']) & (self.command[:, 0] == 0)
        prob_to_speed = (self.command[:, 3] - self.cfg.command_info_cfg['speed_min']) / (self.cfg.command_info_cfg['speed_max'] - self.cfg.command_info_cfg['speed_min'])
        # if it is alrady long standing but speed min is large it is allowed to set new target speed
        commands_to_change = speed_long_idx & (torch.rand(self.num_envs, device=self.device) < prob_to_speed)
        commands_to_change_number = int(commands_to_change.sum().item())
        if (commands_to_change_number > 0):
            self.command[commands_to_change, 3] = 0
            self.command[commands_to_change, 4] = math_utils.sample_uniform(-1.5, 1.5, (commands_to_change_number,), self.device)

    def get_observations(self):
        self.update_and_sample_commands()

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        if self.cfg.add_noise:
            # Apply noise to hip and knee positions
            joint_pos[:, self.noise_hip_knee_indices] += math_utils.sample_gaussian(
                0.0, self.cfg.noise_pos_hip_knee,
                joint_pos[:, self.noise_hip_knee_indices].shape, self.device
            )
            # tilt
            joint_pos[:, [self.rod_body_dof_name_idx[0]]] += math_utils.sample_gaussian(
                0.0, self.cfg.noise_angle_pos,
                joint_pos[:, [self.rod_body_dof_name_idx[0]]].shape, self.device
            )
            # height
            joint_pos[:, [self.rotor_rod_dof_name_idx[0]]] += math_utils.sample_gaussian(
                0.0, self.cfg.noise_height_pos,
                joint_pos[:, [self.rotor_rod_dof_name_idx[0]]].shape, self.device
            )
            # Apply noise to velocities
            joint_vel[:, self.noise_hip_knee_indices] += math_utils.sample_gaussian(
                0.0, self.cfg.noise_vel_hip_knee,
                joint_pos[:, self.noise_hip_knee_indices].shape, self.device
            )
            # tilt
            joint_vel[:, [self.rod_body_dof_name_idx[0]]] += math_utils.sample_gaussian(
                0.0, self.cfg.noise_angle_vel,
                joint_vel[:, [self.rod_body_dof_name_idx[0]]].shape, self.device
            )
            # height
            joint_vel[:, [self.rotor_rod_dof_name_idx[0]]] += math_utils.sample_gaussian(
                0.0, self.cfg.noise_height_vel,
                joint_vel[:, [self.rotor_rod_dof_name_idx[0]]].shape, self.device
            )
            # speed
            joint_vel[:, [self.base_rotor_dof_name_idx[0]]] += math_utils.sample_gaussian(
                0.0, self.cfg.noise_vel,
                joint_vel[:, [self.base_rotor_dof_name_idx[0]]].shape, self.device
            )

        return torch.cat([
            joint_pos[:, self.obs_joint_pos_indices],
            joint_vel,
            self.command[:, [0, 4]],
            self.targets,
        ], dim=-1).float()

    def _scale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        # Scale actions (deltas)
        actions = actions.clamp(-1, 1)
        actions[:, 0] *= self.cfg.action_hip_scale
        actions[:, 1] *= self.cfg.action_hip_scale
        actions[:, 2] *= self.cfg.action_knee_scale
        actions[:, 3] *= self.cfg.action_knee_scale
        return actions

    def apply_actions(self, actions):
        self.actions = actions.clone()
        scaled_actions = self._scale_actions(actions)
        self.targets += scaled_actions
        limits = self.robot.data.soft_joint_pos_limits[:, self.actuated_dof_indices]
        self.targets = torch.clamp(self.targets, min=limits[..., 0], max=limits[..., 1])
        self.robot.set_joint_position_target(self.targets, joint_ids=[
            self.body_right_hip_dof_name_idx[0],
            self.body_left_hip_dof_name_idx[0],
            self.right_hip_shin_dof_name_idx[0],
            self.left_hip_shin_dof_name_idx[0],
        ])

    def _get_left_knee_location(self) -> torch.Tensor:
        left_knee_loc = self.robot.data.body_link_state_w[:, self.left_knee_idx[0], :3]
        return left_knee_loc

    def _get_right_knee_location(self) -> torch.Tensor:
        right_knee_loc = self.robot.data.body_link_state_w[:, self.right_knee_idx[0], :3]
        return right_knee_loc

    def _get_top_torso_location(self) -> torch.Tensor:
        torso_loc = self.robot.data.body_state_w[:, self.body_idx[0], :3]
        torso_rots = self.robot.data.body_state_w[:, self.body_idx[0], 3:7]
        offset = torch.tensor(self.cfg.head_offset_from_torso_loc, device=self.device).expand_as(torso_loc)
        top_torso_loc = torso_loc + math_utils.quat_apply(torso_rots, offset)
        return top_torso_loc, torso_rots

    def _get_left_foot_location(self) -> torch.Tensor:
        foot_loc = self.robot.data.body_state_w[:, self.left_knee_idx[0], :3]
        foot_rots = self.robot.data.body_state_w[:, self.left_knee_idx[0], 3:7]
        offset = torch.tensor(self.cfg.left_foot_offset_from_shin_loc, device=self.device).expand_as(foot_loc)
        foot_offset_loc = foot_loc + math_utils.quat_apply(foot_rots, offset)
        return foot_offset_loc, foot_rots

    def _get_right_foot_location(self) -> torch.Tensor:
        foot_loc = self.robot.data.body_state_w[:, self.right_knee_idx[0], :3]
        foot_rots = self.robot.data.body_state_w[:, self.right_knee_idx[0], 3:7]
        offset = torch.tensor(self.cfg.right_foot_offset_from_shin_loc, device=self.device).expand_as(foot_loc)
        foot_offset_loc = foot_loc + math_utils.quat_apply(foot_rots, offset)
        return foot_offset_loc, foot_rots

    def _get_left_foot_velocity(self) -> torch.Tensor:
        shin_vel = self.robot.data.body_state_w[:, self.left_knee_idx[0], 7:10]
        shin_ang_vel = self.robot.data.body_state_w[:, self.left_knee_idx[0], 10:13]
        shin_rots = self.robot.data.body_state_w[:, self.left_knee_idx[0], 3:7]

        offset = torch.tensor(self.cfg.left_foot_offset_from_shin_loc, device=self.device).expand_as(shin_vel)
        offset_world = math_utils.quat_apply(shin_rots, offset)

        return shin_vel + torch.cross(shin_ang_vel, offset_world, dim=-1)

    def _get_right_foot_velocity(self) -> torch.Tensor:
        shin_vel = self.robot.data.body_state_w[:, self.right_knee_idx[0], 7:10]
        shin_ang_vel = self.robot.data.body_state_w[:, self.right_knee_idx[0], 10:13]
        shin_rots = self.robot.data.body_state_w[:, self.right_knee_idx[0], 3:7]

        offset = torch.tensor(self.cfg.right_foot_offset_from_shin_loc, device=self.device).expand_as(shin_vel)
        offset_world = math_utils.quat_apply(shin_rots, offset)

        return shin_vel + torch.cross(shin_ang_vel, offset_world, dim=-1)

    def visualize(self):
        # Arrow locations for command and speed visualization (not true torso top/bottom)
        torso_base_loc = self.robot.data.body_state_w[:, self.body_idx[0], :3]
        arrow_loc = torch.vstack((torso_base_loc + self.marker_offset * 1.1, torso_base_loc + self.marker_offset))
        head_loc, head_rots = self._get_top_torso_location()

        # Rotation for arrows
        ang_speed = self.joint_vel[:, self.base_rotor_dof_name_idx[0]]
        base_angle = -self.joint_pos[:, self.base_rotor_dof_name_idx[0]]
        up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        rots_actual = math_utils.quat_from_angle_axis(base_angle - torch.pi / 2 - torch.sign(ang_speed) * torch.pi / 2, up_vec)
        rots_command = math_utils.quat_from_angle_axis(base_angle - torch.pi / 2 - torch.sign(self.command[:, 4]) * torch.pi / 2, up_vec)
        arrow_rots = torch.vstack((rots_actual, rots_command))

        # Scaling for arrows
        base_scale = torch.tensor([0.25, 0.25, 0.5], device=self.device)
        command_scale = (1 + torch.abs(self.command[:, 4])).unsqueeze(1) * base_scale
        actual_scale = (1 + torch.abs(ang_speed)).unsqueeze(1) * base_scale
        arrow_scales = torch.vstack((actual_scale, command_scale))

        # Knees
        left_knee_loc = self._get_left_knee_location()
        right_knee_loc = self._get_right_knee_location()
        scales_knee = torch.ones_like(left_knee_loc, device=self.device) * 0.4
        left_hip_rots = self.robot.data.body_state_w[:, self.left_hip_idx[0], 3:7]
        right_hip_rots = self.robot.data.body_state_w[:, self.right_hip_idx[0], 3:7]

        # Marker indices for knees
        num_envs = self.num_envs
        left_knee_indices = torch.full((num_envs,), 2, device=self.device, dtype=torch.long)
        right_knee_indices = torch.full((num_envs,), 2, device=self.device, dtype=torch.long)

        # Check for low knee condition when not sitting
        is_walking_command = self.command[:, 0] == 0

        # Left knee
        left_knee_low = (left_knee_loc[:, 2] < 0.1) & is_walking_command
        left_knee_indices[left_knee_low] = 3  # index for low_knee marker

        # Right knee
        right_knee_low = (right_knee_loc[:, 2] < 0.1) & is_walking_command
        right_knee_indices[right_knee_low] = 3  # index for low_knee marker

        # Feet
        left_foot_loc, left_foot_rots = self._get_left_foot_location()
        right_foot_loc, right_foot_rots = self._get_right_foot_location()
        scales_foot = torch.ones_like(left_foot_loc, device=self.device) * 0.4

        # Marker indices for feet
        left_foot_indices = torch.full((num_envs,), 2, device=self.device, dtype=torch.long)
        right_foot_indices = torch.full((num_envs,), 2, device=self.device, dtype=torch.long)

        # Check for low foot condition when not sitting
        left_foot_low = (left_foot_loc[:, 2] < 0.05) & is_walking_command
        left_foot_indices[left_foot_low] = 3  # index for low_knee marker (re-using for low foot)

        right_foot_low = (right_foot_loc[:, 2] < 0.05) & is_walking_command
        right_foot_indices[right_foot_low] = 3  # index for low_knee marker (re-using for low foot)

        # Feet Velocity Markers
        left_foot_vel = self._get_left_foot_velocity()
        right_foot_vel = self._get_right_foot_velocity()

        left_foot_vel_hor = left_foot_vel[:, :2]
        right_foot_vel_hor = right_foot_vel[:, :2]

        left_foot_speed_hor = torch.norm(left_foot_vel_hor, dim=-1)
        right_foot_speed_hor = torch.norm(right_foot_vel_hor, dim=-1)

        # Rotations for velocity arrows
        up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        left_foot_angle = torch.atan2(left_foot_vel_hor[:, 1], left_foot_vel_hor[:, 0])
        right_foot_angle = torch.atan2(right_foot_vel_hor[:, 1], right_foot_vel_hor[:, 0])

        left_foot_vel_rots = math_utils.quat_from_angle_axis(left_foot_angle, up_vec)
        right_foot_vel_rots = math_utils.quat_from_angle_axis(right_foot_angle, up_vec)

        # Scales for velocity arrows
        foot_arrow_base_scale = torch.tensor([1.0, 0.2, 0.2], device=self.device)
        left_foot_vel_scales = foot_arrow_base_scale.unsqueeze(0).expand(num_envs, 3).clone()
        left_foot_vel_scales[:, 0] *= left_foot_speed_hor

        right_foot_vel_scales = foot_arrow_base_scale.unsqueeze(0).expand(num_envs, 3).clone()
        right_foot_vel_scales[:, 0] *= right_foot_speed_hor

        # Indices for velocity arrows
        left_foot_vel_indices = torch.where(left_foot_speed_hor > 0.1, 5.0, 4.0)
        right_foot_vel_indices = torch.where(right_foot_speed_hor > 0.1, 5.0, 4.0)

        # Stack all marker locations, rotations, and scales
        loc = torch.vstack((arrow_loc, left_knee_loc, right_knee_loc, head_loc, left_foot_loc, right_foot_loc, left_foot_loc, right_foot_loc))
        rots = torch.vstack((arrow_rots, left_hip_rots, right_hip_rots, head_rots, left_foot_rots, right_foot_rots, left_foot_vel_rots, right_foot_vel_rots))
        scales = torch.vstack((arrow_scales, scales_knee, scales_knee, scales_knee, scales_foot, scales_foot, left_foot_vel_scales, right_foot_vel_scales))

        # Marker indices: 0=speed, 1=command, 2=knee, 3=low_knee, 4=foot_vel_ok, 5=foot_vel_bad
        marker_indices = torch.hstack((
            torch.zeros(num_envs, device=self.device),  # speed arrow
            torch.ones(num_envs, device=self.device),  # command arrow
            left_knee_indices,  # left knee
            right_knee_indices,  # right knee
            2 * torch.ones(num_envs, device=self.device),  # head
            left_foot_indices,  # left foot
            right_foot_indices,  # right foot
            left_foot_vel_indices,  # left foot vel
            right_foot_vel_indices,  # right foot vel
        ))
        self.visualization_markers.visualize(loc, rots, marker_indices=marker_indices, scales=scales)


def main():
    # Setup simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Setup scene
    scene_cfg = RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Load policy or agent
    policy = None
    runner_skrl = None
    experiment_cfg = None
    
    if args_cli.policy:
        print(f"[INFO] Loading TorchScript policy from: {args_cli.policy}")
        policy = torch.jit.load(args_cli.policy).to(sim.device)
        policy.eval()

    if args_cli.checkpoint and args_cli.task:
        print(f"[INFO] Loading SKRL agent from: {args_cli.checkpoint}")
        # Load config
        task_name = args_cli.task.split(":")[-1]
        try:
            experiment_cfg = load_cfg_from_registry(task_name, f"skrl_{args_cli.algorithm.lower()}_cfg_entry_point")
        except ValueError:
            experiment_cfg = load_cfg_from_registry(task_name, "skrl_cfg_entry_point")
        
    if policy is None and experiment_cfg is None:
        raise ValueError("Either --policy or --checkpoint and --task must be provided.")

    # Update config based on args
    phys_sps = 1.0 / args_cli.sim_dt
    decimation = args_cli.decimation
    env_cfg = CbriisaaclabEnvCfg()
    env_cfg.command_info_cfg = {
        'sit_min': phys_sps / decimation * 1,
        'sit_max': phys_sps / decimation * 2,
        'walk_min': phys_sps / decimation * 7,
        'walk_max': phys_sps / decimation * 13,
        'speed_min': phys_sps / decimation * 2,
        'speed_max': phys_sps / decimation * 9,
    }

    # Reset simulation to initialize assets
    sim.reset()

    # Initialize runner helper (encapsulating env logic)
    runner = CBRIPolicyRunner(scene, env_cfg)

    # Initialize SKRL agent if needed
    if runner_skrl is None and experiment_cfg is not None:
        dummy_env = SkrlDummyEnv(runner.num_envs, env_cfg.observation_space, env_cfg.action_space, runner.device)
        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        runner_skrl = Runner(dummy_env, experiment_cfg)
        runner_skrl.agent.load(args_cli.checkpoint)
        runner_skrl.agent.set_running_mode("eval")

    runner.reset()

    sim_time = 0.0
    while simulation_app.is_running():
        # Get observations
        obs = runner.get_observations()
        
        actions_policy = None
        actions_skrl = None

        # Run policy
        with torch.no_grad():
            if policy is not None:
                actions_policy = policy(obs)
            
            if runner_skrl is not None:
                # SKRL agent act returns (actions, log_prob, outputs)
                outputs = runner_skrl.agent.act(obs, timestep=0, timesteps=0)
                actions_skrl = outputs[-1].get("mean_actions", outputs[0])
        
        # Select actions to apply (prefer policy if available)
        if actions_policy is not None:
            actions = actions_policy
        elif actions_skrl is not None:
            actions = actions_skrl
        
        # Apply actions
        runner.apply_actions(actions)

        # Print actions and joint positions
        actuated_ids = [
            runner.body_right_hip_dof_name_idx[0],
            runner.body_left_hip_dof_name_idx[0],
            runner.right_hip_shin_dof_name_idx[0],
            runner.left_hip_shin_dof_name_idx[0]
        ]
        tar_deg = np.degrees(runner.targets[0].cpu().numpy())
        pos_deg = np.degrees(runner.joint_pos[0, actuated_ids].cpu().numpy())
        tar_str = np.array2string(tar_deg, precision=4, separator=', ', suppress_small=True)
        pos_str = np.array2string(pos_deg, precision=4, separator=', ', suppress_small=True)
        print(f"Targets (deg): {tar_str} | Joint Pos (deg): {pos_str}")

        if actions_policy is not None and actions_skrl is not None:
            pol_str = np.array2string(actions_policy[0].cpu().numpy(), precision=4, separator=', ', suppress_small=True)
            skrl_str = np.array2string(actions_skrl[0].cpu().numpy(), precision=4, separator=', ', suppress_small=True)
            print(f"Policy Actions: {pol_str}")
            print(f"SKRL Actions:   {skrl_str}")

        # Step physics
        for _ in range(args_cli.decimation):
            scene.write_data_to_sim()
            sim.step()
            scene.update(args_cli.sim_dt)
            runner.visualize()
            sim_time += args_cli.sim_dt

        if sim_time >= 10.0:
            sim_time = 0.0
            runner.reset()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()