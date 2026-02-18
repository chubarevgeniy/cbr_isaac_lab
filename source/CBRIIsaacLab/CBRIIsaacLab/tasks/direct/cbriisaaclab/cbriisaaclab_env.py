# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch

from isaaclab.markers.visualization_markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, sample_gaussian
import isaaclab.utils.math as math_utils  # <-- add import for math utils


from .cbriisaaclab_env_cfg import CbriisaaclabEnvCfg


class CbriisaaclabEnv(DirectRLEnv):
    cfg: CbriisaaclabEnvCfg

    def __init__(self, cfg: CbriisaaclabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.base_rotor_dof_name_idx, _ = self.robot.find_joints(self.cfg.base_rotor_dof_name)
        self.rotor_rod_dof_name_idx, _ = self.robot.find_joints(self.cfg.rotor_rod_dof_name)
        self.rod_body_dof_name_idx, _ = self.robot.find_joints(self.cfg.rod_body_dof_name)
        self.body_right_hip_dof_name_idx, _ = self.robot.find_joints(self.cfg.body_right_hip_dof_name)
        self.body_left_hip_dof_name_idx, _ = self.robot.find_joints(self.cfg.body_left_hip_dof_name)
        self.right_hip_shin_dof_name_idx, _ = self.robot.find_joints(self.cfg.right_hip_shin_dof_name)
        self.left_hip_shin_dof_name_idx, _ = self.robot.find_joints(self.cfg.left_hip_shin_dof_name)
        self.body_idx,_ = self.robot.find_bodies('body')
        self.left_hip_idx,_ = self.robot.find_bodies('left_hip')
        self.right_hip_idx,_ = self.robot.find_bodies('right_hip')
        self.left_knee_idx,_ = self.robot.find_bodies('left_shin')
        self.right_knee_idx,_ = self.robot.find_bodies('right_shin')

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

        # Pre-compute indices for observations to avoid fragile slicing
        self.obs_joint_pos_indices = torch.tensor(
            [i for i in range(self.robot.num_joints) if i != self.base_rotor_dof_name_idx[0]],
            device=self.device
        )
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Initialize command handling
        self.command = torch.zeros((self.cfg.scene.num_envs,5), device=self.device)
        self.command[:,[0,1,2,3,4]] = get_command(device = self.device,sit_time=self.cfg.command_info_cfg['sit_min']//2)
        # Setup visualization for commands.
        self.visualization_markers = define_markers()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)
        self.marker_offset[:, -1] = 0.5  # Offset for visualization

        self.actions = torch.zeros((self.cfg.scene.num_envs, 4), device=self.device)
        self.targets = torch.zeros((self.cfg.scene.num_envs, 4), device=self.device)

    def _setup_scene(self):
        # Initialize the robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Add robot to the scene
        self.scene.articulations["robot"] = self.robot

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

    def update_and_sample_commands(self):
        # update timers
        self.command[:,[1,2,3]] += 1

        # from sit to standing
        sit_long_idx = (self.command[:,1] >= self.cfg.command_info_cfg['sit_min']) & (self.command[:,0] == 1)
        prob_to_stand = (self.command[:,1] - self.cfg.command_info_cfg['sit_min'])/(self.cfg.command_info_cfg['sit_max'] - self.cfg.command_info_cfg['sit_min'])
        commands_to_change = (torch.rand(self.cfg.scene.num_envs, device=self.device) < prob_to_stand) & sit_long_idx
        self.command[commands_to_change,0] = 0
        self.command[commands_to_change,1] = 0
        self.command[commands_to_change,2] = 0
        self.command[commands_to_change,3] = 0
        self.command[commands_to_change,4] = 0

        #from standing to sit
        walk_long_idx = (self.command[:,2] >= self.cfg.command_info_cfg['walk_min']) & (self.command[:,0] == 0)
        prob_to_sit = (self.command[:,2] - self.cfg.command_info_cfg['walk_min'])/(self.cfg.command_info_cfg['walk_max'] - self.cfg.command_info_cfg['walk_min'])
        commands_to_change = (torch.rand(self.cfg.scene.num_envs, device=self.device) < prob_to_sit) & walk_long_idx
        self.command[commands_to_change,0] = 1
        self.command[commands_to_change,1] = 0
        self.command[commands_to_change,2] = 0
        self.command[commands_to_change,3] = 0
        self.command[commands_to_change,4] = 0

        #set speed for long walking
        speed_long_idx = (self.command[:,3] >= self.cfg.command_info_cfg['speed_min']) & (self.command[:,0] == 0)
        prob_to_speed = (self.command[:,3] - self.cfg.command_info_cfg['speed_min'])/(self.cfg.command_info_cfg['speed_max'] - self.cfg.command_info_cfg['speed_min'])
        # if it is alrady long standing but speed min is large it is allowed to set new target speed
        commands_to_change = speed_long_idx & (torch.rand(self.cfg.scene.num_envs, device=self.device) < prob_to_speed)
        commands_to_change_number = int(commands_to_change.sum().item())
        if(commands_to_change_number>0):
            self.command[commands_to_change,3] = 0
            self.command[commands_to_change,4] = sample_uniform(-1.5,1.5,(commands_to_change_number,),self.device)

    def _pre_physics_step(self, actions):
        self.actions = actions.clone()
        scaled_actions = self._scale_actions(actions)
        self.targets += scaled_actions
        limits = self.robot.data.soft_joint_pos_limits[:, self.actuated_dof_indices]
        self.targets = torch.clamp(self.targets, min=limits[..., 0], max=limits[..., 1])
        self._visualize_markers()

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

    def _visualize_markers(self):
        # Arrow locations for command and speed visualization (not true torso top/bottom)
        torso_base_loc = self.robot.data.body_state_w[:, self.body_idx[0], :3]
        arrow_loc = torch.vstack((torso_base_loc + self.marker_offset * 1.1, torso_base_loc + self.marker_offset))
        head_loc, head_rots = self._get_top_torso_location()

        # Rotation for arrows
        ang_speed = self.joint_vel[:, self.base_rotor_dof_name_idx[0]]
        base_angle = -self.joint_pos[:, self.base_rotor_dof_name_idx[0]]
        up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        rots_actual = math_utils.quat_from_angle_axis(base_angle - torch.pi/2 - torch.sign(ang_speed)*torch.pi/2, up_vec)
        rots_command = math_utils.quat_from_angle_axis(base_angle - torch.pi/2 - torch.sign(self.command[:, 4])*torch.pi/2, up_vec)
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
        num_envs = self.cfg.scene.num_envs
        left_knee_indices = torch.full((num_envs,), 2, device=self.device, dtype=torch.long)
        right_knee_indices = torch.full((num_envs,), 2, device=self.device, dtype=torch.long)

        # Check for low knee condition when not sitting
        is_walking_command = self.command[:, 0] == 0
        
        # Left knee
        left_knee_low = (left_knee_loc[:, 2] < 0.1) & is_walking_command
        left_knee_indices[left_knee_low] = 3 # index for low_knee marker

        # Right knee
        right_knee_low = (right_knee_loc[:, 2] < 0.1) & is_walking_command
        right_knee_indices[right_knee_low] = 3 # index for low_knee marker

        # Feet
        left_foot_loc, left_foot_rots = self._get_left_foot_location()
        right_foot_loc, right_foot_rots = self._get_right_foot_location()
        scales_foot = torch.ones_like(left_foot_loc, device=self.device) * 0.4

        # Marker indices for feet
        left_foot_indices = torch.full((num_envs,), 2, device=self.device, dtype=torch.long)
        right_foot_indices = torch.full((num_envs,), 2, device=self.device, dtype=torch.long)

        # Check for low foot condition when not sitting
        left_foot_low = (left_foot_loc[:, 2] < 0.05) & is_walking_command
        left_foot_indices[left_foot_low] = 3 # index for low_knee marker (re-using for low foot)

        right_foot_low = (right_foot_loc[:, 2] < 0.05) & is_walking_command
        right_foot_indices[right_foot_low] = 3 # index for low_knee marker (re-using for low foot)

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
            2*torch.ones(num_envs, device=self.device), # head
            left_foot_indices, # left foot
            right_foot_indices, # right foot
            left_foot_vel_indices, # left foot vel
            right_foot_vel_indices, # right foot vel
        ))
        self.visualization_markers.visualize(loc, rots, marker_indices=marker_indices, scales=scales)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.targets, joint_ids=[
            self.body_right_hip_dof_name_idx[0],
            self.body_left_hip_dof_name_idx[0],
            self.right_hip_shin_dof_name_idx[0],
            self.left_hip_shin_dof_name_idx[0],
        ])

    def _get_observations(self):
        self.update_and_sample_commands()

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        if self.cfg.add_noise:
            # Apply noise to hip and knee positions
            joint_pos[:, self.noise_hip_knee_indices] += sample_gaussian(
                0.0, self.cfg.noise_pos_hip_knee,
                joint_pos[:, self.noise_hip_knee_indices].shape, self.device
            )
            # tilt
            joint_pos[:, [self.rod_body_dof_name_idx[0]]] += sample_gaussian(
                0.0, self.cfg.noise_angle_pos,
                joint_pos[:, [self.rod_body_dof_name_idx[0]]].shape, self.device
            )
            # height
            joint_pos[:, [self.rotor_rod_dof_name_idx[0]]] += sample_gaussian(
                0.0, self.cfg.noise_height_pos,
                joint_pos[:, [self.rotor_rod_dof_name_idx[0]]].shape, self.device
            )
            # Apply noise to velocities
            joint_vel[:, self.noise_hip_knee_indices] += sample_gaussian(
                0.0, self.cfg.noise_vel_hip_knee,
                joint_pos[:, self.noise_hip_knee_indices].shape, self.device
            )
            # tilt
            joint_vel[:, [self.rod_body_dof_name_idx[0]]] += sample_gaussian(
                0.0, self.cfg.noise_angle_vel,
                joint_vel[:, [self.rod_body_dof_name_idx[0]]].shape, self.device
            )
            # height
            joint_vel[:, [self.rotor_rod_dof_name_idx[0]]] += sample_gaussian(
                0.0, self.cfg.noise_height_vel,
                joint_vel[:, [self.rotor_rod_dof_name_idx[0]]].shape, self.device
            )
            # speed
            joint_vel[:, [self.base_rotor_dof_name_idx[0]]] += sample_gaussian(
                0.0, self.cfg.noise_vel,
                joint_vel[:, [self.base_rotor_dof_name_idx[0]]].shape, self.device
            )
        
        # The base rotor is not part of the observation space for the policy
        return {
            "policy": torch.cat([
                joint_pos[:, self.obs_joint_pos_indices],
                joint_vel,
                self.command[:,[0,4]],
                self.targets,
            ], dim=-1)
        }
    
    def _get_rewards(self):
        return compute_rewards(
            body_vel=self.joint_vel[:, self.base_rotor_dof_name_idx],
            body_height=self.joint_pos[:, self.rotor_rod_dof_name_idx],
            body_angle=self.joint_pos[:, self.rod_body_dof_name_idx],
            right_hip_angle=self.joint_pos[:, self.body_right_hip_dof_name_idx],
            left_hip_angle=self.joint_pos[:, self.body_left_hip_dof_name_idx],
            right_knee_angle=self.joint_pos[:, self.right_hip_shin_dof_name_idx],
            left_knee_angle=self.joint_pos[:, self.left_hip_shin_dof_name_idx],
            right_hip_vel=self.joint_vel[:, self.body_right_hip_dof_name_idx],
            left_hip_vel=self.joint_vel[:, self.body_left_hip_dof_name_idx],
            right_knee_vel=self.joint_vel[:, self.right_hip_shin_dof_name_idx],
            left_knee_vel=self.joint_vel[:, self.left_hip_shin_dof_name_idx],
            left_knee_location=self._get_left_knee_location(),
            right_knee_location=self._get_right_knee_location(),
            left_foot_location=self._get_left_foot_location()[0],
            right_foot_location=self._get_right_foot_location()[0],
            left_foot_vel=self._get_left_foot_velocity(),
            right_foot_vel=self._get_right_foot_velocity(),
            reset_terminated=self.reset_terminated,
            command=self.command[:,[0,4]],
            actions=self.actions,
        )

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.joint_pos[:, self.rotor_rod_dof_name_idx[0]] > self.cfg.termination_rod_angle
        head_loc, head_rots = self._get_top_torso_location()
        died |= head_loc[:, 2] < self.cfg.termination_head_height
        return died, time_out

    def _reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        
        num_resets = len(env_ids)
        
        # Get default joint states
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        # Set initial command to sitting for all resetting envs
        self.command[env_ids, :] = get_command(device=self.device, sit_time=self.cfg.command_info_cfg['sit_min'] // 2)

        # -- Standing initial state for 70% of environments
        # Determine which envs will be standing
        stand_mask = torch.rand(num_resets, device=self.device) < 0.7
        stand_indices = env_ids[stand_mask]
        num_standing = len(stand_indices)

        if num_standing > 0:
            # Set standing command
            self.command[stand_indices, :] = get_command(sit=0,device=self.device)
            
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

        # Apply initial tilt variation to all resetting envs
        joint_pos[:, self.rod_body_dof_name_idx] += sample_uniform(
            -self.cfg.initial_tilt_angle_variation,
            self.cfg.initial_tilt_angle_variation,
            joint_pos[:, self.rod_body_dof_name_idx].shape,
            joint_pos.device,
        )

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.targets[env_ids] = joint_pos[:, self.actuated_dof_indices]
        self.actions[env_ids] = 0.0

    def _scale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        # Scale actions (deltas)
        actions = actions.clamp(-1, 1)
        actions[:,0] *= self.cfg.action_hip_scale
        actions[:,1] *= self.cfg.action_hip_scale
        actions[:,2] *= self.cfg.action_knee_scale
        actions[:,3] *= self.cfg.action_knee_scale
        return actions
    
@torch.jit.script
def compute_rewards(
    body_vel: torch.Tensor,
    body_height: torch.Tensor,
    body_angle: torch.Tensor,
    right_hip_angle: torch.Tensor,
    left_hip_angle: torch.Tensor,
    right_knee_angle: torch.Tensor,
    left_knee_angle: torch.Tensor,
    right_hip_vel: torch.Tensor,
    left_hip_vel: torch.Tensor,
    right_knee_vel: torch.Tensor,
    left_knee_vel: torch.Tensor,
    left_knee_location: torch.Tensor,
    right_knee_location: torch.Tensor,
    left_foot_location: torch.Tensor,
    right_foot_location: torch.Tensor,
    left_foot_vel: torch.Tensor,
    right_foot_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
    command: torch.Tensor,
    actions: torch.Tensor,
):
    # command[:, 0] is the sit/stand command (1 for sit, 0 for walk)
    # command[:, 1] is the target speed
    is_sitting_command = command[:, 0] == 1

    # Common rewards/penalties for all envs
    termination_penalty = reset_terminated.float() * -10
    alive_reward = (1.0 - reset_terminated.float()) * 0.05

    # --- Rewards for walking ---
    # Penalize deviation from target speed and encourage standing height
    walk_reward = (body_vel.squeeze(-1) - command[:, 1]).abs() * -0.15
    walk_reward += right_hip_vel.abs().squeeze(-1) * -0.00001
    walk_reward += left_hip_vel.abs().squeeze(-1) * -0.00001
    walk_reward += right_knee_vel.abs().squeeze(-1) * -0.00001
    walk_reward += left_knee_vel.abs().squeeze(-1) * -0.00001
    walk_reward += body_height.squeeze(-1) * -0.5
    walk_reward += (body_angle).abs().squeeze(dim=-1) * -0.05

    # moving_command = command[:, 1].abs() > 0.15
    walk_reward += (~is_sitting_command & (left_knee_location[:, 2] < 0.1)).float() * -0.05
    walk_reward += (~is_sitting_command & (right_knee_location[:, 2] < 0.1)).float() * -0.05

    # Penalty for feet dragging
    feet_drag_penalty = torch.exp(-left_foot_location[:, 2] * 15.0) * torch.norm(left_foot_vel[:, :2], dim=-1)
    feet_drag_penalty += torch.exp(-right_foot_location[:, 2] * 15.0) * torch.norm(right_foot_vel[:, :2], dim=-1)
    walk_reward += feet_drag_penalty * -0.03

    # Penalty for both feet on the ground when commanded to move
    # left_foot_low = (left_foot_location[:, 2] < 0.07) | (left_foot_location[:, 2] > left_knee_location[:, 2])
    # right_foot_low = (right_foot_location[:, 2] < 0.07) | (right_foot_location[:, 2] > right_knee_location[:, 2])
    # walk_reward += (moving_command & left_foot_low & right_foot_low).float() * -0.03


    # --- Rewards for sitting ---
    # Penalize any velocity to encourage being still.
    # You could also add a reward for being at a low height.
    sit_reward = (body_height-5.2 * torch.pi / 180.0).abs().squeeze(dim=-1) * -0.1
    sit_reward += body_vel.abs().squeeze(-1) * -0.1
    sit_reward += (body_angle+80.0 * torch.pi / 180.0).abs().squeeze(dim=-1) * -0.05
    sit_reward += (right_hip_angle).abs().squeeze(dim=-1) * -0.1
    sit_reward += (left_hip_angle).abs().squeeze(dim=-1) * -0.1
    sit_reward += (right_knee_angle+124.0 * torch.pi / 180.0 * 0.99).abs().squeeze(dim=-1) * -0.1
    sit_reward += (left_knee_angle-124.0 * torch.pi / 180.0 * 0.99).abs().squeeze(dim=-1) * -0.1

    # Penalty for action magnitude (energy/effort)
    action_penalty = torch.sum(actions ** 2, dim=-1) * -0.00001

    # Select the appropriate reward based on the command
    total_reward = torch.where(is_sitting_command, sit_reward*0.5, walk_reward)

    # Add common rewards
    total_reward += alive_reward + termination_penalty + action_penalty
    return total_reward

# @torch.jit.script
# def compute_rewards(
#     body_vel: torch.Tensor,
#     body_height: torch.Tensor,
#     body_angle: torch.Tensor,
#     right_hip_angle: torch.Tensor,
#     left_hip_angle: torch.Tensor,
#     right_knee_angle: torch.Tensor,
#     left_knee_angle: torch.Tensor,
#     left_knee_location: torch.Tensor,
#     right_knee_location: torch.Tensor,
#     left_foot_location: torch.Tensor,
#     right_foot_location: torch.Tensor,
#     reset_terminated: torch.Tensor,
#     command: torch.Tensor,
# ):
#     # command[:, 0] is the sit/stand command (1 for sit, 0 for walk)
#     # command[:, 1] is the target speed
#     is_sitting_command = command[:, 0] == 1

#     # Common rewards/penalties for all envs
#     termination_penalty = reset_terminated.float() * -25.0
#     alive_reward = (1.0 - reset_terminated.float()) * 0.1

#     # --- Rewards for walking ---
#     # Penalize deviation from target speed and encourage standing height
#     walk_reward = (body_vel.squeeze(-1) - command[:, 1]).abs() * -0.15
#     walk_reward += body_height.sum(dim=-1) * -0.35
#     walk_reward += (body_angle).abs().squeeze(dim=-1) * -0.05

#     moving_command = command[:, 1].abs() > 0.15
#     walk_reward += (moving_command & (left_knee_location[:, 2] < 0.1)).float() * -0.1
#     walk_reward += (moving_command & (right_knee_location[:, 2] < 0.1)).float() * -0.1

#     # Penalty for both feet on the ground when commanded to move
#     left_foot_low = (left_foot_location[:, 2] < 0.07) | (left_foot_location[:, 2] > left_knee_location[:, 2])
#     right_foot_low = (right_foot_location[:, 2] < 0.07) | (right_foot_location[:, 2] > right_knee_location[:, 2])
#     walk_reward += (moving_command & left_foot_low & right_foot_low).float() * -0.03


#     # --- Rewards for sitting ---
#     # Penalize any velocity to encourage being still.
#     # You could also add a reward for being at a low height.
#     sit_reward = (body_height-5.2 * torch.pi / 180.0).abs().squeeze(dim=-1) * -0.1
#     sit_reward += (body_angle+80.0 * torch.pi / 180.0).abs().squeeze(dim=-1) * -0.05
#     sit_reward += (right_hip_angle).abs().squeeze(dim=-1) * -0.05
#     sit_reward += (left_hip_angle).abs().squeeze(dim=-1) * -0.05
#     sit_reward += (right_knee_angle+124.0 * torch.pi / 180.0 * 0.99).abs().squeeze(dim=-1) * -0.05
#     sit_reward += (left_knee_angle-124.0 * torch.pi / 180.0 * 0.99).abs().squeeze(dim=-1) * -0.05

#     # Select the appropriate reward based on the command
#     total_reward = torch.where(is_sitting_command, sit_reward*0.5, walk_reward)

#     # Add common rewards
#     total_reward += alive_reward + termination_penalty
#     return total_reward

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


def get_command(sit: float = 1, sit_time: float = 0, walk_time: float = 0, speed_time: float = 0,speed:float = 0,device = "cpu"):
    return torch.tensor([sit, sit_time, walk_time, speed_time, speed], dtype=torch.float32, device=device)