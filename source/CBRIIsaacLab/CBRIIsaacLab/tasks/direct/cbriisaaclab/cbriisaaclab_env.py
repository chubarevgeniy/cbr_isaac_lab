# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import sample_gaussian, sample_uniform

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
        self.left_foot_contact_idx, _ = self.feet_contact_sensor.find_sensors("left_shin")
        self.right_foot_contact_idx, _ = self.feet_contact_sensor.find_sensors("right_shin")
        self.feet_contact_ids = [self.left_foot_contact_idx[0], self.right_foot_contact_idx[0]]

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
        self.joint_pos = self.robot.data.joint_pos.torch
        self.joint_vel = self.robot.data.joint_vel.torch

        # Constant tensors used in every environment step. Creating them inside the
        # kinematics helpers launches extra CUDA work and allocator operations.
        self._head_offset = torch.tensor(self.cfg.head_offset_from_torso_loc, device=self.device)
        self._left_foot_offset = torch.tensor(self.cfg.left_foot_offset_from_shin_loc, device=self.device)
        self._right_foot_offset = torch.tensor(self.cfg.right_foot_offset_from_shin_loc, device=self.device)
        self._up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self._marker_base_scale = torch.tensor([0.25, 0.25, 0.5], device=self.device)
        self._foot_arrow_base_scale = torch.tensor([1.0, 0.2, 0.2], device=self.device)
        self._sit_reset_command = get_command(
            device=self.device,
            sit_time=self.cfg.command_info_cfg['sit_min'] // 2,
        )

        # Initialize command handling
        self.command = torch.zeros((self.num_envs, 5), device=self.device)
        self.command[:] = self._sit_reset_command
        # Setup visualization for commands.
        self.visualization_markers = define_markers()
        self.marker_offset = torch.zeros((self.num_envs, 3), device=self.device)
        self.marker_offset[:, -1] = 0.5  # Offset for visualization

        self.actions = torch.zeros((self.num_envs, 4), device=self.device)
        self.targets = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_delta = torch.zeros((self.num_envs, 4), device=self.device)

    def _setup_scene(self):
        # Initialize the robot
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # Contact forces are read from the shin rigid bodies (the feet are part of them).
        self.feet_contact_sensor = ContactSensor(self.cfg.feet_contact_sensor_cfg)
        self.scene.sensors["feet_contact_sensor"] = self.feet_contact_sensor

        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Clone environments after registering assets and sensors.
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

    def update_and_sample_commands(self):
        # update timers
        self.command[:, 1:4].add_(1)

        # from sit to standing
        sit_long_idx = (self.command[:,1] >= self.cfg.command_info_cfg['sit_min']) & (self.command[:,0] == 1)
        prob_to_stand = (self.command[:,1] - self.cfg.command_info_cfg['sit_min'])/(self.cfg.command_info_cfg['sit_max'] - self.cfg.command_info_cfg['sit_min'])
        commands_to_change = (torch.rand(self.num_envs, device=self.device) < prob_to_stand) & sit_long_idx
        self.command[commands_to_change,0] = 0
        self.command[commands_to_change,1] = 0
        self.command[commands_to_change,2] = 0
        self.command[commands_to_change,3] = 0
        self.command[commands_to_change,4] = 0

        #from standing to sit
        walk_long_idx = (self.command[:,2] >= self.cfg.command_info_cfg['walk_min']) & (self.command[:,0] == 0)
        prob_to_sit = (self.command[:,2] - self.cfg.command_info_cfg['walk_min'])/(self.cfg.command_info_cfg['walk_max'] - self.cfg.command_info_cfg['walk_min'])
        commands_to_change = (torch.rand(self.num_envs, device=self.device) < prob_to_sit) & walk_long_idx
        self.command[commands_to_change,0] = 1
        self.command[commands_to_change,1] = 0
        self.command[commands_to_change,2] = 0
        self.command[commands_to_change,3] = 0
        self.command[commands_to_change,4] = 0

        #set speed for long walking
        speed_long_idx = (self.command[:,3] >= self.cfg.command_info_cfg['speed_min']) & (self.command[:,0] == 0)
        prob_to_speed = (self.command[:,3] - self.cfg.command_info_cfg['speed_min'])/(self.cfg.command_info_cfg['speed_max'] - self.cfg.command_info_cfg['speed_min'])
        # if it is alrady long standing but speed min is large it is allowed to set new target speed
        commands_to_change = speed_long_idx & (torch.rand(self.num_envs, device=self.device) < prob_to_speed)
        sampled_speeds = sample_uniform(-1.5, 1.5, (self.num_envs,), self.device)
        self.command[:, 3].masked_fill_(commands_to_change, 0.0)
        self.command[:, 4] = torch.where(commands_to_change, sampled_speeds, self.command[:, 4])

    def _pre_physics_step(self, actions):
        # Interpret the policy output as an absolute normalized target. The
        # target is mapped directly into the joint limits; smoothness is learned
        # through the quadratic target-change penalty instead of a hard limiter.
        actions = actions.clone().clamp(-1.0, 1.0)
        limits = self.robot.data.soft_joint_pos_limits.torch[:, self.actuated_dof_indices]
        desired_targets = limits[..., 0] + 0.5 * (actions + 1.0) * (limits[..., 1] - limits[..., 0])
        self.target_delta.copy_(desired_targets)
        self.target_delta.sub_(self.targets)
        self.targets.copy_(desired_targets)
        if self._should_visualize_markers():
            self._visualize_markers()

    def _should_visualize_markers(self) -> bool:
        """Return whether an active, unpaused visualizer needs marker data."""
        if not self.render_enabled:
            return False
        return any(
            visualizer.supports_markers()
            and getattr(visualizer.cfg, "enable_markers", True)
            and not visualizer.is_rendering_paused()
            for visualizer in self.sim.visualizers
        )

    def _get_left_knee_location(self) -> torch.Tensor:
        left_knee_loc = self.robot.data.body_link_pose_w.torch[:, self.left_knee_idx[0], :3]
        return left_knee_loc

    def _get_right_knee_location(self) -> torch.Tensor:
        right_knee_loc = self.robot.data.body_link_pose_w.torch[:, self.right_knee_idx[0], :3]
        return right_knee_loc

    def _get_top_torso_location(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torso_pose_w = self.robot.data.body_link_pose_w.torch
        torso_pose = (
            torso_pose_w[:, self.body_idx[0]]
            if env_ids is None
            else torso_pose_w[env_ids, self.body_idx[0]]
        )
        torso_loc = torso_pose[:, :3]
        torso_rots = torso_pose[:, 3:7]
        top_torso_loc = torso_loc + math_utils.quat_apply(torso_rots, self._head_offset.expand_as(torso_loc))
        return top_torso_loc, torso_rots

    def _get_foot_state(
        self,
        body_idx: int,
        local_offset: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute foot position, rotation and velocity with one quaternion transform."""
        body_pose_w = self.robot.data.body_link_pose_w.torch
        body_velocity_w = self.robot.data.body_link_vel_w.torch
        if env_ids is None:
            foot_pose = body_pose_w[:, body_idx]
            body_velocity = body_velocity_w[:, body_idx]
        else:
            foot_pose = body_pose_w[env_ids, body_idx]
            body_velocity = body_velocity_w[env_ids, body_idx]
        foot_rots = foot_pose[:, 3:7]
        offset_world = math_utils.quat_apply(foot_rots, local_offset.expand(foot_rots.shape[0], -1))
        foot_location = foot_pose[:, :3] + offset_world

        foot_velocity = body_velocity[:, :3] + torch.cross(body_velocity[:, 3:6], offset_world, dim=-1)
        return foot_location, foot_rots, foot_velocity

    def _visualize_markers(self):
        visible_env_ids = self._get_marker_env_ids()
        if visible_env_ids is None:
            env_selection = slice(None)
            num_envs = self.num_envs
        else:
            if visible_env_ids.numel() == 0:
                return
            env_selection = visible_env_ids
            num_envs = visible_env_ids.numel()

        # Arrow locations for command and speed visualization (not true torso top/bottom)
        torso_base_loc = self.robot.data.body_link_pose_w.torch[env_selection, self.body_idx[0], :3]
        marker_offset = self.marker_offset[env_selection]
        arrow_loc = torch.vstack((torso_base_loc + marker_offset * 1.1, torso_base_loc + marker_offset))
        head_loc, head_rots = self._get_top_torso_location(visible_env_ids)

        # Rotation for arrows
        ang_speed = self.joint_vel[env_selection, self.base_rotor_dof_name_idx[0]]
        base_angle = -self.joint_pos[env_selection, self.base_rotor_dof_name_idx[0]]
        command_speed = self.command[env_selection, 4]
        rots_actual = math_utils.quat_from_angle_axis(base_angle - torch.pi/2 - torch.sign(ang_speed)*torch.pi/2, self._up_vec)
        rots_command = math_utils.quat_from_angle_axis(base_angle - torch.pi/2 - torch.sign(command_speed)*torch.pi/2, self._up_vec)
        arrow_rots = torch.vstack((rots_actual, rots_command))

        # Scaling for arrows
        command_scale = (1 + torch.abs(command_speed)).unsqueeze(1) * self._marker_base_scale
        actual_scale = (1 + torch.abs(ang_speed)).unsqueeze(1) * self._marker_base_scale
        arrow_scales = torch.vstack((actual_scale, command_scale))

        # Knees
        body_pose_w = self.robot.data.body_link_pose_w.torch
        left_knee_loc = body_pose_w[env_selection, self.left_knee_idx[0], :3]
        right_knee_loc = body_pose_w[env_selection, self.right_knee_idx[0], :3]
        scales_knee = torch.ones_like(left_knee_loc, device=self.device) * 0.4
        left_hip_rots = body_pose_w[env_selection, self.left_hip_idx[0], 3:7]
        right_hip_rots = body_pose_w[env_selection, self.right_hip_idx[0], 3:7]
        
        # Marker indices for knees
        left_knee_indices = torch.full((num_envs,), 2, device=self.device, dtype=torch.long)
        right_knee_indices = torch.full((num_envs,), 2, device=self.device, dtype=torch.long)

        # Check for low knee condition when not sitting
        is_walking_command = self.command[env_selection, 0] == 0
        
        # Left knee
        left_knee_low = (left_knee_loc[:, 2] < 0.1) & is_walking_command
        left_knee_indices[left_knee_low] = 3 # index for low_knee marker

        # Right knee
        right_knee_low = (right_knee_loc[:, 2] < 0.1) & is_walking_command
        right_knee_indices[right_knee_low] = 3 # index for low_knee marker

        # Feet
        left_foot_loc, left_foot_rots, left_foot_vel = self._get_foot_state(
            self.left_knee_idx[0], self._left_foot_offset, visible_env_ids
        )
        right_foot_loc, right_foot_rots, right_foot_vel = self._get_foot_state(
            self.right_knee_idx[0], self._right_foot_offset, visible_env_ids
        )
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
        left_foot_vel_hor = left_foot_vel[:, :2]
        right_foot_vel_hor = right_foot_vel[:, :2]
        
        left_foot_speed_hor = torch.norm(left_foot_vel_hor, dim=-1)
        right_foot_speed_hor = torch.norm(right_foot_vel_hor, dim=-1)
        
        # Rotations for velocity arrows
        left_foot_angle = torch.atan2(left_foot_vel_hor[:, 1], left_foot_vel_hor[:, 0])
        right_foot_angle = torch.atan2(right_foot_vel_hor[:, 1], right_foot_vel_hor[:, 0])
        
        left_foot_vel_rots = math_utils.quat_from_angle_axis(left_foot_angle, self._up_vec)
        right_foot_vel_rots = math_utils.quat_from_angle_axis(right_foot_angle, self._up_vec)
        
        # Scales for velocity arrows
        left_foot_vel_scales = self._foot_arrow_base_scale.unsqueeze(0).expand(num_envs, 3).clone()
        left_foot_vel_scales[:, 0] *= left_foot_speed_hor
        
        right_foot_vel_scales = self._foot_arrow_base_scale.unsqueeze(0).expand(num_envs, 3).clone()
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

    def _get_marker_env_ids(self) -> torch.Tensor | None:
        """Return the environment IDs selected by the active marker visualizer.

        ``None`` means that every environment should be visualized. The cap-only,
        non-random case is resolved here because ``BaseVisualizer`` represents it as
        ``env_ids=None`` plus ``max_visible_envs``.
        """
        for visualizer in self.sim.visualizers:
            if not visualizer.supports_markers():
                continue

            env_ids = visualizer.get_visualized_env_ids()
            max_visible_envs = getattr(visualizer.cfg, "max_visible_envs", None)

            if env_ids is None:
                if max_visible_envs is None:
                    return None
                env_ids = range(min(max(int(max_visible_envs), 0), self.num_envs))
            elif max_visible_envs is not None:
                env_ids = env_ids[:max(int(max_visible_envs), 0)]

            return torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)

        return None

    def _apply_action(self):
        self.robot.set_joint_position_target_index(target=self.targets, joint_ids=[
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
        contact_forces_w = self.feet_contact_sensor.data.net_forces_w_history.torch
        feet_contact = (
            torch.linalg.vector_norm(contact_forces_w[:, :, self.feet_contact_ids, :], dim=-1).amax(dim=1)
            > self.cfg.feet_contact_force_threshold
        )
        left_foot_contact = feet_contact[:, 0]
        right_foot_contact = feet_contact[:, 1]
        left_foot_location, _, left_foot_velocity = self._get_foot_state(
            self.left_knee_idx[0], self._left_foot_offset
        )
        right_foot_location, _, right_foot_velocity = self._get_foot_state(
            self.right_knee_idx[0], self._right_foot_offset
        )

        rewards, feet_slide_penalty, feet_clearance_reward = compute_rewards(
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
            left_foot_location=left_foot_location,
            right_foot_location=right_foot_location,
            left_foot_vel=left_foot_velocity,
            right_foot_vel=right_foot_velocity,
            left_foot_contact=left_foot_contact,
            right_foot_contact=right_foot_contact,
            reset_terminated=self.reset_terminated,
            command=self.command[:,[0,4]],
            target_delta=self.target_delta,
            target_change_penalty_scale=self.cfg.target_change_penalty_scale,
            feet_slide_penalty_scale=self.cfg.feet_slide_penalty_scale,
            feet_clearance_reward_scale=self.cfg.feet_clearance_reward_scale,
            feet_clearance_height_scale=self.cfg.feet_clearance_height_scale,
            feet_clearance_speed_scale=self.cfg.feet_clearance_speed_scale,
            moving_command_threshold=self.cfg.moving_command_threshold,
        )

        # SKRL calls .item() for every scalar under environment_info. Publishing
        # these values every step serializes the CUDA stream, so log them sparsely.
        self.extras.pop("log", None)
        if self.common_step_counter % self.cfg.reward_log_interval == 0:
            self.extras["log"] = {
                "Reward/feet_slide_penalty": feet_slide_penalty.mean(),
                "Reward/feet_clearance": feet_clearance_reward.mean(),
                "Metrics/left_foot_contact_rate": left_foot_contact.float().mean(),
                "Metrics/right_foot_contact_rate": right_foot_contact.float().mean(),
            }
        return rewards

    def _get_dones(self):
        # Isaac Lab 3.0 exposes simulation buffers through explicit torch views.
        self.joint_pos = self.robot.data.joint_pos.torch
        self.joint_vel = self.robot.data.joint_vel.torch

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
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()

        # Set initial command to sitting for all resetting envs
        self.command[env_ids, :] = self._sit_reset_command

        # -- Standing initial state for 70% of environments
        # Determine which envs will be standing
        stand_mask = torch.rand(num_resets, device=self.device) < 0.7
        self.command[env_ids, 0] = (~stand_mask).float()

        # Split standing environments into two poses without tensor-to-Python
        # conditions. Empty boolean selections are valid and need no special case.
        pose_a_mask = stand_mask & (torch.rand(num_resets, device=self.device) < 0.5)
        pose_b_mask = stand_mask & ~pose_a_mask

        joint_pos[pose_a_mask, self.rotor_rod_dof_name_idx] = self.cfg.default_standing_state_a['rotor_rod']
        joint_pos[pose_a_mask, self.rod_body_dof_name_idx] = self.cfg.default_standing_state_a['rod_body']
        joint_pos[pose_a_mask, self.body_right_hip_dof_name_idx] = self.cfg.default_standing_state_a['body_right_hip']
        joint_pos[pose_a_mask, self.body_left_hip_dof_name_idx] = self.cfg.default_standing_state_a['body_left_hip']
        joint_pos[pose_a_mask, self.right_hip_shin_dof_name_idx] = self.cfg.default_standing_state_a['right_hip_shin']
        joint_pos[pose_a_mask, self.left_hip_shin_dof_name_idx] = self.cfg.default_standing_state_a['left_hip_shin']

        joint_pos[pose_b_mask, self.rotor_rod_dof_name_idx] = self.cfg.default_standing_state_b['rotor_rod']
        joint_pos[pose_b_mask, self.rod_body_dof_name_idx] = self.cfg.default_standing_state_b['rod_body']
        joint_pos[pose_b_mask, self.body_right_hip_dof_name_idx] = self.cfg.default_standing_state_b['body_right_hip']
        joint_pos[pose_b_mask, self.body_left_hip_dof_name_idx] = self.cfg.default_standing_state_b['body_left_hip']
        joint_pos[pose_b_mask, self.right_hip_shin_dof_name_idx] = self.cfg.default_standing_state_b['right_hip_shin']
        joint_pos[pose_b_mask, self.left_hip_shin_dof_name_idx] = self.cfg.default_standing_state_b['left_hip_shin']

        # Apply initial tilt variation to all resetting envs
        joint_pos[:, self.rod_body_dof_name_idx] += sample_uniform(
            -self.cfg.initial_tilt_angle_variation,
            self.cfg.initial_tilt_angle_variation,
            joint_pos[:, self.rod_body_dof_name_idx].shape,
            joint_pos.device,
        )

        default_root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        default_root_vel = self.robot.data.default_root_vel.torch[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        self.targets[env_ids] = joint_pos[:, self.actuated_dof_indices]
        self.actions[env_ids] = 0.0
        self.target_delta[env_ids] = 0.0

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
    left_foot_contact: torch.Tensor,
    right_foot_contact: torch.Tensor,
    reset_terminated: torch.Tensor,
    command: torch.Tensor,
    target_delta: torch.Tensor,
    target_change_penalty_scale: float,
    feet_slide_penalty_scale: float,
    feet_clearance_reward_scale: float,
    feet_clearance_height_scale: float,
    feet_clearance_speed_scale: float,
    moving_command_threshold: float,
):
    # command[:, 0] is the sit/stand command (1 for sit, 0 for walk)
    # command[:, 1] is the target speed
    is_sitting_command = command[:, 0] == 1

    # Common rewards/penalties for all envs
    termination_penalty = reset_terminated.float() * -20
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

    moving_command = (~is_sitting_command) & (command[:, 1].abs() > moving_command_threshold)
    walk_reward += (~is_sitting_command & (left_knee_location[:, 2] < 0.1)).float() * -0.05
    walk_reward += (~is_sitting_command & (right_knee_location[:, 2] < 0.1)).float() * -0.05

    # Previous height-proxy approach (kept for reference, but disabled).
    # feet_drag_penalty = torch.exp(-left_foot_location[:, 2] * 15.0) * torch.norm(left_foot_vel[:, :2], dim=-1)
    # feet_drag_penalty += torch.exp(-right_foot_location[:, 2] * 15.0) * torch.norm(right_foot_vel[:, :2], dim=-1)
    # walk_reward += feet_drag_penalty * -0.03

    # Penalize horizontal motion only while a foot is physically in contact.
    left_foot_speed = torch.norm(left_foot_vel[:, :2], dim=-1)
    right_foot_speed = torch.norm(right_foot_vel[:, :2], dim=-1)
    feet_slide_penalty = (
        left_foot_contact.float() * left_foot_speed
        + right_foot_contact.float() * right_foot_speed
    )
    walk_reward -= feet_slide_penalty_scale * feet_slide_penalty

    # Reward upward motion of the swing foot above the stance foot. Relative height
    # makes the term work on uneven terrain. tanh has no hard target/cutoff, so a
    # higher lift is always worth a little more without making the reward unbounded.
    single_support = torch.logical_xor(left_foot_contact, right_foot_contact)
    swing_foot_height = torch.where(
        left_foot_contact,
        right_foot_location[:, 2] - left_foot_location[:, 2],
        left_foot_location[:, 2] - right_foot_location[:, 2],
    )
    swing_foot_up_velocity = torch.where(
        left_foot_contact,
        right_foot_vel[:, 2],
        left_foot_vel[:, 2],
    ).clamp(min=0.0)
    clearance_score = torch.tanh(
        swing_foot_height.clamp(min=0.0) / feet_clearance_height_scale
    )
    lift_motion = torch.tanh(feet_clearance_speed_scale * swing_foot_up_velocity)
    feet_clearance_reward = (
        moving_command.float() * single_support.float() * clearance_score * lift_motion
    )
    walk_reward += feet_clearance_reward_scale * feet_clearance_reward

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

    # Penalize target changes, not the absolute desired target. Since the policy
    # step is fixed, delta-target squared is proportional to commanded speed^2.
    target_change_penalty = (
        torch.sum(target_delta ** 2, dim=-1) * -target_change_penalty_scale
    )

    # Select the appropriate reward based on the command
    total_reward = torch.where(is_sitting_command, sit_reward*0.5, walk_reward)

    # Add common rewards
    total_reward += alive_reward + termination_penalty + target_change_penalty
    return total_reward, feet_slide_penalty, feet_clearance_reward

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
