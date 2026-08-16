# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import sample_gaussian, sample_uniform

from CBRIIsaacLab.robots.coupled_leg_actuator import CoupledLegPDActuator

from .cbriisaaclab_env_cfg import CbriisaaclabEnvCfg
from .coordinate_conventions import canonical_actuated_to_raw, raw_actuated_to_canonical
from .initial_pose_randomization import (
    InitialPoseIndices,
    apply_sitting_reset_variation,
    sample_ground_safe_initial_pose,
    sample_initial_commands,
)


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
        self.rock_idx,_ = self.robot.find_bodies('Rock')
        self.left_hip_idx,_ = self.robot.find_bodies('left_hip')
        self.right_hip_idx,_ = self.robot.find_bodies('right_hip')
        self.left_knee_idx,_ = self.robot.find_bodies('left_shin')
        self.right_knee_idx,_ = self.robot.find_bodies('right_shin')

        leg_actuator = self.robot.actuators["coupled_leg_actuator"]
        if not isinstance(leg_actuator, CoupledLegPDActuator):
            raise TypeError(
                "The 'coupled_leg_actuator' group must use CoupledLegPDActuator "
                "to expose motor-space effort telemetry."
            )
        self.leg_actuator = leg_actuator

        self.initial_pose_indices = InitialPoseIndices(
            rotor_rod=self.rotor_rod_dof_name_idx[0],
            rod_body=self.rod_body_dof_name_idx[0],
            body_right_hip=self.body_right_hip_dof_name_idx[0],
            body_left_hip=self.body_left_hip_dof_name_idx[0],
            right_hip_shin=self.right_hip_shin_dof_name_idx[0],
            left_hip_shin=self.left_hip_shin_dof_name_idx[0],
            left_shin_body=self.left_knee_idx[0],
            right_shin_body=self.right_knee_idx[0],
        )
        collision_body_ids = []
        for body_name in (
            "Rock",
            "bottom_rotor",
            "rod_1",
            "body",
            "right_hip",
            "right_shin",
            "left_hip",
            "left_shin",
        ):
            body_ids, _ = self.robot.find_bodies(body_name)
            collision_body_ids.append(body_ids[0])
        self.collision_body_indices = torch.tensor(collision_body_ids, device=self.device, dtype=torch.long)
        self.left_foot_offset = torch.tensor(self.cfg.left_foot_offset_from_shin_loc, device=self.device)
        self.right_foot_offset = torch.tensor(self.cfg.right_foot_offset_from_shin_loc, device=self.device)

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
        self._actuated_dof_indices_tensor = torch.tensor(self.actuated_dof_indices, device=self.device)
        self._histogram_joint_names = ("right_hip", "left_hip", "right_knee", "left_knee")

        # Pre-compute indices for observations to avoid fragile slicing
        self.obs_joint_pos_indices = torch.tensor(
            [i for i in range(self.robot.num_joints) if i != self.base_rotor_dof_name_idx[0]],
            device=self.device
        )
        obs_index_by_joint = {
            int(joint_index): obs_index
            for obs_index, joint_index in enumerate(self.obs_joint_pos_indices.tolist())
        }
        self.obs_rotor_rod_pos_index = obs_index_by_joint[self.rotor_rod_dof_name_idx[0]]
        self.obs_actuated_pos_indices = torch.tensor(
            [obs_index_by_joint[index] for index in self.actuated_dof_indices],
            device=self.device,
            dtype=torch.long,
        )
        self.joint_pos = self.robot.data.joint_pos.torch
        self.joint_vel = self.robot.data.joint_vel.torch
        self._observation_delay_steps = self._configure_observation_delay()
        history_length = max(self._observation_delay_steps, 1)
        self._observation_joint_pos_history = self.joint_pos.unsqueeze(0).repeat(
            history_length, 1, 1
        ).clone()
        self._observation_joint_vel_history = self.joint_vel.unsqueeze(0).repeat(
            history_length, 1, 1
        ).clone()
        self._observation_history_index = 0
        self._observation_delay_mask = torch.zeros(
            self.joint_pos.shape[0], device=self.device, dtype=torch.bool
        )

        self._histogram_writer = None
        if getattr(self.cfg, "log_dir", None):
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._histogram_writer = SummaryWriter(
                    log_dir=self.cfg.log_dir,
                    max_queue=100,
                    flush_secs=30,
                )
            except ImportError:
                print("[WARN] TensorBoard is unavailable; action histograms will not be written.")

        # Initialize command handling
        self.command = torch.zeros((self.cfg.scene.num_envs,5), device=self.device)
        self.command[:,[0,1,2,3,4]] = get_command(device = self.device,sit_time=self.cfg.command_info_cfg['sit_min']//2)
        # Setup visualization for commands.
        self.visualization_markers = define_markers()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)
        self.marker_offset[:, -1] = 0.5  # Offset for visualization

        self.actions = torch.zeros((self.cfg.scene.num_envs, 4), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.previous_previous_actions = torch.zeros_like(self.actions)
        self.targets = torch.zeros((self.cfg.scene.num_envs, 4), device=self.device)
        self._canonical_action_offset = torch.tensor(
            self.cfg.action_default_target,
            device=self.device,
        )
        self._canonical_action_scale = torch.tensor(
            [
                self.cfg.action_hip_scale,
                self.cfg.action_hip_scale,
                self.cfg.action_knee_scale,
                self.cfg.action_knee_scale,
            ],
            device=self.device,
        )
        self._canonical_target_min = torch.tensor(
            [
                self.cfg.canonical_hip_min,
                self.cfg.canonical_hip_min,
                self.cfg.canonical_knee_min,
                self.cfg.canonical_knee_min,
            ],
            device=self.device,
        )
        self._canonical_target_max = torch.tensor(
            [
                self.cfg.canonical_hip_max,
                self.cfg.canonical_hip_max,
                self.cfg.canonical_knee_max,
                self.cfg.canonical_knee_max,
            ],
            device=self.device,
        )
        self._policy_action_abs_limit = compute_policy_action_abs_limit(
            self._canonical_target_min,
            self._canonical_target_max,
            self._canonical_action_offset,
            self._canonical_action_scale,
            self.cfg.action_limit_range_margin,
        )

    def _raw_to_canonical_actuated(self, raw: torch.Tensor) -> torch.Tensor:
        return raw_actuated_to_canonical(raw, self.cfg.canonical_hip_down_angle)

    def _canonical_to_raw_actuated(self, canonical: torch.Tensor) -> torch.Tensor:
        return canonical_actuated_to_raw(canonical, self.cfg.canonical_hip_down_angle)

    def _get_action_acceleration_scale(self) -> float:
        """Return the action-acceleration coefficient for this trainer step."""

        return compute_action_acceleration_scale(
            timestep=float(self.common_step_counter),
            start_scale=float(self.cfg.rewards.action_acceleration_scale_start),
            end_scale=float(self.cfg.rewards.action_acceleration_scale_end),
            start_timestep=float(
                self.cfg.rewards.action_acceleration_schedule_start_timestep
            ),
            end_timestep=float(
                self.cfg.rewards.action_acceleration_schedule_end_timestep
            ),
        )

    def _configure_observation_delay(self) -> int:
        """Validate the configured latency and convert it to control steps."""

        mode = str(self.cfg.observation_delay_mode).strip().lower()
        if mode not in {"current", "delayed", "random"}:
            raise ValueError(
                "observation_delay_mode must be 'current', 'delayed', or 'random'; "
                f"got {self.cfg.observation_delay_mode!r}"
            )
        probability = float(self.cfg.observation_delay_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "observation_delay_probability must be in [0, 1]; "
                f"got {probability}"
            )

        delay_s = float(self.cfg.observation_delay_s)
        if delay_s < 0.0:
            raise ValueError(f"observation_delay_s must be non-negative; got {delay_s}")
        control_dt = float(self.cfg.sim.dt) * int(self.cfg.decimation)
        delay_steps_float = delay_s / control_dt
        delay_steps = int(round(delay_steps_float))
        if not math.isclose(delay_steps_float, delay_steps, rel_tol=1.0e-6, abs_tol=1.0e-6):
            raise ValueError(
                "observation_delay_s must be an integer number of policy steps; "
                f"got {delay_s} s for control dt {control_dt} s"
            )

        self._observation_delay_mode = mode
        self._observation_delay_probability = probability
        return delay_steps

    def _reset_observation_delay(
        self,
        env_ids: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> None:
        """Reset latency selection and history for newly reset environments."""

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if self._observation_delay_mode == "current":
            self._observation_delay_mask[env_ids] = False
        elif self._observation_delay_mode == "delayed":
            self._observation_delay_mask[env_ids] = True
        else:
            self._observation_delay_mask[env_ids] = (
                torch.rand(len(env_ids), device=self.device)
                < self._observation_delay_probability
            )

        # Fill every history slot so a reset cannot expose the previous
        # episode's state while the delay pipeline warms up.
        if self._observation_delay_steps > 0:
            self._observation_joint_pos_history[:, env_ids] = joint_pos.unsqueeze(0)
            self._observation_joint_vel_history[:, env_ids] = joint_vel.unsqueeze(0)

    def _get_observation_joint_state(
        self,
        current_joint_pos: torch.Tensor,
        current_joint_vel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the selected current/delayed joint state and advance history."""

        if self._observation_delay_steps == 0:
            return current_joint_pos, current_joint_vel

        delayed_joint_pos = self._observation_joint_pos_history[
            self._observation_history_index
        ].clone()
        delayed_joint_vel = self._observation_joint_vel_history[
            self._observation_history_index
        ].clone()

        if self._observation_delay_mode == "current":
            selected_joint_pos = current_joint_pos
            selected_joint_vel = current_joint_vel
        elif self._observation_delay_mode == "delayed":
            selected_joint_pos = delayed_joint_pos
            selected_joint_vel = delayed_joint_vel
        else:
            delay_mask = self._observation_delay_mask.unsqueeze(-1)
            selected_joint_pos = torch.where(delay_mask, delayed_joint_pos, current_joint_pos)
            selected_joint_vel = torch.where(delay_mask, delayed_joint_vel, current_joint_vel)

        self._observation_joint_pos_history[self._observation_history_index] = current_joint_pos
        self._observation_joint_vel_history[self._observation_history_index] = current_joint_vel
        self._observation_history_index = (
            self._observation_history_index + 1
        ) % self._observation_delay_steps
        return selected_joint_pos, selected_joint_vel

    def _actions_to_canonical_targets(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert raw actions to direct canonical joint-position targets."""

        return self._canonical_action_offset + actions * self._canonical_action_scale

    def _clip_policy_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Bound actions using the farther joint limit plus a range margin."""

        return torch.clamp(
            actions,
            min=-self._policy_action_abs_limit,
            max=self._policy_action_abs_limit,
        )

    def _canonical_targets_to_actions(self, targets: torch.Tensor) -> torch.Tensor:
        """Convert canonical joint targets back to normalized policy actions."""

        return (targets - self._canonical_action_offset) / self._canonical_action_scale

    def _setup_scene(self):
        # Initialize the robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # Sparse, kinematic cuboids are cloned with the environments and moved
        # to a new layout at every reset. They are registered in the scene so
        # the common scene reset/write/update lifecycle also handles them.
        self.uneven_ground = None
        if self.cfg.uneven_ground_enabled:
            self.uneven_ground = RigidObjectCollection(self.cfg.uneven_ground_cfg)

        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Clone environments before registering assets, matching the Isaac Lab 3.0
        # direct-workflow scene setup order.
        self.scene.clone_environments(copy_from_source=False)

        # Add robot to the scene
        self.scene.articulations["robot"] = self.robot
        if self.uneven_ground is not None:
            self.scene.rigid_object_collections["uneven_ground"] = self.uneven_ground

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

    def _randomize_uneven_ground(self, env_ids: torch.Tensor) -> None:
        """Place sparse 2 cm bumps around the rotor while protecting reset poses.

        Candidate positions are sampled uniformly by area in an annulus around
        the actual Rock body. At reset we reject points close to any collision
        body in the freshly written robot pose and to previously accepted
        bumps. This keeps the initial state free of artificial penetrations;
        after reset the bumps remain fixed until the next reset.
        """

        if self.uneven_ground is None or len(env_ids) == 0:
            return

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        num_envs = len(env_ids)
        num_bumps = len(self.cfg.uneven_ground_cfg.rigid_objects)
        radial_min, radial_max = self.cfg.uneven_ground_radial_range
        min_robot_distance = float(self.cfg.uneven_ground_min_distance_to_robot)
        min_bump_distance = float(self.cfg.uneven_ground_min_distance_between_bumps)
        attempts = max(1, int(self.cfg.uneven_ground_resample_attempts))

        # ``body_link_pose_w`` is in the same world frame expected by the
        # collection write API. The pivot is taken from the actual fixed Rock
        # body rather than assuming that every USD revision uses the origin.
        rock_pose = self.robot.data.body_link_pose_w.torch[env_ids, self.rock_idx[0]]
        pivot_xy = rock_pose[:, :2]
        body_xy = self.robot.data.body_link_pose_w.torch[env_ids][:, self.collision_body_indices, :2]
        left_foot_xy = self._get_left_foot_location()[0][env_ids, :2]
        right_foot_xy = self._get_right_foot_location()[0][env_ids, :2]
        protected_xy = torch.cat(
            (body_xy, left_foot_xy.unsqueeze(1), right_foot_xy.unsqueeze(1)), dim=1
        )

        bump_xy = torch.empty(
            (num_envs, num_bumps, 2), device=self.device, dtype=rock_pose.dtype
        )
        radial_min_sq = float(radial_min) ** 2
        radial_max_sq = float(radial_max) ** 2

        for bump_index in range(num_bumps):
            radius = torch.sqrt(
                radial_min_sq
                + (radial_max_sq - radial_min_sq)
                * torch.rand((num_envs,), device=self.device, dtype=rock_pose.dtype)
            )
            angle = (2.0 * math.pi) * torch.rand(
                (num_envs,), device=self.device, dtype=rock_pose.dtype
            ) - math.pi
            candidate = pivot_xy + torch.stack(
                (radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1
            )

            for _ in range(attempts):
                protected_distance = torch.linalg.vector_norm(
                    candidate.unsqueeze(1) - protected_xy, dim=-1
                ).amin(dim=1)
                valid = protected_distance >= min_robot_distance
                if bump_index > 0:
                    accepted_distance = torch.linalg.vector_norm(
                        candidate.unsqueeze(1) - bump_xy[:, :bump_index], dim=-1
                    ).amin(dim=1)
                    valid &= accepted_distance >= min_bump_distance
                if bool(valid.all().item()):
                    break

                new_radius = torch.sqrt(
                    radial_min_sq
                    + (radial_max_sq - radial_min_sq)
                    * torch.rand((num_envs,), device=self.device, dtype=rock_pose.dtype)
                )
                new_angle = (2.0 * math.pi) * torch.rand(
                    (num_envs,), device=self.device, dtype=rock_pose.dtype
                ) - math.pi
                new_candidate = pivot_xy + torch.stack(
                    (new_radius * torch.cos(new_angle), new_radius * torch.sin(new_angle)), dim=-1
                )
                candidate = torch.where(valid.unsqueeze(-1), candidate, new_candidate)

            bump_xy[:, bump_index] = candidate

        bump_height = float(self.cfg.uneven_ground_bump_height)
        poses = torch.zeros(
            (num_envs, num_bumps, 7), device=self.device, dtype=rock_pose.dtype
        )
        poses[..., :2] = bump_xy
        poses[..., 2] = bump_height * 0.5
        yaw = (2.0 * math.pi) * torch.rand(
            (num_envs, num_bumps), device=self.device, dtype=rock_pose.dtype
        ) - math.pi
        poses[..., 5] = torch.sin(yaw * 0.5)
        poses[..., 6] = torch.cos(yaw * 0.5)

        self.uneven_ground.write_body_link_pose_to_sim_index(
            body_poses=poses,
            env_ids=env_ids,
        )

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
        self.previous_previous_actions.copy_(self.previous_actions)
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(self._clip_policy_actions(actions))
        self.targets = self._actions_to_canonical_targets(self.actions)
        self._visualize_markers()

    def _get_left_knee_location(self) -> torch.Tensor:
        left_knee_loc = self.robot.data.body_link_pose_w.torch[:, self.left_knee_idx[0], :3]
        return left_knee_loc

    def _get_right_knee_location(self) -> torch.Tensor:
        right_knee_loc = self.robot.data.body_link_pose_w.torch[:, self.right_knee_idx[0], :3]
        return right_knee_loc

    def _get_top_torso_location(self) -> torch.Tensor:
        torso_pose = self.robot.data.body_link_pose_w.torch[:, self.body_idx[0]]
        torso_loc = torso_pose[:, :3]
        torso_rots = torso_pose[:, 3:7]
        offset = torch.tensor(self.cfg.head_offset_from_torso_loc, device=self.device).expand_as(torso_loc)
        top_torso_loc = torso_loc + math_utils.quat_apply(torso_rots, offset)
        return top_torso_loc, torso_rots
    
    def _get_left_foot_location(self) -> torch.Tensor:
        foot_pose = self.robot.data.body_link_pose_w.torch[:, self.left_knee_idx[0]]
        foot_loc = foot_pose[:, :3]
        foot_rots = foot_pose[:, 3:7]
        offset = torch.tensor(self.cfg.left_foot_offset_from_shin_loc, device=self.device).expand_as(foot_loc)
        foot_offset_loc = foot_loc + math_utils.quat_apply(foot_rots, offset)
        return foot_offset_loc, foot_rots
    
    def _get_right_foot_location(self) -> torch.Tensor:
        foot_pose = self.robot.data.body_link_pose_w.torch[:, self.right_knee_idx[0]]
        foot_loc = foot_pose[:, :3]
        foot_rots = foot_pose[:, 3:7]
        offset = torch.tensor(self.cfg.right_foot_offset_from_shin_loc, device=self.device).expand_as(foot_loc)
        foot_offset_loc = foot_loc + math_utils.quat_apply(foot_rots, offset)
        return foot_offset_loc, foot_rots

    def _get_left_foot_velocity(self) -> torch.Tensor:
        shin_vel = self.robot.data.body_link_vel_w.torch[:, self.left_knee_idx[0], :3]
        shin_ang_vel = self.robot.data.body_link_vel_w.torch[:, self.left_knee_idx[0], 3:6]
        shin_rots = self.robot.data.body_link_pose_w.torch[:, self.left_knee_idx[0], 3:7]
        
        offset = torch.tensor(self.cfg.left_foot_offset_from_shin_loc, device=self.device).expand_as(shin_vel)
        offset_world = math_utils.quat_apply(shin_rots, offset)
        
        return shin_vel + torch.cross(shin_ang_vel, offset_world, dim=-1)

    def _get_right_foot_velocity(self) -> torch.Tensor:
        shin_vel = self.robot.data.body_link_vel_w.torch[:, self.right_knee_idx[0], :3]
        shin_ang_vel = self.robot.data.body_link_vel_w.torch[:, self.right_knee_idx[0], 3:6]
        shin_rots = self.robot.data.body_link_pose_w.torch[:, self.right_knee_idx[0], 3:7]
        
        offset = torch.tensor(self.cfg.right_foot_offset_from_shin_loc, device=self.device).expand_as(shin_vel)
        offset_world = math_utils.quat_apply(shin_rots, offset)
        
        return shin_vel + torch.cross(shin_ang_vel, offset_world, dim=-1)

    def _visualize_markers(self):
        # Arrow locations for command and speed visualization (not true torso top/bottom)
        torso_base_loc = self.robot.data.body_link_pose_w.torch[:, self.body_idx[0], :3]
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
        left_hip_rots = self.robot.data.body_link_pose_w.torch[:, self.left_hip_idx[0], 3:7]
        right_hip_rots = self.robot.data.body_link_pose_w.torch[:, self.right_hip_idx[0], 3:7]
        
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

        # The marker point-instancer contains nine entries per environment. Isaac Lab's
        # generic partial-visualization filter can only infer the environment mapping when
        # an instancer contains exactly one entry per environment, so apply the visualizer's
        # selected environment IDs explicitly here.
        visible_env_ids = self._get_marker_env_ids()
        if visible_env_ids is not None:
            marker_env_ids = torch.cat(
                [visible_env_ids + block * num_envs for block in range(9)]
            )
            loc = loc[marker_env_ids]
            rots = rots[marker_env_ids]
            scales = scales[marker_env_ids]
            marker_indices = marker_indices[marker_env_ids]

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
                env_ids = range(min(max(int(max_visible_envs), 0), self.cfg.scene.num_envs))
            elif max_visible_envs is not None:
                env_ids = env_ids[:max(int(max_visible_envs), 0)]

            return torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)

        return None

    def _apply_action(self):
        raw_targets = self._canonical_to_raw_actuated(self.targets)
        self.robot.set_joint_position_target_index(target=raw_targets, joint_ids=[
            self.body_right_hip_dof_name_idx[0],
            self.body_left_hip_dof_name_idx[0],
            self.right_hip_shin_dof_name_idx[0],
            self.left_hip_shin_dof_name_idx[0],
        ])

    def _get_observations(self):
        self.update_and_sample_commands()

        current_joint_pos = self.joint_pos.clone()
        current_joint_vel = self.joint_vel.clone()
        joint_pos, joint_vel = self._get_observation_joint_state(
            current_joint_pos, current_joint_vel
        )

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
        
        # Targets are direct functions of the action. Expose the two most
        # recent actions so the policy can distinguish a smooth command ramp
        # from a reversal, which is needed by the second-order action penalty.
        canonical_joint_pos = joint_pos[:, self.obs_joint_pos_indices].clone()
        canonical_joint_pos[:, self.obs_rotor_rod_pos_index] = (
            -canonical_joint_pos[:, self.obs_rotor_rod_pos_index]
        )
        canonical_joint_pos[:, self.obs_actuated_pos_indices] = self._raw_to_canonical_actuated(
            joint_pos[:, self._actuated_dof_indices_tensor]
        )

        return {
            "policy": torch.cat([
                canonical_joint_pos,
                joint_vel,
                self.command[:,[0,4]],
                self.actions,
                self.previous_actions,
            ], dim=-1)
        }
    
    def _get_rewards(self):
        # Observations keep angular velocities raw.  Reward terms that compare
        # against linear commands use one-metre tangential proxies instead.
        body_vel = (
            self.joint_vel[:, self.base_rotor_dof_name_idx]
            * self.cfg.longitudinal_velocity_proxy_lever_arm
        )
        body_height = (
            -self.joint_pos[:, self.rotor_rod_dof_name_idx]
            * self.cfg.height_proxy_lever_arm
        )
        body_vertical_vel = (
            -self.joint_vel[:, self.rotor_rod_dof_name_idx]
            * self.cfg.height_velocity_proxy_lever_arm
        )
        body_angle = self.joint_pos[:, self.rod_body_dof_name_idx]
        body_angular_vel = self.joint_vel[:, self.rod_body_dof_name_idx]
        actuated_joint_pos = self._raw_to_canonical_actuated(
            self.joint_pos[:, self._actuated_dof_indices_tensor]
        )
        actuated_joint_vel = self.joint_vel[:, self._actuated_dof_indices_tensor]
        raw_actuated_joint_pos = self.joint_pos[:, self._actuated_dof_indices_tensor]
        raw_actuated_joint_limits = self.robot.data.soft_joint_pos_limits.torch[:, self._actuated_dof_indices_tensor]
        joint_pos_limits = (
            (raw_actuated_joint_limits[..., 0] - raw_actuated_joint_pos).clamp_min(0.0)
            + (raw_actuated_joint_pos - raw_actuated_joint_limits[..., 1]).clamp_min(0.0)
        ).sum(dim=-1)
        target_joint_limit_violation = (
            (self._canonical_target_min - self.targets).clamp_min(0.0)
            + (self.targets - self._canonical_target_max).clamp_min(0.0)
        )
        motor_effort_limit = self.leg_actuator.effort_limit.clamp_min(1.0e-6)
        normalized_motor_effort = self.leg_actuator.applied_motor_effort / motor_effort_limit

        right_hip_angle = actuated_joint_pos[:, 0:1]
        left_hip_angle = actuated_joint_pos[:, 1:2]
        right_knee_angle = actuated_joint_pos[:, 2:3]
        left_knee_angle = actuated_joint_pos[:, 3:4]
        right_hip_vel = self.joint_vel[:, self.body_right_hip_dof_name_idx]
        left_hip_vel = self.joint_vel[:, self.body_left_hip_dof_name_idx]
        right_knee_vel = self.joint_vel[:, self.right_hip_shin_dof_name_idx]
        left_knee_vel = self.joint_vel[:, self.left_hip_shin_dof_name_idx]
        left_knee_location = self._get_left_knee_location()
        right_knee_location = self._get_right_knee_location()
        left_foot_location = self._get_left_foot_location()[0]
        right_foot_location = self._get_right_foot_location()[0]
        left_foot_vel = self._get_left_foot_velocity()
        right_foot_vel = self._get_right_foot_velocity()
        foot_height = torch.stack(
            (left_foot_location[:, 2], right_foot_location[:, 2]), dim=-1
        )
        foot_horizontal_speed = torch.stack(
            (
                torch.linalg.vector_norm(left_foot_vel[:, :2], dim=-1),
                torch.linalg.vector_norm(right_foot_vel[:, :2], dim=-1),
            ),
            dim=-1,
        )
        command = self.command[:, [0, 4]]
        action_acceleration_scale = self._get_action_acceleration_scale()

        rewards = compute_rewards(
            body_vel=body_vel,
            body_height=body_height,
            body_vertical_vel=body_vertical_vel,
            body_angular_vel=body_angular_vel,
            body_angle=body_angle,
            actuated_joint_pos=actuated_joint_pos,
            actuated_joint_vel=actuated_joint_vel,
            joint_pos_limits=joint_pos_limits,
            target_joint_limit_violation=target_joint_limit_violation,
            normalized_motor_effort=normalized_motor_effort,
            foot_height=foot_height,
            foot_horizontal_speed=foot_horizontal_speed,
            reset_terminated=self.reset_terminated,
            command=command,
            actions=self.actions,
            previous_actions=self.previous_actions,
            previous_previous_actions=self.previous_previous_actions,
            action_target_scale=self._canonical_action_scale,
            action_acceleration_scale=action_acceleration_scale,
            alive_reward_scale=self.cfg.rewards.alive_reward_scale,
            death_reward_scale=self.cfg.rewards.death_reward_scale,
            walk_velocity_tracking_scale=self.cfg.rewards.walk_velocity_tracking_scale,
            walk_velocity_tracking_std=self.cfg.rewards.walk_velocity_tracking_std,
            base_vertical_velocity_scale=self.cfg.rewards.base_vertical_velocity_scale,
            base_angular_velocity_scale=self.cfg.rewards.base_angular_velocity_scale,
            joint_velocity_scale=self.cfg.rewards.joint_velocity_scale,
            joint_position_limits_scale=self.cfg.rewards.joint_position_limits_scale,
            action_target_limits_scale=self.cfg.rewards.action_target_limits_scale,
            motor_effort_scale=self.cfg.rewards.motor_effort_scale,
            foot_slip_scale=self.cfg.rewards.foot_slip_scale,
            foot_slip_height_scale=self.cfg.rewards.foot_slip_height_scale,
            joint_deviation_waist_scale=self.cfg.rewards.joint_deviation_waist_scale,
            joint_deviation_legs_scale=self.cfg.rewards.joint_deviation_legs_scale,
            flat_orientation_scale=self.cfg.rewards.flat_orientation_scale,
            walk_base_height_target=self.cfg.rewards.walk_base_height_target,
            walk_base_height_scale=self.cfg.rewards.walk_base_height_scale,
            walk_body_angle_target=self.cfg.rewards.walk_body_angle_target,
            sit_body_height_target=self.cfg.rewards.sit_body_height_target,
            sit_body_height_scale=self.cfg.rewards.sit_body_height_scale,
            sit_body_angle_target=self.cfg.rewards.sit_body_angle_target,
            sit_right_hip_angle_target=self.cfg.rewards.sit_right_hip_angle_target,
            sit_left_hip_angle_target=self.cfg.rewards.sit_left_hip_angle_target,
            sit_right_knee_angle_target=self.cfg.rewards.sit_right_knee_angle_target,
            sit_left_knee_angle_target=self.cfg.rewards.sit_left_knee_angle_target,
            sit_pose_angle_multiplier=self.cfg.rewards.sit_pose_angle_multiplier,
        )

        self.extras.pop("log", None)
        if self.common_step_counter % self.cfg.metrics_log_interval == 0:
            self.extras["log"] = self._get_physical_metrics(
                body_vel=body_vel,
                body_height=body_height,
                body_angle=body_angle,
                right_hip_angle=right_hip_angle,
                left_hip_angle=left_hip_angle,
                right_knee_angle=right_knee_angle,
                left_knee_angle=left_knee_angle,
                right_hip_vel=right_hip_vel,
                left_hip_vel=left_hip_vel,
                right_knee_vel=right_knee_vel,
                left_knee_vel=left_knee_vel,
                left_knee_location=left_knee_location,
                right_knee_location=right_knee_location,
                left_foot_location=left_foot_location,
                right_foot_location=right_foot_location,
                left_foot_vel=left_foot_vel,
                right_foot_vel=right_foot_vel,
                command=command,
                normalized_motor_effort=normalized_motor_effort,
            )
        if self.common_step_counter % self.cfg.histogram_log_interval == 0:
            self._log_action_histograms()
        return rewards

    def _log_action_histograms(self):
        """Write action and target distributions to the run's TensorBoard log."""
        if self._histogram_writer is None:
            return

        with torch.no_grad():
            raw_actions = self.actions
            # Reference only: this clipped copy is not used for control or reward.
            clipped_actions_reference = raw_actions.clamp(-1.0, 1.0)
            canonical_action_target = self._actions_to_canonical_targets(raw_actions)
            unnoisy_joint_state = self._raw_to_canonical_actuated(
                self.joint_pos.index_select(1, self._actuated_dof_indices_tensor)
            )
            target_error = self.targets - unnoisy_joint_state
            applied_motor_effort = self.leg_actuator.applied_motor_effort
            motor_effort_limit = self.leg_actuator.effort_limit.clamp_min(1.0e-6)
            normalized_motor_effort = applied_motor_effort / motor_effort_limit

            distributions = {
                "action/raw": raw_actions,
                "action/clipped_reference": clipped_actions_reference,
                "action/target_canonical": canonical_action_target,
                "target/canonical": self.targets,
                "target/error_to_unnoisy_joint": target_error,
                "state/unnoisy_joint_canonical": unnoisy_joint_state,
                "motor/applied_effort": applied_motor_effort,
                "motor/applied_effort_normalized": normalized_motor_effort,
            }
            for name, values in distributions.items():
                for joint_index, joint_name in enumerate(self._histogram_joint_names):
                    self._histogram_writer.add_histogram(
                        f"PhysicalHistogram/{name}/{joint_name}",
                        values[:, joint_index],
                        global_step=self.common_step_counter,
                    )

    def close(self):
        """Close TensorBoard writers and the simulator."""
        try:
            super().close()
        finally:
            if self._histogram_writer is not None:
                self._histogram_writer.flush()
                self._histogram_writer.close()
                self._histogram_writer = None

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute a mean on selected environments without synchronizing the device."""
        mask_float = mask.to(dtype=values.dtype)
        return (values * mask_float).sum() / mask_float.sum().clamp_min(1.0)

    def _get_physical_metrics(
        self,
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
        command: torch.Tensor,
        normalized_motor_effort: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return physical diagnostics grouped by sitting and walking commands.

        The returned values stay on the environment device. skrl converts them
        to scalars only when it consumes ``extras['log']``; keeping the
        calculations here on the device avoids an explicit CPU round-trip.
        """
        body_vel = body_vel.squeeze(-1)
        body_height = body_height.squeeze(-1)
        body_angle = body_angle.squeeze(-1)
        right_hip_angle = right_hip_angle.squeeze(-1)
        left_hip_angle = left_hip_angle.squeeze(-1)
        right_knee_angle = right_knee_angle.squeeze(-1)
        left_knee_angle = left_knee_angle.squeeze(-1)
        right_hip_vel = right_hip_vel.squeeze(-1)
        left_hip_vel = left_hip_vel.squeeze(-1)
        right_knee_vel = right_knee_vel.squeeze(-1)
        left_knee_vel = left_knee_vel.squeeze(-1)
        rotor_rod_vel = self.joint_vel[:, self.rotor_rod_dof_name_idx].squeeze(-1)
        rod_body_vel = self.joint_vel[:, self.rod_body_dof_name_idx].squeeze(-1)

        sitting = command[:, 0] == 1
        walking = ~sitting
        target_speed = command[:, 1]
        moving = walking & (target_speed.abs() >= self.cfg.metrics_speed_command_threshold)
        positive_speed = moving & (target_speed > 0.0)
        negative_speed = moving & (target_speed < 0.0)
        speed_error = body_vel - target_speed

        body_pose = self.robot.data.body_link_pose_w.torch[:, self.body_idx[0]]
        torso_height = body_pose[:, 2]
        head_height = self._get_top_torso_location()[0][:, 2]
        left_foot_height = left_foot_location[:, 2]
        right_foot_height = right_foot_location[:, 2]
        left_knee_height = left_knee_location[:, 2]
        right_knee_height = right_knee_location[:, 2]
        left_foot_speed = torch.linalg.vector_norm(left_foot_vel[:, :2], dim=-1)
        right_foot_speed = torch.linalg.vector_norm(right_foot_vel[:, :2], dim=-1)

        sit_rotor_target = self.cfg.rewards.sit_body_height_target
        sit_rod_target = self.cfg.rewards.sit_body_angle_target
        sit_right_hip_target = self.cfg.rewards.sit_right_hip_angle_target
        sit_left_hip_target = self.cfg.rewards.sit_left_hip_angle_target
        sit_right_knee_target = self.cfg.rewards.sit_right_knee_angle_target
        sit_left_knee_target = self.cfg.rewards.sit_left_knee_angle_target
        sit_rotor_error = body_height - sit_rotor_target
        sit_rod_error = body_angle - sit_rod_target
        sit_right_hip_error = right_hip_angle - sit_right_hip_target
        sit_left_hip_error = left_hip_angle - sit_left_hip_target
        sit_right_knee_error = right_knee_angle - sit_right_knee_target
        sit_left_knee_error = left_knee_angle - sit_left_knee_target
        sit_joint_angle_error = torch.stack(
            (
                sit_rotor_error,
                sit_rod_error,
                sit_right_hip_error,
                sit_left_hip_error,
                sit_right_knee_error,
                sit_left_knee_error,
            ),
            dim=-1,
        )
        sit_joint_velocity = torch.stack(
            (
                rotor_rod_vel,
                rod_body_vel,
                body_vel,
                right_hip_vel,
                left_hip_vel,
                right_knee_vel,
                left_knee_vel,
            ),
            dim=-1,
        )

        metrics = {
            "Physical/command/walking_fraction": walking.float().mean(),
            "Physical/command/sitting_fraction": sitting.float().mean(),
            "Physical/command/moving_fraction": moving.float().mean(),
            "Physical/command/positive_speed_fraction": positive_speed.float().mean(),
            "Physical/command/negative_speed_fraction": negative_speed.float().mean(),
            "Physical/termination/terminated_rate": self.reset_terminated.float().mean(),
            "Physical/termination/timeout_rate": self.reset_time_outs.float().mean(),
            "Physical/motor/applied_effort_abs_normalized": normalized_motor_effort.abs().mean(),
            "Physical/motor/applied_effort_l2_normalized": (
                torch.square(normalized_motor_effort).sum(dim=-1).mean()
            ),
            "Physical/motor/saturation_fraction": (normalized_motor_effort.abs() >= 0.999).float().mean(),
            "Physical/walk/torso_height": self._masked_mean(torso_height, walking),
            "Physical/walk/head_height": self._masked_mean(head_height, walking),
            "Physical/walk/left_knee_height": self._masked_mean(left_knee_height, walking),
            "Physical/walk/right_knee_height": self._masked_mean(right_knee_height, walking),
            "Physical/walk/left_foot_height": self._masked_mean(left_foot_height, walking),
            "Physical/walk/right_foot_height": self._masked_mean(right_foot_height, walking),
            "Physical/walk/mean_foot_height": self._masked_mean(
                (left_foot_height + right_foot_height) * 0.5, walking
            ),
            "Physical/walk/height_proxy_m": self._masked_mean(body_height, walking),
            "Physical/walk/rod_body_angle": self._masked_mean(body_angle, walking),
            "Physical/walk/height_proxy_m_abs": self._masked_mean(body_height.abs(), walking),
            "Physical/walk/rod_body_angle_abs": self._masked_mean(body_angle.abs(), walking),
            "Physical/walk/rotor_rod_angular_velocity_abs": self._masked_mean(rotor_rod_vel.abs(), walking),
            "Physical/walk/rod_body_angular_velocity_abs": self._masked_mean(rod_body_vel.abs(), walking),
            "Physical/walk/body_velocity": self._masked_mean(body_vel, walking),
            "Physical/walk/body_velocity_abs": self._masked_mean(body_vel.abs(), walking),
            "Physical/walk/target_speed": self._masked_mean(target_speed, moving),
            "Physical/walk/speed_error_signed": self._masked_mean(speed_error, moving),
            "Physical/walk/speed_error_abs": self._masked_mean(speed_error.abs(), moving),
            "Physical/walk/speed_error_positive_command_signed": self._masked_mean(
                speed_error, positive_speed
            ),
            "Physical/walk/speed_error_positive_command_abs": self._masked_mean(
                speed_error.abs(), positive_speed
            ),
            "Physical/walk/speed_error_negative_command_signed": self._masked_mean(
                speed_error, negative_speed
            ),
            "Physical/walk/speed_error_negative_command_abs": self._masked_mean(
                speed_error.abs(), negative_speed
            ),
            "Physical/walk/left_foot_horizontal_speed": self._masked_mean(left_foot_speed, walking),
            "Physical/walk/right_foot_horizontal_speed": self._masked_mean(right_foot_speed, walking),
            "Physical/walk/mean_foot_horizontal_speed": self._masked_mean(
                (left_foot_speed + right_foot_speed) * 0.5, walking
            ),
            "Physical/walk/left_foot_vertical_velocity": self._masked_mean(left_foot_vel[:, 2], walking),
            "Physical/walk/right_foot_vertical_velocity": self._masked_mean(right_foot_vel[:, 2], walking),
            "Physical/sit/torso_height": self._masked_mean(torso_height, sitting),
            "Physical/sit/head_height": self._masked_mean(head_height, sitting),
            "Physical/sit/height_proxy_error_m_signed": self._masked_mean(sit_rotor_error, sitting),
            "Physical/sit/height_proxy_error_m_abs": self._masked_mean(sit_rotor_error.abs(), sitting),
            "Physical/sit/rod_body_angle_error_signed": self._masked_mean(sit_rod_error, sitting),
            "Physical/sit/rod_body_angle_error_abs": self._masked_mean(sit_rod_error.abs(), sitting),
            "Physical/sit/right_hip_angle_error_abs": self._masked_mean(sit_right_hip_error.abs(), sitting),
            "Physical/sit/left_hip_angle_error_abs": self._masked_mean(sit_left_hip_error.abs(), sitting),
            "Physical/sit/right_knee_angle_error_abs": self._masked_mean(sit_right_knee_error.abs(), sitting),
            "Physical/sit/left_knee_angle_error_abs": self._masked_mean(sit_left_knee_error.abs(), sitting),
            "Physical/sit/mean_joint_angle_error_abs": self._masked_mean(
                sit_joint_angle_error.abs().mean(dim=-1), sitting
            ),
            "Physical/sit/rotor_rod_angular_velocity_abs": self._masked_mean(rotor_rod_vel.abs(), sitting),
            "Physical/sit/rod_body_angular_velocity_abs": self._masked_mean(rod_body_vel.abs(), sitting),
            "Physical/sit/body_velocity_abs": self._masked_mean(body_vel.abs(), sitting),
            "Physical/sit/mean_joint_velocity_abs": self._masked_mean(
                sit_joint_velocity.abs().mean(dim=-1), sitting
            ),
        }
        return metrics

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

        # Reset distribution is selected explicitly in the config. Active
        # standing environments are replaced below by the shared pose sampler.
        reset_mode = self.cfg.initial_reset_mode
        commands = sample_initial_commands(
            num_resets, self.cfg, self.device, mode=reset_mode
        )
        self.command[env_ids, :] = commands

        joint_pos = apply_sitting_reset_variation(
            self.robot.data.default_joint_pos.torch[env_ids],
            commands,
            self.cfg,
            self.initial_pose_indices.rod_body,
            mode=reset_mode,
        )
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel.torch[env_ids])
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_vel = torch.zeros_like(self.robot.data.default_root_vel.torch[env_ids])
        root_pose[:, :3] += self.scene.env_origins[env_ids]

        active_mask = commands[:, 0] == 0
        if bool(active_mask.any().item()):
            active_env_ids = env_ids[active_mask]
            active_result = sample_ground_safe_initial_pose(
                robot=self.robot,
                env_ids=active_env_ids,
                default_joint_pos=joint_pos[active_mask],
                default_joint_vel=joint_vel[active_mask],
                root_pose=root_pose[active_mask],
                soft_joint_pos_limits=self.robot.data.soft_joint_pos_limits.torch[env_ids][active_mask],
                cfg=self.cfg,
                indices=self.initial_pose_indices,
                collision_body_indices=self.collision_body_indices,
                left_foot_offset=self.left_foot_offset,
                right_foot_offset=self.right_foot_offset,
                mode=reset_mode,
                ground_check=self.cfg.initial_ground_safety_check,
                forward_fn=self.sim.forward,
            )
            joint_pos[active_mask] = active_result.joint_pos
            joint_vel[active_mask] = active_result.joint_vel

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        # Keep root position, including z, at the configured value. Height
        # correction happens only in bottom_rotor.
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        # Refresh FK before protecting the freshly sampled robot pose from
        # terrain bumps. The normal reset path performs another forward after
        # this method; this one is needed for partial resets during training.
        self.sim.forward()
        self._randomize_uneven_ground(env_ids)

        reset_targets = self._raw_to_canonical_actuated(
            joint_pos[:, self._actuated_dof_indices_tensor]
        )
        reset_actions = self._canonical_targets_to_actions(reset_targets)
        self.targets[env_ids] = reset_targets
        # The reset pose is already the active target. Seed all action-history
        # slots with its inverse affine action so the first policy command is
        # compared against the pose that was actually commanded, not zeros.
        self.actions[env_ids] = reset_actions
        self.previous_actions[env_ids] = reset_actions
        self.previous_previous_actions[env_ids] = reset_actions
        self._reset_observation_delay(env_ids, joint_pos, joint_vel)


def compute_action_acceleration_scale(
    timestep: float,
    start_scale: float,
    end_scale: float,
    start_timestep: float,
    end_timestep: float,
) -> float:
    """Linearly interpolate a reward scale over trainer timesteps."""

    if end_timestep <= start_timestep:
        return end_scale
    progress = min(
        max((timestep - start_timestep) / (end_timestep - start_timestep), 0.0),
        1.0,
    )
    return start_scale + progress * (end_scale - start_scale)


def compute_policy_action_abs_limit(
    canonical_target_min: torch.Tensor,
    canonical_target_max: torch.Tensor,
    action_offset: torch.Tensor,
    action_scale: torch.Tensor,
    range_margin: float,
) -> torch.Tensor:
    """Return symmetric action bounds based on the farther target limit."""

    target_range = canonical_target_max - canonical_target_min
    target_abs_limit = torch.maximum(
        (canonical_target_min - action_offset).abs(),
        (canonical_target_max - action_offset).abs(),
    ) + range_margin * target_range
    return target_abs_limit / action_scale.abs()


@torch.jit.script
def compute_target_second_difference(
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    previous_previous_actions: torch.Tensor,
    action_target_scale: torch.Tensor,
) -> torch.Tensor:
    """Return the second difference of the affine target command."""

    return (
        (actions - previous_actions)
        - (previous_actions - previous_previous_actions)
    ) * action_target_scale


@torch.jit.script
def compute_rewards(
    body_vel: torch.Tensor,
    body_height: torch.Tensor,
    body_vertical_vel: torch.Tensor,
    body_angular_vel: torch.Tensor,
    body_angle: torch.Tensor,
    actuated_joint_pos: torch.Tensor,
    actuated_joint_vel: torch.Tensor,
    joint_pos_limits: torch.Tensor,
    target_joint_limit_violation: torch.Tensor,
    normalized_motor_effort: torch.Tensor,
    foot_height: torch.Tensor,
    foot_horizontal_speed: torch.Tensor,
    reset_terminated: torch.Tensor,
    command: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    previous_previous_actions: torch.Tensor,
    action_target_scale: torch.Tensor,
    action_acceleration_scale: float,
    alive_reward_scale: float,
    death_reward_scale: float,
    walk_velocity_tracking_scale: float,
    walk_velocity_tracking_std: float,
    base_vertical_velocity_scale: float,
    base_angular_velocity_scale: float,
    joint_velocity_scale: float,
    joint_position_limits_scale: float,
    action_target_limits_scale: float,
    motor_effort_scale: float,
    foot_slip_scale: float,
    foot_slip_height_scale: float,
    joint_deviation_waist_scale: float,
    joint_deviation_legs_scale: float,
    flat_orientation_scale: float,
    walk_base_height_target: float,
    walk_base_height_scale: float,
    walk_body_angle_target: float,
    sit_body_height_target: float,
    sit_body_height_scale: float,
    sit_body_angle_target: float,
    sit_right_hip_angle_target: float,
    sit_left_hip_angle_target: float,
    sit_right_knee_angle_target: float,
    sit_left_knee_angle_target: float,
    sit_pose_angle_multiplier: float,
):
    # command[:, 0] is the sit/stand command (1 for sit, 0 for walk)
    # command[:, 1] is the target speed
    is_sitting_command = command[:, 0] == 1
    mode_multiplier = torch.where(
        is_sitting_command,
        torch.full_like(command[:, 1], sit_pose_angle_multiplier),
        torch.ones_like(command[:, 1]),
    )

    body_vel_value = body_vel.squeeze(-1)
    body_height_value = body_height.squeeze(-1)
    body_vertical_vel_value = body_vertical_vel.squeeze(-1)
    body_angular_vel_value = body_angular_vel.squeeze(-1)
    body_angle_value = body_angle.squeeze(-1)

    sit_joint_target = torch.zeros_like(actuated_joint_pos)
    sit_joint_target[:, 0] = sit_right_hip_angle_target
    sit_joint_target[:, 1] = sit_left_hip_angle_target
    sit_joint_target[:, 2] = sit_right_knee_angle_target
    sit_joint_target[:, 3] = sit_left_knee_angle_target
    joint_target = torch.where(
        is_sitting_command.unsqueeze(-1), sit_joint_target, torch.zeros_like(actuated_joint_pos)
    )
    body_angle_target = torch.where(
        is_sitting_command,
        torch.full_like(body_angle_value, sit_body_angle_target),
        torch.full_like(body_angle_value, walk_body_angle_target),
    )
    body_height_target = torch.where(
        is_sitting_command,
        torch.full_like(body_height_value, sit_body_height_target),
        torch.full_like(body_height_value, walk_base_height_target),
    )
    body_height_scale = torch.where(
        is_sitting_command,
        torch.full_like(body_height_value, sit_body_height_scale),
        torch.full_like(body_height_value, walk_base_height_scale),
    )
    velocity_command = torch.where(
        is_sitting_command, torch.zeros_like(command[:, 1]), command[:, 1]
    )

    # Common reward terms. CBR-I additionally assigns an explicit penalty to
    # terminal deaths; timeouts are not included in reset_terminated.
    alive_reward = (1.0 - reset_terminated.float()) * alive_reward_scale
    death_reward = reset_terminated.float() * death_reward_scale
    common_reward = alive_reward + death_reward
    common_reward += torch.square(body_vertical_vel_value) * base_vertical_velocity_scale
    common_reward += torch.square(body_angular_vel_value) * base_angular_velocity_scale
    common_reward += torch.sum(torch.square(actuated_joint_vel), dim=-1) * joint_velocity_scale
    # The direct action is an affine target command.  Its constant offset
    # cancels in the second difference, while the target scale converts the
    # normalized action acceleration to physical radians.
    target_second_difference = compute_target_second_difference(
        actions,
        previous_actions,
        previous_previous_actions,
        action_target_scale,
    )
    common_reward += (
        torch.sum(torch.square(target_second_difference), dim=-1)
        * action_acceleration_scale
    )
    common_reward += joint_pos_limits * joint_position_limits_scale
    common_reward += (
        torch.sum(torch.square(target_joint_limit_violation), dim=-1)
        * action_target_limits_scale
    )
    common_reward += torch.sum(torch.square(normalized_motor_effort), dim=-1) * motor_effort_scale
    foot_ground_weight = torch.exp(-foot_height / foot_slip_height_scale)
    foot_slip_penalty = torch.sum(
        foot_ground_weight * foot_horizontal_speed, dim=-1
    )
    foot_slip_penalty *= (~is_sitting_command).to(dtype=foot_slip_penalty.dtype)
    common_reward += foot_slip_penalty * foot_slip_scale

    # Unitree track_lin_vel_xy_exp reduced to the one available longitudinal
    # speed proxy. Sitting is the same stand-still command with v_target=0.
    velocity_error = body_vel_value - velocity_command
    mode_reward = torch.exp(
        -(velocity_error ** 2) / (walk_velocity_tracking_std ** 2)
    ) * walk_velocity_tracking_scale

    # Unitree joint_deviation_waists, joint_deviation_legs and
    # flat_orientation_l2. For sitting the same terms point to the sitting
    # target and are doubled to make the pose precise.
    mode_reward += (
        torch.abs(body_angle_value - body_angle_target) * joint_deviation_waist_scale
        + torch.sum(torch.abs(actuated_joint_pos - joint_target), dim=-1) * joint_deviation_legs_scale
        + torch.square(body_angle_value - body_angle_target) * flat_orientation_scale
    ) * mode_multiplier
    mode_reward += torch.square(body_height_value - body_height_target) * body_height_scale

    return common_reward + mode_reward

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
