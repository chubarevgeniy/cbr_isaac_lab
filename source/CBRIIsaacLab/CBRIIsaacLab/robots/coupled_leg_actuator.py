# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actuator model for the coupled CBR-I hip/knee transmission."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    from collections.abc import Sequence


class CoupledLegPDActuator(IdealPDActuator):
    r"""PD actuator for ``theta_knee = q_knee_motor - q_hip_motor``.

    The articulation keeps the physically useful joint coordinates
    ``(theta_hip, theta_knee)``.  For each configured leg pair this actuator
    evaluates the controller in motor coordinates

    .. math::

        q_h = \theta_h, \qquad q_k = \theta_k + \theta_h,

    clips both motor torques independently, and maps them back to articulation
    efforts using virtual work:

    .. math::

        \tau_{\theta_h} = \tau_{q_h} + \tau_{q_k}, \qquad
        \tau_{\theta_k} = \tau_{q_k}.

    Consequently the knee's existing revolute-joint limit remains a hard
    limit on the physical relative knee angle; no fake link or tendon is
    needed.  The signs in ``transmission_pairs`` convert each authored USD
    coordinate to the common canonical coordinate used by the equations.
    """

    cfg: CoupledLegPDActuatorCfg

    def __init__(self, cfg: CoupledLegPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # Isaac Lab's standard effort telemetry is expressed in articulation
        # coordinates.  Keep the motor-space values as separate buffers so
        # torque-limit diagnostics do not mistake a valid 9 N m physical hip
        # effort for a violation of either 4.5 N m motor limit.
        self.computed_motor_effort = torch.zeros_like(self.computed_effort)
        self.applied_motor_effort = torch.zeros_like(self.applied_effort)

        local_index = {name: index for index, name in enumerate(self.joint_names)}
        self._pairs: list[tuple[int, int, float, float]] = []
        used_joint_names: set[str] = set()

        for hip_name, knee_name, hip_sign, knee_sign in cfg.transmission_pairs:
            if hip_name not in local_index or knee_name not in local_index:
                raise ValueError(
                    "Every coupled-transmission joint must belong to the actuator group; "
                    f"got pair ({hip_name!r}, {knee_name!r}) for {self.joint_names!r}."
                )
            if hip_name in used_joint_names or knee_name in used_joint_names:
                raise ValueError(f"Joint appears in more than one transmission pair: {(hip_name, knee_name)!r}")
            if hip_sign not in (-1.0, 1.0) or knee_sign not in (-1.0, 1.0):
                raise ValueError("Transmission coordinate signs must be either -1.0 or 1.0.")

            self._pairs.append((local_index[hip_name], local_index[knee_name], hip_sign, knee_sign))
            used_joint_names.update((hip_name, knee_name))

        if used_joint_names != set(self.joint_names):
            missing = sorted(set(self.joint_names) - used_joint_names)
            raise ValueError(f"All joints in a CoupledLegPDActuator must be paired; unpaired joints: {missing!r}")

    def reset(self, env_ids: Sequence[int]):
        # This ideal actuator has no temporal state.
        pass

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        if control_action.joint_positions is None:
            position_error = torch.zeros_like(joint_pos)
        else:
            position_error = control_action.joint_positions - joint_pos

        if control_action.joint_velocities is None:
            velocity_error = -joint_vel
        else:
            velocity_error = control_action.joint_velocities - joint_vel

        motor_feedforward = control_action.joint_efforts
        if motor_feedforward is None:
            motor_feedforward = torch.zeros_like(joint_pos)

        computed_physical_effort = torch.zeros_like(joint_pos)
        applied_physical_effort = torch.zeros_like(joint_pos)
        computed_motor_effort = torch.zeros_like(joint_pos)
        applied_motor_effort = torch.zeros_like(joint_pos)

        for hip_index, knee_index, hip_sign, knee_sign in self._pairs:
            # Convert physical-joint errors to the two motor-coordinate errors.
            hip_position_error = hip_sign * position_error[:, hip_index]
            knee_position_error = knee_sign * position_error[:, knee_index]
            hip_velocity_error = hip_sign * velocity_error[:, hip_index]
            knee_velocity_error = knee_sign * velocity_error[:, knee_index]

            hip_motor_position_error = hip_position_error
            knee_motor_position_error = knee_position_error + hip_position_error
            hip_motor_velocity_error = hip_velocity_error
            knee_motor_velocity_error = knee_velocity_error + hip_velocity_error

            computed_hip_motor_effort = (
                self.stiffness[:, hip_index] * hip_motor_position_error
                + self.damping[:, hip_index] * hip_motor_velocity_error
                + motor_feedforward[:, hip_index]
            )
            computed_knee_motor_effort = (
                self.stiffness[:, knee_index] * knee_motor_position_error
                + self.damping[:, knee_index] * knee_motor_velocity_error
                + motor_feedforward[:, knee_index]
            )

            applied_hip_motor_effort = torch.clamp(
                computed_hip_motor_effort,
                min=-self.effort_limit[:, hip_index],
                max=self.effort_limit[:, hip_index],
            )
            applied_knee_motor_effort = torch.clamp(
                computed_knee_motor_effort,
                min=-self.effort_limit[:, knee_index],
                max=self.effort_limit[:, knee_index],
            )

            computed_motor_effort[:, hip_index] = computed_hip_motor_effort
            computed_motor_effort[:, knee_index] = computed_knee_motor_effort
            applied_motor_effort[:, hip_index] = applied_hip_motor_effort
            applied_motor_effort[:, knee_index] = applied_knee_motor_effort

            # Map motor efforts to canonical physical efforts, then restore
            # the signs of the authored right/left USD joint coordinates.
            computed_physical_effort[:, hip_index] = hip_sign * (
                computed_hip_motor_effort + computed_knee_motor_effort
            )
            computed_physical_effort[:, knee_index] = knee_sign * computed_knee_motor_effort
            applied_physical_effort[:, hip_index] = hip_sign * (
                applied_hip_motor_effort + applied_knee_motor_effort
            )
            applied_physical_effort[:, knee_index] = knee_sign * applied_knee_motor_effort

        self.computed_effort = computed_physical_effort
        self.applied_effort = applied_physical_effort
        self.computed_motor_effort = computed_motor_effort
        self.applied_motor_effort = applied_motor_effort
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class CoupledLegPDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for :class:`CoupledLegPDActuator`.

    Each tuple contains ``(hip_joint, knee_joint, hip_sign, knee_sign)``.
    The signs are the derivatives of canonical physical angles with respect
    to the corresponding authored USD joint coordinates.
    """

    class_type: type[CoupledLegPDActuator] = CoupledLegPDActuator

    transmission_pairs: list[tuple[str, str, float, float]] = MISSING
