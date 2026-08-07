# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the coupled hip/knee motor transmission."""

from __future__ import annotations

import torch

from isaaclab.utils.types import ArticulationActions

from CBRIIsaacLab.robots.coupled_leg_actuator import CoupledLegPDActuator, CoupledLegPDActuatorCfg


JOINT_NAMES = [
    "body_Revolute_4",
    "right_hip_Revolute_6",
    "body_Revolute_5",
    "left_hip_Revolute_7",
]


def _make_actuator(*, stiffness: float = 1.0, damping: float = 0.0, effort_limit: float = 4.5):
    cfg = CoupledLegPDActuatorCfg(
        joint_names_expr=JOINT_NAMES,
        transmission_pairs=[
            ("body_Revolute_4", "right_hip_Revolute_6", -1.0, -1.0),
            ("body_Revolute_5", "left_hip_Revolute_7", 1.0, 1.0),
        ],
        effort_limit=effort_limit,
        effort_limit_sim=2.0 * effort_limit,
        velocity_limit=100.0,
        velocity_limit_sim=100.0,
        stiffness=stiffness,
        damping=damping,
    )
    return CoupledLegPDActuator(
        cfg,
        JOINT_NAMES,
        torch.arange(len(JOINT_NAMES)),
        num_envs=1,
        device="cpu",
    )


def _position_action(target: torch.Tensor) -> ArticulationActions:
    return ArticulationActions(
        joint_positions=target,
        joint_velocities=torch.zeros_like(target),
        joint_efforts=torch.zeros_like(target),
        joint_indices=torch.arange(target.shape[-1]),
    )


def test_pd_error_is_evaluated_in_motor_coordinates_for_both_usd_signs() -> None:
    actuator = _make_actuator()
    current = torch.zeros((1, 4))

    # Both legs request canonical physical errors theta_h=2, theta_k=3.
    # Therefore motor errors are q_h=2 and q_k=theta_k-theta_h=1.
    target = torch.tensor([[-2.0, -3.0, 2.0, 3.0]])
    result = actuator.compute(_position_action(target), current, torch.zeros_like(current))

    # Motor efforts (2, 1) map to canonical physical efforts (1, 1).
    # The right-leg USD coordinates have the opposite sign.
    expected = torch.tensor([[-1.0, -1.0, 1.0, 1.0]])
    torch.testing.assert_close(actuator.computed_effort, expected)
    torch.testing.assert_close(actuator.applied_effort, expected)
    torch.testing.assert_close(result.joint_efforts, expected)
    assert result.joint_positions is None
    assert result.joint_velocities is None


def test_each_motor_is_clipped_before_efforts_are_mapped_to_physical_joints() -> None:
    actuator = _make_actuator(effort_limit=4.5)
    current = torch.zeros((1, 4))

    # theta_h=10, theta_k=0 produces motor errors q_h=10, q_k=-10.
    # Unclipped physical efforts are (20, -10); clipping each motor first
    # gives (tau_h, tau_k)=(9, -4.5).
    target = torch.tensor([[-10.0, 0.0, 10.0, 0.0]])
    actuator.compute(_position_action(target), current, torch.zeros_like(current))

    torch.testing.assert_close(
        actuator.computed_effort,
        torch.tensor([[-20.0, 10.0, 20.0, -10.0]]),
    )
    torch.testing.assert_close(
        actuator.applied_effort,
        torch.tensor([[-9.0, 4.5, 9.0, -4.5]]),
    )
    torch.testing.assert_close(
        actuator.computed_motor_effort,
        torch.tensor([[10.0, -10.0, 10.0, -10.0]]),
    )
    torch.testing.assert_close(
        actuator.applied_motor_effort,
        torch.tensor([[4.5, -4.5, 4.5, -4.5]]),
    )


def test_velocity_feedback_uses_knee_motor_velocity_difference() -> None:
    actuator = _make_actuator(stiffness=0.0, damping=1.0, effort_limit=100.0)
    current = torch.zeros((1, 4))
    # Canonical physical velocities are theta_h_dot=2, theta_k_dot=3 on
    # both legs, so motor velocities are q_h_dot=2 and q_k_dot=1.
    raw_velocity = torch.tensor([[-2.0, -3.0, 2.0, 3.0]])
    target = torch.zeros_like(current)
    actuator.compute(_position_action(target), current, raw_velocity)

    # Damping commands motor efforts (-2, -1), mapping to canonical
    # physical efforts (-1, -1), with the authored right signs reversed.
    torch.testing.assert_close(
        actuator.applied_effort,
        torch.tensor([[1.0, 1.0, -1.0, -1.0]]),
    )
