"""Regression tests for the explicit coupled-actuator stability settings."""

from __future__ import annotations

import pytest

from CBRIIsaacLab.robots.CBRI import CBR_I_CONFIG
from CBRIIsaacLab.tasks.direct.cbriisaaclab.cbriisaaclab_env_cfg import CbriisaaclabEnvCfg


def test_coupled_actuator_uses_measured_stable_armature() -> None:
    actuator_cfg = CBR_I_CONFIG.actuators["coupled_leg_actuator"]

    assert actuator_cfg.armature == pytest.approx(0.02)
    assert actuator_cfg.friction == pytest.approx(0.10)
    assert actuator_cfg.dynamic_friction == pytest.approx(0.08)
    assert actuator_cfg.viscous_friction == pytest.approx(0.01)


def test_explicit_pd_randomization_does_not_exceed_nominal_damping() -> None:
    env_cfg = CbriisaaclabEnvCfg()
    gain_params = env_cfg.events.robot_joint_stiffness_and_damping.params

    assert gain_params["stiffness_distribution_params"] == pytest.approx((0.9, 1.1))
    assert gain_params["damping_distribution_params"] == pytest.approx((0.9, 1.0))
    assert CBR_I_CONFIG.spawn.articulation_props.solver_position_iteration_count == 4
    assert CBR_I_CONFIG.spawn.articulation_props.solver_velocity_iteration_count == 0


def test_joint_friction_is_randomized_around_nominal_values() -> None:
    env_cfg = CbriisaaclabEnvCfg()
    friction_params = env_cfg.events.robot_joint_friction.params

    assert friction_params["friction_distribution_params"] == pytest.approx((0.5, 1.5))
    assert friction_params["operation"] == "scale"
