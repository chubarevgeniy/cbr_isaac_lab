import os

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from .coupled_leg_actuator import CoupledLegPDActuatorCfg

usd_path = os.path.join(os.path.dirname(__file__), "CBR-I.usda")

CBR_I_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "Rock_Revolute_1": 0.0,
            "bottom_rotor_Revolute_2": 5.2 * np.pi / 180.0,
            "rod_1_Revolute_3": -80.0 * np.pi / 180.0,
            "body_Revolute_4": 0,  # Right_hip
            "body_Revolute_5": 0,  # Left_hip
            # Keep the reset infinitesimally inside the authored USD limit;
            # the canonical task target remains the nominal 124 deg.
            "right_hip_Revolute_6": -124.0 * np.pi / 180.0 * 0.9999,
            "left_hip_Revolute_7": 124.0 * np.pi / 180.0 * 0.9999,
        },
        joint_vel={
            "Rock_Revolute_1": 0.0,
            "bottom_rotor_Revolute_2": 0.0,
            "rod_1_Revolute_3": 0.0,
            "body_Revolute_4": 0.0,
            "body_Revolute_5": 0.0,
            "right_hip_Revolute_6": 0.0,
            "left_hip_Revolute_7": 0.0,
        },
    ),
    actuators={
        "base_rotor_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Rock_Revolute_1"],
            effort_limit_sim=100.0,  # Physics solver effort limit [N m]
            velocity_limit_sim=572957.0,  # Adjusted velocity limit
            stiffness=0.0,
            damping=0.0,
        ),
        "rotor_rod_actuator": ImplicitActuatorCfg(
            joint_names_expr=["bottom_rotor_Revolute_2"],
            effort_limit_sim=100.0,  # Physics solver effort limit [N m]
            velocity_limit_sim=572957.0,  # Adjusted velocity limit
            stiffness=0.0,
            damping=0.0,
        ),
        "rod_body_actuator": ImplicitActuatorCfg(
            joint_names_expr=["rod_1_Revolute_3"],
            effort_limit_sim=100.0,  # Physics solver effort limit [N m]
            velocity_limit_sim=572957.0,  # Adjusted velocity limit
            stiffness=0.0,
            damping=0.0,
        ),
        "coupled_leg_actuator": CoupledLegPDActuatorCfg(
            joint_names_expr=[
                "body_Revolute_4",
                "body_Revolute_5",
                "right_hip_Revolute_6",
                "left_hip_Revolute_7",
            ],
            # In canonical coordinates theta_knee = q_knee_motor - q_hip_motor,
            # hence q_knee_motor = theta_knee + theta_hip for both legs.
            # Canonical physical coordinates use the opposite sign on the
            # right leg and the authored sign on the left leg.
            transmission_pairs=[
                ("body_Revolute_4", "right_hip_Revolute_6", -1.0, -1.0),
                ("body_Revolute_5", "left_hip_Revolute_7", 1.0, 1.0),
            ],
            # These are motor-space limits.  After the transmission mapping,
            # the physical hip joint can receive up to the sum of both motor
            # torques while each motor remains limited to 3.5 N m.
            effort_limit=5.0,
            effort_limit_sim={
                "body_Revolute_4": 10.0,
                "body_Revolute_5": 10.0,
                "right_hip_Revolute_6": 5.0,
                "left_hip_Revolute_7": 5.0,
            },
            velocity_limit=572957.0,
            velocity_limit_sim=572957.0,
            # Reflected output-side inertia of the 5008 motors with 12:1
            # reduction.  Besides matching the missing rotor inertia, this
            # keeps the explicit motor-space PD stable at the 250 Hz physics
            # rate.  The coupled transmission is still evaluated explicitly
            # by CoupledLegPDActuator.
            armature=0.02,
            # Joint-space approximation of motor, 12:1 gearbox, and bearing
            # losses.  Isaac Sim 6 interprets static/dynamic values as joint
            # effort [N m] and viscous friction as [N m s/rad].
            friction=0.12,
            dynamic_friction=0.096,
            viscous_friction=0.012,
            stiffness=73.3,
            damping=3.67,
        ),
    },
)
