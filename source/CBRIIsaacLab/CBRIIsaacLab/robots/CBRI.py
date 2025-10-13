import os

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

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
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "Rock_Revolute_1": 0.0,
            "bottom_rotor_Revolute_2": 5.2 * np.pi / 180.0,
            "rod_1_Revolute_3": -80.0 * np.pi / 180.0,
            "body_Revolute_4": 0,  # Right_hip
            "body_Revolute_5": 0,  # Left_hip
            "right_hip_Revolute_6": -124.0 * np.pi / 180.0 * 0.99,
            "left_hip_Revolute_7": 124.0 * np.pi / 180.0 * 0.99,
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
            effort_limit=100.0,  # Adjusted effort limit
            velocity_limit_sim=572957.0,  # Adjusted velocity limit
            stiffness=0.0,
            damping=0.0,
        ),
        "rotor_rod_actuator": ImplicitActuatorCfg(
            joint_names_expr=["bottom_rotor_Revolute_2"],
            effort_limit=100.0,  # Adjusted effort limit
            velocity_limit_sim=572957.0,  # Adjusted velocity limit
            stiffness=0.0,
            damping=0.0,
        ),
        "rod_body_actuator": ImplicitActuatorCfg(
            joint_names_expr=["rod_1_Revolute_3"],
            effort_limit=100.0,  # Adjusted effort limit
            velocity_limit_sim=572957.0,  # Adjusted velocity limit
            stiffness=0.0,
            damping=0.0,
        ),
        "body_right_hip_actuator": ImplicitActuatorCfg(
            joint_names_expr=["body_Revolute_4"],
            effort_limit=5,
            velocity_limit_sim=572957.0,
            stiffness=27.9,
            damping=2.15,
        ),
        "body_left_hip_actuator": ImplicitActuatorCfg(
            joint_names_expr=["body_Revolute_5"],
            effort_limit=5,
            velocity_limit_sim=572957.0,
            stiffness=27.9,
            damping=2.15,
        ),
        "right_hip_shin_actuator": ImplicitActuatorCfg(
            joint_names_expr=["right_hip_Revolute_6"],
            effort_limit=5,
            velocity_limit_sim=572957.0,
            stiffness=27.9,
            damping=2.15,
        ),
        "left_hip_shin_actuator": ImplicitActuatorCfg(
            joint_names_expr=["left_hip_Revolute_7"],
            effort_limit=5,
            velocity_limit_sim=572957.0,
            stiffness=27.9,
            damping=2.15,
        ),
    },
)