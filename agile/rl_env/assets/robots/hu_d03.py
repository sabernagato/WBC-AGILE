# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""HU_D03 robot configuration for WBC-AGILE.

The gains below are conservative training defaults derived from the limits in
the HU_D03_03 asset. They must be replaced with identified hardware values
before sim-to-real deployment.
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

from agile.rl_env.mdp.actuators import DelayedDCMotorCfg

MIN_DELAY_PHY_STEPS = 0
MAX_DELAY_PHY_STEPS = 4

_DEFAULT_DESCRIPTION_ROOT = (
    Path(__file__).resolve().parents[5] / "limx_oli_description" / "HU_D03_description"
)
HU_D03_DESCRIPTION_ROOT = Path(
    os.environ.get("HU_D03_DESCRIPTION_ROOT", _DEFAULT_DESCRIPTION_ROOT)
).expanduser()
HU_D03_USD_PATH = os.environ.get(
    "HU_D03_USD_PATH",
    str(HU_D03_DESCRIPTION_ROOT / "usd" / "HU_D03_03.usd"),
)

ROOT_BODY_NAME = "base_link"
TORSO_BODY_NAME = "waist_pitch_link"
HEAD_BODY_NAME = "head_pitch_link"

LEG_JOINT_NAMES = [
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
]
ANKLE_JOINT_NAMES = [".*_ankle_.*_joint"]
WAIST_JOINT_NAMES = ["waist_.*_joint"]
ARM_JOINT_NAMES = [".*_shoulder_.*_joint", ".*_elbow_joint"]
WRIST_JOINT_NAMES = [".*_wrist_.*_joint", ".*_hand_yaw_joint"]
HEAD_JOINT_NAMES = ["head_.*_joint"]
CONTROLLED_JOINT_NAMES = LEG_JOINT_NAMES + ["waist_roll_joint", "waist_pitch_joint"]

FEET_LINK_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
DEFAULT_BASE_HEIGHT = 0.92
DEFAULT_FEET_DISTANCE = 0.242

# Contacts with these links are expected while recovering from a fall but are
# penalized so the learned terminal pose prefers both feet on the ground.
UNDESIRED_CONTACTS_LINKS = [
    ROOT_BODY_NAME,
    "waist_.*_link",
    "head_.*_link",
    ".*_hip_.*_link",
    ".*_shoulder_.*_link",
    ".*_elbow_link",
    ".*_wrist_.*_link",
    ".*_hand_yaw_link",
]

# Software limits from HU_D03_03.urdf. The higher saturation values represent
# the preliminary hardware peak classes and are not validated continuous
# torque ratings.
HU_D03_DELAYED_DC_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=HU_D03_USD_PATH,
        # The source USD's Robot variant references a file that is not shipped.
        # Physics already contains the complete articulation, so disable that
        # optional payload and the unused sensor payload for training.
        variants={"Physics": "PhysX", "Sensor": "None", "Robot": "None"},
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    articulation_root_prim_path=f"/{ROOT_BODY_NAME}",
    soft_joint_pos_limit_factor=0.9,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            ".*_hip_pitch_joint": -0.15,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.15,
            "left_shoulder_roll_joint": 0.20,
            "right_shoulder_roll_joint": -0.20,
            ".*_elbow_joint": 0.30,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": DelayedDCMotorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=120.0,
            velocity_limit_sim=12.0,
            stiffness={
                ".*_hip_.*_joint": 100.0,
                ".*_knee_joint": 150.0,
            },
            damping={
                ".*_hip_.*_joint": 4.0,
                ".*_knee_joint": 5.0,
            },
            friction=0.01,
            armature=0.15257125,
            saturation_effort=200.0,
            min_delay=MIN_DELAY_PHY_STEPS,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "feet": DelayedDCMotorCfg(
            joint_names_expr=ANKLE_JOINT_NAMES,
            effort_limit_sim=45.0,
            velocity_limit_sim=16.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.01,
            armature=0.094889232,
            saturation_effort=75.0,
            min_delay=MIN_DELAY_PHY_STEPS,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "waist": DelayedDCMotorCfg(
            joint_names_expr=WAIST_JOINT_NAMES,
            effort_limit_sim=45.0,
            velocity_limit_sim=16.0,
            stiffness=100.0,
            damping=3.0,
            friction=0.01,
            armature=0.094889232,
            saturation_effort=75.0,
            min_delay=MIN_DELAY_PHY_STEPS,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "arms": DelayedDCMotorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            effort_limit_sim=30.0,
            velocity_limit_sim=20.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.01,
            armature=0.02,
            saturation_effort=50.0,
            min_delay=MIN_DELAY_PHY_STEPS,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "wrists": DelayedDCMotorCfg(
            joint_names_expr=WRIST_JOINT_NAMES,
            effort_limit_sim=18.0,
            velocity_limit_sim=14.0,
            stiffness=15.0,
            damping=0.5,
            friction=0.005,
            armature=0.01,
            saturation_effort=30.0,
            min_delay=MIN_DELAY_PHY_STEPS,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "head": DelayedDCMotorCfg(
            joint_names_expr=HEAD_JOINT_NAMES,
            effort_limit_sim=18.0,
            velocity_limit_sim=14.0,
            stiffness=10.0,
            damping=0.5,
            friction=0.005,
            armature=0.01,
            saturation_effort=30.0,
            min_delay=MIN_DELAY_PHY_STEPS,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
    },
)

# Keep commanded position offsets comfortably inside the software torque
# limits. Isaac Lab accepts regex-to-scale mappings for JointPositionActionCfg.
HU_D03_ACTION_SCALE_LOWER = {
    ".*_hip_.*_joint": 0.30,
    ".*_knee_joint": 0.20,
    ".*_ankle_.*_joint": 0.25,
    "waist_roll_joint": 0.15,
    "waist_pitch_joint": 0.15,
}

# Full-body reference tracking needs enough arm range for gestures while
# keeping the preliminary, uncalibrated wrist/head groups conservative.
HU_D03_ACTION_SCALE_FULL_BODY = {
    ".*_hip_.*_joint": 0.30,
    ".*_knee_joint": 0.20,
    ".*_ankle_.*_joint": 0.25,
    "waist_yaw_joint": 0.20,
    "waist_roll_joint": 0.15,
    "waist_pitch_joint": 0.15,
    ".*_shoulder_.*_joint": 0.35,
    ".*_elbow_joint": 0.35,
    ".*_wrist_.*_joint": 0.25,
    ".*_hand_yaw_joint": 0.25,
    "head_.*_joint": 0.20,
}
