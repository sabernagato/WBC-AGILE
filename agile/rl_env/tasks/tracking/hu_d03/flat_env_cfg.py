# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""HU_D03 full-body reference-motion tracking on flat ground."""

import os
from dataclasses import MISSING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from agile.common.hu_d03_motion import HU_D03_MOTION_JOINT_NAMES, HU_D03_TRACKED_BODY_NAMES
from agile.rl_env.assets.robots import hu_d03
from agile.rl_env.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class HUD03FlatEnvCfg(TrackingEnvCfg):
    """Track a 50 Hz, 31-DoF HU_D03 motion clip."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = hu_d03.HU_D03_DELAYED_DC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = hu_d03.HU_D03_ACTION_SCALE_FULL_BODY

        self.commands.motion.anchor_body_name = hu_d03.TORSO_BODY_NAME
        self.commands.motion.debug_vis = False
        self.commands.motion.motion_file = os.environ.get("MOTION_FILE", MISSING)
        self.commands.motion.body_names = HU_D03_TRACKED_BODY_NAMES
        # HU_D03 motion files intentionally contain only the 15 tracked bodies,
        # in the canonical order shared by the generator and validator.
        self.commands.motion.motion_body_names = HU_D03_TRACKED_BODY_NAMES
        self.commands.motion.motion_joint_names = HU_D03_MOTION_JOINT_NAMES
        self.commands.motion.require_name_metadata = True
        self.commands.motion.joint_position_range = (-0.12, 0.12)

        # Rebind the G1-specific inherited randomization targets.
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=hu_d03.ROOT_BODY_NAME
        )
        self.events.randomize_base_mass.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=hu_d03.ROOT_BODY_NAME
        )
        self.events.apply_external_force_torque.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=hu_d03.TORSO_BODY_NAME
        )
        self.events.apply_external_force_torque_extremities.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[".*_hand_yaw_link"]
        )

        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                r"(?!left_hand_yaw_link$)(?!right_hand_yaw_link$).+$"
            ],
        )
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_hand_yaw_link",
            "right_hand_yaw_link",
        ]
