# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""HU_D03 full-body recovery from cached fallen poses.

This task specializes the proven Booster T1 stand-up MDP while preserving its
reward, observation, lift-assist, and terrain curriculum structure. All
robot-specific body names, joint groups, dimensions, and height thresholds are
rebound to HU_D03 below.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from agile.rl_env import mdp
from agile.rl_env.assets.robots import hu_d03
from agile.rl_env.tasks.stand_up.t1.stand_up_env_cfg import (
    CurriculumCfg as T1CurriculumCfg,
)
from agile.rl_env.tasks.stand_up.t1.stand_up_env_cfg import T1StandUpEnvCfg


@configclass
class HUD03StandUpCurriculumCfg(T1CurriculumCfg):
    """T1 recovery curriculum plus staged arbitrary fallen poses."""

    random_fallen_states = CurrTerm(
        func=mdp.update_event_param_after_curriculum,
        params={
            "event_term": "reset_base",
            "param_name": "random_fallen_ratio",
            "start_value": 0.25,
            "terminal_value": 1.0,
            "prerequisite_curriculum": "terrain_levels",
            "prerequisite_threshold": 2.0,
            "delay_steps": 1_000,
            "num_steps": 48_000,
        },
    )


@configclass
class HUD03StandUpEnvCfg(T1StandUpEnvCfg):
    """Recover HU_D03 from supine, prone, and side-lying configurations."""

    curriculum: HUD03StandUpCurriculumCfg = HUD03StandUpCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # Robot and ground-height sensing.
        self.scene.robot = hu_d03.HU_D03_DELAYED_DC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_measurement_sensor.prim_path = (
            f"{{ENV_REGEX_NS}}/Robot/{hu_d03.ROOT_BODY_NAME}"
        )

        # The policy controls all 31 active joints. Relative actions let the
        # policy build coordinated recovery trajectories from arbitrary poses.
        self.actions.joint_pos.joint_names = [".*"]
        self.actions.joint_pos.scale = 0.1
        self.actions.joint_pos.clip = {".*": (-1.0, 1.0)}
        self.actions.joint_pos.use_zero_offset = True
        self.actions.joint_pos.preserve_order = True

        # Early training receives an upward force capped at 90% of robot weight.
        # The inherited adaptive curriculum removes it as recovery improves.
        self.actions.lift.link_to_lift = hu_d03.HEAD_BODY_NAME
        self.actions.lift.stiffness_forces = 5_000.0
        self.actions.lift.damping_forces = 800.0
        self.actions.lift.force_limit = 0.0
        self.actions.lift.force_limit_weight_fraction = 0.9
        self.actions.lift.damping_torques = 120.0
        self.actions.lift.torque_limit = 350.0
        self.actions.lift.target_height = hu_d03.DEFAULT_BASE_HEIGHT
        self.actions.lift.start_lifting_time_s = 2.0
        self.actions.lift.lifting_duration_s = 10.0

        # Height-shaped recovery rewards.
        for reward_name in ("base_height_rough", "base_height_medium", "base_height_fine"):
            getattr(self.rewards, reward_name).params["target_height"] = hu_d03.DEFAULT_BASE_HEIGHT

        standing_threshold = hu_d03.DEFAULT_BASE_HEIGHT * 0.8
        self.rewards.joint_deviation_l1.params["standing_height_threshold"] = standing_threshold
        self.rewards.joint_deviation_l1_upper_body.params.update(
            {
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(
                        hu_d03.WAIST_JOINT_NAMES
                        + hu_d03.ARM_JOINT_NAMES
                        + hu_d03.WRIST_JOINT_NAMES
                        + hu_d03.HEAD_JOINT_NAMES
                    ),
                ),
                "standing_height_threshold": standing_threshold,
            }
        )
        self.rewards.ankle_torques.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=hu_d03.ANKLE_JOINT_NAMES
        )
        self.rewards.orientation.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[hu_d03.ROOT_BODY_NAME]
        )
        self.rewards.not_moving.params["standing_height_threshold"] = standing_threshold
        self.rewards.illegal_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=hu_d03.UNDESIRED_CONTACTS_LINKS
        )
        self.rewards.feet_distance.params.update(
            {
                "asset_cfg": SceneEntityCfg("robot", body_names=hu_d03.FEET_LINK_NAMES),
                "ref_distance": hu_d03.DEFAULT_FEET_DISTANCE,
                "standing_height_threshold": standing_threshold,
            }
        )
        self.rewards.feet_yaw_mean.params.update(
            {
                "feet_asset_cfg": SceneEntityCfg("robot", body_names=hu_d03.FEET_LINK_NAMES),
                "base_body_cfg": SceneEntityCfg("robot", body_names=hu_d03.ROOT_BODY_NAME),
                "standing_height_threshold": standing_threshold,
            }
        )
        self.rewards.root_acc.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=hu_d03.ROOT_BODY_NAME
        )

        # Progress is measured at the pelvis/root, not the G1/T1 torso name.
        self.terminations.no_height_progress.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=hu_d03.ROOT_BODY_NAME
        )
        self.terminations.no_height_progress.params["height_increase_threshold"] = 0.25
        self.terminations.no_height_progress.params["time_limit_s"] = 12.0

        # Domain randomization and disturbances use HU_D03 body names.
        self.events.randomize_base_mass.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=hu_d03.ROOT_BODY_NAME
        )
        self.events.randomize_base_com.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=hu_d03.ROOT_BODY_NAME
        )
        self.events.apply_external_force_torque.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=hu_d03.TORSO_BODY_NAME
        )
        self.events.apply_external_force_torque_extremities.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            body_names=[
                ".*_hand_yaw_link",
                ".*_ankle_roll_link",
            ],
        )
        self.events.reset_base.params.update(
            {
                "standing_ratio": 0.1,
                "height_offset": 0.03,
                "random_fallen_ratio": 0.25,
            }
        )

        # Adapt inherited success thresholds to the taller HU_D03.
        self.curriculum.terrain_levels.params["min_height"] = standing_threshold
        self.curriculum.adaptive_lift.params["standing_height_threshold"] = (
            hu_d03.DEFAULT_BASE_HEIGHT - 0.1
        )

        # The larger robot gets extra time to complete a recovery sequence.
        self.episode_length_s = 20.0
