# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # noqa: F401

from agile.rl_env import mdp
from agile.rl_env.assets.robots import hu_d03
from agile.rl_env.mdp.terrains import LESS_ROUGH_TERRAIN_CFG, MEDIUM_ROUGH_TERRAIN_CFG  # noqa: F401

# Define controlled joints for HU_D03 (legs + waist roll/pitch for locomotion)
CONTROLLED_JOINT_NAMES = hu_d03.CONTROLLED_JOINT_NAMES
UPPER_BODY_HOLD_JOINT_NAMES = hu_d03.UPPER_BODY_HOLD_JOINT_NAMES

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane" or "generator"
        terrain_generator=MEDIUM_ROUGH_TERRAIN_CFG,  # None or LESS_ROUGH_TERRAIN_CFG, MEDIUM_ROUGH_TERRAIN_CFG
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                f"TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot = hu_d03.HU_D03_DELAYED_DC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )

    height_measurement_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.0, 0.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
    )

    height_measurement_sensor_left_foot = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.01, 0.0, 1.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.2, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
    )
    height_measurement_sensor_right_foot = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.01, 0.0, 1.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.2, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformNullVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.10,
        rel_single_axis_envs=0.5,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformNullVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=CONTROLLED_JOINT_NAMES,
        scale=hu_d03.HU_D03_ACTION_SCALE_LOWER,
        use_default_offset=True,
        clip={".*": (-10.0, 10.0)},
    )

    # This zero-dimensional helper keeps the remaining joints at their default
    # targets without changing the 14-DoF policy action interface.
    upper_body_default = mdp.DefaultJointPositionActionCfg(
        asset_name="robot",
        joint_names=UPPER_BODY_HOLD_JOINT_NAMES,
        scale=1.0,
    )


@configclass
class PhaseHarnessActionsCfg(ActionsCfg):
    """Lower-body actions plus a zero-dimensional training-only harness."""

    harness = mdp.HarnessActionCfg(
        asset_name="robot",
        root_name=hu_d03.ROOT_BODY_NAME,
        stiffness_torques=1000.0,
        damping_torques=100.0,
        stiffness_forces=3000.0,
        damping_forces=300.0,
        force_limit=600.0,
        torque_limit=1000.0,
        height_sensor="height_measurement_sensor",
        target_height=hu_d03.DEFAULT_BASE_HEIGHT,
        command_name=None,
    )


@configclass
class PhaseResidualHarnessActionsCfg(PhaseHarnessActionsCfg):
    """Learned lower-body residuals around an anti-phase gait reference."""

    joint_pos = mdp.PhaseReferenceJointPositionActionCfg(
        asset_name="robot",
        joint_names=CONTROLLED_JOINT_NAMES,
        scale=hu_d03.HU_D03_ACTION_SCALE_LOWER,
        use_default_offset=True,
        clip={".*": (-10.0, 10.0)},
        frequency=1.5,
        command_name="base_velocity",
        reference_speed=0.45,
        hip_pitch_amplitude=0.25,
        knee_amplitude=0.40,
        ankle_pitch_amplitude=0.20,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class HistoryPolicyCfg(ObsGroup):
        """Observations for policy group with history."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        controlled_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
        )
        controlled_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = False
            self.flatten_history_dim = False

    @configclass
    class PrivilegedVelocityCriticCfg(ObsGroup):
        """Observations for policy group."""

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: HistoryPolicyCfg = HistoryPolicyCfg()
    critic: PrivilegedVelocityCriticCfg = PrivilegedVelocityCriticCfg()


@configclass
class PhaseObservationsCfg(ObservationsCfg):
    """Velocity observations extended with an explicit biped gait clock."""

    @configclass
    class PhaseHistoryPolicyCfg(ObservationsCfg.HistoryPolicyCfg):
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"frequency": 1.5})

    @configclass
    class PhasePrivilegedVelocityCriticCfg(ObservationsCfg.PrivilegedVelocityCriticCfg):
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"frequency": 1.5})

    policy: PhaseHistoryPolicyCfg = PhaseHistoryPolicyCfg()
    critic: PhasePrivilegedVelocityCriticCfg = PhasePrivilegedVelocityCriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "std": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped_command,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "command_slice": slice(0, 2),
            "threshold": 0.4,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=hu_d03.FEET_LINK_NAMES),
        },
    )

    base_height = RewTerm(
        func=mdp.base_height_exp,
        weight=2.5,
        params={
            "target_height": hu_d03.DEFAULT_BASE_HEIGHT,
            "std": 0.1,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )

    orientation = RewTerm(
        func=mdp.flat_body_orientation_exp,
        weight=5.0,
        params={
            "std": math.radians(10.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link", "waist_pitch_link"]),
        },
    )

    torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
    )

    ankle_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*ankle.*")},
    )

    ankle_roll_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*ankle_roll.*")},
    )

    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    dof_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
    )

    action_rate_rate = RewTerm(
        func=mdp.action_rate_rate_l2,
        weight=-0.025,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
    )

    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
    )

    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES), "soft_ratio": 0.9},
    )

    torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
    )

    feet_slip = RewTerm(
        func=mdp.feet_slip,
        weight=-0.05,
        params={
            "contact_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=hu_d03.FEET_LINK_NAMES),
            "robot_cfg": SceneEntityCfg("robot", body_names=hu_d03.FEET_LINK_NAMES),
        },
    )

    feet_roll = RewTerm(
        func=mdp.feet_roll_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=hu_d03.FEET_LINK_NAMES)},
    )

    feet_yaw_diff = RewTerm(
        func=mdp.feet_yaw_diff_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=hu_d03.FEET_LINK_NAMES)},
    )

    feet_yaw_mean = RewTerm(
        func=mdp.feet_yaw_mean_vs_base,
        weight=-2.0,
        params={
            "feet_asset_cfg": SceneEntityCfg("robot", body_names=hu_d03.FEET_LINK_NAMES),
            "base_body_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    root_acc = RewTerm(
        func=mdp.body_acc_l2,  # type: ignore
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    feet_distance = RewTerm(
        func=mdp.feet_distance_from_ref,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=hu_d03.FEET_LINK_NAMES),
            "ref_distance": 0.2,
        },
    )

    jumping = RewTerm(
        func=mdp.jumping,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=hu_d03.FEET_LINK_NAMES),
            "threshold": 10.0,
        },
    )


@configclass
class PhaseRewardsCfg(RewardsCfg):
    """Bootstrap rewards that require actual alternating single support."""

    gait_load_transfer = RewTerm(
        func=mdp.biped_gait_load_transfer,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "frequency": 1.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=hu_d03.FEET_LINK_NAMES),
        },
    )

    gait_contact_schedule = RewTerm(
        func=mdp.biped_gait_contact_schedule,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "frequency": 1.5,
            "force_threshold": 10.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=hu_d03.FEET_LINK_NAMES),
        },
    )

    gait_foot_clearance = RewTerm(
        func=mdp.biped_gait_foot_clearance,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "frequency": 1.5,
            "target_clearance": 0.08,
            "std": 0.06,
            "asset_cfg": SceneEntityCfg("robot", body_names=hu_d03.FEET_LINK_NAMES),
        },
    )


@configclass
class PhaseReferenceRewardsCfg(PhaseRewardsCfg):
    """Phase rewards plus a direct anti-phase leg reference scaffold."""

    gait_joint_reference = RewTerm(
        func=mdp.biped_gait_joint_reference,
        weight=25.0,
        params={
            "command_name": "base_velocity",
            "frequency": 1.5,
            "std": 0.15,
            "left_joint_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_hip_pitch_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                ],
            ),
            "right_joint_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "right_hip_pitch_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                ],
            ),
            "hip_pitch_amplitude": 0.25,
            "knee_amplitude": 0.40,
            "ankle_pitch_amplitude": 0.20,
        },
    )


@configclass
class PhaseResidualAdaptiveRewardsCfg(PhaseReferenceRewardsCfg):
    """Residual gait rewards with a non-saturating upright penalty."""

    orientation_tilt_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["base_link", "waist_pitch_link"])},
    )


@configclass
class PhaseResidualAdaptiveYawRewardsCfg(PhaseResidualAdaptiveRewardsCfg):
    """Adaptive gait rewards with direct yaw-rate error shaping."""

    yaw_rate_error_l2 = RewTerm(
        func=mdp.track_ang_vel_z_l2,
        weight=-2.0,
        params={"command_name": "base_velocity"},
    )

@configclass
class PhaseResidualHarnessCurriculumCfg:
    """Linearly remove the training harness after a short adaptation window."""

    remove_harness = CurrTerm(
        func=mdp.remove_harness,
        params={
            "harness_action_name": "harness",
            "start": 240,
            "num_steps": 4320,
        },
    )


@configclass
class PhaseResidualAdaptiveHarnessCurriculumCfg:
    """Remove assistance only while the batch remains nearly upright."""

    adaptive_harness = CurrTerm(
        func=mdp.adaptive_force_decay,
        params={
            "action_name": "harness",
            "metric_name": "orientation_error",
            "decay_when": "below",
            "threshold": 0.12,
            "ema_alpha": 0.01,
            "decay": 0.9995,
            "disable_threshold": 0.005,
            "initial_scale": 0.5,
        },
    )


@configclass
class PhaseResidualAdaptiveLowHarnessCurriculumCfg:
    """Continue adaptive removal from the validated 27% assistance stage."""

    adaptive_harness = CurrTerm(
        func=mdp.adaptive_force_decay,
        params={
            "action_name": "harness",
            "metric_name": "orientation_error",
            "decay_when": "below",
            "threshold": 0.12,
            "ema_alpha": 0.01,
            "decay": 0.999,
            "disable_threshold": 0.005,
            "initial_scale": 0.27,
        },
    )

@configclass
class PhaseResidualMixedHarnessCurriculumCfg:
    """Keep supported environments at 27% assistance while zero-support peers train beside them."""

    mixed_harness = CurrTerm(
        func=mdp.adaptive_force_decay,
        params={
            "action_name": "harness",
            "metric_name": "orientation_error",
            "decay_when": "below",
            "threshold": 0.12,
            "ema_alpha": 0.01,
            "decay": 1.0,
            "disable_threshold": 0.005,
            "initial_scale": 0.27,
        },
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_pitch_link"),
            "limit_angle": math.radians(30.0),
        },
    )

    illegal_contacts = DoneTerm(
        func=mdp.illegal_ground_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base_link", "waist_pitch_link"],
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link", "waist_pitch_link"]),
            "threshold": 20.0,
            "min_height": 0.55,
        },
    )

    illegal_base_height = DoneTerm(
        func=mdp.illegal_base_height,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "height_threshold": hu_d03.DEFAULT_BASE_HEIGHT - 0.3,
        },
    )


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    eye: tuple[float, float, float] = (0.0, -5.0, 2.0)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.5)
    cam_prim_path: str = "/OmniverseKit_Persp"
    resolution: tuple[int, int] = (1280, 720)
    origin_type = "asset_root"
    asset_name: str = "robot"
    env_index: int = 0


##
# Environment configuration
##


@configclass
class LocomotionEventCfg:
    """Configuration for events."""

    randomize_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.2, 1.5),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 2.0),
            "operation": "scale",
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
            "friction_distribution_params": (0.0, 0.005),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    randomize_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_bodies_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )

    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 5.0),
            "operation": "add",
        },
    )

    randomize_bodies_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        },
    )

    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.15, 0.25), "y": (-0.05, 0.05), "z": (-0.15, 0.15)},
        },
    )

    apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-5.0, 5.0),
        },
    )

    apply_external_force_torque_extremities = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*wrist_yaw_link.*", ".*ankle_roll_link.*"]),
            "force_range": (-5.0, 5.0),
            "torque_range": (-0.5, 0.5),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-2.5, 2.5),
                "y": (-2.5, 2.5),
                "z": (-0.0, 0.0),
                "yaw": (-3.14, 3.14),
                "roll": (-math.radians(10), math.radians(10)),
                "pitch": (-math.radians(10), math.radians(10)),
            },
            "velocity_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "z": (-0.0, 0.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll_link.*"]),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (-1.0, 1.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
        },
    )

    reset_upper_body_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (1.0, 1.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY_HOLD_JOINT_NAMES),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.25, 0.25),
            }
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel_curriculum,
        params={
            "command_name": "base_velocity",
            "move_up_distance": 4.0,
            "move_down_distance": 2.0,
            "n_successes": 4,
            "n_failures": 10,
            "p_random_move_up": 0.00,
            "p_random_move_down": 0.00,
        },
    )

    increase_action_rate_regularization = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "action_rate",
            "start_step": 50_000,
            "num_steps": 100_000,
            "terminal_weight": -1.0,
            "use_log_space": False,
        },
    )

    increase_action_rate_rate_regularization = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "action_rate_rate",
            "start_step": 60_000,
            "num_steps": 100_000,
            "terminal_weight": -0.5,
            "use_log_space": False,
        },
    )


@configclass
class HUD03LowerVelocityHistoryEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the HU_D03 velocity tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: LocomotionEventCfg = LocomotionEventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.controller_freq = 50.0
        self.physics_freq = 200.0
        self.episode_length_s = 30.0
        self.max_episode_length_offset_s = 0.0

        self.decimation = int(self.physics_freq / self.controller_freq)
        self.sim.dt = 1.0 / self.physics_freq
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.solver_type = 1
        self.sim.physx.gpu_max_rigid_patch_count = 2**20

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.height_measurement_sensor is not None:
            self.scene.height_measurement_sensor.update_period = self.sim.dt
        if self.scene.height_measurement_sensor_left_foot is not None:
            self.scene.height_measurement_sensor_left_foot.update_period = self.sim.dt
        if self.scene.height_measurement_sensor_right_foot is not None:
            self.scene.height_measurement_sensor_right_foot.update_period = self.sim.dt

        self.only_positive_rewards = False

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    def eval(self):
        self.observations.eval = mdp.EvaluationObservationsCfg()
        self.observations.policy.enable_corruption = False
        self.rewards = None
        self.curriculum = None
        # Training disturbances must not leak into deterministic evaluation.
        self.events.apply_external_force_torque = None
        self.events.apply_external_force_torque_extremities = None
        self.events.push_robot = None


@configclass
class HUD03LowerVelocityHistoryBootstrapEnvCfg(HUD03LowerVelocityHistoryEnvCfg):
    """Flat-ground curriculum used only to bootstrap a genuine stepping gait."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum = None

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_single_axis_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # The plain exponential gives a stationary policy substantial reward
        # (exp(-(0.2..0.5)^2 / 0.5^2)), which makes standing a strong local
        # optimum. During gait bootstrap, also require the achieved velocity
        # to point along the commanded direction: standing receives zero and
        # backwards motion receives a negative reward.
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp_aligned
        self.rewards.track_lin_vel_xy_exp.weight = 10.0
        self.rewards.track_ang_vel.weight = 2.0
        self.rewards.base_height.weight = 1.5
        self.rewards.orientation.weight = 2.5
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.action_rate.weight = -0.05
        self.rewards.action_rate_rate.weight = -0.005

        self.events.randomize_physics_material = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_joint_friction = None
        self.events.randomize_joint_armature = None
        self.events.randomize_bodies_mass = None
        self.events.randomize_base_mass = None
        self.events.randomize_bodies_com = None
        self.events.randomize_base_com = None
        self.events.apply_external_force_torque = None
        self.events.apply_external_force_torque_extremities = None
        self.events.push_robot = None

        self.events.reset_base.params["pose_range"].update({"roll": (0.0, 0.0), "pitch": (0.0, 0.0)})
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


@configclass
class HUD03LowerVelocityHistoryPhaseBootstrapEnvCfg(HUD03LowerVelocityHistoryBootstrapEnvCfg):
    """Flat gait bootstrap with an explicit phase and dense load-transfer target."""

    observations: PhaseObservationsCfg = PhaseObservationsCfg()
    rewards: PhaseRewardsCfg = PhaseRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Preserve balance while requiring true, phase-matched single support.
        self.rewards.termination_penalty.weight = -200.0
        self.rewards.track_lin_vel_xy_exp.weight = 8.0
        self.rewards.base_height.weight = 3.0
        self.rewards.orientation.weight = 5.0
        self.rewards.feet_air_time.weight = 3.0


@configclass
class HUD03LowerVelocityHistoryPhaseHarnessEnvCfg(
    HUD03LowerVelocityHistoryPhaseBootstrapEnvCfg
):
    """Phase gait bootstrap with training-only height/orientation assistance."""

    actions: PhaseHarnessActionsCfg = PhaseHarnessActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        # With falls temporarily arrested, prioritize learning actual leg
        # alternation rather than merely shifting load between planted feet.
        self.rewards.feet_air_time.weight = 5.0
        self.rewards.gait_contact_schedule.weight = 10.0
        self.rewards.gait_foot_clearance.weight = 10.0

    def eval(self):
        super().eval()
        if hasattr(self.actions, "harness"):
            del self.actions.harness


@configclass
class HUD03LowerVelocityHistoryPhaseReferenceHarnessEnvCfg(
    HUD03LowerVelocityHistoryPhaseHarnessEnvCfg
):
    """Training-only harness plus an explicit alternating leg reference."""

    rewards: PhaseReferenceRewardsCfg = PhaseReferenceRewardsCfg()


@configclass
class HUD03LowerVelocityHistoryPhaseResidualHarnessEnvCfg(
    HUD03LowerVelocityHistoryPhaseReferenceHarnessEnvCfg
):
    """Periodic gait reference with learned residuals and training-only harness."""

    actions: PhaseResidualHarnessActionsCfg = PhaseResidualHarnessActionsCfg()


@configclass
class HUD03LowerVelocityHistoryPhaseResidualHarnessDecayEnvCfg(
    HUD03LowerVelocityHistoryPhaseResidualHarnessEnvCfg
):
    """Residual gait task that linearly removes all harness assistance."""

    curriculum: PhaseResidualHarnessCurriculumCfg = PhaseResidualHarnessCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # The flat bootstrap parent disables its terrain curriculum. Restore only harness removal.
        self.curriculum = PhaseResidualHarnessCurriculumCfg()


@configclass
class HUD03LowerVelocityHistoryPhaseResidualHarnessAdaptiveEnvCfg(
    HUD03LowerVelocityHistoryPhaseResidualHarnessEnvCfg
):
    """Residual gait task with performance-gated harness removal."""

    rewards: PhaseResidualAdaptiveRewardsCfg = PhaseResidualAdaptiveRewardsCfg()
    curriculum: PhaseResidualAdaptiveHarnessCurriculumCfg = PhaseResidualAdaptiveHarnessCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # The flat bootstrap parent disables its terrain curriculum. Restore only
        # performance-gated harness removal and strengthen the cost of falling.
        self.curriculum = PhaseResidualAdaptiveHarnessCurriculumCfg()
        self.rewards.termination_penalty.weight = -500.0


@configclass
class HUD03LowerVelocityHistoryPhaseResidualHarnessAdaptiveLowEnvCfg(
    HUD03LowerVelocityHistoryPhaseResidualHarnessAdaptiveEnvCfg
):
    """Low-assistance continuation with tighter velocity and yaw tracking."""

    rewards: PhaseResidualAdaptiveYawRewardsCfg = PhaseResidualAdaptiveYawRewardsCfg()
    curriculum: PhaseResidualAdaptiveLowHarnessCurriculumCfg = PhaseResidualAdaptiveLowHarnessCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Continue from the assistance level reached by the previous stage.
        self.curriculum = PhaseResidualAdaptiveLowHarnessCurriculumCfg()
        self.rewards.track_ang_vel.weight = 5.0
        self.rewards.track_lin_vel_xy_exp.weight = 12.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.3


@configclass
class HUD03LowerVelocityHistoryPhaseResidualHarnessMixedEnvCfg(
    HUD03LowerVelocityHistoryPhaseResidualHarnessAdaptiveLowEnvCfg
):
    """Train one shared policy on supported and fully unassisted environments."""

    curriculum: PhaseResidualMixedHarnessCurriculumCfg = PhaseResidualMixedHarnessCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.curriculum = PhaseResidualMixedHarnessCurriculumCfg()
        self.actions.harness.unassisted_env_fraction = 0.25

        amplitudes = {
            "hip_pitch_amplitude": 0.1875,
            "knee_amplitude": 0.30,
            "ankle_pitch_amplitude": 0.15,
        }
        for name, value in amplitudes.items():
            setattr(self.actions.joint_pos, name, value)
            self.rewards.gait_joint_reference.params[name] = value
