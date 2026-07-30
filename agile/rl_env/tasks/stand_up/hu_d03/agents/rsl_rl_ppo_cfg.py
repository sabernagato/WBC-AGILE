# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from agile.rl_env.mdp.events import FallenStateDatasetCfg
from agile.rl_env.mdp.symmetry import lr_mirror_HU_D03
from agile.rl_env.rsl_rl import (
    RslRlL2C2Cfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRewardNormalizationCfg,
    RslRlSymmetryCfg,
)


@configclass
class HUD03StandUpPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for HU_D03 full-body fall recovery."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 100_000
    save_interval = 250
    experiment_name = "stand_up_hu_d03"
    run_name = "stand_up_hu_d03"
    wandb_project = "StandUp-HU-D03"
    empirical_normalization = False
    enable_entropy_coef_annealing = False
    entropy_coef_annealing_start_progress = 0.2
    enable_entropy_coef_annealing_success_rate = 0.9

    # Start with a repeatable supine recovery problem. A second dataset covers
    # arbitrary orientations and randomized joints; the environment curriculum
    # gradually raises its reset probability.
    fallen_state_dataset_cfg: FallenStateDatasetCfg | None = FallenStateDatasetCfg(
        num_spawns_per_level=2,
        fall_duration_s=1.5,
        spawn_height_offset=0.6,
        spawn_orientation="on_back",
        spawn_pitch_range=(-1.7, -1.4),
        spawn_joint_mode="default",
        initial_lin_vel_range=0.0,
        initial_ang_vel_range=0.0,
    )
    fallen_state_dataset_secondary_cfg: FallenStateDatasetCfg | None = FallenStateDatasetCfg(
        num_spawns_per_level=2,
        fall_duration_s=1.5,
        spawn_height_offset=0.6,
        spawn_orientation="random",
        spawn_joint_mode="random",
        initial_lin_vel_range=0.5,
        initial_ang_vel_range=0.5,
    )

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=lr_mirror_HU_D03,
        ),
        l2c2_cfg=RslRlL2C2Cfg(
            lambda_actor=1.0,
            lambda_critic=0.1,
        ),
        reward_normalization_cfg=RslRlRewardNormalizationCfg(
            decay=0.999,
            epsilon=1e-2,
            return_scale_decay=0.999,
        ),
    )
