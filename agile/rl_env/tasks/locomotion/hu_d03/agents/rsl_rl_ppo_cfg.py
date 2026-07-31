# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from agile.rl_env.mdp.symmetry import lr_mirror_HU_D03
from agile.rl_env.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)


@configclass
class HUD03VelocityPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for HU_D03 lower-body velocity tracking."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 50_000
    save_interval = 250
    experiment_name = "velocity_hu_d03_lower"
    run_name = "velocity_hu_d03_lower"
    wandb_project = "Velocity-HU-D03-Lower"
    empirical_normalization = False
    enable_entropy_coef_annealing = True
    entropy_coef_annealing_start_progress = 0.3
    enable_entropy_coef_annealing_success_rate = 0.8
    entropy_annealing_decay_rate = 0.9995
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
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=lr_mirror_HU_D03,
        ),
    )


@configclass
class HUD03VelocityBootstrapPpoRunnerCfg(HUD03VelocityPpoRunnerCfg):
    """Lower-noise PPO setup that makes the deterministic policy learn the gait."""

    save_interval = 100
    enable_entropy_coef_annealing = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=lr_mirror_HU_D03,
        ),
    )
