# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from agile.rl_env.rsl_rl import (
    RslRlL2C2Cfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRewardNormalizationCfg,
)


@configclass
class HUD03FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for HU_D03 full-body motion tracking."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 250
    experiment_name = "hu_d03_flat_tracking"
    run_name = "hu_d03_dance"
    wandb_project = "Tracking-HU-D03"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
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
        reward_normalization_cfg=RslRlRewardNormalizationCfg(
            decay=0.999,
            epsilon=1e-2,
        ),
        l2c2_cfg=RslRlL2C2Cfg(
            lambda_actor=1.0,
            lambda_critic=0.1,
        ),
    )
