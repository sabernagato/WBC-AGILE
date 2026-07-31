# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

gym.register(
    id="Velocity-HU-D03-History-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_history_env_cfg:HUD03LowerVelocityHistoryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HUD03VelocityPpoRunnerCfg",
    },
)

gym.register(
    id="Velocity-HU-D03-History-Bootstrap-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_history_env_cfg:"
            "HUD03LowerVelocityHistoryBootstrapEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:HUD03VelocityBootstrapPpoRunnerCfg"
        ),
    },
)

gym.register(
    id="Velocity-HU-D03-History-Bootstrap-Explore-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_history_env_cfg:"
            "HUD03LowerVelocityHistoryBootstrapEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:HUD03VelocityPpoRunnerCfg"
        ),
    },
)

gym.register(
    id="Velocity-HU-D03-History-Bootstrap-Phase-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_history_env_cfg:"
            "HUD03LowerVelocityHistoryPhaseBootstrapEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:HUD03VelocityBootstrapPpoRunnerCfg"
        ),
    },
)

gym.register(
    id="Velocity-HU-D03-History-Bootstrap-Phase-Harness-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_history_env_cfg:"
            "HUD03LowerVelocityHistoryPhaseHarnessEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:HUD03VelocityBootstrapPpoRunnerCfg"
        ),
    },
)

gym.register(
    id="Velocity-HU-D03-History-Bootstrap-Phase-Reference-Harness-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_history_env_cfg:"
            "HUD03LowerVelocityHistoryPhaseReferenceHarnessEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:HUD03VelocityBootstrapPpoRunnerCfg"
        ),
    },
)

gym.register(
    id="Velocity-HU-D03-History-Bootstrap-Phase-Residual-Harness-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_history_env_cfg:"
            "HUD03LowerVelocityHistoryPhaseResidualHarnessEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:HUD03VelocityBootstrapPpoRunnerCfg"
        ),
    },
)

gym.register(
    id="Velocity-HU-D03-History-Bootstrap-Phase-Residual-Harness-Decay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_history_env_cfg:"
            "HUD03LowerVelocityHistoryPhaseResidualHarnessDecayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:HUD03VelocityBootstrapPpoRunnerCfg"
        ),
    },
)


gym.register(
    id="Velocity-HU-D03-History-Bootstrap-Phase-Residual-Harness-Adaptive-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_history_env_cfg:"
            "HUD03LowerVelocityHistoryPhaseResidualHarnessAdaptiveEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:HUD03VelocityBootstrapPpoRunnerCfg"
        ),
    },
)


gym.register(
    id="Velocity-HU-D03-History-Bootstrap-Phase-Residual-Harness-Adaptive-Low-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_history_env_cfg:"
            "HUD03LowerVelocityHistoryPhaseResidualHarnessAdaptiveLowEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:HUD03VelocityBootstrapPpoRunnerCfg"
        ),
    },
)

gym.register(
    id="Velocity-HU-D03-History-Bootstrap-Phase-Residual-Harness-Mixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_history_env_cfg:"
            "HUD03LowerVelocityHistoryPhaseResidualHarnessMixedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:HUD03VelocityBootstrapPpoRunnerCfg"
        ),
    },
)
