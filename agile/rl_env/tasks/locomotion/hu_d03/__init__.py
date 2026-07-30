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
