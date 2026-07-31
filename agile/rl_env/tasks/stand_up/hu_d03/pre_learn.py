# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""HU_D03 fallen-state collection hook.

The implementation is robot-agnostic. Re-exporting it locally keeps the task
registration self-contained while sharing the dual-dataset cache pipeline.
"""

from agile.rl_env.tasks.stand_up.g1.pre_learn import pre_learn

__all__ = ["pre_learn"]
