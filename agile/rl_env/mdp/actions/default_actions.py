# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointAction, JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class DefaultJointPositionAction(JointAction):
    """Hold selected joints at their articulation default positions without policy inputs."""

    cfg: actions_cfg.DefaultJointPositionActionCfg

    def __init__(self, cfg: actions_cfg.DefaultJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._default_positions = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._processed_actions = self._default_positions.clone()
        self._export_IO_descriptor = False

    @property
    def action_dim(self) -> int:
        """This helper term does not consume policy actions."""
        return 0

    def process_actions(self, actions: torch.Tensor) -> None:
        """Keep the fixed targets unchanged."""
        if actions.shape[1] != 0:
            raise ValueError(f"Expected an empty action slice, received shape {tuple(actions.shape)}.")

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Restore the default targets for reset environments."""
        if env_ids is None:
            env_ids = slice(None)
        self._processed_actions[env_ids] = self._default_positions[env_ids]

    def apply_actions(self) -> None:
        """Write the default targets before every physics step."""
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)


class PhaseReferenceJointPositionAction(JointPositionAction):
    """Add a deterministic anti-phase leg reference to learned residual actions."""

    cfg: actions_cfg.PhaseReferenceJointPositionActionCfg

    def __init__(self, cfg: actions_cfg.PhaseReferenceJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._env = env
        required_names = (
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
        )
        missing = [name for name in required_names if name not in self._joint_names]
        if missing:
            raise ValueError(f"Phase reference joints are missing from the action term: {missing}")
        self._phase_joint_indices = {
            name: self._joint_names.index(name) for name in required_names
        }

    def process_actions(self, actions: torch.Tensor) -> None:
        """Apply learned residuals around a command-gated periodic leg target."""
        super().process_actions(actions)

        steps = self._env.episode_length_buf.float()
        phase = 2.0 * torch.pi * self.cfg.frequency * steps * self._env.step_dt
        phase_sin = torch.sin(phase)
        phase_cos = torch.cos(phase)
        left_swing = torch.relu(-phase_sin)
        right_swing = torch.relu(phase_sin)
        command_speed = torch.norm(
            self._env.command_manager.get_command(self.cfg.command_name)[:, :2], dim=1
        )
        gait_scale = torch.clamp(command_speed / self.cfg.reference_speed, 0.0, 1.25)
        phase_sin = phase_sin * gait_scale
        phase_cos = phase_cos * gait_scale
        left_swing = left_swing * gait_scale
        right_swing = right_swing * gait_scale

        idx = self._phase_joint_indices
        self._processed_actions[:, idx["left_hip_pitch_joint"]] += (
            -self.cfg.hip_pitch_amplitude * phase_sin
        )
        self._processed_actions[:, idx["right_hip_pitch_joint"]] += (
            self.cfg.hip_pitch_amplitude * phase_sin
        )
        self._processed_actions[:, idx["left_knee_joint"]] += self.cfg.knee_amplitude * left_swing
        self._processed_actions[:, idx["right_knee_joint"]] += self.cfg.knee_amplitude * right_swing
        self._processed_actions[:, idx["left_ankle_pitch_joint"]] += (
            -self.cfg.ankle_pitch_amplitude * left_swing
        )
        self._processed_actions[:, idx["right_ankle_pitch_joint"]] += (
            -self.cfg.ankle_pitch_amplitude * right_swing
        )
        yaw_phase_correction = self.cfg.yaw_phase_amplitude * phase_cos
        self._processed_actions[:, idx["left_hip_yaw_joint"]] += yaw_phase_correction
        self._processed_actions[:, idx["right_hip_yaw_joint"]] += yaw_phase_correction

        if self.cfg.yaw_rate_damping_gain != 0.0 and self.cfg.yaw_rate_damping_limit > 0.0:
            commanded_yaw_rate = self._env.command_manager.get_command(self.cfg.command_name)[:, 2]
            yaw_rate_error = self._asset.data.root_ang_vel_b[:, 2] - commanded_yaw_rate
            yaw_correction = torch.clamp(
                self.cfg.yaw_rate_damping_gain * yaw_rate_error,
                min=-self.cfg.yaw_rate_damping_limit,
                max=self.cfg.yaw_rate_damping_limit,
            )
            self._processed_actions[:, idx["left_hip_yaw_joint"]] += yaw_correction
            self._processed_actions[:, idx["right_hip_yaw_joint"]] += yaw_correction
