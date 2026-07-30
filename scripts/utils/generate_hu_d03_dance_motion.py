#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""Generate a small, licensed HU_D03 starter dance reference with MuJoCo FK.

This is an integration clip, not a production choreography. Replace it with a
retargeted human or authored motion after the training pipeline is verified.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agile.common.hu_d03_motion import (  # noqa: E402
    HU_D03_MOTION_FPS,
    HU_D03_MOTION_JOINT_NAMES,
    HU_D03_TRACKED_BODY_NAMES,
)


def _default_description_root() -> Path:
    return REPO_ROOT.parent / "limx_oli_description" / "HU_D03_description"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", type=Path, required=True, help="Output .npz path.")
    parser.add_argument(
        "--description-root",
        type=Path,
        default=Path(os.environ.get("HU_D03_DESCRIPTION_ROOT", _default_description_root())),
        help="Path to HU_D03_description.",
    )
    parser.add_argument("--duration", type=float, default=8.0, help="Clip duration in seconds.")
    parser.add_argument("--fps", type=int, default=HU_D03_MOTION_FPS, help="Output rate; keep at 50 for training.")
    return parser.parse_args()


def _joint_pose(time_s: float, duration: float) -> dict[str, float]:
    """Return a loop-friendly whole-body warm-up dance pose."""

    beat = 2.0 * np.pi * 0.75 * time_s
    phrase = 2.0 * np.pi * time_s / duration
    bounce = 0.5 - 0.5 * np.cos(2.0 * beat)
    sway = np.sin(beat)
    arm_wave = np.sin(2.0 * beat)

    pose = dict.fromkeys(HU_D03_MOTION_JOINT_NAMES, 0.0)

    for side in ("left", "right"):
        side_sign = 1.0 if side == "left" else -1.0
        pose[f"{side}_hip_pitch_joint"] = -0.15 - 0.10 * bounce
        pose[f"{side}_hip_roll_joint"] = 0.05 * sway
        pose[f"{side}_hip_yaw_joint"] = 0.04 * side_sign * np.sin(phrase)
        pose[f"{side}_knee_joint"] = 0.30 + 0.20 * bounce
        pose[f"{side}_ankle_pitch_joint"] = -0.15 - 0.10 * bounce
        pose[f"{side}_ankle_roll_joint"] = -0.04 * sway

    pose["waist_yaw_joint"] = 0.18 * np.sin(phrase)
    pose["waist_roll_joint"] = 0.08 * sway
    pose["waist_pitch_joint"] = 0.04 * np.sin(2.0 * beat)
    pose["head_yaw_joint"] = -0.12 * np.sin(phrase)
    pose["head_pitch_joint"] = 0.06 * np.sin(beat + np.pi / 4.0)

    pose["left_shoulder_pitch_joint"] = 0.25 * np.sin(phrase)
    pose["right_shoulder_pitch_joint"] = -0.25 * np.sin(phrase)
    pose["left_shoulder_roll_joint"] = 0.55 + 0.28 * arm_wave
    pose["right_shoulder_roll_joint"] = -0.55 - 0.28 * arm_wave
    pose["left_shoulder_yaw_joint"] = 0.22 * np.sin(beat + np.pi / 2.0)
    pose["right_shoulder_yaw_joint"] = -0.22 * np.sin(beat + np.pi / 2.0)
    pose["left_elbow_joint"] = 0.45 + 0.22 * np.sin(2.0 * beat + np.pi / 2.0)
    pose["right_elbow_joint"] = 0.45 - 0.22 * np.sin(2.0 * beat + np.pi / 2.0)
    pose["left_wrist_yaw_joint"] = 0.25 * np.sin(2.0 * beat)
    pose["right_wrist_yaw_joint"] = -0.25 * np.sin(2.0 * beat)
    pose["left_wrist_pitch_joint"] = 0.18 * np.sin(beat)
    pose["right_wrist_pitch_joint"] = 0.18 * np.sin(beat)
    pose["left_hand_yaw_joint"] = 0.35 * np.sin(2.0 * beat + np.pi / 3.0)
    pose["right_hand_yaw_joint"] = -0.35 * np.sin(2.0 * beat + np.pi / 3.0)
    return pose


def _angular_velocity(quaternions_wxyz: np.ndarray, dt: float) -> np.ndarray:
    frame_count, body_count, _ = quaternions_wxyz.shape
    velocity = np.zeros((frame_count, body_count, 3), dtype=np.float32)
    for body_index in range(body_count):
        quaternions_xyzw = quaternions_wxyz[:, body_index, [1, 2, 3, 0]]
        rotations = Rotation.from_quat(quaternions_xyzw)
        relative = rotations[2:] * rotations[:-2].inv()
        central = relative.as_rotvec() / (2.0 * dt)
        velocity[1:-1, body_index] = central
        velocity[0, body_index] = central[0]
        velocity[-1, body_index] = central[-1]
    return velocity


def generate_motion(xml_path: Path, duration: float, fps: int) -> dict[str, np.ndarray]:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "The generator needs the project dependency 'mujoco>=3.3.6'. "
            "Install WBC-AGILE dependencies first."
        ) from exc

    if not xml_path.is_file():
        raise FileNotFoundError(f"HU_D03 MJCF not found: {xml_path}")
    if duration < 2.0:
        raise ValueError("duration must be at least 2 seconds")
    if fps != HU_D03_MOTION_FPS:
        raise ValueError(f"fps must be {HU_D03_MOTION_FPS} for Tracking-Flat-HU-D03-v0")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    joint_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in HU_D03_MOTION_JOINT_NAMES
    }
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in HU_D03_TRACKED_BODY_NAMES
    ]
    missing_joints = [name for name, joint_id in joint_ids.items() if joint_id < 0]
    missing_bodies = [
        name for name, body_id in zip(HU_D03_TRACKED_BODY_NAMES, body_ids, strict=True) if body_id < 0
    ]
    if missing_joints or missing_bodies:
        raise ValueError(f"MJCF names missing: joints={missing_joints}, bodies={missing_bodies}")

    frame_count = max(3, int(round(duration * fps)))
    times = np.arange(frame_count, dtype=np.float64) / fps
    joint_pos = np.zeros((frame_count, len(HU_D03_MOTION_JOINT_NAMES)), dtype=np.float32)
    body_pos = np.zeros((frame_count, len(HU_D03_TRACKED_BODY_NAMES), 3), dtype=np.float32)
    body_quat = np.zeros((frame_count, len(HU_D03_TRACKED_BODY_NAMES), 4), dtype=np.float32)

    for frame_index, time_s in enumerate(times):
        data.qpos[:] = model.qpos0
        pose = _joint_pose(float(time_s), duration)
        for joint_index, joint_name in enumerate(HU_D03_MOTION_JOINT_NAMES):
            joint_id = joint_ids[joint_name]
            qpos_address = model.jnt_qposadr[joint_id]
            data.qpos[qpos_address] = pose[joint_name]
            joint_pos[frame_index, joint_index] = pose[joint_name]

        mujoco.mj_forward(model, data)
        body_pos[frame_index] = data.xpos[body_ids]
        body_quat[frame_index] = data.xquat[body_ids]

    dt = 1.0 / fps
    return {
        "fps": np.asarray(fps, dtype=np.float32),
        "joint_pos": joint_pos,
        "joint_vel": np.gradient(joint_pos, dt, axis=0).astype(np.float32),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": np.gradient(body_pos, dt, axis=0).astype(np.float32),
        "body_ang_vel_w": _angular_velocity(body_quat, dt),
        "joint_names": np.asarray(HU_D03_MOTION_JOINT_NAMES),
        "body_names": np.asarray(HU_D03_TRACKED_BODY_NAMES),
    }


def main() -> int:
    args = _parse_args()
    description_root = args.description_root.expanduser().resolve()
    xml_path = description_root / "xml" / "HU_D03_03.xml"
    motion = generate_motion(xml_path, args.duration, args.fps)

    output_path = args.output_file.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **motion)
    print(
        f"Saved HU_D03 starter dance: {output_path} "
        f"({motion['joint_pos'].shape[0]} frames @ {args.fps} fps)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
