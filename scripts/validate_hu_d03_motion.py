#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""Validate a motion clip before HU_D03 reference-tracking training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agile.common.hu_d03_motion import (  # noqa: E402
    HU_D03_MOTION_FPS,
    HU_D03_MOTION_JOINT_NAMES,
    HU_D03_MOTION_KEYS,
    HU_D03_TRACKED_BODY_NAMES,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_file", type=Path, help="HU_D03 tracking motion in .npz format.")
    return parser.parse_args()


def _decode_names(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values.reshape(-1).tolist()
    ]


def validate_motion(path: Path) -> tuple[list[str], list[str]]:
    """Return errors and warnings for one HU_D03 motion file."""

    errors: list[str] = []
    warnings: list[str] = []
    if not path.is_file():
        return [f"Motion file not found: {path}"], warnings

    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        return [f"Unable to load motion file: {exc}"], warnings

    missing_keys = [key for key in HU_D03_MOTION_KEYS if key not in data]
    if missing_keys:
        errors.append(f"Missing keys: {missing_keys}")
        return errors, warnings

    try:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    except (TypeError, ValueError, IndexError) as exc:
        errors.append(f"Invalid fps value: {exc}")
        fps = 0.0
    if not np.isfinite(fps) or not np.isclose(fps, HU_D03_MOTION_FPS):
        errors.append(
            f"Expected {HU_D03_MOTION_FPS} fps (the task control rate), found {fps!r}"
        )

    joint_names = _decode_names(data["joint_names"])
    body_names = _decode_names(data["body_names"])
    if joint_names != HU_D03_MOTION_JOINT_NAMES:
        errors.append(
            "joint_names do not match the canonical 31-joint HU_D03 order. "
            "Regenerate or explicitly reorder the clip."
        )
    if body_names != HU_D03_TRACKED_BODY_NAMES:
        errors.append(
            "body_names do not match the canonical 15-body HU_D03 tracking order. "
            "Regenerate or explicitly reorder the clip."
        )

    joint_pos = np.asarray(data["joint_pos"])
    joint_vel = np.asarray(data["joint_vel"])
    body_pos = np.asarray(data["body_pos_w"])
    body_quat = np.asarray(data["body_quat_w"])
    body_lin_vel = np.asarray(data["body_lin_vel_w"])
    body_ang_vel = np.asarray(data["body_ang_vel_w"])

    if joint_pos.ndim != 2:
        errors.append(f"joint_pos must have shape (frames, 31), found {joint_pos.shape}")
        frame_count = 0
    else:
        frame_count = joint_pos.shape[0]
        if joint_pos.shape[1] != len(HU_D03_MOTION_JOINT_NAMES):
            errors.append(f"joint_pos must have 31 joints, found {joint_pos.shape[1]}")
        if frame_count < 3:
            errors.append(f"Motion must contain at least 3 frames, found {frame_count}")

    expected_shapes = {
        "joint_vel": (frame_count, len(HU_D03_MOTION_JOINT_NAMES)),
        "body_pos_w": (frame_count, len(HU_D03_TRACKED_BODY_NAMES), 3),
        "body_quat_w": (frame_count, len(HU_D03_TRACKED_BODY_NAMES), 4),
        "body_lin_vel_w": (frame_count, len(HU_D03_TRACKED_BODY_NAMES), 3),
        "body_ang_vel_w": (frame_count, len(HU_D03_TRACKED_BODY_NAMES), 3),
    }
    arrays = {
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
    }
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            errors.append(f"{name} must have shape {expected_shape}, found {arrays[name].shape}")

    for name, array in {"joint_pos": joint_pos, **arrays}.items():
        if not np.issubdtype(array.dtype, np.number):
            errors.append(f"{name} must be numeric, found dtype {array.dtype}")
        elif not np.all(np.isfinite(array)):
            errors.append(f"{name} contains NaN or infinity")

    if body_quat.shape == expected_shapes["body_quat_w"] and np.issubdtype(
        body_quat.dtype, np.number
    ):
        quat_norm_error = np.max(np.abs(np.linalg.norm(body_quat, axis=-1) - 1.0))
        if quat_norm_error > 1.0e-3:
            errors.append(
                f"body_quat_w is not normalized; maximum norm error is {quat_norm_error:.3e}"
            )

    if frame_count and fps > 0:
        duration = (frame_count - 1) / fps
        if duration < 2.0:
            warnings.append(
                f"Clip duration is only {duration:.2f}s; use a longer clip for useful dance training."
            )

    return errors, warnings


def main() -> int:
    args = _parse_args()
    motion_path = args.motion_file.expanduser().resolve()
    errors, warnings = validate_motion(motion_path)

    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    data = np.load(motion_path, allow_pickle=False)
    frame_count = int(data["joint_pos"].shape[0])
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    print(
        f"HU_D03 motion OK: {frame_count} frames, {frame_count / fps:.2f}s, "
        f"{len(HU_D03_MOTION_JOINT_NAMES)} joints, {len(HU_D03_TRACKED_BODY_NAMES)} bodies"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
