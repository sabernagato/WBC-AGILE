#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""Validate HU_D03 assets needed by the WBC-AGILE locomotion task."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

EXPECTED_ACTIVE_JOINTS = {
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "hip_pitch",
        "hip_roll",
        "hip_yaw",
        "knee",
        "ankle_pitch",
        "ankle_roll",
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_yaw",
        "wrist_pitch",
        "hand_yaw",
    )
}
EXPECTED_ACTIVE_JOINTS.update(
    {
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "head_yaw_joint",
        "head_pitch_joint",
    }
)

POLICY_JOINTS = {
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "hip_pitch",
        "hip_roll",
        "hip_yaw",
        "knee",
        "ankle_pitch",
        "ankle_roll",
    )
}
POLICY_JOINTS.update({"waist_roll_joint", "waist_pitch_joint"})


def _default_description_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "limx_oli_description"
        / "HU_D03_description"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--description-root",
        type=Path,
        default=Path(
            os.environ.get("HU_D03_DESCRIPTION_ROOT", _default_description_root())
        ),
        help="Path to HU_D03_description.",
    )
    return parser.parse_args()


def _validate_urdf(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    root = ET.parse(path).getroot()
    active = {
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") in {"revolute", "continuous", "prismatic"}
    }
    if active != EXPECTED_ACTIVE_JOINTS:
        errors.append(
            f"URDF active-joint mismatch: missing={sorted(EXPECTED_ACTIVE_JOINTS - active)}, "
            f"extra={sorted(active - EXPECTED_ACTIVE_JOINTS)}"
        )
    collision_count = len(root.findall("./link/collision"))
    if collision_count < 10:
        warnings.append(
            f"URDF has only {collision_count} active collision elements; use the USD physics "
            "layer for Isaac training."
        )
    return errors, warnings


def _validate_mjcf(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    root = ET.parse(path).getroot()
    actuator_joints = {
        motor.attrib["joint"]
        for motor in root.findall("./actuator/motor")
        if "joint" in motor.attrib
    }
    if len(actuator_joints) != 31:
        errors.append(f"MJCF expected 31 motor actuators, found {len(actuator_joints)}")

    direct_policy_actuators = POLICY_JOINTS & actuator_joints
    missing_direct = POLICY_JOINTS - direct_policy_actuators
    expected_transmission_joints = {
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    }
    if missing_direct != expected_transmission_joints:
        errors.append(
            "Unexpected MJCF policy-to-actuator mapping gap: "
            f"{sorted(missing_direct)}"
        )
    else:
        warnings.append(
            "MJCF uses parallel-linkage actuators for ankle pitch/roll and waist roll/pitch; "
            "sim-to-MuJoCo needs a transmission mapping before policy evaluation."
        )
    return errors, warnings


def _validate_usd(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        from pxr import Usd, UsdPhysics
    except ImportError:
        return _validate_usd_with_cli(path)

    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
    if stage is None:
        return [f"Unable to open USD stage: {path}"], warnings
    root = stage.GetDefaultPrim()
    if not root:
        return ["USD has no default prim"], warnings

    variants = root.GetVariantSets()
    variants.GetVariantSet("Robot").SetVariantSelection("None")
    variants.GetVariantSet("Sensor").SetVariantSelection("None")
    variants.GetVariantSet("Physics").SetVariantSelection("PhysX")
    stage.Load()

    articulation_prim = stage.GetPrimAtPath(f"{root.GetPath()}/base_link")
    if not articulation_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        errors.append("USD base_link does not have PhysicsArticulationRootAPI")

    revolute_joints = {
        prim.GetName()
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.RevoluteJoint)
    }
    if revolute_joints != EXPECTED_ACTIVE_JOINTS:
        errors.append(
            f"USD revolute-joint mismatch: missing={sorted(EXPECTED_ACTIVE_JOINTS - revolute_joints)}, "
            f"extra={sorted(revolute_joints - EXPECTED_ACTIVE_JOINTS)}"
        )

    collision_bodies = {
        prim.GetParent().GetName()
        for prim in stage.Traverse()
        if prim.GetName() == "collisions"
    }
    if len(collision_bodies) < 30:
        errors.append(
            f"USD expected collision groups on at least 30 bodies, found {len(collision_bodies)}"
        )

    robot_variant = root.GetVariantSets().GetVariantSet("Robot")
    if "Robot" in robot_variant.GetVariantNames():
        warnings.append(
            "The source USD's Robot=Robot payload is missing; WBC-AGILE explicitly selects Robot=None."
        )
    return errors, warnings


def _validate_usd_with_cli(path: Path) -> tuple[list[str], list[str]]:
    """Validate the source layers when pxr is only available through USD CLIs."""

    errors: list[str] = []
    warnings: list[str] = [
        "Python pxr bindings are unavailable; USD structure was checked with usdcat."
    ]
    physics_path = path.parent / "configuration" / "HU_D03_03_physics.usd"
    try:
        root_result = subprocess.run(
            ["usdcat", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        physics_result = subprocess.run(
            ["usdcat", str(physics_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ["Neither Python pxr bindings nor usdcat are available"], warnings
    except subprocess.CalledProcessError as exc:
        return [f"usdcat failed for HU_D03 USD: {exc.stderr.strip()}"], warnings

    root_text = root_result.stdout
    physics_text = physics_result.stdout
    if 'defaultPrim = "HU_D03_03"' not in root_text:
        errors.append("USD root layer does not declare HU_D03_03 as defaultPrim")
    if "PhysicsArticulationRootAPI" not in physics_text:
        errors.append("USD physics layer has no PhysicsArticulationRootAPI")

    revolute_joints = set(re.findall(r'def PhysicsRevoluteJoint "([^"]+)"', physics_text))
    if revolute_joints != EXPECTED_ACTIVE_JOINTS:
        errors.append(
            f"USD revolute-joint mismatch: missing={sorted(EXPECTED_ACTIVE_JOINTS - revolute_joints)}, "
            f"extra={sorted(revolute_joints - EXPECTED_ACTIVE_JOINTS)}"
        )

    collision_group_count = len(re.findall(r'def Xform "collisions"', physics_text))
    if collision_group_count < 30:
        errors.append(
            "USD expected collision groups on at least 30 bodies, "
            f"found {collision_group_count}"
        )
    if "HU_D03_03_robot.usd" in root_text:
        warnings.append(
            "The source USD's Robot=Robot payload is missing; WBC-AGILE explicitly selects Robot=None."
        )
    return errors, warnings


def main() -> int:
    args = _parse_args()
    root = args.description_root.expanduser().resolve()
    paths = {
        "URDF": root / "urdf" / "HU_D03_03.urdf",
        "USD": root / "usd" / "HU_D03_03.usd",
        "MJCF": root / "xml" / "HU_D03_03.xml",
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        print("ERROR missing assets:")
        print("\n".join(f"  - {item}" for item in missing))
        return 1

    checks = (
        ("URDF", _validate_urdf(paths["URDF"])),
        ("USD", _validate_usd(paths["USD"])),
        ("MJCF", _validate_mjcf(paths["MJCF"])),
    )
    errors = [(name, item) for name, (items, _) in checks for item in items]
    warnings = [(name, item) for name, (_, items) in checks for item in items]

    print(f"HU_D03 description root: {root}")
    print(f"Expected active joints: {len(EXPECTED_ACTIVE_JOINTS)}")
    for name, item in warnings:
        print(f"WARNING [{name}] {item}")
    for name, item in errors:
        print(f"ERROR [{name}] {item}")
    if errors:
        print(f"FAILED with {len(errors)} error(s)")
        return 1
    print(f"PASSED with {len(warnings)} known warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
