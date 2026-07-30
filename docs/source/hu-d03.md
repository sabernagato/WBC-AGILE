# HU_D03 Integration

This fork adds two HU_D03 training milestones:

- lower-body velocity tracking with 12 leg joints plus waist roll and pitch;
- full-body recovery from cached supine, prone, and side-lying poses.

For a reproducible new-machine setup and an agent-oriented execution
checklist, read the repository-root
[`HU_D03_AGENT_HANDOFF.md`](../../HU_D03_AGENT_HANDOFF.md) before training.

## Asset layout

By default, AGILE expects the robot description repository next to this
repository:

```text
third_party/
├── WBC-AGILE/
└── limx_oli_description/
    └── HU_D03_description/
        ├── urdf/HU_D03_03.urdf
        ├── usd/HU_D03_03.usd
        └── xml/HU_D03_03.xml
```

For a different layout, set either the description root or the USD path:

```bash
export HU_D03_DESCRIPTION_ROOT=/absolute/path/to/HU_D03_description
# Or only override the Isaac asset:
export HU_D03_USD_PATH=/absolute/path/to/HU_D03_03.usd
```

## Validate the assets

Run the independent model check before starting Isaac Sim:

```bash
python scripts/validate_hu_d03_assets.py
```

The check verifies the 31 active joints, the USD articulation and collision
groups, and the MuJoCo actuators. Known warnings document two source-asset
constraints:

- The root USD's optional `Robot=Robot` payload is missing. The AGILE robot
  configuration selects `Robot=None`; the complete articulation comes from
  the `Physics=PhysX` layer.
- The MJCF represents the ankle and waist parallel mechanisms with physical
  A/B actuators. A transmission mapping is required before using the generic
  sim-to-MuJoCo evaluator.

## Smoke test and train

Use a small environment count for the first launch:

```bash
python scripts/train.py \
  --task Velocity-HU-D03-History-v0 \
  --num_envs 32 \
  --max_iterations 10
```

After the robot settles, contacts are correct, and observations/actions
resolve without errors, start a headless run:

```bash
python scripts/train.py \
  --task Velocity-HU-D03-History-v0 \
  --num_envs 2048 \
  --headless
```

Evaluate a checkpoint with:

```bash
python scripts/eval.py \
  --task Velocity-HU-D03-History-v0 \
  --num_envs 32 \
  --checkpoint /absolute/path/to/model.pt
```

## Fallen-pose stand-up training

`StandUp-HU-D03-v0` controls all 31 active joints and always tries to recover
to a stable standing pose. It does not command the robot to fall or lie down.

Before training, its `pre_learn` hook automatically builds and caches two
fallen-state datasets:

- a primary set with repeatable supine poses and default joint positions;
- a secondary set with random root orientations, joint positions, and small
  initial velocities, covering prone, side-lying, and irregular poses.

Inspect the generated fallen states before learning:

```bash
python scripts/play.py \
  --task StandUp-HU-D03-v0 \
  --num_envs 16 \
  --validate-fallen-states \
  --num_steps 500
```

Run a short headless smoke training:

```bash
python scripts/train.py \
  --task StandUp-HU-D03-v0 \
  --num_envs 64 \
  --max_iterations 10 \
  --headless \
  --logger tensorboard
```

The lift action applies an early-training upward assist capped at 90% of robot
weight. Curriculum learning removes the assist and increases arbitrary fallen
poses. A final policy is useful only after it succeeds without lift assistance.

The first launch can take several minutes because it simulates and caches the
fallen poses. Delete a cache only when intentionally changing the robot asset,
terrain, or fallen-state configuration.

## Before sim-to-real

The actuator gains in `agile/rl_env/assets/robots/hu_d03.py` are conservative
training defaults. Replace them with identified hardware values and validate:

1. standing base height and all joint zero positions;
2. joint axes, signs, limits, and policy ordering;
3. output torque and velocity limits;
4. position-loop stiffness/damping, friction, armature, delay, and saturation;
5. total mass, link inertias, center of mass, and foot contact geometry;
6. ankle and waist actuator-to-joint transmission;
7. hardware state/action bridge, watchdog, fall detection, and torque shutdown.

Do not deploy a policy trained with the default gains directly to hardware.
