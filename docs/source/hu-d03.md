# HU_D03 Integration

This fork adds the first HU_D03 training milestone: lower-body velocity
tracking with 12 leg joints plus waist roll and pitch.

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
