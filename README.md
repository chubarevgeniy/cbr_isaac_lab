# Template for Isaac Lab Projects

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

### Robot asset

Place the CBR-I articulation at:

```text
source/CBRIIsaacLab/CBRIIsaacLab/robots/CBR-I.usda
```

The environment cannot be created without this file. The repository's
`.gitattributes` stores USD assets with Git LFS.

The leg transmission is modeled in the actuator layer rather than with extra
USD links.  The articulation coordinates remain the physical relative hip and
knee angles, while the controller uses
`q_hip_motor = theta_hip` and
`q_knee_motor = theta_knee + theta_hip`.  Equivalently, the physical knee
coordinate is `theta_knee = q_knee_motor - q_hip_motor`.  Motor torques are
clipped separately before being mapped back to physical joint efforts.  This
keeps the authored knee limits as hard limits on the physical knee angle and
avoids artificial link mass or tendon compliance.

The four driven articulation joints use `0.02 kg m^2` armature as a diagonal
approximation of the reflected 5008-rotor inertia after the 12:1 reductions.
The coupled PD is explicit, so damping randomization is capped at the nominal
`3.67 N m s/rad`; increasing it by 10% was numerically unstable at the 250 Hz
physics rate.  The articulation remains at 4 position and 0 velocity solver
iterations: measured increases did not improve the contact diagnostic.
The nominal joint-friction approximation is `0.12 N m` static, `0.096 N m`
dynamic, and `0.012 N m s/rad` viscous. Each component is independently
randomized by `0.5...1.5` during training, with dynamic friction clamped not
to exceed static friction.

Observation latency is configurable without changing the 23-element policy
observation. In `CbriisaaclabEnvCfg`, `observation_delay_s = 0.02` is one
policy step at the current `250 Hz` physics rate and `decimation = 5`.
`observation_delay_mode` can be set to `"current"`, `"delayed"`, or
`"random"`; the last option selects one mode per environment on reset.
Only joint position/velocity measurements are delayed; the task command and
the two action-history fields remain current.

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/CBRIIsaacLab

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running this task in the Isaac Sim (Kit) visualizer while rendering at most 16 environments:

        ```bash
        python scripts/skrl/train.py \
            --task=Template-Cbriisaaclab-Direct-v0 \
            --viz kit \
            --max_visible_envs 16
        ```

        `--max_visible_envs` changes rendering only; all simulated `env_*` prims remain listed in the Stage panel.
        Use `--num_envs 16` instead if only 16 environments should be simulated and created.

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Inspect randomized initial poses

To create exactly 16 standing-like reset poses with a reproducible seed, print
their FK/collision diagnostics, and leave Isaac Sim open for visual inspection:

```bash
../IsaacLab/isaaclab.sh -p scripts/inspect_initial_poses.py --seed 123 --viz newton
```

The inspector forces `sit=0` and `speed=0`, writes zero root/joint velocities,
and does not call `env.step()`. Newton shows all 16 environments with joints
and collision geometry enabled. Use `--viz kit` for the Kit viewport, or
`--seed -1` to generate a seed at startup; the resolved seed is printed to the
console.

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/CBRIIsaacLab/CBRIIsaacLab/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Training comparison workflow

The reproducible experiment protocol is documented in [TRAINING_PLAN.md](TRAINING_PLAN.md).
The quick comparison uses a fixed seed, environment count, and `32,000` environment steps:

```bash
VIRTUAL_ENV=/path/to/env_isaaclab ../IsaacLab/isaaclab.sh -p scripts/skrl/train.py \
    --task=Template-Cbriisaaclab-Direct-v0 \
    --num_envs=2048 \
    --max_iterations=1000 \
    --seed=42
```

`max_iterations=1000` means `1000 * rollouts(24) = 24,000` trainer steps. Every run stores
the environment and PPO configuration, git provenance, checkpoints, TensorBoard event files,
and the full training trajectory under `logs/skrl/cbr_i_ppo/`. Compare all recorded steps,
not just the tail of a run.

### Warm-start and NumPy export for ROS

To adapt a trained policy to a robot with weaker motors, use `--warm_start` together
with the source checkpoint. It restores only the policy and observation normalizer;
the value model, PPO optimizer, value normalizer, and rollout memory start fresh:

```bash
../IsaacLab/isaaclab.sh -p scripts/skrl/train.py \
    --task=Template-Cbriisaaclab-Direct-v0 \
    --checkpoint=/absolute/path/to/agent_650000.pt \
    --warm_start
```

To run the policy in a ROS Python node without importing PyTorch, export it from the
checkpoint using the Isaac Lab Python environment (Isaac Sim itself is not launched):

```bash
VIRTUAL_ENV=/path/to/env_isaaclab python scripts/skrl/export_numpy_policy.py \
    --checkpoint=/absolute/path/to/agent_650000.pt \
    --output_file=policy.npz
```

This produces `policy.npz` and `policy.json` next to the checkpoint. The archive contains
the MLP and observation scaler; the JSON records the 23-element observation layout,
coordinate signs, action offset/scales, canonical hip reference, and both configured
`default_standing_state` poses. Copy `scripts/skrl/numpy_policy.py` and the two exported
files into the ROS package. The runtime exposes `build_observation`, `predict`, and
`action_to_raw_target`; keep feeding the exact previous and two-steps-previous policy
actions as the last eight observation values. At reset, initialize both history
slots with the inverse-affine action of the commanded reset pose; use zeros only
when that pose corresponds to a zero action.
See the detailed
[NumPy/ROS export contract](scripts/skrl/export_numpy_policy.md) for the exact
coordinate formulas and a ROS example.

### Single-factor and cross-factor experiments

Keep one-factor branches for attribution, then add cross-factor branches to measure interactions.
The initial factors are:

| Factor | Change |
| --- | --- |
| A | `add_noise=False` |
| B | `initial_tilt_angle_variation=0.0` |
| C | `agent.mini_batches=4` |

For a fast triage pass, run the combined gate `A+B+C` first. If its full training trajectory,
physical metrics do not show a meaningful difference from baseline, the three candidates can be
deprioritized together. If the gate changes behavior, run `A`, `B`, and
`C` independently, followed by selected combinations such as `A+B`, `A+C`, and `B+C` to attribute
the effect. Each branch must change only the listed values relative to the common base; the report
should distinguish the individual effect from the interaction effect.

### Parallel runs and GPU memory

Parallel training is allowed only when the available VRAM has been checked with `nvidia-smi`.
On an 8 GB GPU, two processes using `2048` environments may not fit. If two processes are run
in parallel, use the same reduced `--num_envs` for the whole comparison cohort (for example,
`1024` per process), and do not compare that cohort's raw speed directly with a `2048`-environment
cohort. Keep each process on a separate log directory and stop the pair if memory use approaches
the device limit.

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/CBRIIsaacLab"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```


Подробный план системного сравнения обучения: [TRAINING_PLAN.md](TRAINING_PLAN.md).
