# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--max_timesteps",
    type=int,
    default=None,
    help="Exact number of environment timesteps for this process (overrides --max_iterations).",
)
parser.add_argument(
    "--action_mode",
    type=str,
    default=None,
    choices=["delta", "absolute"],
    help="Action semantics: bounded delta targets or bounded absolute joint targets.",
)
parser.add_argument(
    "--reward_profile",
    type=str,
    default=None,
    choices=["baseline", "survival_clearance_speed", "smooth_clearance", "task_balanced"],
    help="Named reward bundle used for an experiment.",
)
parser.add_argument(
    "--policy_clip_actions",
    action="store_true",
    help="Clip sampled policy actions to the bounded [-1, 1] action space.",
)
parser.add_argument(
    "--policy_initial_log_std",
    type=float,
    default=None,
    help="Optional initial Gaussian log standard deviation for bounded-policy runs.",
)
parser.add_argument(
    "--policy_max_log_std",
    type=float,
    default=None,
    help="Optional upper bound for Gaussian log standard deviation.",
)
parser.add_argument(
    "--disable_observation_noise",
    action="store_true",
    help="Disable observation noise for a curriculum warm-up stage.",
)
parser.add_argument(
    "--initial_tilt_deg",
    type=float,
    default=None,
    help="Initial rod-body tilt variation in degrees for a curriculum stage.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=None,
    help="Override the PPO learning rate for this stage.",
)
parser.add_argument(
    "--learning_rate_min",
    type=float,
    default=None,
    help="Override KLAdaptiveLR min_lr for this stage.",
)
parser.add_argument(
    "--reset_optimizer_scheduler",
    action="store_true",
    help=(
        "After loading a checkpoint, keep model/preprocessor weights but recreate the PPO "
        "optimizer and learning-rate scheduler from the current stage configuration."
    ),
)
parser.add_argument(
    "--experiment_label",
    type=str,
    default=None,
    help="Optional human-readable suffix stored in the run directory name.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import itertools
import math
import numpy as np
import os
import random
import re
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "2.1.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import CBRIIsaacLab.tasks  # noqa: F401

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


def get_git_metadata() -> dict:
    """Collect the repository state at the moment the training run starts."""
    repo_root = Path(__file__).resolve().parents[2]

    def run_git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
        commit = run_git("rev-parse", "--short", "HEAD")
        status_output = run_git("status", "--porcelain=1", "--untracked-files=all")
        dirty_files = status_output.splitlines()
        return {
            "repository": str(repo_root),
            "branch": branch,
            "commit": commit,
            "worktree": "dirty" if dirty_files else "clean",
            "dirty_files": dirty_files,
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "repository": str(repo_root),
            "branch": "unknown",
            "commit": "unknown",
            "worktree": "unknown",
            "dirty_files": [],
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "error": str(exc),
        }


def sanitize_run_component(value: str) -> str:
    """Convert a git value into a filesystem-safe run-name component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "unknown"


def reset_optimizer_and_scheduler(agent) -> None:
    """Reset PPO optimization state while preserving loaded model weights.

    skrl checkpoints include policy/value preprocessors as well as the Adam
    optimizer. A staged run retains the learned policy and normalization
    statistics, but starts the next reward/environment stage with fresh Adam
    moments and a fresh learning-rate scheduler.
    """
    if agent.policy is None or agent.value is None:
        raise RuntimeError("Cannot reset PPO optimizer without policy and value models")

    if agent.policy is agent.value:
        parameters = agent.policy.parameters()
    else:
        parameters = itertools.chain(agent.policy.parameters(), agent.value.parameters())

    learning_rate = agent.cfg.learning_rate
    if isinstance(learning_rate, (list, tuple)):
        learning_rate = learning_rate[0]
    agent.optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
    agent.checkpoint_modules["optimizer"] = agent.optimizer

    scheduler_cfg = agent.cfg.learning_rate_scheduler
    scheduler_cls = scheduler_cfg[0] if isinstance(scheduler_cfg, (list, tuple)) else scheduler_cfg
    if scheduler_cls is None:
        agent.scheduler = None
        agent.checkpoint_modules.pop("scheduler", None)
        return

    scheduler_kwargs = agent.cfg.learning_rate_scheduler_kwargs
    if isinstance(scheduler_kwargs, (list, tuple)):
        scheduler_kwargs = scheduler_kwargs[0]
    agent.scheduler = scheduler_cls(agent.optimizer, **dict(scheduler_kwargs))
    agent.checkpoint_modules["scheduler"] = agent.scheduler


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Apply experiment controls before the run configuration is dumped. This
    # keeps every run self-describing and avoids relying on an editable install
    # that may point to a different git worktree.
    if args_cli.action_mode is not None:
        env_cfg.action_mode = args_cli.action_mode
    if args_cli.reward_profile is not None:
        env_cfg.reward_profile = args_cli.reward_profile
    if args_cli.disable_observation_noise:
        env_cfg.add_noise = False
    if args_cli.initial_tilt_deg is not None:
        if args_cli.initial_tilt_deg < 0.0:
            raise ValueError("--initial_tilt_deg must be non-negative")
        env_cfg.initial_tilt_angle_variation = math.radians(args_cli.initial_tilt_deg)
    if args_cli.policy_clip_actions:
        agent_cfg["models"]["policy"]["clip_actions"] = True
        agent_cfg["models"]["policy"]["clip_mean_actions"] = True
        # The integer action-space shorthand is unbounded to skrl. Expose the
        # normalized domain when policy-side clipping is requested.
        env_cfg.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
    if args_cli.policy_initial_log_std is not None:
        agent_cfg["models"]["policy"]["initial_log_std"] = args_cli.policy_initial_log_std
    if args_cli.policy_max_log_std is not None:
        agent_cfg["models"]["policy"]["max_log_std"] = args_cli.policy_max_log_std
    if args_cli.learning_rate is not None:
        if args_cli.learning_rate <= 0.0:
            raise ValueError("--learning_rate must be positive")
        agent_cfg["agent"]["learning_rate"] = args_cli.learning_rate
    if args_cli.learning_rate_min is not None:
        if args_cli.learning_rate_min <= 0.0:
            raise ValueError("--learning_rate_min must be positive")
        scheduler_kwargs = agent_cfg["agent"].get("learning_rate_scheduler_kwargs", {})
        if scheduler_kwargs is None:
            scheduler_kwargs = {}
        scheduler_kwargs["min_lr"] = args_cli.learning_rate_min
        agent_cfg["agent"]["learning_rate_scheduler_kwargs"] = scheduler_kwargs

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max timesteps for training. ``--max_timesteps`` is useful when resuming
    # from a checkpoint because skrl starts the trainer loop at timestep zero
    # for the new process.
    if args_cli.max_timesteps is not None:
        if args_cli.max_timesteps <= 0:
            raise ValueError("--max_timesteps must be positive")
        agent_cfg["trainer"]["timesteps"] = args_cli.max_timesteps
    elif args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    git_metadata = get_git_metadata()
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_branch = sanitize_run_component(git_metadata["branch"])
    run_commit = sanitize_run_component(git_metadata["commit"])
    run_worktree = sanitize_run_component(git_metadata["worktree"])
    # Include source-control provenance in the default run directory name.
    log_dir = (
        f"{run_timestamp}_{run_branch}_{run_commit}_{run_worktree}_{algorithm}_{args_cli.ml_framework}"
    )
    print(
        f"[INFO] Git state: branch={git_metadata['branch']}, commit={git_metadata['commit']}, "
        f"worktree={git_metadata['worktree']}"
    )
    if git_metadata["dirty_files"]:
        print(f"[INFO] Uncommitted paths at launch: {len(git_metadata['dirty_files'])}; see params/git.yaml")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    if args_cli.experiment_label:
        log_dir += f"_{sanitize_run_component(args_cli.experiment_label)}"
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(
        os.path.join(log_dir, "params", "launch.yaml"),
        {"argparse": vars(args_cli), "hydra_args": hydra_args},
    )
    git_metadata["run_name"] = log_dir
    dump_yaml(os.path.join(log_dir, "params", "git.yaml"), git_metadata)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # make the run directory available to the environment and its managers
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    # skrl does not register the scheduler in PPO checkpoints by default.
    # Register it here so subsequent stage-1 resume checkpoints preserve its
    # state just like the Adam optimizer. Older checkpoints simply omit this
    # key and remain loadable.
    if getattr(runner.agent, "scheduler", None) is not None:
        runner.agent.checkpoint_modules["scheduler"] = runner.agent.scheduler

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        if args_cli.reset_optimizer_scheduler:
            if algorithm != "ppo":
                raise ValueError("--reset_optimizer_scheduler is currently supported only for PPO")
            reset_optimizer_and_scheduler(runner.agent)
            print("[INFO] Reset PPO optimizer, Adam moments, and learning-rate scheduler")

    # run training
    runner.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
