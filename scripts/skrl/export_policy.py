# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to export an RL agent to TorchScript.

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import torch
import torch.nn as nn

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Export an RL agent to TorchScript.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--output_file", type=str, default="policy.pt", help="Output TorchScript file name.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from skrl.utils.runner.torch import Runner
from skrl.resources.preprocessors.torch import RunningStandardScaler

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import CBRIIsaacLab.tasks  # noqa: F401

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"

class ExportPolicy(nn.Module):
    def __init__(self, policy, scaler):
        super().__init__()
        self.policy = policy
        # Register scaler parameters as buffers so they are saved with the model
        self.with_scaler = scaler is not None

        print("With scaler:", self.with_scaler)

        if scaler is not None:
            self.register_buffer("mean", scaler.running_mean.clone())
            self.register_buffer("var", scaler.running_variance.clone())
            self.clip_threshold = float(scaler.clip_threshold)

    def forward(self, x):
        # Apply scaler
        if self.with_scaler:
            x = (x - self.mean) / torch.sqrt(self.var + 1e-8)
            x = torch.clamp(x, -self.clip_threshold, self.clip_threshold)
        
        # Apply policy
        # Note: skrl models usually return (output, ...) tuple in compute
        # We assume the first element is the action/mean
        out = self.policy.compute({"states": x}, role="policy")
        if isinstance(out, tuple):
            return out[0]
        return out

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Export skrl agent to TorchScript."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = 1 
    if hasattr(args_cli, "device") and args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = 0

    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # get checkpoint path
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # configure and instantiate the skrl runner
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    runner = Runner(env, agent_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # Extract policy and scaler
    policy = runner.agent.policy
    state_preprocessor = runner.agent._state_preprocessor
    
    scaler = None
    if isinstance(state_preprocessor, RunningStandardScaler):
        scaler = state_preprocessor
        print("[INFO] Found RunningStandardScaler.")
    else:
        print(f"[INFO] Preprocessor is {type(state_preprocessor)}. Assuming no scaling needed or not supported for export.")

    device = runner.agent.device

    # Create export wrapper
    print("Creating ExportPolicy wrapper...")
    export_model = ExportPolicy(policy, scaler).to(device)
    export_model.eval()

    # Dummy input for tracing
    obs_shape = env.observation_space.shape
    dummy_input = torch.randn(1, *obs_shape, device=device)

    # Trace the model
    print("Tracing model with dummy input...")
    traced_script = torch.jit.trace(export_model, dummy_input)

    # Save the traced model
    output_path = args_cli.output_file
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(resume_path), output_path)

    traced_script.save(output_path)
    print(f"Exported TorchScript model saved to {output_path}")

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()