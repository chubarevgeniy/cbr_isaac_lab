#!/usr/bin/env python3
"""Run staged PPO curriculum experiments in isolated git worktrees.

Each variant consists of sequential PPO stages. The stage process receives the
previous stage's final policy checkpoint; ``train.py`` loads the policy/value
and observation preprocessors, then resets the optimizer, scheduler and PPO
rollout memory before learning under the new stage configuration.

The default cohort is sized for an overnight run on the current 8 GB GPU:
four variants, three 64k-step stages each, with two simulator processes in
parallel. A previous 64k-step process pair took about 87--90 minutes per
wave, so the cohort is expected to finish in roughly nine hours.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from overnight_experiments import (
    BASELINE,
    CHECK_INTERVAL,
    format_status,
    latest,
    median,
    scalar_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
ISAACLAB = ROOT.parent / "IsaacLab"
VIRTUAL_ENV = Path("/home/evgenii/ws/isaac/env_isaaclab")
TASK = "Template-Cbriisaaclab-Direct-v0"
NUM_ENVS = 2048
MAX_ITERATIONS = 2000  # 2000 * rollouts(32) = 64,000 environment timesteps
MAX_CONCURRENT = 2
EARLY_STOP_AFTER = 30 * 60


@dataclass(frozen=True)
class Stage:
    name: str
    reward_profile: str
    disable_observation_noise: bool = False
    initial_tilt_deg: float | None = None


@dataclass(frozen=True)
class Variant:
    name: str
    branch: str
    stages: tuple[Stage, ...]
    seed: int = 42
    action_mode: str = "delta"

    @property
    def worktree(self) -> Path:
        return ROOT.parent / f"cbr_i_staged_{self.name.replace('-', '_')}"


@dataclass(frozen=True)
class StageIdentity:
    variant: Variant
    stage: Stage
    stage_index: int

    @property
    def name(self) -> str:
        return f"{self.variant.name}/{self.stage.name}"


@dataclass
class Running:
    identity: StageIdentity
    process: subprocess.Popen[str]
    stdout_path: Path
    started_at: float
    last_status_at: float = 0.0
    last_checkpoint: int = 0
    last_checkpoint_mtime: float = 0.0
    run_dir: Path | None = None
    stopped_reason: str | None = None

    @property
    def variant(self) -> Variant:
        """Compatibility view for the existing overnight status formatter."""
        return self.identity.variant


VARIANTS = (
    Variant(
        "reward-curriculum",
        "experiment/staged-reward-curriculum",
        (
            Stage("survival", "survival_clearance_speed"),
            Stage("task", "task_balanced"),
            Stage("tracking", "baseline"),
        ),
    ),
    Variant(
        "easy-to-robust",
        "experiment/staged-easy-to-robust",
        (
            Stage("easy-survival", "survival_clearance_speed", True, 0.0),
            Stage("robust-task", "task_balanced"),
            Stage("robust-tracking", "baseline"),
        ),
    ),
    Variant(
        "easy-task-to-robust",
        "experiment/staged-easy-task-to-robust",
        (
            Stage("easy-task", "task_balanced", True, 0.0),
            Stage("robust-task", "task_balanced"),
            Stage("robust-tracking", "baseline"),
        ),
    ),
    Variant(
        "staged-control",
        "experiment/staged-control",
        (
            Stage("task-1", "task_balanced"),
            Stage("task-2", "task_balanced"),
            Stage("task-3", "task_balanced"),
        ),
        seed=43,
    ),
)


def run_git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def prepare_worktree(variant: Variant, base_commit: str) -> None:
    path = variant.worktree
    if path.exists():
        current_branch = run_git("-C", str(path), "rev-parse", "--abbrev-ref", "HEAD")
        if current_branch != variant.branch:
            raise RuntimeError(f"{path} already exists on {current_branch}, expected {variant.branch}")
        dirty = run_git("-C", str(path), "status", "--porcelain=1", "--untracked-files=all")
        if dirty:
            raise RuntimeError(f"staged worktree is dirty: {path}")
        return

    branches = set(run_git("branch", "--format=%(refname:short)").splitlines())
    if variant.branch in branches:
        subprocess.run(["git", "worktree", "add", str(path), variant.branch], cwd=ROOT, check=True)
    else:
        subprocess.run(
            ["git", "worktree", "add", "-b", variant.branch, str(path), base_commit],
            cwd=ROOT,
            check=True,
        )


def command_for(identity: StageIdentity, checkpoint: Path | None) -> list[str]:
    variant = identity.variant
    stage = identity.stage
    command = [
        str(ISAACLAB / "isaaclab.sh"),
        "-p",
        "scripts/skrl/train.py",
        "--task",
        TASK,
        "--num_envs",
        str(NUM_ENVS),
        "--max_iterations",
        str(MAX_ITERATIONS),
        "--seed",
        str(variant.seed),
        "--action_mode",
        variant.action_mode,
        "--reward_profile",
        stage.reward_profile,
        "--policy_clip_actions",
        "--policy_initial_log_std=-0.7",
        "--policy_max_log_std=0.0",
        "--experiment_label",
        f"staged-{variant.name}-{stage.name}",
    ]
    if stage.disable_observation_noise:
        command.append("--disable_observation_noise")
    if stage.initial_tilt_deg is not None:
        command.extend(["--initial_tilt_deg", str(stage.initial_tilt_deg)])
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint), "--reset_optimizer_scheduler"])
    return command


def discover_run_dir(identity: StageIdentity, started_at: float) -> Path | None:
    root = identity.variant.worktree / "logs" / "skrl" / "cbr_i_ppo"
    if not root.exists():
        return None
    label = f"staged-{identity.variant.name}-{identity.stage.name}"
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and label in path.name and path.stat().st_mtime >= started_at - 5
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def checkpoint_state(run_dir: Path | None) -> tuple[int, float]:
    if run_dir is None:
        return 0, 0.0
    parsed: list[tuple[int, float]] = []
    for checkpoint in (run_dir / "checkpoints").glob("agent_*.pt"):
        try:
            step = int(checkpoint.stem.removeprefix("agent_"))
        except ValueError:
            continue
        parsed.append((step, checkpoint.stat().st_mtime))
    return max(parsed, default=(0, 0.0))


def checkpoint_path(run_dir: Path | None) -> Path | None:
    if run_dir is None:
        return None
    checkpoints = []
    for path in (run_dir / "checkpoints").glob("agent_*.pt"):
        try:
            checkpoints.append((int(path.stem.removeprefix("agent_")), path))
        except ValueError:
            continue
    return max(checkpoints, default=(0, None))[1]


def obvious_failure(running: Running, snapshot: dict[str, Any]) -> str | None:
    """Stop only clear multi-signal failures, preserving partial checkpoints."""
    age = time.time() - running.started_at
    if age < EARLY_STOP_AFTER:
        return None
    if running.last_checkpoint == 0 and age > 40 * 60:
        return "no checkpoint after 40 minutes"
    if running.last_checkpoint > 0 and time.time() - running.last_checkpoint_mtime > 20 * 60:
        return f"no new checkpoint for more than 20 minutes (last={running.last_checkpoint})"
    if not snapshot or "error" in snapshot:
        return None

    values = {
        key: median(snapshot, key)
        for key in ("lifetime", "termination", "speed_error", "foot_height", "sit_error")
    }
    if any(value is None for value in values.values()):
        return None

    bad_signals = 0
    if values["lifetime"] < 0.35 * BASELINE["lifetime"] and values["termination"] > 4.0 * BASELINE["termination"]:
        bad_signals += 2
    if values["speed_error"] > 0.80 and values["foot_height"] < 0.008:
        bad_signals += 1
    if values["sit_error"] > 0.60 and values["lifetime"] < 0.60 * BASELINE["lifetime"]:
        bad_signals += 1

    worsening = 0
    for key, lower_is_better in (
        ("lifetime", False),
        ("speed_error", True),
        ("foot_height", False),
        ("sit_error", True),
    ):
        prior = median(snapshot, key, "prior")
        recent = median(snapshot, key, "recent")
        if prior is None or recent is None:
            continue
        if lower_is_better and recent > prior * 1.08:
            worsening += 1
        elif not lower_is_better and recent < prior * 0.92:
            worsening += 1
    if worsening >= 2 and bad_signals >= 1:
        bad_signals += 1

    if bad_signals >= 3:
        return (
            f"multi-signal failure: lifetime={values['lifetime']:.1f}, "
            f"termination={values['termination']:.4f}, speed_error={values['speed_error']:.3f}, "
            f"foot_height={values['foot_height']:.4f}, sit_error={values['sit_error']:.3f}"
        )
    return None


def stop_process(running: Running, reason: str) -> None:
    running.stopped_reason = reason
    try:
        os.killpg(running.process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        running.process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(running.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def write_status(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stage_record(
    running: Running,
    returncode: int | None,
    status: str,
) -> dict[str, Any]:
    return {
        "stage": running.identity.stage.name,
        "stage_index": running.identity.stage_index,
        "reward_profile": running.identity.stage.reward_profile,
        "disable_observation_noise": running.identity.stage.disable_observation_noise,
        "initial_tilt_deg": running.identity.stage.initial_tilt_deg,
        "branch": running.identity.variant.branch,
        "seed": running.identity.variant.seed,
        "run_dir": str(running.run_dir) if running.run_dir else None,
        "stdout": str(running.stdout_path),
        "last_checkpoint": running.last_checkpoint,
        "returncode": returncode,
        "status": status,
        "stopped_reason": running.stopped_reason,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true", help="Prepare worktrees and print all stage commands.")
    parser.add_argument("--prepare-only", action="store_true", help="Create worktrees without starting training.")
    args = parser.parse_args()

    if not ISAACLAB.joinpath("isaaclab.sh").exists():
        raise SystemExit(f"IsaacLab launcher not found: {ISAACLAB / 'isaaclab.sh'}")
    if run_git("status", "--porcelain=1", "--untracked-files=all"):
        raise SystemExit("Working tree must be clean before preparing staged worktrees")

    base_commit = run_git("rev-parse", "HEAD")
    for variant in VARIANTS:
        prepare_worktree(variant, base_commit)

    if args.prepare_only:
        print(f"[staged] prepared {len(VARIANTS)} worktrees at base {base_commit}")
        return 0

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    state_root = ROOT / "logs" / "staged" / timestamp
    state_root.mkdir(parents=True, exist_ok=True)
    status_path = state_root / "status.json"

    manifest: dict[str, Any] = {
        variant.name: {
            "branch": variant.branch,
            "worktree": str(variant.worktree),
            "seed": variant.seed,
            "action_mode": variant.action_mode,
            "stages": [
                {
                    "name": stage.name,
                    "reward_profile": stage.reward_profile,
                    "disable_observation_noise": stage.disable_observation_noise,
                    "initial_tilt_deg": stage.initial_tilt_deg,
                }
                for stage in variant.stages
            ],
        }
        for variant in VARIANTS
    }
    (state_root / "commands.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        for variant in VARIANTS:
            for index, stage in enumerate(variant.stages):
                checkpoint = Path("<previous-stage-checkpoint>") if index else None
                identity = StageIdentity(variant, stage, index)
                print(identity.name, "\t", " ".join(command_for(identity, checkpoint)))
        return 0

    queue: list[StageIdentity] = [
        StageIdentity(variant, variant.stages[0], 0) for variant in VARIANTS
    ]
    running: dict[str, Running] = {}
    finished: dict[str, list[dict[str, Any]]] = {variant.name: [] for variant in VARIANTS}
    latest_checkpoints: dict[str, Path] = {}
    failed_variants: set[str] = set()
    stop_requested = False

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[staged] received signal {signum}; stopping active runs", flush=True)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while queue or running:
        if stop_requested:
            for active in list(running.values()):
                stop_process(active, "supervisor interrupted")

        while not stop_requested and queue and len(running) < max(1, args.max_concurrent):
            identity = queue.pop(0)
            if identity.variant.name in failed_variants:
                continue
            checkpoint = latest_checkpoints.get(identity.variant.name)
            stdout_path = state_root / f"{identity.variant.name}__{identity.stage.name}.stdout.log"
            stdout = stdout_path.open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["VIRTUAL_ENV"] = str(VIRTUAL_ENV)
            environment["PYTHONUNBUFFERED"] = "1"
            source_path = str(identity.variant.worktree / "source" / "CBRIIsaacLab")
            environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
            process = subprocess.Popen(
                command_for(identity, checkpoint),
                cwd=identity.variant.worktree,
                env=environment,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            stdout.close()
            running[identity.name] = Running(identity, process, stdout_path, time.time())
            print(
                f"[staged] launched {identity.name} pid={process.pid} "
                f"checkpoint={checkpoint or 'none'}",
                flush=True,
            )

        for name, active in list(running.items()):
            if active.run_dir is None:
                active.run_dir = discover_run_dir(active.identity, active.started_at)
            active.last_checkpoint, active.last_checkpoint_mtime = checkpoint_state(active.run_dir)
            snapshot = scalar_snapshot(active.run_dir)
            now = time.time()
            if now - active.last_status_at >= 300:
                print(format_status(active, snapshot), flush=True)
                active.last_status_at = now

            reason = obvious_failure(active, snapshot)
            if reason:
                print(f"[staged] stopping {name}: {reason}", flush=True)
                stop_process(active, reason)

            if active.process.poll() is not None:
                returncode = active.process.returncode
                succeeded = returncode == 0 and checkpoint_path(active.run_dir) is not None
                status = "completed" if succeeded else "failed"
                if active.stopped_reason:
                    status = "stopped_early"
                finished[active.identity.variant.name].append(
                    stage_record(active, returncode, status)
                )
                if succeeded:
                    latest_checkpoints[active.identity.variant.name] = checkpoint_path(active.run_dir)  # type: ignore[assignment]
                    next_index = active.identity.stage_index + 1
                    stages = active.identity.variant.stages
                    if next_index < len(stages) and not stop_requested:
                        queue.append(StageIdentity(active.identity.variant, stages[next_index], next_index))
                else:
                    failed_variants.add(active.identity.variant.name)
                del running[name]
                print(f"[staged] finished {name} returncode={returncode} status={status}", flush=True)

        state = {
            "started_at": timestamp,
            "base_commit": base_commit,
            "queue": [identity.name for identity in queue],
            "running": {
                name: {
                    "pid": active.process.pid,
                    "stage": active.identity.stage.name,
                    "branch": active.identity.variant.branch,
                    "run_dir": str(active.run_dir) if active.run_dir else None,
                    "last_checkpoint": active.last_checkpoint,
                    "stdout": str(active.stdout_path),
                }
                for name, active in running.items()
            },
            "finished": finished,
            "failed_variants": sorted(failed_variants),
        }
        write_status(status_path, state)
        if queue or running:
            time.sleep(CHECK_INTERVAL)

    print(f"[staged] cohort complete; state={status_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
