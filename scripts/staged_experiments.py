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
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

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
ROLLOUTS_PER_ITERATION = 32
TARGET_STEPS = MAX_ITERATIONS * ROLLOUTS_PER_ITERATION
MAX_CONCURRENT = 2
EARLY_STOP_AFTER = 30 * 60

# These files describe the task/robot/agent behavior. Experiment orchestration
# and documentation files are intentionally excluded: changing the supervisor
# must not make an unchanged training hypothesis look like a new experiment.
TRAINING_FINGERPRINT_ROOTS = (
    "source/CBRIIsaacLab/CBRIIsaacLab/tasks/direct/cbriisaaclab",
    "source/CBRIIsaacLab/CBRIIsaacLab/robots",
)

SIGNATURE_DEFAULTS: dict[str, Any] = {
    "timesteps": None,
    "distributed": False,
    "checkpoint": None,
    "policy_clip_actions": False,
    "policy_initial_log_std": None,
    "policy_max_log_std": None,
    "disable_observation_noise": False,
    "initial_tilt_deg": None,
    "learning_rate": None,
    "learning_rate_min": None,
    "reset_optimizer_scheduler": False,
    "ml_framework": "torch",
    "algorithm": "PPO",
}
SIGNATURE_KEYS = (
    "task",
    "num_envs",
    "timesteps",
    "seed",
    "distributed",
    "checkpoint",
    "action_mode",
    "reward_profile",
    "policy_clip_actions",
    "policy_initial_log_std",
    "policy_max_log_std",
    "disable_observation_noise",
    "initial_tilt_deg",
    "learning_rate",
    "learning_rate_min",
    "reset_optimizer_scheduler",
    "ml_framework",
    "algorithm",
)
BOOL_SIGNATURE_KEYS = {
    "distributed",
    "policy_clip_actions",
    "disable_observation_noise",
    "reset_optimizer_scheduler",
}


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


@dataclass(frozen=True)
class ExistingRun:
    run_dir: Path
    checkpoint: Path | None
    checkpoint_step: int
    fingerprint: str
    signature: dict[str, Any]


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
    input_checkpoint: Path | None = None

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


def _git_output(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


@lru_cache(maxsize=None)
def training_fingerprint(commit: str) -> str | None:
    """Hash task behavior files at a recorded git commit.

    The experiment supervisor and train CLI are deliberately not included:
    their additions can provide new controls without changing a run that did
    not use those controls. The resolved launch arguments below capture the
    controls that were actually enabled for each run.
    """
    path_listing = _git_output("ls-tree", "-r", "--name-only", commit, "--", *TRAINING_FINGERPRINT_ROOTS)
    if path_listing is None:
        return None

    digest = hashlib.sha256()
    paths = [path for path in path_listing.splitlines() if path]
    for path in paths:
        blob = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if blob.returncode != 0:
            return None
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(blob.stdout)
        digest.update(b"\0")
    return digest.hexdigest()


def normalize_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def normalize_checkpoint(value: Any) -> str | None:
    if value in (None, "", "null"):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


def effective_timesteps(arguments: dict[str, Any]) -> int | None:
    explicit = arguments.get("max_timesteps")
    if explicit is not None:
        return int(explicit)
    iterations = arguments.get("max_iterations")
    if iterations is not None:
        return int(iterations) * ROLLOUTS_PER_ITERATION
    return None


def launch_signature(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return only effective experiment controls, ignoring run labels."""
    signature: dict[str, Any] = {}
    for key in SIGNATURE_KEYS:
        value = effective_timesteps(arguments) if key == "timesteps" else arguments.get(key, SIGNATURE_DEFAULTS.get(key))
        if key in BOOL_SIGNATURE_KEYS:
            value = normalize_bool(value)
        elif key == "checkpoint":
            value = normalize_checkpoint(value)
        signature[key] = value
    return signature


def expected_launch_signature(
    identity: StageIdentity,
    checkpoint: Path | None,
    *,
    timesteps: int | None = None,
    reset_optimizer_scheduler: bool | None = None,
) -> dict[str, Any]:
    stage = identity.stage
    variant = identity.variant
    return launch_signature(
        {
            "task": TASK,
            "num_envs": NUM_ENVS,
            "max_timesteps": TARGET_STEPS if timesteps is None else timesteps,
            "seed": variant.seed,
            "distributed": False,
            "checkpoint": checkpoint,
            "action_mode": variant.action_mode,
            "reward_profile": stage.reward_profile,
            "policy_clip_actions": True,
            "policy_initial_log_std": -0.7,
            "policy_max_log_std": 0.0,
            "disable_observation_noise": stage.disable_observation_noise,
            "initial_tilt_deg": stage.initial_tilt_deg,
            "learning_rate": None,
            "learning_rate_min": None,
            "reset_optimizer_scheduler": (
                checkpoint is not None
                if reset_optimizer_scheduler is None
                else reset_optimizer_scheduler
            ),
            "ml_framework": "torch",
            "algorithm": "PPO",
        }
    )


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


def experiment_log_roots() -> list[Path]:
    roots = [ROOT / "logs" / "skrl" / "cbr_i_ppo"]
    roots.extend(ROOT.parent.glob("cbr_i_*/logs/skrl/cbr_i_ppo"))
    unique: dict[str, Path] = {}
    for root in roots:
        if root.exists():
            unique[str(root.resolve())] = root
    return sorted(unique.values(), key=str)


def existing_runs() -> list[ExistingRun]:
    """Read prior clean runs from this repository's local worktrees."""
    discovered: list[ExistingRun] = []
    for root in experiment_log_roots():
        for launch_path in sorted(root.glob("*/params/launch.yaml")):
            run_dir = launch_path.parent.parent
            git_path = run_dir / "params" / "git.yaml"
            try:
                launch_data = yaml.safe_load(launch_path.read_text(encoding="utf-8")) or {}
                git_data = yaml.safe_load(git_path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                continue

            if not isinstance(launch_data, dict) or not isinstance(git_data, dict):
                continue
            arguments = launch_data.get("argparse", {})
            if not isinstance(arguments, dict):
                continue
            if git_data.get("dirty_files"):
                # A dirty run may contain uncommitted task changes that are not
                # represented by its commit hash, so it is not safe to match.
                continue
            commit = git_data.get("commit")
            if not isinstance(commit, str) or not commit:
                continue
            fingerprint = training_fingerprint(commit)
            if fingerprint is None:
                continue
            checkpoint = checkpoint_path(run_dir)
            checkpoint_step, _ = checkpoint_state(run_dir)
            discovered.append(
                ExistingRun(
                    run_dir=run_dir,
                    checkpoint=checkpoint,
                    checkpoint_step=checkpoint_step,
                    fingerprint=fingerprint,
                    signature=launch_signature(arguments),
                )
            )
    return discovered


def find_duplicate(
    identity: StageIdentity,
    checkpoint: Path | None,
    fingerprint: str,
) -> ExistingRun | None:
    expected = expected_launch_signature(identity, checkpoint)
    matches = [
        run
        for run in existing_runs()
        if run.fingerprint == fingerprint and run.signature == expected
    ]
    if not matches:
        return None
    # Prefer the run with the most progress, then the newest directory. This
    # matters when an interrupted duplicate and a completed duplicate coexist.
    return max(matches, key=lambda run: (run.checkpoint_step, run.run_dir.stat().st_mtime))


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
    duplicate_of: Path | None = None,
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
        "input_checkpoint": str(running.input_checkpoint) if running.input_checkpoint else None,
        "last_checkpoint": running.last_checkpoint,
        "returncode": returncode,
        "status": status,
        "stopped_reason": running.stopped_reason,
        "duplicate_of": str(duplicate_of) if duplicate_of else None,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def skipped_duplicate_record(
    identity: StageIdentity,
    duplicate: ExistingRun,
    checkpoint: Path | None,
) -> dict[str, Any]:
    return {
        "stage": identity.stage.name,
        "stage_index": identity.stage_index,
        "reward_profile": identity.stage.reward_profile,
        "disable_observation_noise": identity.stage.disable_observation_noise,
        "initial_tilt_deg": identity.stage.initial_tilt_deg,
        "branch": identity.variant.branch,
        "seed": identity.variant.seed,
        "run_dir": None,
        "stdout": None,
        "input_checkpoint": str(checkpoint) if checkpoint else None,
        "last_checkpoint": duplicate.checkpoint_step,
        "returncode": None,
        "status": "skipped_duplicate",
        "stopped_reason": None,
        "duplicate_of": str(duplicate.run_dir),
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true", help="Prepare worktrees and print all stage commands.")
    parser.add_argument("--prepare-only", action="store_true", help="Create worktrees without starting training.")
    parser.add_argument(
        "--allow-duplicate",
        action="store_true",
        help="Run a stage even when an equivalent clean run already exists locally.",
    )
    args = parser.parse_args()

    if not ISAACLAB.joinpath("isaaclab.sh").exists():
        raise SystemExit(f"IsaacLab launcher not found: {ISAACLAB / 'isaaclab.sh'}")
    if run_git("status", "--porcelain=1", "--untracked-files=all"):
        raise SystemExit("Working tree must be clean before preparing staged worktrees")

    base_commit = run_git("rev-parse", "HEAD")
    for variant in VARIANTS:
        prepare_worktree(variant, base_commit)

    variant_fingerprints: dict[str, str] = {}
    for variant in VARIANTS:
        worktree_commit = run_git("-C", str(variant.worktree), "rev-parse", "HEAD")
        fingerprint = training_fingerprint(worktree_commit)
        if fingerprint is None:
            raise SystemExit(
                f"Could not fingerprint training files for {variant.name} at {worktree_commit}"
            )
        variant_fingerprints[variant.name] = fingerprint

    if args.prepare_only:
        print(f"[staged] prepared {len(VARIANTS)} worktrees at base {base_commit}")
        return 0

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    state_root = ROOT / "logs" / "staged" / timestamp
    state_root.mkdir(parents=True, exist_ok=True)
    status_path = state_root / "status.json"

    manifest: dict[str, Any] = {
        "deduplication": {
            "enabled": not args.allow_duplicate,
            "allow_duplicate_override": "--allow-duplicate",
            "target_steps": TARGET_STEPS,
            "training_fingerprints": variant_fingerprints,
            "fingerprint_roots": list(TRAINING_FINGERPRINT_ROOTS),
            "ignored_launch_field": "experiment_label",
        }
    }
    manifest.update({
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
    })
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
    blocked_variants: set[str] = set()
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
            if identity.variant.name in failed_variants or identity.variant.name in blocked_variants:
                continue
            checkpoint = latest_checkpoints.get(identity.variant.name)

            if not args.allow_duplicate:
                duplicate = find_duplicate(
                    identity,
                    checkpoint,
                    variant_fingerprints[identity.variant.name],
                )
                if duplicate is not None:
                    finished[identity.variant.name].append(
                        skipped_duplicate_record(identity, duplicate, checkpoint)
                    )
                    print(
                        f"[staged] skipped {identity.name}: duplicate of {duplicate.run_dir} "
                        f"(checkpoint={duplicate.checkpoint_step or 'none'})",
                        flush=True,
                    )
                    if duplicate.checkpoint is not None:
                        latest_checkpoints[identity.variant.name] = duplicate.checkpoint
                        next_index = identity.stage_index + 1
                        stages = identity.variant.stages
                        if next_index < len(stages) and not stop_requested:
                            queue.append(StageIdentity(identity.variant, stages[next_index], next_index))
                    else:
                        blocked_variants.add(identity.variant.name)
                        print(
                            f"[staged] blocked {identity.variant.name}: duplicate has no checkpoint; "
                            "use --allow-duplicate to retry intentionally",
                            flush=True,
                        )
                    continue

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
            running[identity.name] = Running(
                identity,
                process,
                stdout_path,
                time.time(),
                input_checkpoint=checkpoint,
            )
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
            "blocked_variants": sorted(blocked_variants),
            "deduplication": {
                "enabled": not args.allow_duplicate,
                "training_fingerprints": variant_fingerprints,
            },
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
