#!/usr/bin/env python3
"""Run and conservatively monitor the overnight 64k-step experiment cohort.

The cohort deliberately uses git worktrees and explicit train.py arguments. A
run therefore records both the branch and the resolved action/reward settings,
even when several related edits are bundled into one hypothesis.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


BASELINE = {
    "lifetime": 378.2,
    "termination": 0.00262,
    "speed_error": 0.429,
    "foot_height": 0.0167,
    "sit_error": 0.299,
}


@dataclass(frozen=True)
class Variant:
    name: str
    branch: str
    action_mode: str
    reward_profile: str
    seed: int = 42
    bounded_policy: bool = True

    @property
    def worktree(self) -> Path:
        return ROOT.parent / f"cbr_i_overnight_{self.name.replace('-', '_')}"


@dataclass
class Running:
    variant: Variant
    process: subprocess.Popen[str]
    stdout_path: Path
    started_at: float
    last_status_at: float = 0.0
    last_checkpoint: int = 0
    last_checkpoint_mtime: float = 0.0
    run_dir: Path | None = None
    stopped_reason: str | None = None


VARIANTS = (
    Variant("long-baseline", "experiment/overnight-long-baseline", "delta", "baseline", bounded_policy=False),
    Variant("delta-bounded", "experiment/overnight-delta-bounded", "delta", "baseline"),
    Variant("delta-survival", "experiment/overnight-delta-survival", "delta", "survival_clearance_speed"),
    Variant("delta-smooth-clearance", "experiment/overnight-delta-smooth-clearance", "delta", "smooth_clearance"),
    Variant("absolute-target", "experiment/overnight-absolute-target", "absolute", "baseline"),
    Variant("absolute-safe-task", "experiment/overnight-absolute-safe-task", "absolute", "task_balanced"),
    Variant("absolute-clearance", "experiment/overnight-absolute-clearance", "absolute", "survival_clearance_speed"),
    Variant("delta-task-repeat", "experiment/overnight-delta-task-repeat", "delta", "task_balanced", seed=43),
)


ROOT = Path(__file__).resolve().parents[1]
ISAACLAB = ROOT.parent / "IsaacLab"
VIRTUAL_ENV = Path("/home/evgenii/ws/isaac/env_isaaclab")
TASK = "Template-Cbriisaaclab-Direct-v0"
NUM_ENVS = 2048
MAX_ITERATIONS = 2000  # 2000 * rollouts(32) = 64,000 environment timesteps
MAX_CONCURRENT = 2
CHECK_INTERVAL = 60
EARLY_STOP_AFTER = 30 * 60


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


def command_for(variant: Variant) -> list[str]:
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
        variant.reward_profile,
        "--experiment_label",
        f"overnight-{variant.name}",
    ]
    if variant.bounded_policy:
        command += [
            "--policy_clip_actions",
            "--policy_initial_log_std=-0.7",
            "--policy_max_log_std=0.0",
        ]
    return command


def discover_run_dir(variant: Variant, started_at: float) -> Path | None:
    root = variant.worktree / "logs" / "skrl" / "cbr_i_ppo"
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and f"overnight-{variant.name}" in path.name
        and path.stat().st_mtime >= started_at - 5
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def checkpoint_state(run_dir: Path | None) -> tuple[int, float]:
    if run_dir is None:
        return 0, 0.0
    checkpoints = list((run_dir / "checkpoints").glob("agent_*.pt"))
    parsed: list[tuple[int, float]] = []
    for checkpoint in checkpoints:
        try:
            step = int(checkpoint.stem.removeprefix("agent_"))
        except ValueError:
            continue
        parsed.append((step, checkpoint.stat().st_mtime))
    return max(parsed, default=(0, 0.0))


def scalar_snapshot(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator = EventAccumulator(
            str(run_dir),
            size_guidance={"scalars": 0, "histograms": 0, "tensors": 0},
        )
        accumulator.Reload()
    except Exception as exc:  # TensorBoard may see a file while Kit is writing it.
        return {"error": str(exc)}

    wanted = {
        "lifetime": "Episode / Total timesteps (mean)",
        "termination": "Physical/termination/terminated_rate",
        "speed_error": "Physical/walk/speed_error_abs",
        "foot_height": "Physical/walk/mean_foot_height",
        "sit_error": "Physical/sit/mean_joint_angle_error_abs",
        "moving": "Physical/command/moving_fraction",
        "action_rate": "Physical/action/mean_abs_rate",
        "saturation": "Physical/action/saturation_fraction",
        "learning_rate": "Learning / Learning rate",
    }
    result: dict[str, Any] = {}
    for key, tag in wanted.items():
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        events = accumulator.Scalars(tag)
        if not events:
            continue
        values = [float(event.value) for event in events]
        result[key] = {
            "latest": values[-1],
            "recent": values[-20:],
            "prior": values[-40:-20],
            "step": int(events[-1].step),
        }
    return result


def latest(snapshot: dict[str, Any], key: str) -> float | None:
    value = snapshot.get(key)
    return float(value["latest"]) if isinstance(value, dict) else None


def median(snapshot: dict[str, Any], key: str, part: str = "recent") -> float | None:
    value = snapshot.get(key)
    if not isinstance(value, dict) or not value.get(part):
        return None
    return float(statistics.median(value[part]))


def obvious_failure(running: Running, snapshot: dict[str, Any]) -> str | None:
    """Return a reason only for a conservative, multi-signal hard failure."""
    age = time.time() - running.started_at
    if age < EARLY_STOP_AFTER:
        return None
    if running.last_checkpoint == 0 and age > 40 * 60:
        return "no checkpoint after 40 minutes"
    if running.last_checkpoint > 0 and time.time() - running.last_checkpoint_mtime > 20 * 60:
        return f"no new checkpoint for more than 20 minutes (last={running.last_checkpoint})"
    if not snapshot or "error" in snapshot:
        return None

    lifetime = median(snapshot, "lifetime")
    termination = median(snapshot, "termination")
    speed_error = median(snapshot, "speed_error")
    foot_height = median(snapshot, "foot_height")
    sit_error = median(snapshot, "sit_error")
    learning_rate = latest(snapshot, "learning_rate")
    if any(value is None for value in (lifetime, termination, speed_error, foot_height, sit_error)):
        return None

    bad_signals = 0
    if lifetime < 0.35 * BASELINE["lifetime"] and termination > 4.0 * BASELINE["termination"]:
        bad_signals += 2
    if speed_error > 0.80 and foot_height < 0.008:
        bad_signals += 1
    if sit_error > 0.60 and lifetime < 0.60 * BASELINE["lifetime"]:
        bad_signals += 1
    if learning_rate is not None and learning_rate <= 2.05e-4 and (
        speed_error > 0.65 or termination > 3.0 * BASELINE["termination"]
    ):
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
            f"multi-signal failure: lifetime={lifetime:.1f}, termination={termination:.4f}, "
            f"speed_error={speed_error:.3f}, foot_height={foot_height:.4f}, sit_error={sit_error:.3f}, "
            f"lr={learning_rate if learning_rate is not None else 'n/a'}"
        )
    return None


def format_status(running: Running, snapshot: dict[str, Any]) -> str:
    age_minutes = (time.time() - running.started_at) / 60.0
    step = latest(snapshot, "lifetime")
    step_text = f"log-step={int(snapshot['lifetime']['step'])}" if isinstance(snapshot.get("lifetime"), dict) else "log-step=?"
    values = []
    for key, fmt in (("lifetime", ".1f"), ("termination", ".4f"), ("speed_error", ".3f"), ("foot_height", ".4f"), ("sit_error", ".3f")):
        value = latest(snapshot, key)
        values.append(f"{key}={value:{fmt}}" if value is not None else f"{key}=?")
    return f"[monitor] {running.variant.name} age={age_minutes:.1f}m {step_text} " + " ".join(values)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true", help="Prepare worktrees and print commands only.")
    args = parser.parse_args()

    if not ISAACLAB.joinpath("isaaclab.sh").exists():
        raise SystemExit(f"IsaacLab launcher not found: {ISAACLAB / 'isaaclab.sh'}")
    base_commit = run_git("rev-parse", "HEAD")
    for variant in VARIANTS:
        prepare_worktree(variant, base_commit)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    state_root = ROOT / "logs" / "overnight" / timestamp
    state_root.mkdir(parents=True, exist_ok=True)
    status_path = state_root / "status.json"
    command_manifest = {
        variant.name: {
            "branch": variant.branch,
            "worktree": str(variant.worktree),
            "seed": variant.seed,
            "action_mode": variant.action_mode,
            "reward_profile": variant.reward_profile,
            "bounded_policy": variant.bounded_policy,
            "command": command_for(variant),
        }
        for variant in VARIANTS
    }
    (state_root / "commands.json").write_text(json.dumps(command_manifest, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        for variant in VARIANTS:
            print(variant.name, "\t", " ".join(command_for(variant)))
        return 0

    queue = list(VARIANTS)
    running: dict[str, Running] = {}
    finished: dict[str, dict[str, Any]] = {}
    stop_requested = False

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[monitor] received signal {signum}; stopping active runs", flush=True)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while queue or running:
        if stop_requested:
            for active in list(running.values()):
                stop_process(active, "supervisor interrupted")

        while not stop_requested and queue and len(running) < max(1, args.max_concurrent):
            variant = queue.pop(0)
            stdout_path = state_root / f"{variant.name}.stdout.log"
            stdout = stdout_path.open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["VIRTUAL_ENV"] = str(VIRTUAL_ENV)
            environment["PYTHONUNBUFFERED"] = "1"
            environment["PYTHONPATH"] = str(variant.worktree / "source" / "CBRIIsaacLab") + os.pathsep + environment.get("PYTHONPATH", "")
            process = subprocess.Popen(
                command_for(variant),
                cwd=variant.worktree,
                env=environment,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            stdout.close()
            running[variant.name] = Running(variant, process, stdout_path, time.time())
            print(f"[monitor] launched {variant.name} pid={process.pid} branch={variant.branch}", flush=True)

        for name, active in list(running.items()):
            if active.run_dir is None:
                active.run_dir = discover_run_dir(active.variant, active.started_at)
            active.last_checkpoint, active.last_checkpoint_mtime = checkpoint_state(active.run_dir)
            snapshot = scalar_snapshot(active.run_dir)
            now = time.time()
            if now - active.last_status_at >= 300:
                print(format_status(active, snapshot), flush=True)
                active.last_status_at = now

            reason = obvious_failure(active, snapshot)
            if reason:
                print(f"[monitor] stopping {name}: {reason}", flush=True)
                stop_process(active, reason)

            if active.process.poll() is not None:
                finished[name] = {
                    "branch": active.variant.branch,
                    "run_dir": str(active.run_dir) if active.run_dir else None,
                    "returncode": active.process.returncode,
                    "stopped_reason": active.stopped_reason,
                    "last_checkpoint": active.last_checkpoint,
                    "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                }
                del running[name]
                print(f"[monitor] finished {name} returncode={active.process.returncode}", flush=True)

        state = {
            "started_at": timestamp,
            "base_commit": base_commit,
            "queue": [variant.name for variant in queue],
            "running": {
                name: {
                    "pid": active.process.pid,
                    "branch": active.variant.branch,
                    "run_dir": str(active.run_dir) if active.run_dir else None,
                    "last_checkpoint": active.last_checkpoint,
                    "stdout": str(active.stdout_path),
                }
                for name, active in running.items()
            },
            "finished": finished,
        }
        write_status(status_path, state)
        if queue or running:
            time.sleep(CHECK_INTERVAL)

    print(f"[monitor] cohort complete; state={status_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
