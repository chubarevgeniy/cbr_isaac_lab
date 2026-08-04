#!/usr/bin/env python3
"""Resume the staged PPO cohort without repeating equivalent experiments.

The supervisor has two different checkpoint semantics:

* an interrupted stage resumes from its latest checkpoint, keeping the loaded
  Adam state and scheduler state when available;
* a completed stage starts the next curriculum stage from its policy/value
  checkpoint with a fresh optimizer and scheduler.

The requested timestep count is explicit. This is important because skrl's
trainer starts its loop at zero in every new process; ``--max_timesteps``
therefore represents the amount of work for this process, not a cumulative
counter from the previous process.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from staged_experiments import (
    BASELINE,
    CHECK_INTERVAL,
    EARLY_STOP_AFTER,
    ISAACLAB,
    MAX_CONCURRENT,
    NUM_ENVS,
    ROOT,
    ROLLOUTS_PER_ITERATION,
    TASK,
    TARGET_STEPS,
    TRAINING_FINGERPRINT_ROOTS,
    VARIANTS,
    ExistingRun,
    StageIdentity,
    Variant,
    checkpoint_path,
    checkpoint_state,
    effective_timesteps,
    existing_runs,
    expected_launch_signature,
    format_status,
    median,
    normalize_checkpoint,
    obvious_failure,
    prepare_worktree,
    run_git,
    scalar_snapshot,
    training_fingerprint,
)


VIRTUAL_ENV = Path("/home/evgenii/ws/isaac/env_isaaclab")
CHECKPOINT_INTERVAL = 5000
COMPLETION_CHECKPOINT = TARGET_STEPS - CHECKPOINT_INTERVAL


@dataclass(frozen=True)
class StageAction:
    identity: StageIdentity
    input_checkpoint: Path | None
    timesteps: int
    reset_optimizer_scheduler: bool
    mode: str
    source_run: Path | None = None
    resume_from_step: int = 0

    @property
    def name(self) -> str:
        return self.identity.name

    @property
    def variant(self) -> Variant:
        return self.identity.variant


@dataclass
class ActiveRun:
    action: StageAction
    process: subprocess.Popen[str]
    stdout_path: Path
    started_at: float
    last_status_at: float = 0.0
    last_checkpoint: int = 0
    last_checkpoint_mtime: float = 0.0
    run_dir: Path | None = None
    stopped_reason: str | None = None

    @property
    def identity(self) -> StageIdentity:
        return self.action.identity

    @property
    def variant(self) -> Variant:
        return self.identity.variant


def run_is_complete(run: ExistingRun) -> bool:
    return run.checkpoint_step >= COMPLETION_CHECKPOINT


def run_in_worktree(run: ExistingRun, worktree: Path) -> bool:
    try:
        run.run_dir.resolve().relative_to(worktree.resolve())
    except ValueError:
        return False
    return True


def variant_fingerprint(variant: Variant) -> str:
    commit = run_git("-C", str(variant.worktree), "rev-parse", "HEAD")
    fingerprint = training_fingerprint(commit)
    if fingerprint is None:
        raise RuntimeError(f"Could not fingerprint {variant.name} at {commit}")
    return fingerprint


def expected_signature(
    identity: StageIdentity,
    checkpoint: Path | None,
    *,
    timesteps: int,
    reset_optimizer_scheduler: bool,
) -> dict[str, Any]:
    return expected_launch_signature(
        identity,
        checkpoint,
        timesteps=timesteps,
        reset_optimizer_scheduler=reset_optimizer_scheduler,
    )


def matching_runs(
    identity: StageIdentity,
    checkpoint: Path | None,
    *,
    timesteps: int,
    reset_optimizer_scheduler: bool,
    fingerprint: str,
    runs: list[ExistingRun],
) -> list[ExistingRun]:
    signature = expected_signature(
        identity,
        checkpoint,
        timesteps=timesteps,
        reset_optimizer_scheduler=reset_optimizer_scheduler,
    )
    return [run for run in runs if run.fingerprint == fingerprint and run.signature == signature]


def current_incomplete_stage1(
    identity: StageIdentity,
    fingerprint: str,
    runs: list[ExistingRun],
) -> ExistingRun | None:
    candidates = [
        run
        for run in runs
        if run_in_worktree(run, identity.variant.worktree)
        and run.checkpoint is not None
        and not run_is_complete(run)
        and run.fingerprint == fingerprint
        and run.signature
        == expected_signature(
            identity,
            None,
            timesteps=TARGET_STEPS,
            reset_optimizer_scheduler=False,
        )
    ]
    return max(candidates, key=lambda run: (run.checkpoint_step, run.run_dir.stat().st_mtime), default=None)


def best_completed(
    candidates: list[ExistingRun],
) -> ExistingRun | None:
    complete = [run for run in candidates if run_is_complete(run) and run.checkpoint is not None]
    return max(complete, key=lambda run: (run.checkpoint_step, run.run_dir.stat().st_mtime), default=None)


def plan_stage(
    identity: StageIdentity,
    input_checkpoint: Path | None,
    *,
    fingerprint: str,
    runs: list[ExistingRun],
    allow_duplicate: bool,
) -> tuple[StageAction | None, ExistingRun | None, str]:
    """Plan one stage.

    Returns ``(action, reused_run, reason)``. ``action is None`` means that a
    completed equivalent run can be reused. An incomplete matching run with a
    checkpoint is resumed rather than started from scratch.
    """
    stage_index = identity.stage_index
    reset_for_new_stage = stage_index > 0

    if stage_index == 0 and input_checkpoint is None and not allow_duplicate:
        incomplete = current_incomplete_stage1(identity, fingerprint, runs)
        if incomplete is not None:
            remaining = TARGET_STEPS - incomplete.checkpoint_step
            return (
                StageAction(
                    identity=identity,
                    input_checkpoint=incomplete.checkpoint,
                    timesteps=remaining,
                    reset_optimizer_scheduler=False,
                    mode="resume_stage1",
                    source_run=incomplete.run_dir,
                    resume_from_step=incomplete.checkpoint_step,
                ),
                None,
                "resume current incomplete stage 1",
            )

    candidates = matching_runs(
        identity,
        input_checkpoint,
        timesteps=TARGET_STEPS,
        reset_optimizer_scheduler=reset_for_new_stage,
        fingerprint=fingerprint,
        runs=runs,
    )
    completed = best_completed(candidates)
    if completed is not None and not allow_duplicate:
        return None, completed, "reuse completed equivalent run"

    resumable = max(
        [run for run in candidates if run.checkpoint is not None and not run_is_complete(run)],
        key=lambda run: (run.checkpoint_step, run.run_dir.stat().st_mtime),
        default=None,
    )
    if resumable is not None and not allow_duplicate:
        remaining = TARGET_STEPS - resumable.checkpoint_step
        return (
            StageAction(
                identity=identity,
                input_checkpoint=resumable.checkpoint,
                timesteps=remaining,
                reset_optimizer_scheduler=False,
                mode="resume_incomplete",
                source_run=resumable.run_dir,
                resume_from_step=resumable.checkpoint_step,
            ),
            None,
            "resume matching incomplete run",
        )

    return (
        StageAction(
            identity=identity,
            input_checkpoint=input_checkpoint,
            timesteps=TARGET_STEPS,
            reset_optimizer_scheduler=reset_for_new_stage,
            mode="fresh_stage",
        ),
        None,
        "launch new stage",
    )


def command_for(action: StageAction) -> list[str]:
    variant = action.identity.variant
    stage = action.identity.stage
    command = [
        str(ISAACLAB / "isaaclab.sh"),
        "-p",
        "scripts/skrl/train.py",
        "--task",
        TASK,
        "--num_envs",
        str(NUM_ENVS),
        "--max_timesteps",
        str(action.timesteps),
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
        f"resume-{variant.name}-{stage.name}",
    ]
    if stage.disable_observation_noise:
        command.append("--disable_observation_noise")
    if stage.initial_tilt_deg is not None:
        command.extend(["--initial_tilt_deg", str(stage.initial_tilt_deg)])
    if action.input_checkpoint is not None:
        command.extend(["--checkpoint", str(action.input_checkpoint)])
    if action.reset_optimizer_scheduler:
        command.append("--reset_optimizer_scheduler")
    return command


def discover_run_dir(action: StageAction, started_at: float) -> Path | None:
    root = action.identity.variant.worktree / "logs" / "skrl" / "cbr_i_ppo"
    if not root.exists():
        return None
    label = f"resume-{action.identity.variant.name}-{action.identity.stage.name}"
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and label in path.name and path.stat().st_mtime >= started_at - 5
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def stage_record(active: ActiveRun, returncode: int | None, status: str) -> dict[str, Any]:
    return {
        "stage": active.identity.stage.name,
        "stage_index": active.identity.stage_index,
        "variant": active.variant.name,
        "branch": active.variant.branch,
        "mode": active.action.mode,
        "reward_profile": active.identity.stage.reward_profile,
        "disable_observation_noise": active.identity.stage.disable_observation_noise,
        "initial_tilt_deg": active.identity.stage.initial_tilt_deg,
        "seed": active.variant.seed,
        "input_checkpoint": str(active.action.input_checkpoint) if active.action.input_checkpoint else None,
        "resume_from_step": active.action.resume_from_step,
        "requested_timesteps": active.action.timesteps,
        "reset_optimizer_scheduler": active.action.reset_optimizer_scheduler,
        "source_run": str(active.action.source_run) if active.action.source_run else None,
        "run_dir": str(active.run_dir) if active.run_dir else None,
        "stdout": str(active.stdout_path),
        "last_checkpoint": active.last_checkpoint,
        "returncode": returncode,
        "status": status,
        "stopped_reason": active.stopped_reason,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def reused_record(
    identity: StageIdentity,
    run: ExistingRun,
    reason: str,
) -> dict[str, Any]:
    return {
        "stage": identity.stage.name,
        "stage_index": identity.stage_index,
        "variant": identity.variant.name,
        "branch": identity.variant.branch,
        "mode": "reuse_completed",
        "reward_profile": identity.stage.reward_profile,
        "seed": identity.variant.seed,
        "run_dir": str(run.run_dir),
        "checkpoint": str(run.checkpoint) if run.checkpoint else None,
        "last_checkpoint": run.checkpoint_step,
        "status": "skipped_duplicate",
        "reason": reason,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def stop_process(active: ActiveRun, reason: str) -> None:
    active.stopped_reason = reason
    try:
        os.killpg(active.process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        active.process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(active.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def write_status(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true", help="Plan actions without starting Isaac Sim.")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare missing staged worktrees only.")
    parser.add_argument(
        "--allow-duplicate",
        action="store_true",
        help="Disable completed-run deduplication; incomplete runs are still resumed when possible.",
    )
    args = parser.parse_args()

    if not ISAACLAB.joinpath("isaaclab.sh").exists():
        raise SystemExit(f"IsaacLab launcher not found: {ISAACLAB / 'isaaclab.sh'}")
    if run_git("status", "--porcelain=1", "--untracked-files=all"):
        raise SystemExit("Working tree must be clean before starting the resume supervisor")

    base_commit = run_git("rev-parse", "HEAD")
    for variant in VARIANTS:
        prepare_worktree(variant, base_commit)
    if args.prepare_only:
        print(f"[resume-staged] prepared {len(VARIANTS)} worktrees")
        return 0

    fingerprints = {variant.name: variant_fingerprint(variant) for variant in VARIANTS}
    runs = existing_runs()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    state_root = ROOT / "logs" / "staged_resume" / timestamp
    state_root.mkdir(parents=True, exist_ok=True)
    status_path = state_root / "status.json"

    manifest = {
        "base_commit": base_commit,
        "target_steps": TARGET_STEPS,
        "rollouts_per_iteration": ROLLOUTS_PER_ITERATION,
        "max_concurrent": max(1, args.max_concurrent),
        "deduplication_enabled": not args.allow_duplicate,
        "fingerprint_roots": list(TRAINING_FINGERPRINT_ROOTS),
        "training_fingerprints": fingerprints,
        "variants": {
            variant.name: {
                "branch": variant.branch,
                "worktree": str(variant.worktree),
                "seed": variant.seed,
                "stages": [stage.name for stage in variant.stages],
            }
            for variant in VARIANTS
        },
    }
    (state_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    queue: list[StageAction] = []
    running: dict[str, ActiveRun] = {}
    finished: dict[str, list[dict[str, Any]]] = {variant.name: [] for variant in VARIANTS}
    latest_checkpoints: dict[str, Path] = {}
    failed_variants: set[str] = set()
    stop_requested = False

    def add_next_action(
        variant: Variant,
        next_index: int,
        input_checkpoint: Path | None,
        destination: list[StageAction] | None = None,
    ) -> None:
        if next_index >= len(variant.stages):
            return
        identity = StageIdentity(variant, variant.stages[next_index], next_index)
        action, reused, reason = plan_stage(
            identity,
            input_checkpoint,
            fingerprint=fingerprints[variant.name],
            runs=existing_runs(),
            allow_duplicate=args.allow_duplicate,
        )
        if reused is not None:
            finished[variant.name].append(reused_record(identity, reused, reason))
            if reused.checkpoint is None:
                failed_variants.add(variant.name)
                return
            latest_checkpoints[variant.name] = reused.checkpoint
            add_next_action(variant, next_index + 1, reused.checkpoint, destination)
        elif action is not None:
            (destination if destination is not None else queue).append(action)

    # Resolve all already-completed stages first. Incomplete current stage-1
    # runs are deliberately queued before new stage-2 actions.
    initial_actions: list[StageAction] = []
    later_actions: list[StageAction] = []
    for variant in VARIANTS:
        identity = StageIdentity(variant, variant.stages[0], 0)
        action, reused, reason = plan_stage(
            identity,
            None,
            fingerprint=fingerprints[variant.name],
            runs=runs,
            allow_duplicate=args.allow_duplicate,
        )
        if reused is not None:
            finished[variant.name].append(reused_record(identity, reused, reason))
            if reused.checkpoint is None:
                failed_variants.add(variant.name)
                continue
            latest_checkpoints[variant.name] = reused.checkpoint
            add_next_action(variant, 1, reused.checkpoint, later_actions)
        elif action is not None:
            (initial_actions if action.identity.stage_index == 0 else later_actions).append(action)
    queue = initial_actions + later_actions

    if args.dry_run:
        for variant, records in finished.items():
            for record in records:
                print(
                    f"[resume-staged] {variant}/{record['stage']} "
                    f"{record['status']} {record.get('run_dir', '')}"
                )
        for action in queue:
            print(
                f"[resume-staged] {action.name} mode={action.mode} "
                f"timesteps={action.timesteps} checkpoint={action.input_checkpoint or 'none'}"
            )
            print(" ", " ".join(command_for(action)))
        return 0

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        queue.clear()
        print(f"[resume-staged] received signal {signum}; stopping active runs", flush=True)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while queue or running:
        while not stop_requested and queue and len(running) < max(1, args.max_concurrent):
            action = queue.pop(0)
            if action.variant.name in failed_variants:
                continue
            stdout_path = state_root / f"{action.variant.name}__{action.identity.stage.name}.stdout.log"
            environment = os.environ.copy()
            environment["VIRTUAL_ENV"] = str(VIRTUAL_ENV)
            environment["PYTHONUNBUFFERED"] = "1"
            source_path = str(action.variant.worktree / "source" / "CBRIIsaacLab")
            environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
            stdout = stdout_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command_for(action),
                cwd=action.variant.worktree,
                env=environment,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            stdout.close()
            running[action.name] = ActiveRun(action, process, stdout_path, time.time())
            print(
                f"[resume-staged] launched {action.name} mode={action.mode} "
                f"timesteps={action.timesteps} checkpoint={action.input_checkpoint or 'none'} "
                f"pid={process.pid}",
                flush=True,
            )

        for name, active in list(running.items()):
            if active.run_dir is None:
                active.run_dir = discover_run_dir(active.action, active.started_at)
            active.last_checkpoint, active.last_checkpoint_mtime = checkpoint_state(active.run_dir)
            snapshot = scalar_snapshot(active.run_dir)
            now = time.time()
            if now - active.last_status_at >= 300:
                print(format_status(active, snapshot), flush=True)
                active.last_status_at = now

            reason = obvious_failure(active, snapshot)
            if reason:
                print(f"[resume-staged] stopping {name}: {reason}", flush=True)
                stop_process(active, reason)

            if active.process.poll() is not None:
                returncode = active.process.returncode
                output_checkpoint = checkpoint_path(active.run_dir)
                succeeded = returncode == 0 and output_checkpoint is not None
                status = "completed" if succeeded else "failed"
                if active.stopped_reason:
                    status = "stopped_early"
                finished[active.variant.name].append(stage_record(active, returncode, status))
                if succeeded:
                    latest_checkpoints[active.variant.name] = output_checkpoint  # type: ignore[assignment]
                    add_next_action(
                        active.variant,
                        active.identity.stage_index + 1,
                        output_checkpoint,
                    )
                else:
                    failed_variants.add(active.variant.name)
                del running[name]
                print(f"[resume-staged] finished {name} returncode={returncode} status={status}", flush=True)

        state = {
            "started_at": timestamp,
            "base_commit": base_commit,
            "queue": [action.name for action in queue],
            "running": {
                name: {
                    "pid": active.process.pid,
                    "stage": active.identity.stage.name,
                    "variant": active.variant.name,
                    "mode": active.action.mode,
                    "run_dir": str(active.run_dir) if active.run_dir else None,
                    "last_checkpoint": active.last_checkpoint,
                    "stdout": str(active.stdout_path),
                }
                for name, active in running.items()
            },
            "finished": finished,
            "failed_variants": sorted(failed_variants),
            "deduplication_enabled": not args.allow_duplicate,
        }
        write_status(status_path, state)
        if queue or running:
            time.sleep(CHECK_INTERVAL)

    print(f"[resume-staged] cohort complete; state={status_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
