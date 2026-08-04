#!/usr/bin/env python3
"""Run the two-anchor action-regularization experiment cohort.

The cohort starts from two existing checkpoints:

* the unbounded noisy baseline, where raw policy actions are the main
  hypothesis under test;
* the bounded ``delta-task-repeat`` policy, where the question is whether
  additional raw action-rate/magnitude penalties can improve smoothness
  without losing its task-balanced survival.

Eight 64k screening jobs are run in two slots.  After all screening jobs have
finished, the supervisor selects one regularized winner per anchor using the
physical metrics and queues a 128k continuation.  If both continuations
finish before the wall-clock deadline, it queues a final dose-response stage
with doubled non-default regularization coefficients.

Use ``--dry-run`` to print the complete initial cohort and command templates.
The script never starts Isaac Sim in dry-run mode.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ISAACLAB = ROOT.parent / "IsaacLab"
VIRTUAL_ENV = Path("/home/evgenii/ws/isaac/env_isaaclab")
TASK = "Template-Cbriisaaclab-Direct-v0"
NUM_ENVS = 2048
ROLLOUTS = 32
DEFAULT_MAGNITUDE = 1.0e-5
DEFAULT_INITIAL_TIMESTEPS = 64_000
DEFAULT_CONTINUATION_TIMESTEPS = 128_000
DEFAULT_STRENGTHEN_TIMESTEPS = 64_000
DEFAULT_MAX_CONCURRENT = 2
DEFAULT_WALL_HOURS = 12.0
CHECK_INTERVAL = 60
STATUS_INTERVAL = 300
NO_CHECKPOINT_TIMEOUT = 45 * 60

BASELINE_CHECKPOINT = Path(
    "/home/evgenii/ws/isaac/cbr_i_overnight_long_baseline/logs/skrl/cbr_i_ppo/"
    "2026-08-03_22-43-25_experiment_overnight-long-baseline_7a882e0_clean_ppo_torch_"
    "overnight-long-baseline/checkpoints/agent_60000.pt"
)
TASK_BALANCED_CHECKPOINT = Path(
    "/home/evgenii/ws/isaac/cbr_i_overnight_delta_task_repeat/logs/skrl/cbr_i_ppo/"
    "2026-08-04_03-11-42_experiment_overnight-delta-task-repeat_7a882e0_clean_ppo_torch_"
    "overnight-delta-task-repeat/checkpoints/agent_60000.pt"
)

METRIC_TAGS = {
    "lifetime": "Episode / Total timesteps (mean)",
    "termination": "Physical/termination/terminated_rate",
    "speed_error": "Physical/walk/speed_error_abs",
    "foot_height": "Physical/walk/mean_foot_height",
    "sit_error": "Physical/sit/mean_joint_angle_error_abs",
    "action_rate": "Physical/action/mean_abs_rate",
    "action_magnitude": "Physical/action/mean_abs",
    "target_step": "Physical/target/mean_abs_step",
    "saturation": "Physical/action/saturation_fraction",
    "learning_rate": "Learning / Learning rate",
}


@dataclass(frozen=True)
class Job:
    name: str
    phase: str
    anchor: str
    kind: str
    reward_profile: str
    seed: int
    policy_clip_actions: bool
    policy_initial_log_std: float | None
    policy_max_log_std: float | None
    action_magnitude_scale: float
    action_rate_scale: float
    timesteps: int
    checkpoint: Path
    parent: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "phase": self.phase,
            "anchor": self.anchor,
            "kind": self.kind,
            "reward_profile": self.reward_profile,
            "seed": self.seed,
            "policy_clip_actions": self.policy_clip_actions,
            "policy_initial_log_std": self.policy_initial_log_std,
            "policy_max_log_std": self.policy_max_log_std,
            "action_magnitude_scale": self.action_magnitude_scale,
            "action_rate_scale": self.action_rate_scale,
            "timesteps": self.timesteps,
            "checkpoint": str(self.checkpoint),
            "parent": self.parent,
        }
        return result


@dataclass
class Running:
    job: Job
    process: subprocess.Popen[str]
    stdout_path: Path
    started_at: float
    run_dir: Path | None = None
    last_checkpoint: int = 0
    last_checkpoint_mtime: float = 0.0
    last_status_at: float = 0.0
    stopped_reason: str | None = None


def run_git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def format_float(value: float) -> str:
    return f"{value:.12g}"


def initial_jobs(initial_timesteps: int) -> list[Job]:
    jobs: list[Job] = []

    baseline_specs = (
        ("control", DEFAULT_MAGNITUDE, 0.0),
        ("magnitude", 5.0e-5, 0.0),
        ("rate", DEFAULT_MAGNITUDE, 1.5e-3),
        ("combined", 5.0e-5, 1.5e-3),
    )
    for kind, magnitude, rate in baseline_specs:
        jobs.append(
            Job(
                name=f"baseline-{kind}",
                phase="initial",
                anchor="baseline",
                kind=kind,
                reward_profile="baseline",
                seed=42,
                policy_clip_actions=False,
                policy_initial_log_std=None,
                policy_max_log_std=None,
                action_magnitude_scale=magnitude,
                action_rate_scale=rate,
                timesteps=initial_timesteps,
                checkpoint=BASELINE_CHECKPOINT,
            )
        )

    task_specs = (
        ("control", DEFAULT_MAGNITUDE, 1.5e-3),
        ("magnitude", 5.0e-5, 1.5e-3),
        ("rate", DEFAULT_MAGNITUDE, 3.0e-3),
        ("combined", 5.0e-5, 3.0e-3),
    )
    for kind, magnitude, rate in task_specs:
        jobs.append(
            Job(
                name=f"task-balanced-{kind}",
                phase="initial",
                anchor="task-balanced",
                kind=kind,
                reward_profile="task_balanced",
                seed=43,
                policy_clip_actions=True,
                policy_initial_log_std=-0.7,
                policy_max_log_std=0.0,
                action_magnitude_scale=magnitude,
                action_rate_scale=rate,
                timesteps=initial_timesteps,
                checkpoint=TASK_BALANCED_CHECKPOINT,
            )
        )

    return jobs


def command_for(job: Job) -> list[str]:
    command = [
        str(ISAACLAB / "isaaclab.sh"),
        "-p",
        "scripts/skrl/train.py",
        "--task",
        TASK,
        "--num_envs",
        str(NUM_ENVS),
        "--max_timesteps",
        str(job.timesteps),
        "--seed",
        str(job.seed),
        "--action_mode",
        "delta",
        "--reward_profile",
        job.reward_profile,
        "--action_magnitude_scale",
        format_float(job.action_magnitude_scale),
        "--action_rate_scale",
        format_float(job.action_rate_scale),
        "--checkpoint",
        str(job.checkpoint),
        "--reset_optimizer_scheduler",
        "--experiment_label",
        f"action-reg-{job.name}",
        "--headless",
    ]
    if job.policy_clip_actions:
        command.extend(
            [
                "--policy_clip_actions",
                f"--policy_initial_log_std={format_float(job.policy_initial_log_std or 0.0)}",
                f"--policy_max_log_std={format_float(job.policy_max_log_std or 0.0)}",
            ]
        )
    return command


def command_text(job: Job) -> str:
    return " ".join(command_for(job))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def discover_run_dir(job: Job, started_at: float) -> Path | None:
    root = ROOT / "logs" / "skrl" / "cbr_i_ppo"
    if not root.exists():
        return None
    label = f"action-reg-{job.name}"
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and label in path.name and path.stat().st_mtime >= started_at - 5
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def checkpoint_state(run_dir: Path | None) -> tuple[int, float]:
    if run_dir is None:
        return 0, 0.0
    checkpoints: list[tuple[int, float]] = []
    for path in (run_dir / "checkpoints").glob("agent_*.pt"):
        try:
            step = int(path.stem.removeprefix("agent_"))
        except ValueError:
            continue
        checkpoints.append((step, path.stat().st_mtime))
    return max(checkpoints, default=(0, 0.0))


def latest_checkpoint(run_dir: Path | None) -> Path | None:
    if run_dir is None:
        return None
    candidates: list[tuple[int, Path]] = []
    for path in (run_dir / "checkpoints").glob("agent_*.pt"):
        try:
            candidates.append((int(path.stem.removeprefix("agent_")), path))
        except ValueError:
            continue
    return max(candidates, default=(0, None))[1]


def read_metrics(run_dir: Path | None) -> dict[str, dict[str, float | int]]:
    if run_dir is None:
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator = EventAccumulator(
            str(run_dir), size_guidance={"scalars": 0, "histograms": 0, "tensors": 0}
        )
        accumulator.Reload()
    except Exception as exc:
        return {"_error": {"message": str(exc)}}

    metrics: dict[str, dict[str, float | int]] = {}
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    for key, tag in METRIC_TAGS.items():
        if tag not in scalar_tags:
            continue
        events = accumulator.Scalars(tag)
        if not events:
            continue
        values = [float(event.value) for event in events]
        recent = values[-20:]
        metrics[key] = {
            "latest": values[-1],
            "median": float(statistics.median(recent)),
            "step": int(events[-1].step),
        }
    return metrics


def metric(record: dict[str, Any] | None, key: str) -> float | None:
    if not record:
        return None
    value = (record.get("metrics") or {}).get(key)
    if not isinstance(value, dict):
        return None
    candidate = value.get("median", value.get("latest"))
    return float(candidate) if candidate is not None else None


def record_for(job: Job, running: Running, returncode: int | None, status: str) -> dict[str, Any]:
    metrics = read_metrics(running.run_dir)
    return {
        "job": job.as_dict(),
        "run_dir": str(running.run_dir) if running.run_dir else None,
        "stdout": str(running.stdout_path),
        "last_checkpoint": running.last_checkpoint,
        "returncode": returncode,
        "status": status,
        "stopped_reason": running.stopped_reason,
        "metrics": metrics,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def format_status(running: Running, metrics: dict[str, Any]) -> str:
    def value(key: str, fmt: str) -> str:
        data = metrics.get(key)
        if not isinstance(data, dict):
            return "?"
        return format(float(data.get("median", data.get("latest", 0.0))), fmt)

    age = (time.time() - running.started_at) / 60.0
    step = metrics.get("lifetime", {}).get("step", "?") if isinstance(metrics.get("lifetime"), dict) else "?"
    return (
        f"[monitor] {running.job.name} age={age:.1f}m step={step} "
        f"checkpoint={running.last_checkpoint} "
        f"lifetime={value('lifetime', '.1f')} termination={value('termination', '.4f')} "
        f"speed={value('speed_error', '.3f')} sit={value('sit_error', '.3f')} "
        f"action_rate={value('action_rate', '.3f')} target_step={value('target_step', '.4f')}"
    )


def stop_process(running: Running, reason: str) -> None:
    running.stopped_reason = reason
    try:
        os.killpg(running.process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        running.process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(running.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def choose_winner(records: dict[str, dict[str, Any]], anchor: str) -> dict[str, Any] | None:
    initial = [
        record
        for record in records.values()
        if record["job"]["phase"] == "initial" and record["job"]["anchor"] == anchor
    ]
    if not initial:
        return None

    controls = [record for record in initial if record["job"]["kind"] == "control"]
    control = controls[0] if controls else None
    control_lifetime = metric(control, "lifetime")
    control_speed = metric(control, "speed_error")
    control_sit = metric(control, "sit_error")

    regularized = [record for record in initial if record["job"]["kind"] != "control"]
    viable: list[dict[str, Any]] = []
    for record in regularized:
        lifetime = metric(record, "lifetime")
        speed = metric(record, "speed_error")
        sit = metric(record, "sit_error")
        if lifetime is None:
            continue
        if control_lifetime is not None and lifetime < 0.90 * control_lifetime:
            continue
        if control_speed is not None and speed is not None and speed > 1.10 * control_speed:
            continue
        if control_sit is not None and sit is not None and sit > 1.10 * control_sit:
            continue
        viable.append(record)

    pool = viable or [record for record in regularized if metric(record, "lifetime") is not None]
    if not pool:
        return control

    def sort_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
        action_rate = metric(record, "action_rate")
        termination = metric(record, "termination")
        lifetime = metric(record, "lifetime")
        speed = metric(record, "speed_error")
        return (
            action_rate if action_rate is not None else float("inf"),
            termination if termination is not None else float("inf"),
            -(lifetime if lifetime is not None else 0.0),
            speed if speed is not None else float("inf"),
        )

    return min(pool, key=sort_key)


def job_from_record(record: dict[str, Any]) -> Job:
    spec = record["job"]
    return Job(
        name=spec["name"],
        phase=spec["phase"],
        anchor=spec["anchor"],
        kind=spec["kind"],
        reward_profile=spec["reward_profile"],
        seed=int(spec["seed"]),
        policy_clip_actions=bool(spec["policy_clip_actions"]),
        policy_initial_log_std=spec["policy_initial_log_std"],
        policy_max_log_std=spec["policy_max_log_std"],
        action_magnitude_scale=float(spec["action_magnitude_scale"]),
        action_rate_scale=float(spec["action_rate_scale"]),
        timesteps=int(spec["timesteps"]),
        checkpoint=Path(spec["checkpoint"]),
        parent=spec.get("parent"),
    )


def make_continuation(record: dict[str, Any], timesteps: int) -> Job | None:
    checkpoint = latest_checkpoint(Path(record["run_dir"]) if record.get("run_dir") else None)
    if checkpoint is None:
        return None
    base = job_from_record(record)
    return replace(
        base,
        name=f"{base.name}-continue",
        phase="continue",
        timesteps=timesteps,
        checkpoint=checkpoint,
        parent=base.name,
    )


def make_strengthen(record: dict[str, Any], timesteps: int) -> Job | None:
    checkpoint = latest_checkpoint(Path(record["run_dir"]) if record.get("run_dir") else None)
    if checkpoint is None:
        return None
    base = job_from_record(record)
    magnitude = base.action_magnitude_scale
    rate = base.action_rate_scale
    if magnitude > DEFAULT_MAGNITUDE:
        magnitude *= 2.0
    if rate > 0.0:
        rate *= 2.0
    if magnitude == base.action_magnitude_scale and rate == base.action_rate_scale:
        return None
    return replace(
        base,
        name=f"{base.name}-strengthen",
        phase="strengthen",
        action_magnitude_scale=magnitude,
        action_rate_scale=rate,
        timesteps=timesteps,
        checkpoint=checkpoint,
        parent=base.name,
    )


def preflight(allow_dirty: bool) -> None:
    if not ISAACLAB.joinpath("isaaclab.sh").exists():
        raise SystemExit(f"IsaacLab launcher not found: {ISAACLAB / 'isaaclab.sh'}")
    if not VIRTUAL_ENV.joinpath("bin/python").exists():
        raise SystemExit(f"IsaacLab virtualenv not found: {VIRTUAL_ENV}")
    for path in (BASELINE_CHECKPOINT, TASK_BALANCED_CHECKPOINT):
        if not path.is_file():
            raise SystemExit(f"Required checkpoint not found: {path}")
    if not allow_dirty and run_git("status", "--porcelain=1", "--untracked-files=all"):
        raise SystemExit("Working tree must be clean before starting the experiment supervisor")


def run_supervisor(args: argparse.Namespace) -> int:
    preflight(args.allow_dirty)
    jobs = initial_jobs(args.initial_timesteps)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    state_root = ROOT / "logs" / "action_regularization" / timestamp
    state_root.mkdir(parents=True, exist_ok=True)
    status_path = state_root / "status.json"
    manifest_path = state_root / "manifest.json"
    deadline = time.time() + args.wall_hours * 3600.0

    write_json(
        manifest_path,
        {
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": run_git("rev-parse", "--short", "HEAD"),
            "root": str(ROOT),
            "task": TASK,
            "num_envs": NUM_ENVS,
            "rollouts": ROLLOUTS,
            "wall_hours": args.wall_hours,
            "initial_jobs": [job.as_dict() | {"command": command_for(job)} for job in jobs],
        },
    )

    queue = list(jobs)
    running: dict[str, Running] = {}
    finished: dict[str, dict[str, Any]] = {}
    phase = "initial"
    followups_added = False
    strengthen_added = False
    stop_requested = False

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[monitor] received signal {signum}; stopping active jobs", flush=True)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while queue or running:
        if stop_requested or time.time() >= deadline:
            reason = "supervisor interrupted" if stop_requested else "wall-clock deadline reached"
            for active in list(running.values()):
                stop_process(active, reason)
            queue.clear()

        while not stop_requested and time.time() < deadline and queue and len(running) < args.max_concurrent:
            job = queue.pop(0)
            stdout_path = state_root / f"{job.name}.stdout.log"
            stdout = stdout_path.open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["VIRTUAL_ENV"] = str(VIRTUAL_ENV)
            environment["PYTHONUNBUFFERED"] = "1"
            environment["PYTHONPATH"] = (
                str(ROOT / "source" / "CBRIIsaacLab")
                + os.pathsep
                + environment.get("PYTHONPATH", "")
            )
            process = subprocess.Popen(
                command_for(job),
                cwd=ROOT,
                env=environment,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            stdout.close()
            running[job.name] = Running(job=job, process=process, stdout_path=stdout_path, started_at=time.time())
            print(f"[monitor] launched {job.name} pid={process.pid}", flush=True)

        for name, active in list(running.items()):
            if active.run_dir is None:
                active.run_dir = discover_run_dir(active.job, active.started_at)
            active.last_checkpoint, active.last_checkpoint_mtime = checkpoint_state(active.run_dir)
            metrics = read_metrics(active.run_dir)
            now = time.time()
            if now - active.last_status_at >= STATUS_INTERVAL:
                print(format_status(active, metrics), flush=True)
                active.last_status_at = now
            if active.last_checkpoint == 0 and now - active.started_at > NO_CHECKPOINT_TIMEOUT:
                stop_process(active, "no checkpoint after 45 minutes")

            returncode = active.process.poll()
            if returncode is not None:
                status = "completed" if returncode == 0 else "failed"
                finished[name] = record_for(active.job, active, returncode, status)
                print(
                    f"[monitor] finished {name} returncode={returncode} "
                    f"checkpoint={active.last_checkpoint}",
                    flush=True,
                )
                del running[name]

        if (
            not stop_requested
            and time.time() < deadline
            and phase == "initial"
            and not queue
            and not running
            and not followups_added
        ):
            continuation_jobs: list[Job] = []
            selected: dict[str, str | None] = {}
            for anchor in ("baseline", "task-balanced"):
                winner = choose_winner(finished, anchor)
                selected[anchor] = winner["job"]["name"] if winner else None
                if winner and winner.get("status") == "completed":
                    continuation = make_continuation(winner, args.continuation_timesteps)
                    if continuation is not None:
                        continuation_jobs.append(continuation)
            queue.extend(continuation_jobs)
            followups_added = True
            phase = "continue"
            print(f"[monitor] initial cohort complete; selected={selected}", flush=True)

        elif (
            not stop_requested
            and time.time() < deadline
            and phase == "continue"
            and not queue
            and not running
            and not strengthen_added
        ):
            strengthen_jobs: list[Job] = []
            for anchor in ("baseline", "task-balanced"):
                candidates = [
                    record
                    for record in finished.values()
                    if record["job"]["phase"] == "continue"
                    and record["job"]["anchor"] == anchor
                    and record.get("status") == "completed"
                ]
                if candidates:
                    strengthen = make_strengthen(candidates[0], args.strengthen_timesteps)
                    if strengthen is not None:
                        strengthen_jobs.append(strengthen)
            queue.extend(strengthen_jobs)
            strengthen_added = True
            phase = "strengthen"
            print(f"[monitor] continuation phase complete; queued {len(strengthen_jobs)} strengthen jobs", flush=True)

        state = {
            "phase": phase,
            "started_at": timestamp,
            "deadline": datetime.fromtimestamp(deadline).astimezone().isoformat(timespec="seconds"),
            "queue": [job.as_dict() for job in queue],
            "running": {
                name: {
                    "job": active.job.as_dict(),
                    "pid": active.process.pid,
                    "run_dir": str(active.run_dir) if active.run_dir else None,
                    "stdout": str(active.stdout_path),
                    "last_checkpoint": active.last_checkpoint,
                }
                for name, active in running.items()
            },
            "finished": finished,
        }
        write_json(status_path, state)

        if queue or running:
            time.sleep(CHECK_INTERVAL)

    print(f"[monitor] cohort complete; status={status_path}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without starting Isaac Sim.")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--wall-hours", type=float, default=DEFAULT_WALL_HOURS)
    parser.add_argument("--initial-timesteps", type=int, default=DEFAULT_INITIAL_TIMESTEPS)
    parser.add_argument("--continuation-timesteps", type=int, default=DEFAULT_CONTINUATION_TIMESTEPS)
    parser.add_argument("--strengthen-timesteps", type=int, default=DEFAULT_STRENGTHEN_TIMESTEPS)
    parser.add_argument("--allow-dirty", action="store_true", help="Allow a dirty worktree for an intentional run.")
    args = parser.parse_args()

    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be positive")
    if args.wall_hours <= 0.0:
        parser.error("--wall-hours must be positive")
    for name in ("initial_timesteps", "continuation_timesteps", "strengthen_timesteps"):
        if getattr(args, name) <= 0 or getattr(args, name) % ROLLOUTS:
            parser.error(f"--{name.replace('_', '-')} must be a positive multiple of {ROLLOUTS}")

    if args.dry_run:
        jobs = initial_jobs(args.initial_timesteps)
        print(f"root={ROOT}")
        print(f"max_concurrent={args.max_concurrent} wall_hours={args.wall_hours}")
        print("initial cohort:")
        for job in jobs:
            print(f"  {job.name}: checkpoint={job.checkpoint} timesteps={job.timesteps}")
            print(f"    {command_text(job)}")
        print("follow-ups:")
        print("  after each anchor's initial cohort: continue the selected regularized winner for the configured timesteps")
        print("  after both continuations: double each non-default regularization coefficient for a final dose-response stage")
        return 0

    return run_supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())
