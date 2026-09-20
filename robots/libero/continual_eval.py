# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a frozen-policy, four-suite LIBERO memory-learning experiment.

Use from a source checkout: ``python -m robots.libero.continual_eval --help``.
The script never counts planner prose or a process exit code as task success.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = "openai-chat:gemini-3.5-flash-lite"
PUBLISHED = ("MEMORY.md", "global", "suite", "task_only")


def parse_seeds(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of seeds and inclusive ranges."""
    seeds: set[int] = set()
    try:
        for part in value.split(","):
            bounds = [int(item) for item in part.split("-")]
            if len(bounds) == 1:
                seeds.add(bounds[0])
            elif len(bounds) == 2 and bounds[0] <= bounds[1]:
                seeds.update(range(bounds[0], bounds[1] + 1))
            else:
                raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use seeds such as 1-10 or 1,3,5") from exc
    if not seeds or min(seeds) < 1:
        raise argparse.ArgumentTypeError(
            "evaluation seeds must be positive; seed 0 is for exploration"
        )
    return tuple(sorted(seeds))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, required=True, help="new experiment directory"
    )
    parser.add_argument(
        "--resume", action="store_true", help="continue a recorded experiment"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print schedule without creating files"
    )
    parser.add_argument("--order", nargs=4, choices=SUITES, default=SUITES)
    parser.add_argument(
        "--tasks-per-suite", type=int, default=10, help="first N tasks, 1-10"
    )
    parser.add_argument(
        "--libero-type", choices=("standard", "pro", "plus"), default="standard"
    )
    parser.add_argument("--eval-seeds", type=parse_seeds, default=tuple(range(1, 11)))
    parser.add_argument(
        "--model", default=MODEL, help="OpenAI-compatible vision/tool model"
    )
    parser.add_argument("--base-url", default=GEMINI_BASE_URL)
    parser.add_argument(
        "--api-key-env",
        default="GEMINI_API_KEY",
        help="environment variable holding the key",
    )
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--planner-timeout-s", type=int, default=1200)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--explore-sessions", type=int, default=3)
    parser.add_argument("--explore-attempts-per-session", type=int, default=5)
    parser.add_argument("--cuda-device", type=int)
    parser.add_argument("--vla-endpoint", help="optional prestarted Pi0.5 service")
    parser.add_argument("--sam3-endpoint", help="optional prestarted SAM3 service")
    parser.add_argument(
        "--skip-api-check",
        action="store_true",
        help="skip the text-only connectivity probe",
    )
    return parser


def schedule(order: tuple[str, ...], tasks: int, seeds: tuple[int, ...]):
    """Yield baseline and four stage jobs in the committed order."""
    for stage in range(5):
        if stage:
            for task in range(tasks):
                yield stage, "explore", order[stage - 1], task, 0
        for suite in order:
            for task in range(tasks):
                for seed in seeds:
                    yield stage, "eval", suite, task, seed


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _digest(root: Path) -> str:
    """Hash names and contents, including unpublished inbox drafts when present."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _snapshot(live: Path, target: Path) -> str:
    if target.exists():
        raise RuntimeError(
            f"unrecorded snapshot exists: {target}; inspect before resuming"
        )
    target.mkdir(parents=True)
    for name in PUBLISHED:
        source = live / name
        destination = target / name
        if source.is_dir():
            shutil.copytree(source, destination)
        elif source.is_file():
            shutil.copy2(source, destination)
    return _digest(target)


def _success(output_dir: Path, *, explore: bool) -> bool:
    manifests = (
        sorted((output_dir / "sessions").glob("session_*/states.json"))
        if explore
        else [output_dir / "states.json"]
    )
    if not manifests or not all(path.is_file() for path in manifests):
        raise ValueError(f"missing environment state manifest in {output_dir}")
    records = [
        step for path in manifests for step in json.loads(path.read_text())["steps"]
    ]
    if not records:
        raise ValueError(f"empty environment state manifest in {output_dir}")
    return any(step["terminated"] for step in records)


def _report(root: Path, manifest: dict[str, Any]) -> None:
    rows = []
    for job in manifest["jobs"].values():
        if job["status"] != "complete" or job["phase"] != "eval":
            continue
        rows.append(
            {
                key: job[key]
                for key in ("stage", "suite", "task", "seed", "success", "output_dir")
            }
        )
    rows.sort(key=lambda row: (row["stage"], row["suite"], row["task"], row["seed"]))
    with (root / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("stage", "suite", "task", "seed", "success", "output_dir"),
        )
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[tuple[int, str], list[int]] = {}
    for row in rows:
        counter = counts.setdefault((row["stage"], row["suite"]), [0, 0])
        counter[0] += int(row["success"])
        counter[1] += 1
    expected = manifest["config"]["tasks_per_suite"] * len(
        manifest["config"]["eval_seeds"]
    )
    with (root / "matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "stage",
                "learned_through",
                "suite",
                "successes",
                "completed",
                "expected",
                "success_rate_if_complete",
            )
        )
        for stage in range(5):
            for suite in manifest["config"]["order"]:
                successes, completed = counts.get((stage, suite), [0, 0])
                writer.writerow(
                    (
                        stage,
                        "baseline"
                        if stage == 0
                        else manifest["config"]["order"][stage - 1],
                        suite,
                        successes,
                        completed,
                        expected,
                        successes / expected if completed == expected else "",
                    )
                )
    with (root / "retention.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "suite",
                "baseline",
                "pre_learning",
                "after_learning",
                "final",
                "final_minus_baseline",
                "forgetting_after_learning",
            )
        )
        for index, suite in enumerate(manifest["config"]["order"]):
            rates = []
            for stage in (0, index, index + 1, 4):
                successes, completed = counts.get((stage, suite), [0, 0])
                rates.append(successes / expected if completed == expected else None)
            baseline, pre_learning, after_learning, final = rates
            writer.writerow(
                (
                    suite,
                    baseline,
                    pre_learning,
                    after_learning,
                    final,
                    final - baseline
                    if final is not None and baseline is not None
                    else "",
                    after_learning - final
                    if final is not None and after_learning is not None
                    else "",
                )
            )


def _command(
    args: argparse.Namespace, root: Path, job: dict[str, Any], output: Path
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "rpent.cli.main",
        "--robot",
        "libero",
        "--libero-type",
        args.libero_type,
        "--suite",
        job["suite"],
        "--task",
        str(job["task"]),
        "--seed",
        str(job["seed"]),
        "--planner",
        "api",
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--memory-dir",
        str(
            root / "memory_live"
            if job["phase"] == "explore"
            else root / "snapshots" / f"stage_{job['stage']:02d}"
        ),
        "--memory-profile",
        "local",
        "--memory-ablation",
        "--output-dir",
        str(output),
        "--max-turns",
        str(args.max_turns),
        "--max-tokens",
        str(args.max_tokens),
        "--planner-timeout-s",
        str(args.planner_timeout_s),
        "--max-episode-steps",
        str(args.max_episode_steps),
    ]
    if job["phase"] == "explore":
        command.extend(
            (
                "--explore",
                "--explore-sessions",
                str(args.explore_sessions),
                "--explore-attempts-per-session",
                str(args.explore_attempts_per_session),
            )
        )
    if args.cuda_device is not None:
        command.extend(("--cuda-device", str(args.cuda_device)))
    for name in ("vla", "sam3"):
        endpoint = getattr(args, f"{name}_endpoint")
        if endpoint:
            command.extend((f"--{name}-endpoint", endpoint))
    return command


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(set(args.order)) != 4 or not 1 <= args.tasks_per_suite <= 10:
        parser.error(
            "--order must contain each suite once and --tasks-per-suite must be 1-10"
        )
    if not args.model.startswith("openai-chat:"):
        parser.error("the OpenAI-compatible endpoint requires an openai-chat: model")
    if (
        min(
            args.max_turns,
            args.max_tokens,
            args.planner_timeout_s,
            args.max_episode_steps,
            args.explore_sessions,
        )
        <= 0
    ):
        parser.error(
            "turn, timeout, episode-step, and exploration-session limits must be positive"
        )
    if args.explore_attempts_per_session < 0:
        parser.error("--explore-attempts-per-session must be nonnegative")
    order = tuple(args.order)
    plan = list(schedule(order, args.tasks_per_suite, args.eval_seeds))
    if args.dry_run:
        print(
            f"{len(plan)} runs: {4 * args.tasks_per_suite} explorations, "
            f"{5 * 4 * args.tasks_per_suite * len(args.eval_seeds)} evaluations"
        )
        print("order:", " -> ".join(order))
        return 0
    key = os.environ.get(args.api_key_env)
    if not key:
        parser.error(
            f"set {args.api_key_env} before running; the key is not saved in results"
        )
    root = args.output_root.expanduser().resolve()
    config = {
        "order": list(order),
        "tasks_per_suite": args.tasks_per_suite,
        "libero_type": args.libero_type,
        "eval_seeds": list(args.eval_seeds),
        "model": args.model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
        "planner_timeout_s": args.planner_timeout_s,
        "max_episode_steps": args.max_episode_steps,
        "explore_sessions": args.explore_sessions,
        "explore_attempts_per_session": args.explore_attempts_per_session,
        "cuda_device": args.cuda_device,
        "vla_endpoint": args.vla_endpoint,
        "sam3_endpoint": args.sam3_endpoint,
    }
    manifest_path = root / "experiment.json"
    if args.resume:
        if not manifest_path.is_file():
            parser.error("--resume requires an existing experiment.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config"] != config:
            parser.error("resume configuration differs from the recorded experiment")
        if _digest(root / "memory_live") != manifest["memory_digest"]:
            raise RuntimeError(
                "live memory differs from the last completed stage; inspect before resuming"
            )
    else:
        root.mkdir(parents=True, exist_ok=False)
        live = root / "memory_live"
        live.mkdir()
        (live / "MEMORY.md").write_text(
            "# Layered memory index\n\nNo learned memory yet.\n", encoding="utf-8"
        )
        manifest = {
            "config": config,
            "jobs": {},
            "snapshots": {},
            "memory_digest": _digest(live),
        }
        _write_json(manifest_path, manifest)
    child_env = os.environ.copy()
    child_env["OPENAI_API_KEY"] = key
    if not args.skip_api_check:
        probe = [
            sys.executable,
            "-m",
            "rpent.cli.check_llm",
            "--planner",
            "api",
            "--model",
            args.model,
            "--base-url",
            args.base_url,
        ]
        if subprocess.run(probe, env=child_env).returncode:
            print("API preflight failed; no episodes were started", file=sys.stderr)
            return 1
    for stage, phase, suite, task, seed in plan:
        snapshot_key = str(stage)
        if phase == "eval" and snapshot_key not in manifest["snapshots"]:
            snapshot = root / "snapshots" / f"stage_{stage:02d}"
            fingerprint = _snapshot(root / "memory_live", snapshot)
            manifest["snapshots"][snapshot_key] = fingerprint
            _write_json(manifest_path, manifest)
        if phase == "eval":
            snapshot = root / "snapshots" / f"stage_{stage:02d}"
            if _digest(snapshot) != manifest["snapshots"][snapshot_key]:
                raise RuntimeError(f"stage {stage} evaluation memory snapshot changed")
        job_id = f"{stage}:{phase}:{suite}:{task}:{seed}"
        previous = manifest["jobs"].get(job_id)
        if previous and previous["status"] == "complete":
            continue
        if previous and previous["status"] == "running":
            raise RuntimeError(
                f"interrupted job {job_id}: inspect {previous['output_dir']} before resuming"
            )
        if previous and previous["status"] == "failed" and phase == "explore":
            if previous["memory_before"] != previous["memory_after"]:
                raise RuntimeError(
                    f"failed exploration {job_id} changed live memory; "
                    "inspect it before manually recovering"
                )
            if _digest(root / "memory_live") != previous["memory_after"]:
                raise RuntimeError(
                    f"live memory changed after failed exploration {job_id}"
                )
        attempt = 1 if previous is None else previous["attempt"] + 1
        output = (
            root
            / "runs"
            / f"stage_{stage:02d}"
            / phase
            / suite
            / f"task_{task:02d}"
            / f"seed_{seed:02d}"
            / f"attempt_{attempt:03d}"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise RuntimeError(f"refusing to reuse existing output directory: {output}")
        job = {
            "stage": stage,
            "phase": phase,
            "suite": suite,
            "task": task,
            "seed": seed,
            "status": "running",
            "attempt": attempt,
            "output_dir": str(output),
        }
        if phase == "explore":
            job["memory_before"] = _digest(root / "memory_live")
        manifest["jobs"][job_id] = job
        _write_json(manifest_path, manifest)
        print(
            f"[{len([item for item in manifest['jobs'].values() if item['status'] == 'complete']) + 1}/{len(plan)}] {job_id}",
            flush=True,
        )
        started = time.monotonic()
        result = subprocess.run(_command(args, root, job, output), env=child_env)
        job["elapsed_s"] = round(time.monotonic() - started, 2)
        job["exit_code"] = result.returncode
        try:
            job["success"] = (
                _success(output, explore=phase == "explore")
                if result.returncode == 0
                else False
            )
            job["status"] = "complete" if result.returncode == 0 else "failed"
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
        if phase == "explore":
            job["memory_after"] = _digest(root / "memory_live")
            if job["status"] == "complete":
                manifest["memory_digest"] = job["memory_after"]
        manifest["jobs"][job_id] = job
        _write_json(manifest_path, manifest)
        _report(root, manifest)
        if phase == "eval" and _digest(snapshot) != manifest["snapshots"][snapshot_key]:
            raise RuntimeError(f"stage {stage} evaluation modified its memory snapshot")
        if job["status"] != "complete":
            print(
                f"Stopped at {job_id}; inspect {output / 'run.log'}, then use --resume",
                file=sys.stderr,
            )
            return 1
    print(
        f"Complete: {root / 'matrix.csv'}, {root / 'retention.csv'}, {root / 'episodes.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
