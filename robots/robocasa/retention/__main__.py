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

"""Prepare, train, evaluate, and summarize the RoboCasa retention experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from robots.robocasa.retention.prepare import PRESETS, initialize, target_paths
from robots.robocasa.retention.protocol import Experiment
from rpent.evaluation import write_json_atomic
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("retention")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser(
        "init", help="create a stage configuration with absolute local paths"
    )
    setup.add_argument("--preset", required=True, choices=list(PRESETS))
    setup.add_argument("--workspace", required=True, type=Path)
    setup.add_argument("--with-planners", action="store_true")
    setup.add_argument("--fsdp-devices", type=int, default=1)
    setup.add_argument(
        "--output-root",
        type=Path,
        help="use a fresh result directory when repeating a preset",
    )
    download = commands.add_parser(
        "download-data", help="list selected Target data; download with --execute"
    )
    download.add_argument("--execute", action="store_true")
    doctor = commands.add_parser(
        "doctor", help="check local prerequisites without starting training"
    )
    doctor.add_argument("--stage", choices=["train", "evaluate", "all"], default="all")
    doctor.add_argument(
        "--runtime",
        action="store_true",
        help="also check configured Python imports and CUDA devices",
    )
    commands.add_parser(
        "plan", help="write the immutable protocol and declared grid counts"
    )
    training = commands.add_parser(
        "train", help="run configured independent or joint OpenPI SFT jobs"
    )
    select = training.add_mutually_exclusive_group(required=True)
    select.add_argument("--run-id")
    select.add_argument("--all", action="store_true")
    training.add_argument("--resume", action="store_true")
    evaluation = commands.add_parser("evaluate", help="run missing paired cells")
    evaluation.add_argument(
        "--checkpoint", required=True, help="base, a run id from plan, or all"
    )
    evaluation.add_argument("--mode", required=True, help="a configured mode, or all")
    evaluation.add_argument("--shard-index", type=int, default=0)
    evaluation.add_argument("--num-shards", type=int, default=1)
    evaluation.add_argument(
        "--tasks",
        nargs="+",
        help="execute a subset without changing the declared evaluation scope",
    )
    evaluation.add_argument("--max-cells", type=int)
    commands.add_parser(
        "summarize", help="write coverage, paired changes, and task matrices"
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    if args.command in {"init", "download-data", "doctor"}:
        init_output_dir(config_path.parent / f"{config_path.stem}-setup-logs")
    if args.command == "init":
        experiment = initialize(
            config_path,
            preset=args.preset,
            workspace=args.workspace,
            with_planners=args.with_planners,
            fsdp_devices=args.fsdp_devices,
            output_root=args.output_root,
        )
        logger.info("config: %s; output: %s", config_path, experiment.output_root)
        return
    experiment = Experiment.load(config_path)
    if args.command == "download-data":
        for task, path in target_paths(experiment).items():
            logger.info("Target Human %s -> %s", task, path)
        if args.execute:
            subprocess.run(
                [
                    experiment.simulator_python,
                    "-m",
                    "robots.robocasa.retention.prepare",
                    "--config",
                    str(config_path),
                ],
                cwd=get_repo_root(),
                check=True,
            )
        else:
            logger.info("preview only; add --execute to download these datasets")
        return
    if args.command == "doctor":
        from robots.robocasa.retention.preflight import inspect

        report = inspect(
            experiment, config_path, stage=args.stage, runtime=args.runtime
        )
        output = config_path.with_suffix(".doctor.json")
        write_json_atomic(output, report)
        for check in report["checks"]:
            logger.info(
                "%s %s: %s",
                "PASS" if check["ok"] else "FAIL",
                check["name"],
                check["detail"],
            )
        logger.info("%s; report: %s", report["scope"], output)
        if not report["ok"]:
            raise SystemExit(1)
        return
    root = Path(experiment.output_root)
    root.mkdir(parents=True, exist_ok=True)
    init_output_dir(root / "driver")
    plan_path = root / "plan.json"
    if plan_path.exists():
        stored = json.loads(plan_path.read_text(encoding="utf-8"))
        if stored["protocol_id"] != experiment.protocol_id:
            parser.error(
                "output_root already belongs to another protocol; choose a fresh directory"
            )
    else:
        write_json_atomic(plan_path, experiment.plan())
    if args.command == "plan":
        plan = experiment.plan()
        logger.info(
            "%s adaptation checkpoints; %s total evaluation cells (including baseline)",
            len(plan["adaptations"]),
            plan["total_cells"],
        )
        logger.info("protocol: %s", plan_path)
    elif args.command == "train":
        runs = experiment.runs() if args.all else [experiment.run(args.run_id)]
        for run in runs:
            command = [
                experiment.openpi_python,
                "-m",
                "robots.robocasa.retention.train",
                "--config",
                str(config_path),
                "--run-id",
                run["id"],
            ]
            if args.resume:
                command.append("--resume")
            subprocess.run(command, cwd=get_repo_root(), check=True)
    elif args.command == "evaluate":
        from robots.robocasa.retention.evaluate import evaluate

        checkpoints = (
            ["base", *(run["id"] for run in experiment.runs())]
            if args.checkpoint == "all"
            else [args.checkpoint]
        )
        modes = experiment.modes if args.mode == "all" else [args.mode]
        for checkpoint in checkpoints:
            for mode in modes:
                result = evaluate(
                    experiment,
                    config_path,
                    checkpoint,
                    mode,
                    shard_index=args.shard_index,
                    num_shards=args.num_shards,
                    tasks=args.tasks,
                    max_cells=args.max_cells,
                )
                logger.info("evaluation %s %s: %s", checkpoint, mode, result)
                if result["infrastructure_errors"]:
                    raise SystemExit(1)
    else:
        from robots.robocasa.retention.summarize import summarize

        report = summarize(experiment)
        logger.info("summary: %s (complete=%s)", root / "summary", report["complete"])
        if not report["complete"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
