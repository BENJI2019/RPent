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

"""Run sharded evaluation and its optional Qwen service inside one Linux job."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import socket
import subprocess
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit

import httpx

from robots.robocasa.retention.protocol import Experiment
from robots.robocasa.retention.recovery import execution_lock
from rpent.evaluation import write_json_atomic
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("retention.cluster_evaluate")


def require_free_planner_port(url: str) -> None:
    """Reject existing listeners instead of accidentally using another service."""
    address = urlsplit(url)
    if (
        address.scheme != "http"
        or address.hostname not in {"localhost", "127.0.0.1"}
        or not address.port
    ):
        raise ValueError(
            "batch evaluation starts a local http://127.0.0.1:PORT/v1 planner"
        )
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", address.port))
        except OSError as exc:
            raise RuntimeError(
                f"planner port {address.port} is unavailable; stop the existing service before automatic evaluation"
            ) from exc


@contextmanager
def process_group(
    command: list[str], log: Path, env: dict
) -> Iterator[subprocess.Popen]:
    """Own a Linux process group, including subprocesses started by the service."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            yield process
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def wait_for_planner(
    process: subprocess.Popen, url: str, model: str, *, timeout: float = 900
) -> None:
    """Require the owned server to advertise the requested model before dispatch."""
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=5) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Qwen service exited; inspect its batch planner log")
            try:
                response = client.get(url.rstrip("/") + "/models")
                response.raise_for_status()
                if model in {item["id"] for item in response.json().get("data", [])}:
                    return
            except (httpx.HTTPError, ValueError, KeyError):
                pass
            time.sleep(1)
    raise TimeoutError("Qwen service did not become ready")


def worker_configs(experiment: Experiment, mode: str, gpus: list[int]) -> list[Path]:
    """Assign physical GPUs while preserving the declared evaluation denominator."""
    paths = []
    for gpu in gpus:
        worker = dataclasses.replace(
            experiment, env_cuda_device=gpu, vla_cuda_device=gpu
        )
        if worker.protocol_id != experiment.protocol_id:
            raise ValueError("GPU placement unexpectedly changed the protocol")
        path = Path(experiment.output_root) / "worker_configs" / f"{mode}-gpu{gpu}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, dataclasses.asdict(worker))
        paths.append(path)
    return paths


def run(
    experiment: Experiment, config: Path, mode: str, gpus: list[int], checkpoint: str
) -> None:
    """Keep planners and environment workers on the same platform job and host."""
    if (
        mode not in experiment.modes
        or not gpus
        or len(set(gpus)) != len(gpus)
        or min(gpus) < 0
    ):
        raise ValueError("select a configured mode and unique physical GPU ordinals")
    root = Path(experiment.output_root)
    with (
        execution_lock(root / ".execution.lock", shared=True),
        execution_lock(root / ".batch-evaluate.lock"),
        ExitStack() as stack,
    ):
        plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
        if (
            plan["protocol_id"] != experiment.protocol_id
            or (root / ".refresh-plan.json").exists()
        ):
            raise ValueError("run plan/refresh-plan before starting evaluation")
        logs = root / "batch_logs" / mode
        env = dict(os.environ, OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY", "local"))
        service = None
        workers = gpus
        if mode != "direct":
            require_free_planner_port(experiment.planners[mode]["base_url"])
            # A local pilot shares one GPU. On MTP reserve the last for Qwen.
            workers = gpus[:-1] or gpus
            script = Path(__file__).parent / "cluster/python.sh"
            service_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpus[-1]))
            service = stack.enter_context(
                process_group(
                    [
                        "bash",
                        str(script),
                        "qwen",
                        os.environ["RETENTION_QWEN_PREFIX"],
                        "-m",
                        "robots.robocasa.retention.serve_planner",
                        "--config",
                        str(config),
                        "--mode",
                        mode,
                    ],
                    logs / "planner.log",
                    service_env,
                )
            )
            wait_for_planner(
                service,
                experiment.planners[mode]["base_url"],
                experiment.planners[mode]["model"],
            )
        processes = []
        for index, path in enumerate(worker_configs(experiment, mode, workers)):
            processes.append(
                stack.enter_context(
                    process_group(
                        [
                            experiment.simulator_python,
                            "-m",
                            "robots.robocasa.retention",
                            "--config",
                            str(path),
                            "evaluate",
                            "--checkpoint",
                            checkpoint,
                            "--mode",
                            mode,
                            "--num-shards",
                            str(len(workers)),
                            "--shard-index",
                            str(index),
                        ],
                        logs / f"worker-{index}.log",
                        env,
                    )
                )
            )
        while any(p.poll() is None for p in processes):
            if service is not None and service.poll() is not None:
                raise RuntimeError("planner stopped during evaluation")
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(
                    "an evaluation worker failed; inspect batch_logs and rerun after repair"
                )
            time.sleep(1)
        if any(p.returncode for p in processes):
            raise RuntimeError(
                "an evaluation worker failed; completed cells are preserved"
            )
        logger.info("completed %s evaluation with %s worker(s)", mode, len(workers))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--checkpoint", default="all")
    args = parser.parse_args()
    experiment = Experiment.load(args.config)
    init_output_dir(Path(experiment.output_root) / "driver")

    # Convert platform cancellation into normal cleanup of all owned groups.
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    run(
        experiment,
        args.config,
        args.mode,
        [int(g) for g in args.gpus.split(",")],
        args.checkpoint,
    )


if __name__ == "__main__":
    main()
