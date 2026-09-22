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

"""Resumable, sharded paired evaluation with one owned VLA server per worker."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Iterator

from robots.robocasa.retention.openpi_adapter import norm_digest
from robots.robocasa.retention.protocol import Experiment
from rpent.evaluation import write_json_atomic
from rpent.utils.config import get_repo_root
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.logging import get_logger
from rpent.utils.rpc import wait_for_ready
from rpent.utils.rpc.http_rpc import HttpRpcClient

logger = get_logger("retention.evaluate")


def resolve_checkpoint(experiment: Experiment, checkpoint_id: str) -> tuple[Path, str]:
    """Resolve base or a completed adaptation with matching provenance."""
    if checkpoint_id == "base":
        path, method = Path(experiment.base_checkpoint), "full"
    else:
        run = experiment.run(checkpoint_id)
        record_path = (
            Path(experiment.output_root)
            / "training"
            / checkpoint_id
            / "checkpoint.json"
        )
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["protocol_id"] != experiment.protocol_id:
            raise ValueError(
                "checkpoint was trained under a different experiment protocol"
            )
        if record["base_norm_sha256"] != norm_digest(experiment.base_checkpoint):
            raise ValueError("base checkpoint normalization has changed")
        if record["run"] != run or record["method"] != run["method"]:
            raise ValueError("checkpoint does not belong to the requested adaptation")
        path, method = Path(record["checkpoint"]), record["method"]
    if not (path / "params").is_dir():
        raise FileNotFoundError(f"OpenPI JAX checkpoint params are missing: {path}")
    norm_digest(path)
    return path.resolve(), method


def cell_identity(
    experiment: Experiment,
    checkpoint_id: str,
    mode: str,
    split: str,
    task: str,
    seed: int,
) -> dict:
    return {
        "protocol_id": experiment.protocol_id,
        "checkpoint_id": checkpoint_id,
        "mode": mode,
        "environment_split": split,
        "task": task,
        "seed": seed,
    }


def cell_dir(
    experiment: Experiment,
    checkpoint_id: str,
    mode: str,
    split: str,
    task: str,
    seed: int,
) -> Path:
    return (
        Path(experiment.output_root)
        / "evaluation"
        / checkpoint_id
        / mode
        / split
        / task
        / str(seed)
    )


def completed_record(path: Path, identity: dict) -> dict | None:
    """Only completed environment-authoritative records are resumable."""
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("identity") != identity:
        raise ValueError(f"result identity mismatch: {path}")
    if record.get("status") != "completed":
        return None
    if (
        type(record.get("success")) is not bool
        or record.get("success_source") != "env.check_success"
    ):
        raise ValueError(f"invalid success record: {path}")
    if not record.get("initial_state_sha256") or record.get("reset_count") != 1:
        raise ValueError(f"missing paired episode evidence: {path}")
    return record


def selected_cells(
    experiment: Experiment,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    tasks: list[str] | None = None,
) -> Iterator[tuple[str, str, int]]:
    """Shard deterministically without changing the declared evaluation denominator."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    allowed = {task for _, task, _ in experiment.grid()}
    if tasks and set(tasks) - allowed:
        raise ValueError(f"unknown evaluation tasks: {sorted(set(tasks) - allowed)}")
    for index, cell in enumerate(experiment.grid()):
        if index % num_shards == shard_index and (not tasks or cell[1] in tasks):
            yield cell


def evaluate(
    experiment: Experiment,
    config_path: Path,
    checkpoint_id: str,
    mode: str,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    tasks: list[str] | None = None,
    max_cells: int | None = None,
) -> dict:
    """Execute missing cells; infrastructure errors remain missing, never failures."""
    if mode not in experiment.modes:
        raise ValueError(f"mode is not part of this protocol: {mode}")
    if max_cells is not None and max_cells < 1:
        raise ValueError("max-cells must be positive")
    checkpoint, method = resolve_checkpoint(experiment, checkpoint_id)
    pending = []
    for split, task, seed in selected_cells(
        experiment, shard_index=shard_index, num_shards=num_shards, tasks=tasks
    ):
        identity = cell_identity(experiment, checkpoint_id, mode, split, task, seed)
        output = cell_dir(experiment, checkpoint_id, mode, split, task, seed)
        if completed_record(output / "retention_result.json", identity) is None:
            pending.append((split, task, seed, output, identity))
            if max_cells is not None and len(pending) >= max_cells:
                break
    if not pending:
        return {"executed": 0, "infrastructure_errors": 0}
    logs = (
        Path(experiment.output_root)
        / "services"
        / checkpoint_id
        / mode
        / f"shard_{shard_index}"
    )
    logs.mkdir(parents=True, exist_ok=True)
    port = pick_free_port()
    endpoint = f"http://127.0.0.1:{port}"
    server = ProcessDaemon(
        "pi05",
        [
            experiment.openpi_python,
            "-m",
            "robots.robocasa.pi05_server",
            "--model-path",
            str(checkpoint),
            "--method",
            method,
            "--port",
            str(port),
            "--openpi-root",
            experiment.openpi_root,
            "--cuda-device",
            str(experiment.vla_cuda_device),
            "--parent-watch",
        ],
        env_overrides={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        log_path=str(logs / "pi05.log"),
    )
    rpc = HttpRpcClient(endpoint, enable_sessions=True)
    errors = 0
    try:
        server.start()
        wait_for_ready(rpc, daemon=server, timeout_s=600)
        metadata = rpc.call("vla.get_modality_config")
        if (
            metadata.get("backend") != "pi05"
            or Path(metadata["checkpoint"]).resolve() != checkpoint
            or metadata.get("method") != method
            or metadata.get("normalization_sha256") != norm_digest(checkpoint)
        ):
            raise ValueError("VLA server did not load the requested checkpoint")
        if metadata["action_horizon"] < experiment.action_steps:
            raise ValueError("policy action horizon is shorter than action_steps")
        for split, task, seed, output, identity in pending:
            output.mkdir(parents=True, exist_ok=True)
            # One owner per cell. A crashed worker leaves the lock for inspection.
            lock = output / ".running"
            with lock.open("x", encoding="utf-8") as file:
                file.write(str(os.getpid()))
            try:
                cmd = [
                    experiment.simulator_python,
                    "-m",
                    "robots.robocasa.retention.cell",
                    "--config",
                    str(config_path.resolve()),
                    "--checkpoint-id",
                    checkpoint_id,
                    "--mode",
                    mode,
                    "--split",
                    split,
                    "--task",
                    task,
                    "--seed",
                    str(seed),
                    "--vla-endpoint",
                    endpoint,
                ]
                with (output / "worker.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        cmd,
                        cwd=get_repo_root(),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=experiment.cell_timeout_s,
                        check=False,
                    )
                record = completed_record(output / "retention_result.json", identity)
                if result.returncode or record is None:
                    errors += 1
                    if record is None:
                        write_json_atomic(
                            output / "retention_result.json",
                            {
                                "identity": identity,
                                "status": "infrastructure_error",
                                "success": None,
                                "error_type": "CellWorkerFailed",
                            },
                        )
            except subprocess.TimeoutExpired:
                errors += 1
                write_json_atomic(
                    output / "retention_result.json",
                    {
                        "identity": identity,
                        "status": "infrastructure_error",
                        "success": None,
                        "error_type": "CellWorkerTimeout",
                    },
                )
            finally:
                lock.unlink()
            logger.info("%s %s %s seed=%s", checkpoint_id, mode, task, seed)
    finally:
        rpc.close()
        server.stop()
    return {"executed": len(pending), "infrastructure_errors": errors}
