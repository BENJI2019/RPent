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

"""Inspect prerequisites without starting optimization or a simulator episode."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from robots.robocasa.retention.openpi_adapter import norm_file
from robots.robocasa.retention.prepare import dataset_issues, target_paths
from robots.robocasa.retention.protocol import Experiment, load_catalog
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("retention.preflight")


def inspect(
    experiment: Experiment, config_path: Path, *, stage: str, runtime: bool
) -> dict:
    """Collect all static failures and optional isolated runtime-probe results."""
    checks = []

    def record(name, check):
        try:
            detail = check()
            checks.append({"name": name, "ok": True, "detail": str(detail)})
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            checks.append({"name": name, "ok": False, "detail": str(exc)})

    def checkpoint():
        params = Path(experiment.base_checkpoint) / "params"
        if not params.is_dir() or not any(params.iterdir()):
            raise ValueError(f"missing or empty JAX params directory: {params}")
        stats = norm_file(experiment.base_checkpoint)
        json.loads(stats.read_text(encoding="utf-8"))
        return f"JAX params and normalization asset found: {stats}"

    def checkout():
        revision = subprocess.check_output(
            ["git", "-C", experiment.openpi_root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=30,
        ).strip()
        if revision != load_catalog()["openpi_revision"]:
            raise ValueError(f"OpenPI revision mismatch: {revision}")
        return revision

    def executable(value):
        path = shutil.which(value)
        if path is None:
            raise ValueError(f"Python executable unavailable: {value}")
        return path

    record("base_checkpoint", checkpoint)
    record("openpi_revision", checkout)
    record("openpi_python", lambda: executable(experiment.openpi_python))
    if stage in {"evaluate", "all"}:
        record("simulator_python", lambda: executable(experiment.simulator_python))
    if stage in {"train", "all"}:
        for task, path in target_paths(experiment).items():

            def dataset(path=path):
                if issues := dataset_issues(path):
                    raise ValueError(f"{path}: {'; '.join(issues)}")
                return str(path)

            record(f"target_data/{task}", dataset)
    if runtime:
        probes = []
        if stage in {"train", "all"}:
            probes.append(("openpi-train", experiment.openpi_python))
        if stage in {"evaluate", "all"}:
            probes.extend(
                (
                    ("openpi-evaluate", experiment.openpi_python),
                    ("simulator", experiment.simulator_python),
                )
            )
        for kind, python in probes:

            def probe(kind=kind, python=python):
                env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
                if kind == "openpi-evaluate":
                    env["CUDA_VISIBLE_DEVICES"] = str(experiment.vla_cuda_device)
                elif kind == "simulator":
                    env.pop("CUDA_VISIBLE_DEVICES", None)
                result = subprocess.run(
                    [
                        python,
                        "-m",
                        "robots.robocasa.retention.preflight",
                        "--config",
                        str(config_path),
                        "--probe",
                        kind,
                    ],
                    cwd=get_repo_root(),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                detail = (result.stdout + result.stderr)[-6000:]
                if result.returncode:
                    raise RuntimeError(detail or f"probe exited {result.returncode}")
                return detail

            record(f"runtime/{kind}", probe)
    return {
        "ok": all(check["ok"] for check in checks),
        "stage": stage,
        "runtime_checked": runtime,
        "checks": checks,
        "scope": "prerequisite checks only; no checkpoint restore, optimization, video decoding, simulator reset, or planner calls",
    }


def probe_runtime(experiment: Experiment, kind: str) -> dict:
    """Check actual interpreter imports and GPU visibility without allocating a model."""
    if sys.platform != "linux":
        raise RuntimeError("GPU execution requires the documented Linux environment")
    if kind.startswith("openpi"):
        import jax

        from robots.robocasa.retention.openpi_adapter import (
            policy_config,
            verify_openpi_checkout,
        )

        verify_openpi_checkout(experiment.openpi_root)
        policy_config(experiment.base_checkpoint)
        devices = jax.devices()
        if not devices or any(d.platform != "gpu" for d in devices):
            raise RuntimeError(f"JAX CUDA devices unavailable: {devices}")
        if kind == "openpi-train" and (
            len(devices) % experiment.fsdp_devices
            or experiment.batch_size % len(devices)
        ):
            raise ValueError(
                "visible GPU count must be divisible by fsdp_devices; batch_size must divide evenly across visible GPUs"
            )
        return {"devices": [str(d) for d in devices], "jax": jax.__version__}
    import robocasa
    import robosuite
    import torch

    if (
        not torch.cuda.is_available()
        or experiment.env_cuda_device >= torch.cuda.device_count()
    ):
        raise RuntimeError(f"simulator GPU {experiment.env_cuda_device} is unavailable")
    asset_path = os.environ.get("ROBOCASA_ASSETS_PATH")
    if (
        not asset_path
        or not Path(asset_path).is_dir()
        or not any(Path(asset_path).iterdir())
    ):
        raise ValueError(
            "set ROBOCASA_ASSETS_PATH to the downloaded external asset directory"
        )
    return {
        "gpu": torch.cuda.get_device_name(experiment.env_cuda_device),
        "torch": torch.__version__,
        "mujoco": importlib.metadata.version("mujoco"),
        "robocasa": robocasa.__file__,
        "robosuite": robosuite.__file__,
        "asset_directory": asset_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--probe",
        choices=["openpi-train", "openpi-evaluate", "simulator"],
        required=True,
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    init_output_dir(config_path.parent / f"{config_path.stem}-setup-logs" / args.probe)
    logger.info(
        "%s", json.dumps(probe_runtime(Experiment.load(args.config), args.probe))
    )


if __name__ == "__main__":
    main()
