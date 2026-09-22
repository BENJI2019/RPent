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

"""Create stage configurations and download only their Target Human datasets."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from robots.robocasa.retention.protocol import (
    DEFAULT_PLANNERS,
    MODES,
    Experiment,
    load_catalog,
)
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("retention.prepare")
PRESETS = {
    "pilot1": "examples/pilot1.json",
    "pilot3-joint": "examples/pilot3-joint.json",
    "pilot3-independent": "examples/pilot3-independent.json",
    "joint50": "examples/joint50.json",
    "independent50": "example.json",
}
SHARED_MODEL_ROOTS = (Path("/home/ma-user/work/model"), Path("/opt/huawei/quoteModel"))
SHARED_DATASET_ROOTS = (Path("/home/ma-user/work/dataset"), Path("/opt/huawei/dataset"))


def initialize(
    destination: Path,
    *,
    preset: str,
    workspace: Path | None = None,
    profile: str = "generic",
    with_planners: bool = False,
    fsdp_devices: int = 1,
    output_root: Path | None = None,
) -> Experiment:
    """Write a new configuration with local absolute paths; never replace an existing one."""
    if destination.exists():
        raise FileExistsError(destination)
    if profile == "huawei":
        resources = cluster_resources(destination, preset, output_root)
        workspace = Path(os.environ["RETENTION_HOME"])
    elif profile == "generic" and workspace is not None:
        resources = {}
        workspace = workspace.expanduser().resolve()
    else:
        raise ValueError(
            "generic init requires --workspace; use the launchers for --profile huawei"
        )
    template = Path(__file__).parent / PRESETS[preset]
    config = json.loads(template.read_text(encoding="utf-8"))
    openpi = workspace / "openpi"
    config.update(
        base_checkpoint=str(
            workspace
            / "checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000"
        ),
        openpi_root=str(openpi),
        datasets_root=str(workspace / "datasets"),
        output_root=str(
            output_root.expanduser().resolve()
            if output_root
            else workspace / "runs" / preset
        ),
        openpi_python=str(openpi / ".venv/bin/python"),
        simulator_python=sys.executable,
        fsdp_devices=fsdp_devices,
        env_cuda_device=0,
        vla_cuda_device=0,
        planners={
            name: {
                **values,
                "max_model_len": 32768,
                "max_num_seqs": 1,
                "gpu_memory_utilization": 0.4,
            }
            for name, values in DEFAULT_PLANNERS.items()
        },
    )
    if with_planners:
        config["modes"] = list(MODES)
    config.update(resources)
    experiment = Experiment(**config)
    experiment.validate()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
        file.write("\n")
    return experiment


def cluster_resources(
    destination: Path, preset: str, output_root: Path | None
) -> dict[str, str]:
    """Bind a Huawei configuration to this package and shared Conda interpreters."""
    required = (
        "RETENTION_HOME",
        "RETENTION_CODE_ROOT",
        "RETENTION_DATASETS",
        "RETENTION_CHECKPOINT",
        "RETENTION_SIM_PREFIX",
        "RETENTION_OPENPI_PREFIX",
    )
    values = {}
    for name in required:
        value = os.environ.get(name, "")
        if not value or not Path(value).is_absolute():
            raise ValueError(
                f"{name} must be an absolute path; use run_local.sh or run_mtp.sh"
            )
        # Keep logical shared-volume paths: resolve() differs between local and MTP.
        values[name] = Path(value)
    platform = os.environ.get("RETENTION_EXECUTION")
    if sys.platform != "linux" or platform not in {"local", "mtp"}:
        raise ValueError("Huawei preparation requires a Linux local/MTP launcher")
    home = values["RETENTION_HOME"]
    shared_roots = (*SHARED_MODEL_ROOTS, *SHARED_DATASET_ROOTS)
    if not any(home.is_relative_to(root) and home != root for root in shared_roots):
        raise ValueError(
            "RETENTION_HOME must be a user directory on a shared dataset/model volume"
        )
    if not destination.resolve().is_relative_to(home.resolve()):
        raise ValueError(
            "Huawei configuration and setup reports must be under RETENTION_HOME"
        )
    output = output_root or home / "retention_outputs" / platform / preset
    if not output.resolve().is_relative_to(home.resolve()):
        raise ValueError("Huawei output_root must be under RETENTION_HOME")
    root = values["RETENTION_CODE_ROOT"]
    if root.resolve() != Path(__file__).resolve().parents[3]:
        raise ValueError("RPent was imported from outside the submitted package")
    wrappers = home / "runtime" / platform / destination.stem
    result = {}
    for role, variable, key in (
        ("simulator", "RETENTION_SIM_PREFIX", "simulator_python"),
        ("openpi", "RETENTION_OPENPI_PREFIX", "openpi_python"),
    ):
        prefix = values[variable]
        allowed = shared_roots
        if not any(prefix.is_relative_to(p) for p in allowed):
            raise ValueError(
                f"{variable} must refer to an environment on a shared volume"
            )
        script = wrappers / f"{role}-python"
        content = (
            "#!/bin/bash\nset -euo pipefail\n"
            'exec bash "${RETENTION_CODE_ROOT:?Use a retention launcher}/robots/robocasa/retention/cluster/python.sh" '
            f'{role} {shlex.quote(str(prefix))} "$@"\n'
        )
        if script.exists() and script.read_text(encoding="utf-8") != content:
            raise ValueError(
                f"interpreter environment changed: {script}; use a new experiment name"
            )
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(content, encoding="utf-8")
        script.chmod(0o755)
        result[key] = str(script)
    return {
        **result,
        "base_checkpoint": str(values["RETENTION_CHECKPOINT"]),
        "datasets_root": str(values["RETENTION_DATASETS"]),
        "openpi_root": str(root / "vendor/openpi"),
        "output_root": str(output),
    }


def target_paths(experiment: Experiment) -> dict[str, Path]:
    """Resolve only configured adaptation tasks under the official Target data root."""
    root = Path(experiment.datasets_root).resolve()
    catalog = load_catalog()
    tasks = sorted({task for run in experiment.runs() for task in run["tasks"]})
    paths = {}
    for task in tasks:
        path = (root / catalog["tasks"][task]["target_dataset"]).resolve()
        if not path.is_relative_to(root) or "/target/" not in path.as_posix():
            raise ValueError(f"Target dataset escapes configured root: {path}")
        paths[task] = path
    return paths


def dataset_issues(path: Path) -> list[str]:
    """Check metadata and per-episode file counts without decoding trajectories."""
    issues = []
    episodes = None
    for name in (
        "info.json",
        "modality.json",
        "tasks.jsonl",
        "episodes.jsonl",
        "stats.json",
    ):
        file = path / "meta" / name
        if not file.is_file() or not file.stat().st_size:
            issues.append(f"missing or empty meta/{name}")
        elif file.suffix == ".json":
            try:
                metadata = json.loads(file.read_text(encoding="utf-8"))
                if name == "info.json":
                    episodes = (
                        metadata.get("total_episodes")
                        if isinstance(metadata, dict)
                        else None
                    )
                    if type(episodes) is not int or episodes < 1:
                        issues.append(
                            "meta/info.json must declare a positive total_episodes"
                        )
                        episodes = None
            except (ValueError, OSError) as exc:
                issues.append(f"invalid meta/{name}: {exc}")
    for pattern in (
        "data/*/*.parquet",
        *(
            f"videos/*/observation.images.{camera}/*.mp4"
            for camera in (
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            )
        ),
    ):
        count = sum(p.is_file() and p.stat().st_size > 0 for p in path.glob(pattern))
        if episodes is not None and count != episodes:
            issues.append(
                f"expected {episodes} nonempty files matching {pattern}, found {count}"
            )
        elif not count:
            issues.append(f"no nonempty files matching {pattern}")
    return issues


def download_selected(experiment: Experiment) -> None:
    """Call the installed official downloader with a process-local dataset root."""
    import robocasa.macros as macros

    # get_ds_meta reads this module object; the downloader also keeps an imported copy.
    macros.DATASET_BASE_PATH = str(Path(experiment.datasets_root).resolve())
    from robocasa.scripts import download_datasets as upstream

    upstream.DATASET_BASE_PATH = macros.DATASET_BASE_PATH
    paths = target_paths(experiment)
    for task, path in paths.items():
        metadata = upstream.get_ds_meta(task=task, split="target", source="human")
        if metadata is None or Path(metadata["path"]).resolve() != path:
            raise ValueError(
                f"installed RoboCasa registry disagrees with pinned Target path: {task}"
            )
        if path.exists() and (issues := dataset_issues(path)):
            raise ValueError(
                f"incomplete dataset at {path}; inspect or move it aside before retrying: {issues}"
            )
    upstream.download_datasets(
        split=["target"],
        tasks=list(paths),
        source=["human"],
        all_data=False,
        overwrite=False,
        dryrun=False,
    )
    # Upstream may print a network failure and return normally. Do not report success then.
    failures = {
        task: issues for task, path in paths.items() if (issues := dataset_issues(path))
    }
    if failures:
        raise RuntimeError(f"dataset download is incomplete: {failures}")
    logger.info("Target Human data ready for %s task(s)", len(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    init_output_dir(config_path.parent / f"{config_path.stem}-setup-logs" / "download")
    download_selected(Experiment.load(args.config))


if __name__ == "__main__":
    main()
