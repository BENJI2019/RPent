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

"""Pinned task catalog, experiment identities, and complete evaluation grids."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Iterator

CATALOG_PATH = Path(__file__).with_name("catalog.json")
MODES = ("direct", "qwen3_vl_4b", "qwen35_4b")
DEFAULT_PLANNERS = {
    "qwen3_vl_4b": {
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "base_url": "http://127.0.0.1:8000/v1",
    },
    "qwen35_4b": {
        "model": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "base_url": "http://127.0.0.1:8001/v1",
    },
}


def fingerprint(value: Any) -> str:
    """Hash the effective protocol, including task catalog and resource settings."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def episode_seed(split: str, task: str, replicate: int) -> int:
    """Derive task-specific random streams, identical before and after adaptation."""
    return int(fingerprint([split, task, replicate])[:8], 16)


def load_catalog() -> dict:
    """Read the official task registry snapshot without loading simulator code."""
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    old = data["pretrain300"]
    groups = data["target_splits"]
    target = [task for group in groups.values() for task in group]
    if len(old) != 300 or len(set(old)) != 300:
        raise ValueError("catalog must contain exactly 300 unique pretraining tasks")
    if [
        len(groups[k]) for k in ("atomic_seen", "composite_seen", "composite_unseen")
    ] != [18, 16, 16]:
        raise ValueError("Target50 split counts must be 18/16/16")
    if len(set(target)) != 50 or len(set(old) & set(target)) != 34:
        raise ValueError(
            "Target50 must have 50 unique tasks, 34 overlapping Pretrain300"
        )
    return data


@dataclass(frozen=True)
class Experiment:
    """Resources and fixed budgets for independent or joint Target-task SFT."""

    base_checkpoint: str
    openpi_root: str
    datasets_root: str
    output_root: str
    checkpoint_revision: str
    openpi_python: str = "python"
    simulator_python: str = "python"
    methods: list[str] = field(default_factory=lambda: ["full"])
    train_seeds: list[int] = field(default_factory=lambda: [0])
    eval_seeds: list[int] = field(default_factory=lambda: list(range(1000, 1010)))
    adaptation_tasks: list[str] | None = None
    adaptation_mode: str = "independent"
    evaluation_tasks: dict[str, list[str]] | None = None
    modes: list[str] = field(default_factory=lambda: list(MODES))
    train_steps: int = 10000
    batch_size: int = 16
    num_workers: int = 4
    learning_rate: float = 1e-5
    fsdp_devices: int = 1
    demo_fraction: float = 1.0
    action_steps: int = 8
    max_turns: int = 100
    planner_timeout_s: int = 600
    cell_timeout_s: int = 1200
    env_cuda_device: int = 0
    vla_cuda_device: int = 1
    epsilon: float = 0.05
    planners: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            key: dict(value) for key, value in DEFAULT_PLANNERS.items()
        }
    )

    @classmethod
    def load(cls, path: str | Path) -> Experiment:
        """Load JSON and resolve configured paths relative to that file."""
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("base_checkpoint", "openpi_root", "datasets_root", "output_root"):
            if key in data:
                value = Path(data[key]).expanduser()
                data[key] = str(
                    value if value.is_absolute() else (path.parent / value).resolve()
                )
        for key in ("openpi_python", "simulator_python"):
            if key in data and ("/" in data[key] or "\\" in data[key]):
                value = Path(data[key]).expanduser()
                data[key] = str(
                    value if value.is_absolute() else (path.parent / value).resolve()
                )
        try:
            config = cls(**data)
        except TypeError as exc:
            raise ValueError(f"invalid experiment configuration: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        """Reject ambiguous identities, invalid budgets, and unsupported models."""
        catalog = load_catalog()
        targets = {t for group in catalog["target_splits"].values() for t in group}
        if self.adaptation_mode not in {"independent", "joint"}:
            raise ValueError("adaptation_mode must be independent or joint")
        if self.evaluation_tasks is not None:
            if not isinstance(self.evaluation_tasks, dict) or set(
                self.evaluation_tasks
            ) - {"pretrain", "target"}:
                raise ValueError(
                    "evaluation_tasks must map pretrain/target to task lists"
                )
            for split, values in self.evaluation_tasks.items():
                if not isinstance(values, list) or any(
                    not isinstance(t, str) for t in values
                ):
                    raise ValueError(
                        "evaluation_tasks values must be lists of task names"
                    )
                allowed = (
                    set(catalog["pretrain300"]) if split == "pretrain" else targets
                )
                if len(values) != len(set(values)) or set(values) - allowed:
                    raise ValueError(
                        f"evaluation_tasks.{split} must contain unique official tasks in that split"
                    )
            if not any(self.evaluation_tasks.values()):
                raise ValueError("evaluation_tasks must select at least one task")
        for name, values in (
            ("methods", self.methods),
            ("modes", self.modes),
            ("train_seeds", self.train_seeds),
            ("eval_seeds", self.eval_seeds),
        ):
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be nonempty and unique")
        if set(self.methods) - {"full", "lora"} or set(self.modes) - set(MODES):
            raise ValueError(
                "methods must be full/lora; modes must be direct/qwen3_vl_4b/qwen35_4b"
            )
        for seed in self.train_seeds + self.eval_seeds:
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError("seeds must be integers in [0, 2**32)")
        if self.adaptation_tasks is not None:
            if not self.adaptation_tasks or len(set(self.adaptation_tasks)) != len(
                self.adaptation_tasks
            ):
                raise ValueError("adaptation_tasks must be nonempty and unique")
            if set(self.adaptation_tasks) - targets:
                raise ValueError(
                    "adaptation_tasks must be members of official Target50"
                )
        for name in (
            "train_steps",
            "batch_size",
            "fsdp_devices",
            "action_steps",
            "max_turns",
            "planner_timeout_s",
            "cell_timeout_s",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.num_workers) is not int
            or self.num_workers < 0
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError(
                "num_workers must be nonnegative and learning_rate positive"
            )
        if not 0 < self.demo_fraction <= 1 or not 0 < self.epsilon < 1:
            raise ValueError("demo_fraction must lie in (0, 1]; epsilon in (0, 1)")
        if self.batch_size % self.fsdp_devices:
            raise ValueError("batch_size must be divisible by fsdp_devices")
        for name in ("env_cuda_device", "vla_cuda_device"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not self.checkpoint_revision.strip():
            raise ValueError(
                "checkpoint_revision must identify the immutable source checkpoint"
            )
        for mode in set(self.modes) - {"direct"}:
            planner = self.planners.get(mode, {})
            if not {"model", "base_url"}.issubset(planner):
                raise ValueError(f"{mode} requires model and base_url")
            if not planner["base_url"].startswith(("http://", "https://")):
                raise ValueError(f"{mode} requires an HTTP API URL")
            if not planner["model"].strip() or re.search(
                r"qwen3-4b", planner["model"], re.I
            ):
                raise ValueError(
                    "use a vision model such as Qwen3-VL-4B-Instruct, not text-only Qwen3-4B"
                )

    @cached_property
    def protocol_id(self) -> str:
        settings = asdict(self)
        for key in (
            "openpi_python",
            "simulator_python",
            "env_cuda_device",
            "vla_cuda_device",
        ):
            settings.pop(key)
        for planner in settings["planners"].values():
            planner.pop("base_url")
        root = Path(__file__).resolve().parents[3]
        sources = sorted((root / "rpent").rglob("*.py")) + sorted(
            (root / "robots" / "robocasa").rglob("*.py")
        )
        implementation = fingerprint(
            {
                str(path.relative_to(root).as_posix()): path.read_text(encoding="utf-8")
                for path in sources
            }
        )
        return fingerprint(
            {
                "experiment": settings,
                "catalog": load_catalog(),
                "implementation": implementation,
            }
        )

    def runs(self) -> list[dict]:
        """Enumerate adaptations; every run starts at the same base checkpoint."""
        catalog = load_catalog()
        tasks = self.adaptation_tasks or [
            task for group in catalog["target_splits"].values() for task in group
        ]
        if self.adaptation_mode == "joint":
            tasks = sorted(tasks)
            label = (
                "joint_target50"
                if len(tasks) == 50
                else f"joint_{len(tasks)}_{fingerprint(tasks)[:8]}"
            )
            groups = [(label, tasks)]
        else:
            groups = [(task, [task]) for task in tasks]
        return [
            {
                "id": f"{method}/{label}/seed_{seed}",
                "label": label,
                "task": run_tasks[0] if self.adaptation_mode == "independent" else None,
                "tasks": list(run_tasks),
                "adaptation_mode": self.adaptation_mode,
                "method": method,
                "train_seed": seed,
                "source_checkpoint": self.base_checkpoint,
            }
            for method in self.methods
            for label, run_tasks in groups
            for seed in self.train_seeds
        ]

    def run(self, run_id: str) -> dict:
        for run in self.runs():
            if run["id"] == run_id:
                return run
        raise ValueError(f"unknown adaptation run: {run_id}")

    def evaluation_sets(self) -> dict[str, list[str]]:
        """Resolve the declared evaluation scope; omitted split keys select no tasks."""
        if self.evaluation_tasks is not None:
            return {
                split: list(self.evaluation_tasks.get(split, []))
                for split in ("pretrain", "target")
            }
        catalog = load_catalog()
        return {
            "pretrain": list(catalog["pretrain300"]),
            "target": [t for group in catalog["target_splits"].values() for t in group],
        }

    def grid(self) -> Iterator[tuple[str, str, int]]:
        """Yield the declared task/scene pairs, keeping the two scene domains separate."""
        for split, tasks in self.evaluation_sets().items():
            for task in tasks:
                for seed in self.eval_seeds:
                    yield split, task, seed

    def plan(self) -> dict:
        """Describe the complete grid without materializing hundreds of thousands of cells."""
        evaluation = self.evaluation_sets()
        task_count = sum(map(len, evaluation.values()))
        per_checkpoint = task_count * len(self.eval_seeds) * len(self.modes)
        runs = self.runs()
        return {
            "schema_version": 2,
            "protocol_id": self.protocol_id,
            "design": "joint_multitask_adaptation"
            if self.adaptation_mode == "joint"
            else "independent_single_task_adaptation",
            "experiment": asdict(self),
            "catalog_source": load_catalog()["source_url"],
            "adaptations": runs,
            "evaluation_sets": evaluation,
            "evaluation_scope": "full_benchmark"
            if task_count == 350
            else "task_subset",
            "evaluated_old_tasks": len(evaluation["pretrain"]),
            "evaluated_target_tasks": len(evaluation["target"]),
            "training_steps_per_run": self.train_steps,
            "total_training_steps": self.train_steps * len(runs),
            "training_sampling": "upstream_length_power_0.4"
            if any(len(run["tasks"]) > 1 for run in runs)
            else "single_task",
            "baseline_cells": per_checkpoint,
            "adapted_cells": per_checkpoint * len(runs),
            "total_cells": per_checkpoint * (1 + len(runs)),
            "old_tasks": 300,
            "target_tasks": 50,
            "overlap_tasks": 34,
            "old_tasks_excluding_all_target_tasks": 266,
            "success_source": "native_environment_check_success",
            "memory": "empty_per_cell_no_cross_episode_learning",
            "normalization": "frozen_base_checkpoint_stats",
        }
