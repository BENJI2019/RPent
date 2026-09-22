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

"""Build official RoboCasa OpenPI configs without reading pretraining datasets."""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
from pathlib import Path
from typing import Any

from robots.robocasa.retention.protocol import Experiment, load_catalog


def verify_openpi_checkout(root: str | Path) -> None:
    """Require the inspected official fork and reject a mismatched installation."""
    from openpi.training import config

    root = Path(root).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != load_catalog()["openpi_revision"]:
        raise ValueError("OpenPI checkout does not match catalog.openpi_revision")
    if not Path(config.__file__).resolve().is_relative_to(root):
        raise ValueError(
            "installed openpi is not imported from the configured official fork"
        )
    if subprocess.check_output(
        ["git", "-C", str(root), "diff", "--name-only", "HEAD", "--", "src", "scripts"],
        text=True,
    ).strip():
        raise ValueError(
            "OpenPI source has local changes; use the pinned, unmodified fork"
        )


def norm_file(checkpoint: str | Path) -> Path:
    """Locate one unambiguous normalization asset; never infer it from datasets."""
    files = sorted((Path(checkpoint) / "assets").rglob("norm_stats.json"))
    if len(files) != 1:
        raise ValueError(
            f"{checkpoint}: expected one assets/**/norm_stats.json, got {len(files)}"
        )
    return files[0]


def norm_digest(checkpoint: str | Path) -> str:
    return hashlib.sha256(norm_file(checkpoint).read_bytes()).hexdigest()


def policy_config(checkpoint: str | Path, method: str = "full") -> tuple[Any, Any]:
    """Reuse the pinned official architecture and transforms with explicit stats."""
    from openpi.shared import normalize
    from openpi.training import config

    if method not in {"full", "lora"}:
        raise ValueError("method must be full or lora")
    base = config.get_config("pi05_pretrain_human300")
    if not base.model.pi05:
        raise ValueError("pi05_pretrain_human300 must resolve to a pi0.5 model")
    stats = normalize.load(norm_file(checkpoint).parent)
    model = base.model
    if method == "lora":
        model = dataclasses.replace(
            model,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
    # Empty data_dirs prevents the upstream factory from falling back to source
    # dataset metadata to compute normalization during inference.
    data = dataclasses.replace(
        base.data,
        data_dirs=[],
        base_config=config.DataConfig(norm_stats=stats),
        assets=config.AssetsConfig(),
    )
    return dataclasses.replace(base, model=model, data=data), stats


def training_config(experiment: Experiment, run: dict, *, resume: bool = False) -> Any:
    """Create single-task or mixed-task SFT from the immutable Human300 parent."""
    from openpi.training import optimizer, weight_loaders

    cfg, _ = policy_config(experiment.base_checkpoint, run["method"])
    catalog = load_catalog()
    root = Path(experiment.datasets_root).resolve()
    demo_count = int(500 * experiment.demo_fraction)
    if demo_count < 1:
        raise ValueError("demo_fraction selects no demonstrations")
    datasets = []
    for task in run["tasks"]:
        metadata = catalog["tasks"][task]
        dataset = (root / metadata["target_dataset"]).resolve()
        if not dataset.is_relative_to(root) or "/target/" not in dataset.as_posix():
            raise ValueError("training dataset must be an official Target Human path")
        if not (dataset / "meta" / "info.json").is_file():
            raise FileNotFoundError(
                f"Target Human LeRobot dataset is missing: {dataset}"
            )
        datasets.append(
            {
                "path": str(dataset),
                "task": task,
                "split": "target",
                "source": "human",
                "horizon": metadata["horizon"],
                "filter_key": f"{demo_count}_demos",
            }
        )
    name = f"pi05_retention_{run['method']}"
    return dataclasses.replace(
        cfg,
        name=name,
        exp_name=f"{run['label']}_seed_{run['train_seed']}",
        # The pinned upstream mixture samples using dataset length ** 0.4.
        # Its explicit-weight branch is broken; retain the supported default.
        data=dataclasses.replace(cfg.data, data_dirs=datasets, dataset_weights=None),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(Path(experiment.base_checkpoint) / "params")
        ),
        freeze_filter=cfg.model.get_freeze_filter(),
        ema_decay=None if run["method"] == "lora" else cfg.ema_decay,
        checkpoint_base_dir=str(Path(experiment.output_root) / "checkpoints"),
        assets_base_dir=str(Path(experiment.output_root) / "assets"),
        seed=run["train_seed"],
        num_train_steps=experiment.train_steps,
        batch_size=experiment.batch_size,
        num_workers=experiment.num_workers,
        fsdp_devices=experiment.fsdp_devices,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=min(1000, max(1, experiment.train_steps // 10)),
            peak_lr=experiment.learning_rate,
            decay_steps=experiment.train_steps,
            decay_lr=experiment.learning_rate / 10,
        ),
        save_interval=min(1000, experiment.train_steps),
        keep_period=None,
        wandb_enabled=False,
        overwrite=False,
        resume=resume,
    )
