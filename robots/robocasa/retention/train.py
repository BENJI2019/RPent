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

"""Run the official OpenPI SFT loop on selected Target Human tasks."""

from __future__ import annotations

import argparse
import json
import os
import runpy
from pathlib import Path

from robots.robocasa.retention.openpi_adapter import (
    norm_digest,
    training_config,
    verify_openpi_checkout,
)
from robots.robocasa.retention.protocol import Experiment, load_catalog
from robots.robocasa.retention.recovery import execution_lock
from rpent.evaluation import write_json_atomic
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("retention.train")


def train(experiment: Experiment, run_id: str, *, resume: bool = False) -> Path:
    """Train one configured full/LoRA model and record its exact final checkpoint."""
    run = experiment.run(run_id)
    root = Path(experiment.output_root)
    with (
        execution_lock(root / ".execution.lock", shared=True),
        execution_lock(root / "training" / run_id / ".training.lock"),
    ):
        return train_locked(experiment, run, resume=resume)


def train_locked(experiment: Experiment, run: dict, *, resume: bool) -> Path:
    """Run the optimizer with exclusive ownership of this training run."""
    run_id = run["id"]
    verify_openpi_checkout(experiment.openpi_root)
    output = Path(experiment.output_root) / "training" / run_id
    output.mkdir(parents=True, exist_ok=True)
    init_output_dir(output)
    record = {
        "protocol_id": experiment.protocol_id,
        "training_id": experiment.training_id,
        "run": run,
        "base_norm_sha256": norm_digest(experiment.base_checkpoint),
        "openpi_revision": load_catalog()["openpi_revision"],
        "pretraining_samples_used": 0,
    }
    provenance = output / "provenance.json"
    if provenance.exists():
        previous = json.loads(provenance.read_text(encoding="utf-8"))
        expected = {k: v for k, v in record.items() if k != "protocol_id"}
        actual = {k: v for k, v in previous.items() if k != "protocol_id"}
        if actual != expected:
            raise ValueError(
                "refusing to reuse training output with different training provenance"
            )
    completed = output / "checkpoint.json"
    if completed.exists():
        if resume:
            from robots.robocasa.retention.evaluate import resolve_checkpoint

            final, _ = resolve_checkpoint(experiment, run_id)
            logger.info("already completed %s: %s", run_id, final)
            return final
        raise FileExistsError(f"run already completed: {completed}")
    cfg = training_config(experiment, run, resume=resume)
    write_json_atomic(provenance, record)
    entry = Path(experiment.openpi_root) / "scripts" / "train.py"
    if not entry.is_file():
        raise FileNotFoundError(
            f"official OpenPI training entry point missing: {entry}"
        )
    # Execute the installed upstream training entry point, not a copied optimizer.
    upstream = runpy.run_path(str(entry), run_name="rpent_openpi_training")
    # This upstream revision hardcodes ~/.cache/jax in main. Redirect that one
    # setting without changing HOME, source checkout, or optimization semantics.
    if cache := os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        import jax

        update = jax.config.update

        def configure(name, value):
            update(name, cache if name == "jax_compilation_cache_dir" else value)

        jax.config.update = configure
        try:
            upstream["main"](cfg)
        finally:
            jax.config.update = update
    else:
        upstream["main"](cfg)
    final = cfg.checkpoint_dir / str(experiment.train_steps - 1)
    if not (final / "params").is_dir():
        raise RuntimeError(
            f"training did not produce the expected final checkpoint: {final}"
        )
    if norm_digest(final) != record["base_norm_sha256"]:
        # Serialization may differ, so compare parsed numerical statistics.
        from robots.robocasa.retention.openpi_adapter import norm_file

        before = json.loads(norm_file(experiment.base_checkpoint).read_text())
        after = json.loads(norm_file(final).read_text())
        if before != after:
            raise RuntimeError("training changed normalization statistics")
    write_json_atomic(
        completed, {**record, "checkpoint": str(final), "method": run["method"]}
    )
    logger.info("completed %s: %s", run_id, final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(Experiment.load(args.config), args.run_id, resume=args.resume)


if __name__ == "__main__":
    main()
