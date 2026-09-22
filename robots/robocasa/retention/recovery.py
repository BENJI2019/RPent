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

"""Training recovery and explicit evaluation-plan refresh after Harness debugging."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Iterator

from robots.robocasa.retention.protocol import Experiment
from rpent.evaluation import write_json_atomic


@contextlib.contextmanager
def execution_lock(path: Path, *, shared: bool = False) -> Iterator[None]:
    """Hold an OS lock, released on process exit; never infer liveness from a PID."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            if path.stat().st_size == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            operation = msvcrt.LK_NBRLCK if shared else msvcrt.LK_NBLCK
            try:
                msvcrt.locking(lock.fileno(), operation, 1)
            except OSError as exc:
                raise RuntimeError(f"another process holds {path}") from exc
        else:
            import fcntl

            operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            try:
                fcntl.flock(lock, operation | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"another process holds {path}") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def checkpoint_directory(experiment: Experiment, run: dict) -> Path:
    """Locate the upstream checkpoint directory without importing GPU packages."""
    return (
        Path(experiment.output_root)
        / "checkpoints"
        / f"pi05_retention_{run['method']}"
        / f"{run['label']}_seed_{run['train_seed']}"
    )


def training_status(experiment: Experiment) -> dict:
    """Report saved-state candidates; Orbax remains the authority on restoration."""
    root = Path(experiment.output_root)
    runs = []
    for run in experiment.runs():
        directory = checkpoint_directory(experiment, run)
        steps = sorted(
            int(p.name)
            for p in directory.glob("*")
            if p.name.isdigit()
            and (p / "params").is_dir()
            and (p / "train_state").is_dir()
        )
        complete = root / "training" / run["id"] / "checkpoint.json"
        runs.append(
            {
                "run_id": run["id"],
                "completion_record": complete.is_file(),
                "checkpoint_directory": str(directory),
                "saved_step_candidates": steps,
                "next_step_if_restorable": steps[-1] + 1 if steps else 0,
            }
        )
    plan_file = root / "plan.json"
    stored = (
        json.loads(plan_file.read_text(encoding="utf-8")) if plan_file.exists() else {}
    )
    return {
        "protocol_matches": stored.get("protocol_id") == experiment.protocol_id,
        "training_matches": stored.get("training_id") == experiment.training_id,
        "refresh_pending": (root / ".refresh-plan.json").exists(),
        "runs": runs,
        "evaluation_running_markers": [
            str(p) for p in (root / "evaluation").rglob(".running")
        ],
        "note": "Directory inspection only. Orbax validates the saved optimizer state on resume; the upstream data-loader cursor is not restored.",
    }


def refresh_plan(experiment: Experiment) -> Path | None:
    """Archive old evaluation outputs while retaining compatible optimizer states.

    The caller must hold the exclusive experiment lock. A journal makes an
    interrupted archive recoverable by rerunning this command with the same config.
    """
    root = Path(experiment.output_root)
    plan_file = root / "plan.json"
    journal = root / ".refresh-plan.json"
    new = experiment.plan()
    if journal.exists():
        transaction = json.loads(journal.read_text(encoding="utf-8"))
        if transaction["new_plan"] != new:
            raise ValueError(
                "finish the pending refresh with its original code/config first"
            )
        old = transaction["old_plan"]
    else:
        old = json.loads(plan_file.read_text(encoding="utf-8"))
        if old["protocol_id"] == new["protocol_id"]:
            return None
        if old.get("training_id") != new["training_id"]:
            raise ValueError(
                "training settings/adapter changed or legacy plan lacks training_id; use a new experiment"
            )
        if any((root / "evaluation").rglob(".running")):
            raise ValueError(
                "evaluation .running markers exist; confirm the old workers stopped before clearing those markers"
            )
        write_json_atomic(journal, {"old_plan": old, "new_plan": new})
    identity = old["protocol_id"]
    if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("invalid archived protocol identity")
    archive = root / "history" / identity
    archive.mkdir(parents=True, exist_ok=True)
    write_json_atomic(archive / "plan.json", old)
    for name in ("evaluation", "summary", "services"):
        source, destination = root / name, archive / name
        if source.exists():
            if destination.exists():
                raise FileExistsError(
                    f"refresh cannot overwrite archived outputs: {destination}"
                )
            source.rename(destination)
    write_json_atomic(plan_file, new)
    journal.unlink()
    return archive
