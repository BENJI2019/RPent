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

"""Offline regression checks for training continuation and evaluation isolation."""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.robocasa.retention import cluster_evaluate, prepare, recovery
from robots.robocasa.retention.protocol import Experiment
from rpent.evaluation import write_json_atomic


@pytest.fixture
def configuration(tmp_path):
    cfg = prepare.initialize(
        tmp_path / "pilot.json", preset="pilot1", workspace=tmp_path
    )
    Path(cfg.output_root).mkdir(parents=True)
    write_json_atomic(Path(cfg.output_root) / "plan.json", cfg.plan())
    return cfg


@pytest.mark.parametrize(
    "changes",
    [
        {"learning_rate": 2e-5},
        {"batch_size": 8},
        {"adaptation_tasks": ["PrepareCoffee"]},
        {"fsdp_devices": 8},
        {"train_steps": 2000},
    ],
)
def test_changed_training_cannot_refresh_in_place(configuration, changes):
    cfg = dataclasses.replace(configuration, **changes)
    root = Path(cfg.output_root)
    original = (root / "plan.json").read_bytes()
    with pytest.raises(ValueError, match="training settings"):
        recovery.refresh_plan(cfg)
    assert (root / "plan.json").read_bytes() == original
    assert not (root / ".refresh-plan.json").exists()


def test_refresh_preserves_training_and_separates_old_evaluation(configuration):
    root = Path(configuration.output_root)
    for name in ("training", "checkpoints", "evaluation", "summary", "services"):
        (root / name).mkdir()
        (root / name / "record.txt").write_text(name)
    cfg = dataclasses.replace(configuration, action_steps=4, save_interval=50)
    assert cfg.training_id == configuration.training_id
    archive = recovery.refresh_plan(cfg)
    for name in ("training", "checkpoints"):
        assert (root / name / "record.txt").read_text() == name
    for name in ("evaluation", "summary", "services"):
        assert not (root / name).exists()
        assert (archive / name / "record.txt").read_text() == name
    assert (
        json.loads((archive / "plan.json").read_text())["protocol_id"]
        == configuration.protocol_id
    )
    assert (
        json.loads((root / "plan.json").read_text())["protocol_id"] == cfg.protocol_id
    )
    assert recovery.refresh_plan(cfg) is None


def test_refresh_recovers_after_partial_archive(configuration, monkeypatch):
    root = Path(configuration.output_root)
    for name in ("evaluation", "services"):
        (root / name).mkdir()
    cfg = dataclasses.replace(configuration, action_steps=4)
    rename = Path.rename

    def interrupted(path, target):
        if path.name == "services":
            raise OSError("simulated interruption")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", interrupted)
    with pytest.raises(OSError, match="interruption"):
        recovery.refresh_plan(cfg)
    assert (root / ".refresh-plan.json").is_file()
    monkeypatch.setattr(Path, "rename", rename)
    archive = recovery.refresh_plan(cfg)
    assert (archive / "evaluation").is_dir() and (archive / "services").is_dir()
    assert not (root / ".refresh-plan.json").exists()


def test_refresh_refuses_unresolved_episode_marker(configuration):
    marker = Path(configuration.output_root) / "evaluation/cell/.running"
    marker.parent.mkdir(parents=True)
    marker.write_text("old process evidence")
    with pytest.raises(ValueError, match="markers exist"):
        recovery.refresh_plan(dataclasses.replace(configuration, action_steps=4))
    assert marker.read_text() == "old process evidence"


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_interrupted_training_resumes_and_completed_model_is_reused(
    configuration, monkeypatch, cache_enabled
):
    from robots.robocasa.retention import train as trainer
    from robots.robocasa.retention.evaluate import resolve_checkpoint

    cfg = configuration
    run = cfg.runs()[0]
    base = Path(cfg.base_checkpoint)
    (base / "assets").mkdir(parents=True)
    (base / "assets/norm_stats.json").write_text("{}")
    directory = recovery.checkpoint_directory(cfg, run)
    entry = Path(cfg.openpi_root) / "scripts/train.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("# offline upstream stand-in")
    seen = []
    updates = []
    jax_config = SimpleNamespace(
        update=lambda name, value: updates.append((name, value))
    )
    original_update = jax_config.update
    if cache_enabled:
        monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(config=jax_config))
        monkeypatch.setenv(
            "JAX_COMPILATION_CACHE_DIR", str(Path(cfg.output_root) / "cache")
        )
    else:
        monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setattr(trainer, "verify_openpi_checkout", lambda root: None)
    monkeypatch.setattr(
        trainer,
        "training_config",
        lambda exp, r, resume: SimpleNamespace(checkpoint_dir=directory, resume=resume),
    )

    def upstream(config):
        if cache_enabled:
            jax_config.update("jax_compilation_cache_dir", "~/.cache/jax")
            jax_config.update("other_setting", True)
        seen.append(config.resume)
        if len(seen) == 1:
            for part in ("params", "train_state"):
                (directory / "100" / part).mkdir(parents=True)
            raise RuntimeError("job interrupted after a saved state")
        assert (directory / "100/train_state").is_dir() and config.resume
        final = directory / str(cfg.train_steps - 1)
        for part in ("params", "assets", "train_state"):
            (final / part).mkdir(parents=True)
        (final / "assets/norm_stats.json").write_text("{}")

    monkeypatch.setattr(trainer.runpy, "run_path", lambda *a, **k: {"main": upstream})
    with pytest.raises(RuntimeError, match="interrupted"):
        trainer.train(cfg, run["id"], resume=True)
    assert recovery.training_status(cfg)["runs"][0]["next_step_if_restorable"] == 101
    changed = dataclasses.replace(cfg, action_steps=4)
    recovery.refresh_plan(changed)
    final = trainer.train(changed, run["id"], resume=True)
    assert final.name == "999"
    assert trainer.train(changed, run["id"], resume=True) == final.resolve()
    assert len(seen) == 2
    assert resolve_checkpoint(changed, run["id"])[0] == final.resolve()
    if cache_enabled:
        assert jax_config.update is original_update
        assert (
            updates
            == [
                ("jax_compilation_cache_dir", str(Path(cfg.output_root) / "cache")),
                ("other_setting", True),
            ]
            * 2
        )


def test_execution_lock_blocks_a_refresh_and_releases_on_exit(tmp_path):
    path = tmp_path / ".lock"
    code = "from pathlib import Path; from robots.robocasa.retention.recovery import execution_lock; import sys;\nwith execution_lock(Path(sys.argv[1])): pass"
    with recovery.execution_lock(path, shared=True):
        busy = subprocess.run(
            [sys.executable, "-c", code, str(path)], capture_output=True, text=True
        )
        assert busy.returncode != 0 and "another process holds" in busy.stderr
    free = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True
    )
    assert free.returncode == 0, free.stderr


@pytest.mark.parametrize(
    "mode,gpus,worker_count",
    [
        ("direct", [0], 1),
        ("direct", list(range(8)), 8),
        ("qwen35_4b", [0], 1),
        ("qwen35_4b", list(range(8)), 7),
    ],
)
def test_batch_evaluation_places_qwen_and_workers_in_one_job(
    configuration, monkeypatch, mode, gpus, worker_count
):
    cfg = dataclasses.replace(configuration, modes=[mode])
    write_json_atomic(Path(cfg.output_root) / "plan.json", cfg.plan())
    launched, closed = [], []

    @contextmanager
    def process(command, log, env):
        launched.append((command, env))
        try:
            yield SimpleNamespace(poll=lambda: 0, returncode=0)
        finally:
            closed.append(command)

    monkeypatch.setattr(cluster_evaluate, "process_group", process)
    monkeypatch.setattr(cluster_evaluate, "wait_for_planner", lambda *a: None)
    monkeypatch.setattr(cluster_evaluate, "require_free_planner_port", lambda url: None)
    monkeypatch.setenv("RETENTION_QWEN_PREFIX", "/shared/qwen")
    cluster_evaluate.run(cfg, Path("config.json"), mode, gpus, "all")
    offset = int(mode != "direct")
    assert len(launched) == len(closed) == worker_count + offset
    if offset:
        assert launched[0][1]["CUDA_VISIBLE_DEVICES"] == str(gpus[-1])
    for index, (command, _) in enumerate(launched[offset:]):
        assert command[command.index("--num-shards") + 1] == str(worker_count)
        assert command[command.index("--shard-index") + 1] == str(index)
        worker = Experiment.load(command[command.index("--config") + 1])
        assert worker.env_cuda_device == worker.vla_cuda_device == index
        assert worker.protocol_id == cfg.protocol_id


def test_batch_rejects_an_existing_planner_listener():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        url = f"http://127.0.0.1:{listener.getsockname()[1]}/v1"
        with pytest.raises(RuntimeError, match="existing service"):
            cluster_evaluate.require_free_planner_port(url)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group ownership")
def test_owned_process_is_terminated_when_batch_scope_exits(tmp_path):
    with cluster_evaluate.process_group(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        tmp_path / "worker.log",
        dict(os.environ),
    ) as process:
        assert process.poll() is None
    assert process.poll() is not None
