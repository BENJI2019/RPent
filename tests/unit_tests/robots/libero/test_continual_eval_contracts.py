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

"""Offline contracts for the frozen-memory LIBERO evaluation schedule."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[4] / "robots" / "libero" / "continual_eval.py"
LAUNCHER = SCRIPT.with_name("run_local_qwen.sh")
SPEC = importlib.util.spec_from_file_location("continual_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
continual_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(continual_eval)


def _args(root: Path, *, resume: bool = False) -> list[str]:
    args = [
        "--output-root",
        str(root),
        "--tasks-per-suite",
        "1",
        "--eval-seeds",
        "1",
        "--skip-api-check",
    ]
    return [*args, "--resume"] if resume else args


def _fake_runner(root: Path, *, fail_once: bool = False, change_memory: bool = False):
    commands = []

    def run(command, *, env):
        assert env["OPENAI_API_KEY"] == "test-key"
        commands.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        memory = Path(command[command.index("--memory-dir") + 1])
        output.mkdir(parents=True)
        explore = "--explore" in command
        if explore:
            assert memory == root / "memory_live"
            if change_memory:
                (memory / "MEMORY.md").write_text("learned", encoding="utf-8")
            states = output / "sessions" / "session_001" / "states.json"
        else:
            assert memory.parent == root / "snapshots"
            states = output / "states.json"
        states.parent.mkdir(parents=True, exist_ok=True)
        states.write_text(
            json.dumps({"steps": [{"terminated": explore}]}), encoding="utf-8"
        )
        return SimpleNamespace(returncode=2 if fail_once and len(commands) == 1 else 0)

    return commands, run


def test_schedule_keeps_baseline_and_all_four_suite_probes() -> None:
    plan = list(continual_eval.schedule(continual_eval.SUITES, 10, tuple(range(1, 11))))
    assert len(plan) == 2040
    assert sum(job[1] == "explore" for job in plan) == 40
    assert sum(job[1] == "eval" for job in plan) == 2000
    assert plan[0] == (0, "eval", "libero_spatial", 0, 1)
    assert plan[400] == (1, "explore", "libero_spatial", 0, 0)
    assert plan[-1] == (4, "eval", "libero_10", 9, 10)
    assert all(job[4] == 0 for job in plan if job[1] == "explore")


def test_outputs_use_environment_success_and_frozen_stage_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "experiment"
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    commands, runner = _fake_runner(root, change_memory=True)
    monkeypatch.setattr(continual_eval.subprocess, "run", runner)

    assert continual_eval.main(_args(root)) == 0
    assert len(commands) == 24
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    assert len(manifest["snapshots"]) == 5
    assert len(manifest["jobs"]) == 24
    assert "No learned memory" in (root / "snapshots/stage_00/MEMORY.md").read_text()
    assert (root / "snapshots/stage_01/MEMORY.md").read_text() == "learned"
    assert all(
        job["success"] is False
        for job in manifest["jobs"].values()
        if job["phase"] == "eval"
    )
    assert "elapsed_s" in next(iter(manifest["jobs"].values()))
    assert "0.0" in (root / "matrix.csv").read_text(encoding="utf-8")
    assert "0.0" in (root / "retention.csv").read_text(encoding="utf-8")
    assert all("--memory-ablation" in command for command in commands)
    assert all(
        command[command.index("--libero-type") + 1] == "standard"
        for command in commands
    )
    assert all(
        command[command.index("--max-tokens") + 1] == "8192" for command in commands
    )


def test_failed_evaluation_can_resume_without_repeating_completed_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "experiment"
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    first_commands, first_runner = _fake_runner(root, fail_once=True)
    monkeypatch.setattr(continual_eval.subprocess, "run", first_runner)
    assert continual_eval.main(_args(root)) == 1
    assert len(first_commands) == 1

    subsequent_commands, runner = _fake_runner(root)
    monkeypatch.setattr(continual_eval.subprocess, "run", runner)
    assert continual_eval.main(_args(root, resume=True)) == 0
    assert len(subsequent_commands) == 24
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    assert manifest["jobs"]["0:eval:libero_spatial:0:1"]["attempt"] == 2


def test_failed_exploration_with_partial_memory_refuses_automatic_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "experiment"
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = 0

    def runner(command, *, env):
        nonlocal calls
        calls += 1
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        if "--explore" in command:
            memory = root / "memory_live" / "MEMORY.md"
            memory.write_text("partial", encoding="utf-8")
            return SimpleNamespace(returncode=1)
        (output / "states.json").write_text(
            json.dumps({"steps": [{"terminated": False}]}), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(continual_eval.subprocess, "run", runner)
    assert continual_eval.main(_args(root)) == 1
    assert calls == 5
    with pytest.raises(RuntimeError, match="live memory differs"):
        continual_eval.main(_args(root, resume=True))
    assert calls == 5


def test_dry_run_does_not_need_key_or_write_outputs(tmp_path: Path) -> None:
    root = tmp_path / "unused"
    assert continual_eval.main(["--output-root", str(root), "--dry-run"]) == 0
    assert not root.exists()


def test_local_launcher_checks_assets_before_starting_services(tmp_path: Path) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is unavailable on this platform")
    environment = os.environ.copy()
    environment["PI05_CHECKPOINT_PATH"] = str(tmp_path / "missing-pi05")
    result = subprocess.run(
        [bash, str(LAUNCHER), str(tmp_path / "experiment")],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Pi0.5 directory missing" in result.stderr
    assert not (tmp_path / "experiment.services").exists()


def test_local_launcher_resolves_sam3_checkpoint_from_install_directory(
    tmp_path: Path,
) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is unavailable on this platform")
    pi05_dir = tmp_path / "pi05"
    sam3_dir = tmp_path / "sam3"
    pi05_dir.mkdir()
    sam3_dir.mkdir()
    environment = os.environ.copy()
    environment["PI05_CHECKPOINT_PATH"] = str(pi05_dir)
    environment["SAM3_MODEL_DIR"] = str(sam3_dir)
    environment.pop("SAM3_CHECKPOINT_PATH", None)
    result = subprocess.run(
        [bash, str(LAUNCHER), str(tmp_path / "experiment")],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert f"SAM3 checkpoint missing: {sam3_dir / 'sam3.pt'}" in result.stderr
