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

"""Offline checks for server preparation, dataset ownership and readiness reports."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.robocasa.retention import preflight, prepare
from robots.robocasa.retention.protocol import Experiment, load_catalog


@pytest.fixture
def setup_config(tmp_path):
    path = tmp_path / "configs/pilot.json"
    cfg = prepare.initialize(path, preset="pilot1", workspace=tmp_path / "workspace")
    cfg = dataclasses.replace(
        cfg, openpi_python=sys.executable, simulator_python=sys.executable
    )
    path.write_text(json.dumps(dataclasses.asdict(cfg)), encoding="utf-8")
    return cfg, path


def create_dataset(path):
    (path / "meta").mkdir(parents=True)
    for name in (
        "info.json",
        "modality.json",
        "stats.json",
        "tasks.jsonl",
        "episodes.jsonl",
    ):
        (path / "meta" / name).write_text("{}\n", encoding="utf-8")
    (path / "meta/info.json").write_text('{"total_episodes": 1}', encoding="utf-8")
    files = [
        "data/chunk-000/episode_000000.parquet",
        *(
            f"videos/chunk-000/observation.images.{camera}/episode_000000.mp4"
            for camera in (
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            )
        ),
    ]
    for name in files:
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"offline fixture, not a decodable trajectory")


def test_initialize_resolves_paths_and_protects_existing_configuration(tmp_path):
    path = tmp_path / "pilot.json"
    cfg = prepare.initialize(
        path, preset="pilot3-joint", workspace=tmp_path, with_planners=True
    )
    assert cfg.modes == ["direct", "qwen3_vl_4b", "qwen35_4b"]
    assert cfg.env_cuda_device == cfg.vla_cuda_device == 0
    assert cfg.openpi_python == str(tmp_path / "openpi/.venv/bin/python")
    assert cfg.simulator_python == sys.executable
    assert len(cfg.runs()) == 1 and len(cfg.runs()[0]["tasks"]) == 3
    assert not Path(cfg.output_root).exists()
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        prepare.initialize(path, preset="joint50", workspace=tmp_path)
    assert path.read_bytes() == original
    multi = prepare.initialize(
        tmp_path / "joint50.json", preset="joint50", workspace=tmp_path, fsdp_devices=8
    )
    assert multi.fsdp_devices == 8 and multi.batch_size % 8 == 0
    repeated = prepare.initialize(
        tmp_path / "pilot-new.json",
        preset="pilot1",
        workspace=tmp_path,
        output_root=tmp_path / "runs/pilot-new",
    )
    assert repeated.output_root == str(tmp_path / "runs/pilot-new")


def test_huawei_cluster_resources_accepts_user_workspace_on_shared_dataset(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "dataset"
    home = dataset / "Common_wl/hyy_vla_retention"
    code_root = Path(prepare.__file__).resolve().parents[3]
    monkeypatch.setattr(prepare, "SHARED_MODEL_ROOTS", (tmp_path / "model",))
    monkeypatch.setattr(prepare, "SHARED_DATASET_ROOTS", (dataset,))
    monkeypatch.setattr(prepare.sys, "platform", "linux")
    for name, value in {
        "RETENTION_HOME": home,
        "RETENTION_CODE_ROOT": code_root,
        "RETENTION_DATASETS": dataset / "target",
        "RETENTION_CHECKPOINT": home / "checkpoint",
        "RETENTION_SIM_PREFIX": dataset / "Common_wl/envs/simulator",
        "RETENTION_OPENPI_PREFIX": dataset / "Common_wl/envs/openpi",
        "RETENTION_EXECUTION": "local",
    }.items():
        monkeypatch.setenv(name, str(value))

    destination = home / "configs/local/pilot.json"
    result = prepare.cluster_resources(destination, "pilot1", None)

    assert result["output_root"] == str(home / "retention_outputs/local/pilot1")
    assert Path(result["openpi_python"]).is_relative_to(home)


def test_training_data_plan_excludes_old_evaluation_probes(setup_config):
    cfg, _ = setup_config
    paths = prepare.target_paths(cfg)
    assert list(paths) == ["OpenDrawer"]
    assert all("/target/" in p.as_posix() for p in paths.values())
    assert not set(paths) & set(cfg.evaluation_tasks["pretrain"])


def test_dataset_checks_detect_partially_extracted_episodes(tmp_path):
    create_dataset(tmp_path / "dataset")
    info = tmp_path / "dataset/meta/info.json"
    info.write_text('{"total_episodes": 500}')
    issues = prepare.dataset_issues(tmp_path / "dataset")
    assert len(issues) == 4
    assert all("expected 500" in issue and "found 1" in issue for issue in issues)


@pytest.fixture
def downloader(setup_config, monkeypatch):
    cfg, _ = setup_config
    paths = prepare.target_paths(cfg)
    calls = []
    macros = SimpleNamespace(DATASET_BASE_PATH=None)

    def download(**kwargs):
        calls.append(kwargs)
        for task in kwargs["tasks"]:
            create_dataset(paths[task])

    upstream = SimpleNamespace(
        get_ds_meta=lambda task, **kwargs: {"path": str(paths[task])},
        download_datasets=download,
    )
    scripts = SimpleNamespace(download_datasets=upstream)
    monkeypatch.setitem(
        sys.modules, "robocasa", SimpleNamespace(macros=macros, scripts=scripts)
    )
    monkeypatch.setitem(sys.modules, "robocasa.macros", macros)
    monkeypatch.setitem(sys.modules, "robocasa.scripts", scripts)
    return upstream, calls, paths


def test_download_uses_configured_root_and_only_target_human(setup_config, downloader):
    cfg, _ = setup_config
    upstream, calls, paths = downloader
    prepare.download_selected(cfg)
    assert upstream.DATASET_BASE_PATH == str(Path(cfg.datasets_root).resolve())
    assert calls == [
        {
            "split": ["target"],
            "tasks": ["OpenDrawer"],
            "source": ["human"],
            "all_data": False,
            "overwrite": False,
            "dryrun": False,
        }
    ]
    assert not prepare.dataset_issues(paths["OpenDrawer"])


def test_download_does_not_hide_upstream_failure_or_overwrite_partial_data(
    setup_config, downloader
):
    cfg, _ = setup_config
    upstream, calls, paths = downloader
    upstream.download_datasets = lambda **kwargs: None
    with pytest.raises(RuntimeError, match="incomplete"):
        prepare.download_selected(cfg)
    path = paths["OpenDrawer"]
    path.mkdir(parents=True)
    sentinel = path / "partial-download"
    sentinel.write_bytes(b"keep this")
    with pytest.raises(ValueError, match="incomplete dataset"):
        prepare.download_selected(cfg)
    assert sentinel.read_bytes() == b"keep this" and not calls


def test_download_rejects_registry_drift_before_downloading(setup_config, downloader):
    cfg, _ = setup_config
    upstream, calls, _ = downloader
    upstream.get_ds_meta = lambda **kwargs: {"path": cfg.datasets_root + "/wrong"}
    with pytest.raises(ValueError, match="registry disagrees"):
        prepare.download_selected(cfg)
    assert not calls


def test_doctor_distinguishes_training_and_evaluation_prerequisites(
    setup_config, monkeypatch
):
    cfg, path = setup_config
    base = Path(cfg.base_checkpoint)
    (base / "params").mkdir(parents=True)
    (base / "params/manifest.ocdbt").write_bytes(b"placeholder")
    (base / "assets").mkdir()
    (base / "assets/norm_stats.json").write_text("{}")
    monkeypatch.setattr(
        preflight.subprocess,
        "check_output",
        lambda *args, **kwargs: load_catalog()["openpi_revision"],
    )
    train = preflight.inspect(cfg, path, stage="train", runtime=False)
    assert not train["ok"]
    assert [c["name"] for c in train["checks"] if not c["ok"]] == [
        "target_data/OpenDrawer"
    ]
    evaluation = preflight.inspect(cfg, path, stage="evaluate", runtime=False)
    assert evaluation["ok"] and not evaluation["runtime_checked"]
    assert not any(c["name"].startswith("target_data/") for c in evaluation["checks"])
    assert "no checkpoint restore" in evaluation["scope"]
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="missing CUDA"
        ),
    )
    runtime = preflight.inspect(cfg, path, stage="evaluate", runtime=True)
    assert not runtime["ok"]
    assert any(c["detail"] == "missing CUDA" for c in runtime["checks"])


def test_preparation_cli_does_not_lock_protocol_and_doctor_fails_cleanly(
    tmp_path, monkeypatch
):
    from robots.robocasa.retention.__main__ import main

    path = tmp_path / "config.json"
    prefix = ["retention", "--config", str(path)]
    monkeypatch.setattr(
        sys,
        "argv",
        [*prefix, "init", "--preset", "pilot1", "--workspace", str(tmp_path)],
    )
    main()
    cfg = Experiment.load(path)
    monkeypatch.setattr(sys, "argv", [*prefix, "download-data"])
    main()
    assert not Path(cfg.output_root).exists()
    monkeypatch.setattr(sys, "argv", [*prefix, "doctor", "--stage", "train"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    report = json.loads(path.with_suffix(".doctor.json").read_text())
    assert not report["ok"] and len([c for c in report["checks"] if not c["ok"]]) >= 3
    assert not Path(cfg.output_root).exists()


def test_shared_a800_planner_limits_are_forwarded_and_validated(setup_config):
    from robots.robocasa.retention.serve_planner import serve_command

    cfg, _ = setup_config
    cfg = dataclasses.replace(cfg, modes=["qwen3_vl_4b", "qwen35_4b"])
    for mode in cfg.modes:
        command = serve_command(cfg, mode)
        for flag, value in (
            ("--gpu-memory-utilization", "0.4"),
            ("--max-model-len", "32768"),
            ("--max-num-seqs", "1"),
        ):
            assert command[command.index(flag) + 1] == value
        for key, value in (
            ("gpu_memory_utilization", 1.0),
            ("max_num_seqs", 0),
            ("max_model_len", "8192"),
        ):
            bad = dataclasses.replace(
                cfg, planners={**cfg.planners, mode: {**cfg.planners[mode], key: value}}
            )
            with pytest.raises(ValueError, match=key):
                serve_command(bad, mode)


@pytest.mark.parametrize(
    "device_count,platform,batch,valid",
    [
        (8, "gpu", 16, True),
        (1, "gpu", 16, False),
        (8, "gpu", 4, False),
        (8, "cpu", 16, False),
    ],
)
def test_runtime_probe_checks_fsdp_against_visible_devices(
    setup_config, monkeypatch, device_count, platform, batch, valid
):
    from robots.robocasa.retention import openpi_adapter

    cfg, _ = setup_config
    cfg = dataclasses.replace(cfg, fsdp_devices=8, batch_size=batch)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(
            devices=lambda: [SimpleNamespace(platform=platform)] * device_count,
            __version__="test",
        ),
    )
    monkeypatch.setattr(openpi_adapter, "verify_openpi_checkout", lambda root: None)
    monkeypatch.setattr(openpi_adapter, "policy_config", lambda checkpoint: None)
    if valid:
        assert len(preflight.probe_runtime(cfg, "openpi-train")["devices"]) == 8
    else:
        with pytest.raises((RuntimeError, ValueError)):
            preflight.probe_runtime(cfg, "openpi-train")
