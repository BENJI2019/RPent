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

"""Offline Linux checks for shared-volume paths and local/MTP environment setup."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from robots.robocasa.retention import prepare

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux cluster scripts")
CLUSTER = Path(prepare.__file__).parent / "cluster"


def shell(code, **env):
    return subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1/common.sh"; ' + code,
            "check",
            str(CLUSTER),
        ],
        env={**os.environ, **env},
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    "metadata,error",
    [
        ({}, "MA_NUM_HOSTS"),
        ({"MA_NUM_HOSTS": "1"}, "MA_NUM_GPUS"),
        ({"MA_NUM_HOSTS": "2", "MA_NUM_GPUS": "8"}, "multi-host"),
        (
            {"MA_NUM_HOSTS": "1", "MA_NUM_GPUS": "8", "VC_TASK_INDEX": "1"},
            "VC_TASK_INDEX",
        ),
    ],
)
def test_mtp_refuses_invalid_platform_before_touching_mounts(metadata, error):
    env = {
        "MA_NUM_HOSTS": "",
        "MA_NUM_GPUS": "",
        "VC_TASK_INDEX": "",
        "VC_WORKER_HOSTS": "",
        "RETENTION_CODE_ROOT": str(CLUSTER.parents[3]),
    }
    env.update(metadata)
    result = shell("validate_mtp", **env)
    assert result.returncode != 0 and error in result.stderr


def test_valid_mtp_metadata_still_refuses_local_algorithm_path():
    result = shell(
        "validate_mtp",
        MA_NUM_HOSTS="1",
        MA_NUM_GPUS="8",
        VC_TASK_INDEX="0",
        VC_WORKER_HOSTS="worker0",
        RETENTION_CODE_ROOT="/home/ma-user/work/algorithm/hyy_vla/rpent",
    )
    assert result.returncode != 0 and "snapshot" in result.stderr


def test_mount_alias_is_idempotent_but_never_replaces_a_real_directory(tmp_path):
    target, alias = tmp_path / "target", tmp_path / "alias"
    target.mkdir()
    command = f"ensure_link {shlex.quote(str(target))} {shlex.quote(str(alias))}"
    assert shell(command).returncode == 0
    assert shell(command).returncode == 0
    assert alias.resolve() == target
    real = tmp_path / "real"
    real.mkdir()
    (real / "keep").write_text("untouched")
    failed = shell(f"ensure_link {shlex.quote(str(target))} {shlex.quote(str(real))}")
    assert failed.returncode != 0 and "refusing to replace" in failed.stderr
    assert (real / "keep").read_text() == "untouched"


def test_role_activation_removes_old_cudnn_and_uses_requested_prefix(tmp_path):
    sim, openpi, qwen = (tmp_path / name for name in ("sim", "openpi", "qwen"))
    for prefix in (sim, openpi, qwen):
        for part in (
            "bin",
            "conda-meta",
            "lib/python3.11/site-packages/nvidia/cudnn/lib",
        ):
            (prefix / part).mkdir(parents=True)
        (prefix / "bin/python").write_text("#!/bin/sh\nexit 0\n")
        (prefix / "bin/python").chmod(0o755)
    conda = tmp_path / "miniconda/etc/profile.d/conda.sh"
    conda.parent.mkdir(parents=True)
    conda.write_text(
        'conda() { export CONDA_PREFIX="$2"; export PATH="$2/bin:$PATH"; export LD_LIBRARY_PATH="$RETENTION_SIM_PREFIX/lib:$2/lib"; }\n'
    )
    result = shell(
        'shared_path() { :; }; activate_role "$RETENTION_OPENPI_PREFIX"; printf "%s\\n" "$CONDA_PREFIX" "$PATH" "$LD_LIBRARY_PATH" "$CUDNN_HOME"',
        RETENTION_SIM_PREFIX=str(sim),
        RETENTION_OPENPI_PREFIX=str(openpi),
        RETENTION_QWEN_PREFIX=str(qwen),
        MINICONDA_PATH=str(tmp_path / "miniconda"),
        CONDA_PREFIX=str(sim),
        CUDA_HOME="/shared/cuda-12.8",
        LD_LIBRARY_PATH=f"{sim}/lib:{sim}/lib/python3.11/site-packages/nvidia/cudnn/lib:/system/lib",
        CUDNN_HOME=str(sim / "lib/python3.11/site-packages/nvidia/cudnn"),
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == str(openpi)
    assert str(sim) not in result.stdout and str(qwen) not in result.stdout
    assert lines[1].startswith("/shared/cuda-12.8/bin:")
    assert lines[3] == str(openpi / "lib/python3.11/site-packages/nvidia/cudnn")


def test_huawei_init_keeps_code_in_package_and_outputs_on_model_volume(
    tmp_path, monkeypatch
):
    model, dataset = tmp_path / "model", tmp_path / "dataset"
    home = model / "xiaoyi_tmpstorage/hyy_files/rpent"
    monkeypatch.setattr(prepare, "SHARED_MODEL_ROOTS", (model,))
    monkeypatch.setattr(prepare, "SHARED_DATASET_ROOTS", (dataset,))
    root = Path(prepare.__file__).resolve().parents[3]
    for name, value in {
        "RETENTION_HOME": home,
        "RETENTION_CODE_ROOT": root,
        "RETENTION_DATASETS": dataset / "target",
        "RETENTION_CHECKPOINT": home / "base",
        "RETENTION_SIM_PREFIX": dataset / "sim",
        "RETENTION_OPENPI_PREFIX": dataset / "openpi",
        "RETENTION_EXECUTION": "local",
    }.items():
        monkeypatch.setenv(name, str(value))
    destination = home / "configs/local/pilot.json"
    cfg = prepare.initialize(destination, profile="huawei", preset="pilot1")
    assert cfg.openpi_root == str(root / "vendor/openpi")
    assert cfg.output_root == str(home / "retention_outputs/local/pilot1")
    wrapper = Path(cfg.openpi_python)
    assert os.access(wrapper, os.X_OK)
    assert str(dataset / "openpi") in wrapper.read_text()
    assert "RETENTION_CODE_ROOT" in wrapper.read_text()
    with pytest.raises(FileExistsError):
        prepare.initialize(destination, profile="huawei", preset="pilot1")
    with pytest.raises(ValueError, match="under RETENTION_HOME"):
        prepare.initialize(tmp_path / "outside.json", profile="huawei", preset="pilot1")
    with pytest.raises(ValueError, match="output_root"):
        prepare.initialize(
            home / "configs/other.json",
            profile="huawei",
            preset="pilot1",
            output_root=root / "results",
        )
