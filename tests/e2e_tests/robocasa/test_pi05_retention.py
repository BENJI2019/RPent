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


"""Opt-in real π0.5 inference, bounded native actions, and Harness tool chain."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from robots.robocasa.retention.cell import run_direct
from robots.robocasa.robot_spec import get_robot_spec
from tests.e2e_tests.common import (
    ScriptedToolCall,
    parse_runtime_args,
    run_scripted_policy_chain,
    runtime_phase,
    selected_cuda_ordinal,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RPENT_RUN_PI05_INTEGRATION") != "1",
    reason="requires Linux, GPU, RoboCasa assets, π0.5 checkpoint and OpenPI environment",
)


def test_pi05_actions_and_harness_chain(tmp_path: Path) -> None:
    arguments = [
        "--task-name",
        "OpenDrawer",
        "--split",
        "target",
        "--seed",
        "1000",
        "--vla-backend",
        "pi05",
        "--vla-model-path",
        os.environ["PI05_CHECKPOINT_PATH"],
        "--vla-python",
        os.environ["PI05_PYTHON"],
        "--cuda-device",
        str(selected_cuda_ordinal()),
        "--env-max-steps",
        "80",
        "--no-task-memory",
    ]
    spec = get_robot_spec()
    args = parse_runtime_args(spec, arguments)
    with runtime_phase(spec, args, tmp_path / "direct", {"env", "vla"}) as runtime:
        env, vla = runtime["env_client"], runtime["vla_client"]
        assert vla.get_modality_config()["backend"] == "pi05"
        run_direct(env, vla, seed=1000, horizon=8, action_steps=8)
        stats = env.get_episode_stats()
        assert stats["steps"] > 0 or env.check_success()
        assert stats["steps"] <= 8 and stats["reset_count"] == 1
        assert stats["initial_state_sha256"]
    result = run_scripted_policy_chain(
        robot="robocasa",
        robot_argv=arguments,
        output_dir=tmp_path / "harness",
        action=ScriptedToolCall("rldx_skill", {"max_chunks": 1, "n_action_steps": 8}),
        action_count_field="steps_applied",
    )
    assert result["simulator_action_count"] > 0
