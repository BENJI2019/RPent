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

"""One bounded simulator episode, with direct VLA or the existing RPent agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from robots.robocasa.env_client import RoboCasaEnvClient
from robots.robocasa.retention.evaluate import cell_dir, cell_identity
from robots.robocasa.retention.protocol import Experiment, episode_seed, load_catalog
from robots.robocasa.rldx_skill import RLDXSkill
from robots.robocasa.vla_client import RoboCasaVLAClient
from rpent.evaluation import write_json_atomic
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.rpc import wait_for_ready
from rpent.utils.rpc.http_rpc import HttpRpcClient


def run_direct(
    env: RoboCasaEnvClient,
    vla: RoboCasaVLAClient,
    *,
    seed: int,
    horizon: int,
    action_steps: int,
) -> None:
    """Execute π0.5 chunks without planner, scripted skills, or settle heuristics."""
    env.reset()
    vla.reset_session(seed=seed)
    prompt = env.get_task_language()
    if not prompt:
        raise ValueError("environment did not provide its task instruction")
    skill = RLDXSkill(env, vla_client=vla)
    skill._load()
    steps = 0
    while steps < horizon and not env.check_success():
        skill._seed_hist(prompt)
        actions = vla.predict(skill._build_obs(prompt), {})
        chunk = actions["action.gripper_close"].shape[1]
        if chunk < action_steps:
            raise ValueError("policy action horizon is shorter than action_steps")
        for index in range(min(action_steps, horizon - steps)):
            action = skill._build_env_action(
                actions["action.end_effector_position"][0, index],
                actions["action.end_effector_rotation"][0, index],
                actions["action.gripper_close"][0, index],
                actions["action.base_motion"][0, index],
                actions["action.control_mode"][0, index],
            )
            env.step(action)
            steps += 1
            if env.check_success():
                break


def agent_command(
    experiment: Experiment,
    mode: str,
    split: str,
    task: str,
    seed: int,
    output: Path,
    env_endpoint: str,
    vla_endpoint: str,
    memory_dir: Path,
) -> list[str]:
    planner = experiment.planners[mode]
    return [
        sys.executable,
        "-m",
        "rpent.cli.main",
        "--robot",
        "robocasa",
        "--task-name",
        task,
        "--split",
        split,
        "--seed",
        str(seed),
        "--planner",
        "api",
        "--model",
        f"openai-chat:{planner['model']}",
        "--base-url",
        planner["base_url"],
        "--reasoning-effort",
        "none",
        "--max-turns",
        str(experiment.max_turns),
        "--planner-timeout-s",
        str(experiment.planner_timeout_s),
        "--env-endpoint",
        env_endpoint,
        "--vla-endpoint",
        vla_endpoint,
        "--vla-backend",
        "pi05",
        "--memory-profile",
        "local",
        "--memory-dir",
        str(memory_dir),
        "--no-task-memory",
        "--output-dir",
        str(output),
    ]


def run_cell(
    experiment: Experiment,
    checkpoint_id: str,
    mode: str,
    split: str,
    task: str,
    seed: int,
    vla_endpoint: str,
) -> dict:
    """Record final native success and initial-state evidence before cleanup."""
    output = cell_dir(experiment, checkpoint_id, mode, split, task, seed)
    output.mkdir(parents=True, exist_ok=True)
    horizon = load_catalog()["tasks"][task]["horizon"]
    identity = cell_identity(experiment, checkpoint_id, mode, split, task, seed)
    native_seed = episode_seed(split, task, seed)
    port = pick_free_port()
    env_endpoint = f"http://127.0.0.1:{port}"
    daemon = ProcessDaemon(
        "retention_env",
        [
            sys.executable,
            "-m",
            "robots.robocasa.env_server",
            "--task-name",
            task,
            "--split",
            split,
            "--seed",
            str(native_seed),
            "--max-steps",
            str(horizon),
            "--cuda-device",
            str(experiment.env_cuda_device),
            "--port",
            str(port),
            "--parent-watch",
        ],
        env_overrides={"MUJOCO_GL": "egl", "RLDX_RESET_SEED": ""},
        log_path=str(output / "env_server.log"),
    )
    rpc = HttpRpcClient(env_endpoint)
    vla_rpc = HttpRpcClient(vla_endpoint, enable_sessions=True)
    start = time.monotonic()
    record = {
        "identity": identity,
        "status": "infrastructure_error",
        "success": None,
        "environment_seed": native_seed,
    }
    try:
        daemon.start()
        wait_for_ready(rpc, daemon=daemon, timeout_s=120)
        wait_for_ready(vla_rpc, timeout_s=300)
        env = RoboCasaEnvClient(
            rpc,
            expected_meta={
                "task_name": task,
                "split": split,
                "seed": native_seed,
                "camera_h": 256,
                "camera_w": 256,
            },
        )
        valid = True
        termination = "completed"
        if mode == "direct":
            run_direct(
                env,
                RoboCasaVLAClient(vla_rpc),
                seed=native_seed,
                horizon=horizon,
                action_steps=experiment.action_steps,
            )
        else:
            # Never reuse task recipes or memory created by another cell/checkpoint.
            memory = Path(tempfile.mkdtemp(prefix="empty_memory_", dir=output))
            agent_output = Path(tempfile.mkdtemp(prefix="agent_", dir=output))
            os.environ.update(
                {
                    "RLDX_ALLOW_RESET": "0",
                    "RLDX_ACTION_STEPS_PER_CHUNK": str(experiment.action_steps),
                    "RLDX_MAX_CHUNKS": "10",
                    "RLDX_SETTLE_PATIENCE": "999999",
                }
            )
            os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
            from rpent.cli.main import main as run_agent

            command = agent_command(
                experiment,
                mode,
                split,
                task,
                native_seed,
                agent_output,
                env_endpoint,
                vla_endpoint,
                memory,
            )
            previous_argv = sys.argv
            try:
                sys.argv = command[2:]
                returncode = run_agent()
            finally:
                sys.argv = previous_argv
            record["agent_output"] = str(agent_output)
            result_path = agent_output / "result.json"
            result = (
                json.loads(result_path.read_text()) if result_path.is_file() else {}
            )
            stats = rpc.call("env.get_episode_stats")
            # A hard simulator budget is an ordinary unsuccessful episode; an
            # API/startup failure before that budget must not become a robot failure.
            valid = (
                bool(result.get("valid"))
                and result.get("termination_reason") != "planner_timeout"
            ) or stats["steps"] >= horizon
            termination = result.get("termination_reason", "agent_process_error")
            if returncode and not result:
                valid = False
        stats = rpc.call("env.get_episode_stats")
        success = bool(env.check_success())
        valid = (valid or success) and stats["reset_count"] == 1
        record.update(
            status="completed" if valid else "infrastructure_error",
            success=success if valid else None,
            success_source="env.check_success",
            termination_reason="step_budget"
            if stats["steps"] >= horizon
            else termination,
            **stats,
        )
    except Exception as exc:
        record["error_type"] = type(exc).__name__
    finally:
        record["elapsed_s"] = round(time.monotonic() - start, 3)
        write_json_atomic(output / "retention_result.json", record)
        vla_rpc.close()
        rpc.close()
        daemon.stop()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--split", choices=("pretrain", "target"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--vla-endpoint", required=True)
    args = parser.parse_args()
    experiment = Experiment.load(args.config)
    if args.mode not in experiment.modes or (
        args.split,
        args.task,
        args.seed,
    ) not in set(experiment.grid()):
        parser.error("cell does not belong to the configured evaluation grid")
    if args.checkpoint_id != "base":
        experiment.run(args.checkpoint_id)
    result = run_cell(
        experiment,
        args.checkpoint_id,
        args.mode,
        args.split,
        args.task,
        args.seed,
        args.vla_endpoint,
    )
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
