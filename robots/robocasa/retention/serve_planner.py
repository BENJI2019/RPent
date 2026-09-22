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


"""Launch one pinned Qwen vision planner in a separate vLLM environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

from robots.robocasa.retention.protocol import Experiment


def serve_command(experiment: Experiment, mode: str) -> list[str]:
    """Use greedy, non-thinking generation with the model's native tool syntax."""
    if mode == "direct" or mode not in experiment.modes:
        raise ValueError("select a configured Qwen planner mode")
    planner = experiment.planners[mode]
    address = urlsplit(planner["base_url"])
    if (
        address.scheme != "http"
        or address.hostname not in {"localhost", "127.0.0.1"}
        or not address.port
    ):
        raise ValueError("the local launcher requires http://127.0.0.1:PORT/v1")
    if not planner.get("revision"):
        raise ValueError("a pinned planner revision is required by the local launcher")
    for key, default in (("max_model_len", 32768), ("max_num_seqs", 4)):
        value = planner.get(key, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    memory = float(planner.get("gpu_memory_utilization", 0.85))
    if not math.isfinite(memory) or not 0 < memory < 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1)")
    parser = "hermes" if mode == "qwen3_vl_4b" else "qwen3_coder"
    command = [
        str(Path(sys.executable).with_name("vllm")),
        "serve",
        planner["model"],
        "--served-model-name",
        planner["model"],
        "--revision",
        planner["revision"],
        "--host",
        "127.0.0.1",
        "--port",
        str(address.port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(planner.get("max_model_len", 32768)),
        "--max-num-seqs",
        str(planner.get("max_num_seqs", 4)),
        "--gpu-memory-utilization",
        str(memory),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        parser,
        "--generation-config",
        "vllm",
        "--override-generation-config",
        '{"temperature":0.0}',
        "--seed",
        "0",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
    ]
    if mode == "qwen35_4b":
        command.extend(["--reasoning-parser", "qwen3"])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()
    experiment = Experiment.load(args.config)
    command = serve_command(experiment, args.mode)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("vllm", "transformers", "torch")
    }
    record = {
        "protocol_id": experiment.protocol_id,
        "command": command,
        "versions": versions,
    }
    path = Path(experiment.output_root) / "services" / f"{args.mode}_launch.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != record:
        raise ValueError(
            "planner launch changed; use the original runtime or a fresh output_root"
        )
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
