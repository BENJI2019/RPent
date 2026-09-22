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

"""π0.5 RoboCasa RPC adapter using the official benchmark OpenPI checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

from robots.robocasa.retention.openpi_adapter import (
    norm_digest,
    policy_config,
    verify_openpi_checkout,
)
from rpent.robots.components.vla_facade_base import BaseVLAFacade
from rpent.utils.logging import get_logger

logger = get_logger("robocasa.pi05")

STATE_KEYS = (
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.base_position",
    "state.base_rotation",
    "state.gripper_qpos",
)
CAMERAS = {
    "observation/image": "video.robot0_agentview_left",
    "observation/wrist_image": "video.robot0_eye_in_hand",
    "observation/right_image": "video.robot0_agentview_right",
}
ACTION_SLICES = {
    "action.end_effector_position": (0, 3),
    "action.end_effector_rotation": (3, 6),
    "action.gripper_close": (6, 7),
    "action.base_motion": (7, 11),
    "action.control_mode": (11, 12),
}


def encode_observation(obs: dict, resize_image: Any) -> dict:
    """Map the existing RoboCasa RPC observation to the official OpenPI layout."""
    state_parts = []
    for key, size in zip(STATE_KEYS, (3, 4, 3, 4, 2), strict=True):
        value = np.asarray(obs[key], dtype=np.float32)
        if value.shape != (1, 1, size) or not np.isfinite(value).all():
            raise ValueError(f"{key} must be a finite (1, 1, {size}) array")
        state_parts.append(value[0, 0])
    prompt = obs["annotation.human.task_description"]
    if (
        not isinstance(prompt, list)
        or len(prompt) != 1
        or not isinstance(prompt[0], str)
        or not prompt[0]
    ):
        raise ValueError("one nonempty task instruction is required")
    result = {"observation/state": np.concatenate(state_parts), "prompt": prompt[0]}
    for destination, source in CAMERAS.items():
        value = np.asarray(obs[source])
        if (
            value.ndim != 5
            or value.shape[:2] != (1, 1)
            or value.shape[-1] != 3
            or value.dtype != np.uint8
        ):
            raise ValueError(f"{source} must be uint8 with shape (1, 1, H, W, 3)")
        # RoboCasaEnvClient already flips native rendering into dataset orientation.
        result[destination] = np.ascontiguousarray(resize_image(value[0, 0]))
    return result


def decode_actions(actions: Any) -> dict[str, np.ndarray]:
    """Retain dataset-space values; the environment owns controller conversion."""
    actions = np.asarray(actions, dtype=np.float32)
    if (
        actions.ndim != 2
        or actions.shape[1] != 12
        or not len(actions)
        or not np.isfinite(actions).all()
    ):
        raise ValueError(
            "OpenPI must return a nonempty finite (horizon, 12) action array"
        )
    return {
        name: actions[None, :, start:end]
        for name, (start, end) in ACTION_SLICES.items()
    }


class RoboCasaPi05Facade(BaseVLAFacade):
    """Serve one model with session-isolated, reproducible sampling noise."""

    def __init__(
        self,
        policy: Any,
        *,
        action_horizon: int,
        action_dim: int,
        resize_image: Any,
        metadata: dict | None = None,
    ) -> None:
        self.policy = policy
        self._shape = (action_horizon, action_dim)
        self._resize = resize_image
        self._metadata = metadata or {}
        self._rngs: dict[str, np.random.Generator] = {}
        self._seeds: dict[str, int] = {}
        super().__init__(enable_sessions=True)

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["vla.get_modality_config"] = self.get_modality_config
        self._rpc["vla.reset_session"] = self.reset_session
        self._readonly_methods.add("vla.get_modality_config")

    def get_modality_config(self, *, session_id: str | None = None) -> dict:
        return {
            "video_delta_indices": [0],
            "hist_maxlen": 1,
            "backend": "pi05",
            "action_horizon": self._shape[0],
            **self._metadata,
        }

    def reset_session(self, *, session_id: str, seed: int | None = None) -> dict:
        if seed is None:
            seed = self._seeds.get(session_id, 0)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        self._seeds[session_id] = seed
        self._rngs[session_id] = np.random.default_rng(seed)
        return {"ok": True}

    def _on_session_drop(self, session_id: str) -> None:
        self._rngs.pop(session_id, None)
        self._seeds.pop(session_id, None)

    def predict(self, obs_dict: dict, options: dict | None, *, session_id: str) -> dict:
        if "session_ids" in (options or {}):
            raise ValueError("session_ids are owned by the RPC facade")
        if session_id not in self._rngs:
            self.reset_session(session_id=session_id)
        obs = encode_observation(obs_dict, self._resize)
        noise = self._rngs[session_id].standard_normal(self._shape).astype(np.float32)
        return decode_actions(self.policy.infer(obs, noise=noise)["actions"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--openpi-root", default=None)
    parser.add_argument("--method", choices=("full", "lora"), default="full")
    parser.add_argument("--cuda-device", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--transport", choices=("http", "socket"), default="http")
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    from openpi.policies.policy_config import create_trained_policy
    from openpi_client import image_tools

    if args.openpi_root:
        verify_openpi_checkout(args.openpi_root)

    cfg, stats = policy_config(args.model_path, args.method)
    policy = create_trained_policy(cfg, args.model_path, norm_stats=stats)
    facade = RoboCasaPi05Facade(
        policy,
        action_horizon=cfg.model.action_horizon,
        action_dim=cfg.model.action_dim,
        resize_image=lambda image: image_tools.convert_to_uint8(
            image_tools.resize_with_pad(image, 224, 224)
        ),
        metadata={
            "checkpoint": str(Path(args.model_path).resolve()),
            "normalization_sha256": norm_digest(args.model_path),
            "method": args.method,
        },
    )
    logger.info("π0.5 RoboCasa policy ready: %s", args.model_path)
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
