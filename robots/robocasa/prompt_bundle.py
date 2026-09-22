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

"""RoboCasa prompt bundle assembly."""

from __future__ import annotations

from collections.abc import Mapping

from robots.robocasa import prompts as robocasa_prompt
from rpent.prompt import common as base_prompt
from rpent.prompt.utils import PromptNode


def system_prompt(
    variables: Mapping[str, object] | None = None,
) -> dict[str, PromptNode]:
    """Return the system prompt tree."""
    variables = variables or {}
    return {
        "Intro": robocasa_prompt.PREAMBLE,
        "Goal": robocasa_prompt.GOAL,
        "Rules": robocasa_prompt.RULES,
        "Memory": (
            "No prior task memory is available. Solve from the current observations. "
            "Do not search for or create cross-episode recipes."
            if variables.get("no_task_memory")
            else robocasa_prompt.MEMORY
        ),
        "Localization": robocasa_prompt.LOCALIZATION,
        "Navigation": robocasa_prompt.NAVIGATION,
        "Primitives": robocasa_prompt.PRIMITIVES,
        "VLA_Rules": (
            "Every rldx_skill / rldx_arm call must pass the complete live "
            "task_language verbatim. Do not substitute an atomic subtask. "
            "The pi0.5 policy uses the current camera observation. "
            "The runner fixes chunk and action budgets; omit max_chunks, "
            "n_action_steps and settle_patience. Continue after a chunk cap "
            "when contact or task progress is improving. Re-stage from current "
            "observations when repeated calls make no progress."
            if variables.get("vla_backend") == "pi05"
            else robocasa_prompt.VLA_RULES
        ),
        "VLA_Backend": (
            "The rldx_skill and rldx_arm tool names are compatibility names. "
            "They call the selected pi0.5 checkpoint in this run, not RLDX. "
            "VLA calls receive the environment's complete task instruction."
            if variables.get("vla_backend") == "pi05"
            else ""
        ),
        "Gripper_Rules": robocasa_prompt.GRIPPER_RULES,
        "Workflow": robocasa_prompt.WORKFLOW,
        "Environment": robocasa_prompt.ENVIRONMENT,
        "Output": base_prompt.OUTPUT,
        "Next": robocasa_prompt.NEXT,
    }


def user_prompt(
    variables: Mapping[str, object] | None = None,
) -> dict[str, PromptNode]:
    """Return the first user message tree."""
    return {
        "Task": """
        - task:    {{task_name}} / {{split}}
        - seed:    {{seed}}
        - output_dir: {{output_dir}}
        - output:  {{output_dir}}/
          - audit filename:  {{recipe_tag}}.json
        """,
        "Mode": robocasa_prompt.USER_MODE,
    }
