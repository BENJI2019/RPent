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

"""Paired task-weighted retention estimates with explicit missing-cell accounting."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from robots.robocasa.retention.evaluate import cell_dir, cell_identity, completed_record
from robots.robocasa.retention.protocol import Experiment, load_catalog
from rpent.evaluation import write_json_atomic


def paired_summary(
    pairs: dict[str, list[tuple[bool, bool]]], *, epsilon: float, alpha: float = 0.05
) -> dict:
    """Compute a conditional, per-comparison Hoeffding noninferiority bound.

    Each task has equal weight; episodes within a task have equal weight.
    The bound assumes independent paired episodes on the fixed task set and
    says nothing about unknown tasks or simultaneous validity across comparisons.
    """
    if not pairs or any(not rows for rows in pairs.values()):
        raise ValueError("every task must contain paired episodes")
    if not 0 < alpha < 1 or not 0 < epsilon < 1:
        raise ValueError("alpha and epsilon must lie in (0, 1)")
    count = len(pairs)
    baseline = (
        sum(sum(a for a, _ in rows) / len(rows) for rows in pairs.values()) / count
    )
    adapted = (
        sum(sum(b for _, b in rows) / len(rows) for rows in pairs.values()) / count
    )
    delta = adapted - baseline
    weight_squares = sum(1 / (count * count * len(rows)) for rows in pairs.values())
    radius = math.sqrt(2 * math.log(1 / alpha) * weight_squares)
    lower = max(-1.0, delta - radius)
    worst = min(
        sum(int(b) - int(a) for a, b in rows) / len(rows) for rows in pairs.values()
    )
    return {
        "tasks": count,
        "paired_episodes": sum(map(len, pairs.values())),
        "baseline_success_rate": baseline,
        "adapted_success_rate": adapted,
        "delta": delta,
        "forgetting": baseline - adapted,
        "worst_task_delta": worst,
        "delta_lower_bound": lower,
        "noninferior": lower > -epsilon,
        "epsilon": epsilon,
        "alpha": alpha,
        "bound": "one_sided_hoeffding_fixed_tasks_independent_paired_episodes",
        "simultaneous_guarantee": False,
    }


def comparison_groups(
    adapted_tasks: list[str], *, joint: bool
) -> dict[str, tuple[str, list[str]]]:
    catalog = load_catalog()
    old = catalog["pretrain300"]
    targets = [task for group in catalog["target_splits"].values() for task in group]
    target_set = set(targets)
    groups = {
        "adapted_targets" if joint else "adapted_target": ("target", adapted_tasks),
        "target50": ("target", targets),
        "pretrain300": ("pretrain", old),
        "old_excluding_adapted_tasks" if joint else "old_excluding_adapted_task": (
            "pretrain",
            [t for t in old if t not in adapted_tasks],
        ),
        "old_exclusive266": ("pretrain", [t for t in old if t not in target_set]),
        "old_target_overlap34": ("pretrain", [t for t in old if t in target_set]),
    }
    groups.update(
        {name: ("target", tasks) for name, tasks in catalog["target_splits"].items()}
    )
    return groups


def summarize(experiment: Experiment) -> dict:
    """Require all expected pairs for each score; never average completed cells only."""
    root = Path(experiment.output_root)
    results = []
    task_rows = []
    baseline_cache = {}
    evaluation = experiment.evaluation_sets()
    full_benchmark = sum(map(len, evaluation.values())) == 350
    for run in experiment.runs():
        for mode in experiment.modes:
            adapted_cache = {}
            groups = comparison_groups(
                run["tasks"], joint=experiment.adaptation_mode == "joint"
            )
            for reference_group, (split, reference_tasks) in groups.items():
                tasks = [t for t in reference_tasks if t in evaluation[split]]
                if not tasks:
                    continue
                subset = len(tasks) != len(reference_tasks)
                group = f"{reference_group}_subset" if subset else reference_group
                pairs = {}
                missing = []
                for task in tasks:
                    rows = []
                    for seed in experiment.eval_seeds:
                        key = (mode, split, task, seed)
                        if key not in baseline_cache:
                            identity = cell_identity(
                                experiment, "base", mode, split, task, seed
                            )
                            baseline_cache[key] = completed_record(
                                cell_dir(experiment, "base", mode, split, task, seed)
                                / "retention_result.json",
                                identity,
                            )
                        if key not in adapted_cache:
                            identity = cell_identity(
                                experiment, run["id"], mode, split, task, seed
                            )
                            adapted_cache[key] = completed_record(
                                cell_dir(experiment, run["id"], mode, split, task, seed)
                                / "retention_result.json",
                                identity,
                            )
                        before, after = baseline_cache[key], adapted_cache[key]
                        if before is None or after is None:
                            missing.append(
                                {
                                    "split": split,
                                    "task": task,
                                    "seed": seed,
                                    "baseline_missing": before is None,
                                    "adapted_missing": after is None,
                                }
                            )
                            continue
                        if (
                            before["initial_state_sha256"]
                            != after["initial_state_sha256"]
                        ):
                            raise ValueError(
                                f"initial simulator states differ: {run['id']} {key}"
                            )
                        if before["max_steps"] != after["max_steps"]:
                            raise ValueError(f"step budgets differ: {run['id']} {key}")
                        rows.append((before["success"], after["success"]))
                    if len(rows) == len(experiment.eval_seeds):
                        pairs[task] = rows
                        if reference_group in ("target50", "pretrain300"):
                            task_rows.append(
                                {
                                    "run_id": run["id"],
                                    "adapted_task": run["task"],
                                    "adapted_tasks": json.dumps(run["tasks"]),
                                    "adaptation_mode": experiment.adaptation_mode,
                                    "mode": mode,
                                    "split": split,
                                    "task": task,
                                    "episodes": len(rows),
                                    "before": sum(a for a, _ in rows) / len(rows),
                                    "after": sum(b for _, b in rows) / len(rows),
                                    "delta": sum(int(b) - int(a) for a, b in rows)
                                    / len(rows),
                                }
                            )
                row = {
                    "run_id": run["id"],
                    "mode": mode,
                    "group": group,
                    "reference_group": reference_group,
                    "reference_task_count": len(reference_tasks),
                    "evaluated_task_count": len(tasks),
                    "scope": "task_subset" if subset else "complete_group",
                    "complete": not missing,
                    "expected_pairs": len(tasks) * len(experiment.eval_seeds),
                    "missing_pairs": len(missing),
                    "missing_examples": missing[:5],
                    "metrics": None,
                }
                if not missing:
                    row["metrics"] = paired_summary(pairs, epsilon=experiment.epsilon)
                results.append(row)
    report = {
        "protocol_id": experiment.protocol_id,
        "complete": all(r["complete"] for r in results),
        "full_benchmark_complete": full_benchmark
        and all(r["complete"] for r in results),
        "evaluation_scope": "full_benchmark" if full_benchmark else "task_subset",
        "evaluation_sets": evaluation,
        "retention_evaluation_configured": bool(evaluation["pretrain"]),
        "adaptation_mode": experiment.adaptation_mode,
        "interpretation": (
            "one jointly adapted model per method and training seed; scores cover only the declared evaluation tasks"
            if experiment.adaptation_mode == "joint"
            else "independent specialists; the adapted_target diagonal is not one model's Target50 score"
        ),
        "comparisons": results,
    }
    destination = root / "summary"
    destination.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination / "summary.json", report)
    for filename, rows, fieldnames in (
        (
            "per_task.csv",
            task_rows,
            [
                "run_id",
                "adapted_task",
                "adapted_tasks",
                "adaptation_mode",
                "mode",
                "split",
                "task",
                "episodes",
                "before",
                "after",
                "delta",
            ],
        ),
        (
            "comparisons.csv",
            [
                {
                    **{
                        k: row[k]
                        for k in (
                            "run_id",
                            "mode",
                            "group",
                            "reference_group",
                            "reference_task_count",
                            "evaluated_task_count",
                            "scope",
                            "complete",
                            "expected_pairs",
                            "missing_pairs",
                        )
                    },
                    **(row["metrics"] or {}),
                }
                for row in results
            ],
            [
                "run_id",
                "mode",
                "group",
                "reference_group",
                "reference_task_count",
                "evaluated_task_count",
                "scope",
                "complete",
                "expected_pairs",
                "missing_pairs",
                "baseline_success_rate",
                "adapted_success_rate",
                "delta",
                "forgetting",
                "worst_task_delta",
                "delta_lower_bound",
                "noninferior",
            ],
        ),
    ):
        with (destination / filename).open(
            "w", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return report
