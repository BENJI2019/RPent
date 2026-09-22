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

"""Offline behavior checks for the retention protocol and π0.5 boundary."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robocasa.env_server import RoboCasaEnvFacade
from robots.robocasa.pi05_server import (
    CAMERAS,
    STATE_KEYS,
    RoboCasaPi05Facade,
    decode_actions,
    encode_observation,
)
from robots.robocasa.retention.evaluate import (
    cell_dir,
    cell_identity,
    completed_record,
    selected_cells,
)
from robots.robocasa.retention.openpi_adapter import norm_file
from robots.robocasa.retention.protocol import Experiment, episode_seed, load_catalog
from robots.robocasa.retention.summarize import (
    comparison_groups,
    paired_summary,
    summarize,
)
from rpent.evaluation import write_json_atomic


@pytest.fixture
def experiment(tmp_path):
    return Experiment(
        base_checkpoint=str(tmp_path / "base"),
        checkpoint_revision="test-immutable-revision",
        openpi_root=str(tmp_path / "openpi"),
        datasets_root=str(tmp_path / "datasets"),
        output_root=str(tmp_path / "results"),
        eval_seeds=[1000, 1001],
        modes=["direct"],
        adaptation_tasks=["OpenDrawer"],
    )


def test_full_plan_has_independent_parents_and_separate_scene_domains(experiment):
    full = dataclasses.replace(
        experiment, adaptation_tasks=None, modes=["direct", "qwen3_vl_4b", "qwen35_4b"]
    )
    full.validate()
    plan = full.plan()
    assert len(plan["adaptations"]) == 50
    assert {r["source_checkpoint"] for r in plan["adaptations"]} == {
        full.base_checkpoint
    }
    assert plan["total_cells"] == 51 * 350 * 2 * 3
    cells = list(full.grid())
    assert ("pretrain", "OpenDrawer", 1000) in cells
    assert ("target", "OpenDrawer", 1000) in cells
    catalog = load_catalog()
    target = {t for group in catalog["target_splits"].values() for t in group}
    assert len(set(catalog["pretrain300"]) - target) == 266
    assert all(
        "/target/" in catalog["tasks"][task]["target_dataset"] for task in target
    )
    assert all("pretrain_dataset" not in row for row in catalog["tasks"].values())


def test_protocol_rejects_text_model_and_duplicate_or_unknown_tasks(experiment):
    invalid = dataclasses.replace(
        experiment,
        modes=["qwen3_vl_4b"],
        planners={
            "qwen3_vl_4b": {
                "model": "Qwen/Qwen3-4B-Instruct-2507",
                "base_url": "http://localhost/v1",
            }
        },
    )
    with pytest.raises(ValueError, match="vision model"):
        invalid.validate()
    with pytest.raises(ValueError, match="Target50"):
        dataclasses.replace(experiment, adaptation_tasks=["NotATask"]).validate()
    with pytest.raises(ValueError, match="unique"):
        dataclasses.replace(experiment, eval_seeds=[1, 1]).validate()


def test_sharding_is_disjoint_complete_and_task_specific(experiment):
    shards = [
        set(selected_cells(experiment, num_shards=3, shard_index=i)) for i in range(3)
    ]
    assert set.union(*shards) == set(experiment.grid())
    assert not any(shards[i] & shards[j] for i in range(3) for j in range(i))
    assert episode_seed("target", "OpenDrawer", 1000) != episode_seed(
        "pretrain", "OpenDrawer", 1000
    )
    assert len({episode_seed(*cell) for cell in experiment.grid()}) == len(
        list(experiment.grid())
    )
    with pytest.raises(ValueError):
        list(selected_cells(experiment, shard_index=3, num_shards=3))


def test_budget_changes_invalidate_resume_but_device_assignment_does_not(experiment):
    assert (
        experiment.protocol_id
        != dataclasses.replace(experiment, action_steps=9).protocol_id
    )
    assert (
        experiment.protocol_id
        != dataclasses.replace(experiment, checkpoint_revision="other").protocol_id
    )
    assert (
        experiment.protocol_id
        == dataclasses.replace(experiment, vla_cuda_device=7).protocol_id
    )


def observation():
    obs = {
        key: np.arange(size, dtype=np.float32)[None, None] + offset
        for offset, (key, size) in enumerate(
            zip(STATE_KEYS, (3, 4, 3, 4, 2), strict=True)
        )
    }
    for key in CAMERAS.values():
        obs[key] = np.arange(6 * 7 * 3, dtype=np.uint8).reshape(1, 1, 6, 7, 3)
    obs["annotation.human.task_description"] = ["Open the drawer"]
    return obs


def test_pi05_state_camera_order_and_native_action_values():
    obs = observation()
    encoded = encode_observation(obs, lambda image: image)
    np.testing.assert_array_equal(
        encoded["observation/state"], np.concatenate([obs[k][0, 0] for k in STATE_KEYS])
    )
    assert encoded["observation/state"].shape == (16,)
    for key, source in CAMERAS.items():
        np.testing.assert_array_equal(encoded[key], obs[source][0, 0])
    actions = np.arange(24, dtype=np.float32).reshape(2, 12) / 24
    decoded = decode_actions(actions)
    np.testing.assert_array_equal(
        decoded["action.gripper_close"][0, :, 0], actions[:, 6]
    )
    np.testing.assert_array_equal(decoded["action.base_motion"][0], actions[:, 7:11])
    for bad in (np.zeros((2, 7)), np.full((2, 12), np.nan), np.empty((0, 12))):
        with pytest.raises(ValueError, match="action array"):
            decode_actions(bad)
    obs[STATE_KEYS[0]] = np.zeros((3,))
    with pytest.raises(ValueError, match="finite"):
        encode_observation(obs, lambda image: image)


def test_pi05_sampling_is_private_to_session_and_reproducible_after_reset():
    class Policy:
        def infer(self, obs, *, noise):
            return {"actions": noise[:, :12]}

    facade = RoboCasaPi05Facade(
        Policy(), action_horizon=8, action_dim=32, resize_image=lambda image: image
    )
    facade.reset_session(session_id="a", seed=123)
    facade.reset_session(session_id="b", seed=123)
    first = facade.predict(observation(), {}, session_id="a")
    second = facade.predict(observation(), {}, session_id="a")
    other = facade.predict(observation(), {}, session_id="b")
    np.testing.assert_array_equal(
        first["action.base_motion"], other["action.base_motion"]
    )
    assert not np.array_equal(first["action.base_motion"], second["action.base_motion"])
    with pytest.raises(ValueError, match="owned"):
        facade.predict(observation(), {"session_ids": ["b"]}, session_id="a")
    facade._on_session_drop("a")
    assert "a" not in facade._rngs


def test_normalization_never_falls_back_to_dataset_statistics(tmp_path):
    with pytest.raises(ValueError, match="expected one"):
        norm_file(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "norm_stats.json").write_text("{}")
    assert norm_file(tmp_path) == assets / "norm_stats.json"
    (assets / "other").mkdir()
    (assets / "other" / "norm_stats.json").write_text("{}")
    with pytest.raises(ValueError, match="got 2"):
        norm_file(tmp_path)


def test_paired_task_weighting_and_small_sample_noninferiority():
    stats = paired_summary(
        {"a": [(True, False)], "b": [(False, True)] * 9}, epsilon=0.05
    )
    assert stats["delta"] == 0
    assert stats["baseline_success_rate"] == 0.5
    unchanged = paired_summary({"a": [(True, True)] * 10}, epsilon=0.05)
    assert unchanged["delta_lower_bound"] < -0.05
    assert not unchanged["noninferior"]


def write_cell(experiment, run_id, split, task, seed, success, digest="scene"):
    output = cell_dir(experiment, run_id, "direct", split, task, seed)
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "identity": cell_identity(experiment, run_id, "direct", split, task, seed),
        "status": "completed",
        "success": success,
        "success_source": "env.check_success",
        "initial_state_sha256": digest,
        "reset_count": 1,
        "max_steps": 100,
    }
    write_json_atomic(output / "retention_result.json", record)
    return output / "retention_result.json"


def test_resume_rejects_wrong_identity_and_never_counts_infrastructure_error(
    experiment,
):
    path = write_cell(experiment, "base", "target", "OpenDrawer", 1000, False)
    identity = cell_identity(experiment, "base", "direct", "target", "OpenDrawer", 1000)
    assert completed_record(path, identity)["success"] is False
    with pytest.raises(ValueError, match="identity"):
        completed_record(path, {**identity, "seed": 1})
    record = json.loads(path.read_text())
    record.update(status="infrastructure_error", success=None)
    write_json_atomic(path, record)
    assert completed_record(path, identity) is None


def test_missing_pairs_do_not_shrink_denominator_and_mismatched_states_fail(experiment):
    run = experiment.runs()[0]["id"]
    for seed in experiment.eval_seeds:
        write_cell(experiment, "base", "target", "OpenDrawer", seed, True)
    write_cell(experiment, run, "target", "OpenDrawer", 1000, False)
    report = summarize(experiment)
    diagonal = next(r for r in report["comparisons"] if r["group"] == "adapted_target")
    assert diagonal["metrics"] is None and diagonal["missing_pairs"] == 1
    write_cell(experiment, run, "target", "OpenDrawer", 1001, False)
    report = summarize(experiment)
    diagonal = next(r for r in report["comparisons"] if r["group"] == "adapted_target")
    assert diagonal["metrics"]["delta"] == -1
    assert not report["complete"]
    write_cell(experiment, run, "target", "OpenDrawer", 1001, False, digest="different")
    with pytest.raises(ValueError, match="states differ"):
        summarize(experiment)


def test_simulator_budget_prevents_extra_actions_and_resets():
    calls = []
    native = SimpleNamespace(
        action_dim=12,
        step=lambda action: (calls.append(action.copy()) or {}, 0, False, {}),
        reset=lambda: {},
        sim=SimpleNamespace(
            get_state=lambda: SimpleNamespace(flatten=lambda: np.zeros(4))
        ),
    )
    facade = object.__new__(RoboCasaEnvFacade)
    facade.env = native
    facade.seed = 12
    facade.max_steps = 2
    facade._reset_count = 0
    facade._episode_steps = 0
    facade._initial_state_sha256 = None
    facade.reset()
    facade.step(np.zeros(12))
    facade.step(np.zeros(12))
    with pytest.raises(RuntimeError, match="budget"):
        facade.step(np.ones(12))
    with pytest.raises(RuntimeError, match="reset"):
        facade.reset()
    assert len(calls) == 2
    assert facade.get_episode_stats()["reset_count"] == 1


def test_pi05_reset_keeps_the_configured_episode_seed():
    class Policy:
        def infer(self, obs, *, noise):
            return {"actions": noise[:, :12]}

    facade = RoboCasaPi05Facade(
        Policy(), action_horizon=8, action_dim=32, resize_image=lambda im: im
    )
    facade.reset_session(session_id="episode", seed=19)
    first = facade.predict(observation(), {}, session_id="episode")
    facade.reset_session(session_id="episode")
    again = facade.predict(observation(), {}, session_id="episode")
    np.testing.assert_array_equal(
        first["action.base_motion"], again["action.base_motion"]
    )


@pytest.mark.parametrize("joint", [False, True])
def test_training_selects_only_target_data_and_keeps_parent_stats(
    experiment, monkeypatch, joint
):
    import sys

    from robots.robocasa.retention import openpi_adapter as adapter

    if joint:
        experiment = dataclasses.replace(
            experiment,
            adaptation_mode="joint",
            adaptation_tasks=["OpenDrawer", "PrepareCoffee", "ArrangeBreadBasket"],
        )

    @dataclasses.dataclass
    class Model:
        def get_freeze_filter(self):
            return "frozen_lora_backbone"

    @dataclasses.dataclass
    class Data:
        data_dirs: list
        base_config: object
        dataset_weights: object = None

    fields = [
        ("model", object),
        ("data", object),
        ("ema_decay", object),
        *[
            (key, object, dataclasses.field(default=None))
            for key in (
                "name",
                "exp_name",
                "weight_loader",
                "freeze_filter",
                "checkpoint_base_dir",
                "assets_base_dir",
                "seed",
                "num_train_steps",
                "batch_size",
                "num_workers",
                "fsdp_devices",
                "lr_schedule",
                "save_interval",
                "keep_period",
                "wandb_enabled",
                "overwrite",
                "resume",
            )
        ],
    ]
    Config = dataclasses.make_dataclass("Config", fields)
    stats = object()
    calls = []

    def base_policy(parent, method):
        calls.append((parent, method))
        return Config(Model(), Data([], SimpleNamespace(norm_stats=stats)), 0.99), stats

    monkeypatch.setattr(adapter, "policy_config", base_policy)
    training = SimpleNamespace(
        optimizer=SimpleNamespace(CosineDecaySchedule=lambda **kwargs: kwargs),
        weight_loaders=SimpleNamespace(CheckpointWeightLoader=lambda path: path),
    )
    monkeypatch.setitem(sys.modules, "openpi", SimpleNamespace(training=training))
    monkeypatch.setitem(sys.modules, "openpi.training", training)
    run = experiment.runs()[0]
    paths = []
    for task in run["tasks"]:
        path = (
            Path(experiment.datasets_root)
            / load_catalog()["tasks"][task]["target_dataset"]
        )
        (path / "meta").mkdir(parents=True)
        (path / "meta" / "info.json").write_text("{}")
        paths.append(path)
    result = adapter.training_config(experiment, run)
    assert calls == [(experiment.base_checkpoint, "full")]
    assert result.weight_loader == str(Path(experiment.base_checkpoint) / "params")
    assert result.data.base_config.norm_stats is stats
    assert [d["path"] for d in result.data.data_dirs] == [
        str(p.resolve()) for p in paths
    ]
    assert {d["split"] for d in result.data.data_dirs} == {"target"}
    assert {d["filter_key"] for d in result.data.data_dirs} == {"500_demos"}
    assert result.data.dataset_weights is None
    assert result.exp_name == f"{run['label']}_seed_0"
    assert result.freeze_filter == "frozen_lora_backbone"
    assert result.ema_decay == 0.99
    assert not result.overwrite
    result = adapter.training_config(experiment, {**run, "method": "lora"})
    assert result.ema_decay is None
    (paths[-1] / "meta" / "info.json").unlink()
    with pytest.raises(FileNotFoundError, match="LeRobot"):
        adapter.training_config(experiment, run)


def test_completed_training_resume_does_not_restart_optimizer(experiment, monkeypatch):
    from robots.robocasa.retention import train as trainer
    from robots.robocasa.retention.openpi_adapter import norm_digest

    base = Path(experiment.base_checkpoint)
    (base / "params").mkdir(parents=True)
    (base / "assets").mkdir()
    (base / "assets" / "norm_stats.json").write_text("{}")
    run = experiment.runs()[0]
    output = Path(experiment.output_root) / "training" / run["id"]
    output.mkdir(parents=True)
    write_json_atomic(
        output / "checkpoint.json",
        {
            "protocol_id": experiment.protocol_id,
            "run": run,
            "base_norm_sha256": norm_digest(base),
            "checkpoint": str(base),
            "method": "full",
        },
    )
    monkeypatch.setattr(trainer, "verify_openpi_checkout", lambda root: None)
    monkeypatch.setattr(trainer, "init_output_dir", lambda output: None)
    monkeypatch.setattr(
        trainer,
        "training_config",
        lambda *args, **kwargs: pytest.fail("optimizer restarted"),
    )
    assert trainer.train(experiment, run["id"], resume=True) == base.resolve()
    with pytest.raises(FileExistsError):
        trainer.train(experiment, run["id"])


def test_planner_launcher_uses_pinned_vision_models_and_tool_parsers(experiment):
    from robots.robocasa.retention.serve_planner import serve_command

    cfg = dataclasses.replace(experiment, modes=["qwen3_vl_4b", "qwen35_4b"])
    for mode, parser in (("qwen3_vl_4b", "hermes"), ("qwen35_4b", "qwen3_coder")):
        command = serve_command(cfg, mode)
        assert command[command.index("--tool-call-parser") + 1] == parser
        assert (
            command[command.index("--revision") + 1] == cfg.planners[mode]["revision"]
        )
        assert json.loads(
            command[command.index("--override-generation-config") + 1]
        ) == {"temperature": 0.0}
    with pytest.raises(ValueError):
        serve_command(cfg, "direct")


def test_joint50_uses_one_parent_and_checkpoint_per_method_and_training_seed(
    experiment,
):
    cfg = dataclasses.replace(
        experiment,
        adaptation_tasks=None,
        adaptation_mode="joint",
        methods=["full", "lora"],
        train_seeds=[0, 1],
    )
    cfg.validate()
    plan = cfg.plan()
    assert len(plan["adaptations"]) == 4
    assert {r["source_checkpoint"] for r in cfg.runs()} == {cfg.base_checkpoint}
    assert all(len(r["tasks"]) == 50 and r["task"] is None for r in cfg.runs())
    assert cfg.run("full/joint_target50/seed_0")["adaptation_mode"] == "joint"
    assert plan["total_cells"] == 5 * 350 * 2
    assert plan["total_training_steps"] == 4 * cfg.train_steps
    assert plan["training_sampling"] == "upstream_length_power_0.4"
    groups = comparison_groups(cfg.runs()[0]["tasks"], joint=True)
    assert len(groups["old_excluding_adapted_tasks"][1]) == 266


def test_joint_subset_names_identify_the_task_set_and_reject_reusing_results(
    experiment,
):
    tasks = ["OpenDrawer", "PrepareCoffee", "ArrangeBreadBasket"]
    cfg = dataclasses.replace(
        experiment, adaptation_mode="joint", adaptation_tasks=tasks
    )
    reversed_cfg = dataclasses.replace(cfg, adaptation_tasks=list(reversed(tasks)))
    different = dataclasses.replace(cfg, adaptation_tasks=tasks[:2])
    assert len(cfg.runs()) == 1
    assert cfg.runs()[0]["id"] == reversed_cfg.runs()[0]["id"]
    assert cfg.runs()[0]["id"] != different.runs()[0]["id"]
    assert cfg.protocol_id != different.protocol_id
    assert len(dataclasses.replace(cfg, adaptation_mode="independent").runs()) == 3


@pytest.mark.parametrize(
    "scope",
    [
        {},
        {"target": []},
        {"old": ["OpenDrawer"]},
        {"pretrain": ["ArrangeBreadBasket"]},
        {"target": ["CloseDrawer"]},
        {"target": ["OpenDrawer", "OpenDrawer"]},
        {"target": "OpenDrawer"},
    ],
)
def test_invalid_evaluation_scope_is_rejected(experiment, scope):
    with pytest.raises(ValueError, match="evaluation_tasks"):
        dataclasses.replace(experiment, evaluation_tasks=scope).validate()


def test_unknown_adaptation_mode_is_rejected(experiment):
    with pytest.raises(ValueError, match="adaptation_mode"):
        dataclasses.replace(experiment, adaptation_mode="sequential").validate()


def test_declared_pilot_grid_keeps_scene_domains_and_execution_filter_separate(
    experiment,
):
    cfg = dataclasses.replace(
        experiment,
        evaluation_tasks={
            "target": ["OpenDrawer"],
            "pretrain": ["OpenDrawer", "CloseDrawer"],
        },
    )
    cfg.validate()
    assert len(list(cfg.grid())) == 6
    shards = [set(selected_cells(cfg, num_shards=2, shard_index=i)) for i in range(2)]
    assert shards[0] | shards[1] == set(cfg.grid())
    assert not shards[0] & shards[1]
    assert len(list(selected_cells(cfg, tasks=["OpenDrawer"]))) == 4
    assert cfg.plan()["total_cells"] == 12
    assert cfg.plan()["evaluation_scope"] == "task_subset"
    assert cfg.protocol_id != experiment.protocol_id


@pytest.mark.parametrize("joint", [False, True])
def test_complete_pilot_summary_labels_subsets_and_excludes_trained_old_tasks(
    experiment, joint
):
    cfg = dataclasses.replace(
        experiment,
        adaptation_mode="joint" if joint else "independent",
        adaptation_tasks=["OpenDrawer", "ArrangeBreadBasket"]
        if joint
        else ["OpenDrawer"],
        evaluation_tasks={
            "target": ["OpenDrawer"],
            "pretrain": ["OpenDrawer", "CloseDrawer"],
        },
    )
    cfg.validate()
    run_id = cfg.runs()[0]["id"]
    for split, task, seed in cfg.grid():
        write_cell(cfg, "base", split, task, seed, True)
        write_cell(cfg, run_id, split, task, seed, task == "OpenDrawer")
    report = summarize(cfg)
    assert report["complete"] and not report["full_benchmark_complete"]
    assert report["retention_evaluation_configured"]
    old = next(
        r for r in report["comparisons"] if r["reference_group"] == "pretrain300"
    )
    assert old["group"] == "pretrain300_subset"
    assert (old["reference_task_count"], old["evaluated_task_count"]) == (300, 2)
    assert old["metrics"]["delta"] == -0.5
    excluded = next(
        r
        for r in report["comparisons"]
        if r["reference_group"].startswith("old_excluding_adapted")
    )
    assert excluded["evaluated_task_count"] == 1
    assert excluded["metrics"]["delta"] == -1
    adapted = next(
        r
        for r in report["comparisons"]
        if r["reference_group"].startswith("adapted_target")
    )
    assert adapted["reference_task_count"] == (2 if joint else 1)
    assert adapted["scope"] == ("task_subset" if joint else "complete_group")
    if joint:
        assert "diagonal" not in report["interpretation"]
    missing = (
        cell_dir(cfg, run_id, "direct", "pretrain", "CloseDrawer", cfg.eval_seeds[-1])
        / "retention_result.json"
    )
    missing.unlink()
    incomplete = summarize(cfg)
    assert not incomplete["complete"]
    old = next(
        r for r in incomplete["comparisons"] if r["reference_group"] == "pretrain300"
    )
    assert old["metrics"] is None and old["expected_pairs"] == 4


def test_target_only_pilot_has_no_retention_claim(experiment):
    cfg = dataclasses.replace(experiment, evaluation_tasks={"target": ["OpenDrawer"]})
    for split, task, seed in cfg.grid():
        for run_id in ("base", cfg.runs()[0]["id"]):
            write_cell(cfg, run_id, split, task, seed, True)
    report = summarize(cfg)
    assert report["complete"] and not report["retention_evaluation_configured"]
    assert all(
        "pretrain" not in r["group"] and not r["group"].startswith("old")
        for r in report["comparisons"]
    )


@pytest.mark.parametrize(
    "name,checkpoints,cells",
    [
        ("examples/pilot1.json", 1, 24),
        ("examples/pilot3-joint.json", 1, 36),
        ("examples/pilot3-independent.json", 3, 72),
        ("examples/joint50.json", 1, 21000),
        ("example.json", 50, 535500),
    ],
)
def test_stage_presets_match_documented_training_and_evaluation_budgets(
    name, checkpoints, cells
):
    from robots.robocasa.retention.protocol import CATALOG_PATH

    cfg = Experiment.load(CATALOG_PATH.parent / name)
    plan = cfg.plan()
    assert len(plan["adaptations"]) == checkpoints
    assert plan["total_cells"] == cells
    assert plan["total_training_steps"] == checkpoints * cfg.train_steps
